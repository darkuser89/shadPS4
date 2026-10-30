// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {

struct RasterParameters {
    GeometryAssemblyParameters mesh;
    u32 first;
    u32 count;
    u32 primitive_slots;
    u32 source_words;
    u32 viewport_word;
    u32 viewport_count;
    u32 last_provoking;
};
static_assert(sizeof(RasterParameters) == 108);
static_assert(TessellationExports::VerticesPerPatch == 4288 &&
              TessellationDomains::PointsPerPatch == 4225);

constexpr std::string_view TessellationRasterSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct MeshParameters {
    uint invocations; uint max_vertices; uint vertex_words; uint mesh_vertices;
    uint topology[4]; uint mesh_counts[4]; uint mesh_offsets[4];
    uint stream_stride_words; uint mesh_stride_words; uint meshes_per_invocation; uint jobs;
};
struct Parameters {
    MeshParameters mesh;
    uint first; uint count; uint primitive_slots; uint source_words;
    uint viewport_word; uint viewport_count; uint last_provoking;
};
struct Primitive { uint vertex_offsets[3]; uint patch; uint instance; uint active; };
static_assert(sizeof(MeshParameters) == 80 && sizeof(Parameters) == 108);
static_assert(sizeof(Primitive) == 24);
kernel void metal4_pack_tessellation_mesh(
    device const Primitive* primitives [[buffer(0)]],
    device const uint* vertices [[buffer(1)]], device uint* emissions [[buffer(2)]],
    device uint* output [[buffer(3)]], constant Parameters& p [[buffer(4)]],
    device atomic_uint& validation_errors [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong linear_job = ulong(group.y) * grid.x + group.x;
    if (linear_job >= p.mesh.jobs) return;
    const uint width = p.mesh.topology[0] + 1u;
    const uint per_job = p.mesh.mesh_vertices / width;
    const uint first = uint(linear_job) * per_job;
    const uint count = min(per_job, p.count - first);
    device uint* payload = emissions + linear_job * 4u * p.mesh.stream_stride_words;
    device uint* records = output + linear_job * p.mesh.mesh_stride_words;
    uint errors = 0;
    for (uint local = lane; local < count; local += 32u) {
        const uint index = p.first + first + local;
        const Primitive primitive = primitives[index];
        bool valid = primitive.active <= 1u;
        if (!valid) errors |= 0x400u;
        if (valid && primitive.active) {
            const ulong patch_begin = ulong(index / p.primitive_slots) * 4288u * p.mesh.vertex_words;
            const ulong patch_end = patch_begin + 4225u * p.mesh.vertex_words;
            for (uint vertex_index = 0; vertex_index < width; ++vertex_index) {
                const uint offset = primitive.vertex_offsets[vertex_index];
                const bool in_bounds = !(offset % p.mesh.vertex_words) &&
                    offset >= patch_begin && ulong(offset) + p.mesh.vertex_words <= patch_end &&
                    ulong(offset) + p.mesh.vertex_words <= p.source_words;
                if (!in_bounds || (in_bounds && vertices[offset] != 1u)) valid = false;
            }
            if (!valid) errors |= 0x800u;
            if (valid && p.viewport_word) {
                const uint offset = primitive.vertex_offsets[p.last_provoking ? width - 1u : 0u];
                if (vertices[offset + p.viewport_word] >= p.viewport_count) {
                    valid = false;
                    errors |= 0x1000u;
                }
            }
        }
        for (uint vertex_index = 0; vertex_index < width; ++vertex_index) {
            device uint* row = payload + 4u + ulong(local * width + vertex_index) * p.mesh.vertex_words;
            for (uint word = 0; word < p.mesh.vertex_words; ++word)
                row[word] = valid && primitive.active ? vertices[primitive.vertex_offsets[vertex_index] + word] : 0u;
        }
        device uint* record = records + 4u + ulong(local) * 8u;
        for (uint component = 0; component < 3; ++component)
            record[component] = local * width + min(component, width - 1u);
        record[3] = valid && primitive.active ? local * width : 0xffffffffu;
        record[4] = valid && primitive.active ? local * width + width - 1u : 0xffffffffu;
        record[5] = record[6] = record[7] = 0u;
    }
    errors = simd_or(errors);
    if (!lane) {
        payload[0] = count * width;
        payload[1] = payload[2] = payload[3] = 0u;
        records[0] = 0u;
        records[1] = count * width;
        records[2] = errors ? 0u : count;
        records[3] = errors;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}
)metal";

bool ValidAllocation(const std::shared_ptr<StagingBufferPool::Allocation>& allocation,
                     id<MTLDevice> device, u64 bytes) {
    return allocation && allocation->GetBuffer() && allocation->GetBuffer().device == device &&
           allocation->SizeBytes() >= bytes && allocation->GetOffset() % sizeof(u32) == 0 &&
           allocation->GetOffset() <= allocation->GetBuffer().length &&
           allocation->SizeBytes() <= allocation->GetBuffer().length - allocation->GetOffset();
}

} // namespace

id<MTLComputePipelineState> GeometryAssembly::TessellationRasterPipeline() {
    if (!tessellation_raster_pipeline) {
        auto library =
            compiler.CompileLibrary(TessellationRasterSource, "shadPS4:TessellationMesh");
        auto function = [[MTL4LibraryFunctionDescriptor alloc] init];
        function.library = library;
        function.name = @"metal4_pack_tessellation_mesh";
        auto descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.label = @"shadPS4:TessellationMesh";
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 32;
        tessellation_raster_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return tessellation_raster_pipeline;
}

void GeometryAssembly::AssembleTessellationRaster(const TessellationExports::Batch& batch,
                                                  const MeshShader& mesh, const BufferRange& errors,
                                                  const std::function<void(Result&&)>& consume,
                                                  u32 viewport_count) {
    const auto device = instance.GetDevice();
    const u32 width = batch.GetPrimitiveWidth();
    const u32 words = batch.GetVertexWords();
    const auto& domains = batch.GetDomains();
    u32 viewport_word{};
    if (mesh.viewport_extent[0] && mesh.viewport_extent[1]) {
        const auto output = std::ranges::find_if(mesh.attributes, [](const auto& attribute) {
            return attribute.attribute == Shader::IR::Attribute::ViewportIndex &&
                   !attribute.component && !attribute.stream;
        });
        if (output == mesh.attributes.end() || !output->word || output->word >= words ||
            !viewport_count || viewport_count > 16) {
            throw std::runtime_error("Native direct TES viewport packing interface is invalid");
        }
        viewport_word = output->word;
    }
    const u64 source_primitives = u64{domains.patch_count} * batch.GetPrimitiveSlotsPerPatch();
    const u64 source_words =
        u64{domains.patch_count} * TessellationExports::VerticesPerPatch * words;
    if (!consume || !batch.IsRaster() || batch.GetGuestRing() || !batch.IsValidated() ||
        !mesh.mesh || !mesh.shader || mesh.stream || width < 1 || width > 3 ||
        mesh.topology != width - 1 || mesh.vertex_words != words || !words ||
        mesh.mesh->vertices < width || mesh.mesh->vertices > 256 ||
        mesh.mesh->primitives < mesh.mesh->vertices / width ||
        !std::ranges::equal(
            mesh.attributes, batch.GetRasterAttributes(),
            [](const auto& left, const auto& right) {
                return left.attribute == right.attribute && left.component == right.component &&
                       left.stream == right.stream && left.word == right.word;
            }) ||
        !source_primitives || source_primitives > std::numeric_limits<u32>::max() ||
        !source_words || source_words > std::numeric_limits<u32>::max() ||
        !ValidAllocation(batch.GetExports(), device, source_words * sizeof(u32)) ||
        !ValidAllocation(batch.GetPrimitives(), device,
                         source_primitives * sizeof(TessellationPrimitive)) ||
        !errors.buffer || errors.buffer.device != device || errors.offset % sizeof(u32) ||
        errors.offset > errors.buffer.length || errors.size < sizeof(u32) ||
        errors.size > errors.buffer.length - errors.offset) {
        throw std::runtime_error("Invalid native direct TES mesh batch or raster interface");
    }
    for (const auto& allocation : {batch.GetExports(), batch.GetPrimitives()}) {
        if (allocation->GetBuffer() == errors.buffer &&
            allocation->GetOffset() < errors.offset + sizeof(u32) &&
            errors.offset < allocation->GetOffset() + allocation->SizeBytes()) {
            throw std::runtime_error("Native direct TES error storage aliases its input");
        }
    }
    const auto layout = GeometryAssemblyLayout(1, mesh.mesh->vertices, words, mesh.mesh->vertices,
                                               {mesh.topology, 0, 0, 0}, 1);
    const u64 capacity =
        std::min({u64{device.maxBufferLength} / layout.EmissionSizeBytes(),
                  u64{device.maxBufferLength} / layout.PrimitiveSizeBytes(),
                  (u64{32} << 20) / (layout.EmissionSizeBytes() + layout.PrimitiveSizeBytes()),
                  u64{std::numeric_limits<u32>::max()}});
    if (!capacity) {
        throw std::runtime_error("Native direct TES mesh exceeds buffer capacity");
    }
    const u32 per_job = mesh.mesh->vertices / width;
    const u64 primitive_capacity = capacity * per_job;
    const auto pipeline = TessellationRasterPipeline();
    for (u64 first = 0; first < source_primitives;) {
        const u32 count = u32(std::min(primitive_capacity, source_primitives - first));
        const u32 jobs = u32((u64{count} + per_job - 1) / per_job);
        auto emissions = Allocate(jobs, mesh.mesh->vertices, words, mesh.mesh->vertices,
                                  {mesh.topology, 0, 0, 0}, 1);
        const auto& parameters = emissions.GetParameters();
        const RasterParameters work{
            parameters,        u32(first),    count,          batch.GetPrimitiveSlotsPerPatch(),
            u32(source_words), viewport_word, viewport_count, mesh.last_provoking_vertex};
        auto uniforms = staging.Allocate(sizeof(work));
        std::memcpy(uniforms.UploadData().data(), &work, sizeof(work));
        auto output = std::make_shared<StagingBufferPool::Allocation>(
            staging.Allocate(parameters.PrimitiveSizeBytes()));
        const auto source = emissions.GetRange();
        ResourceBindings bindings{instance, 6, 0, 0};
        bindings.SetBuffer(0, batch.GetPrimitives()->GetBuffer(),
                           batch.GetPrimitives()->GetOffset());
        bindings.SetBuffer(1, batch.GetExports()->GetBuffer(), batch.GetExports()->GetOffset());
        bindings.SetBuffer(2, source.buffer, source.offset);
        bindings.SetBuffer(3, output->GetBuffer(), output->GetOffset());
        bindings.SetBuffer(4, uniforms.GetBuffer(), uniforms.GetOffset());
        scheduler.UseResource(errors.buffer);
        bindings.SetBuffer(5, errors.buffer, errors.offset);
        bindings.RetainUploadAllocation(batch.GetPrimitives());
        bindings.RetainUploadAllocation(batch.GetExports());
        emissions.Retain(bindings);
        bindings.RetainUploadAllocation(output);
        bindings.RetainUploadAllocation(std::move(uniforms));
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
        commands.BindCompute(pipeline, bindings);
        const u32 grid_width = std::min(jobs, 65535u);
        commands.Dispatch(MTLSizeMake(grid_width, (u64{jobs} + grid_width - 1) / grid_width, 1),
                          MTLSizeMake(32, 1, 1));
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
        consume({std::move(emissions), std::move(output)});
        first += count;
    }
}

} // namespace Metal
