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

constexpr std::string_view ViewportPackingSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct MeshParameters {
    uint invocations; uint max_vertices; uint vertex_words; uint mesh_vertices;
    uint topology[4]; uint mesh_counts[4]; uint mesh_offsets[4];
    uint stream_stride_words; uint mesh_stride_words; uint meshes_per_invocation; uint jobs;
};
struct Parameters {
    MeshParameters input; MeshParameters output; ulong first;
    uint count; uint slots; uint viewport_word; uint viewport_count; uint last_provoking; uint padding;
};
static_assert(sizeof(MeshParameters) == 80 && sizeof(Parameters) == 192);
kernel void metal4_pack_geometry_viewports(
    device const uint* emissions [[buffer(0)]], device const uint* primitives [[buffer(1)]],
    device uint* vertices [[buffer(2)]], device uint* packed [[buffer(3)]],
    constant Parameters& p [[buffer(4)]], device atomic_uint& validation_errors [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong job = ulong(group.y) * grid.x + group.x;
    if (job >= p.output.jobs) return;
    const uint width = p.output.topology[0] + 1u;
    const uint per_job = p.output.mesh_vertices / width;
    const uint first = uint(job) * per_job;
    const uint count = min(per_job, p.count - first);
    device uint* payload = vertices + job * 4u * p.output.stream_stride_words;
    device uint* destination = packed + job * p.output.mesh_stride_words;
    uint errors = 0;
    for (uint local = lane; local < count; local += 32u) {
        const ulong slot = p.first + first + local;
        const ulong source_job = slot / p.slots;
        const uint primitive = uint(slot % p.slots);
        bool valid = false;
        bool active = false;
        uint indices[3] = {};
        uint first_provoking = 0u, last_provoking = 0u, primitive_id = 0u;
        device const uint* source = emissions;
        if (source_job >= p.input.jobs) {
            errors |= 0x2000u;
        } else {
            device const uint* records = primitives + source_job * p.input.mesh_stride_words;
            const uint start = records[0], source_vertices = records[1], source_count = records[2];
            valid = !records[3] && start <= p.input.max_vertices &&
                source_vertices <= p.input.mesh_vertices &&
                source_vertices <= p.input.max_vertices - start && source_count <= p.slots &&
                (!source_count || source_vertices);
            if (!valid) errors |= 0x2000u;
            active = valid && primitive < source_count;
            if (active) {
                const ulong invocation = source_job / p.input.meshes_per_invocation;
                source = emissions + invocation * 4u * p.input.stream_stride_words +
                    4u + ulong(start) * p.input.vertex_words;
                device const uint* record = records + 4u + ulong(primitive) * 8u;
                primitive_id = record[5];
                uint first_match = 0xffffffffu, last_match = 0xffffffffu;
                for (uint component = 0; component < width; ++component) {
                    indices[component] = record[component];
                    valid &= indices[component] < source_vertices;
                    if (indices[component] == record[3]) first_match = component;
                    if (indices[component] == record[4]) last_match = component;
                }
                valid &= first_match < width && last_match < width;
                first_provoking = first_match;
                last_provoking = last_match;
                if (!valid) {
                    errors |= 0x2000u;
                } else {
                    const uint selected = record[p.last_provoking ? 4 : 3];
                    if (source[ulong(selected) * p.input.vertex_words + p.viewport_word] >= p.viewport_count) {
                        valid = false;
                        errors |= 0x1000u;
                    }
                }
            }
        }
        for (uint component = 0; component < width; ++component) {
            device uint* row = payload + 4u + ulong(local * width + component) * p.output.vertex_words;
            for (uint word = 0; word < p.output.vertex_words; ++word)
                row[word] = active && valid ? source[ulong(indices[component]) * p.input.vertex_words + word] : 0u;
        }
        device uint* record = destination + 4u + ulong(local) * 8u;
        for (uint component = 0; component < 3; ++component)
            record[component] = local * width + min(component, width - 1u);
        record[3] = active && valid ? local * width + first_provoking : 0xffffffffu;
        record[4] = active && valid ? local * width + last_provoking : 0xffffffffu;
        record[5] = active && valid ? primitive_id : 0u;
        record[6] = record[7] = 0u;
    }
    errors = simd_or(errors);
    if (!lane) {
        payload[0] = count * width;
        payload[1] = payload[2] = payload[3] = 0u;
        destination[0] = 0u;
        destination[1] = count * width;
        destination[2] = errors ? 0u : count;
        destination[3] = errors;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}
struct ViewportControl {
    MeshParameters mesh;
    uint unused[13]; uint dispatch[3]; uint mesh_dispatch[3]; uint flags;
};
static_assert(sizeof(ViewportControl) == 160);
bool viewport_layout_matches(device const MeshParameters& source, constant MeshParameters& capacity) {
    if (source.max_vertices != capacity.max_vertices || source.vertex_words != capacity.vertex_words ||
        source.mesh_vertices != capacity.mesh_vertices || source.stream_stride_words != capacity.stream_stride_words ||
        source.mesh_stride_words != capacity.mesh_stride_words || source.meshes_per_invocation != capacity.meshes_per_invocation)
        return false;
    for (uint stream = 0u; stream < 4u; ++stream)
        if (source.topology[stream] != capacity.topology[stream] ||
            source.mesh_counts[stream] != capacity.mesh_counts[stream] ||
            source.mesh_offsets[stream] != capacity.mesh_offsets[stream]) return false;
    return true;
}
kernel void metal4_prepare_viewport_dispatch(device const MeshParameters& input [[buffer(0)]],
    constant Parameters& capacity [[buffer(1)]], device Parameters& parameters [[buffer(2)]],
    device ViewportControl& control [[buffer(3)]], device atomic_uint& errors [[buffer(4)]]) {
    Parameters result = capacity;
    ViewportControl dispatch = {};
    result.count = result.output.invocations = result.output.jobs = 0u;
    bool empty = true;
    device const uchar* bytes = reinterpret_cast<device const uchar*>(&input);
    for (uint byte = 0u; byte < sizeof(MeshParameters); ++byte)
        empty = empty && bytes[byte] == 0u;
    result.input = input;
    bool valid = !capacity.padding && capacity.input.topology[0] <= 2u &&
        capacity.output.topology[0] == capacity.input.topology[0] &&
        capacity.input.mesh_vertices > capacity.input.topology[0] &&
        capacity.output.mesh_vertices > capacity.output.topology[0] &&
        capacity.output.mesh_vertices <= 256u && capacity.output.vertex_words == capacity.input.vertex_words &&
        capacity.slots && capacity.slots <= capacity.input.mesh_vertices - capacity.input.topology[0] &&
        capacity.viewport_word && capacity.viewport_word < capacity.input.vertex_words &&
        capacity.viewport_count && capacity.viewport_count <= 16u && capacity.last_provoking <= 1u &&
        capacity.output.meshes_per_invocation == 1u && capacity.output.mesh_counts[0] == 1u &&
        capacity.output.invocations == capacity.output.jobs &&
        capacity.first <= ulong(capacity.input.jobs) * capacity.slots &&
        capacity.count <= ulong(capacity.input.jobs) * capacity.slots - capacity.first;
    if (valid && !empty)
        valid = viewport_layout_matches(input, capacity.input) &&
            input.invocations <= capacity.input.invocations && input.jobs <= capacity.input.jobs &&
            ulong(input.invocations) * input.meshes_per_invocation == input.jobs;
    if (!valid) atomic_fetch_or_explicit(&errors, 0x2000u, memory_order_relaxed);
    else if (!empty) {
        const ulong total = ulong(input.jobs) * capacity.slots;
        const uint count = capacity.first < total ? uint(min(ulong(capacity.count), total - capacity.first)) : 0u;
        const uint per_job = capacity.output.mesh_vertices / (capacity.output.topology[0] + 1u);
        const uint jobs = uint((ulong(count) + per_job - 1u) / per_job);
        if (jobs > capacity.output.jobs) atomic_fetch_or_explicit(&errors, 0x2000u, memory_order_relaxed);
        else {
            result.count = count;
            result.output.invocations = result.output.jobs = jobs;
            if (jobs) {
                dispatch.mesh = result.output;
                const uint width = min(jobs, 65535u);
                for (uint component = 0u; component < 3u; ++component) {
                    const uint value = component == 0u ? width : component == 1u ? uint((ulong(jobs) + width - 1u) / width) : 1u;
                    dispatch.dispatch[component] = dispatch.mesh_dispatch[component] = value;
                }
                dispatch.flags = 1u;
            }
        }
    }
    parameters = result;
    control = dispatch;
}

)metal";

} // namespace

id<MTLComputePipelineState> GeometryAssembly::ViewportPackingPipeline() {
    if (!viewport_packing_pipeline) {
        auto library = compiler.CompileLibrary(ViewportPackingSource, "shadPS4:GeometryViewports");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_pack_geometry_viewports";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.label = @"shadPS4:GeometryViewports";
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 32;
        viewport_packing_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return viewport_packing_pipeline;
}

id<MTLComputePipelineState> GeometryAssembly::ViewportPackingDispatchPipeline() {
    if (!viewport_packing_dispatch_pipeline) {
        auto library =
            compiler.CompileLibrary(ViewportPackingSource, "shadPS4:GeometryViewportDispatch");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_prepare_viewport_dispatch";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.label = @"shadPS4:GeometryViewportDispatch";
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 1;
        viewport_packing_dispatch_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return viewport_packing_dispatch_pipeline;
}

void GeometryAssembly::PackViewportBatches(const Result& source, const MeshShader& mesh,
                                           const BufferRange& errors, u32 viewport_count,
                                           const std::function<void(Result&&)>& consume) {
    const auto& input = source.emissions.GetParameters();
    if (!consume || !mesh.mesh || !mesh.shader || mesh.stream || !mesh.viewport_extent[0] ||
        !mesh.viewport_extent[1] || !viewport_count || viewport_count > 16 ||
        input.vertex_words != mesh.vertex_words || input.mesh_vertices != mesh.mesh->vertices ||
        input.topology[0] != mesh.topology || mesh.topology > 2 ||
        input.mesh_vertices <= mesh.topology || input.mesh_counts[1] || input.mesh_counts[2] ||
        input.mesh_counts[3] || input.mesh_offsets[0] || !input.mesh_counts[0] ||
        input.meshes_per_invocation != input.mesh_counts[0] ||
        u64{input.invocations} * input.meshes_per_invocation != input.jobs) {
        throw std::runtime_error("Invalid native geometry viewport packing layout");
    }
    if (!input.jobs) {
        return;
    }
    const auto output = std::ranges::find_if(mesh.attributes, [](const auto& attribute) {
        return attribute.attribute == Shader::IR::Attribute::ViewportIndex &&
               !attribute.component && !attribute.stream;
    });
    const auto device = instance.GetDevice();
    const auto valid = [&](const BufferRange& range, u64 bytes) {
        return range.buffer && range.buffer.device == device && range.offset % sizeof(u32) == 0 &&
               range.offset <= range.buffer.length && range.size >= bytes &&
               range.size <= range.buffer.length - range.offset;
    };
    const auto vertices = source.emissions.GetRange();
    const BufferRange primitives{source.primitives ? source.primitives->GetBuffer() : nil,
                                 source.primitives ? source.primitives->GetOffset() : 0,
                                 source.primitives ? source.primitives->SizeBytes() : 0};
    if (output == mesh.attributes.end() || !output->word || output->word >= input.vertex_words ||
        !valid(vertices, input.EmissionSizeBytes()) ||
        !valid(primitives, input.PrimitiveSizeBytes()) || !valid(errors, sizeof(u32))) {
        throw std::runtime_error("Invalid native geometry viewport packing buffers");
    }
    const BufferRange indirect{source.indirect ? source.indirect->GetBuffer() : nil,
                               source.indirect ? source.indirect->GetOffset() : 0,
                               source.indirect ? source.indirect->SizeBytes() : 0};
    if (source.indirect &&
        (!valid(indirect, sizeof(RasterCapturePackWork)) ||
         source.retained_bytes < input.EmissionSizeBytes() + input.PrimitiveSizeBytes() +
                                     sizeof(RasterCapturePackWork) ||
         !source.primitive_slots || source.primitive_slots > input.mesh_vertices - mesh.topology)) {
        throw std::runtime_error("Invalid native geometry viewport indirect source");
    }
    for (const auto& range : {vertices, primitives, indirect}) {
        if (range.buffer == errors.buffer && range.offset < errors.offset + sizeof(u32) &&
            errors.offset < range.offset + range.size) {
            throw std::runtime_error("Native geometry viewport errors alias their source");
        }
    }
    const auto layout = GeometryAssemblyLayout(1, mesh.mesh->vertices, mesh.vertex_words,
                                               mesh.mesh->vertices, {mesh.topology, 0, 0, 0}, 1);
    const u64 budget = std::min(u64{32} << 20, u64{device.maxBufferLength});
    u64 available = u64{32} << 20;
    if (source.indirect) {
        const u64 minimum =
            ViewportPackingMinimumSizeBytes(mesh.vertex_words, mesh.mesh->vertices, mesh.topology);
        if (minimum > budget || source.retained_bytes > budget - minimum) {
            throw std::runtime_error("Native geometry viewport source exceeds aggregate storage");
        }
        available = budget - source.retained_bytes - ViewportPackingFixedSizeBytes();
    }
    const u64 jobs_capacity =
        std::min({u64{device.maxBufferLength} / layout.EmissionSizeBytes(),
                  u64{device.maxBufferLength} / layout.PrimitiveSizeBytes(),
                  available / (layout.EmissionSizeBytes() + layout.PrimitiveSizeBytes())});
    if (!jobs_capacity) {
        throw std::runtime_error("Native geometry viewport packing exceeds device capacity");
    }
    const u32 width = mesh.topology + 1;
    const u32 per_job = mesh.mesh->vertices / width;
    const u32 slots =
        source.indirect ? source.primitive_slots : input.mesh_vertices - mesh.topology;
    const u64 total = u64{input.jobs} * slots;
    const auto pipeline = ViewportPackingPipeline();
    for (u64 first = 0; first < total;) {
        const u32 count = u32(std::min(
            {total - first, jobs_capacity * per_job, u64{std::numeric_limits<u32>::max()}}));
        const u32 jobs = u32((u64{count} + per_job - 1) / per_job);
        auto emissions = Allocate(jobs, mesh.mesh->vertices, mesh.vertex_words, mesh.mesh->vertices,
                                  {mesh.topology, 0, 0, 0}, 1);
        const auto& parameters = emissions.GetParameters();
        const ViewportPackingParameters work{
            input, parameters,   first,          count,
            slots, output->word, viewport_count, mesh.last_provoking_vertex};
        auto uniforms =
            std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(sizeof(work)));
        std::memcpy(uniforms->UploadData().data(), &work, sizeof(work));
        auto packed = std::make_shared<StagingBufferPool::Allocation>(
            staging.Allocate(parameters.PrimitiveSizeBytes()));
        std::shared_ptr<StagingBufferPool::Allocation> runtime, control;
        if (source.indirect) {
            runtime = std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
                sizeof(ViewportPackingParameters), alignof(ViewportPackingParameters)));
            control = std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
                sizeof(RasterCapturePackWork), alignof(RasterCapturePackWork)));
            ResourceBindings prepare{instance, 5, 0, 0};
            prepare.SetBuffer(0, indirect.buffer, indirect.offset);
            prepare.SetBuffer(1, uniforms->GetBuffer(), uniforms->GetOffset());
            prepare.SetBuffer(2, runtime->GetBuffer(), runtime->GetOffset());
            prepare.SetBuffer(3, control->GetBuffer(), control->GetOffset());
            scheduler.UseResource(errors.buffer);
            prepare.SetBuffer(4, errors.buffer, errors.offset);
            prepare.RetainUploadAllocation(source.indirect);
            prepare.RetainUploadAllocation(uniforms);
            prepare.RetainUploadAllocation(runtime);
            prepare.RetainUploadAllocation(control);
            scheduler.EndEncoding();
            scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
            commands.BindCompute(ViewportPackingDispatchPipeline(), prepare);
            commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
            scheduler.ProducerBarrier(MTLStageDispatch, MTLStageDispatch);
        }
        const auto destination = emissions.GetRange();
        ResourceBindings bindings{instance, 6, 0, 0};
        bindings.SetBuffer(0, vertices.buffer, vertices.offset);
        bindings.SetBuffer(1, primitives.buffer, primitives.offset);
        bindings.SetBuffer(2, destination.buffer, destination.offset);
        bindings.SetBuffer(3, packed->GetBuffer(), packed->GetOffset());
        bindings.SetBuffer(4, runtime ? runtime->GetBuffer() : uniforms->GetBuffer(),
                           runtime ? runtime->GetOffset() : uniforms->GetOffset());
        scheduler.UseResource(errors.buffer);
        bindings.SetBuffer(5, errors.buffer, errors.offset);
        source.emissions.Retain(bindings);
        bindings.RetainUploadAllocation(source.primitives);
        emissions.Retain(bindings);
        bindings.RetainUploadAllocation(packed);
        if (runtime) {
            bindings.RetainUploadAllocation(runtime);
            bindings.RetainUploadAllocation(control);
        }
        bindings.RetainUploadAllocation(uniforms);
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
        commands.BindCompute(pipeline, bindings);
        const u32 grid_width = std::min(jobs, 65535u);
        if (control) {
            commands.DispatchIndirect(
                {control->GetBuffer(),
                 control->GetOffset() + offsetof(RasterCapturePackWork, dispatch),
                 sizeof(RasterCapturePackWork::dispatch)},
                MTLSizeMake(32, 1, 1));
        } else {
            commands.Dispatch(MTLSizeMake(grid_width, (u64{jobs} + grid_width - 1) / grid_width, 1),
                              MTLSizeMake(32, 1, 1));
        }
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
        const u64 retained = control ? source.retained_bytes + ViewportPackingFixedSizeBytes() +
                                           parameters.EmissionSizeBytes() +
                                           parameters.PrimitiveSizeBytes()
                                     : 0;
        consume({std::move(emissions), std::move(packed), std::move(control), retained, per_job,
                 source.primitive_ids});
        first += count;
    }
}

} // namespace Metal
