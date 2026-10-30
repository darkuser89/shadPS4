// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tessellation_geometry.h"

namespace Metal {
namespace {

struct Parameters {
    u32 first;
    u32 count;
    u32 source_primitives;
    u32 source_words;
    u32 width;
    u32 vertex_words;
    u32 export_items;
    u32 geometry_items;
    u32 gs_invocations;
    u32 geometry_wave_bytes;
};
static_assert(sizeof(Parameters) == 40);

constexpr std::string_view GeometrySource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Primitive { uint vertex_offsets[3]; uint patch; uint instance; uint active; };
struct Input { uint ring_offset; uint wave_id; uint offsets[6]; uint patch; uint invocation; uint active; };
struct Parameters {
    uint first; uint count; uint source_primitives; uint source_words; uint width;
    uint vertex_words; uint export_items; uint geometry_items; uint gs_invocations;
    uint geometry_wave_bytes;
};
static_assert(sizeof(Primitive) == 24);
static_assert(sizeof(Input) == 44);
static_assert(sizeof(Parameters) == 40);
kernel void gather_tessellation_geometry_exports(
    device const Primitive* primitives [[buffer(0)]],
    device const uint* source [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device atomic_uint* errors [[buffer(6)]],
    constant Parameters& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong item = (ulong(group.y) * grid.x + group.x) * 64u + lane;
    if (item >= ulong(p.export_items) * p.vertex_words)
        return;
    const uint row = uint(item / (ulong(p.vertex_words) * 64u));
    const uint component = uint(item / 64u) % p.vertex_words;
    const uint packed = row * 64u + lane;
    uint value = 0;
    uint error = 0;
    if (packed < p.count * p.width) {
        const uint primitive_index = p.first + packed / p.width;
        if (primitive_index >= p.source_primitives) {
            error = 1u;
        } else {
            const Primitive primitive = primitives[primitive_index];
            if (primitive.active > 1u) {
                error = 2u;
            } else if (primitive.active) {
                const uint base = primitive.vertex_offsets[packed % p.width];
                const ulong address = ulong(base) + ulong(component) * 64u;
                if (base % (p.vertex_words * 64u) >= 64u || address >= p.source_words)
                    error = 4u;
                else
                    value = source[address];
            }
        }
    }
    output[item] = value;
    if (error)
        atomic_fetch_or_explicit(errors, error, memory_order_relaxed);
}
kernel void build_tessellation_geometry_inputs(
    device const Primitive* primitives [[buffer(0)]],
    device Input* inputs [[buffer(3)]], device uint* group_map [[buffer(4)]],
    device atomic_uint* dispatch [[buffer(5)]], device atomic_uint* errors [[buffer(6)]],
    constant Parameters& p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint item = group * 64u + lane;
    if (group >= p.geometry_items / 64u)
        return;
    threadgroup atomic_uint active;
    threadgroup atomic_uint failures;
    if (!lane) {
        atomic_store_explicit(&active, 0u, memory_order_relaxed);
        atomic_store_explicit(&failures, 0u, memory_order_relaxed);
        if (!group) {
            atomic_store_explicit(dispatch + 1, 1u, memory_order_relaxed);
            atomic_store_explicit(dispatch + 2, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Input result = {};
    uint error = 0;
    if (item < p.count * p.gs_invocations &&
        !atomic_load_explicit(errors, memory_order_relaxed)) {
        const uint primitive_index = p.first + item / p.gs_invocations;
        if (primitive_index >= p.source_primitives) {
            error = 8u;
        } else {
            const Primitive primitive = primitives[primitive_index];
            if (primitive.active > 1u) {
                error = 16u;
            } else if (primitive.active) {
                result.ring_offset = group * p.geometry_wave_bytes;
                result.wave_id = group;
                for (uint slot = 0; slot < p.width; ++slot) {
                    const uint packed = (item / p.gs_invocations) * p.width + slot;
                    result.offsets[slot] =
                        (packed / 64u) * p.vertex_words * 64u + packed % 64u;
                }
                result.patch = primitive.patch;
                result.invocation = item % p.gs_invocations;
                result.active = 1u;
            }
        }
    }
    inputs[item] = result;
    if (result.active)
        atomic_fetch_add_explicit(&active, 1u, memory_order_relaxed);
    if (error)
        atomic_fetch_or_explicit(&failures, error, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) {
        const uint failure = atomic_load_explicit(&failures, memory_order_relaxed);
        if (failure)
            atomic_fetch_or_explicit(errors, failure, memory_order_relaxed);
        else if (atomic_load_explicit(&active, memory_order_relaxed)) {
            const uint slot = atomic_fetch_add_explicit(dispatch, 1u, memory_order_relaxed);
            group_map[slot] = group;
        }
    }
}
)metal";

bool ValidAllocation(const std::shared_ptr<StagingBufferPool::Allocation>& allocation,
                     id<MTLDevice> device, u64 bytes) {
    return allocation && allocation->GetBuffer() && allocation->GetBuffer().device == device &&
           allocation->SizeBytes() >= bytes && allocation->GetOffset() % alignof(u32) == 0 &&
           allocation->GetOffset() <= allocation->GetBuffer().length &&
           allocation->SizeBytes() <= allocation->GetBuffer().length - allocation->GetOffset();
}

} // namespace

TessellationGeometry::TessellationGeometry(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

u32 TessellationGeometry::MaxBatchPrimitives(const AmdGpu::Regs& regs,
                                             const TessellationExports::Batch& batch) const {
    const u32 width = batch.GetPrimitiveWidth();
    const u32 invocations =
        regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
    if (!batch.IsValidated() || !batch.GetGuestRing() || width < 1 || width > 3 ||
        !batch.GetVertexWords() || regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs ||
        regs.vgt_esgs_ring_itemsize != batch.GetVertexWords()) {
        throw std::runtime_error("Invalid Metal tessellation geometry batch planning context");
    }
    const auto topology = width == 1   ? AmdGpu::PrimitiveType::PointList
                          : width == 2 ? AmdGpu::PrimitiveType::LineList
                                       : AmdGpu::PrimitiveType::TriangleList;
    const auto rings =
        BuildGeometryRingLayout(regs, GeometryDrawLayout(topology, width, 1, invocations, 0, 0, 0));
    const u64 source_primitives =
        u64{batch.GetDomains().patch_count} * batch.GetPrimitiveSlotsPerPatch();
    u64 upper = std::min({source_primitives,
                          std::min(rings.export_capacity, batch.GetGuestRing()->size) /
                              rings.export_wave_bytes * 64 / width,
                          rings.geometry_capacity / rings.geometry_wave_bytes * 64 / invocations});
    upper =
        std::min<u64>(upper, (std::numeric_limits<u32>::max() - 63) / std::max(width, invocations));
    const auto fits = [&](u64 count) {
        const u64 exports = (count * width + 63) & ~u64{63};
        const u64 geometry = (count * invocations + 63) & ~u64{63};
        const u64 export_bytes = exports * batch.GetVertexWords() * sizeof(u32);
        const u64 input_bytes =
            geometry * sizeof(GeometryKernelInput) + geometry / 64 * sizeof(u32);
        const u64 wave_bytes = geometry / 64 * sizeof(GeometryKernelWave);
        const u64 capacity = instance.GetDevice().maxBufferLength;
        return export_bytes <= capacity && input_bytes <= capacity && wave_bytes <= capacity &&
               export_bytes + input_bytes + wave_bytes + 4 * sizeof(u32) <= 32 * 1024 * 1024;
    };
    u64 lower{};
    while (lower < upper) {
        const u64 middle = lower + (upper - lower + 1) / 2;
        if (fits(middle)) {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    return u32(lower);
}

void TessellationGeometry::RunBatches(const AmdGpu::Regs& regs,
                                      const TessellationExports::Batch& batch,
                                      const std::function<void(Inputs&&)>& consume,
                                      u32 maximum_primitives) {
    if (!consume) {
        throw std::runtime_error("Metal tessellation geometry requires a batch consumer");
    }
    const u32 capacity = std::min(MaxBatchPrimitives(regs, batch), maximum_primitives);
    const u64 count = u64{batch.GetDomains().patch_count} * batch.GetPrimitiveSlotsPerPatch();
    if (!capacity || count > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Metal tessellation geometry cannot fit one bounded batch");
    }
    for (u32 first = 0; first < count;) {
        const u32 size = std::min(capacity, u32(count - first));
        consume(Prepare(regs, batch, first, size));
        first += size;
    }
}

TessellationGeometry::Inputs TessellationGeometry::Prepare(const AmdGpu::Regs& regs,
                                                           const TessellationExports::Batch& batch,
                                                           u32 first_primitive,
                                                           u32 primitive_count) {
    const auto device = instance.GetDevice();
    const u32 width = batch.GetPrimitiveWidth();
    const u32 words = batch.GetVertexWords();
    const u64 source_primitives =
        u64{batch.GetDomains().patch_count} * batch.GetPrimitiveSlotsPerPatch();
    if (!batch.IsValidated() || !batch.GetGuestRing() || width < 1 || width > 3 || !words ||
        regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs ||
        regs.vgt_esgs_ring_itemsize != words ||
        regs.tess_config.type != batch.GetDomains().config.tessellation.type ||
        regs.tess_config.partitioning != batch.GetDomains().config.tessellation.partitioning ||
        regs.tess_config.topology != batch.GetDomains().config.tessellation.topology ||
        !primitive_count || primitive_count > std::numeric_limits<u32>::max() / width ||
        source_primitives > std::numeric_limits<u32>::max() ||
        first_primitive > source_primitives ||
        primitive_count > source_primitives - first_primitive ||
        !ValidAllocation(batch.GetPrimitives(), device,
                         source_primitives * sizeof(TessellationPrimitive)) ||
        !ValidAllocation(batch.GetExports(), device,
                         u64{batch.GetDomains().patch_count} *
                             TessellationExports::VerticesPerPatch * words * sizeof(u32)) ||
        batch.GetExports()->SizeBytes() / sizeof(u32) > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Metal tessellation geometry inputs do not match their exports");
    }
    const auto topology = width == 1   ? AmdGpu::PrimitiveType::PointList
                          : width == 2 ? AmdGpu::PrimitiveType::LineList
                                       : AmdGpu::PrimitiveType::TriangleList;
    const u32 invocations =
        regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
    const auto draw =
        GeometryDrawLayout(topology, primitive_count * width, 1, invocations, 0, 0, 0);
    const auto rings = BuildGeometryRingLayout(regs, draw);
    const u64 export_bytes = u64{draw.export_items} * words * sizeof(u32);
    const u64 register_bytes = u64{draw.geometry_items} * sizeof(GeometryKernelInput);
    const u64 groups = draw.geometry_items / 64;
    const u64 input_bytes = register_bytes + groups * sizeof(u32);
    const u64 wave_bytes = groups * sizeof(GeometryKernelWave);
    if (export_bytes > batch.GetGuestRing()->size || export_bytes > device.maxBufferLength ||
        input_bytes > device.maxBufferLength || wave_bytes > device.maxBufferLength ||
        input_bytes / sizeof(u32) > std::numeric_limits<u32>::max() ||
        export_bytes + input_bytes + wave_bytes + 4 * sizeof(u32) > 32 * 1024 * 1024) {
        throw std::runtime_error("Metal tessellation geometry batch exceeds native storage bounds");
    }
    if (!export_pipeline || !input_pipeline) {
        auto library = compiler.CompileLibrary(GeometrySource, "metal4_tessellation_geometry");
        const auto compile = [&](NSString* name) {
            auto function = [MTL4LibraryFunctionDescriptor new];
            function.library = library;
            function.name = name;
            auto descriptor = [MTL4ComputePipelineDescriptor new];
            descriptor.computeFunctionDescriptor = function;
            descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
            descriptor.maxTotalThreadsPerThreadgroup = 64;
            return compiler.CompileComputePipeline(descriptor);
        };
        export_pipeline = compile(@"gather_tessellation_geometry_exports");
        input_pipeline = compile(@"build_tessellation_geometry_inputs");
    }
    Inputs result{draw, rings, *batch.GetGuestRing()};
    using Attribute = Shader::IR::Attribute;
    const u64 last_export =
        u64{draw.export_items / 64 - 1} * (rings.export_wave_bytes / sizeof(u32)) + 63;
    for (u32 slot = 0; slot < 6; ++slot) {
        result.address_envelope.push_back(
            {Attribute::GeometryVertexOffset, slot, {0, slot < width ? u32(last_export) : 0}});
    }
    result.address_envelope.push_back(
        {Attribute::GeometryRingOffset, 0, {0, u32((groups - 1) * rings.geometry_wave_bytes)}});
    result.address_envelope.push_back({Attribute::GeometryWaveId, 0, {0, u32(groups - 1)}});
    result.address_envelope.push_back(
        {Attribute::PrimitiveId, 0, {0, std::numeric_limits<u32>::max()}});
    result.address_envelope.push_back({Attribute::InvocationId, 0, {0, invocations - 1}});
    result.address_envelope.push_back({Attribute::GeometryInputActive, 0, {0, 1}});
    result.exports = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(export_bytes, alignof(u32)));
    result.registers = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(input_bytes, alignof(u32)));
    result.arguments = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(sizeof(MTLDispatchThreadgroupsIndirectArguments), alignof(u32)));
    result.geometry_waves =
        std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(wave_bytes, alignof(u32)));
    for (u32 group = 0; group < groups; ++group) {
        const GeometryKernelWave wave{group * rings.geometry_wave_bytes, group};
        std::memcpy(result.geometry_waves->UploadData().data() + u64{group} * sizeof(wave), &wave,
                    sizeof(wave));
    }
    auto errors =
        std::make_shared<StagingBufferPool::Allocation>(staging.AllocateReadback(sizeof(u32)));
    const Parameters params{first_primitive,
                            primitive_count,
                            u32(source_primitives),
                            u32(batch.GetExports()->SizeBytes() / sizeof(u32)),
                            width,
                            words,
                            draw.export_items,
                            draw.geometry_items,
                            invocations,
                            rings.geometry_wave_bytes};
    auto parameters = staging.Allocate(sizeof(params), alignof(Parameters));
    std::memcpy(parameters.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 8, 0, 0};
    const std::array allocations{
        batch.GetPrimitives(), batch.GetExports(), result.exports, result.registers,
        result.registers,      result.arguments,   errors};
    for (u32 index = 0; index < allocations.size(); ++index) {
        bindings.SetBuffer(index, allocations[index]->GetBuffer(),
                           allocations[index]->GetOffset() + (index == 4 ? register_bytes : 0));
        bindings.RetainUploadAllocation(allocations[index]);
    }
    bindings.SetBuffer(7, parameters.GetBuffer(), parameters.GetOffset());
    bindings.RetainUploadAllocation(std::move(parameters));
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.FillBuffer({errors->GetBuffer(), errors->GetOffset(), sizeof(u32)}, 0);
    commands.FillBuffer({result.arguments->GetBuffer(), result.arguments->GetOffset(),
                         result.arguments->SizeBytes()},
                        0);
    commands.BindCompute(export_pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageDispatch);
    const u64 export_groups = export_bytes / (64 * sizeof(u32));
    const u32 grid_width = u32(std::min<u64>(export_groups, 65535));
    commands.Dispatch(MTLSizeMake(grid_width, (export_groups + grid_width - 1) / grid_width, 1),
                      MTLSizeMake(64, 1, 1));
    scheduler.EncoderBarrier(MTLStageDispatch, MTLStageDispatch);
    commands.BindCompute(input_pipeline, bindings);
    commands.Dispatch(MTLSizeMake(groups, 1, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.Finish();
    u32 failure;
    std::memcpy(&failure, errors->ReadbackData().data(), sizeof(failure));
    if (failure) {
        throw std::runtime_error("Metal tessellation geometry gather/input validation failed");
    }
    return result;
}

} // namespace Metal
