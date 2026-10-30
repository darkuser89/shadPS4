// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
constexpr std::string_view GeometryCopyInputsSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint work_items; uint invocations; uint max_vertices;
    uint components; uint ring_words; uint base_word_offset;
};
struct KernelWave { uint ring_offset; uint wave_id; };
struct CopyWave { uint system_sgprs[16]; uint vgprs[3]; float viewport[4]; };
struct CopyInput {
    uint system_sgprs[16]; uint vgprs[4]; uint invocation; uint vertex_ordinal; float viewport[4];
};
static_assert(sizeof(Parameters) == 24);
static_assert(sizeof(KernelWave) == 8 && sizeof(CopyWave) == 92 && sizeof(CopyInput) == 104);
kernel void metal4_prepare_geometry_copy_inputs(
    device const KernelWave* geometry_waves [[buffer(0)]],
    device const CopyWave* copy_waves [[buffer(1)]], device CopyInput* inputs [[buffer(2)]],
    constant Parameters& p [[buffer(3)]], device atomic_uint& errors [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    const ulong item = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (item >= p.work_items) return;
    const uint invocation = uint(item) / p.max_vertices;
    const uint ordinal = uint(item) % p.max_vertices;
    const uint lane = invocation % 64;
    const uint scalar = geometry_waves[invocation / 64].ring_offset;
    const ulong index = ulong(p.base_word_offset) + scalar / 4 +
                        ulong(lane / 16) * p.components * p.max_vertices * 16 +
                        ulong(ordinal) * 16 + lane % 16;
    const ulong last_byte = index * 4 + ulong(p.components - 1) * p.max_vertices * 64 + 3;
    CopyInput result{};
    if (!(scalar & 3) && last_byte < ulong(p.ring_words) * 4) {
        const CopyWave wave = copy_waves[item / 64];
        for (uint i = 0; i < 16; ++i) result.system_sgprs[i] = wave.system_sgprs[i];
        result.vgprs[0] = uint(index);
        for (uint i = 0; i < 3; ++i) result.vgprs[i + 1] = wave.vgprs[i];
        result.invocation = invocation;
        result.vertex_ordinal = ordinal;
        for (uint i = 0; i < 4; ++i) result.viewport[i] = wave.viewport[i];
    } else {
        result.invocation = 0xffffffffu;
        atomic_fetch_or_explicit(&errors, 8u, memory_order_relaxed);
    }
    inputs[item] = result;
}
struct CopyDispatchParameters {
    uint work_items; uint invocations; uint max_vertices; uint stream; uint stream_stride;
};
static_assert(sizeof(CopyDispatchParameters) == 20);
kernel void metal4_prepare_geometry_copy_dispatch(
    device const CopyInput* inputs [[buffer(0)]], device const uint* emissions [[buffer(1)]],
    device uint* group_map [[buffer(2)]], device atomic_uint* dispatch [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]],
    constant CopyDispatchParameters& p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong original_group = ulong(group.y) * grid.x + group.x;
    if (original_group >= p.work_items / 64u)
        return;
    threadgroup atomic_uint active;
    threadgroup atomic_uint failures;
    if (!lane) {
        atomic_store_explicit(&active, 0u, memory_order_relaxed);
        atomic_store_explicit(&failures, 0u, memory_order_relaxed);
        if (!original_group) {
            atomic_store_explicit(dispatch + 1, 1u, memory_order_relaxed);
            atomic_store_explicit(dispatch + 2, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const CopyInput input = inputs[original_group * 64u + lane];
    bool enabled = false;
    uint failure = 0;
    if (input.invocation >= p.invocations || input.vertex_ordinal >= p.max_vertices) {
        failure = 0x100u;
    } else {
        const uint count = emissions[(ulong(input.invocation) * 4u + p.stream) * p.stream_stride];
        if (count > p.max_vertices)
            failure = 0x200u;
        else
            enabled = input.vertex_ordinal < count;
    }
    if (enabled)
        atomic_fetch_add_explicit(&active, 1u, memory_order_relaxed);
    if (failure)
        atomic_fetch_or_explicit(&failures, failure, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) {
        const uint error = atomic_load_explicit(&failures, memory_order_relaxed);
        if (error)
            atomic_fetch_or_explicit(errors, error, memory_order_relaxed);
        else if (atomic_load_explicit(&active, memory_order_relaxed)) {
            const uint slot = atomic_fetch_add_explicit(dispatch, 1u, memory_order_relaxed);
            group_map[slot] = uint(original_group);
        }
    }
}
)metal";
} // namespace

id<MTLComputePipelineState> GeometryAssembly::CopyInputsPipeline() {
    if (!copy_inputs_pipeline) {
        auto library =
            compiler.CompileLibrary(GeometryCopyInputsSource, "shadPS4:GeometryCopyInputs");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_prepare_geometry_copy_inputs";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        copy_inputs_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return copy_inputs_pipeline;
}

std::shared_ptr<StagingBufferPool::Allocation> GeometryAssembly::PrepareCopyInputs(
    const GeometryCopyParameters& parameters, const BufferRange& geometry_waves,
    const BufferRange& copy_waves, const BufferRange& errors, bool compact_workgroups) {
    const auto& p = parameters;
    if (p.ring_words > std::numeric_limits<u32>::max() / sizeof(u32) ||
        p != GeometryCopyWorkLayout(p.invocations, p.max_vertices, p.components,
                                    p.ring_words * sizeof(u32), p.base_word_offset)) {
        throw std::runtime_error("Invalid native geometry copy parameters");
    }
    const auto valid = [&](const BufferRange& range, u64 bytes) {
        return range.buffer && range.buffer.device == instance.GetDevice() &&
               range.offset % sizeof(u32) == 0 && range.offset <= range.buffer.length &&
               range.size >= bytes && range.size <= range.buffer.length - range.offset;
    };
    const u64 geometry_bytes = u64{p.invocations / 64} * sizeof(GeometryKernelWave);
    const u64 wave_bytes = u64{p.work_items / 64} * sizeof(GeometryCopyWave);
    const u64 input_bytes = u64{p.work_items} * sizeof(GeometryCopyInput) +
                            (compact_workgroups ? u64{p.work_items / 64} * sizeof(u32) : 0);
    const auto overlaps_errors = [&](const BufferRange& range, u64 bytes) {
        return range.buffer == errors.buffer && range.offset < errors.offset + sizeof(u32) &&
               errors.offset < range.offset + bytes;
    };
    if (!valid(geometry_waves, geometry_bytes) || !valid(copy_waves, wave_bytes) ||
        !valid(errors, sizeof(u32)) || input_bytes > instance.GetDevice().maxBufferLength ||
        input_bytes / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1 ||
        overlaps_errors(geometry_waves, geometry_bytes) ||
        overlaps_errors(copy_waves, wave_bytes)) {
        throw std::runtime_error("Invalid native geometry copy wave input ranges");
    }
    const auto compute_pipeline = CopyInputsPipeline();
    scheduler.UseResource(geometry_waves.buffer);
    scheduler.UseResource(copy_waves.buffer);
    scheduler.UseResource(errors.buffer);
    auto data = staging.Allocate(sizeof(p));
    std::memcpy(data.UploadData().data(), &p, sizeof(p));
    auto inputs = std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(input_bytes));
    ResourceBindings bindings{instance, 5, 0, 0};
    bindings.SetBuffer(0, geometry_waves.buffer, geometry_waves.offset);
    bindings.SetBuffer(1, copy_waves.buffer, copy_waves.offset);
    bindings.SetBuffer(2, inputs->GetBuffer(), inputs->GetOffset());
    bindings.SetBuffer(3, data.GetBuffer(), data.GetOffset());
    bindings.SetBuffer(4, errors.buffer, errors.offset);
    bindings.RetainUploadAllocation(std::move(data));
    bindings.RetainUploadAllocation(inputs);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(compute_pipeline, bindings);
    const u32 groups = p.work_items / 64;
    const u32 width = std::min(groups, 65535u);
    const u32 height = u32((u64{groups} + width - 1) / width);
    commands.Dispatch(MTLSizeMake(width, height, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return inputs;
}

id<MTLComputePipelineState> GeometryAssembly::CopyDispatchPipeline() {
    if (!copy_dispatch_pipeline) {
        auto library =
            compiler.CompileLibrary(GeometryCopyInputsSource, "metal4_geometry_copy_dispatch");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_prepare_geometry_copy_dispatch";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        copy_dispatch_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return copy_dispatch_pipeline;
}

void GeometryAssembly::BuildCopyDispatch(PreparedKernel& kernel) {
    const auto& request = *kernel.copy_dispatch;
    const auto& config = request.config;
    struct Parameters {
        u32 work_items;
        u32 invocations;
        u32 max_vertices;
        u32 stream;
        u32 stream_stride;
    };
    static_assert(sizeof(Parameters) == 20);
    const Parameters parameters{config.work_items, config.emissions.invocations,
                                config.emissions.max_vertices, config.stream,
                                config.emissions.stream_stride_words};
    auto data = staging.Allocate(sizeof(parameters), alignof(Parameters));
    std::memcpy(data.UploadData().data(), &parameters, sizeof(parameters));
    ResourceBindings bindings{instance, 6, 0, 0};
    bindings.SetBuffer(0, request.inputs->GetBuffer(), request.inputs->GetOffset());
    const auto source = request.source.GetRange();
    bindings.SetBuffer(1, source.buffer, source.offset);
    bindings.SetBuffer(2, request.inputs->GetBuffer(),
                       request.inputs->GetOffset() + config.RegisterSizeBytes());
    bindings.SetBuffer(3, kernel.arguments->GetBuffer(), kernel.arguments->GetOffset());
    bindings.SetBuffer(4, request.errors.buffer, request.errors.offset);
    bindings.SetBuffer(5, data.GetBuffer(), data.GetOffset());
    request.source.Retain(bindings);
    bindings.RetainUploadAllocation(request.inputs);
    bindings.RetainUploadAllocation(kernel.arguments);
    bindings.RetainUploadAllocation(std::move(data));
    scheduler.UseResource(request.errors.buffer);
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.FillBuffer({kernel.arguments->GetBuffer(), kernel.arguments->GetOffset(),
                         kernel.arguments->SizeBytes()},
                        0);
    commands.BindCompute(CopyDispatchPipeline(), bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageDispatch);
    const u32 groups = config.work_items / 64;
    const u32 width = std::min(groups, 65535u);
    commands.Dispatch(MTLSizeMake(width, (u64{groups} + width - 1) / width, 1),
                      MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
}

GeometryAssembly::Emissions GeometryAssembly::Copy(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    const Emissions& emissions, const GeometryCopyParameters& parameters,
    const BufferRange& geometry_waves, const BufferRange& copy_waves, const BufferRange& errors,
    bool neo_mode, std::span<const GuestBufferExtent> guest_extents) {
    if (!shader.geometry_copy || emissions.GetParameters() != shader.geometry_copy->emissions ||
        parameters.invocations != shader.geometry_copy->emissions.invocations ||
        parameters.max_vertices != shader.geometry_copy->emissions.max_vertices ||
        parameters.work_items != shader.geometry_copy->work_items) {
        throw std::runtime_error("Native copy work differs from its compiled kernel");
    }
    auto inputs = PrepareCopyInputs(parameters, geometry_waves, copy_waves, errors,
                                    shader.geometry_copy->compact_workgroups);
    auto prepared = PrepareCopy(shader_cache, resources, shader, emissions, std::move(inputs),
                                neo_mode, 0, guest_extents, {}, errors);
    const std::array kernels{&prepared.kernel};
    Execute(kernels);
    return std::move(prepared.emissions);
}

} // namespace Metal
