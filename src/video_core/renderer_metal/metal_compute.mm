// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_metal/metal_compute.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
constexpr std::string_view DispatchArgumentsSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void metal4_dispatch_extent(device const uint* dimensions [[buffer(0)]],
                                   device uint* extent [[buffer(1)]],
                                   constant uint* start [[buffer(2)]]) {
    for (uint component = 0; component < 3; ++component) {
        const uint end = dimensions[component];
        extent[component] = end > start[component] ? end - start[component] : 0;
    }
}
)metal";

void ValidateDispatchMode(const AmdGpu::ComputeProgram& program) {
    if (program.dispatch_initiator & (1u << 1)) {
        throw std::runtime_error("Metal partial compute threadgroups require native lowering");
    }
    if (program.dispatch_initiator & (1u << 3)) {
        throw std::runtime_error("Metal ordered-append compute requires native wave-ID lowering");
    }
    if (program.dispatch_initiator & (1u << 5)) {
        throw std::runtime_error(
            "Metal thread-dimension compute requires native dispatch lowering");
    }
}
} // namespace

ComputeRenderer::ComputeRenderer(const Instance& instance, Scheduler& scheduler_,
                                 ShaderCache& shader_cache_, GuestResources& resources_,
                                 BufferCache& buffer_cache_, AmdGpu::Liverpool& liverpool_)
    : instance{instance}, scheduler{scheduler_}, shader_cache{shader_cache_}, resources{resources_},
      buffer_cache{buffer_cache_}, liverpool{liverpool_}, commands{instance, scheduler},
      compiler{instance}, staging{instance, scheduler} {}

ShaderCache::Variant& ComputeRenderer::Shader() {
    const auto& program = liverpool.GetCsRegs();
    if (!program.Address<u32*>()) {
        throw std::runtime_error("Metal guest compute requires a valid shader program");
    }
    Shader::RuntimeInfo runtime{};
    runtime.Initialize(Shader::HwStage::Compute, Shader::SwStage::Compute);
    runtime.props.num_user_data = program.settings.num_user_regs;
    runtime.props.num_allocated_vgprs = (program.settings.num_vgprs + 1) * 4;
    runtime.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
    runtime.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
    runtime.props.fp_round_mode32 = program.settings.fp_round_mode32;
    runtime.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    runtime.props.dx10_clamp = program.settings.dx10_clamp;
    runtime.hw.cs.workgroup_size = {program.num_thread_x.full, program.num_thread_y.full,
                                    program.num_thread_z.full};
    runtime.hw.cs.tgid_enable = {program.IsTgidEnabled(0), program.IsTgidEnabled(1),
                                 program.IsTgidEnabled(2)};
    runtime.hw.cs.shared_memory_size = program.SharedMemSize();
    Shader::Backend::Bindings bindings{};
    return shader_cache.GetShader(
        Shader::HwStage::Compute, Shader::SwStage::Compute,
        shader_cache.GetParams(program.Address<VAddr>(), program.user_data), runtime, bindings);
}

std::unique_ptr<ResourceBindings> ComputeRenderer::Bind(ShaderCache::Variant& variant,
                                                        GuestResources::Stage& stage) {
    std::array stages{&stage};
    resources.MarkWrites(stages);
    auto snapshot = resources.Snapshot(stage);
    id<MTLComputePipelineState> pipeline = shader_cache.GetComputePipeline(variant);
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    commands.BindCompute(pipeline, *snapshot);
    return snapshot;
}

void ComputeRenderer::DispatchDirect(bool neo_mode) {
    const auto& program = liverpool.GetCsRegs();
    if (!(program.dispatch_initiator & 1)) {
        return;
    }
    const auto extent = program.WorkgroupExtent();
    if (!extent[0] || !extent[1] || !extent[2]) {
        return;
    }
    ValidateDispatchMode(program);
    auto& variant = Shader();
    auto stage = resources.Prepare(variant, neo_mode);
    std::array stages{stage.get()};
    resources.Acquire(stages);
    const auto snapshot = Bind(variant, *stage);
    const auto& size = variant.specialization.runtime_info.hw.cs.workgroup_size;
    commands.Dispatch(MTLSizeMake(extent[0], extent[1], extent[2]),
                      MTLSizeMake(size[0], size[1], size[2]));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
}

void ComputeRenderer::DispatchIndirect(VAddr address, u32 offset, u32 size, bool neo_mode) {
    const auto& program = liverpool.GetCsRegs();
    if (!(program.dispatch_initiator & 1)) {
        return;
    }
    ValidateDispatchMode(program);
    if (size < 3 * sizeof(u32) || address >= (u64{1} << 40) || offset > (u64{1} << 40) - address ||
        size > (u64{1} << 40) - address - offset || (address + offset) % sizeof(u32)) {
        throw std::runtime_error("Invalid Metal guest indirect dispatch arguments");
    }
    auto& variant = Shader();
    if (std::ranges::any_of(variant.info.buffers, [](const auto& buffer) {
            return buffer.buffer_type == Shader::BufferType::SharedMemory;
        })) {
        throw std::runtime_error(
            "Metal indirect shared-memory spill requires GPU-sized scratch allocation");
    }
    buffer_cache.PreparePagedBuffer(address + offset, size);
    auto stage = resources.Prepare(variant, neo_mode);
    buffer_cache.ObtainPagedBuffer(address + offset, size, false);
    std::array stages{stage.get()};
    resources.Acquire(stages);
    auto arguments = GatherIndirectArguments(address + offset);
    const auto start = program.WorkgroupStart();
    if (std::ranges::any_of(start, [](u32 value) { return value != 0; })) {
        arguments = AdjustIndirectArguments(arguments, start);
    }
    const auto snapshot = Bind(variant, *stage);
    const auto& group_size = variant.specialization.runtime_info.hw.cs.workgroup_size;
    commands.DispatchIndirect(arguments, MTLSizeMake(group_size[0], group_size[1], group_size[2]));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
}

BufferRange ComputeRenderer::GatherIndirectArguments(VAddr address) {
    constexpr u64 size = 3 * sizeof(u32);
    const auto ranges = buffer_cache.GetPagedBufferRanges(address, size);
    if (ranges.size() == 1) {
        return ranges.front();
    }
    if (!indirect_arguments) {
        indirect_arguments = instance.CreateBuffer(size, MTLResourceStorageModePrivate);
    }
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    u64 offset{};
    for (const auto& range : ranges) {
        commands.CopyBuffer(range, {indirect_arguments, offset, range.size}, range.size);
        offset += range.size;
    }
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch);
    return {indirect_arguments, 0, size};
}

BufferRange ComputeRenderer::AdjustIndirectArguments(const BufferRange& arguments,
                                                     const std::array<u32, 3>& start) {
    if (!arguments_pipeline) {
        auto library = compiler.CompileLibrary(DispatchArgumentsSource, "shadPS4:DispatchExtent");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_dispatch_extent";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 1;
        arguments_pipeline = compiler.CompileComputePipeline(descriptor);
        adjusted_arguments = instance.CreateBuffer(3 * sizeof(u32), MTLResourceStorageModePrivate);
    }
    auto allocation = staging.Allocate(sizeof(start), 16);
    std::memcpy(allocation.UploadData().data(), start.data(), sizeof(start));
    ResourceBindings bindings{instance, 3, 0, 0};
    bindings.SetBuffer(0, arguments.buffer, arguments.offset);
    bindings.SetBuffer(1, adjusted_arguments);
    bindings.SetBuffer(2, allocation.GetBuffer(), allocation.GetOffset());
    bindings.RetainUploadAllocation(std::move(allocation));
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    commands.BindCompute(arguments_pipeline, bindings);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageDispatch, MTLStageDispatch);
    return {adjusted_arguments, 0, 3 * sizeof(u32)};
}

} // namespace Metal
