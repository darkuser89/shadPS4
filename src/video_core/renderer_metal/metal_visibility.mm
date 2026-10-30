// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <stdexcept>
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_visibility.h"

namespace Metal {
namespace {
constexpr std::string_view CounterSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters { uint counter; uint pairs; uint stride_qwords; uint reset; };
static_assert(sizeof(Parameters) == 16);
kernel void metal4_pixel_counter(device const ulong* counters [[buffer(0)]],
                                 device ulong* baselines [[buffer(1)]],
                                 device ulong* output [[buffer(2)]],
                                 constant Parameters& parameters [[buffer(3)]]) {
    ulong total = 0;
    for (uint mask = 1; mask < 16; ++mask) {
        if (mask & (1u << parameters.counter)) total += counters[mask];
    }
    if (parameters.reset) {
        baselines[parameters.counter] = total;
        return;
    }
    const ulong value = (total - baselines[parameters.counter]) & 0x7ffffffffffffffful;
    for (uint rb = 0; rb < parameters.pairs; ++rb) {
        output[rb * parameters.stride_qwords] =
            (rb == 0 ? value : 0ul) | 0x8000000000000000ul;
    }
}
)metal";
} // namespace

VisibilityQueries::VisibilityQueries(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler},
      buffer{instance.CreateBuffer(16 * sizeof(u64), MTLResourceStorageModeShared)},
      baselines{instance.CreateBuffer(4 * sizeof(u64), MTLResourceStorageModeShared)} {
    std::memset(buffer.contents, 0, buffer.length);
    std::memset(baselines.contents, 0, baselines.length);
}

void VisibilityQueries::Prepare(RenderState& render, GraphicsState& graphics, u32 control) {
    if (control & 0x00fff000) {
        throw std::runtime_error(
            "Metal failed-sample pixel counters require native query lowering");
    }
    if (control & 1)
        return;
    const u32 enabled = (control >> 8) & 0xf;
    const u32 even = (control >> 24) & enabled;
    const u32 odd = (control >> 28) & enabled;
    if (even != odd || (control & 0x70) != 0) {
        throw std::runtime_error("Metal pixel counter mode requires native query lowering");
    }
    if (!even)
        return;
    render.visibility_buffer = buffer;
    render.visibility_result_type = MTLVisibilityResultTypeAccumulate;
    graphics.visibility_mode = MTLVisibilityResultModeCounting;
    graphics.visibility_offset = even * sizeof(u64);
}

id<MTLComputePipelineState> VisibilityQueries::Pipeline() {
    if (!pipeline) {
        auto library = compiler.CompileLibrary(CounterSource, "shadPS4:PixelCounter");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_pixel_counter";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 1;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return pipeline;
}

void VisibilityQueries::Dump(u32 counter, const BufferRange& destination, u32 pairs, u32 stride) {
    if ((pairs != 8 && pairs != 16) || (stride != 8 && stride != 16 && stride != 32) ||
        !destination.buffer || destination.buffer.device != instance.GetDevice() ||
        destination.offset % sizeof(u64) || destination.offset > destination.buffer.length ||
        destination.size < u64{pairs - 1} * stride + sizeof(u64) ||
        destination.size > destination.buffer.length - destination.offset) {
        throw std::runtime_error("Invalid Metal pixel counter output range");
    }
    Update(counter, destination, pairs, stride, false);
}

void VisibilityQueries::Reset(u32 counter) {
    Update(counter, {baselines, 0, baselines.length}, 0, 0, true);
}

void VisibilityQueries::Update(u32 counter, const BufferRange& destination, u32 pairs, u32 stride,
                               bool reset) {
    if (counter >= 4) {
        throw std::runtime_error("Invalid Metal pixel counter index");
    }
    const auto compute_pipeline = Pipeline();
    struct Parameters {
        u32 counter;
        u32 pairs;
        u32 stride_qwords;
        u32 reset;
    };
    static_assert(sizeof(Parameters) == 16);
    const Parameters parameters{counter, pairs, stride / u32(sizeof(u64)), u32(reset)};
    auto allocation = staging.Allocate(sizeof(parameters));
    std::memcpy(allocation.UploadData().data(), &parameters, sizeof(parameters));
    ResourceBindings bindings{instance, 4, 0, 0};
    bindings.SetBuffer(0, buffer);
    bindings.SetBuffer(1, baselines);
    bindings.SetBuffer(2, destination.buffer, destination.offset);
    bindings.SetBuffer(3, allocation.GetBuffer(), allocation.GetOffset());
    bindings.RetainUploadAllocation(std::move(allocation));
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    commands.BindCompute(compute_pipeline, bindings);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
}

} // namespace Metal
