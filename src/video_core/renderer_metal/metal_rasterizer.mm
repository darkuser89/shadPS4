// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include "common/recursive_lock.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_rasterizer.h"

namespace Metal {

static constexpr std::string_view FillSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct FillParameters { uint count; uint value; };
kernel void fill_words(device uint* destination [[buffer(0)]],
                       constant FillParameters& parameters [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    if (index < parameters.count) destination[index] = parameters.value;
}
)metal";

static constexpr std::string_view WriteDataSource = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void metal4_write_one_address(device const uint* source [[buffer(0)]],
                                     device volatile uint* destination [[buffer(1)]],
                                     constant uint& count [[buffer(2)]]) {
    for (uint word = 0; word < count; ++word) destination[0] = source[word];
}
)metal";

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool& liverpool_, bool neo_mode_, bool host_markers_,
                       bool guest_markers_)
    : VideoCore::Rasterizer{host_markers_}, instance{instance_}, scheduler{scheduler_},
      liverpool{liverpool_}, memory{*Core::Memory::Instance()},
      page_manager{static_cast<VideoCore::Rasterizer*>(this), true},
      buffer_cache{instance, scheduler, liverpool, page_manager},
      texture_cache{instance, scheduler, buffer_cache},
      shader_cache{instance, buffer_cache, BufferCache::PageShift}, pipelines{instance},
      resources{instance, scheduler, buffer_cache, texture_cache, liverpool},
      graphics{instance,  scheduler,    shader_cache,  pipelines,
               resources, buffer_cache, texture_cache, liverpool},
      compute{instance, scheduler, shader_cache, resources, buffer_cache, liverpool},
      commands{instance, scheduler}, compiler{instance}, staging{instance, scheduler},
      neo_mode{neo_mode_}, host_markers{host_markers_}, guest_markers{guest_markers_} {
    liverpool.BindRasterizer(static_cast<VideoCore::Rasterizer*>(this));
    memory.SetRasterizer(static_cast<VideoCore::Rasterizer*>(this));
}

Rasterizer::~Rasterizer() {
    PrepareShutdown();
    RunOnGpu([this] {
        texture_cache.ResolveAll();
        buffer_cache.ReleaseMemory();
        while (!markers.empty())
            ScopeMarkerEnd(markers.back());
        scheduler.Finish();
        texture_cache.CheckTilingErrors();
        graphics.CheckIndirectErrors();
        liverpool.BindRasterizer(static_cast<VideoCore::Rasterizer*>(nullptr));
        memory.SetRasterizer(static_cast<VideoCore::Rasterizer*>(nullptr));
    });
}

void Rasterizer::PrepareShutdown() {
    liverpool.WaitGpuIdle();
    RunOnGpu([this] {
        eop_interrupts_enabled->store(false, std::memory_order_release);
        scheduler.Finish();
        scheduler.WaitForDeferredOperations();
        liverpool.DrainCommands();
        scheduler.Finish();
    });
}

void Rasterizer::RequireGpuThread() const {
    if (std::this_thread::get_id() != liverpool.GetGpuCommandProcessorThread()) {
        throw std::runtime_error("Metal encoding requires the GPU command processor thread");
    }
}

void Rasterizer::RunOnGpu(const std::function<void()>& function) {
    const auto guarded = [&] {
        const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
        function();
    };
    if (std::this_thread::get_id() == liverpool.GetGpuCommandProcessorThread()) {
        guarded();
        return;
    }
    liverpool.SendCommand<true>(guarded);
}

bool Rasterizer::RequiresMemoryMappingDispatch() const {
    return std::this_thread::get_id() != liverpool.GetGpuCommandProcessorThread();
}

void Rasterizer::DispatchMemoryMapping(const std::function<void()>& function) {
    RunOnGpu(function);
}

void Rasterizer::OnMemoryOperationException() noexcept {
    std::terminate();
}

void Rasterizer::Draw(bool indexed, u32 index_offset) {
    RequireGpuThread();
    graphics.Draw(indexed, index_offset, neo_mode);
}

void Rasterizer::DrawIndirect(bool indexed, VAddr address, u32 offset, u32 stride, u32 max_count,
                              VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RequireGpuThread();
    graphics.DrawIndirect(indexed, address, offset, stride, max_count, count_address,
                          vertex_sgpr_offset, instance_sgpr_offset, neo_mode);
}

void Rasterizer::DispatchDirect() {
    RequireGpuThread();
    compute.DispatchDirect(neo_mode);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RequireGpuThread();
    compute.DispatchIndirect(address, offset, size, neo_mode);
}

bool Rasterizer::MarkersEnabled(bool from_guest) const {
    return from_guest ? guest_markers : host_markers;
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if (!MarkersEnabled(from_guest))
        return;
    RequireGpuThread();
    scheduler.PushDebugGroup(str);
    markers.push_back(from_guest);
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if (!MarkersEnabled(from_guest))
        return;
    RequireGpuThread();
    if (markers.empty() || markers.back() != from_guest) {
        throw std::runtime_error("Unbalanced Metal debug marker group");
    }
    scheduler.PopDebugGroup();
    markers.pop_back();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    ScopeMarkerBegin(str, from_guest);
    ScopeMarkerEnd(from_guest);
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, u32 color, bool from_guest) {
    if (!MarkersEnabled(from_guest))
        return;
    ScopedMarkerInsert(fmt::format("{} [{:08x}]", str, color), from_guest);
}

BufferRange Rasterizer::GdsRange(VAddr address, u32 size) {
    auto range = buffer_cache.GetGdsBuffer();
    if (!size || address > range.size || size > range.size - address) {
        throw std::runtime_error("Invalid Metal guest GDS transfer range");
    }
    range.offset += address;
    range.size = size;
    return range;
}

id<MTLComputePipelineState> Rasterizer::FillPipeline() {
    if (!fill_pipeline.valid()) {
        const auto library = compiler.CompileLibraryAsync(FillSource, "metal4_fill_words");
        MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
        function.library = library.get();
        function.name = @"fill_words";
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        fill_pipeline = compiler.CompileComputePipelineAsync(descriptor);
    }
    return fill_pipeline.get();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    RequireGpuThread();
    if (!num_bytes)
        return;
    if (address % sizeof(u32) || num_bytes % sizeof(u32)) {
        throw std::runtime_error("Metal guest word fill requires four-byte alignment");
    }
    if (!is_gds) {
        buffer_cache.PreparePagedBuffer(address, num_bytes);
        buffer_cache.ObtainPagedBuffer(address, num_bytes, true);
    }
    const bool byte_fill = value == u32(u8(value)) * 0x01010101u;
    const auto pipeline = byte_fill ? nil : FillPipeline();
    const auto destinations = is_gds ? std::vector{GdsRange(address, num_bytes)}
                                     : buffer_cache.GetPagedBufferRanges(address, num_bytes);
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, byte_fill ? MTLStageBlit : MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit,
                             byte_fill ? MTLStageBlit : MTLStageDispatch);
    for (const auto& destination : destinations) {
        if (byte_fill) {
            commands.FillBuffer(destination, u8(value));
            continue;
        }
        const std::array parameters{u32(destination.size / sizeof(u32)), value};
        auto allocation = staging.Allocate(sizeof(parameters), alignof(u32));
        std::memcpy(allocation.UploadData().data(), parameters.data(), sizeof(parameters));
        ResourceBindings bindings{instance, 2, 0, 0};
        bindings.SetBuffer(0, destination.buffer, destination.offset);
        bindings.SetBuffer(1, allocation.GetBuffer(), allocation.GetOffset());
        bindings.RetainUploadAllocation(std::move(allocation));
        commands.BindCompute(pipeline, bindings);
        commands.Dispatch(MTLSizeMake((u64(parameters[0]) + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
    }
    const auto stage = byte_fill ? MTLStageBlit : MTLStageDispatch;
    scheduler.ProducerBarrier(stage, MTLStageAll);
    scheduler.EncoderBarrier(stage, MTLStageDispatch | MTLStageBlit);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    RequireGpuThread();
    if (!num_bytes)
        return;
    if (!src_gds)
        buffer_cache.PreparePagedBuffer(src, num_bytes);
    if (!dst_gds)
        buffer_cache.PreparePagedBuffer(dst, num_bytes);
    if (!src_gds)
        buffer_cache.ObtainPagedBuffer(src, num_bytes, false);
    if (!dst_gds)
        buffer_cache.ObtainPagedBuffer(dst, num_bytes, true);
    const auto source = src_gds ? std::vector{GdsRange(src, num_bytes)}
                                : buffer_cache.GetPagedBufferRanges(src, num_bytes);
    const auto destination = dst_gds ? std::vector{GdsRange(dst, num_bytes)}
                                     : buffer_cache.GetPagedBufferRanges(dst, num_bytes);
    const auto copy_ranges = [&](const auto& inputs, const auto& outputs) {
        size_t input{}, output{};
        u64 input_offset{}, output_offset{};
        while (input < inputs.size()) {
            const auto& from = inputs[input];
            const auto& to = outputs[output];
            const u64 length = std::min(from.size - input_offset, to.size - output_offset);
            commands.CopyBuffer({from.buffer, from.offset + input_offset, length},
                                {to.buffer, to.offset + output_offset, length}, length);
            input_offset += length;
            output_offset += length;
            if (input_offset == from.size) {
                ++input;
                input_offset = 0;
            }
            if (output_offset == to.size) {
                ++output;
                output_offset = 0;
            }
        }
    };
    const bool overlap =
        src_gds == dst_gds && src != dst && src < dst + num_bytes && dst < src + num_bytes;
    std::vector<BufferRange> scratch;
    if (overlap) {
        scratch.reserve(source.size());
        copy_scratch.resize(std::max(copy_scratch.size(), source.size()));
        for (size_t index = 0; index < source.size(); ++index) {
            const u64 size = source[index].size;
            if (!copy_scratch[index] || copy_scratch[index].length < size) {
                copy_scratch[index] = instance.CreateBuffer(size, MTLResourceStorageModePrivate);
            }
            scratch.push_back({copy_scratch[index], 0, size});
        }
    }
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    if (overlap) {
        copy_ranges(source, scratch);
        scheduler.EncoderBarrier(MTLStageBlit, MTLStageBlit);
        copy_ranges(scratch, destination);
    } else {
        copy_ranges(source, destination);
    }
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch | MTLStageBlit);
}

void Rasterizer::SynchronizeDma() {
    Finish();
}

bool Rasterizer::CopyData(const AmdGpu::PM4CmdCopyData& packet) {
    RequireGpuThread();
    using Source = AmdGpu::CopyDataSrc;
    using Destination = AmdGpu::CopyDataDst;
    const auto source = packet.src_sel.Value();
    const auto destination = packet.dst_sel.Value();
    if (packet.engine_sel != AmdGpu::CopyDataEngine::Me ||
        (source != Source::Memory && source != Source::TCL2 && source != Source::Gds &&
         source != Source::Immediate) ||
        (destination != Destination::MemorySync && destination != Destination::MemoryAsync &&
         destination != Destination::TCL2 && destination != Destination::Gds)) {
        throw std::runtime_error("Metal COPY_DATA source, destination or engine requires lowering");
    }
    const u32 size = packet.count_sel.Value() ? sizeof(u64) : sizeof(u32);
    const VAddr src = packet.SrcAddress<VAddr>();
    const VAddr dst = packet.DstAddress<VAddr>();
    const bool src_gds = source == Source::Gds;
    const bool dst_gds = destination == Destination::Gds;
    if (dst % size || (source != Source::Immediate && src % size)) {
        throw std::runtime_error("Metal COPY_DATA requires aligned guest addresses");
    }
    if (source == Source::Immediate) {
        if (!dst_gds) {
            buffer_cache.PreparePagedBuffer(dst, size);
            buffer_cache.ObtainPagedBuffer(dst, size, true);
        }
        const auto ranges = dst_gds ? std::vector{GdsRange(dst, size)}
                                    : buffer_cache.GetPagedBufferRanges(dst, size);
        auto allocation = staging.Allocate(size, alignof(u64));
        std::memcpy(allocation.UploadData().data(), &src, size);
        (void)scheduler.BeginCompute();
        scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
        scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
        u64 offset{};
        for (const auto& range : ranges) {
            commands.CopyBuffer(
                {allocation.GetBuffer(), allocation.GetOffset() + offset, range.size}, range,
                range.size);
            offset += range.size;
        }
        scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
        scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch | MTLStageBlit);
    } else {
        CopyBuffer(dst, src, size, dst_gds, src_gds);
    }
    if (packet.wr_confirm.Value() || destination == Destination::MemorySync) {
        if (dst_gds) {
            Finish();
        } else {
            buffer_cache.ReadMemory(dst, size);
            texture_cache.CheckTilingErrors();
            graphics.CheckIndirectErrors();
        }
    }
    return true;
}

id<MTLComputePipelineState> Rasterizer::WriteDataPipeline() {
    if (!write_data_pipeline.valid()) {
        auto library = compiler.CompileLibraryAsync(WriteDataSource, "shadPS4:WriteData");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library.get();
        function.name = @"metal4_write_one_address";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 1;
        write_data_pipeline = compiler.CompileComputePipelineAsync(descriptor);
    }
    return write_data_pipeline.get();
}

bool Rasterizer::WriteData(const AmdGpu::PM4CmdWriteData& packet) {
    RequireGpuThread();
    const u32 destination = packet.dst_sel.Value();
    constexpr u32 supported_control = (0xfu << 8) | (1u << 16) | (1u << 20);
    if (packet.header.NumWords() < 4 || (packet.raw & ~supported_control) ||
        packet.header.predicate != AmdGpu::PM4Predicate::PredDisable ||
        (destination != 1 && destination != 2 && destination != 3 && destination != 5)) {
        throw std::runtime_error(
            "Metal WRITE_DATA packet, engine or destination requires lowering");
    }
    const u32 data_size = packet.Size();
    const bool one_address = packet.wr_one_addr.Value();
    const u32 size = one_address ? sizeof(u32) : data_size;
    const VAddr address = std::bit_cast<VAddr>(packet.Address<void*>());
    const bool is_gds = destination == 3;
    if (address % sizeof(u32)) {
        throw std::runtime_error("Metal WRITE_DATA requires a four-byte aligned destination");
    }
    const auto pipeline = one_address ? WriteDataPipeline() : nil;
    if (!is_gds) {
        buffer_cache.PreparePagedBuffer(address, size);
        buffer_cache.ObtainPagedBuffer(address, size, true);
    }
    const auto ranges = is_gds ? std::vector{GdsRange(address, size)}
                               : buffer_cache.GetPagedBufferRanges(address, size);
    auto allocation = staging.Allocate(u64{data_size} + sizeof(u32));
    const u32 count = data_size / sizeof(u32);
    std::memcpy(allocation.UploadData().data(), &count, sizeof(count));
    std::memcpy(allocation.UploadData().data() + sizeof(count), packet.data, data_size);
    const auto stage = one_address ? MTLStageDispatch : MTLStageBlit;
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, stage);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, stage);
    if (one_address) {
        ResourceBindings bindings{instance, 3, 0, 0};
        bindings.SetBuffer(0, allocation.GetBuffer(), allocation.GetOffset() + sizeof(count));
        bindings.SetBuffer(1, ranges.front().buffer, ranges.front().offset);
        bindings.SetBuffer(2, allocation.GetBuffer(), allocation.GetOffset());
        bindings.RetainUploadAllocation(std::move(allocation));
        commands.BindCompute(pipeline, bindings);
        commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    } else {
        u64 offset = sizeof(count);
        for (const auto& range : ranges) {
            commands.CopyBuffer(
                {allocation.GetBuffer(), allocation.GetOffset() + offset, range.size}, range,
                range.size);
            offset += range.size;
        }
    }
    scheduler.ProducerBarrier(stage, MTLStageAll);
    scheduler.EncoderBarrier(stage, MTLStageDispatch | MTLStageBlit);
    if (packet.wr_confirm.Value() || destination == 1) {
        if (is_gds) {
            Finish();
        } else {
            buffer_cache.ReadMemory(address, size);
            texture_cache.CheckTilingErrors();
            graphics.CheckIndirectErrors();
        }
    }
    return true;
}

bool Rasterizer::PixelStatistics(const AmdGpu::PM4CmdEventWrite& packet, u32 num_counter_pairs) {
    RequireGpuThread();
    using Event = AmdGpu::EventType;
    const auto event = packet.event_type.Value();
    if (event != Event::PixelPipeStatControl && event != Event::PixelPipeStatDump &&
        event != Event::PixelPipeStatReset && event != Event::ZpassDone) {
        return false;
    }
    if ((num_counter_pairs != 8 && num_counter_pairs != 16) ||
        packet.header.predicate != AmdGpu::PM4Predicate::PredDisable) {
        throw std::runtime_error("Metal pixel counter packet requires native query lowering");
    }
    if (event == Event::PixelPipeStatReset) {
        if (packet.header.NumWords() != 1 || packet.event_index != AmdGpu::EventIndex::Other) {
            throw std::runtime_error("Invalid Metal pixel counter reset packet");
        }
        graphics.ResetPixelCounter(pixel_counter_index);
        return true;
    }
    if (packet.header.NumWords() != 3 || packet.event_index != AmdGpu::EventIndex::ZpassDone) {
        throw std::runtime_error("Invalid Metal pixel counter packet size or index");
    }
    if (event == Event::PixelPipeStatControl) {
        const u32 control = packet.address[0];
        const u32 counter = (control >> 3) & 0x3f;
        const u32 stride = 4u << ((control >> 9) & 3);
        const u64 instances = (u64(packet.address[1]) << 21) | (control >> 11);
        if ((control & 7) || counter >= 4 || stride < sizeof(u64) ||
            instances != (u64{1} << num_counter_pairs) - 1) {
            throw std::runtime_error(
                "Metal pixel counter selection requires native query lowering");
        }
        pixel_counter_index = counter;
        pixel_counter_stride = stride;
        return true;
    }
    const VAddr address = packet.Address<VAddr>();
    const u32 stride = event == Event::ZpassDone ? 2 * sizeof(u64) : pixel_counter_stride;
    const u64 size = u64{num_counter_pairs - 1} * stride + sizeof(u64);
    if (address % sizeof(u64) || address >= (u64{1} << 40) || size > (u64{1} << 40) - address) {
        throw std::runtime_error("Invalid Metal pixel counter destination range");
    }
    buffer_cache.PreparePagedBuffer(address, size);
    buffer_cache.ObtainPagedBuffer(address, size, true);
    const auto destinations = buffer_cache.GetPagedBufferRanges(address, size);
    const u32 counter = event == Event::ZpassDone ? 0 : pixel_counter_index;
    if (destinations.size() == 1) {
        graphics.DumpPixelCounter(counter, destinations.front(), num_counter_pairs, stride);
        return true;
    }
    if (!pixel_counter_output || pixel_counter_output.length < size) {
        pixel_counter_output = instance.CreateBuffer(size, MTLResourceStorageModePrivate);
    }
    graphics.DumpPixelCounter(counter, {pixel_counter_output, 0, size}, num_counter_pairs, stride);
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    for (u32 pair = 0; pair < num_counter_pairs; ++pair) {
        const u64 offset = u64{pair} * stride;
        const auto ranges = buffer_cache.GetPagedBufferRanges(address + offset, sizeof(u64));
        u64 copied{};
        for (const auto& range : ranges) {
            commands.CopyBuffer({pixel_counter_output, offset + copied, range.size}, range,
                                range.size);
            copied += range.size;
        }
    }
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch | MTLStageBlit);
    return true;
}

bool Rasterizer::EventWriteEop(const AmdGpu::PM4CmdEventWriteEop& packet) {
    RequireGpuThread();
    if (packet.header.NumWords() != 5 ||
        packet.header.predicate != AmdGpu::PM4Predicate::PredDisable) {
        throw std::runtime_error("Invalid Metal EOP packet size or predicate");
    }
    const bool gpu_fence = (packet.data_sel == AmdGpu::DataSelect::Data32Low ||
                            packet.data_sel == AmdGpu::DataSelect::Data64) &&
                           std::bit_cast<VAddr>(packet.Address<void>()) % sizeof(u32) == 0;
    if (packet.data_sel == AmdGpu::DataSelect::None &&
        (packet.int_sel == AmdGpu::InterruptSelect::IrqOnly ||
         packet.int_sel == AmdGpu::InterruptSelect::IrqWhenWriteConfirm)) {
        scheduler.Flush();
    } else if (!gpu_fence) {
        OnFence();
    }
    packet.SignalFence(
        [this, gpu_fence](void* address, u64 data, u32 size) {
            const auto guest_address = std::bit_cast<VAddr>(address);
            if (!gpu_fence) {
                WriteMemory(guest_address, {reinterpret_cast<const u8*>(&data), size});
                return;
            }
            WriteDataFence(guest_address, data, size);
        },
        [this, gpu_fence] { QueueFenceInterrupt(u32(Platform::InterruptId::GfxEop), gpu_fence); });
    return true;
}

bool Rasterizer::ReleaseMem(const AmdGpu::PM4CmdReleaseMem& packet, u32 pipe_id) {
    RequireGpuThread();
    if ((packet.data_sel != AmdGpu::DataSelect::Data32Low &&
         packet.data_sel != AmdGpu::DataSelect::Data64 &&
         packet.data_sel != AmdGpu::DataSelect::GdsMemStore) ||
        packet.dst_sel != 0 || packet.Address<VAddr>() % sizeof(u32)) {
        return false;
    }
    if (packet.header.NumWords() != 6 ||
        packet.header.predicate != AmdGpu::PM4Predicate::PredDisable ||
        pipe_id >= AmdGpu::Liverpool::NumComputePipes) {
        throw std::runtime_error("Invalid Metal RELEASE_MEM packet or pipe");
    }
    packet.SignalFence(
        [this](void* address, u64 data, u32 size) {
            WriteDataFence(std::bit_cast<VAddr>(address), data, size);
        },
        [this, pipe_id] { QueueFenceInterrupt(pipe_id, true); },
        [this](VAddr address, u16 offset, u16 words) {
            const u32 size = u32{words} * sizeof(u32);
            if (offset % sizeof(u32)) {
                throw std::runtime_error("Invalid Metal RELEASE_MEM GDS offset");
            }
            (void)GdsRange(offset, size);
            CopyBuffer(address, offset, size, false, true);
        });
    return true;
}

void Rasterizer::WriteDataFence(VAddr address, u64 data, u32 size) {
    constexpr u32 MemoryUsingL2Control = 5u << 8;
    const u32 count = size / sizeof(u32);
    alignas(AmdGpu::PM4CmdWriteData) const std::array<u32, 6> write_packet{
        AmdGpu::PM4Type3Header{AmdGpu::PM4ItOpcode::WriteData, count + 2}.raw,
        MemoryUsingL2Control,
        u32(address),
        u32(address >> 32),
        u32(data),
        u32(data >> 32),
    };
    WriteData(*reinterpret_cast<const AmdGpu::PM4CmdWriteData*>(write_packet.data()));
}

void Rasterizer::QueueFenceInterrupt(u32 interrupt_id, bool gpu_fence) {
    if (gpu_fence) {
        scheduler.Flush();
    }
    auto* processor = &liverpool;
    auto enabled = eop_interrupts_enabled;
    scheduler.DeferOperation(scheduler.CurrentTick() - 1,
                             [this, processor, enabled, gpu_fence, interrupt_id] {
                                 processor->SendCommand([this, enabled, gpu_fence, interrupt_id] {
                                     if (enabled->load(std::memory_order_acquire)) {
                                         if (gpu_fence) {
                                             OnSubmit();
                                         }
                                         Platform::IrqC::Instance()->Signal(
                                             static_cast<Platform::InterruptId>(interrupt_id));
                                     }
                                 });
                             });
}

void Rasterizer::PrepareWaitRegMem(const AmdGpu::PM4CmdWaitRegMem& packet) {
    RequireGpuThread();
    if (packet.mem_space != AmdGpu::PM4CmdWaitRegMem::MemSpace::Memory ||
        packet.function == AmdGpu::PM4CmdWaitRegMem::Function::Always) {
        return;
    }
    PrepareMemoryAccess(packet.Address<VAddr>(), sizeof(u32));
}

void Rasterizer::PrepareMemoryAccess(VAddr address, u64 size, bool is_write) {
    RequireGpuThread();
    if (!IsMapped(address, size)) {
        return;
    }
    buffer_cache.ReadMemory(address, size, is_write);
    texture_cache.CheckTilingErrors();
    graphics.CheckIndirectErrors();
}

u32 Rasterizer::ReadDataFromGds(u32 offset) {
    RequireGpuThread();
    return buffer_cache.ReadDataFromGds(offset);
}

bool Rasterizer::StoreGds(VAddr address, u32 offset, u32 size) {
    RequireGpuThread();
    const auto gds = buffer_cache.GetGdsBuffer();
    if (!size || address % sizeof(u32) || offset % sizeof(u32) || size % sizeof(u32) ||
        offset > gds.size || size > gds.size - offset) {
        throw std::runtime_error("Invalid Metal guest GDS store range");
    }
    CopyBuffer(address, offset, size, false, true);
    buffer_cache.ReadMemory(address, size);
    texture_cache.CheckTilingErrors();
    graphics.CheckIndirectErrors();
    return true;
}

bool Rasterizer::WriteMemory(VAddr address, std::span<const u8> data) {
    RequireGpuThread();
    buffer_cache.WriteMemory(address, data);
    return true;
}

bool Rasterizer::WithMemoryWriteWatch(VAddr address, u64 size, const std::function<void()>& write) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    RequireGpuThread();
    if (!IsMapped(address, size)) {
        return false;
    }
    buffer_cache.WithMemoryWriteWatch(address, size, write);
    return true;
}

bool Rasterizer::InvalidateMemory(VAddr address, u64 size, bool assume_locks) {
    if (!IsMapped(address, size))
        return false;
    buffer_cache.InvalidateMemory(address, size, assume_locks);
    return true;
}

bool Rasterizer::ReadMemory(VAddr address, u64 size, bool assume_locks) {
    if (!IsMapped(address, size))
        return false;
    buffer_cache.ReadMemory(address, size, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr address, u64 size) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    if (!size || address >= (u64{1} << 40) || size > (u64{1} << 40) - address)
        return false;
    Common::RecursiveSharedLock lock{mapped_mutex};
    return mapped_ranges.Contains(address, size);
}

void Rasterizer::MapMemory(VAddr address, u64 size) {
    if (!size || address >= (u64{1} << 40) || size > (u64{1} << 40) - address) {
        throw std::runtime_error("Invalid Metal guest mapping range");
    }
    RunOnGpu([this, address, size] {
        GuestRangeMap<bool> prepared;
        std::vector<VAddr> replaced;
        {
            std::shared_lock lock{mapped_mutex};
            mapped_ranges.StageRange(address, size, prepared, replaced);
        }
        prepared.Add(address, size, true);
        buffer_cache.MapMemory(address, size);
        std::unique_lock lock{mapped_mutex};
        mapped_ranges.CommitPrepared(prepared, replaced);
    });
}

void Rasterizer::RegisterMemory(VAddr address, u64 size) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    if (!size)
        return;
    constexpr u64 Limit = u64{1} << 40;
    if (address < Limit) {
        page_manager.OnGpuMap(address, std::min(size, Limit - address));
    }
}

void Rasterizer::UnmapMemory(VAddr address, u64 size) {
    if (!size)
        return;
    RunOnGpu([this, address, size] {
        GuestRangeMap<bool> prepared;
        std::vector<VAddr> replaced;
        {
            std::shared_lock lock{mapped_mutex};
            mapped_ranges.StageRange(address, size, prepared, replaced);
        }
        prepared.Subtract(address, size);
        buffer_cache.UnmapMemory(address, size);
        texture_cache.UnmapMemory(address, size);
        page_manager.OnGpuUnmap(address, size);
        std::unique_lock lock{mapped_mutex};
        mapped_ranges.CommitPrepared(prepared, replaced);
    });
}

bool Rasterizer::ProtectGuestMemory(VAddr address, u64 size, u32 prot) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    return page_manager.ProtectGuestMemory(address, size, prot);
}

bool Rasterizer::HandleGuestMemoryFault(VAddr address, u64 size, bool is_write, bool is_execute,
                                        bool assume_locks) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    auto access = is_execute ? Core::MemoryPermission::Execute : Core::MemoryPermission::Read;
    if (is_write) {
        access = is_execute ? Core::MemoryPermission::Execute | Core::MemoryPermission::Write
                            : Core::MemoryPermission::Write;
    }
    if (!IsMapped(address, size) || !page_manager.IsGuestAccessAllowed(address, size, access)) {
        return false;
    }
    return is_write ? InvalidateMemory(address, size, assume_locks)
                    : ReadMemory(address, size, assume_locks);
}

u64 Rasterizer::Flush() {
    RequireGpuThread();
    const u64 tick = scheduler.CurrentTick();
    scheduler.Flush();
    return tick;
}

void Rasterizer::Finish() {
    RequireGpuThread();
    scheduler.Finish();
    texture_cache.CheckTilingErrors();
    graphics.CheckIndirectErrors();
}

void Rasterizer::OnSubmit() {
    RequireGpuThread();
    scheduler.ReclaimCompletedResources();
    texture_cache.CheckTilingErrors();
    graphics.CheckIndirectErrors();
}

void Rasterizer::OnFence() {
    RequireGpuThread();
    scheduler.Finish();
    texture_cache.CheckTilingErrors();
    graphics.CheckIndirectErrors();
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool.GetGpuCommandProcessorThread();
}

} // namespace Metal
