// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <functional>
#include "common/shared_first_mutex.h"
#include "video_core/page_manager.h"
#include "video_core/rasterizer.h"
#include "video_core/renderer_metal/metal_compute.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_guest_ranges.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

class Rasterizer final : public VideoCore::Rasterizer {
public:
    Rasterizer(const Instance& instance, Scheduler& scheduler, AmdGpu::Liverpool& liverpool,
               bool neo_mode, bool host_markers, bool guest_markers);
    ~Rasterizer() override;

    void Draw(bool indexed, u32 index_offset = 0) override;
    void DrawIndirect(bool indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset,
                      u16 instance_sgpr_offset) override;
    void DispatchDirect() override;
    void DispatchIndirect(VAddr address, u32 offset, u32 size) override;

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false) override;
    void ScopeMarkerEnd(bool from_guest = false) override;
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false) override;
    void ScopedMarkerInsertColor(const std::string_view& str, u32 color,
                                 bool from_guest = false) override;

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) override;
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) override;
    void SynchronizeDma() override;
    bool CopyData(const AmdGpu::PM4CmdCopyData& packet) override;
    bool WriteData(const AmdGpu::PM4CmdWriteData& packet) override;
    bool PixelStatistics(const AmdGpu::PM4CmdEventWrite& packet, u32 num_counter_pairs) override;
    bool EventWriteEop(const AmdGpu::PM4CmdEventWriteEop& packet) override;
    bool ReleaseMem(const AmdGpu::PM4CmdReleaseMem& packet, u32 pipe_id) override;
    void PrepareWaitRegMem(const AmdGpu::PM4CmdWaitRegMem& packet) override;
    void PrepareMemoryAccess(VAddr address, u64 size, bool is_write = false) override;
    u32 ReadDataFromGds(u32 offset) override;
    bool StoreGds(VAddr address, u32 offset, u32 size) override;
    bool WriteMemory(VAddr address, std::span<const u8> data) override;
    bool WithMemoryWriteWatch(VAddr address, u64 size, const std::function<void()>& write) override;
    bool InvalidateMemory(VAddr address, u64 size, bool assume_locks = false) override;
    bool ReadMemory(VAddr address, u64 size, bool assume_locks = false) override;
    bool IsMapped(VAddr address, u64 size) override;
    bool RequiresMemoryMappingDispatch() const override;
    void DispatchMemoryMapping(const std::function<void()>& function) override;
    void OnMemoryOperationException() noexcept override;
    void MapMemory(VAddr address, u64 size) override;
    void RegisterMemory(VAddr address, u64 size) override;
    void UnmapMemory(VAddr address, u64 size) override;
    bool ProtectGuestMemory(VAddr address, u64 size, u32 prot) override;
    bool HandleGuestMemoryFault(VAddr address, u64 size, bool is_write, bool is_execute,
                                bool assume_locks) override;

    u64 Flush() override;
    void Finish() override;
    void OnSubmit() override;
    void OnFence() override;
    std::thread::id GetGpuCommandProcessorThread() override;

    void RunOnGpu(const std::function<void()>& function);
    void PrepareShutdown();
    [[nodiscard]] TextureCache& GetTextureCache() {
        return texture_cache;
    }

private:
    void RequireGpuThread() const;
    void WriteDataFence(VAddr address, u64 data, u32 size);
    void QueueFenceInterrupt(u32 interrupt_id, bool gpu_fence);
    [[nodiscard]] bool MarkersEnabled(bool from_guest) const;
    [[nodiscard]] BufferRange GdsRange(VAddr address, u32 size);
    [[nodiscard]] id<MTLComputePipelineState> FillPipeline();
    [[nodiscard]] id<MTLComputePipelineState> WriteDataPipeline();

    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool& liverpool;
    Core::MemoryManager& memory;
    Common::SharedFirstMutex mapped_mutex;
    GuestRangeMap<bool> mapped_ranges;
    VideoCore::PageManager page_manager;
    BufferCache buffer_cache;
    TextureCache texture_cache;
    ShaderCache shader_cache;
    GraphicsPipelineCache pipelines;
    GuestResources resources;
    GraphicsRenderer graphics;
    ComputeRenderer compute;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    std::shared_future<id<MTLComputePipelineState>> fill_pipeline;
    std::shared_future<id<MTLComputePipelineState>> write_data_pipeline;
    std::vector<id<MTLBuffer>> copy_scratch;
    std::vector<bool> markers;
    std::shared_ptr<std::atomic<bool>> eop_interrupts_enabled{
        std::make_shared<std::atomic<bool>>(true)};
    id<MTLBuffer> pixel_counter_output;
    u32 pixel_counter_index{};
    u32 pixel_counter_stride{2 * sizeof(u64)};
    const bool neo_mode;
    const bool host_markers;
    const bool guest_markers;
};

} // namespace Metal
