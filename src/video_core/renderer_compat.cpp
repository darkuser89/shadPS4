// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <map>
#include <mutex>
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "video_core/presenter.h"
#include "video_core/rasterizer.h"
#include "video_core/renderer_compat.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace VideoCore {

void UpdateGameResolution(u32 width, u32 height) {
    DebugState.game_resolution.store({width, height}, std::memory_order_relaxed);
}

void UpdateFsrState(bool enabled) {
    DebugState.is_using_fsr.store(enabled, std::memory_order_relaxed);
}
void UpdateOutputResolution(u32 width, u32 height) {
    DebugState.output_resolution.store({width, height}, std::memory_order_relaxed);
}
void CompleteGameFlip() {
    DebugState.IncFlipFrameNum();
}

namespace {

class RasterizerAdapter final : public Rasterizer {
public:
    explicit RasterizerAdapter(Vulkan::Rasterizer& rasterizer_)
        : Rasterizer{EmulatorSettings.IsVkHostMarkersEnabled()}, rasterizer{rasterizer_} {}

    Vulkan::Rasterizer& GetRasterizer() const {
        return rasterizer;
    }

    void Draw(bool indexed, u32 offset) override {
        rasterizer.Draw(indexed, offset);
    }

    void DrawIndirect(bool indexed, VAddr address, u32 offset, u32 size, u32 count,
                      VAddr count_address, u16 vertex_offset, u16 instance_offset) override {
        rasterizer.DrawIndirect(indexed, address, offset, size, count, count_address, vertex_offset,
                                instance_offset);
    }

    void DispatchDirect() override {
        rasterizer.DispatchDirect();
    }

    void DispatchIndirect(VAddr address, u32 offset, u32 size) override {
        rasterizer.DispatchIndirect(address, offset, size);
    }

    void ScopeMarkerBegin(const std::string_view& str, bool guest) override {
        rasterizer.ScopeMarkerBegin(str, guest);
    }

    void ScopeMarkerEnd(bool guest) override {
        rasterizer.ScopeMarkerEnd(guest);
    }

    void ScopedMarkerInsert(const std::string_view& str, bool guest) override {
        rasterizer.ScopedMarkerInsert(str, guest);
    }

    void ScopedMarkerInsertColor(const std::string_view& str, u32 color, bool guest) override {
        rasterizer.ScopedMarkerInsertColor(str, color, guest);
    }

    void FillBuffer(VAddr address, u32 size, u32 value, bool gds) override {
        rasterizer.FillBuffer(address, size, value, gds);
    }

    void CopyBuffer(VAddr dst, VAddr src, u32 size, bool dst_gds, bool src_gds) override {
        rasterizer.CopyBuffer(dst, src, size, dst_gds, src_gds);
    }

    u32 ReadDataFromGds(u32 offset) override {
        return rasterizer.ReadDataFromGds(offset);
    }

    bool InvalidateMemory(VAddr address, u64 size, bool assume_locks) override {
        return rasterizer.InvalidateMemory(address, size, assume_locks);
    }

    bool ReadMemory(VAddr address, u64 size, bool assume_locks) override {
        return rasterizer.ReadMemory(address, size, assume_locks);
    }

    bool IsMapped(VAddr address, u64 size) override {
        return rasterizer.IsMapped(address, size);
    }

    void MapMemory(VAddr address, u64 size) override {
        rasterizer.MapMemory(address, size);
    }

    void RegisterMemory(VAddr address, u64 size) override {
        rasterizer.RegisterMemory(address, size);
    }

    void UnmapMemory(VAddr address, u64 size) override {
        rasterizer.UnmapMemory(address, size);
    }

    u64 Flush() override {
        return rasterizer.Flush();
    }

    void Finish() override {
        rasterizer.Finish();
    }

    void OnSubmit() override {
        rasterizer.OnSubmit();
    }

    void OnFence() override {
        rasterizer.OnFence();
    }

    std::thread::id GetGpuCommandProcessorThread() override {
        return rasterizer.GetGpuCommandProcessorThread();
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() override {
        return rasterizer.GetGpuCommandProcessorThreadId();
    }
#endif

private:
    Vulkan::Rasterizer& rasterizer;
};

struct FrameAdapter final : Frame {
    explicit FrameAdapter(Vulkan::Frame* frame_) : frame{frame_} {}
    Vulkan::Frame* frame;
};

class PresenterAdapter final : public Presenter {
public:
    explicit PresenterAdapter(std::unique_ptr<Vulkan::Presenter> presenter_)
        : Presenter{presenter_->GetWindow()}, presenter{std::move(presenter_)},
          rasterizer{presenter->GetRasterizer()} {
        SetGamma(presenter->GetPPSettingsRef().gamma);
        SetHdrEnabled(presenter->GetPPSettingsRef().hdr != 0);
        const auto& fsr = presenter->GetFsrSettingsRef();
        SetFsrEnabled(fsr.enable);
        SetRcasEnabled(fsr.use_rcas);
        SetRcasAttenuation(fsr.rcas_attenuation);
    }

    Rasterizer& GetRasterizer() const override {
        return rasterizer;
    }

    bool IsHDRSupported() const override {
        return presenter->IsHDRSupported();
    }

    void SetHDR(bool enable) override {
        presenter->SetHDR(enable);
        SetHdrEnabled(presenter->GetPPSettingsRef().hdr != 0);
    }

    void RegisterVideoOutSurface(const Libraries::VideoOut::BufferAttributeGroup& group,
                                 VAddr address) override {
        presenter->RegisterVideoOutSurface(group, address);
    }

    Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& group,
                        VAddr address) override {
        SyncSettings();
        return Wrap(presenter->PrepareFrame(group, address));
    }

    Frame* PrepareBlankFrame(bool present_thread) override {
        SyncSettings();
        return Wrap(presenter->PrepareBlankFrame(present_thread));
    }

    Frame* PrepareLastFrame() override {
        SyncSettings();
        return Wrap(presenter->PrepareLastFrame());
    }

    void Present(Frame* frame, bool reuse, bool game) override {
        auto* wrapped = dynamic_cast<FrameAdapter*>(frame);
        ASSERT(wrapped);
        presenter->Present(wrapped->frame, reuse, game);
    }

private:
    void SyncSettings() {
        const auto settings = GetDisplaySettings();
        presenter->GetPPSettingsRef().gamma = settings.post_processing.gamma;
        auto& fsr = presenter->GetFsrSettingsRef();
        fsr = {settings.fsr.enable, settings.fsr.use_rcas, settings.fsr.rcas_attenuation};
    }

    Frame* Wrap(Vulkan::Frame* frame) {
        if (!frame) {
            return nullptr;
        }
        std::scoped_lock lock{frame_mutex};
        auto& wrapped = frames[frame];
        if (!wrapped) {
            wrapped = std::make_unique<FrameAdapter>(frame);
        }
        return wrapped.get();
    }

    std::unique_ptr<Vulkan::Presenter> presenter;
    mutable RasterizerAdapter rasterizer;
    std::mutex frame_mutex;
    std::map<Vulkan::Frame*, std::unique_ptr<FrameAdapter>> frames;
};

} // namespace

std::unique_ptr<Rasterizer> AdaptRasterizer(Vulkan::Rasterizer* rasterizer) {
    ASSERT(rasterizer);
    return std::make_unique<RasterizerAdapter>(*rasterizer);
}

std::unique_ptr<Presenter> AdaptPresenter(std::unique_ptr<Vulkan::Presenter> presenter) {
    ASSERT(presenter);
    return std::make_unique<PresenterAdapter>(std::move(presenter));
}

Vulkan::Rasterizer* GetVulkanRasterizer(Rasterizer& rasterizer) {
    const auto* adapter = dynamic_cast<RasterizerAdapter*>(&rasterizer);
    return adapter ? &adapter->GetRasterizer() : nullptr;
}

} // namespace VideoCore
