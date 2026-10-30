// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>
#include "common/types.h"
#include "core/libraries/videoout/buffer.h"

namespace Frontend {
class WindowSDL;
}

namespace VideoCore {

class Rasterizer;
void UpdateGameResolution(u32 width, u32 height);
void UpdateOutputResolution(u32 width, u32 height);
void UpdateFsrState(bool enabled);
void CompleteGameFlip();

struct Frame {
    virtual ~Frame() = default;
};

struct PostProcessingSettings {
    float gamma = 1.0f;
    u32 hdr = 0;
    u32 srgb_input = 0;
};

struct FsrSettings {
    bool enable{true};
    bool use_rcas{true};
    float rcas_attenuation{0.25f};
};

struct DisplaySettings {
    PostProcessingSettings post_processing;
    FsrSettings fsr;
    bool metal_fx{};
};

class Presenter {
public:
    virtual ~Presenter() = default;

    [[nodiscard]] DisplaySettings GetDisplaySettings() const {
        std::scoped_lock lock{settings_mutex};
        return display_settings;
    }

    void SetGamma(float gamma) {
        std::scoped_lock lock{settings_mutex};
        display_settings.post_processing.gamma = gamma;
    }

    void SetFsrEnabled(bool enabled) {
        std::scoped_lock lock{settings_mutex};
        display_settings.fsr.enable = enabled;
        if (enabled) {
            display_settings.metal_fx = false;
        }
    }

    void SetMetalFxEnabled(bool enabled) {
        if (enabled && !SupportsMetalFx()) {
            throw std::runtime_error("MetalFX requires a supported native Metal 4 renderer");
        }
        std::scoped_lock lock{settings_mutex};
        display_settings.metal_fx = enabled;
        if (enabled) {
            display_settings.fsr.enable = false;
        }
    }

    void SetRcasEnabled(bool enabled) {
        std::scoped_lock lock{settings_mutex};
        display_settings.fsr.use_rcas = enabled;
    }

    void SetRcasAttenuation(float attenuation) {
        std::scoped_lock lock{settings_mutex};
        display_settings.fsr.rcas_attenuation = attenuation;
    }

    [[nodiscard]] Frontend::WindowSDL& GetWindow() const {
        return window;
    }

    [[nodiscard]] virtual Rasterizer& GetRasterizer() const = 0;
    [[nodiscard]] virtual bool IsHDRSupported() const = 0;
    [[nodiscard]] virtual bool SupportsMetalFx() const {
        return false;
    }
    virtual void SetHDR(bool enable) = 0;
    virtual void RegisterVideoOutSurface(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                                         VAddr cpu_address) = 0;
    [[nodiscard]] virtual Frame* PrepareFrame(
        const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) = 0;
    [[nodiscard]] virtual Frame* PrepareBlankFrame(bool present_thread) = 0;
    [[nodiscard]] virtual Frame* PrepareLastFrame() = 0;
    virtual void Present(Frame* frame, bool is_reusing_frame = false,
                         bool is_game_frame = true) = 0;

protected:
    explicit Presenter(Frontend::WindowSDL& window_) : window{window_} {}

    void SetHdrEnabled(bool enabled) {
        std::scoped_lock lock{settings_mutex};
        display_settings.post_processing.hdr = enabled;
    }

    Frontend::WindowSDL& window;

private:
    mutable std::mutex settings_mutex;
    DisplaySettings display_settings{};
};

} // namespace VideoCore
