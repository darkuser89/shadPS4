// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <deque>
#include <memory>

#include <QuartzCore/CAMetalLayer.h>

#include "video_core/renderer_metal/metal_common.h"

struct SDL_Window;

namespace Metal {

class Instance;
class Scheduler;

class Swapchain {
public:
    Swapchain(const Instance& instance, Scheduler& scheduler, CAMetalLayer* layer,
              SDL_Window* window);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    void Resize(u32 width, u32 height);
    [[nodiscard]] bool SupportsHDR() const;
    void SetHDR(bool enable);
    [[nodiscard]] id<MTLTexture> Acquire();
    void Present();

private:
    struct HdrSupport;
    void RefreshHDRSupport() const;
    const Instance& instance;
    Scheduler& scheduler;
    CAMetalLayer* layer;
    __weak id cocoa_window;
    id<MTLResidencySet> residency_set;
    id<CAMetalDrawable> drawable;
    std::deque<u64> present_ticks;
    std::shared_ptr<HdrSupport> hdr_support;
    bool hdr_enabled{};
};

} // namespace Metal
