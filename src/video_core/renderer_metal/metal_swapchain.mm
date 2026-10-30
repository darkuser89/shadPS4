// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cmath>
#include <stdexcept>
#import <AppKit/AppKit.h>
#include <SDL3/SDL_video.h>
#include <dispatch/dispatch.h>
#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_swapchain.h"

namespace Metal {

struct Swapchain::HdrSupport {
    std::atomic<bool> supported{};
    std::atomic<bool> refresh_pending{};
};

Swapchain::Swapchain(const Instance& instance_, Scheduler& scheduler_, CAMetalLayer* layer_,
                     SDL_Window* window)
    : instance{instance_}, scheduler{scheduler_}, layer{layer_},
      hdr_support{std::make_shared<HdrSupport>()} {
    ASSERT(layer);
    void* native_window = SDL_GetPointerProperty(SDL_GetWindowProperties(window),
                                                 SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr);
    if (!native_window) {
        throw std::runtime_error("Native Metal presentation requires a Cocoa window");
    }
    cocoa_window = (__bridge id)native_window;
    if (layer.device && layer.device != instance.GetDevice()) {
        UNREACHABLE_MSG("Metal presentation layer belongs to another device");
    }
    layer.device = instance.GetDevice();
    layer.framebufferOnly = YES;
    layer.allowsNextDrawableTimeout = YES;
    residency_set = layer.residencySet;
    if (!residency_set) {
        UNREACHABLE_MSG("Failed to obtain Metal presentation residency set");
    }
    [scheduler.GetQueue() addResidencySet:residency_set];
    RefreshHDRSupport();
}

void Swapchain::RefreshHDRSupport() const {
    auto state = hdr_support;
    if (state->refresh_pending.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    __weak id weak_window = cocoa_window;
    const auto query = ^{
      NSWindow* window = [weak_window isKindOfClass:[NSWindow class]] ? weak_window : nil;
      const CGFloat potential =
          window.screen.maximumPotentialExtendedDynamicRangeColorComponentValue;
      state->supported.store(std::isfinite(potential) && potential > 1.0,
                             std::memory_order_release);
      state->refresh_pending.store(false, std::memory_order_release);
    };
    if ([NSThread isMainThread]) {
        query();
    } else {
        dispatch_async(dispatch_get_main_queue(), query);
    }
}

bool Swapchain::SupportsHDR() const {
    RefreshHDRSupport();
    return hdr_support->supported.load(std::memory_order_acquire);
}

void Swapchain::SetHDR(bool enable) {
    ASSERT(!drawable);
    if (enable == hdr_enabled) {
        return;
    }
    if (enable && !SupportsHDR()) {
        throw std::runtime_error("Native Metal HDR requires an EDR-capable screen");
    }
    scheduler.Finish();
    auto colorspace =
        CGColorSpaceCreateWithName(enable ? kCGColorSpaceITUR_2100_PQ : kCGColorSpaceSRGB);
    if (!colorspace) {
        throw std::runtime_error("Failed to create native Metal presentation color space");
    }
    present_ticks.clear();
    layer.pixelFormat = enable ? MTLPixelFormatBGR10A2Unorm : MTLPixelFormatBGRA8Unorm_sRGB;
    layer.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);
    layer.wantsExtendedDynamicRangeContent = enable;
    auto next_residency = layer.residencySet;
    if (!next_residency) {
        throw std::runtime_error("Failed to refresh native Metal drawable residency");
    }
    if (next_residency != residency_set) {
        [scheduler.GetQueue() removeResidencySet:residency_set];
        [scheduler.GetQueue() addResidencySet:next_residency];
        residency_set = next_residency;
    }
    hdr_enabled = enable;
}

Swapchain::~Swapchain() {
    scheduler.Finish();
    scheduler.ReclaimCompletedResources();
    drawable = nil;
    if (hdr_enabled) {
        SetHDR(false);
    }
    [scheduler.GetQueue() removeResidencySet:residency_set];
}

void Swapchain::Resize(u32 width, u32 height) {
    ASSERT(!drawable);
    const CGSize size = CGSizeMake(width, height);
    if (!CGSizeEqualToSize(layer.drawableSize, size)) {
        layer.drawableSize = size;
    }
}

id<MTLTexture> Swapchain::Acquire() {
    ASSERT(!drawable);
    RefreshHDRSupport();
    if (layer.device != instance.GetDevice() || layer.residencySet != residency_set) {
        UNREACHABLE_MSG("Metal presentation layer device changed");
    }
    if (layer.drawableSize.width <= 0 || layer.drawableSize.height <= 0) {
        return nil;
    }
    while (!present_ticks.empty() && scheduler.IsFree(present_ticks.front())) {
        present_ticks.pop_front();
    }
    if (present_ticks.size() >= layer.maximumDrawableCount) {
        scheduler.Wait(present_ticks.front());
        present_ticks.pop_front();
    }
    scheduler.ReclaimCompletedResources();
    @autoreleasepool {
        drawable = [layer nextDrawable];
        if (!drawable) {
            return nil;
        }
        if (drawable.texture.device != instance.GetDevice()) {
            UNREACHABLE_MSG("Metal drawable belongs to another device");
        }
        scheduler.WaitForDrawable(drawable);
        scheduler.UseResource(drawable.texture);
        return drawable.texture;
    }
}

void Swapchain::Present() {
    ASSERT(drawable);
    present_ticks.push_back(scheduler.Present(drawable));
    drawable = nil;
}

} // namespace Metal
