// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>
#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>
#include "common/logging/log.h"
#include "video_core/renderer_metal/metal_big_picture.h"
#include "video_core/renderer_metal/metal_gui.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_swapchain.h"

namespace Metal {
namespace {

CAMetalLayer* PresentationLayer(SDL_MetalView view) {
    auto layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(view);
    if (!layer) {
        throw std::runtime_error("Failed to create the Metal 4 Big Picture surface");
    }
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    auto colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    layer.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);
    return layer;
}

} // namespace

struct BigPictureRenderer::Impl {
    explicit Impl(SDL_Window* window_)
        : window{window_}, scheduler{instance}, gui{instance, scheduler},
          view{SDL_Metal_CreateView(window), SDL_Metal_DestroyView},
          swapchain{instance, scheduler, PresentationLayer(view.get()), window_} {
        auto& io = ImGui::GetIO();
        io.BackendRendererName = "shadPS4_Metal4";
        io.BackendFlags |=
            ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
        LOG_INFO(Render, "Big Picture renderer: native Metal 4 ({})", instance.GetDeviceName());
    }

    ~Impl() {
        scheduler.Finish();
        gui.ReleaseTextures();
        auto& io = ImGui::GetIO();
        io.BackendRendererName = nullptr;
        io.BackendFlags &=
            ~(ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset);
    }

    SDL_Window* window;
    Instance instance;
    Scheduler scheduler;
    GuiRenderer gui;
    std::unique_ptr<void, decltype(&SDL_Metal_DestroyView)> view;
    Swapchain swapchain;
};

BigPictureRenderer::BigPictureRenderer(SDL_Window* window) : impl{std::make_unique<Impl>(window)} {}

BigPictureRenderer::~BigPictureRenderer() = default;

ImGui::Core::TextureManager::NativeTexture BigPictureRenderer::Upload(std::span<const u8> pixels,
                                                                      u32 width, u32 height) {
    return impl->gui.Upload(pixels, width, height);
}

void BigPictureRenderer::Render(ImDrawData& data) {
    @autoreleasepool {
        int width{}, height{};
        SDL_GetWindowSizeInPixels(impl->window, &width, &height);
        if (width <= 0 || height <= 0 || data.DisplaySize.x <= 0 || data.DisplaySize.y <= 0) {
            SDL_Delay(10);
            return;
        }
        impl->scheduler.TickFrame();
        impl->swapchain.Resize(width, height);
        if (auto target = impl->swapchain.Acquire()) {
            impl->gui.Render(data, target);
            impl->swapchain.Present();
        }
    }
}

} // namespace Metal
