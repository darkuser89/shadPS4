// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <imgui.h>

#include "common/types.h"

namespace Frontend {
class WindowSDL;
}
namespace Vulkan {
class Instance;
}
namespace vk {
class Device;
class CommandBuffer;
class ImageView;
struct Extent2D;
struct AllocationCallbacks;
enum class Format : int32_t;
} // namespace vk

union SDL_Event;

namespace Vulkan {
struct Frame;
}

namespace ImGui::Core {

void Initialize(const Vulkan::Instance& instance, const Frontend::WindowSDL& window,
                u32 image_count, vk::Format surface_format,
                const vk::AllocationCallbacks* allocator = nullptr);

void InitializeNative(const Frontend::WindowSDL& window, u32 max_texture_dimension);
void ShutdownNative();

void OnResize();

void OnSurfaceFormatChange(vk::Format surface_format);

void Shutdown(const vk::Device& device);

bool ProcessEvent(SDL_Event* event);

void AcquireGamepadInputCapture();
void ReleaseGamepadInputCapture();
bool IsGamepadInputCaptured();

ImGuiID NewFrame(bool is_reusing_frame = false);

void Render(const vk::CommandBuffer& cmdbuf, const vk::ImageView& image_view,
            const vk::Extent2D& extent);

bool MustKeepDrawing(); // Force the emulator redraw

} // namespace ImGui::Core
