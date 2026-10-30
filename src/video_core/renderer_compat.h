// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

namespace Vulkan {
class Presenter;
class Rasterizer;
} // namespace Vulkan

namespace VideoCore {

class Presenter;
class Rasterizer;

[[nodiscard]] std::unique_ptr<Rasterizer> AdaptRasterizer(Vulkan::Rasterizer* rasterizer);
[[nodiscard]] std::unique_ptr<Presenter> AdaptPresenter(
    std::unique_ptr<Vulkan::Presenter> presenter);
[[nodiscard]] Vulkan::Rasterizer* GetVulkanRasterizer(Rasterizer& rasterizer);

} // namespace VideoCore
