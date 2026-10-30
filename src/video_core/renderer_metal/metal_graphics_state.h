// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

struct Viewport {
    double x{};
    double y{};
    double width{};
    double height{};
    double min_depth{};
    double max_depth{1.0};

    bool operator==(const Viewport&) const = default;
};

struct Scissor {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};

    bool operator==(const Scissor&) const = default;
};

struct DepthBias {
    float constant{};
    float slope{};
    float clamp{};

    bool operator==(const DepthBias&) const = default;
};

struct DepthBounds {
    float minimum{};
    float maximum{1.0f};
    bool enabled{};

    bool operator==(const DepthBounds&) const = default;
};

struct GraphicsState {
    std::array<Viewport, 16> viewports{};
    std::array<Scissor, 16> scissors{};
    u32 viewport_count{1};
    id<MTLDepthStencilState> depth_stencil{};
    MTLCullMode cull_mode{MTLCullModeNone};
    MTLWinding front_face{MTLWindingClockwise};
    MTLDepthClipMode depth_clip{MTLDepthClipModeClip};
    MTLTriangleFillMode fill_mode{MTLTriangleFillModeFill};
    DepthBias depth_bias{};
    DepthBounds depth_bounds{};
    std::array<float, 4> blend_constant{};
    u32 stencil_front_reference{};
    u32 stencil_back_reference{};
    MTLVisibilityResultMode visibility_mode{MTLVisibilityResultModeDisabled};
    u64 visibility_offset{};

    bool operator==(const GraphicsState&) const = default;
};

} // namespace Metal
