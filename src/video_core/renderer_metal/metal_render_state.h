// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

struct RenderAttachment {
    id<MTLTexture> texture{};
    id<MTLTexture> resolve_texture{};
    u32 level{};
    u32 slice{};
    u32 depth_plane{};
    u32 resolve_level{};
    u32 resolve_slice{};
    u32 resolve_depth_plane{};
    MTLLoadAction load_action{MTLLoadActionLoad};
    MTLStoreAction store_action{MTLStoreActionStore};

    bool operator==(const RenderAttachment&) const = default;
};

struct ColorAttachment : RenderAttachment {
    std::array<double, 4> clear_color{};

    bool operator==(const ColorAttachment&) const = default;
};

struct DepthAttachment : RenderAttachment {
    double clear_depth{1.0};
    MTLMultisampleDepthResolveFilter resolve_filter{MTLMultisampleDepthResolveFilterSample0};

    bool operator==(const DepthAttachment&) const = default;
};

struct StencilAttachment : RenderAttachment {
    u32 clear_stencil{};
    MTLMultisampleStencilResolveFilter resolve_filter{MTLMultisampleStencilResolveFilterSample0};

    bool operator==(const StencilAttachment&) const = default;
};

struct RenderState {
    std::array<ColorAttachment, 8> color_attachments{};
    DepthAttachment depth_attachment{};
    StencilAttachment stencil_attachment{};
    id<MTLBuffer> visibility_buffer{};
    MTLVisibilityResultType visibility_result_type{MTLVisibilityResultTypeReset};
    u32 num_color_attachments{};
    u32 width{};
    u32 height{};
    u32 num_layers{1};
    u32 num_samples{1};
    std::array<std::array<u8, 2>, 8> sample_positions{};
    bool custom_sample_positions{};

    bool operator==(const RenderState&) const = default;
};

} // namespace Metal
