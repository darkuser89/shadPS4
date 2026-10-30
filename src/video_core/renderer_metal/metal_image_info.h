// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/videoout/buffer.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_metal/metal_tiling.h"

namespace Shader {
struct ImageResource;
}

namespace Metal {

struct DepthMetadata {
    VAddr address{};
    u32 surface{};
    u32 stencil_info{};
    u32 base_slice{};
    float depth_clear{};
    u32 stencil_clear{};
    bool neo_mode{};
};

struct ImageInfo {
    static ImageInfo FromShader(const AmdGpu::Image& image, const Shader::ImageResource& resource,
                                bool neo_mode);
    static ImageInfo FromColor(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent extent,
                               bool neo_mode);
    static ImageInfo FromFmask(const ImageInfo& color);
    static ImageInfo FromDepth(const AmdGpu::DepthBuffer& buffer, u32 layers,
                               AmdGpu::CbDbExtent extent, bool write_buffer, bool stencil_plane,
                               const DepthMetadata& metadata);
    static ImageInfo FromVideoOut(const Libraries::VideoOut::BufferAttributeGroup& attributes,
                                  VAddr address);

    [[nodiscard]] bool IsTiled() const;
    [[nodiscard]] MTLTextureType TextureType() const;
    [[nodiscard]] MTLTextureDescriptor* CreateDescriptor(
        MTLTextureUsage usage, MTLPixelFormat storage_format = MTLPixelFormatInvalid) const;
    [[nodiscard]] bool operator==(const ImageInfo& other) const;

    VAddr guest_address{};
    MTLPixelFormat format{MTLPixelFormatInvalid};
    u32 width{};
    u32 height{};
    u32 depth{1};
    u32 pitch{};
    u32 layers{1};
    bool is_volume{};
    bool is_depth{};
    bool is_stencil{};
    bool pow2_pad{};
    AmdGpu::CompMapping view_mapping{AmdGpu::IdentityMapping};
    VAddr cmask_address{};
    u64 cmask_slice_bytes{};
    u64 cmask_clear_value{};
    u32 cmask_base_slice{};
    u32 cmask_address_type{};
    bool cmask_linear{};
    bool cmask_fast_clear{};
    VAddr fmask_address{};
    u32 fmask_pitch{};
    u32 fmask_height{};
    u32 fmask_samples{};
    AmdGpu::TileMode fmask_tile_mode{AmdGpu::TileMode::Thin2DThin};
    VAddr htile_address{};
    u32 htile_surface{};
    u32 htile_base_slice{};
    u32 htile_clear_value{};
    AmdGpu::TileMode htile_tile_mode{AmdGpu::TileMode::Depth2DThin64};
    bool htile_stencil_disabled{};
    bool htile_neo{};
    TilingLayout layout;

private:
    friend class TextureCache;

    void UpdateSize();
    void Validate() const;
    void SetSingleMip(u64 size, u32 mip_pitch, u32 mip_height);
};

} // namespace Metal
