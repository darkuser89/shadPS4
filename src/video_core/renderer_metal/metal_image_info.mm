// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include "common/alignment.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_metal/metal_fmask.h"
#include "video_core/renderer_metal/metal_format.h"
#include "video_core/renderer_metal/metal_image_info.h"

namespace Metal {

ImageInfo ImageInfo::FromShader(const AmdGpu::Image& image, const Shader::ImageResource& resource,
                                bool neo_mode) {
    if (AmdGpu::IsFmask(image.GetDataFmt())) {
        if (resource.is_written || resource.is_depth || resource.is_atomic) {
            throw std::runtime_error("Metal Fmask writable or depth views require native lowering");
        }
        return FromShader(PhysicalFmaskImage(image), resource, neo_mode);
    }
    if (neo_mode && image.NumSamples() > 1 && image.compression_en) {
        throw std::runtime_error(
            "Metal compressed multisample shader images require metadata lowering");
    }
    ImageInfo info;
    const auto data = image.GetDataFmt();
    info.format = SurfaceFormat(data, image.GetNumberFmt());
    if (resource.is_depth && !resource.RequiresUnorm32DepthConversion(image)) {
        info.format = DepthFormat(info.format);
        info.is_depth = info.format != MTLPixelFormatStencil8;
        info.is_stencil = info.format == MTLPixelFormatStencil8;
    }
    info.guest_address = image.Address();
    info.width = image.width + 1;
    info.height = image.height + 1;
    info.is_volume = image.GetBaseType() == AmdGpu::ImageType::Color3D;
    info.depth = info.is_volume ? image.depth + 1 : 1;
    info.pitch = image.Pitch();
    info.layers = image.NumLayers();
    info.pow2_pad = image.pow2pad;
    info.layout.num_bits = AmdGpu::NumBitsPerBlock(data);
    info.layout.num_samples = image.NumSamples();
    info.layout.num_mips = image.NumLevels();
    info.layout.tile_mode = image.GetTileMode();
    info.layout.block_compressed = AmdGpu::IsBlockCoded(data);
    info.layout.alt_tile = neo_mode && image.alt_tile_mode;
    info.layout.bank_swizzle = image.alt_tile_mode ? image.GetBankSwizzle() : 0;
    info.UpdateSize();
    return info;
}

ImageInfo ImageInfo::FromColor(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent extent,
                               bool neo_mode) {
    ImageInfo info;
    const auto data = buffer.GetDataFmt();
    info.format = SurfaceFormat(data, buffer.GetNumberFmt());
    info.guest_address = buffer.Address();
    info.width = extent.Valid() ? extent.width : buffer.Pitch();
    info.height = extent.Valid() ? extent.height : buffer.Height();
    info.pitch = buffer.Pitch();
    info.layers = buffer.NumSlices();
    info.layout.num_bits = AmdGpu::NumBitsPerBlock(data);
    info.layout.num_samples = buffer.NumSamples();
    info.layout.block_compressed = AmdGpu::IsBlockCoded(data);
    info.layout.tile_mode = buffer.GetTileMode();
    info.layout.alt_tile = neo_mode && buffer.info.alt_tile_mode;
    info.cmask_address =
        buffer.info.fast_clear || buffer.info.compression ? buffer.CmaskAddress() : 0;
    if (buffer.info.fast_clear || buffer.info.compression) {
        if (!info.cmask_address || buffer.info.dcc_enable) {
            throw std::runtime_error("Invalid Metal legacy CMASK color buffer");
        }
        info.cmask_slice_bytes = (u64{buffer.cmask_slice.tile_max} + 1) * 128;
        info.cmask_clear_value = u64{buffer.clear_word0} | (u64{buffer.clear_word1} << 32);
        info.cmask_base_slice = buffer.BaseSlice();
        info.cmask_address_type = buffer.info.cmask_addr_type;
        info.cmask_linear = buffer.info.cmask_is_linear;
        info.cmask_fast_clear = buffer.info.fast_clear;
    }
    info.fmask_address = buffer.info.compression ? buffer.FmaskAddress() : 0;
    if (buffer.info.compression) {
        info.fmask_pitch = (buffer.pitch.fmask_tile_max + 1) * 8;
        const u32 pixels = (buffer.fmask_slice.tile_max + 1) * 64;
        if (!info.fmask_address || pixels % info.fmask_pitch) {
            throw std::runtime_error("Invalid Metal FMASK allocation");
        }
        info.fmask_height = pixels / info.fmask_pitch;
        info.fmask_samples = 1u << buffer.attrib.num_samples_log2;
        info.fmask_tile_mode = AmdGpu::TileMode(buffer.attrib.fmask_tile_mode_index);
    }
    if (info.IsTiled()) {
        const u64 size = (u64{buffer.slice.tile_max} + 1) * 64 * (info.layout.num_bits / 8) *
                         info.layout.num_samples * info.layers;
        info.SetSingleMip(size, info.pitch, buffer.Height());
    } else {
        info.UpdateSize();
    }
    info.Validate();
    return info;
}

ImageInfo ImageInfo::FromFmask(const ImageInfo& color) {
    if (!color.fmask_address || color.fmask_samples != color.layout.num_samples ||
        (color.fmask_samples != 2 && color.fmask_samples != 4 && color.fmask_samples != 8) ||
        color.fmask_pitch < color.layout.mips[0].pitch ||
        color.fmask_height < color.layout.mips[0].height ||
        AmdGpu::GetArrayMode(color.fmask_tile_mode) != AmdGpu::ArrayMode::Array2DTiledThin1) {
        throw std::runtime_error("Metal FMASK layout or EQAA requires additional lowering");
    }
    ImageInfo info;
    info.guest_address = color.fmask_address;
    info.format = color.fmask_samples == 8 ? MTLPixelFormatR32Uint : MTLPixelFormatR8Uint;
    info.width = color.width;
    info.height = color.height;
    info.pitch = color.fmask_pitch;
    info.layers = color.layers;
    info.layout.num_bits = color.fmask_samples == 8 ? 32 : 8;
    info.layout.tile_mode = color.fmask_tile_mode;
    info.layout.alt_tile = color.layout.alt_tile;
    info.SetSingleMip(u64{info.pitch} * color.fmask_height * info.layers *
                          (info.layout.num_bits / 8),
                      info.pitch, color.fmask_height);
    info.Validate();
    return info;
}

ImageInfo ImageInfo::FromDepth(const AmdGpu::DepthBuffer& buffer, u32 layers,
                               AmdGpu::CbDbExtent extent, bool write_buffer, bool stencil_plane,
                               const DepthMetadata& metadata) {
    ImageInfo info;
    info.is_depth = !stencil_plane;
    info.is_stencil = stencil_plane;
    if (stencil_plane) {
        if (buffer.stencil_info.format == AmdGpu::DepthBuffer::StencilFormat::Invalid) {
            throw std::runtime_error("Invalid Metal guest stencil attachment format");
        }
        info.format = MTLPixelFormatStencil8;
        info.guest_address = write_buffer ? buffer.StencilWriteAddress() : buffer.StencilAddress();
        info.layout.num_bits = 8;
    } else {
        switch (buffer.z_info.format) {
        case AmdGpu::DepthBuffer::ZFormat::Z16:
            info.format = MTLPixelFormatDepth16Unorm;
            break;
        case AmdGpu::DepthBuffer::ZFormat::Z32Float:
            info.format = MTLPixelFormatDepth32Float;
            break;
        default:
            throw std::runtime_error("Invalid Metal guest depth attachment format");
        }
        info.guest_address = write_buffer ? buffer.DepthWriteAddress() : buffer.DepthAddress();
        info.layout.num_bits = buffer.NumBits();
    }
    info.width = extent.Valid() ? extent.width : buffer.Pitch();
    info.height = extent.Valid() ? extent.height : buffer.Height();
    info.pitch = buffer.Pitch();
    info.layers = layers;
    info.layout.num_samples = buffer.NumSamples();
    info.layout.tile_mode =
        stencil_plane ? AmdGpu::TileMode((metadata.stencil_info >> 20) & 7) : buffer.GetTileMode();
    info.htile_stencil_disabled = (metadata.stencil_info >> 29) & 1;
    info.htile_address =
        buffer.z_info.tile_surface_enable && !(stencil_plane && info.htile_stencil_disabled)
            ? metadata.address
            : 0;
    if (info.htile_address) {
        info.htile_surface = metadata.surface;
        info.htile_base_slice = metadata.base_slice;
        info.htile_tile_mode = buffer.GetTileMode();
        info.htile_neo = metadata.neo_mode;
        if (stencil_plane) {
            info.htile_clear_value = metadata.stencil_clear & 0xff;
        } else {
            info.htile_clear_value = std::bit_cast<u32>(metadata.depth_clear);
        }
    } else if (buffer.z_info.tile_surface_enable &&
               !(stencil_plane && info.htile_stencil_disabled)) {
        throw std::runtime_error("Invalid Metal HTILE metadata address");
    }
    if (info.IsTiled()) {
        const u64 size = (u64{buffer.depth_slice.tile_max} + 1) * 64 * (info.layout.num_bits / 8) *
                         info.layout.num_samples * layers;
        info.SetSingleMip(size, info.pitch, buffer.Height());
    } else {
        info.UpdateSize();
    }
    info.Validate();
    return info;
}

ImageInfo ImageInfo::FromVideoOut(const Libraries::VideoOut::BufferAttributeGroup& attributes,
                                  VAddr address) {
    const auto& attribute = attributes.attrib;
    ImageInfo info;
    switch (attribute.pixel_format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        info.format = MTLPixelFormatRGBA8Unorm_sRGB;
        info.layout.num_bits = 32;
        break;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        info.format = MTLPixelFormatBGRA8Unorm_sRGB;
        info.layout.num_bits = 32;
        break;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        info.format = MTLPixelFormatBGR10A2Unorm;
        info.layout.num_bits = 32;
        break;
    case Libraries::VideoOut::PixelFormat::A16R16G16B16Float:
        info.format = MTLPixelFormatRGBA16Float;
        info.view_mapping = {.r = AmdGpu::CompSwizzle::Blue,
                             .g = AmdGpu::CompSwizzle::Green,
                             .b = AmdGpu::CompSwizzle::Red,
                             .a = AmdGpu::CompSwizzle::Alpha};
        info.layout.num_bits = 64;
        break;
    default:
        throw std::runtime_error("Unsupported Metal guest video-out format");
    }
    info.guest_address = address;
    info.width = attribute.width;
    info.height = attribute.height;
    switch (attribute.tiling_mode) {
    case Libraries::VideoOut::TilingMode::Tile:
        info.layout.tile_mode = AmdGpu::TileMode::Display2DThin;
        info.pitch = Common::AlignUp(info.width, 128u);
        break;
    case Libraries::VideoOut::TilingMode::Linear:
        info.layout.tile_mode = AmdGpu::TileMode::DisplayLinearAligned;
        info.pitch = attribute.pitch_in_pixel ? attribute.pitch_in_pixel : info.width;
        break;
    default:
        throw std::runtime_error("Invalid Metal guest video-out tiling mode");
    }
    if (attribute.tiling_mode == Libraries::VideoOut::TilingMode::Linear) {
        info.Validate();
        info.SetSingleMip(u64{info.pitch} * info.height * (info.layout.num_bits / 8), info.pitch,
                          info.height);
        info.Validate();
    } else {
        info.UpdateSize();
    }
    return info;
}

bool ImageInfo::IsTiled() const {
    return layout.tile_mode != AmdGpu::TileMode::DisplayLinearAligned &&
           layout.tile_mode != AmdGpu::TileMode::DisplayLinearGeneral;
}

void ImageInfo::Validate() const {
    if (format == MTLPixelFormatInvalid || !width || !height || !depth || width > 16384 ||
        height > 16384 || pitch < width || pitch > 16384 || !layers || layers > 2048 ||
        (is_volume && (layers != 1 || width > 2048 || height > 2048 || depth > 2048)) ||
        (!is_volume && depth != 1) || !layout.num_mips || layout.num_mips > layout.mips.size() ||
        layout.num_mips > std::bit_width(std::max({width, height, depth})) ||
        !std::has_single_bit(layout.num_samples) || layout.num_samples > 8 ||
        (layout.num_samples > 1 &&
         (is_volume || layout.num_mips != 1 || layout.block_compressed)) ||
        !layout.num_bits || layout.num_bits % 8) {
        throw std::runtime_error("Invalid Metal guest image dimensions or format");
    }
    constexpr u64 address_space_size = u64{1} << 40;
    if (guest_address >= address_space_size ||
        layout.guest_size > address_space_size - guest_address) {
        throw std::runtime_error("Invalid Metal guest image address range");
    }
}

void ImageInfo::SetSingleMip(u64 size, u32 mip_pitch, u32 mip_height) {
    if (!size || size > std::numeric_limits<u32>::max() ||
        mip_pitch > std::numeric_limits<u16>::max() ||
        mip_height > std::numeric_limits<u16>::max() || mip_pitch < width || mip_height < height) {
        throw std::runtime_error("Metal guest attachment exceeds its allocation");
    }
    layout.guest_size = size;
    layout.mips[0] = {u32(size), u16(mip_pitch), u16(mip_height), 0};
}

void ImageInfo::UpdateSize() {
    Validate();
    const auto mode = AmdGpu::GetArrayMode(layout.tile_mode);
    const u32 thickness = AmdGpu::GetMicroTileThickness(mode);
    const bool macro = AmdGpu::IsMacroTiled(mode);
    const u32 bytes = layout.num_bits / 8;
    layout.guest_size = 0;
    layout.micro_tiled_mips = 0;
    for (u32 level = 0; level < layout.num_mips; ++level) {
        u32 mip_pitch = std::max(pitch >> level, 1u);
        u32 mip_height = std::max(height >> level, 1u);
        u32 mip_depth = std::max(depth >> level, 1u);
        if (layout.block_compressed) {
            mip_pitch = (mip_pitch + 3) / 4;
            mip_height = (mip_height + 3) / 4;
        }
        if (pow2_pad) {
            mip_pitch = std::bit_ceil(mip_pitch);
            mip_height = std::bit_ceil(mip_height);
            mip_depth = std::bit_ceil(mip_depth);
        }
        u32 mip_thickness = 1;
        const auto slice_size = [&] {
            return u64{mip_pitch} * mip_height * bytes * layout.num_samples;
        };
        if (mode == AmdGpu::ArrayMode::ArrayLinearGeneral) {
        } else if (mode == AmdGpu::ArrayMode::ArrayLinearAligned) {
            const u32 pitch_alignment = std::max(8u, 64 / bytes);
            const u32 slice_alignment = std::max(64u, 256 / bytes);
            mip_pitch = Common::AlignUp(mip_pitch, pitch_alignment);
            while ((u64{mip_pitch} * mip_height * layout.num_samples) % slice_alignment) {
                mip_pitch += pitch_alignment;
            }
        } else if (macro && VideoCore::IsMacroTiledMip(mip_pitch, mip_height, layout.num_bits,
                                                       layout.num_samples, layout.tile_mode, level,
                                                       layout.alt_tile)) {
            mip_thickness = thickness;
            const auto [pitch_alignment, height_alignment] = VideoCore::GetMacroTileExtents(
                layout.tile_mode, layout.num_bits, layout.num_samples, layout.alt_tile);
            if (!pitch_alignment || !height_alignment) {
                throw std::runtime_error("Invalid Metal guest macrotile alignment");
            }
            mip_pitch = Common::AlignUp(mip_pitch, pitch_alignment);
            mip_height = Common::AlignUp(mip_height, height_alignment);
        } else {
            mip_thickness = std::min(thickness, 4u);
            mip_pitch = Common::AlignUp(mip_pitch, 8u);
            mip_height = Common::AlignUp(mip_height, 8u);
            while ((slice_size() * mip_thickness) % 256) {
                mip_pitch += 8;
            }
            if (macro) {
                layout.micro_tiled_mips |= 1u << level;
            }
        }
        const u64 slices = Common::AlignUp(u64{mip_depth} * layers, u64{mip_thickness});
        const u64 size = slice_size() * slices;
        if (layout.block_compressed) {
            mip_pitch *= 4;
            mip_height *= 4;
        }
        if (size > std::numeric_limits<u32>::max() ||
            layout.guest_size > std::numeric_limits<u32>::max() - size ||
            mip_pitch > std::numeric_limits<u16>::max() ||
            mip_height > std::numeric_limits<u16>::max()) {
            throw std::runtime_error("Metal guest image layout exceeds current tiling limits");
        }
        layout.mips[level] = {u32(size), u16(mip_pitch), u16(mip_height), u32(layout.guest_size)};
        layout.guest_size += size;
    }
    Validate();
}

MTLTextureType ImageInfo::TextureType() const {
    if (is_volume) {
        return MTLTextureType3D;
    }
    if (layout.num_samples > 1) {
        return layers > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
    }
    return layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
}

MTLTextureDescriptor* ImageInfo::CreateDescriptor(MTLTextureUsage usage,
                                                  MTLPixelFormat storage_format) const {
    Validate();
    if (!layout.guest_size) {
        throw std::runtime_error("Metal guest texture requires a complete image layout");
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.pixelFormat = storage_format == MTLPixelFormatInvalid ? format : storage_format;
    descriptor.textureType = TextureType();
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.arrayLength = layers;
    descriptor.mipmapLevelCount = layout.num_mips;
    descriptor.sampleCount = layout.num_samples;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    descriptor.usage = usage | MTLTextureUsagePixelFormatView;
    return descriptor;
}

bool ImageInfo::operator==(const ImageInfo& other) const {
    if (guest_address != other.guest_address ||
        StorageFormat(format) != StorageFormat(other.format) || width != other.width ||
        height != other.height || depth != other.depth || pitch != other.pitch ||
        layers != other.layers || is_volume != other.is_volume || is_depth != other.is_depth ||
        is_stencil != other.is_stencil || pow2_pad != other.pow2_pad ||
        cmask_address != other.cmask_address || fmask_address != other.fmask_address ||
        cmask_slice_bytes != other.cmask_slice_bytes ||
        cmask_clear_value != other.cmask_clear_value ||
        cmask_base_slice != other.cmask_base_slice || cmask_fast_clear != other.cmask_fast_clear ||
        fmask_pitch != other.fmask_pitch || fmask_height != other.fmask_height ||
        fmask_samples != other.fmask_samples || fmask_tile_mode != other.fmask_tile_mode ||
        cmask_address_type != other.cmask_address_type || cmask_linear != other.cmask_linear ||
        htile_address != other.htile_address || htile_surface != other.htile_surface ||
        htile_base_slice != other.htile_base_slice ||
        htile_clear_value != other.htile_clear_value || htile_tile_mode != other.htile_tile_mode ||
        htile_stencil_disabled != other.htile_stencil_disabled || htile_neo != other.htile_neo ||
        layout.num_bits != other.layout.num_bits ||
        layout.num_samples != other.layout.num_samples ||
        layout.num_mips != other.layout.num_mips ||
        layout.micro_tiled_mips != other.layout.micro_tiled_mips ||
        layout.bank_swizzle != other.layout.bank_swizzle ||
        layout.guest_size != other.layout.guest_size ||
        layout.tile_mode != other.layout.tile_mode ||
        layout.block_compressed != other.layout.block_compressed ||
        layout.alt_tile != other.layout.alt_tile) {
        return false;
    }
    return std::ranges::equal(layout.mips, other.layout.mips,
                              [](const auto& left, const auto& right) {
                                  return left.size == right.size && left.pitch == right.pitch &&
                                         left.height == right.height && left.offset == right.offset;
                              });
}

} // namespace Metal
