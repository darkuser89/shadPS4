// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_texture_transfer.h"

namespace Metal {

namespace {

constexpr std::string_view CopyRowsSource = R"(
#include <metal_stdlib>
using namespace metal;

struct RowCopyParameters {
    ulong source_offset;
    ulong destination_offset;
    ulong source_stride;
    ulong destination_stride;
    uint row_bytes;
    uint rows;
};

kernel void copy_rows(device const uchar* source [[buffer(0)]],
                      device uchar* destination [[buffer(1)]],
                      constant RowCopyParameters& params [[buffer(2)]],
                      uint2 position [[thread_position_in_grid]]) {
    const uint first = position.x * 16;
    if (position.y >= params.rows || first >= params.row_bytes)
        return;
    const ulong source_base = params.source_offset + ulong(position.y) * params.source_stride;
    const ulong destination_base =
        params.destination_offset + ulong(position.y) * params.destination_stride;
    const uint count = min(16u, params.row_bytes - first);
    for (uint byte = 0; byte < count; ++byte)
        destination[destination_base + first + byte] = source[source_base + first + byte];
}
)";

struct PixelBlock {
    u32 width;
    u32 height;
    u32 bytes;
};

PixelBlock GetPixelBlock(MTLPixelFormat format, MTLBlitOption options) {
    if (format == MTLPixelFormatDepth32Float_Stencil8) {
        if (options == MTLBlitOptionDepthFromDepthStencil) {
            return {1, 1, 4};
        }
        if (options == MTLBlitOptionStencilFromDepthStencil) {
            return {1, 1, 1};
        }
        UNREACHABLE_MSG("Metal combined depth/stencil transfer requires one plane");
    }
    if (options != MTLBlitOptionNone) {
        UNREACHABLE_MSG("Invalid Metal texture transfer plane options");
    }
    switch (format) {
    case MTLPixelFormatA8Unorm:
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Unorm_sRGB:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
    case MTLPixelFormatStencil8:
        return {1, 1, 1};
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Unorm_sRGB:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatB5G6R5Unorm:
    case MTLPixelFormatA1BGR5Unorm:
    case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm:
    case MTLPixelFormatDepth16Unorm:
        return {1, 1, 2};
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatBGR10A2Unorm:
    case MTLPixelFormatDepth32Float:
        return {1, 1, 4};
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA16Float:
        return {1, 1, 8};
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
    case MTLPixelFormatRGBA32Float:
        return {1, 1, 16};
    case MTLPixelFormatBC1_RGBA:
    case MTLPixelFormatBC1_RGBA_sRGB:
    case MTLPixelFormatBC4_RUnorm:
    case MTLPixelFormatBC4_RSnorm:
        return {4, 4, 8};
    case MTLPixelFormatBC2_RGBA:
    case MTLPixelFormatBC2_RGBA_sRGB:
    case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB:
    case MTLPixelFormatBC5_RGUnorm:
    case MTLPixelFormatBC5_RGSnorm:
    case MTLPixelFormatBC6H_RGBFloat:
    case MTLPixelFormatBC6H_RGBUfloat:
    case MTLPixelFormatBC7_RGBAUnorm:
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
        return {4, 4, 16};
    default:
        UNREACHABLE_MSG("Unsupported Metal texture transfer pixel format: {}", u32(format));
    }
}

u64 CheckedMultiply(u64 left, u64 right) {
    if (right && left > std::numeric_limits<u64>::max() / right) {
        UNREACHABLE_MSG("Metal texture transfer layout overflow");
    }
    return left * right;
}

u64 CheckedAdd(u64 left, u64 right) {
    if (left > std::numeric_limits<u64>::max() - right) {
        UNREACHABLE_MSG("Metal texture transfer layout overflow");
    }
    return left + right;
}

} // namespace

TextureTransfers::~TextureTransfers() = default;

TextureTransfers::TextureTransfers(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {
    if (scheduler.GetQueue().device != instance.GetDevice()) {
        UNREACHABLE_MSG("Metal texture transfers and scheduler must use the same device");
    }
}

void TextureTransfers::ValidateRegion(const TextureRegion& region) const {
    const auto texture = region.texture;
    if (!texture || texture.device != instance.GetDevice() || texture.framebufferOnly ||
        texture.storageMode == MTLStorageModeMemoryless ||
        region.level >= texture.mipmapLevelCount || region.level >= 64) {
        UNREACHABLE_MSG("Invalid Metal texture transfer resource");
    }
    const bool multisampled = texture.sampleCount > 1;
    if (multisampled &&
        (region.level || texture.mipmapLevelCount != 1 ||
         !std::has_single_bit(u64{texture.sampleCount}) || texture.sampleCount > 8 ||
         ![instance.GetDevice() supportsTextureSampleCount:texture.sampleCount] ||
         (texture.textureType != MTLTextureType2DMultisample &&
          texture.textureType != MTLTextureType2DMultisampleArray))) {
        UNREACHABLE_MSG("Invalid Metal multisample transfer resource");
    }
    const u64 width = std::max<u64>(1, texture.width >> region.level);
    const u64 height = std::max<u64>(1, texture.height >> region.level);
    const u64 depth = std::max<u64>(1, texture.depth >> region.level);
    u64 slices = texture.arrayLength;
    switch (texture.textureType) {
    case MTLTextureTypeCube:
    case MTLTextureTypeCubeArray:
        slices *= 6;
        break;
    case MTLTextureType1D:
    case MTLTextureType1DArray:
    case MTLTextureType2D:
    case MTLTextureType2DArray:
    case MTLTextureType2DMultisample:
    case MTLTextureType2DMultisampleArray:
        break;
    case MTLTextureType3D:
        slices = 1;
        break;
    default:
        UNREACHABLE_MSG("Unsupported Metal texture transfer type");
    }
    const auto check_dimension = [](u64 origin, u64 size, u64 extent) {
        return size && origin < extent && size <= extent - origin;
    };
    if (region.slice >= slices || !check_dimension(region.origin.x, region.size.width, width) ||
        !check_dimension(region.origin.y, region.size.height, height) ||
        !check_dimension(region.origin.z, region.size.depth, depth)) {
        UNREACHABLE_MSG("Invalid Metal texture transfer region");
    }
    const auto options = texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8
                             ? MTLBlitOptionDepthFromDepthStencil
                             : MTLBlitOptionNone;
    const auto block = GetPixelBlock(texture.pixelFormat, options);
    if ((multisampled && (block.width != 1 || block.height != 1)) ||
        region.origin.x % block.width || region.origin.y % block.height ||
        (region.size.width % block.width && region.size.width != width - region.origin.x) ||
        (region.size.height % block.height && region.size.height != height - region.origin.y)) {
        UNREACHABLE_MSG("Metal compressed transfer region must cover complete blocks");
    }
}

void TextureTransfers::ValidateBufferLayout(const BufferRange& buffer, const TextureRegion& region,
                                            const TextureBufferLayout& layout) const {
    const auto block = GetPixelBlock(region.texture.pixelFormat, layout.options);
    const u64 columns = (region.size.width + block.width - 1) / block.width;
    const u64 rows = (region.size.height + block.height - 1) / block.height;
    const u64 row_size =
        CheckedMultiply(CheckedMultiply(columns, block.bytes), region.texture.sampleCount);
    if (!buffer.buffer || buffer.buffer.device != instance.GetDevice() ||
        buffer.offset % block.bytes || buffer.offset > buffer.buffer.length ||
        buffer.size > buffer.buffer.length - buffer.offset || layout.bytes_per_row < row_size ||
        layout.bytes_per_row % block.bytes || layout.bytes_per_image % block.bytes ||
        (region.size.depth == 1 && layout.bytes_per_image) ||
        (region.size.depth > 1 &&
         layout.bytes_per_image < CheckedMultiply(layout.bytes_per_row, rows))) {
        UNREACHABLE_MSG("Invalid Metal texture transfer buffer layout");
    }
    const u64 row_end = CheckedAdd(CheckedMultiply(rows - 1, layout.bytes_per_row), row_size);
    const u64 required =
        CheckedAdd(CheckedMultiply(region.size.depth - 1, layout.bytes_per_image), row_end);
    if (required > buffer.size) {
        UNREACHABLE_MSG("Metal texture transfer exceeds buffer range");
    }
}

void TextureTransfers::Upload(const BufferRange& source, const TextureRegion& destination,
                              const TextureBufferLayout& layout) {
    Transfer(source, destination, layout, true);
}

void TextureTransfers::Download(const TextureRegion& source, const BufferRange& destination,
                                const TextureBufferLayout& layout) {
    Transfer(destination, source, layout, false);
}

void TextureTransfers::Transfer(const BufferRange& buffer, const TextureRegion& region,
                                const TextureBufferLayout& layout, bool upload) {
    ValidateRegion(region);
    ValidateBufferLayout(buffer, region, layout);
    scheduler.UseResource(buffer.buffer);
    scheduler.UseResource(region.texture);
    const auto block = GetPixelBlock(region.texture.pixelFormat, layout.options);
    const u64 max_width = region.texture.textureType == MTLTextureType3D ? 2048 : 16384;
    const u64 max_row_bytes = max_width / block.width * block.bytes;
    if (!(buffer.offset % 16) && !(layout.bytes_per_row % 16) && !(layout.bytes_per_image % 16) &&
        layout.bytes_per_row <= max_row_bytes) {
        EncodeTransfer(buffer, region, layout, upload);
        return;
    }
    const u64 bytes = u64{block.bytes} * region.texture.sampleCount;
    const u64 columns = (region.size.width + block.width - 1) / block.width;
    const u64 rows = (region.size.height + block.height - 1) / block.height;
    const u64 stripe_columns = std::min(columns, max_row_bytes / bytes);
    const u64 stripe_row_bytes = (stripe_columns * bytes + 15) & ~u64{15};
    constexpr u64 max_scratch_bytes = 4 * 1024 * 1024;
    const u64 stripe_rows = std::min(rows, max_scratch_bytes / stripe_row_bytes);
    const u64 scratch_size = stripe_row_bytes * stripe_rows;
    if (!scratch || scratch.length < scratch_size) {
        scratch = instance.CreateBuffer(scratch_size, MTLResourceStorageModePrivate);
    }
    scheduler.UseResource(scratch);
    for (u64 z = 0; z < region.size.depth; ++z) {
        for (u64 y = 0; y < rows; y += stripe_rows) {
            const u64 height = std::min(rows - y, stripe_rows);
            for (u64 x = 0; x < columns; x += stripe_columns) {
                const u64 row_bytes = std::min(columns - x, stripe_columns) * bytes;
                const u64 row_stride = (row_bytes + 15) & ~u64{15};
                const u64 offset = buffer.offset + z * layout.bytes_per_image +
                                   y * layout.bytes_per_row + x * bytes;
                TextureRegion stripe = region;
                stripe.origin.x += x * block.width;
                stripe.origin.y += y * block.height;
                stripe.origin.z += z;
                stripe.size = MTLSizeMake(
                    std::min(region.size.width - x * block.width, stripe_columns * block.width),
                    std::min(region.size.height - y * block.height, height * block.height), 1);
                const BufferRange packed{scratch, 0, row_stride * height};
                const TextureBufferLayout packed_layout{row_stride, 0, layout.options};
                const auto copy_rows = [&] {
                    const bool contiguous = height == 1 || (row_bytes == layout.bytes_per_row &&
                                                            row_bytes == row_stride);
                    if (contiguous) {
                        [scheduler.BeginCompute() copyFromBuffer:upload ? buffer.buffer : scratch
                                                    sourceOffset:upload ? offset : 0
                                                        toBuffer:upload ? scratch : buffer.buffer
                                               destinationOffset:upload ? 0 : offset
                                                            size:row_bytes * height];
                    } else {
                        const BufferRange original{buffer.buffer, offset,
                                                   buffer.size - (offset - buffer.offset)};
                        CopyRows(upload ? original : packed, upload ? packed : original,
                                 upload ? layout.bytes_per_row : row_stride,
                                 upload ? row_stride : layout.bytes_per_row, u32(row_bytes),
                                 u32(height));
                    }
                };
                if (upload) {
                    copy_rows();
                } else {
                    EncodeTransfer(packed, stripe, packed_layout, false);
                }
                scheduler.EncoderBarrier(MTLStageBlit, MTLStageBlit);
                if (upload) {
                    EncodeTransfer(packed, stripe, packed_layout, true);
                } else {
                    copy_rows();
                }
                scheduler.EncoderBarrier(MTLStageBlit, MTLStageBlit);
            }
        }
    }
}

void TextureTransfers::CopyRows(const BufferRange& source, const BufferRange& destination,
                                u64 source_stride, u64 destination_stride, u32 row_bytes,
                                u32 rows) {
    if (!row_pipeline) {
        auto library = compiler.CompileLibrary(CopyRowsSource, "metal4_texture_row_copy");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"copy_rows";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        row_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    struct Parameters {
        u64 source_offset;
        u64 destination_offset;
        u64 source_stride;
        u64 destination_stride;
        u32 row_bytes;
        u32 rows;
    };
    static_assert(sizeof(Parameters) == 40);
    const Parameters params{source.offset,      destination.offset, source_stride,
                            destination_stride, row_bytes,          rows};
    auto allocation = staging.Allocate(sizeof(params), 16);
    std::memcpy(allocation.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 3, 0, 0};
    bindings.SetBuffer(0, source.buffer);
    bindings.SetBuffer(1, destination.buffer);
    bindings.SetBuffer(2, allocation.GetBuffer(), allocation.GetOffset());
    bindings.RetainUploadAllocation(std::move(allocation));
    commands.BindCompute(row_pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake((u64{row_bytes} + 1023) / 1024, rows, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageDispatch, MTLStageDispatch | MTLStageBlit);
}

void TextureTransfers::EncodeTransfer(const BufferRange& buffer, const TextureRegion& region,
                                      const TextureBufferLayout& layout, bool upload) {
    if (upload) {
        [scheduler.BeginCompute() copyFromBuffer:buffer.buffer
                                    sourceOffset:buffer.offset
                               sourceBytesPerRow:layout.bytes_per_row
                             sourceBytesPerImage:layout.bytes_per_image
                                      sourceSize:region.size
                                       toTexture:region.texture
                                destinationSlice:region.slice
                                destinationLevel:region.level
                               destinationOrigin:region.origin
                                         options:layout.options];
    } else {
        [scheduler.BeginCompute() copyFromTexture:region.texture
                                      sourceSlice:region.slice
                                      sourceLevel:region.level
                                     sourceOrigin:region.origin
                                       sourceSize:region.size
                                         toBuffer:buffer.buffer
                                destinationOffset:buffer.offset
                           destinationBytesPerRow:layout.bytes_per_row
                         destinationBytesPerImage:layout.bytes_per_image
                                          options:layout.options];
    }
}

void TextureTransfers::Copy(const TextureRegion& source, const TextureRegion& destination) {
    ValidateRegion(source);
    ValidateRegion(destination);
    if (source.texture.sampleCount != destination.texture.sampleCount ||
        source.texture.pixelFormat != destination.texture.pixelFormat ||
        source.size.width != destination.size.width ||
        source.size.height != destination.size.height ||
        source.size.depth != destination.size.depth) {
        UNREACHABLE_MSG("Incompatible Metal texture copy regions");
    }
    if (source.texture == destination.texture && source.level == destination.level &&
        source.slice == destination.slice) {
        const auto overlaps = [](u64 start_a, u64 size_a, u64 start_b, u64 size_b) {
            return start_a < start_b + size_b && start_b < start_a + size_a;
        };
        if (overlaps(source.origin.x, source.size.width, destination.origin.x,
                     destination.size.width) &&
            overlaps(source.origin.y, source.size.height, destination.origin.y,
                     destination.size.height) &&
            overlaps(source.origin.z, source.size.depth, destination.origin.z,
                     destination.size.depth)) {
            UNREACHABLE_MSG("Overlapping Metal texture copy requires an intermediate texture");
        }
    }
    scheduler.UseResource(source.texture);
    scheduler.UseResource(destination.texture);
    [scheduler.BeginCompute() copyFromTexture:source.texture
                                  sourceSlice:source.slice
                                  sourceLevel:source.level
                                 sourceOrigin:source.origin
                                   sourceSize:source.size
                                    toTexture:destination.texture
                             destinationSlice:destination.slice
                             destinationLevel:destination.level
                            destinationOrigin:destination.origin];
}

void TextureTransfers::Copy(id<MTLTexture> source, id<MTLTexture> destination) {
    ValidateRegion(
        {.texture = source, .size = MTLSizeMake(source.width, source.height, source.depth)});
    ValidateRegion({.texture = destination,
                    .size = MTLSizeMake(destination.width, destination.height, destination.depth)});
    if (source.pixelFormat != destination.pixelFormat ||
        source.sampleCount != destination.sampleCount ||
        source.textureType != destination.textureType || source.width != destination.width ||
        source.height != destination.height || source.depth != destination.depth ||
        source.arrayLength != destination.arrayLength ||
        source.mipmapLevelCount != destination.mipmapLevelCount) {
        UNREACHABLE_MSG("Incompatible Metal whole texture copy resources");
    }
    if (source == destination) {
        return;
    }
    scheduler.UseResource(source);
    scheduler.UseResource(destination);
    [scheduler.BeginCompute() copyFromTexture:source toTexture:destination];
}

} // namespace Metal
