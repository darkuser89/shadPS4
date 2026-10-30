// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_cmask.h"
#include "video_core/renderer_metal/metal_htile.h"

#include <map>
#include <memory>
#include <vector>
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_depth_conversion.h"
#include "video_core/renderer_metal/metal_image_info.h"
#include "video_core/renderer_metal/metal_texture_transfer.h"

namespace Metal {

struct ImageViewInfo {
    static ImageViewInfo FromShader(const AmdGpu::Image& image,
                                    const Shader::ImageResource& resource);

    MTLPixelFormat format{MTLPixelFormatInvalid};
    MTLTextureType type{MTLTextureType2D};
    u32 base_level{};
    u32 levels{1};
    u32 base_slice{};
    u32 slices{1};
    AmdGpu::CompMapping mapping{AmdGpu::IdentityMapping};
    float min_lod{};
    bool convert_unorm32_depth{};

    [[nodiscard]] bool operator==(const ImageViewInfo&) const = default;
};

class TextureCache {
public:
    class Image {
    public:
        [[nodiscard]] const ImageInfo& GetInfo() const {
            return info;
        }
        [[nodiscard]] id<MTLTexture> GetTexture() const {
            return texture;
        }

    private:
        friend class TextureCache;

        struct View {
            ImageViewInfo info;
            id<MTLTexture> texture;
        };

        ImageInfo info;
        const TextureCache* owner{};
        id<MTLTexture> texture;
        id<MTLTexture> normalized_depth;
        MTLTextureUsage usage{};
        u64 version{};
        u64 visit_epoch{};
        bool valid{};
        bool gpu_modified{};
        bool normalized_depth_valid{};
        std::vector<View> views;
    };

    TextureCache(const Instance& instance, Scheduler& scheduler, BufferCache& buffer_cache);
    ~TextureCache();

    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    [[nodiscard]] Image& ObtainImage(const ImageInfo& info, MTLTextureUsage usage);
    [[nodiscard]] id<MTLTexture> ObtainView(Image& image, const ImageViewInfo& info);
    [[nodiscard]] id<MTLTexture> ObtainQueryView(const ImageInfo& image, const ImageViewInfo& view);
    void MarkWritten(Image& image);
    void ProcessColorMetadata(const ImageInfo& info, Cmask::Operation operation,
                              std::span<const u8> tiles);
    void ResolveColor(const ImageInfo& source, u32 source_slice, const ImageInfo& destination,
                      u32 destination_slice, u32 slices);
    void ResolveMemory(VAddr address, u64 size);
    void ResolveAll();
    void UnmapMemory(VAddr address, u64 size);
    void CheckTilingErrors();

private:
    struct QueryViewKey {
        MTLPixelFormat format;
        u32 width;
        u32 height;
        u32 depth;
        u32 layers;
        u32 levels;
        u32 samples;
        bool is_volume;
        ImageViewInfo view;

        static QueryViewKey FromImage(const ImageInfo& image, const ImageViewInfo& view) {
            return {image.format,
                    image.width,
                    image.height,
                    image.depth,
                    image.layers,
                    image.layout.num_mips,
                    image.layout.num_samples,
                    image.is_volume,
                    view};
        }

        bool operator==(const QueryViewKey&) const = default;
    };

    struct QueryView {
        QueryViewKey key;
        __weak id<MTLTexture> texture;
    };

    [[nodiscard]] id<MTLTexture> ObtainView(Image& image, const ImageViewInfo& info,
                                            bool query_only);
    [[nodiscard]] Image& FindImage(const ImageInfo& info, MTLTextureUsage usage);
    [[nodiscard]] BufferRange Scratch(u64 size, id<MTLBuffer> __strong& buffer);
    [[nodiscard]] BufferRange GatherBacking(const ImageInfo& info);
    [[nodiscard]] BufferRange GatherBacking(const ImageInfo& info,
                                            id<MTLBuffer> __strong& gather_scratch);
    void StoreBacking(const ImageInfo& info, const BufferRange& source);
    void UpdateUsage(Image& image, MTLTextureUsage usage);
    void Upload(Image& image, const BufferRange& source);
    void Resolve(Image& image);
    void Transfer(Image& image, const BufferRange& linear, bool upload);
    void Copy(const BufferRange& source, const BufferRange& destination);
    void BeginTransfer();
    void EndTransfer();
    void RegisterImage(Image& image);
    void ResolveMemory(VAddr address, u64 size, const Image* except);
    void ValidateImage(const ImageInfo& info, MTLTextureUsage usage) const;
    void ValidateCurrent(const Image& image) const;
    void MaterializeCmask(const ImageInfo& info,
                          Cmask::Operation operation = Cmask::Operation::Expand,
                          std::span<const u8> tiles = {});
    void MaterializeHtile(const ImageInfo& info);
    template <typename Func>
    void ForEachImage(VAddr address, u64 size, Func&& func);

    const Instance& instance;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    Commands commands;
    TextureTransfers transfers;
    DepthConversion depth_conversion;
    TileManager tiling;
    Cmask cmask;
    Htile htile;
    std::map<VAddr, std::vector<std::unique_ptr<Image>>> images;
    std::map<u64, std::vector<Image*>> pages;
    std::vector<QueryView> query_views;
    u64 visit_epoch{};
    id<MTLBuffer> scratch;
    id<MTLBuffer> backing_scratch;
    id<MTLBuffer> fmask_scratch;
    id<MTLBuffer> fmask_backing_scratch;
};

} // namespace Metal
