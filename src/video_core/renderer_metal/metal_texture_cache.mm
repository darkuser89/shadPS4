// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include "common/alignment.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_metal/metal_fmask.h"
#include "video_core/renderer_metal/metal_format.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_texture_cache.h"

namespace Metal {

ImageViewInfo ImageViewInfo::FromShader(const AmdGpu::Image& image,
                                        const Shader::ImageResource& resource) {
    if (AmdGpu::IsFmask(image.GetDataFmt())) {
        if (resource.is_written || resource.is_depth || resource.is_atomic) {
            throw std::runtime_error("Metal Fmask writable or depth views require native lowering");
        }
        auto info = FromShader(PhysicalFmaskImage(image), resource);
        info.mapping = AmdGpu::IdentityMapping;
        return info;
    }
    ImageViewInfo info;
    info.format = SurfaceFormat(image.GetDataFmt(), image.GetNumberFmt());
    if (resource.is_written) {
        info.format = LinearFormat(info.format);
    }
    if (resource.is_depth) {
        info.convert_unorm32_depth = resource.RequiresUnorm32DepthConversion(image);
        info.format =
            info.convert_unorm32_depth ? MTLPixelFormatDepth32Float : DepthFormat(info.format);
    }
    switch (image.GetViewType(resource.is_array)) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color2D:
        info.type = MTLTextureType2D;
        break;
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Cube:
        info.type = MTLTextureType2DArray;
        break;
    case AmdGpu::ImageType::Color3D:
        info.type = MTLTextureType3D;
        break;
    case AmdGpu::ImageType::Color2DMsaa:
        info.type = MTLTextureType2DMultisample;
        break;
    case AmdGpu::ImageType::Color2DMsaaArray:
        info.type = MTLTextureType2DMultisampleArray;
        break;
    default:
        throw std::runtime_error("Invalid Metal guest texture view type");
    }
    info.base_level = image.base_level;
    info.levels = image.NumViewLevels(resource.is_array);
    info.base_slice = std::min<u32>(image.base_array, image.NumLayers() - 1);
    info.slices = image.NumViewLayers(resource.is_array);
    info.mapping = resource.is_written ? AmdGpu::IdentityMapping : image.DstSelect();
    info.min_lod = std::min(float(image.min_lod) / 256.f, float(info.base_level + info.levels - 1));
    return info;
}

TextureCache::TextureCache(const Instance& instance_, Scheduler& scheduler_,
                           BufferCache& buffer_cache_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_},
      commands{instance, scheduler}, transfers{instance, scheduler},
      depth_conversion{instance, scheduler}, tiling{instance, scheduler},
      cmask{instance, scheduler}, htile{instance, scheduler} {
    if (buffer_cache.texture_cache || scheduler.GetQueue().device != instance.GetDevice() ||
        &buffer_cache.scheduler != &scheduler) {
        throw std::runtime_error("Metal texture cache requires an exclusive matching buffer cache");
    }
    buffer_cache.texture_cache = this;
}

TextureCache::~TextureCache() {
    buffer_cache.texture_cache = nullptr;
}

void TextureCache::ValidateImage(const ImageInfo& info, MTLTextureUsage usage) const {
    info.Validate();
    if (!info.layout.guest_size || info.layout.guest_size > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Metal guest texture requires an allocation");
    }
    const u32 block = info.layout.block_compressed ? 4 : 1;
    const u32 bytes = info.layout.num_bits / 8;
    u64 end{};
    for (u32 level = 0; level < info.layout.num_mips; ++level) {
        const auto& mip = info.layout.mips[level];
        const u64 row_bytes =
            u64{(mip.pitch + block - 1) / block} * bytes * info.layout.num_samples;
        const u64 slice_bytes = row_bytes * ((mip.height + block - 1) / block);
        if (mip.pitch < std::max(info.width >> level, 1u) ||
            mip.height < std::max(info.height >> level, 1u) || !slice_bytes ||
            slice_bytes > mip.size ||
            u64{std::max(info.depth >> level, 1u)} * info.layers > mip.size / slice_bytes ||
            mip.offset < end || mip.offset > info.layout.guest_size ||
            mip.size > info.layout.guest_size - mip.offset || mip.offset % bytes) {
            throw std::runtime_error("Invalid Metal guest texture mip layout");
        }
        end = u64{mip.offset} + mip.size;
    }
    if (info.layout.num_samples > 1 &&
        ((usage & (MTLTextureUsageShaderWrite | MTLTextureUsageShaderAtomic)) ||
         ![instance.GetDevice() supportsTextureSampleCount:info.layout.num_samples])) {
        throw std::runtime_error(
            "Metal multisample storage writes or sample counts require native lowering");
    }
    if (info.fmask_address && !info.cmask_address) {
        throw std::runtime_error("Metal guest image metadata requires decompression handling");
    }
    if (info.cmask_address) {
        Cmask::Validate(info);
    }
    if (info.htile_address) {
        (void)Htile::MetadataSize(info);
    }
    if (info.layout.block_compressed && !instance.GetDevice().supportsBCTextureCompression) {
        throw std::runtime_error("Metal guest BC image requires native decompression on this GPU");
    }
    if (info.layout.block_compressed &&
        (usage & (MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite |
                  MTLTextureUsageShaderAtomic))) {
        throw std::runtime_error("Metal compressed images cannot be writable attachments");
    }
    if ((info.is_depth || info.is_stencil) &&
        (usage & (MTLTextureUsageShaderWrite | MTLTextureUsageShaderAtomic))) {
        throw std::runtime_error("Metal depth/stencil storage writes require format lowering");
    }
}

void TextureCache::RegisterImage(Image& image) {
    const auto& info = image.info;
    const u64 first = info.guest_address >> BufferCache::PageShift;
    const u64 last = (info.guest_address + info.layout.guest_size - 1) >> BufferCache::PageShift;
    for (u64 page = first; page <= last; ++page) {
        pages[page].push_back(&image);
    }
}

template <typename Func>
void TextureCache::ForEachImage(VAddr address, u64 size, Func&& func) {
    BufferCache::ValidateRange(address, size);
    if (visit_epoch == std::numeric_limits<u64>::max()) {
        for (auto& [base, list] : images) {
            for (auto& image : list) {
                image->visit_epoch = 0;
            }
        }
        visit_epoch = 0;
    }
    const u64 epoch = ++visit_epoch;
    const auto aliases = buffer_cache.AliasedRanges(address, size);
    aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr begin, VAddr end) {
        const u64 first = begin >> BufferCache::PageShift;
        const u64 last = (end - 1) >> BufferCache::PageShift;
        for (auto it = pages.lower_bound(first); it != pages.end() && it->first <= last; ++it) {
            for (Image* image : it->second) {
                const auto& info = image->info;
                if (image->visit_epoch == epoch || info.guest_address >= end ||
                    begin >= info.guest_address + info.layout.guest_size) {
                    continue;
                }
                image->visit_epoch = epoch;
                func(*image);
            }
        }
    });
}

TextureCache::Image& TextureCache::FindImage(const ImageInfo& info, MTLTextureUsage usage) {
    auto& list = images[info.guest_address];
    for (auto& image : list) {
        if (image->info == info) {
            return *image;
        }
    }
    auto image = std::make_unique<Image>();
    image->info = info;
    image->owner = this;
    image->usage = usage | MTLTextureUsagePixelFormatView;
    image->texture = instance.CreateTexture(
        info.CreateDescriptor(image->usage, LinearFormat(StorageFormat(info.format))));
    auto& result = *image;
    list.push_back(std::move(image));
    RegisterImage(result);
    return result;
}

TextureCache::Image& TextureCache::ObtainImage(const ImageInfo& info, MTLTextureUsage usage) {
    ValidateImage(info, usage);
    if (info.layout.num_samples > 1) {
        usage |= MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    }
    buffer_cache.PreparePagedBuffer(info.guest_address, info.layout.guest_size);
    if (info.cmask_address) {
        MaterializeCmask(info);
    }
    if (info.htile_address) {
        MaterializeHtile(info);
    }
    auto& image = FindImage(info, usage);
    ResolveMemory(info.guest_address, info.layout.guest_size, &image);
    UpdateUsage(image, usage);
    if (image.gpu_modified) {
        ValidateCurrent(image);
        return image;
    }
    buffer_cache.ObtainPagedTextureBuffer(info.guest_address, info.layout.guest_size, false);
    const auto version = buffer_cache.GetRegionVersion(info.guest_address, info.layout.guest_size);
    if (!image.valid || image.version != version) {
        Upload(image, GatherBacking(info));
        image.version = version;
        image.valid = true;
    }
    return image;
}

void TextureCache::ProcessColorMetadata(const ImageInfo& info, Cmask::Operation operation,
                                        std::span<const u8> tiles) {
    ValidateImage(info, MTLTextureUsageShaderRead);
    Cmask::Validate(info);
    if (operation == Cmask::Operation::Expand || tiles.empty()) {
        throw std::runtime_error("Metal color metadata draw requires an explicit tile selection");
    }
    buffer_cache.PreparePagedBuffer(info.guest_address, info.layout.guest_size);
    MaterializeCmask(info, operation, tiles);
}

void TextureCache::MaterializeCmask(const ImageInfo& info, Cmask::Operation operation,
                                    std::span<const u8> tiles) {
    const u64 metadata_size = info.cmask_slice_bytes * info.layers;
    auto& memory = *Core::Memory::Instance();
    const auto backing = memory.GetBackingRanges(info.cmask_address, metadata_size);
    if (std::ranges::any_of(backing, [](const auto& range) { return !range.physical_address; })) {
        throw std::runtime_error("Metal CMASK metadata must be fully mapped");
    }
    const auto aliases = buffer_cache.AliasedRanges(info.cmask_address, metadata_size);
    if (aliases.Intersects(info.guest_address, info.layout.guest_size)) {
        throw std::runtime_error("Metal CMASK metadata overlaps its pixel backing");
    }
    std::optional<ImageInfo> fmask_info;
    if (info.fmask_address) {
        fmask_info = ImageInfo::FromFmask(info);
        ValidateImage(*fmask_info, MTLTextureUsageShaderRead);
        const auto& fmask = *fmask_info;
        const auto fmask_backing =
            memory.GetBackingRanges(fmask.guest_address, fmask.layout.guest_size);
        const auto fmask_aliases =
            buffer_cache.AliasedRanges(fmask.guest_address, fmask.layout.guest_size);
        if (std::ranges::any_of(fmask_backing,
                                [](const auto& range) { return !range.physical_address; }) ||
            fmask_aliases.Intersects(info.guest_address, info.layout.guest_size) ||
            aliases.Intersects(fmask.guest_address, fmask.layout.guest_size)) {
            throw std::runtime_error("Metal FMASK metadata is unmapped or aliases another plane");
        }
        buffer_cache.PreparePagedBuffer(fmask.guest_address, fmask.layout.guest_size);
    }
    buffer_cache.ReadMemory(info.cmask_address, metadata_size);
    std::vector<u8> metadata(metadata_size);
    memory.ReadSparseMemory(info.cmask_address, metadata);
    const auto plan = Cmask::Decode(info, metadata, operation, tiles);
    if (!plan.materialize_count) {
        return;
    }
    ResolveMemory(info.guest_address, info.layout.guest_size);
    if (plan.write_fmask) {
        ResolveMemory(fmask_info->guest_address, fmask_info->layout.guest_size);
    }
    buffer_cache.ObtainPagedTextureBuffer(info.guest_address, info.layout.guest_size, false);
    const auto source = GatherBacking(info);
    const auto linear = Scratch(info.layout.guest_size, scratch);
    tiling.Transform(source, linear, info.layout, false);
    BufferRange fmask_source{}, fmask_linear{};
    if (plan.write_fmask) {
        const auto& fmask = *fmask_info;
        buffer_cache.ObtainPagedTextureBuffer(fmask.guest_address, fmask.layout.guest_size, false);
        fmask_source = GatherBacking(fmask, fmask_backing_scratch);
        fmask_linear = Scratch(fmask.layout.guest_size, fmask_scratch);
        tiling.Transform(fmask_source, fmask_linear, fmask.layout, false);
    }
    cmask.Materialize(info, plan, linear, fmask_linear);
    if (plan.write_pixels) {
        tiling.Transform(source, linear, info.layout, true);
        StoreBacking(info, source);
    }
    if (plan.write_fmask) {
        tiling.Transform(fmask_source, fmask_linear, fmask_info->layout, true);
        StoreBacking(*fmask_info, fmask_source);
    }
    buffer_cache.WriteMemory(info.cmask_address, metadata);
}

void TextureCache::MaterializeHtile(const ImageInfo& info) {
    const u64 metadata_size = Htile::MetadataSize(info);
    auto& memory = *Core::Memory::Instance();
    const auto backing = memory.GetBackingRanges(info.htile_address, metadata_size);
    const auto aliases = buffer_cache.AliasedRanges(info.htile_address, metadata_size);
    if (std::ranges::any_of(backing, [](const auto& range) { return !range.physical_address; }) ||
        aliases.Intersects(info.guest_address, info.layout.guest_size)) {
        throw std::runtime_error("Metal HTILE metadata is unmapped or overlaps its pixel backing");
    }
    buffer_cache.ReadMemory(info.htile_address, metadata_size);
    std::vector<u8> metadata(metadata_size);
    memory.ReadSparseMemory(info.htile_address, metadata);
    const auto plan = Htile::Decode(info, metadata);
    if (plan.clear_count) {
        ResolveMemory(info.guest_address, info.layout.guest_size);
        buffer_cache.ObtainPagedTextureBuffer(info.guest_address, info.layout.guest_size, false);
        const auto source = GatherBacking(info);
        const auto linear = Scratch(info.layout.guest_size, scratch);
        tiling.Transform(source, linear, info.layout, false);
        htile.Materialize(info, plan, linear);
        tiling.Transform(source, linear, info.layout, true);
        StoreBacking(info, source);
    }
    if (plan.metadata_changed) {
        buffer_cache.WriteMemory(info.htile_address, metadata);
    }
}

void TextureCache::UpdateUsage(Image& image, MTLTextureUsage usage) {
    if ((image.usage & usage) == usage) {
        return;
    }
    if (image.gpu_modified) {
        ValidateCurrent(image);
    }
    const auto& info = image.info;
    const bool preserve =
        image.valid &&
        !buffer_cache.IsRegionCpuModified(info.guest_address, info.layout.guest_size) &&
        image.version == buffer_cache.GetRegionVersion(info.guest_address, info.layout.guest_size);
    const auto new_usage = image.usage | usage;
    auto texture = instance.CreateTexture(
        info.CreateDescriptor(new_usage, LinearFormat(StorageFormat(info.format))));
    if (preserve) {
        BeginTransfer();
        transfers.Copy(image.texture, texture);
        EndTransfer();
    }
    image.texture = texture;
    image.usage = new_usage;
    image.views.clear();
    image.normalized_depth = nil;
    image.normalized_depth_valid = false;
    image.valid = preserve;
}

void TextureCache::UnmapMemory(VAddr address, u64 size) {
    ResolveMemory(address, size);
    for (auto it = images.begin(); it != images.end();) {
        std::erase_if(it->second, [&](const auto& image) {
            const auto& info = image->info;
            if (info.guest_address >= address + size ||
                address >= info.guest_address + info.layout.guest_size)
                return false;
            const u64 first = info.guest_address >> BufferCache::PageShift;
            const u64 last =
                (info.guest_address + info.layout.guest_size - 1) >> BufferCache::PageShift;
            for (u64 page = first; page <= last; ++page) {
                auto entry = pages.find(page);
                if (entry == pages.end())
                    continue;
                std::erase(entry->second, image.get());
                if (entry->second.empty())
                    pages.erase(entry);
            }
            return true;
        });
        if (it->second.empty())
            it = images.erase(it);
        else
            ++it;
    }
}

void TextureCache::ValidateCurrent(const Image& image) const {
    const auto& info = image.info;
    if (image.owner != this || !image.valid ||
        buffer_cache.IsRegionCpuModified(info.guest_address, info.layout.guest_size) ||
        image.version !=
            buffer_cache.GetRegionVersion(info.guest_address, info.layout.guest_size)) {
        throw std::runtime_error(
            "Metal guest image changed before its pending writes were resolved");
    }
}

void TextureCache::MarkWritten(Image& image) {
    ValidateCurrent(image);
    const auto& info = image.info;
    bool overlap{};
    ForEachImage(info.guest_address, info.layout.guest_size,
                 [&](Image& other) { overlap |= &other != &image && other.gpu_modified; });
    if (overlap) {
        throw std::runtime_error("Overlapping writable Metal guest attachments require lowering");
    }
    if (!(image.usage & (MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite |
                         MTLTextureUsageShaderAtomic))) {
        throw std::runtime_error("Metal guest image was not prepared for writes");
    }
    buffer_cache.ObtainPagedTextureBuffer(info.guest_address, info.layout.guest_size, true);
    image.version = buffer_cache.GetRegionVersion(info.guest_address, info.layout.guest_size);
    image.gpu_modified = true;
    image.normalized_depth_valid = false;
}

void TextureCache::ResolveColor(const ImageInfo& source, u32 source_slice,
                                const ImageInfo& destination, u32 destination_slice, u32 slices) {
    constexpr auto usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    ValidateImage(source, usage);
    ValidateImage(destination, usage);
    if (source.is_depth || source.is_stencil || destination.is_depth || destination.is_stencil ||
        source.is_volume || destination.is_volume || source.layout.block_compressed ||
        destination.layout.block_compressed || source.format != destination.format ||
        source.width != destination.width || source.height != destination.height ||
        source.layout.num_mips != 1 || destination.layout.num_mips != 1 ||
        destination.layout.num_samples != 1 || !slices || source_slice >= source.layers ||
        slices > source.layers - source_slice || destination_slice >= destination.layers ||
        slices > destination.layers - destination_slice) {
        throw std::runtime_error("Invalid Metal color resolve surfaces or subresources");
    }
    if (source.layout.num_samples > 1) {
        switch (source.format) {
        case MTLPixelFormatR8Unorm:
        case MTLPixelFormatRG8Unorm:
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatR16Float:
        case MTLPixelFormatRG16Float:
        case MTLPixelFormatRGBA16Float:
            break;
        default:
            throw std::runtime_error("Metal color resolve format requires native shader lowering");
        }
    }
    auto& source_image = ObtainImage(source, usage);
    ImageViewInfo source_view;
    source_view.format = source.format;
    source_view.type = source.TextureType();
    source_view.base_slice = source_slice;
    source_view.slices = slices;
    id<MTLTexture> source_texture = ObtainView(source_image, source_view);
    auto& destination_image = ObtainImage(destination, usage);
    if (&source_image == &destination_image && source_slice == destination_slice) {
        return;
    }
    ImageViewInfo destination_view;
    destination_view.format = destination.format;
    destination_view.type = destination.TextureType();
    destination_view.base_slice = destination_slice;
    destination_view.slices = slices;
    const auto destination_texture = ObtainView(destination_image, destination_view);
    const auto extent = MTLSizeMake(source.width, source.height, 1);
    if (&source_image == &destination_image) {
        auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.format
                                                                             width:source.width
                                                                            height:source.height
                                                                         mipmapped:NO];
        descriptor.textureType = slices > 1 ? MTLTextureType2DArray : MTLTextureType2D;
        descriptor.arrayLength = slices;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead;
        const auto snapshot = instance.CreateTexture(descriptor);
        BeginTransfer();
        for (u32 slice = 0; slice < slices; ++slice) {
            transfers.Copy({source_texture, 0, slice, {}, extent},
                           {snapshot, 0, slice, {}, extent});
        }
        EndTransfer();
        source_texture = snapshot;
    }
    MarkWritten(destination_image);
    if (source.layout.num_samples == 1) {
        BeginTransfer();
        for (u32 slice = 0; slice < slices; ++slice) {
            transfers.Copy({source_texture, 0, slice, {}, extent},
                           {destination_texture, 0, slice, {}, extent});
        }
        EndTransfer();
        return;
    }
    for (u32 slice = 0; slice < slices; ++slice) {
        RenderState state;
        state.width = source.width;
        state.height = source.height;
        state.num_samples = source.layout.num_samples;
        state.num_color_attachments = 1;
        auto& attachment = state.color_attachments[0];
        attachment.texture = source_texture;
        attachment.slice = slice;
        attachment.resolve_texture = destination_texture;
        attachment.resolve_slice = slice;
        attachment.store_action = MTLStoreActionStoreAndMultisampleResolve;
        (void)scheduler.BeginRendering(state);
        scheduler.QueueBarrier(MTLStageAll, MTLStageAll);
        scheduler.ProducerBarrier(MTLStageAll, MTLStageAll);
        scheduler.EndEncoding();
    }
}

BufferRange TextureCache::Scratch(u64 size, id<MTLBuffer> __strong& buffer) {
    const u64 capacity = instance.GetDevice().maxBufferLength;
    if (!size || size > capacity) {
        throw std::runtime_error("Metal texture backing exceeds native buffer capacity");
    }
    if (!buffer || buffer.length < size) {
        buffer = instance.CreateBuffer(std::min(Common::AlignUp(size, u64{65536}), capacity),
                                       MTLResourceStorageModePrivate);
    }
    return {buffer, 0, size};
}

BufferRange TextureCache::GatherBacking(const ImageInfo& info) {
    return GatherBacking(info, backing_scratch);
}

BufferRange TextureCache::GatherBacking(const ImageInfo& info,
                                        id<MTLBuffer> __strong& gather_scratch) {
    const auto ranges =
        buffer_cache.GetPagedBufferRanges(info.guest_address, info.layout.guest_size);
    if (ranges.size() == 1) {
        return ranges.front();
    }
    const auto result = Scratch(info.layout.guest_size, gather_scratch);
    BeginTransfer();
    u64 offset{};
    for (const auto& range : ranges) {
        commands.CopyBuffer(range, {result.buffer, result.offset + offset, range.size}, range.size);
        offset += range.size;
    }
    EndTransfer();
    return result;
}

void TextureCache::StoreBacking(const ImageInfo& info, const BufferRange& source) {
    buffer_cache.ObtainPagedTextureBuffer(info.guest_address, info.layout.guest_size, true);
    const auto ranges =
        buffer_cache.GetPagedBufferRanges(info.guest_address, info.layout.guest_size);
    if (ranges.size() == 1 && source.buffer == ranges.front().buffer &&
        source.offset == ranges.front().offset) {
        return;
    }
    BeginTransfer();
    u64 offset{};
    for (const auto& range : ranges) {
        commands.CopyBuffer({source.buffer, source.offset + offset, range.size}, range, range.size);
        offset += range.size;
    }
    EndTransfer();
}

void TextureCache::BeginTransfer() {
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
}

void TextureCache::EndTransfer() {
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch | MTLStageBlit);
}

void TextureCache::Copy(const BufferRange& source, const BufferRange& destination) {
    BeginTransfer();
    commands.CopyBuffer(source, destination, source.size);
    EndTransfer();
}

void TextureCache::Transfer(Image& image, const BufferRange& linear, bool upload) {
    const auto& info = image.info;
    const u32 block = info.layout.block_compressed ? 4 : 1;
    const u32 bytes = info.layout.num_bits / 8;
    BeginTransfer();
    for (u32 level = 0; level < info.layout.num_mips; ++level) {
        const auto& mip = info.layout.mips[level];
        const u64 row_bytes =
            u64{(mip.pitch + block - 1) / block} * bytes * info.layout.num_samples;
        const u64 slice_bytes = row_bytes * ((mip.height + block - 1) / block);
        const u32 depth = std::max(info.depth >> level, 1u);
        if (!slice_bytes || slice_bytes > mip.size ||
            u64{depth} * info.layers > mip.size / slice_bytes) {
            throw std::runtime_error("Metal texture mip exceeds its guest allocation");
        }
        for (u32 layer = 0; layer < info.layers; ++layer) {
            const u64 offset = mip.offset + u64{layer} * slice_bytes;
            TextureRegion region{image.texture, level, layer, MTLOriginMake(0, 0, 0),
                                 MTLSizeMake(std::max(info.width >> level, 1u),
                                             std::max(info.height >> level, 1u), depth)};
            BufferRange buffer{linear.buffer, linear.offset + offset,
                               mip.size - u64{layer} * slice_bytes};
            TextureBufferLayout layout{row_bytes, depth > 1 ? slice_bytes : 0};
            if (upload) {
                transfers.Upload(buffer, region, layout);
            } else {
                transfers.Download(region, buffer, layout);
            }
        }
    }
    EndTransfer();
}

void TextureCache::Upload(Image& image, const BufferRange& source) {
    image.normalized_depth_valid = false;
    if (image.info.IsTiled()) {
        const auto linear = Scratch(image.info.layout.guest_size, scratch);
        tiling.Transform(source, linear, image.info.layout, false);
        Transfer(image, linear, true);
    } else {
        Transfer(image, source, true);
    }
}

void TextureCache::Resolve(Image& image) {
    if (!image.gpu_modified) {
        return;
    }
    ValidateCurrent(image);
    const auto& info = image.info;
    buffer_cache.ObtainPagedTextureBuffer(info.guest_address, info.layout.guest_size, false);
    const auto source = GatherBacking(info);
    const auto linear = Scratch(info.layout.guest_size, scratch);
    if (info.IsTiled()) {
        tiling.Transform(source, linear, info.layout, false);
    } else {
        Copy(source, linear);
    }
    Transfer(image, linear, false);
    if (info.IsTiled()) {
        tiling.Transform(source, linear, info.layout, true);
    } else {
        Copy(linear, source);
    }
    StoreBacking(info, source);
    image.version = buffer_cache.GetRegionVersion(info.guest_address, info.layout.guest_size);
    image.gpu_modified = false;
}

void TextureCache::ResolveMemory(VAddr address, u64 size, const Image* except) {
    ForEachImage(address, size, [&](Image& image) {
        if (&image != except) {
            Resolve(image);
        }
    });
}

void TextureCache::ResolveMemory(VAddr address, u64 size) {
    ResolveMemory(address, size, nullptr);
}

void TextureCache::ResolveAll() {
    for (auto& [address, list] : images) {
        for (auto& image : list) {
            Resolve(*image);
        }
    }
}

void TextureCache::CheckTilingErrors() {
    tiling.CheckErrors();
    cmask.CheckErrors();
}

id<MTLTexture> TextureCache::ObtainView(Image& image, const ImageViewInfo& info) {
    return ObtainView(image, info, false);
}

id<MTLTexture> TextureCache::ObtainQueryView(const ImageInfo& info, const ImageViewInfo& view) {
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (info.layout.num_samples > 1) {
        usage |= MTLTextureUsageRenderTarget;
    }
    ValidateImage(info, usage);
    std::erase_if(query_views, [](const QueryView& cached) { return !cached.texture; });
    const auto key = QueryViewKey::FromImage(info, view);
    for (const auto& cached : query_views) {
        if (cached.key == key) {
            id<MTLTexture> texture = cached.texture;
            if (texture) {
                return texture;
            }
        }
    }
    Image image;
    image.owner = this;
    image.info = info;
    auto query_view = view;
    if (view.convert_unorm32_depth) {
        if (view.format != MTLPixelFormatDepth32Float ||
            LinearFormat(StorageFormat(info.format)) != MTLPixelFormatR32Uint || info.is_volume) {
            throw std::runtime_error("Invalid Metal UNORM32 depth query view");
        }
        image.info.format = MTLPixelFormatDepth32Float;
        image.info.is_depth = true;
        query_view.convert_unorm32_depth = false;
        usage |= MTLTextureUsageRenderTarget;
    }
    image.texture = instance.CreateTexture(
        image.info.CreateDescriptor(usage, LinearFormat(StorageFormat(image.info.format))));
    const auto texture = ObtainView(image, query_view, true);
    query_views.push_back({key, texture});
    return texture;
}

id<MTLTexture> TextureCache::ObtainView(Image& image, const ImageViewInfo& info, bool query_only) {
    const bool multisampled = image.info.layout.num_samples > 1;
    const bool multisample_view =
        info.type == MTLTextureType2DMultisample || info.type == MTLTextureType2DMultisampleArray;
    if (image.owner != this || (!query_only && !image.valid) || multisampled != multisample_view ||
        !info.levels || info.base_level >= image.info.layout.num_mips ||
        info.levels > image.info.layout.num_mips - info.base_level || !info.slices ||
        info.base_slice >= image.info.layers || info.slices > image.info.layers - info.base_slice ||
        !std::isfinite(info.min_lod) || info.min_lod < 0 ||
        (info.convert_unorm32_depth
             ? info.format != MTLPixelFormatDepth32Float ||
                   image.texture.pixelFormat != MTLPixelFormatR32Uint || image.info.is_volume
             : LinearFormat(StorageFormat(info.format)) !=
                   LinearFormat(StorageFormat(image.info.format))) ||
        (info.type != MTLTextureType2D && info.type != MTLTextureType2DArray &&
         info.type != MTLTextureType3D && !multisample_view) ||
        ((info.type == MTLTextureType3D) != image.info.is_volume) ||
        ((info.type == MTLTextureType2D || info.type == MTLTextureType2DMultisample) &&
         info.slices != 1)) {
        throw std::runtime_error("Invalid Metal guest texture view");
    }
    auto source = image.texture;
    if (info.convert_unorm32_depth) {
        if (!query_only) {
            ValidateCurrent(image);
        }
        if (!image.normalized_depth) {
            image.normalized_depth = depth_conversion.CreateTexture(image.texture);
        }
        if (!query_only && !image.normalized_depth_valid) {
            depth_conversion.Convert(image.texture, image.normalized_depth);
            image.normalized_depth_valid = true;
        }
        source = image.normalized_depth;
    }
    for (const auto& view : image.views) {
        if (view.info == info) {
            return view.texture;
        }
    }
    auto descriptor = [MTLTextureViewDescriptor new];
    descriptor.pixelFormat = info.format;
    descriptor.textureType = info.type;
    descriptor.levelRange = NSMakeRange(info.base_level, info.levels);
    descriptor.sliceRange = NSMakeRange(info.base_slice, info.slices);
    descriptor.swizzle = ComponentMapping(info.mapping);
    if (info.min_lod != 0) {
        if (@available(macOS 27.0, *)) {
            descriptor.minLOD = info.min_lod;
        } else {
            throw std::runtime_error("Metal guest minimum texture LOD requires shader lowering");
        }
    }
    id<MTLTexture> texture = [source newTextureViewWithDescriptor:descriptor];
    if (!texture) {
        throw std::runtime_error("Metal device rejected the guest texture view");
    }
    image.views.push_back({info, texture});
    return texture;
}

} // namespace Metal
