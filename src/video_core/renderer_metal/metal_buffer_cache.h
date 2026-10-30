// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <vector>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_guest_page_table.h"
#include "video_core/renderer_metal/metal_guest_physical_mappings.h"
#include "video_core/renderer_metal/metal_guest_ranges.h"
#include "video_core/renderer_metal/metal_guest_read_watches.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace VideoCore {
class MemoryTracker;
class PageManager;
} // namespace VideoCore

namespace Metal {

class TextureCache;

class BufferCache {
public:
    static constexpr u32 PageShift = GuestPageTableLayout::PageBits;
    static constexpr u64 PageSize = GuestPageTableLayout::PageSize;

    BufferCache(const Instance& instance, Scheduler& scheduler, AmdGpu::Liverpool& liverpool,
                VideoCore::PageManager& page_manager);
    ~BufferCache();

    void MapMemory(VAddr address, u64 size);
    void PreparePagedBuffer(VAddr address, u64 size);
    void ObtainPagedBuffer(VAddr address, u64 size, bool is_written);
    [[nodiscard]] std::vector<BufferRange> GetPagedBufferRanges(VAddr address, u64 size);
    [[nodiscard]] BufferRange GetGdsBuffer() const;
    [[nodiscard]] std::shared_ptr<const GuestPageTableSnapshot> CreatePageSnapshot();
    [[nodiscard]] u32 ReadDataFromGds(u32 offset);
    void InvalidateMemory(VAddr address, u64 size, bool assume_locks = false);
    void ReadMemory(VAddr address, u64 size, bool is_write = false, bool assume_locks = false);
    void WriteMemory(VAddr address, std::span<const u8> data);
    void WithMemoryWriteWatch(VAddr address, u64 size, const std::function<void()>& write);
    void UnmapMemory(VAddr address, u64 size);
    void ReleaseMemory();
    [[nodiscard]] bool OverlapsGuestMemory(VAddr left, u64 left_size, VAddr right,
                                           u64 right_size) const;
    [[nodiscard]] bool IsRegionGpuModified(VAddr address, u64 size);
    [[nodiscard]] bool IsRegionCpuModified(VAddr address, u64 size);
    [[nodiscard]] u64 GetRegionVersion(VAddr address, u64 size) const;

private:
    friend class TextureCache;

    struct CachedBuffer {
        VAddr address;
        u64 size;
        id<MTLBuffer> buffer;
        u64 offset{};
    };

    struct PhysicalBuffer {
        PAddr address;
        u64 size;
        id<MTLBuffer> buffer;
        u64 offset;
        u64 mapping_epoch;
    };

    [[nodiscard]] CachedBuffer& FindBuffer(VAddr address, u64 size);
    [[nodiscard]] BufferRange ObtainPhysicalBuffer(PAddr address, u64 size, u64 mapping_epoch);
    void RetirePhysicalBuffers(PAddr address, u64 size);
    [[nodiscard]] GuestRangeSet AliasedRanges(VAddr address, u64 size) const;
    void TrackAliases(VAddr address, u64 size, bool is_written);
    void UpdateCpuReadWatches(VAddr address, u64 size, bool track);
    void SynchronizeMemory(CachedBuffer& buffer, VAddr address, u64 size, bool is_written);
    void DownloadMemory(VAddr address, u64 size);
    void WriteBacking(VAddr address, std::span<const u8> data);
    void Copy(const BufferRange& source, const BufferRange& destination, u64 size);
    static void ValidateRange(VAddr address, u64 size);
    void MarkRegionChanged(VAddr address, u64 size);
    void ObtainPagedTextureBuffer(VAddr address, u64 size, bool is_written);

    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool& liverpool;
    Core::MemoryManager& memory;
    std::unique_ptr<VideoCore::MemoryTracker> memory_tracker;
    VideoCore::PageManager& page_manager;
    StagingBufferPool staging;
    Commands commands;
    std::map<VAddr, CachedBuffer> buffers;
    std::map<PAddr, PhysicalBuffer> physical_buffers;
    GuestPhysicalMappings physical_mappings;
    std::shared_ptr<const GuestPageTableSnapshot> page_snapshot;
    GuestRangeSet gpu_modified_ranges;
    GuestRangeSet initialized_ranges;
    GuestReadWatches read_watches;
    GuestRangeMap<u64> versions;
    u64 current_version{};
    TextureCache* texture_cache{};
    id<MTLBuffer> gds_buffer;
};

} // namespace Metal
