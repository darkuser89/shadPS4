// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <libkern/OSCacheControl.h>
#include "common/alignment.h"
#include "common/small_vector.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/rasterizer.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_texture_cache.h"

namespace Metal {

BufferCache::BufferCache(const Instance& instance_, Scheduler& scheduler_,
                         AmdGpu::Liverpool& liverpool_, VideoCore::PageManager& page_manager_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{*Core::Memory::Instance()},
      memory_tracker{std::make_unique<VideoCore::MemoryTracker>(page_manager_)},
      page_manager{page_manager_}, staging{instance, scheduler}, commands{instance, scheduler},
      gds_buffer{instance.CreateBuffer(64 * 1024, MTLResourceStorageModeShared)} {
    std::memset(gds_buffer.contents, 0, gds_buffer.length);
}

BufferCache::~BufferCache() = default;

void BufferCache::ValidateRange(VAddr address, u64 size) {
    constexpr u64 AddressSpaceSize = u64{1} << 40;
    if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address) {
        throw std::runtime_error("Invalid Metal guest buffer range");
    }
}

void BufferCache::Copy(const BufferRange& source, const BufferRange& destination, u64 size) {
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    commands.CopyBuffer(source, destination, size);
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch | MTLStageBlit);
}

void BufferCache::MapMemory(VAddr address, u64 size) {
    ValidateRange(address, size);
    const auto backing = memory.GetBackingRanges(address, size);
    std::vector<GuestPhysicalMappings::BackingRange> ranges;
    ranges.reserve(backing.size());
    for (const auto& range : backing) {
        ranges.push_back({range.address, range.size, range.physical_address});
    }
    physical_mappings.Map(ranges);
    GuestRangeSet initialized;
    GuestRangeSet modified;
    GuestRangeSet cpu_modified;
    GuestRangeMap<u64> inherited_versions;
    for (const auto& alias : physical_mappings.GetAliases(address, size)) {
        initialized_ranges.ForEachInRange(
            alias.alias_address, alias.size, [&](VAddr first, VAddr last) {
                initialized.Add(alias.address + (first - alias.alias_address), last - first);
                while (first < last) {
                    constexpr u64 CpuPageSize = 4096;
                    const u64 length =
                        std::min<u64>(last - first, CpuPageSize - first % CpuPageSize);
                    if (memory_tracker->IsRegionCpuModified(first, length)) {
                        cpu_modified.Add(alias.address + (first - alias.alias_address), length);
                    }
                    first += length;
                }
            });
        gpu_modified_ranges.ForEachInRange(
            alias.alias_address, alias.size, [&](VAddr first, VAddr last) {
                modified.Add(alias.address + (first - alias.alias_address), last - first);
            });
        versions.ForEachInRange(
            alias.alias_address, alias.size, [&](VAddr first, VAddr last, u64 version) {
                inherited_versions.Add(alias.address + (first - alias.alias_address), last - first,
                                       version);
            });
    }
    initialized.ForEachInRange(address, size, [&](VAddr first, VAddr last) {
        memory_tracker->ForEachUploadRange(first, last - first, false, [](VAddr, u64) {});
        initialized_ranges.Add(first, last - first);
    });
    cpu_modified.ForEachInRange(address, size, [&](VAddr first, VAddr last) {
        memory_tracker->MarkRegionAsCpuModified(first, last - first);
    });
    modified.ForEachInRange(address, size, [&](VAddr first, VAddr last) {
        gpu_modified_ranges.Add(first, last - first);
        memory_tracker->ForEachUploadRange(first, last - first, false, [](VAddr, u64) {});
        UpdateCpuReadWatches(first, last - first, true);
    });
    inherited_versions.ForEach(
        [&](VAddr first, VAddr last, u64 version) { versions.Add(first, last - first, version); });
}

GuestRangeSet BufferCache::AliasedRanges(VAddr address, u64 size) const {
    GuestRangeSet ranges;
    ranges.Add(address, size);
    for (const auto& alias : physical_mappings.GetAliases(address, size)) {
        ranges.Add(alias.alias_address, alias.size);
    }
    return ranges;
}

void BufferCache::TrackAliases(VAddr address, u64 size, bool is_written) {
    constexpr u64 CpuPageSize = 4096;
    const VAddr begin = Common::AlignDown(address, CpuPageSize);
    const VAddr end = Common::AlignUp(address + size, CpuPageSize);
    const auto aliases = AliasedRanges(begin, end - begin);
    aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        memory_tracker->ForEachUploadRange(first, last - first, false, [](VAddr, u64) {});
        initialized_ranges.Add(first, last - first);
        if (is_written) {
            UpdateCpuReadWatches(first, last - first, true);
        }
    });
}

void BufferCache::UpdateCpuReadWatches(VAddr address, u64 size, bool track) {
    read_watches.Update(
        address, size, track,
        [this](std::span<const GuestReadWatches::Range> ranges, bool enabled) {
            for (const auto& range : ranges) {
                VAddr cursor = range.address;
                const VAddr end = cursor + range.size;
                while (cursor < end) {
                    const VAddr base = Common::AlignDown(cursor, VideoCore::HIGHER_PAGE_SIZE);
                    const u64 offset = cursor - base;
                    const u64 length =
                        std::min<u64>(end - cursor, VideoCore::HIGHER_PAGE_SIZE - offset);
                    auto bounds = VideoCore::RegionManager::GetBounds(offset, length);
                    bounds.start_page %= VideoCore::PAGES_PER_WORD;
                    bounds.end_page %= VideoCore::PAGES_PER_WORD;
                    VideoCore::RegionBits write_mask{};
                    VideoCore::RegionBits read_mask{};
                    VideoCore::RegionManager::IterateWords(
                        bounds, [&](u64 index, u64 mask) { read_mask[index] = mask; });
                    page_manager.UpdatePageWatchersForRegion(
                        base, bounds, write_mask, read_mask, VideoCore::PageOp::None,
                        enabled ? VideoCore::PageOp::Track : VideoCore::PageOp::Untrack);
                    cursor += length;
                }
            }
        });
}

BufferRange BufferCache::ObtainPhysicalBuffer(PAddr address, u64 size, u64 mapping_epoch) {
    auto next = physical_buffers.upper_bound(address);
    if (next != physical_buffers.begin()) {
        const auto& previous = std::prev(next)->second;
        if (address < previous.address + previous.size) {
            if (previous.mapping_epoch != mapping_epoch) {
                throw std::runtime_error("Metal physical buffer mapping epoch is stale");
            }
            const u64 offset = address - previous.address;
            return {previous.buffer, previous.offset + offset,
                    std::min(size, previous.size - offset)};
        }
    }
    const u64 length =
        next == physical_buffers.end() ? size : std::min<u64>(size, next->first - address);
    auto buffer = instance.CreateBuffer(length, MTLResourceStorageModePrivate);
    physical_buffers.emplace(address, PhysicalBuffer{address, length, buffer, 0, mapping_epoch});
    return {buffer, 0, length};
}

void BufferCache::RetirePhysicalBuffers(PAddr address, u64 size) {
    auto first = physical_buffers.lower_bound(address);
    if (first != physical_buffers.begin()) {
        const auto previous = std::prev(first);
        if (previous->first + previous->second.size > address) {
            first = previous;
        }
    }
    auto last = first;
    std::map<PAddr, PhysicalBuffer> replacements;
    while (last != physical_buffers.end() && last->first < address + size) {
        const auto& old = last->second;
        for (const auto& range : physical_mappings.GetPhysicalRanges(old.address, old.size)) {
            if (range.mapping_epoch == old.mapping_epoch) {
                replacements.emplace(range.address,
                                     PhysicalBuffer{range.address, range.size, old.buffer,
                                                    old.offset + (range.address - old.address),
                                                    old.mapping_epoch});
            }
        }
        ++last;
    }
    physical_buffers.erase(first, last);
    physical_buffers.merge(replacements);
}

void BufferCache::PreparePagedBuffer(VAddr address, u64 size) {
    ValidateRange(address, size);
    const u64 capacity = Common::AlignDown(u64{instance.GetDevice().maxBufferLength}, PageSize);
    if (!capacity) {
        throw std::runtime_error("Metal device cannot allocate guest pages");
    }
    VAddr begin = Common::AlignDown(address, PageSize);
    const VAddr end = Common::AlignUp(address + size, PageSize);
    while (begin < end) {
        auto next = buffers.upper_bound(begin);
        if (next != buffers.begin()) {
            const auto& previous = std::prev(next)->second;
            if (begin < previous.address + previous.size) {
                begin = std::min<VAddr>(end, previous.address + previous.size);
                continue;
            }
        }
        const VAddr gap_end = next == buffers.end() ? end : std::min(end, next->first);
        const u64 length = std::min<u64>(capacity, gap_end - begin);
        for (const auto& range : physical_mappings.GetRanges(begin, length)) {
            u64 offset{};
            while (offset < range.size) {
                const auto storage =
                    range.physical_address
                        ? ObtainPhysicalBuffer(*range.physical_address + offset,
                                               range.size - offset, range.mapping_epoch)
                        : BufferRange{
                              instance.CreateBuffer(range.size, MTLResourceStorageModePrivate), 0,
                              range.size};
                CachedBuffer buffer{range.address + offset, storage.size, storage.buffer,
                                    storage.offset};
                page_snapshot.reset();
                buffers.emplace(buffer.address, std::move(buffer));
                offset += storage.size;
            }
        }
        begin += length;
    }
}

void BufferCache::ObtainPagedBuffer(VAddr address, u64 size, bool is_written) {
    ValidateRange(address, size);
    if (texture_cache) {
        texture_cache->ResolveMemory(address, size);
    }
    ObtainPagedTextureBuffer(address, size, is_written);
}

void BufferCache::ObtainPagedTextureBuffer(VAddr address, u64 size, bool is_written) {
    ValidateRange(address, size);
    const VAddr end = address + size;
    while (address < end) {
        auto& buffer = FindBuffer(address, 1);
        const u64 length = std::min<u64>(end - address, buffer.address + buffer.size - address);
        SynchronizeMemory(buffer, address, length, is_written);
        address += length;
    }
}

std::vector<BufferRange> BufferCache::GetPagedBufferRanges(VAddr address, u64 size) {
    ValidateRange(address, size);
    std::vector<BufferRange> ranges;
    const VAddr end = address + size;
    while (address < end) {
        const auto& buffer = FindBuffer(address, 1);
        const u64 length = std::min<u64>(end - address, buffer.address + buffer.size - address);
        ranges.push_back({buffer.buffer, buffer.offset + (address - buffer.address), length});
        address += length;
    }
    return ranges;
}

BufferCache::CachedBuffer& BufferCache::FindBuffer(VAddr address, u64 size) {
    ValidateRange(address, size);
    auto it = buffers.upper_bound(address);
    if (it == buffers.begin()) {
        throw std::runtime_error("Metal guest buffer must be prepared before binding");
    }
    auto& buffer = std::prev(it)->second;
    const u64 offset = address - buffer.address;
    if (offset > buffer.size || size > buffer.size - offset) {
        throw std::runtime_error("Metal guest buffer does not cover the requested range");
    }
    return buffer;
}

void BufferCache::SynchronizeMemory(CachedBuffer& buffer, VAddr address, u64 size,
                                    bool is_written) {
    std::vector<std::pair<VAddr, u64>> uploads;
    memory_tracker->ForEachUploadRange(address, size, false, [&](VAddr start, u64 length) {
        if (start < buffer.address || length > buffer.size ||
            start - buffer.address > buffer.size - length) {
            throw std::runtime_error("Metal tracked upload exceeds its guest buffer");
        }
        auto allocation = staging.Allocate(length);
        memory.ReadSparseMemory(start, allocation.UploadData());
        Copy({allocation.GetBuffer(), allocation.GetOffset(), length},
             {buffer.buffer, buffer.offset + (start - buffer.address), length}, length);
        MarkRegionChanged(start, length);
        uploads.emplace_back(start, length);
    });
    for (const auto& [start, length] : uploads) {
        TrackAliases(start, length, false);
    }
    if (is_written) {
        const auto aliases = AliasedRanges(address, size);
        aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
            gpu_modified_ranges.Add(first, last - first);
        });
        TrackAliases(address, size, true);
        MarkRegionChanged(address, size);
    }
}

BufferRange BufferCache::GetGdsBuffer() const {
    return {gds_buffer, 0, gds_buffer.length};
}

std::shared_ptr<const GuestPageTableSnapshot> BufferCache::CreatePageSnapshot() {
    if (page_snapshot) {
        return page_snapshot;
    }
    std::vector<GuestPageTableRange> ranges;
    ranges.reserve(buffers.size());
    for (const auto& [address, buffer] : buffers) {
        ranges.push_back({address, {buffer.buffer, buffer.offset, buffer.size}});
    }
    page_snapshot = std::make_shared<GuestPageTableSnapshot>(instance, staging, ranges);
    return page_snapshot;
}

u32 BufferCache::ReadDataFromGds(u32 offset) {
    if (offset % sizeof(u32) || offset > gds_buffer.length - sizeof(u32)) {
        throw std::runtime_error("Invalid Metal guest GDS read offset");
    }
    scheduler.Finish();
    u32 value;
    std::memcpy(&value, static_cast<const u8*>(gds_buffer.contents) + offset, sizeof(value));
    return value;
}

bool BufferCache::OverlapsGuestMemory(VAddr left, u64 left_size, VAddr right,
                                      u64 right_size) const {
    if (!left_size || !right_size)
        return false;
    ValidateRange(left, left_size);
    ValidateRange(right, right_size);
    return AliasedRanges(left, left_size).Intersects(right, right_size);
}

bool BufferCache::IsRegionGpuModified(VAddr address, u64 size) {
    ValidateRange(address, size);
    return gpu_modified_ranges.Intersects(address, size);
}

bool BufferCache::IsRegionCpuModified(VAddr address, u64 size) {
    ValidateRange(address, size);
    return memory_tracker->IsRegionCpuModified(address, size);
}

void BufferCache::MarkRegionChanged(VAddr address, u64 size) {
    if (current_version == std::numeric_limits<u64>::max()) {
        throw std::runtime_error("Metal guest memory version exhausted");
    }
    const u64 version = ++current_version;
    const auto aliases = AliasedRanges(address, size);
    aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        versions.Add(first, last - first, version);
    });
}

u64 BufferCache::GetRegionVersion(VAddr address, u64 size) const {
    ValidateRange(address, size);
    u64 version{};
    versions.ForEachInRange(address, size,
                            [&](VAddr, VAddr, u64 value) { version = std::max(version, value); });
    return version;
}

void BufferCache::DownloadMemory(VAddr address, u64 size) {
    ValidateRange(address, size);
    constexpr u64 CpuPageSize = 4096;
    const VAddr begin = Common::AlignDown(address, CpuPageSize);
    const VAddr end = Common::AlignUp(address + size, CpuPageSize);
    struct Download {
        VAddr address;
        u64 offset;
        BufferRange source;
    };
    struct Batch {
        std::vector<Download> downloads;
        u64 size{};
    };
    const u64 batch_size = Common::AlignDown(u64{instance.GetDevice().maxBufferLength}, u64{64});
    if (!batch_size) {
        throw std::runtime_error("Metal device cannot allocate aligned readback batches");
    }
    std::vector<Batch> batches;
    GuestRangeSet executable;
    gpu_modified_ranges.ForEachInRange(begin, end - begin, [&](VAddr first, VAddr last) {
        PreparePagedBuffer(first, last - first);
        while (first < last) {
            if (batches.empty() || batches.back().size == batch_size) {
                batches.emplace_back();
            }
            auto& batch = batches.back();
            auto& buffer = FindBuffer(first, 1);
            const u64 count = std::min<u64>(
                {last - first, buffer.address + buffer.size - first, batch_size - batch.size});
            const BufferRange source{buffer.buffer, buffer.offset + (first - buffer.address),
                                     count};
            batch.downloads.push_back({first, batch.size, source});
            batch.size = Common::AlignUp(batch.size + count, u64{64});
            first += count;
        }
    });
    for (const auto& batch : batches) {
        auto allocation = staging.AllocateReadback(batch.size);
        for (const auto& download : batch.downloads) {
            Copy(download.source,
                 {allocation.GetBuffer(), allocation.GetOffset() + download.offset,
                  download.source.size},
                 download.source.size);
        }
        scheduler.Finish();
        const auto data = allocation.ReadbackData();
        for (const auto& download : batch.downloads) {
            WriteBacking(download.address, data.subspan(download.offset, download.source.size));
            const auto aliases = AliasedRanges(download.address, download.source.size);
            aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
                for (VAddr page = Common::AlignDown(first, CpuPageSize);
                     page < Common::AlignUp(last, CpuPageSize); page += CpuPageSize) {
                    if (page_manager.IsGuestAccessAllowed(page, CpuPageSize,
                                                          Core::MemoryPermission::Execute)) {
                        executable.Add(page, CpuPageSize);
                    }
                }
                gpu_modified_ranges.Subtract(first, last - first);
            });
        }
    }
    executable.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        sys_icache_invalidate(reinterpret_cast<void*>(first), last - first);
    });
    const auto aliases = AliasedRanges(begin, end - begin);
    aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        UpdateCpuReadWatches(first, last - first, false);
    });
}

void BufferCache::WriteBacking(VAddr address, std::span<const u8> data) {
    if (!memory.TryWriteSparseMemory(address, data)) {
        throw std::runtime_error("Metal guest write requires mapped backing");
    }
}

void BufferCache::WriteMemory(VAddr address, std::span<const u8> data) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    if (data.empty())
        return;
    ValidateRange(address, data.size());
    ReadMemory(address, data.size(), true);
    SmallVector<u8, 64> bytes(data.size());
    std::memcpy(bytes.data(), data.data(), data.size());
    WithMemoryWriteWatch(address, bytes.size(), [&] { WriteBacking(address, bytes); });
}

void BufferCache::WithMemoryWriteWatch(VAddr address, u64 size,
                                       const std::function<void()>& write) {
    const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
    ValidateRange(address, size);
    constexpr u64 CpuPageSize = 4096;
    const auto aliases = AliasedRanges(address, size);
    GuestRangeSet executable;
    aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        const VAddr end = Common::AlignUp(last, CpuPageSize);
        for (VAddr page = Common::AlignDown(first, CpuPageSize); page < end; page += CpuPageSize) {
            if (page_manager.IsGuestAccessAllowed(page, CpuPageSize,
                                                  Core::MemoryPermission::Execute)) {
                executable.Add(page, CpuPageSize);
            }
        }
    });
    executable.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        UpdateCpuReadWatches(first, last - first, true);
    });
    write();
    executable.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        sys_icache_invalidate(reinterpret_cast<void*>(first), last - first);
    });
    executable.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        UpdateCpuReadWatches(first, last - first, false);
    });
}

void BufferCache::ReadMemory(VAddr address, u64 size, bool is_write, bool assume_locks) {
    ValidateRange(address, size);
    const auto flush = [this, address, size, is_write] {
        const auto memory_operation = VideoCore::GuardMemoryOperation(std::terminate);
        constexpr u64 CpuPageSize = 4096;
        VAddr begin = Common::AlignDown(address, CpuPageSize);
        VAddr end = Common::AlignUp(address + size, CpuPageSize);
        if (!is_write) {
            const VAddr lower = begin > PageSize ? begin - PageSize : 0;
            const VAddr upper = std::min(end + PageSize, u64{1} << 40);
            VAddr read_begin = begin;
            VAddr read_end = end;
            gpu_modified_ranges.ForEachIntersectingRange(
                begin, end - begin, [&](VAddr first, VAddr last) {
                    read_begin = std::min(read_begin,
                                          Common::AlignDown(std::max(first, lower), CpuPageSize));
                    read_end =
                        std::max(read_end, Common::AlignUp(std::min(last, upper), CpuPageSize));
                });
            if (read_begin != begin || read_end != end) {
                const auto backing = memory.GetBackingRanges(read_begin, read_end - read_begin);
                if (std::ranges::all_of(backing, [](const auto& range) {
                        return range.physical_address.has_value();
                    })) {
                    begin = read_begin;
                    end = read_end;
                }
            }
        }
        if (texture_cache) {
            texture_cache->ResolveMemory(begin, end - begin);
        }
        DownloadMemory(begin, end - begin);
        if (is_write) {
            constexpr u64 CpuPageSize = 4096;
            const VAddr begin = Common::AlignDown(address, CpuPageSize);
            const VAddr end = Common::AlignUp(address + size, CpuPageSize);
            const auto aliases = AliasedRanges(begin, end - begin);
            aliases.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
                memory_tracker->MarkRegionAsCpuModified(first, last - first);
            });
        }
    };
    if (assume_locks || std::this_thread::get_id() == liverpool.GetGpuCommandProcessorThread()) {
        flush();
    } else {
        liverpool.SendCommand<true>(flush);
    }
}

void BufferCache::InvalidateMemory(VAddr address, u64 size, bool assume_locks) {
    ValidateRange(address, size);
    ReadMemory(address, size, true, assume_locks);
}

void BufferCache::UnmapMemory(VAddr address, u64 size) {
    ReadMemory(address, size, true);
    UpdateCpuReadWatches(address, size, false);
    const auto backing = physical_mappings.GetRanges(address, size);
    const VAddr begin = Common::AlignUp(address, PageSize);
    const VAddr end = Common::AlignDown(address + size, PageSize);
    if (begin < end) {
        auto first = buffers.lower_bound(begin);
        if (first != buffers.begin()) {
            const auto previous = std::prev(first);
            if (previous->first + previous->second.size > begin) {
                first = previous;
            }
        }
        auto last = first;
        std::map<VAddr, CachedBuffer> replacements;
        const auto preserve = [&](const CachedBuffer& source, VAddr start, u64 length) {
            CachedBuffer replacement{start, length, source.buffer,
                                     source.offset + (start - source.address)};
            replacements.emplace(start, std::move(replacement));
        };
        while (last != buffers.end() && last->first < end) {
            const auto& old = last->second;
            if (old.address < begin) {
                preserve(old, old.address, begin - old.address);
            }
            const VAddr old_end = old.address + old.size;
            if (old_end > end) {
                preserve(old, end, old_end - end);
            }
            ++last;
        }
        page_snapshot.reset();
        buffers.erase(first, last);
        buffers.merge(replacements);
    }
    gpu_modified_ranges.Subtract(address, size);
    initialized_ranges.Subtract(address, size);
    versions.Subtract(address, size);
    physical_mappings.Unmap(address, size);
    for (const auto& range : backing) {
        if (range.physical_address) {
            RetirePhysicalBuffers(*range.physical_address, range.size);
        }
    }
}

void BufferCache::ReleaseMemory() {
    if (std::this_thread::get_id() != liverpool.GetGpuCommandProcessorThread()) {
        throw std::runtime_error("Metal cache release requires the GPU command processor thread");
    }
    const auto initialized = initialized_ranges;
    initialized.ForEachInRange(0, u64{1} << 40, [&](VAddr first, VAddr last) {
        ReadMemory(first, last - first, true, true);
    });
    UpdateCpuReadWatches(0, u64{1} << 40, false);
    page_snapshot.reset();
    buffers.clear();
    physical_buffers.clear();
    gpu_modified_ranges.Clear();
    initialized_ranges.Clear();
    versions.Clear();
    physical_mappings.Clear();
}

} // namespace Metal
