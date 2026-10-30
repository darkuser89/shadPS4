// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <limits>
#include <utility>

#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace Metal {

namespace {
constexpr u64 UploadBlockSize = 16_MB;
constexpr u64 ReadbackMinimumSize = 64_KB;
constexpr u64 UploadIdleFrames = 300;
constexpr u64 DedicatedIdleFrames = 120;
constexpr u64 PrivateCacheBudget = 64_MB;

u64 RoundAllocationSize(u64 size) {
    const u64 step = std::max(std::bit_floor(size) / 4, ReadbackMinimumSize);
    const u64 remainder = size & (step - 1);
    const u64 padding = remainder ? step - remainder : 0;
    if (size > std::numeric_limits<u64>::max() - padding) {
        UNREACHABLE_MSG("Metal staging capacity overflow: {} bytes", size);
    }
    return size + padding;
}
} // namespace

StagingBufferPool::Allocation::Allocation(std::shared_ptr<Chunk> chunk_, Scheduler& scheduler_,
                                          u64 offset_, u64 size_)
    : chunk{std::move(chunk_)}, scheduler{&scheduler_}, offset{offset_}, size{size_},
      allocation_tick{scheduler_.CurrentTick()} {}

id<MTLBuffer> StagingBufferPool::Allocation::GetBuffer() const {
    ASSERT(chunk);
    return chunk->buffer;
}

bool StagingBufferPool::Allocation::IsReady() const {
    ASSERT(chunk);
    return scheduler->IsResourceFree(chunk->buffer);
}

bool StagingBufferPool::Allocation::IsCurrentTick() const {
    ASSERT(chunk);
    return scheduler->CurrentTick() == allocation_tick;
}

std::span<u8> StagingBufferPool::Allocation::UploadData() const {
    if (!chunk || chunk->usage != Usage::Upload ||
        (scheduler->CurrentTick() != allocation_tick && !IsReady())) {
        UNREACHABLE_MSG("Metal upload data requires a valid allocation without pending GPU use");
    }
    auto* mapped = static_cast<u8*>(chunk->buffer.contents);
    return {mapped + offset, static_cast<size_t>(size)};
}

std::span<const u8> StagingBufferPool::Allocation::ReadbackData() const {
    if (!chunk || chunk->usage != Usage::Readback || !IsReady()) {
        UNREACHABLE_MSG("Metal readback data requires a valid allocation and completed GPU use");
    }
    const auto* mapped = static_cast<const u8*>(chunk->buffer.contents);
    return {mapped + offset, static_cast<size_t>(size)};
}

StagingBufferPool::StagingBufferPool(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    if (scheduler.GetQueue().device != instance.GetDevice()) {
        UNREACHABLE_MSG("Metal staging pool and scheduler must use the same device");
    }
    scheduler.RegisterStagingPool(*this);
}

StagingBufferPool::~StagingBufferPool() {
    scheduler.UnregisterStagingPool(*this);
}

void StagingBufferPool::Trim(u64 current_frame) {
    if (current_frame == last_trim_frame) {
        return;
    }
    last_trim_frame = current_frame;
    std::erase_if(chunks, [&](const auto& chunk) {
        const u64 idle_frames = chunk->dedicated ? DedicatedIdleFrames : UploadIdleFrames;
        return current_frame >= chunk->last_used_frame &&
               current_frame - chunk->last_used_frame >= idle_frames && chunk.use_count() == 1 &&
               scheduler.IsResourceFree(chunk->buffer);
    });
}

StagingBufferPool::Allocation StagingBufferPool::Allocate(u64 size, u64 alignment) {
    return Allocate(size, alignment, Usage::Upload);
}

StagingBufferPool::Allocation StagingBufferPool::AllocateReadback(u64 size, u64 alignment) {
    return Allocate(size, alignment, Usage::Readback);
}

StagingBufferPool::Allocation StagingBufferPool::AllocatePrivate(u64 size, u64 alignment) {
    return Allocate(size, alignment, Usage::Private);
}

StagingBufferPool::Allocation StagingBufferPool::Allocate(u64 size, u64 alignment, Usage usage) {
    if (!size || !alignment || (alignment & (alignment - 1)) ||
        size > std::numeric_limits<u64>::max() - (alignment - 1)) {
        UNREACHABLE_MSG("Invalid Metal staging allocation: size {}, alignment {}", size, alignment);
    }
    const u64 max_buffer_size = instance.GetDevice().maxBufferLength;
    if (size > max_buffer_size) {
        UNREACHABLE_MSG("Invalid Metal staging buffer size: {}", size);
    }
    Trim(scheduler.current_frame);
    const bool dedicated = usage != Usage::Upload || size > UploadBlockSize;
    const u64 capacity =
        std::min(max_buffer_size, dedicated ? std::max(usage != Usage::Upload ? ReadbackMinimumSize
                                                                              : UploadBlockSize,
                                                       RoundAllocationSize(size))
                                            : UploadBlockSize);
    const u64 max_capacity =
        std::max(capacity, size + std::min(size / 4, std::numeric_limits<u64>::max() - size));
    const auto allocate = [&](const std::shared_ptr<Chunk>& chunk, u64 offset) {
        chunk->position = offset + size;
        chunk->last_used_frame = scheduler.current_frame;
        scheduler.UseResource(chunk->buffer);
        return Allocation{chunk, scheduler, offset, size};
    };
    const std::shared_ptr<Chunk>* best{};
    u64 best_offset{};
    u64 best_remaining = std::numeric_limits<u64>::max();
    for (auto& chunk : chunks) {
        if (chunk->usage != usage || chunk->dedicated != dedicated) {
            continue;
        }
        if (dedicated) {
            if (chunk.use_count() != 1 || size > chunk->buffer.length ||
                chunk->buffer.length > max_capacity || !scheduler.IsResourceFree(chunk->buffer)) {
                continue;
            }
            if (!best || chunk->buffer.length < (*best)->buffer.length) {
                best = &chunk;
            }
            continue;
        }
        if (chunk.use_count() == 1 && scheduler.IsResourceFree(chunk->buffer)) {
            chunk->position = 0;
        }
        if (chunk->position > std::numeric_limits<u64>::max() - (alignment - 1)) {
            continue;
        }
        const u64 offset = (chunk->position + alignment - 1) & ~(alignment - 1);
        if (offset <= chunk->buffer.length && size <= chunk->buffer.length - offset) {
            const u64 remaining = chunk->buffer.length - offset - size;
            if (!best || remaining < best_remaining) {
                best = &chunk;
                best_offset = offset;
                best_remaining = remaining;
            }
        }
    }
    if (best) {
        return allocate(*best, best_offset);
    }
    if (usage == Usage::Private) {
        u64 remaining = PrivateCacheBudget;
        std::erase_if(chunks, [&](const auto& chunk) {
            if (chunk->usage != Usage::Private || chunk.use_count() != 1 ||
                !scheduler.IsResourceFree(chunk->buffer)) {
                return false;
            }
            if (chunk->buffer.length > remaining) {
                return true;
            }
            remaining -= chunk->buffer.length;
            return false;
        });
    }
    @autoreleasepool {
        const auto options =
            usage == Usage::Private
                ? MTLResourceStorageModePrivate
                : MTLResourceStorageModeShared |
                      (usage == Usage::Readback ? MTLResourceCPUCacheModeDefaultCache
                                                : MTLResourceCPUCacheModeWriteCombined);
        id<MTLBuffer> buffer = instance.CreateBuffer(capacity, options);
        if (usage != Usage::Private && !buffer.contents) {
            UNREACHABLE_MSG("Failed to allocate Metal staging buffer: {} bytes", capacity);
        }
        buffer.label = usage == Usage::Private    ? @"shadPS4:MetalPrivateScratch"
                       : usage == Usage::Readback ? @"shadPS4:MetalReadback"
                                                  : @"shadPS4:MetalUpload";
        auto chunk = std::make_shared<Chunk>();
        chunk->buffer = buffer;
        chunk->usage = usage;
        chunk->dedicated = dedicated;
        chunks.push_back(std::move(chunk));
        return allocate(chunks.back(), 0);
    }
}

} // namespace Metal
