// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

class Instance;
class Scheduler;

class StagingBufferPool {
    enum class Usage { Upload, Readback, Private };
    struct Chunk;

public:
    class Allocation {
    public:
        Allocation(Allocation&&) noexcept = default;
        Allocation& operator=(Allocation&&) noexcept = default;
        Allocation(const Allocation&) = delete;
        Allocation& operator=(const Allocation&) = delete;

        [[nodiscard]] id<MTLBuffer> GetBuffer() const;
        [[nodiscard]] u64 GetOffset() const noexcept {
            return offset;
        }
        [[nodiscard]] u64 SizeBytes() const noexcept {
            return size;
        }
        [[nodiscard]] bool IsReady() const;
        [[nodiscard]] bool IsCurrentTick() const;
        [[nodiscard]] std::span<u8> UploadData() const;
        [[nodiscard]] std::span<const u8> ReadbackData() const;

    private:
        friend class StagingBufferPool;

        Allocation(std::shared_ptr<Chunk> chunk, Scheduler& scheduler, u64 offset, u64 size);

        std::shared_ptr<Chunk> chunk;
        Scheduler* scheduler;
        u64 offset;
        u64 size;
        u64 allocation_tick;
    };

    StagingBufferPool(const Instance& instance, Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    [[nodiscard]] Allocation Allocate(u64 size, u64 alignment = 16);
    [[nodiscard]] Allocation AllocateReadback(u64 size, u64 alignment = 16);
    [[nodiscard]] Allocation AllocatePrivate(u64 size, u64 alignment = 16);

private:
    friend class Scheduler;

    struct Chunk {
        id<MTLBuffer> buffer;
        u64 position{};
        Usage usage{};
        bool dedicated{};
        u64 last_used_frame{};
    };

    [[nodiscard]] Allocation Allocate(u64 size, u64 alignment, Usage usage);
    void Trim(u64 current_frame);

    const Instance& instance;
    Scheduler& scheduler;
    std::vector<std::shared_ptr<Chunk>> chunks;
    u64 last_trim_frame{};
};

} // namespace Metal
