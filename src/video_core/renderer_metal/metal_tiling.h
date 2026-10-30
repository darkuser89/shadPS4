// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <deque>
#include <unordered_map>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/texture_cache/tile.h"

namespace Metal {

struct TilingLayout {
    u32 num_bits{};
    u32 num_samples{1};
    u32 num_mips{1};
    u32 micro_tiled_mips{};
    u32 bank_swizzle{};
    u64 guest_size{};
    AmdGpu::TileMode tile_mode{AmdGpu::TileMode::Thin1DThin};
    bool block_compressed{};
    bool alt_tile{};
    std::array<VideoCore::MipInfo, VideoCore::MAX_MIPS> mips{};
};

class TileManager {
public:
    TileManager(const Instance& instance, Scheduler& scheduler);

    void Transform(const BufferRange& tiled, const BufferRange& linear, const TilingLayout& layout,
                   bool is_tiler);
    void CheckErrors();

private:
    struct Parameters {
        u32 bank_swizzle;
        u32 micro_tiled_mips;
        u32 num_mips;
        u32 element_count;
        std::array<VideoCore::MipInfo, VideoCore::MAX_MIPS> mips;
    };
    static_assert(sizeof(Parameters) == 208);

    using Key = std::array<u32, 14>;
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept;
    };

    [[nodiscard]] Key BuildKey(const TilingLayout& layout, bool is_tiler) const;
    [[nodiscard]] Parameters BuildParameters(const TilingLayout& layout, const Key& key) const;
    [[nodiscard]] id<MTLComputePipelineState> GetPipeline(const Key& key);
    void ValidateBuffer(const BufferRange& buffer, u64 size, u64 alignment) const;

    const Instance& instance;
    Scheduler& scheduler;
    PipelineCompiler compiler;
    Commands commands;
    StagingBufferPool staging;
    std::shared_future<id<MTLLibrary>> library;
    std::unordered_map<Key, std::shared_future<id<MTLComputePipelineState>>, KeyHash> pipelines;
    struct ErrorRecord {
        StagingBufferPool::Allocation allocation;
        u64 tick;
    };
    std::deque<ErrorRecord> errors;
};

} // namespace Metal
