// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <deque>
#include <span>
#include <vector>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_image_info.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace Metal {

class Cmask {
public:
    enum class Operation {
        Expand,
        EliminateFastClear,
        DecompressFmask,
    };

    struct Plan {
        std::vector<u8> tile_modes;
        u32 materialize_count{};
        bool write_pixels{};
        bool write_fmask{};
    };

    Cmask(const Instance& instance, Scheduler& scheduler);
    static void Validate(const ImageInfo& info);
    [[nodiscard]] static Plan Decode(const ImageInfo& info, std::span<u8> metadata,
                                     Operation operation = Operation::Expand,
                                     std::span<const u8> tiles = {});
    void Materialize(const ImageInfo& info, const Plan& plan, const BufferRange& linear,
                     const BufferRange& fmask);
    void CheckErrors();

private:
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
    struct ErrorRecord {
        StagingBufferPool::Allocation allocation;
        u64 tick;
    };
    std::deque<ErrorRecord> errors;
};

} // namespace Metal
