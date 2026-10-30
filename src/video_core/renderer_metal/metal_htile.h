// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <span>
#include <vector>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_image_info.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"
namespace Metal {
class Htile {
public:
    struct Plan {
        std::vector<u8> clear_tiles;
        u32 clear_count{};
        bool metadata_changed{};
    };
    Htile(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] static u64 MetadataSize(const ImageInfo& info);
    [[nodiscard]] static Plan Decode(const ImageInfo& info, std::span<u8> metadata);
    void Materialize(const ImageInfo& info, const Plan& plan, const BufferRange& linear);

private:
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
};
} // namespace Metal
