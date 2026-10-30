// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_image_info.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace Metal {

class MetadataCoverage {
public:
    MetadataCoverage(const Instance& instance, Scheduler& scheduler);
    void Prepare(const ImageInfo& info, RenderState& render);
    [[nodiscard]] std::vector<u8> ReadTiles(const ImageInfo& info, const RenderState& render);

private:
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
    id<MTLTexture> target;
    id<MTLTexture> resolved;
};

} // namespace Metal
