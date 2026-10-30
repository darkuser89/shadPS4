// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"

namespace Metal {

class Instance;
class Scheduler;

class VisibilityQueries {
public:
    VisibilityQueries(const Instance& instance, Scheduler& scheduler);

    void Prepare(RenderState& render, GraphicsState& graphics, u32 control);
    void Dump(u32 counter, const BufferRange& destination, u32 pairs, u32 stride);
    void Reset(u32 counter);

private:
    [[nodiscard]] id<MTLComputePipelineState> Pipeline();
    void Update(u32 counter, const BufferRange& destination, u32 pairs, u32 stride, bool reset);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> buffer;
    id<MTLBuffer> baselines;
};

} // namespace Metal
