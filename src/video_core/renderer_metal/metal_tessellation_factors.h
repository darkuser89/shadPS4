// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include "video_core/amdgpu/regs_vertex.h"
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_tessellation_work.h"

namespace Metal {

class TessellationFactors {
public:
    struct Group {
        u64 source_offset;
        u32 patch_count;
        u32 instance;
        u32 first_patch;
    };
    struct Result {
        std::shared_ptr<StagingBufferPool::Allocation> storage;
        std::shared_ptr<StagingBufferPool::Allocation> errors;
        u32 patch_count;
        AmdGpu::TessellationType type;
    };

    TessellationFactors(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] Result Decode(AmdGpu::TessellationType type,
                                const std::shared_ptr<StagingBufferPool::Allocation>& source,
                                std::span<const Group> groups);
    void Validate(const Result& result) const;
    [[nodiscard]] static u32 Components(AmdGpu::TessellationType type);

private:
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
};

} // namespace Metal
