// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_tessellation_factors.h"

namespace Metal {

struct TessellationDomainPoint {
    u32 u_fixed;
    u32 v_fixed;
};
static_assert(sizeof(TessellationDomainPoint) == 8);

struct TessellationDomainPatch {
    u32 point_count;
    u32 index_count;
    u32 instance;
    u32 patch;
};
static_assert(sizeof(TessellationDomainPatch) == 16);

struct TessellationDomainConfig {
    AmdGpu::TessellationConfig tessellation;
    float minimum_factor;
    float maximum_factor;
};

[[nodiscard]] TessellationDomainConfig BuildTessellationDomainConfig(const AmdGpu::Regs& regs);

class TessellationDomains {
public:
    static constexpr u32 CoordinateOne = 65536;
    static constexpr u32 PointsPerPatch = 4225;
    static constexpr u32 IndicesPerPatch = 24576;

    struct Result {
        std::shared_ptr<StagingBufferPool::Allocation> points;
        std::shared_ptr<StagingBufferPool::Allocation> indices;
        std::shared_ptr<StagingBufferPool::Allocation> patches;
        std::shared_ptr<StagingBufferPool::Allocation> errors;
        std::shared_ptr<StagingBufferPool::Allocation> factors;
        TessellationDomainConfig config;
        u32 first_patch;
        u32 patch_count;
    };

    TessellationDomains(const Instance& instance, Scheduler& scheduler);

    [[nodiscard]] u32 MaxBatchPatches() const;
    [[nodiscard]] Result Generate(const TessellationDomainConfig& config,
                                  const TessellationFactors::Result& factors, u32 first_patch,
                                  u32 patch_count);
    void Validate(const Result& result) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
};

} // namespace Metal
