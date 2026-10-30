// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_tessellation_domains.h"

namespace Metal {

class TessellationEvaluation {
public:
    struct Result {
        std::shared_ptr<StagingBufferPool::Allocation> registers;
        std::shared_ptr<StagingBufferPool::Allocation> arguments;
        std::shared_ptr<StagingBufferPool::Allocation> errors;
        std::vector<GeometryExportInput> address_envelope;
    };

    TessellationEvaluation(const Instance& instance, Scheduler& scheduler);

    [[nodiscard]] std::optional<Result> BuildInputs(
        const TessellationDomains::Result& domains, const TessellationWorkLayout& work,
        const TessellationExportRingLayout& ring, u32 relative_patch, u32 first_point,
        bool raster_output = false, const std::array<float, 4>& viewport = {},
        std::span<const std::array<u32, 16>> system_registers = {});
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
