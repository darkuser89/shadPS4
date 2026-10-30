// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <limits>
#include "video_core/renderer_metal/metal_geometry_work.h"
#include "video_core/renderer_metal/metal_tessellation_exports.h"

namespace Metal {

class TessellationGeometry {
public:
    struct Inputs {
        GeometryDrawParameters draw;
        GeometryRingLayout rings;
        GeometryBufferRange guest_ring;
        std::shared_ptr<StagingBufferPool::Allocation> exports;
        std::shared_ptr<StagingBufferPool::Allocation> registers;
        std::shared_ptr<StagingBufferPool::Allocation> arguments;
        std::shared_ptr<StagingBufferPool::Allocation> geometry_waves;
        std::vector<GeometryAddressInput> address_envelope;
    };

    TessellationGeometry(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] u32 MaxBatchPrimitives(const AmdGpu::Regs& regs,
                                         const TessellationExports::Batch& batch) const;
    void RunBatches(const AmdGpu::Regs& regs, const TessellationExports::Batch& batch,
                    const std::function<void(Inputs&&)>& consume,
                    u32 maximum_primitives = std::numeric_limits<u32>::max());
    [[nodiscard]] Inputs Prepare(const AmdGpu::Regs& regs, const TessellationExports::Batch& batch,
                                 u32 first_primitive, u32 primitive_count);

private:
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> export_pipeline;
    id<MTLComputePipelineState> input_pipeline;
};

} // namespace Metal
