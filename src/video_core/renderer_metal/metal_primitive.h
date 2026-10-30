// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include "video_core/amdgpu/regs_vertex.h"
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"

namespace Metal {

class GuestPageTableSnapshot;

class PrimitiveAssembly {
public:
    struct Result {
        BufferRange indices;
        u32 count;
        MTLPrimitiveType primitive;
    };

    struct IndirectResult {
        BufferRange indices;
        BufferRange arguments;
        BufferRange count;
        MTLPrimitiveType primitive;
    };

    PrimitiveAssembly(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] Result Assemble(AmdGpu::PrimitiveType topology, u32 count,
                                  const BufferRange& indices, MTLIndexType index_type,
                                  bool last_vertex);

    [[nodiscard]] Result AssemblePaged(AmdGpu::PrimitiveType topology, u32 count,
                                       const GuestPageTableSnapshot& pages, VAddr index_address,
                                       u64 index_size, MTLIndexType index_type, bool last_vertex,
                                       const BufferRange& errors);

    [[nodiscard]] BufferRange FinalizeDrawArguments(u32 index_count, u32 instance_count,
                                                    s32 base_vertex, u32 first_instance,
                                                    const BufferRange& errors);

    [[nodiscard]] IndirectResult AssembleIndexedIndirect(
        AmdGpu::PrimitiveType topology, const BufferRange& indices, MTLIndexType index_type,
        const BufferRange& arguments, const BufferRange& count, u32 max_count, u32 stride,
        bool last_vertex, const BufferRange& errors,
        std::optional<u32> restart_index = std::nullopt);

    [[nodiscard]] IndirectResult AssembleIndexedPaged(
        AmdGpu::PrimitiveType topology, VAddr index_address, u64 index_size,
        MTLIndexType index_type, const GuestPageTableSnapshot& pages, VAddr argument_address,
        VAddr count_address, u32 max_count, u32 stride, bool last_vertex, const BufferRange& errors,
        std::optional<u32> restart_index = std::nullopt);

    [[nodiscard]] IndirectResult AssembleIndexedRestart(
        AmdGpu::PrimitiveType topology, u32 count, u32 instance_count, s32 base_vertex,
        u32 first_instance, const BufferRange& indices, MTLIndexType index_type, bool last_vertex,
        u32 restart_index, const BufferRange& errors);

    [[nodiscard]] IndirectResult AssembleIndexedRestartPaged(
        AmdGpu::PrimitiveType topology, u32 count, u32 instance_count, s32 base_vertex,
        u32 first_instance, const GuestPageTableSnapshot& pages, VAddr index_address,
        u64 index_size, MTLIndexType index_type, bool last_vertex, u32 restart_index,
        const BufferRange& errors);

private:
    friend class GraphicsRenderer;

    struct Slot {
        id<MTLBuffer> buffer;
        bool leased{};
    };
    [[nodiscard]] Result AssemblePrimitives(AmdGpu::PrimitiveType topology, u32 count,
                                            const BufferRange& indices, MTLIndexType index_type,
                                            bool last_vertex, const GuestPageTableSnapshot* pages,
                                            VAddr index_address, u64 index_size,
                                            const BufferRange& errors);
    [[nodiscard]] IndirectResult AssembleIndexedCommands(
        AmdGpu::PrimitiveType topology, const BufferRange& indices, MTLIndexType index_type,
        const BufferRange& arguments, const BufferRange& count, u32 max_count, u32 stride,
        bool last_vertex, const BufferRange& errors, std::optional<u32> restart_index,
        const GuestPageTableSnapshot* pages, VAddr argument_address, VAddr count_address,
        VAddr index_address, u64 index_size);
    [[nodiscard]] id<MTLBuffer> Obtain(u64 size, bool arguments = false);
    [[nodiscard]] id<MTLComputePipelineState> Pipeline(bool indirect, bool restart = false,
                                                       bool paged = false);
    [[nodiscard]] id<MTLComputePipelineState> DrawArgumentPipeline();
    void ReleaseAfterDraw(const BufferRange& indices, const BufferRange& arguments);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    std::array<id<MTLComputePipelineState>, 4> pipelines{};
    id<MTLComputePipelineState> draw_argument_pipeline;
    std::array<Slot, 2> slots{};
};

} // namespace Metal
