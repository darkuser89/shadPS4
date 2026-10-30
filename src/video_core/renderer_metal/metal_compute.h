// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_guest_resources.h"

namespace Metal {

class ComputeRenderer {
public:
    ComputeRenderer(const Instance& instance, Scheduler& scheduler, ShaderCache& shader_cache,
                    GuestResources& resources, BufferCache& buffer_cache,
                    AmdGpu::Liverpool& liverpool);

    void DispatchDirect(bool neo_mode);
    void DispatchIndirect(VAddr address, u32 offset, u32 size, bool neo_mode);

private:
    [[nodiscard]] ShaderCache::Variant& Shader();
    [[nodiscard]] std::unique_ptr<ResourceBindings> Bind(ShaderCache::Variant& variant,
                                                         GuestResources::Stage& stage);
    [[nodiscard]] BufferRange GatherIndirectArguments(VAddr address);
    [[nodiscard]] BufferRange AdjustIndirectArguments(const BufferRange& arguments,
                                                      const std::array<u32, 3>& start);

    const Instance& instance;
    Scheduler& scheduler;
    ShaderCache& shader_cache;
    GuestResources& resources;
    BufferCache& buffer_cache;
    AmdGpu::Liverpool& liverpool;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> arguments_pipeline;
    id<MTLBuffer> adjusted_arguments;
    id<MTLBuffer> indirect_arguments;
};

} // namespace Metal
