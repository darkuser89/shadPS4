// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/presenter.h"
#include "video_core/renderer_metal/metal_shader_binding.h"

namespace Metal {

class FsrPass {
public:
    FsrPass(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] id<MTLTexture> Render(id<MTLTexture> source, u32 width, u32 height,
                                        const VideoCore::FsrSettings& settings,
                                        bool perceptual_input);

private:
    struct Kernel {
        id<MTLComputePipelineState> pipeline;
        std::unique_ptr<ShaderBindings> bindings;
    };

    [[nodiscard]] Kernel CreateKernel(std::span<const u32> spirv, std::string_view name);
    [[nodiscard]] id<MTLTexture> Output(id<MTLTexture> __strong& texture, u32 width, u32 height,
                                        MTLPixelFormat format);
    void Dispatch(Kernel& kernel, id<MTLTexture> source, id<MTLTexture> destination,
                  std::span<const u8> constants);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    Kernel easu;
    Kernel rcas;
    Kernel hdr_easu;
    Kernel hdr_rcas;
    id<MTLSamplerState> sampler;
    id<MTLTexture> intermediate;
    id<MTLTexture> output;
};

} // namespace Metal
