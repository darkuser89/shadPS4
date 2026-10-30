// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"

namespace Metal {

class DepthConversion {
public:
    DepthConversion(const Instance& instance, Scheduler& scheduler);

    [[nodiscard]] id<MTLTexture> CreateTexture(id<MTLTexture> source) const;
    void Convert(id<MTLTexture> source, id<MTLTexture> destination);

private:
    [[nodiscard]] id<MTLRenderPipelineState> Pipeline(u32 samples);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    id<MTLLibrary> library;
    id<MTLDepthStencilState> depth_stencil;
    std::map<u32, id<MTLRenderPipelineState>> pipelines;
};

} // namespace Metal
