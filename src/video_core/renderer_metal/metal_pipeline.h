// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <future>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

class Instance;

class PipelineCompiler {
public:
    explicit PipelineCompiler(const Instance& instance);

    [[nodiscard]] id<MTLLibrary> CompileLibrary(std::string_view source, std::string_view name);
    [[nodiscard]] std::shared_future<id<MTLLibrary>> CompileLibraryAsync(std::string_view source,
                                                                         std::string_view name);
    [[nodiscard]] id<MTLComputePipelineState> CompileComputePipeline(
        MTL4ComputePipelineDescriptor* descriptor);
    [[nodiscard]] std::shared_future<id<MTLComputePipelineState>> CompileComputePipelineAsync(
        MTL4ComputePipelineDescriptor* descriptor);
    [[nodiscard]] id<MTLRenderPipelineState> CompileRenderPipeline(
        MTL4PipelineDescriptor* descriptor);
    [[nodiscard]] std::shared_future<id<MTLRenderPipelineState>> CompileRenderPipelineAsync(
        MTL4PipelineDescriptor* descriptor);

private:
    struct LibrarySourceHash {
        using is_transparent = void;

        size_t operator()(std::string_view source) const noexcept;
    };

    const Instance& instance;
    std::mutex library_mutex;
    std::unordered_map<std::string, std::shared_future<id<MTLLibrary>>, LibrarySourceHash,
                       std::equal_to<>>
        libraries;
};

} // namespace Metal
