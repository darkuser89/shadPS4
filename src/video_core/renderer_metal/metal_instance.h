// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>

#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

class PipelineArchive;

class Instance {
public:
    Instance();
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    [[nodiscard]] id<MTLDevice> GetDevice() const noexcept {
        return device;
    }

    [[nodiscard]] id<MTL4Compiler> GetCompiler() const noexcept {
        return compiler;
    }

    [[nodiscard]] PipelineArchive& GetPipelineArchive() const noexcept;

    [[nodiscard]] const std::string& GetDeviceName() const noexcept {
        return device_name;
    }

    [[nodiscard]] bool SupportsDepthBounds() const noexcept {
        return supports_depth_bounds;
    }

    [[nodiscard]] bool SupportsSamplerLodBias() const noexcept {
        return supports_sampler_lod_bias;
    }

    [[nodiscard]] id<MTLBuffer> CreateBuffer(u64 size, MTLResourceOptions options) const;
    [[nodiscard]] id<MTLTexture> CreateTexture(MTLTextureDescriptor* descriptor) const;
    [[nodiscard]] id<MTLSamplerState> CreateSampler(MTLSamplerDescriptor* descriptor) const;
    [[nodiscard]] id<MTLDepthStencilState> CreateDepthStencilState(
        MTLDepthStencilDescriptor* descriptor) const;

private:
    friend class ResourceBindings;

    struct ArgumentTables;

    [[nodiscard]] id<MTL4ArgumentTable> AcquireArgumentTable(u32 buffer_count, u32 texture_count,
                                                             u32 sampler_count) const;
    void ReleaseArgumentTable(id<MTL4ArgumentTable> table, u32 buffer_count, u32 texture_count,
                              u32 sampler_count) const;

    id<MTLDevice> device;
    id<MTL4Compiler> compiler;
    std::string device_name;
    bool supports_depth_bounds{};
    bool supports_sampler_lod_bias{};
    std::unique_ptr<PipelineArchive> pipeline_archive;
    std::unique_ptr<ArgumentTables> argument_tables;
};

} // namespace Metal
