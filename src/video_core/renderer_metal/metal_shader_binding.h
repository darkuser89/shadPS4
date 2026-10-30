// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <memory>
#include "video_core/renderer_metal/metal_binding.h"
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_shader.h"

namespace Metal {

class ShaderBindings {
public:
    ShaderBindings(const Instance& instance, const CompiledShader& shader);

    [[nodiscard]] bool UsesBinding(u32 binding) const {
        return bindings.contains(binding);
    }

    void SetBuffer(u32 binding, id<MTLBuffer> buffer, u64 offset = 0, u32 array_element = 0);
    void SetTexture(u32 binding, id<MTLTexture> texture, u32 array_element = 0);
    void SetSampler(u32 binding, id<MTLSamplerState> sampler, u32 array_element = 0);
    [[nodiscard]] std::unique_ptr<ResourceBindings> CreateSnapshot(
        StagingBufferPool& uploads, std::span<const u8> push_data,
        std::span<const BufferRange> vertex_buffers = {});

private:
    struct Slot {
        id<MTLBuffer> buffer;
        u64 offset{};
        id<MTLTexture> texture;
        id<MTLSamplerState> sampler;
    };

    struct Binding {
        ShaderResourceBinding layout;
        std::vector<Slot> slots;
    };

    [[nodiscard]] Slot& GetSlot(u32 binding, ShaderResourceType type, u32 array_element);
    void InvalidateArguments();

    const Instance& instance;
    std::map<u32, Binding> bindings;
    id<MTLArgumentEncoder> encoder;
    std::weak_ptr<StagingBufferPool::Allocation> argument_buffer;
    const StagingBufferPool* argument_pool{};
    std::vector<id<MTLResource>> resources;
    std::vector<id<MTLSamplerState>> samplers;
    bool uses_push_data;
    Shader::SwStage stage;
};

} // namespace Metal
