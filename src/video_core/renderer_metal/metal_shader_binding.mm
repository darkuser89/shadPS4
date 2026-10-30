// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_shader_binding.h"

namespace Metal {

ShaderBindings::ShaderBindings(const Instance& instance_, const CompiledShader& shader)
    : instance{instance_}, uses_push_data{shader.translated.uses_push_data},
      stage{shader.translated.stage} {
    for (const auto& layout : shader.translated.resources) {
        if (layout.used) {
            bindings.emplace(layout.binding, Binding{layout, std::vector<Slot>(layout.count)});
        }
    }
    if (bindings.empty()) {
        return;
    }
    @autoreleasepool {
        id<MTLLibrary> library = shader.library.get();
        ASSERT(library && library.device == instance.GetDevice());
        const auto& entry = shader.translated.entry_point;
        NSString* name = [[NSString alloc] initWithBytes:entry.data()
                                                  length:entry.size()
                                                encoding:NSUTF8StringEncoding];
        id<MTLFunction> function = [library newFunctionWithName:name];
        encoder =
            [function newArgumentEncoderWithBufferIndex:ShaderCompiler::DescriptorBufferIndex];
        if (!encoder || !encoder.encodedLength) {
            UNREACHABLE_MSG("Failed to reflect Metal guest shader argument-buffer layout");
        }
    }
}

ShaderBindings::Slot& ShaderBindings::GetSlot(u32 binding, ShaderResourceType type,
                                              u32 array_element) {
    const auto it = bindings.find(binding);
    if (it == bindings.end() || it->second.layout.type != type ||
        array_element >= it->second.slots.size()) {
        UNREACHABLE_MSG("Invalid Metal guest shader binding: {} element {}", binding,
                        array_element);
    }
    return it->second.slots[array_element];
}

void ShaderBindings::InvalidateArguments() {
    argument_buffer.reset();
    resources.clear();
    samplers.clear();
}

void ShaderBindings::SetBuffer(u32 binding, id<MTLBuffer> buffer, u64 offset, u32 array_element) {
    if ((buffer && (buffer.device != instance.GetDevice() || offset >= buffer.length)) ||
        (!buffer && offset)) {
        UNREACHABLE_MSG("Invalid Metal guest buffer offset: {}", offset);
    }
    auto& slot = GetSlot(binding, ShaderResourceType::Buffer, array_element);
    if (slot.buffer == buffer && slot.offset == offset) {
        return;
    }
    InvalidateArguments();
    slot.buffer = buffer;
    slot.offset = offset;
}

void ShaderBindings::SetTexture(u32 binding, id<MTLTexture> texture, u32 array_element) {
    if (texture && texture.device != instance.GetDevice()) {
        UNREACHABLE_MSG("Metal guest texture and shader must use the same device");
    }
    auto& slot = GetSlot(binding, ShaderResourceType::Texture, array_element);
    if (slot.texture != texture) {
        InvalidateArguments();
        slot.texture = texture;
    }
}

void ShaderBindings::SetSampler(u32 binding, id<MTLSamplerState> sampler, u32 array_element) {
    if (sampler && sampler.device != instance.GetDevice()) {
        UNREACHABLE_MSG("Metal guest sampler and shader must use the same device");
    }
    auto& slot = GetSlot(binding, ShaderResourceType::Sampler, array_element);
    if (slot.sampler != sampler) {
        InvalidateArguments();
        slot.sampler = sampler;
    }
}

std::unique_ptr<ResourceBindings> ShaderBindings::CreateSnapshot(
    StagingBufferPool& uploads, std::span<const u8> push_data,
    std::span<const BufferRange> vertex_buffers) {
    constexpr u32 first_vertex_buffer = ShaderCompiler::DescriptorBufferIndex + 1;
    if (vertex_buffers.size() > 31 - first_vertex_buffer ||
        (!vertex_buffers.empty() && stage != Shader::SwStage::Vertex)) {
        UNREACHABLE_MSG("Invalid Metal guest vertex buffer bindings");
    }
    auto snapshot = std::make_unique<ResourceBindings>(
        instance, first_vertex_buffer + vertex_buffers.size(), 0, 0);
    for (u32 i = 0; i < vertex_buffers.size(); ++i) {
        const auto& range = vertex_buffers[i];
        if (!range.buffer || !range.size || range.offset >= range.buffer.length ||
            range.size > range.buffer.length - range.offset) {
            UNREACHABLE_MSG("Invalid Metal guest vertex buffer range");
        }
        snapshot->SetBuffer(first_vertex_buffer + i, range.buffer, range.offset);
    }
    if (uses_push_data) {
        if (push_data.empty()) {
            UNREACHABLE_MSG("Metal guest shader requires push data");
        }
        auto allocation = uploads.Allocate(push_data.size(), 16);
        std::memcpy(allocation.UploadData().data(), push_data.data(), push_data.size());
        snapshot->SetBuffer(ShaderCompiler::PushDataBufferIndex, allocation.GetBuffer(),
                            allocation.GetOffset());
        snapshot->RetainUploadAllocation(std::move(allocation));
    }
    if (!encoder) {
        return snapshot;
    }
    auto allocation = argument_pool == &uploads ? argument_buffer.lock() : nullptr;
    if (allocation && !allocation->IsCurrentTick()) {
        allocation.reset();
    }
    if (!allocation) {
        argument_buffer.reset();
        allocation = std::make_shared<StagingBufferPool::Allocation>(
            uploads.Allocate(encoder.encodedLength, encoder.alignment));
        ASSERT(allocation->GetBuffer().device == instance.GetDevice());
        std::memset(allocation->UploadData().data(), 0, allocation->SizeBytes());
        [encoder setArgumentBuffer:allocation->GetBuffer() offset:allocation->GetOffset()];
        resources.clear();
        samplers.clear();
        for (const auto& [binding, data] : bindings) {
            for (size_t i = 0; i < data.slots.size(); ++i) {
                const auto& slot = data.slots[i];
                const u32 index = data.layout.index + i;
                switch (data.layout.type) {
                case ShaderResourceType::Buffer:
                    [encoder setBuffer:slot.buffer offset:slot.offset atIndex:index];
                    if (slot.buffer) {
                        resources.push_back(slot.buffer);
                    }
                    break;
                case ShaderResourceType::Texture:
                    [encoder setTexture:slot.texture atIndex:index];
                    if (slot.texture) {
                        resources.push_back(slot.texture);
                    }
                    break;
                case ShaderResourceType::Sampler:
                    [encoder setSamplerState:slot.sampler atIndex:index];
                    if (slot.sampler) {
                        samplers.push_back(slot.sampler);
                    }
                    break;
                }
            }
        }
        [encoder setArgumentBuffer:nil offset:0];
        argument_pool = &uploads;
        argument_buffer = allocation;
    }
    snapshot->SetBuffer(ShaderCompiler::DescriptorBufferIndex, allocation->GetBuffer(),
                        allocation->GetOffset());
    snapshot->SetIndirectResources(resources, samplers);
    snapshot->RetainUploadAllocation(std::move(allocation));
    return snapshot;
}

} // namespace Metal
