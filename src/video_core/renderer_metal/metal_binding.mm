// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/assert.h"
#include "video_core/renderer_metal/metal_binding.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

ResourceBindings::ResourceBindings(const Instance& instance, u32 buffer_count, u32 texture_count,
                                   u32 sampler_count)
    : storage{std::make_shared<Storage>(instance)} {
    if (buffer_count > 31 || texture_count > 128 || sampler_count > 16) {
        UNREACHABLE_MSG("Resource bindings exceed Metal argument table limits");
    }
    storage->buffers.resize(buffer_count);
    storage->textures.resize(texture_count);
    storage->samplers.resize(sampler_count);
    storage->table = instance.AcquireArgumentTable(buffer_count, texture_count, sampler_count);
}

ResourceBindings::Storage::~Storage() {
    if (!table) {
        return;
    }
    for (u32 i = 0; i < buffers.size(); ++i) {
        if (buffers[i].buffer) {
            [table setAddress:0 atIndex:i];
        }
    }
    for (u32 i = 0; i < textures.size(); ++i) {
        if (textures[i]) {
            [table setTexture:MTLResourceID {} atIndex:i];
        }
    }
    for (u32 i = 0; i < samplers.size(); ++i) {
        if (samplers[i]) {
            [table setSamplerState:MTLResourceID {} atIndex:i];
        }
    }
    instance.ReleaseArgumentTable(table, buffers.size(), textures.size(), samplers.size());
}

void ResourceBindings::SetBuffer(u32 index, id<MTLBuffer> buffer, u64 offset) {
    if (index >= storage->buffers.size() ||
        (buffer && (buffer.device != storage->table.device || offset >= buffer.length)) ||
        (!buffer && offset)) {
        UNREACHABLE_MSG("Invalid Metal buffer binding: index {}, offset {}", index, offset);
    }
    auto& binding = storage->buffers[index];
    if (binding.buffer == buffer && binding.offset == offset) {
        return;
    }
    binding = {buffer, offset};
    [storage->table setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
    ++storage->revision;
}

void ResourceBindings::SetTexture(u32 index, id<MTLTexture> texture) {
    if (index >= storage->textures.size() || (texture && texture.device != storage->table.device)) {
        UNREACHABLE_MSG("Invalid Metal texture binding index: {}", index);
    }
    if (storage->textures[index] == texture) {
        return;
    }
    storage->textures[index] = texture;
    [storage->table setTexture:texture ? texture.gpuResourceID : MTLResourceID {} atIndex:index];
    ++storage->revision;
}

void ResourceBindings::SetSampler(u32 index, id<MTLSamplerState> sampler) {
    if (index >= storage->samplers.size() || (sampler && sampler.device != storage->table.device)) {
        UNREACHABLE_MSG("Invalid Metal sampler binding index: {}", index);
    }
    if (storage->samplers[index] == sampler) {
        return;
    }
    storage->samplers[index] = sampler;
    [storage->table setSamplerState:sampler ? sampler.gpuResourceID : MTLResourceID {}
                            atIndex:index];
    ++storage->revision;
}

void ResourceBindings::SetIndirectResources(std::span<const id<MTLResource>> resources,
                                            std::span<const id<MTLSamplerState>> samplers) {
    for (auto resource : resources) {
        if (!resource || resource.device != storage->table.device) {
            UNREACHABLE_MSG("Invalid indirectly referenced Metal resource");
        }
    }
    for (auto sampler : samplers) {
        if (!sampler || sampler.device != storage->table.device) {
            UNREACHABLE_MSG("Invalid indirectly referenced Metal sampler");
        }
    }
    if (std::ranges::equal(resources, storage->indirect_resources) &&
        std::ranges::equal(samplers, storage->indirect_samplers)) {
        return;
    }
    storage->indirect_resources.assign(resources.begin(), resources.end());
    storage->indirect_samplers.assign(samplers.begin(), samplers.end());
    ++storage->revision;
}

void ResourceBindings::AddIndirectResources(std::span<const id<MTLResource>> resources) {
    for (auto resource : resources) {
        if (!resource || resource.device != storage->table.device) {
            UNREACHABLE_MSG("Invalid indirectly referenced Metal resource");
        }
    }
    for (auto resource : resources) {
        if (std::ranges::find(storage->indirect_resources, resource) ==
            storage->indirect_resources.end()) {
            storage->indirect_resources.push_back(resource);
            ++storage->revision;
        }
    }
}

void ResourceBindings::BindCompute(Scheduler& scheduler) {
    RegisterResources(scheduler, storage);
    [scheduler.BeginCompute() setArgumentTable:storage->table];
}

void ResourceBindings::RetainUploadAllocation(StagingBufferPool::Allocation allocation) {
    if (allocation.GetBuffer().device != storage->table.device) {
        UNREACHABLE_MSG("Metal upload allocation and argument table must use the same device");
    }
    storage->upload_allocations.emplace_back(std::move(allocation));
}

void ResourceBindings::RetainUploadAllocation(
    std::shared_ptr<StagingBufferPool::Allocation> allocation) {
    if (!allocation || allocation->GetBuffer().device != storage->table.device) {
        UNREACHABLE_MSG("Metal upload allocation and argument table must use the same device");
    }
    storage->shared_upload_allocations.emplace_back(std::move(allocation));
}

void ResourceBindings::BindRender(Scheduler& scheduler, id<MTL4RenderCommandEncoder> encoder,
                                  MTLRenderStages stages) {
    ASSERT(encoder);
    RegisterResources(scheduler, storage);
    [encoder setArgumentTable:storage->table atStages:stages];
}

void ResourceBindings::RegisterResources(Scheduler& scheduler,
                                         const std::shared_ptr<Storage>& storage_owner) {
    auto& storage = *storage_owner;
    const auto queue = scheduler.GetQueue();
    if (storage.table.device != queue.device) {
        UNREACHABLE_MSG("Metal resource bindings and scheduler must use the same device");
    }
    if (storage.registered_queue == queue && storage.registered_tick == scheduler.CurrentTick() &&
        storage.registered_revision == storage.revision) {
        return;
    }
    scheduler.RetainLifetime(storage_owner);
    scheduler.RetainObject(storage.table);
    for (const auto& binding : storage.buffers) {
        if (binding.buffer) {
            scheduler.UseResource(binding.buffer);
        }
    }
    for (auto texture : storage.textures) {
        if (texture) {
            scheduler.UseResource(texture);
        }
    }
    for (auto sampler : storage.samplers) {
        if (sampler) {
            scheduler.RetainObject(sampler);
        }
    }
    for (auto resource : storage.indirect_resources) {
        scheduler.UseResource(resource);
    }
    for (auto sampler : storage.indirect_samplers) {
        scheduler.RetainObject(sampler);
    }
    storage.registered_queue = queue;
    storage.registered_tick = scheduler.CurrentTick();
    storage.registered_revision = storage.revision;
}

} // namespace Metal
