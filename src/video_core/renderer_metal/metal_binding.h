// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "video_core/renderer_metal/metal_common.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace Metal {

class Instance;
class Scheduler;
class Commands;

class ResourceBindings {
public:
    ResourceBindings(const Instance& instance, u32 buffer_count, u32 texture_count,
                     u32 sampler_count);

    ResourceBindings(const ResourceBindings&) = delete;
    ResourceBindings& operator=(const ResourceBindings&) = delete;

    void SetBuffer(u32 index, id<MTLBuffer> buffer, u64 offset = 0);
    void SetTexture(u32 index, id<MTLTexture> texture);
    void SetSampler(u32 index, id<MTLSamplerState> sampler);
    void SetIndirectResources(std::span<const id<MTLResource>> resources,
                              std::span<const id<MTLSamplerState>> samplers = {});
    void AddIndirectResources(std::span<const id<MTLResource>> resources);
    void RetainUploadAllocation(StagingBufferPool::Allocation allocation);
    void RetainUploadAllocation(std::shared_ptr<StagingBufferPool::Allocation> allocation);

private:
    friend class Commands;

    void BindCompute(Scheduler& scheduler);
    void BindRender(Scheduler& scheduler, id<MTL4RenderCommandEncoder> encoder,
                    MTLRenderStages stages);

    struct BufferBinding {
        id<MTLBuffer> buffer;
        u64 offset{};
    };

    struct Storage {
        explicit Storage(const Instance& instance_) : instance{instance_} {}
        ~Storage();

        const Instance& instance;
        id<MTL4ArgumentTable> table;
        std::vector<BufferBinding> buffers;
        std::vector<id<MTLTexture>> textures;
        std::vector<id<MTLSamplerState>> samplers;
        std::vector<id<MTLResource>> indirect_resources;
        std::vector<id<MTLSamplerState>> indirect_samplers;
        std::vector<StagingBufferPool::Allocation> upload_allocations;
        std::vector<std::shared_ptr<StagingBufferPool::Allocation>> shared_upload_allocations;
        u64 revision{1};
        id<MTL4CommandQueue> registered_queue;
        u64 registered_tick{};
        u64 registered_revision{};
    };

    static void RegisterResources(Scheduler& scheduler, const std::shared_ptr<Storage>& storage);

    std::shared_ptr<Storage> storage;
};

} // namespace Metal
