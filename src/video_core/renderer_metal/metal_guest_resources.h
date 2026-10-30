// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include "video_core/renderer_metal/metal_geometry_address.h"
#include "video_core/renderer_metal/metal_sampler.h"
#include "video_core/renderer_metal/metal_shader_binding.h"
#include "video_core/renderer_metal/metal_shader_cache.h"
#include "video_core/renderer_metal/metal_texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Metal {

class GuestResources {
public:
    struct GeometryBuffer {
        u32 resource_index;
        std::shared_ptr<StagingBufferPool::Allocation> storage;
    };

    class Stage {
    public:
        [[nodiscard]] bool HasWrites() const noexcept;

        [[nodiscard]] std::span<const GeometryBufferAddressMapping> GetGeometryAddressMappings()
            const noexcept {
            return geometry_address_mappings;
        }

    private:
        friend class GuestResources;

        struct Buffer {
            u32 binding;
            Shader::BufferType type;
            u32 resource_index;
            VAddr address{};
            u32 size{};
            bool written{};
        };
        struct Image {
            u32 binding;
            u32 count;
            ImageInfo info;
            ImageViewInfo view;
            Shader::MipStorageFallbackMode mip_mode;
            bool written;
            bool atomic;
            bool query_only;
            TextureCache::Image* cached{};
        };
        struct Sampler {
            u32 binding;
            AmdGpu::Sampler sharp;
            bool depth;
            bool native_bias;
        };

        const GuestResources* owner{};
        ShaderCache::Variant* variant{};
        ShaderBindings* bindings{};
        std::vector<Buffer> buffers;
        std::vector<Image> images;
        std::vector<Sampler> samplers;
        std::vector<std::shared_ptr<StagingBufferPool::Allocation>> uploads;
        std::shared_ptr<StagingBufferPool::Allocation> tessellation_errors;
        std::vector<u32> flattened_user_data;
        std::vector<GeometryAddressInput> geometry_inputs;
        std::vector<GeometryBufferByteInterval> scalar_reads;
        std::unordered_map<u32, BufferRange> geometry_buffers;
        std::array<u32, 3> workgroups;
        std::array<u32, 4> dispatch_base{};
        std::array<u32, Shader::NUM_BUFFERS> buffer_capacities{};
        std::array<u64, Shader::NUM_BUFFERS> buffer_addresses{};
        std::shared_ptr<const GuestPageTableSnapshot> page_snapshot;
        std::optional<GeometryBufferSnapshot> geometry_snapshot;
        std::vector<GeometryBufferAddressBinding> geometry_address_bindings;
        std::vector<GeometryBufferAddressMapping> geometry_address_mappings;
        Shader::PushData push{};
        bool acquired{};
        bool marked_written{};
        bool snapshotted{};
    };

    GuestResources(const Instance& instance, Scheduler& scheduler, BufferCache& buffer_cache,
                   TextureCache& texture_cache, AmdGpu::Liverpool& liverpool);

    [[nodiscard]] std::unique_ptr<Stage> Prepare(
        ShaderCache::Variant& variant, bool neo_mode,
        std::span<const GeometryBuffer> geometry_buffers = {}, u32 initial_active_work_items = 0,
        std::span<const GuestBufferExtent> guest_extents = {},
        std::span<const GeometryAddressInput> geometry_inputs = {});
    void Acquire(std::span<Stage* const> stages);
    void RefreshPageTables(std::span<Stage* const> stages);
    void ValidateTessellation(const Stage& stage) const;
    void ValidateTessellationRing(const Stage& stage, u32 resource_index, VAddr address, u64 size,
                                  std::span<const ImageInfo> attachments) const;
    void ReplayTessellationRing(Stage& stage, const GeometryBufferRange& origin,
                                std::span<const GeometryBufferByteInterval> ranges,
                                const std::shared_ptr<StagingBufferPool::Allocation>& source,
                                std::span<const ImageInfo> attachments = {});
    void ReplayGeometryRing(Stage& geometry, const GeometryBufferRange& origin,
                            const std::shared_ptr<StagingBufferPool::Allocation>& source,
                            std::span<Stage* const> companions = {},
                            std::span<const ImageInfo> attachments = {});
    void ValidateGeometryOutputRing(const Stage& stage, u32 resource_index, VAddr address, u64 size,
                                    std::span<const ImageInfo> attachments) const;
    void ValidateGeometryCopyRing(const Stage& geometry, const Stage& copy, u64 size,
                                  std::span<const ImageInfo> attachments) const;
    [[nodiscard]] std::vector<GeometryBufferLink> GeometryDependencies(const Stage& producer,
                                                                       const Stage& consumer) const;
    void ValidateAttachments(std::span<Stage* const> stages,
                             std::span<const ImageInfo> attachments) const;
    void MarkWrites(std::span<Stage* const> stages);
    [[nodiscard]] std::unique_ptr<ResourceBindings> Snapshot(
        Stage& stage, std::span<const BufferRange> vertex_buffers = {});

private:
    void ValidateRingBacking(const Stage& stage, u32 resource_index, VAddr address, u64 size,
                             std::span<const ImageInfo> attachments) const;
    void BindBuffer(Stage& stage, const Stage::Buffer& buffer);
    void BindImage(Stage& stage, Stage::Image& image, MTLTextureUsage usage);
    void ValidateAliases(std::span<Stage* const> stages) const;
    [[nodiscard]] BufferRange Upload(Stage& stage, u32 binding, std::span<const u8> data);
    [[nodiscard]] BufferRange SharedMemory(Stage& stage, u64 stride);

    const Instance& instance;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    AmdGpu::Liverpool& liverpool;
    Core::MemoryManager& memory;
    StagingBufferPool staging;
    Commands commands;
    SamplerCache samplers;
    std::unordered_map<const ShaderCache::Variant*, std::unique_ptr<ShaderBindings>> bindings;
    std::unordered_map<ShaderBindings*,
                       std::unordered_map<u32, std::weak_ptr<StagingBufferPool::Allocation>>>
        upload_cache;
};

} // namespace Metal
