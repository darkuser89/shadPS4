// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <span>
#include <vector>
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_graphics_pipeline.h"
#include "video_core/renderer_metal/metal_guest_resources.h"
#include "video_core/renderer_metal/metal_indirect.h"
#include "video_core/renderer_metal/metal_metadata_coverage.h"
#include "video_core/renderer_metal/metal_primitive.h"
#include "video_core/renderer_metal/metal_tessellation_dispatch.h"
#include "video_core/renderer_metal/metal_visibility.h"

namespace Metal {

class GraphicsRenderer {
public:
    GraphicsRenderer(const Instance& instance, Scheduler& scheduler, ShaderCache& shader_cache,
                     GraphicsPipelineCache& pipelines, GuestResources& resources,
                     BufferCache& buffer_cache, TextureCache& texture_cache,
                     AmdGpu::Liverpool& liverpool);

    void Draw(bool indexed, u32 index_offset, bool neo_mode);
    void DrawIndirect(bool indexed, VAddr address, u32 offset, u32 stride, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset,
                      bool neo_mode);

    void CheckIndirectErrors();
    void DumpPixelCounter(u32 counter, const BufferRange& destination, u32 pairs, u32 stride);
    void ResetPixelCounter(u32 counter);

private:
    struct IndirectDraw {
        VAddr address;
        u32 stride;
        u32 max_count;
        VAddr count_address;
        u16 vertex_sgpr_offset;
        u16 instance_sgpr_offset;
    };
    struct GatheredBuffer {
        struct Source {
            __weak id<MTLBuffer> buffer;
            u64 offset;
            u64 size;
        };

        id<MTLBuffer> buffer;
        VAddr address{};
        u64 size{};
        u64 version{};
        std::vector<Source> sources;
        bool valid{};
    };
    void Draw(bool indexed, u32 index_offset, bool neo_mode, const IndirectDraw* indirect);
    void ResolveColor(bool neo_mode);
    [[nodiscard]] Shader::RuntimeInfo Runtime(Shader::HwStage stage) const;
    [[nodiscard]] GraphicsState State(u32 width, u32 height, bool& flip_vertex_y);
    [[nodiscard]] id<MTLDepthStencilState> DepthStencil(GraphicsState& state);
    [[nodiscard]] BufferRange GatherIndices(VAddr address, u64 size);
    [[nodiscard]] BufferRange GatherBuffer(VAddr address, u64 size, GatheredBuffer& cached);
    [[nodiscard]] std::vector<BufferRange> GatherVertexBuffers(
        std::span<const std::pair<VAddr, u32>> ranges);

    const Instance& instance;
    Scheduler& scheduler;
    ShaderCache& shader_cache;
    GraphicsPipelineCache& pipelines;
    GuestResources& resources;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    AmdGpu::Liverpool& liverpool;
    Core::MemoryManager& memory;
    Commands commands;
    PipelineCompiler compiler;
    IndirectDraws indirect_draws;
    PrimitiveAssembly primitive_assembly;
    GeometryAssembly geometry_assembly;
    TessellationDispatch tessellation;
    TessellationGeometry tessellation_geometry;
    VisibilityQueries visibility;
    MetadataCoverage metadata_coverage;
    GatheredBuffer gathered_indices;
    std::array<GatheredBuffer, GraphicsPipelineState::MaxVertexBuffers> gathered_vertices{};
    ShaderCache::Variant depth_fragment;
    ShaderCache::Variant coverage_fragment;
    std::map<std::array<u32, 15>, id<MTLDepthStencilState>> depth_states;
};

} // namespace Metal
