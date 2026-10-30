// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "video_core/renderer_metal/metal_binding.h"
#include "video_core/renderer_metal/metal_graphics_state.h"
#include "video_core/renderer_metal/metal_render_state.h"

namespace Metal {

class Instance;
class Scheduler;

struct BufferRange {
    id<MTLBuffer> buffer;
    u64 offset{};
    u64 size{};
};

class Commands {
public:
    Commands(const Instance& instance, Scheduler& scheduler);

    Commands(const Commands&) = delete;
    Commands& operator=(const Commands&) = delete;

    [[nodiscard]] id<MTL4RenderCommandEncoder> BindGraphics(const RenderState& state,
                                                            const GraphicsState& graphics,
                                                            id<MTLRenderPipelineState> pipeline,
                                                            ResourceBindings* vertex_bindings,
                                                            ResourceBindings* fragment_bindings);
    [[nodiscard]] id<MTL4RenderCommandEncoder> BindMeshGraphics(
        const RenderState& state, const GraphicsState& graphics,
        id<MTLRenderPipelineState> pipeline, ResourceBindings* object_bindings,
        ResourceBindings* mesh_bindings, ResourceBindings* fragment_bindings);
    void Draw(MTLPrimitiveType primitive, u32 vertex_count, u32 instance_count = 1,
              u32 first_vertex = 0, u32 first_instance = 0);
    void DrawIndexed(MTLPrimitiveType primitive, MTLIndexType index_type,
                     const BufferRange& indices, u32 index_count, u32 instance_count = 1,
                     s32 base_vertex = 0, u32 first_instance = 0);
    void DrawIndirect(MTLPrimitiveType primitive, const BufferRange& arguments);
    void DrawIndexedIndirect(MTLPrimitiveType primitive, MTLIndexType index_type,
                             const BufferRange& indices, const BufferRange& arguments);
    void DrawMesh(MTLSize group_count, MTLSize mesh_group_size,
                  MTLSize object_group_size = {1, 1, 1});
    void DrawMeshIndirect(const BufferRange& arguments, MTLSize mesh_group_size,
                          MTLSize object_group_size = {1, 1, 1});

    void ExecuteIndirect(id<MTLIndirectCommandBuffer> commands, const BufferRange& range);

    void BindCompute(id<MTLComputePipelineState> pipeline, ResourceBindings& bindings);
    void Dispatch(MTLSize group_count, MTLSize group_size);
    void DispatchIndirect(const BufferRange& arguments, MTLSize group_size);
    void CopyBuffer(const BufferRange& source, const BufferRange& destination, u64 size);
    void FillBuffer(const BufferRange& destination, u8 value);

private:
    [[nodiscard]] id<MTL4RenderCommandEncoder> RenderEncoder() const;
    [[nodiscard]] id<MTL4ComputeCommandEncoder> ComputeEncoder() const;
    void BeginGraphics(const RenderState& state, const GraphicsState& graphics,
                       id<MTLRenderPipelineState> pipeline);
    [[nodiscard]] id<MTL4RenderCommandEncoder> PrepareDraw(bool mesh = false);
    [[nodiscard]] id<MTL4ComputeCommandEncoder> PrepareDispatch();
    void ValidateRange(const BufferRange& range, u64 minimum_size, u64 alignment) const;
    void ValidateGroupSize(MTLSize group_size) const;
    void ValidateMeshGroupSizes(MTLSize mesh_group_size, MTLSize object_group_size) const;
    void ApplyGraphicsState(const RenderState& state, const GraphicsState& graphics);

    const Instance& instance;
    Scheduler& scheduler;
    id<MTL4RenderCommandEncoder> render_encoder;
    id<MTL4ComputeCommandEncoder> compute_encoder;
    id<MTLRenderPipelineState> render_pipeline;
    id<MTLComputePipelineState> compute_pipeline;
    std::shared_ptr<ResourceBindings::Storage> vertex_resources;
    std::shared_ptr<ResourceBindings::Storage> fragment_resources;
    std::shared_ptr<ResourceBindings::Storage> object_resources;
    std::shared_ptr<ResourceBindings::Storage> mesh_resources;
    std::shared_ptr<ResourceBindings::Storage> compute_resources;
    std::optional<GraphicsState> graphics_state;
    u64 encoding_generation{};
    bool mesh_pipeline{};
};

} // namespace Metal
