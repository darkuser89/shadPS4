// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <stdexcept>

#include "common/assert.h"
#include "video_core/renderer_metal/metal_binding.h"
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

Commands::Commands(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    if (scheduler.GetQueue().device != instance.GetDevice()) {
        UNREACHABLE_MSG("Metal commands and scheduler must use the same device");
    }
}

void Commands::BeginGraphics(const RenderState& state, const GraphicsState& graphics,
                             id<MTLRenderPipelineState> pipeline) {
    if (!pipeline || pipeline.device != instance.GetDevice()) {
        UNREACHABLE_MSG("Invalid Metal graphics pipeline");
    }
    render_encoder = scheduler.BeginRendering(state);
    scheduler.ClaimEncoderState(*this);
    compute_encoder = nil;
    if (encoding_generation != scheduler.EncodingGeneration()) {
        graphics_state.reset();
    }
    if (encoding_generation != scheduler.EncodingGeneration() || render_pipeline != pipeline) {
        [render_encoder setRenderPipelineState:pipeline];
    }
    render_pipeline = pipeline;
    compute_pipeline = nil;
    encoding_generation = scheduler.EncodingGeneration();
    ApplyGraphicsState(state, graphics);
    scheduler.RetainObject(pipeline);
    compute_resources.reset();
    vertex_resources.reset();
    fragment_resources.reset();
    object_resources.reset();
    mesh_resources.reset();
    mesh_pipeline = false;
}

id<MTL4RenderCommandEncoder> Commands::BindGraphics(const RenderState& state,
                                                    const GraphicsState& graphics,
                                                    id<MTLRenderPipelineState> pipeline,
                                                    ResourceBindings* vertex_bindings,
                                                    ResourceBindings* fragment_bindings) {
    BeginGraphics(state, graphics, pipeline);
    vertex_resources = vertex_bindings ? vertex_bindings->storage : nullptr;
    fragment_resources = fragment_bindings ? fragment_bindings->storage : nullptr;
    if (vertex_bindings) {
        vertex_bindings->BindRender(scheduler, render_encoder, MTLRenderStageVertex);
    } else {
        [render_encoder setArgumentTable:nil atStages:MTLRenderStageVertex];
    }
    if (fragment_bindings) {
        fragment_bindings->BindRender(scheduler, render_encoder, MTLRenderStageFragment);
    } else {
        [render_encoder setArgumentTable:nil atStages:MTLRenderStageFragment];
    }
    return render_encoder;
}

id<MTL4RenderCommandEncoder> Commands::BindMeshGraphics(const RenderState& state,
                                                        const GraphicsState& graphics,
                                                        id<MTLRenderPipelineState> pipeline,
                                                        ResourceBindings* object_bindings,
                                                        ResourceBindings* mesh_bindings,
                                                        ResourceBindings* fragment_bindings) {
    if (!pipeline.maxTotalThreadsPerMeshThreadgroup) {
        UNREACHABLE_MSG("Metal mesh graphics requires a mesh pipeline");
    }
    BeginGraphics(state, graphics, pipeline);
    mesh_pipeline = true;
    object_resources = object_bindings ? object_bindings->storage : nullptr;
    mesh_resources = mesh_bindings ? mesh_bindings->storage : nullptr;
    fragment_resources = fragment_bindings ? fragment_bindings->storage : nullptr;
    if (object_bindings) {
        object_bindings->BindRender(scheduler, render_encoder, MTLRenderStageObject);
    } else {
        [render_encoder setArgumentTable:nil atStages:MTLRenderStageObject];
    }
    if (mesh_bindings) {
        mesh_bindings->BindRender(scheduler, render_encoder, MTLRenderStageMesh);
    } else {
        [render_encoder setArgumentTable:nil atStages:MTLRenderStageMesh];
    }
    if (fragment_bindings) {
        fragment_bindings->BindRender(scheduler, render_encoder, MTLRenderStageFragment);
    } else {
        [render_encoder setArgumentTable:nil atStages:MTLRenderStageFragment];
    }
    return render_encoder;
}

void Commands::Draw(MTLPrimitiveType primitive, u32 vertex_count, u32 instance_count,
                    u32 first_vertex, u32 first_instance) {
    if (!vertex_count || !instance_count) {
        return;
    }
    [PrepareDraw() drawPrimitives:primitive
                      vertexStart:first_vertex
                      vertexCount:vertex_count
                    instanceCount:instance_count
                     baseInstance:first_instance];
}

void Commands::DrawIndexed(MTLPrimitiveType primitive, MTLIndexType index_type,
                           const BufferRange& indices, u32 index_count, u32 instance_count,
                           s32 base_vertex, u32 first_instance) {
    if (!index_count || !instance_count) {
        return;
    }
    const u64 element_size = index_type == MTLIndexTypeUInt16 ? 2 : 4;
    if (index_type != MTLIndexTypeUInt16 && index_type != MTLIndexTypeUInt32) {
        UNREACHABLE_MSG("Unsupported Metal index type");
    }
    ValidateRange(indices, static_cast<u64>(index_count) * element_size, element_size);
    scheduler.UseResource(indices.buffer);
    [PrepareDraw() drawIndexedPrimitives:primitive
                              indexCount:index_count
                               indexType:index_type
                             indexBuffer:indices.buffer.gpuAddress + indices.offset
                       indexBufferLength:indices.size
                           instanceCount:instance_count
                              baseVertex:base_vertex
                            baseInstance:first_instance];
}

void Commands::DrawIndirect(MTLPrimitiveType primitive, const BufferRange& arguments) {
    ValidateRange(arguments, sizeof(MTLDrawPrimitivesIndirectArguments), 4);
    scheduler.UseResource(arguments.buffer);
    [PrepareDraw() drawPrimitives:primitive
                   indirectBuffer:arguments.buffer.gpuAddress + arguments.offset];
}

void Commands::DrawIndexedIndirect(MTLPrimitiveType primitive, MTLIndexType index_type,
                                   const BufferRange& indices, const BufferRange& arguments) {
    const u64 element_size = index_type == MTLIndexTypeUInt16 ? 2 : 4;
    if (index_type != MTLIndexTypeUInt16 && index_type != MTLIndexTypeUInt32) {
        UNREACHABLE_MSG("Unsupported Metal index type");
    }
    ValidateRange(indices, element_size, element_size);
    ValidateRange(arguments, sizeof(MTLDrawIndexedPrimitivesIndirectArguments), 4);
    scheduler.UseResource(indices.buffer);
    scheduler.UseResource(arguments.buffer);
    [PrepareDraw() drawIndexedPrimitives:primitive
                               indexType:index_type
                             indexBuffer:indices.buffer.gpuAddress + indices.offset
                       indexBufferLength:indices.size
                          indirectBuffer:arguments.buffer.gpuAddress + arguments.offset];
}

void Commands::DrawMesh(MTLSize group_count, MTLSize mesh_group_size, MTLSize object_group_size) {
    if (!group_count.width || !group_count.height || !group_count.depth) {
        return;
    }
    ValidateMeshGroupSizes(mesh_group_size, object_group_size);
    [PrepareDraw(true) drawMeshThreadgroups:group_count
                threadsPerObjectThreadgroup:object_group_size
                  threadsPerMeshThreadgroup:mesh_group_size];
}

void Commands::DrawMeshIndirect(const BufferRange& arguments, MTLSize mesh_group_size,
                                MTLSize object_group_size) {
    ValidateRange(arguments, sizeof(MTLDispatchThreadgroupsIndirectArguments), 4);
    ValidateMeshGroupSizes(mesh_group_size, object_group_size);
    scheduler.UseResource(arguments.buffer);
    [PrepareDraw(true)
        drawMeshThreadgroupsWithIndirectBuffer:arguments.buffer.gpuAddress + arguments.offset
                   threadsPerObjectThreadgroup:object_group_size
                     threadsPerMeshThreadgroup:mesh_group_size];
}

void Commands::ExecuteIndirect(id<MTLIndirectCommandBuffer> commands, const BufferRange& range) {
    if (!commands || commands.device != instance.GetDevice()) {
        UNREACHABLE_MSG("Invalid Metal indirect command buffer");
    }
    ValidateRange(range, sizeof(MTLIndirectCommandBufferExecutionRange), alignof(u32));
    scheduler.UseResource(commands);
    scheduler.UseResource(range.buffer);
    [PrepareDraw() executeCommandsInBuffer:commands
                            indirectBuffer:range.buffer.gpuAddress + range.offset];
}

void Commands::BindCompute(id<MTLComputePipelineState> pipeline, ResourceBindings& bindings) {
    if (!pipeline || pipeline.device != instance.GetDevice()) {
        UNREACHABLE_MSG("Invalid Metal compute pipeline");
    }
    compute_encoder = scheduler.BeginCompute();
    scheduler.ClaimEncoderState(*this);
    render_encoder = nil;
    if (encoding_generation != scheduler.EncodingGeneration() || compute_pipeline != pipeline) {
        [compute_encoder setComputePipelineState:pipeline];
    }
    compute_pipeline = pipeline;
    render_pipeline = nil;
    encoding_generation = scheduler.EncodingGeneration();
    scheduler.RetainObject(pipeline);
    vertex_resources.reset();
    fragment_resources.reset();
    object_resources.reset();
    mesh_resources.reset();
    mesh_pipeline = false;
    compute_resources = bindings.storage;
    bindings.BindCompute(scheduler);
}

void Commands::Dispatch(MTLSize group_count, MTLSize group_size) {
    if (!group_count.width || !group_count.height || !group_count.depth) {
        return;
    }
    ValidateGroupSize(group_size);
    [PrepareDispatch() dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
}

void Commands::DispatchIndirect(const BufferRange& arguments, MTLSize group_size) {
    ValidateRange(arguments, sizeof(MTLDispatchThreadgroupsIndirectArguments), 4);
    ValidateGroupSize(group_size);
    scheduler.UseResource(arguments.buffer);
    [PrepareDispatch()
        dispatchThreadgroupsWithIndirectBuffer:arguments.buffer.gpuAddress + arguments.offset
                         threadsPerThreadgroup:group_size];
}

void Commands::CopyBuffer(const BufferRange& source, const BufferRange& destination, u64 size) {
    if (!size) {
        return;
    }
    ValidateRange(source, size, 1);
    ValidateRange(destination, size, 1);
    if (source.buffer == destination.buffer) {
        if (source.offset == destination.offset) {
            return;
        }
        if (source.offset < destination.offset + size &&
            destination.offset < source.offset + size) {
            UNREACHABLE_MSG("Overlapping Metal buffer copies require an intermediate allocation");
        }
    }
    scheduler.UseResource(source.buffer);
    scheduler.UseResource(destination.buffer);
    [scheduler.BeginCompute() copyFromBuffer:source.buffer
                                sourceOffset:source.offset
                                    toBuffer:destination.buffer
                           destinationOffset:destination.offset
                                        size:size];
}

void Commands::FillBuffer(const BufferRange& destination, u8 value) {
    if (!destination.size) {
        return;
    }
    ValidateRange(destination, 1, 1);
    scheduler.UseResource(destination.buffer);
    [scheduler.BeginCompute() fillBuffer:destination.buffer
                                   range:NSMakeRange(destination.offset, destination.size)
                                   value:value];
}

id<MTL4RenderCommandEncoder> Commands::RenderEncoder() const {
    if (!render_encoder || encoding_generation != scheduler.EncodingGeneration()) {
        UNREACHABLE_MSG("Metal graphics state must be bound before drawing");
    }
    return render_encoder;
}

id<MTL4ComputeCommandEncoder> Commands::ComputeEncoder() const {
    if (!compute_encoder || encoding_generation != scheduler.EncodingGeneration()) {
        UNREACHABLE_MSG("Metal compute state must be bound before dispatching");
    }
    return compute_encoder;
}

id<MTL4RenderCommandEncoder> Commands::PrepareDraw(bool mesh) {
    const auto encoder = RenderEncoder();
    if (mesh != mesh_pipeline) {
        UNREACHABLE_MSG("Metal draw command does not match the bound graphics pipeline");
    }
    if (mesh) {
        if (object_resources) {
            ResourceBindings::RegisterResources(scheduler, object_resources);
        }
        if (mesh_resources) {
            ResourceBindings::RegisterResources(scheduler, mesh_resources);
        }
    } else if (vertex_resources) {
        ResourceBindings::RegisterResources(scheduler, vertex_resources);
    }
    if (fragment_resources) {
        ResourceBindings::RegisterResources(scheduler, fragment_resources);
    }
    return encoder;
}

id<MTL4ComputeCommandEncoder> Commands::PrepareDispatch() {
    const auto encoder = ComputeEncoder();
    if (compute_resources) {
        ResourceBindings::RegisterResources(scheduler, compute_resources);
    }
    return encoder;
}

void Commands::ValidateRange(const BufferRange& range, u64 minimum_size, u64 alignment) const {
    if (!range.buffer || range.buffer.device != instance.GetDevice() ||
        range.offset > range.buffer.length || range.offset % alignment ||
        range.size < minimum_size || range.size > range.buffer.length - range.offset) {
        UNREACHABLE_MSG("Invalid Metal command buffer range: offset {}, size {}", range.offset,
                        range.size);
    }
}

void Commands::ValidateGroupSize(MTLSize group_size) const {
    (void)ComputeEncoder();
    const auto maximum = instance.GetDevice().maxThreadsPerThreadgroup;
    const u64 limit = compute_pipeline.maxTotalThreadsPerThreadgroup;
    if (!group_size.width || !group_size.height || !group_size.depth ||
        group_size.width > maximum.width || group_size.height > maximum.height ||
        group_size.depth > maximum.depth || group_size.width > limit ||
        group_size.height > limit / group_size.width ||
        group_size.depth > limit / (group_size.width * group_size.height)) {
        UNREACHABLE_MSG("Invalid Metal threadgroup size: {}x{}x{}", group_size.width,
                        group_size.height, group_size.depth);
    }
}

void Commands::ValidateMeshGroupSizes(MTLSize mesh_group_size, MTLSize object_group_size) const {
    (void)RenderEncoder();
    if (!mesh_pipeline) {
        UNREACHABLE_MSG("Metal mesh draw requires bound mesh graphics state");
    }
    const auto maximum = instance.GetDevice().maxThreadsPerThreadgroup;
    const auto validate = [&](MTLSize size, u64 limit, MTLSize required) {
        if (!size.width || !size.height || !size.depth || size.width > maximum.width ||
            size.height > maximum.height || size.depth > maximum.depth || size.width > limit ||
            size.height > limit / size.width || size.depth > limit / (size.width * size.height) ||
            ((required.width || required.height || required.depth) &&
             (size.width != required.width || size.height != required.height ||
              size.depth != required.depth))) {
            UNREACHABLE_MSG("Invalid Metal mesh/object threadgroup size: {}x{}x{}", size.width,
                            size.height, size.depth);
        }
    };
    validate(mesh_group_size, render_pipeline.maxTotalThreadsPerMeshThreadgroup,
             render_pipeline.requiredThreadsPerMeshThreadgroup);
    if (const auto limit = render_pipeline.maxTotalThreadsPerObjectThreadgroup; limit) {
        validate(object_group_size, limit, render_pipeline.requiredThreadsPerObjectThreadgroup);
    }
}

void Commands::ApplyGraphicsState(const RenderState& state, const GraphicsState& graphics) {
    if (!graphics.viewport_count || graphics.viewport_count > graphics.viewports.size() ||
        (graphics.depth_stencil && graphics.depth_stencil.device != instance.GetDevice())) {
        UNREACHABLE_MSG("Invalid Metal graphics state");
    }
    const auto& bounds = graphics.depth_bounds;
    if (bounds.enabled) {
        if (!instance.SupportsDepthBounds()) {
            throw std::runtime_error("Metal depth bounds require native lowering on this GPU");
        }
        if (!state.depth_attachment.texture || !std::isfinite(bounds.minimum) ||
            !std::isfinite(bounds.maximum) || bounds.minimum < 0.0f || bounds.maximum > 1.0f ||
            bounds.minimum > bounds.maximum) {
            throw std::runtime_error("Metal guest depth bounds require native range lowering");
        }
        if (bounds.minimum == 0.0f && bounds.maximum == 1.0f &&
            state.depth_attachment.texture.pixelFormat != MTLPixelFormatDepth16Unorm) {
            throw std::runtime_error(
                "Metal full-range float depth bounds require native stored-depth lowering");
        }
    }
    if ((graphics.visibility_mode != MTLVisibilityResultModeDisabled &&
         graphics.visibility_mode != MTLVisibilityResultModeBoolean &&
         graphics.visibility_mode != MTLVisibilityResultModeCounting) ||
        graphics.visibility_offset % sizeof(u64)) {
        UNREACHABLE_MSG("Invalid Metal visibility mode or offset");
    }
    if (graphics.visibility_mode != MTLVisibilityResultModeDisabled) {
        ValidateRange({state.visibility_buffer, graphics.visibility_offset, sizeof(u64)},
                      sizeof(u64), sizeof(u64));
    }
    if (graphics_state && *graphics_state == graphics) {
        return;
    }
    const auto encoder = RenderEncoder();
    const bool first = !graphics_state;
    if (first || graphics_state->viewport_count != graphics.viewport_count ||
        graphics_state->viewports != graphics.viewports) {
        std::array<MTLViewport, 16> viewports{};
        for (u32 i = 0; i < graphics.viewport_count; ++i) {
            const auto& viewport = graphics.viewports[i];
            if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) ||
                !std::isfinite(viewport.width) || !std::isfinite(viewport.height) ||
                !std::isfinite(viewport.min_depth) || !std::isfinite(viewport.max_depth) ||
                viewport.width <= 0.0 || viewport.height <= 0.0 || viewport.min_depth < 0.0 ||
                viewport.min_depth > 1.0 || viewport.max_depth < 0.0 || viewport.max_depth > 1.0) {
                UNREACHABLE_MSG("Invalid Metal viewport at index {}", i);
            }
            viewports[i] = {viewport.x,      viewport.y,         viewport.width,
                            viewport.height, viewport.min_depth, viewport.max_depth};
        }
        [encoder setViewports:viewports.data() count:graphics.viewport_count];
    }
    if (first || graphics_state->viewport_count != graphics.viewport_count ||
        graphics_state->scissors != graphics.scissors) {
        std::array<MTLScissorRect, 16> scissors{};
        for (u32 i = 0; i < graphics.viewport_count; ++i) {
            const auto& scissor = graphics.scissors[i];
            if (scissor.x > state.width || scissor.y > state.height ||
                scissor.width > state.width - scissor.x ||
                scissor.height > state.height - scissor.y) {
                UNREACHABLE_MSG("Metal scissor exceeds the render area at index {}", i);
            }
            scissors[i] = {scissor.x, scissor.y, scissor.width, scissor.height};
        }
        [encoder setScissorRects:scissors.data() count:graphics.viewport_count];
    }
    if (first || graphics_state->depth_stencil != graphics.depth_stencil) {
        [encoder setDepthStencilState:graphics.depth_stencil];
        if (graphics.depth_stencil) {
            scheduler.RetainObject(graphics.depth_stencil);
        }
    }
    if (first || graphics_state->cull_mode != graphics.cull_mode) {
        [encoder setCullMode:graphics.cull_mode];
    }
    if (first || graphics_state->front_face != graphics.front_face) {
        [encoder setFrontFacingWinding:graphics.front_face];
    }
    if (first || graphics_state->depth_clip != graphics.depth_clip) {
        [encoder setDepthClipMode:graphics.depth_clip];
    }
    if (first || graphics_state->fill_mode != graphics.fill_mode) {
        [encoder setTriangleFillMode:graphics.fill_mode];
    }
    if (first || graphics_state->depth_bias != graphics.depth_bias) {
        [encoder setDepthBias:graphics.depth_bias.constant
                   slopeScale:graphics.depth_bias.slope
                        clamp:graphics.depth_bias.clamp];
    }
    if (instance.SupportsDepthBounds() &&
        (first || graphics_state->depth_bounds != graphics.depth_bounds)) {
        [encoder setDepthTestMinBound:bounds.enabled ? bounds.minimum : 0.0f
                             maxBound:bounds.enabled ? bounds.maximum : 1.0f];
    }
    if (first || graphics_state->blend_constant != graphics.blend_constant) {
        [encoder setBlendColorRed:graphics.blend_constant[0]
                            green:graphics.blend_constant[1]
                             blue:graphics.blend_constant[2]
                            alpha:graphics.blend_constant[3]];
    }
    if (first || graphics_state->stencil_front_reference != graphics.stencil_front_reference ||
        graphics_state->stencil_back_reference != graphics.stencil_back_reference) {
        [encoder setStencilFrontReferenceValue:graphics.stencil_front_reference
                            backReferenceValue:graphics.stencil_back_reference];
    }
    if (first || graphics_state->visibility_mode != graphics.visibility_mode ||
        graphics_state->visibility_offset != graphics.visibility_offset) {
        [encoder setVisibilityResultMode:graphics.visibility_mode
                                  offset:graphics.visibility_offset];
    }
    graphics_state = graphics;
}

} // namespace Metal
