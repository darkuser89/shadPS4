// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bitset>
#include <limits>
#include <stdexcept>
#include "common/hash.h"
#include "video_core/renderer_metal/metal_graphics_pipeline.h"
#include "video_core/renderer_metal/metal_instance.h"

namespace Metal {

GraphicsPipelineCache::GraphicsPipelineCache(const Instance& instance_)
    : instance{instance_}, compiler{instance} {}

size_t GraphicsPipelineCache::KeyHash::operator()(const Key& key) const noexcept {
    return (*this)(KeyView{key.vertex, key.mesh.get(), key.fragment, key.state});
}

size_t GraphicsPipelineCache::KeyHash::operator()(const KeyView& key) const noexcept {
    u64 hash{};
    const auto add = [&](u64 value) { hash = HashCombine(hash, value); };
    add(std::hash<const ShaderCache::Variant*>{}(key.vertex));
    add(std::hash<const CompiledShader*>{}(key.mesh));
    add(std::hash<const ShaderCache::Variant*>{}(key.fragment));
    for (const auto& color : key.state.colors) {
        add(color.format);
        add(color.blending);
        add(color.source_rgb);
        add(color.destination_rgb);
        add(color.rgb_operation);
        add(color.source_alpha);
        add(color.destination_alpha);
        add(color.alpha_operation);
        add(color.write_mask);
    }
    add(key.state.attributes.size());
    for (const auto& attribute : key.state.attributes) {
        add(attribute.location);
        add(attribute.format);
        add(attribute.buffer_index);
        add(attribute.offset);
    }
    add(key.state.buffers.size());
    for (const auto& buffer : key.state.buffers) {
        add(buffer.stride);
        add(buffer.step_function);
        add(buffer.step_rate);
    }
    add(key.state.topology);
    add(key.state.sample_count);
    add(key.state.alpha_to_coverage);
    add(key.state.alpha_to_one);
    add(key.state.rasterization);
    add(key.state.indirect_commands);
    return hash;
}

void GraphicsPipelineCache::ValidateState(const CompiledShader& vertex,
                                          const ShaderCache::Variant* fragment,
                                          const GraphicsPipelineState& state) const {
    const bool mesh = vertex.translated.stage == Shader::SwStage::Geometry;
    if ((!mesh && vertex.translated.stage != Shader::SwStage::Vertex) ||
        (fragment && (fragment->info.sw_stage != Shader::SwStage::Fragment ||
                      fragment->shader.translated.stage != Shader::SwStage::Fragment))) {
        throw std::runtime_error("Invalid Metal graphics shader stages");
    }
    if (state.rasterization && !fragment) {
        throw std::runtime_error("Metal 4 rasterization requires a fragment function");
    }
    if (!state.sample_count ||
        ![instance.GetDevice() supportsTextureSampleCount:state.sample_count]) {
        throw std::runtime_error("Unsupported Metal graphics sample count");
    }
    if (mesh) {
        if (!state.attributes.empty() || !state.buffers.empty()) {
            throw std::runtime_error("Metal mesh pipelines cannot use a vertex descriptor");
        }
        const auto& size = vertex.translated.mesh_workgroup_size;
        const auto maximum = instance.GetDevice().maxThreadsPerThreadgroup;
        constexpr u64 limit = std::numeric_limits<NSUInteger>::max();
        if (!size[0] || !size[1] || !size[2] || size[0] > maximum.width ||
            size[1] > maximum.height || size[2] > maximum.depth || size[0] > limit ||
            size[1] > limit / size[0] || size[2] > limit / (u64{size[0]} * size[1])) {
            throw std::runtime_error("Metal mesh workgroup exceeds device limits");
        }
    }
    if (state.buffers.size() > GraphicsPipelineState::MaxVertexBuffers ||
        state.attributes.size() > 31) {
        throw std::runtime_error("Metal vertex layout exceeds device binding limits");
    }
    std::bitset<31> locations;
    for (const auto& attribute : state.attributes) {
        if (attribute.location >= locations.size() || locations[attribute.location] ||
            attribute.format == MTLVertexFormatInvalid ||
            attribute.buffer_index < GraphicsPipelineState::FirstVertexBufferIndex ||
            attribute.buffer_index - GraphicsPipelineState::FirstVertexBufferIndex >=
                state.buffers.size()) {
            throw std::runtime_error("Invalid Metal graphics vertex attribute");
        }
        locations.set(attribute.location);
    }
    for (const auto& buffer : state.buffers) {
        if (buffer.step_function != MTLVertexStepFunctionConstant &&
            buffer.step_function != MTLVertexStepFunctionPerVertex &&
            buffer.step_function != MTLVertexStepFunctionPerInstance) {
            throw std::runtime_error("Metal vertex layout requires stage lowering");
        }
        if (!buffer.stride ||
            (buffer.step_function == MTLVertexStepFunctionConstant ? buffer.step_rate != 0
                                                                   : !buffer.step_rate)) {
            throw std::runtime_error("Invalid Metal graphics vertex layout");
        }
    }
}

id<MTLRenderPipelineState> GraphicsPipelineCache::GetPipeline(const ShaderCache::Variant& vertex,
                                                              const ShaderCache::Variant* fragment,
                                                              const GraphicsPipelineState& state) {
    if (vertex.info.sw_stage != vertex.shader.translated.stage) {
        throw std::runtime_error("Inconsistent Metal graphics shader stage metadata");
    }
    if (const auto found = pipelines.find(KeyView{&vertex, nullptr, fragment, state});
        found != pipelines.end()) {
        return found->second.get();
    }
    ValidateState(vertex.shader, fragment, state);
    auto pending = CompilePipeline(vertex.shader, fragment, state);
    return pipelines.emplace(Key{&vertex, nullptr, fragment, state}, std::move(pending))
        .first->second.get();
}

id<MTLRenderPipelineState> GraphicsPipelineCache::GetMeshPipeline(
    std::shared_ptr<const CompiledShader> mesh, const ShaderCache::Variant* fragment,
    const GraphicsPipelineState& state) {
    if (!mesh || mesh->translated.stage != Shader::SwStage::Geometry) {
        throw std::runtime_error("Metal geometry assembly requires a mesh shader");
    }
    if (const auto found = pipelines.find(KeyView{nullptr, mesh.get(), fragment, state});
        found != pipelines.end()) {
        return found->second.get();
    }
    ValidateState(*mesh, fragment, state);
    auto pending = CompilePipeline(*mesh, fragment, state);
    return pipelines.emplace(Key{nullptr, std::move(mesh), fragment, state}, std::move(pending))
        .first->second.get();
}

std::shared_future<id<MTLRenderPipelineState>> GraphicsPipelineCache::CompilePipeline(
    const CompiledShader& vertex, const ShaderCache::Variant* fragment,
    const GraphicsPipelineState& state) {
    @autoreleasepool {
        const auto function_descriptor = [](const CompiledShader& shader) {
            MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
            function.library = shader.library.get();
            const auto& name = shader.translated.entry_point;
            function.name = [[NSString alloc] initWithBytes:name.data()
                                                     length:name.size()
                                                   encoding:NSUTF8StringEncoding];
            return function;
        };
        const auto configure = [&](auto* descriptor) {
            if (fragment) {
                descriptor.fragmentFunctionDescriptor = function_descriptor(fragment->shader);
            }
            descriptor.supportIndirectCommandBuffers =
                state.indirect_commands ? MTL4IndirectCommandBufferSupportStateEnabled
                                        : MTL4IndirectCommandBufferSupportStateDisabled;
            descriptor.rasterizationEnabled = state.rasterization;
            descriptor.rasterSampleCount = state.sample_count;
            descriptor.alphaToCoverageState = state.alpha_to_coverage
                                                  ? MTL4AlphaToCoverageStateEnabled
                                                  : MTL4AlphaToCoverageStateDisabled;
            descriptor.alphaToOneState =
                state.alpha_to_one ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
            for (u32 i = 0; i < state.colors.size(); ++i) {
                const auto& color = state.colors[i];
                auto attachment = descriptor.colorAttachments[i];
                attachment.pixelFormat = color.format;
                attachment.blendingState =
                    color.blending ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
                attachment.sourceRGBBlendFactor = color.source_rgb;
                attachment.destinationRGBBlendFactor = color.destination_rgb;
                attachment.rgbBlendOperation = color.rgb_operation;
                attachment.sourceAlphaBlendFactor = color.source_alpha;
                attachment.destinationAlphaBlendFactor = color.destination_alpha;
                attachment.alphaBlendOperation = color.alpha_operation;
                attachment.writeMask = color.write_mask;
            }
        };
        MTL4PipelineDescriptor* descriptor;
        if (vertex.translated.stage == Shader::SwStage::Geometry) {
            auto mesh = [MTL4MeshRenderPipelineDescriptor new];
            mesh.meshFunctionDescriptor = function_descriptor(vertex);
            const auto& size = vertex.translated.mesh_workgroup_size;
            mesh.requiredThreadsPerMeshThreadgroup = MTLSizeMake(size[0], size[1], size[2]);
            mesh.maxTotalThreadsPerMeshThreadgroup = u64{size[0]} * size[1] * size[2];
            configure(mesh);
            descriptor = mesh;
        } else {
            auto render = [MTL4RenderPipelineDescriptor new];
            render.vertexFunctionDescriptor = function_descriptor(vertex);
            render.inputPrimitiveTopology = state.topology;
            configure(render);
            descriptor = render;
            if (!state.attributes.empty()) {
                MTLVertexDescriptor* layout = [MTLVertexDescriptor new];
                for (const auto& attribute : state.attributes) {
                    auto destination = layout.attributes[attribute.location];
                    destination.format = attribute.format;
                    destination.bufferIndex = attribute.buffer_index;
                    destination.offset = attribute.offset;
                }
                for (u32 i = 0; i < state.buffers.size(); ++i) {
                    const auto& buffer = state.buffers[i];
                    auto destination =
                        layout.layouts[GraphicsPipelineState::FirstVertexBufferIndex + i];
                    destination.stride = buffer.stride;
                    destination.stepFunction = buffer.step_function;
                    destination.stepRate = buffer.step_rate;
                }
                render.vertexDescriptor = layout;
            }
        }
        return compiler.CompileRenderPipelineAsync(descriptor);
    }
}

} // namespace Metal
