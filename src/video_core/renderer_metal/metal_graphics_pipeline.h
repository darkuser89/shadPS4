// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include "video_core/renderer_metal/metal_shader_cache.h"

namespace Metal {

struct ColorPipelineState {
    MTLPixelFormat format{MTLPixelFormatInvalid};
    bool blending{};
    MTLBlendFactor source_rgb{MTLBlendFactorOne};
    MTLBlendFactor destination_rgb{MTLBlendFactorZero};
    MTLBlendOperation rgb_operation{MTLBlendOperationAdd};
    MTLBlendFactor source_alpha{MTLBlendFactorOne};
    MTLBlendFactor destination_alpha{MTLBlendFactorZero};
    MTLBlendOperation alpha_operation{MTLBlendOperationAdd};
    MTLColorWriteMask write_mask{MTLColorWriteMaskAll};

    bool operator==(const ColorPipelineState&) const = default;
};

struct VertexAttribute {
    u32 location{};
    MTLVertexFormat format{MTLVertexFormatInvalid};
    u32 buffer_index{};
    u32 offset{};

    bool operator==(const VertexAttribute&) const = default;
};

struct VertexBufferLayout {
    u32 stride{};
    MTLVertexStepFunction step_function{MTLVertexStepFunctionPerVertex};
    u32 step_rate{1};

    bool operator==(const VertexBufferLayout&) const = default;
};

struct GraphicsPipelineState {
    static constexpr u32 FirstVertexBufferIndex = ShaderCompiler::DescriptorBufferIndex + 1;
    static constexpr u32 MaxVertexBuffers = 31 - FirstVertexBufferIndex;

    std::array<ColorPipelineState, 8> colors{};
    std::vector<VertexAttribute> attributes;
    std::vector<VertexBufferLayout> buffers;
    MTLPrimitiveTopologyClass topology{MTLPrimitiveTopologyClassTriangle};
    u32 sample_count{1};
    bool alpha_to_coverage{};
    bool alpha_to_one{};
    bool rasterization{true};
    bool indirect_commands{};

    bool operator==(const GraphicsPipelineState&) const = default;
};

class GraphicsPipelineCache {
public:
    explicit GraphicsPipelineCache(const Instance& instance);

    [[nodiscard]] id<MTLRenderPipelineState> GetPipeline(const ShaderCache::Variant& vertex,
                                                         const ShaderCache::Variant* fragment,
                                                         const GraphicsPipelineState& state);
    [[nodiscard]] id<MTLRenderPipelineState> GetMeshPipeline(
        std::shared_ptr<const CompiledShader> mesh, const ShaderCache::Variant* fragment,
        const GraphicsPipelineState& state);

private:
    struct Key {
        const ShaderCache::Variant* vertex;
        std::shared_ptr<const CompiledShader> mesh;
        const ShaderCache::Variant* fragment;
        GraphicsPipelineState state;

        bool operator==(const Key&) const = default;
    };

    struct KeyView {
        const ShaderCache::Variant* vertex;
        const CompiledShader* mesh;
        const ShaderCache::Variant* fragment;
        const GraphicsPipelineState& state;
    };

    struct KeyHash {
        using is_transparent = void;

        size_t operator()(const Key& key) const noexcept;
        size_t operator()(const KeyView& key) const noexcept;
    };

    struct KeyEqual {
        using is_transparent = void;

        bool operator()(const Key& left, const Key& right) const noexcept {
            return left == right;
        }

        bool operator()(const Key& left, const KeyView& right) const noexcept {
            return left.vertex == right.vertex && left.mesh.get() == right.mesh &&
                   left.fragment == right.fragment && left.state == right.state;
        }

        bool operator()(const KeyView& left, const Key& right) const noexcept {
            return (*this)(right, left);
        }
    };

    void ValidateState(const CompiledShader& vertex, const ShaderCache::Variant* fragment,
                       const GraphicsPipelineState& state) const;
    [[nodiscard]] std::shared_future<id<MTLRenderPipelineState>> CompilePipeline(
        const CompiledShader& vertex, const ShaderCache::Variant* fragment,
        const GraphicsPipelineState& state);

    const Instance& instance;
    PipelineCompiler compiler;
    std::unordered_map<Key, std::shared_future<id<MTLRenderPipelineState>>, KeyHash, KeyEqual>
        pipelines;
};

} // namespace Metal
