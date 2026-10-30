// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include "common/small_vector.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_metal/metal_shader.h"
#include "video_core/renderer_metal/metal_tessellation_constants.h"

namespace Metal {

class BufferCache;

class ShaderCache {
public:
    struct SamplerLodState {
        float bias{};
        float minimum{};
        float maximum{};
        bool anisotropic{};

        bool operator==(const SamplerLodState&) const = default;
    };

    struct Variant {
        struct FetchProgram {
            u32 sgpr_base;
            std::vector<u32> code;
        };

        struct VertexFetchFormat {
            AmdGpu::DataFormat data;
            AmdGpu::NumberFormat number;

            bool operator==(const VertexFetchFormat&) const = default;
        };

        Shader::Info info;
        std::vector<FetchProgram> fetch_programs;
        SmallVector<VertexFetchFormat, 32> vertex_fetch_formats;
        Shader::StageSpecialization specialization;
        u32 shared_memory_size{};
        std::optional<GeometryKernelConfig> geometry;
        std::optional<GeometryCopyConfig> geometry_copy;
        std::optional<GeometryExportConfig> geometry_export;
        std::optional<TessellationKernelConfig> tessellation;
        bool flip_vertex_y{};
        bool sample_rate_shading{};
        Shader::Backend::Bindings binding_count{};
        std::array<SamplerLodState, Shader::NUM_SAMPLERS> sampler_lod{};
        std::bitset<Shader::NUM_BUFFERS> thread_id_buffers;
        CompiledShader shader;
        std::shared_future<id<MTLComputePipelineState>> compute_pipeline;
    };

    [[nodiscard]] static Shader::StageSpecialization SpecializeShader(
        Shader::Info& info, const Shader::RuntimeInfo& runtime, const Shader::Profile& profile,
        Shader::Backend::Bindings bindings, bool native_tessellation = false);

    ShaderCache(const Instance& instance, BufferCache& buffer_cache, u32 sparse_page_shift);

    [[nodiscard]] Shader::ShaderParams GetParams(
        VAddr address, std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data);
    [[nodiscard]] Variant& GetShader(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                                     const Shader::ShaderParams& params,
                                     const Shader::RuntimeInfo& runtime,
                                     Shader::Backend::Bindings& bindings,
                                     bool flip_vertex_y = false, bool sample_rate_shading = false,
                                     const GeometryKernelConfig* geometry = nullptr,
                                     const GeometryCopyConfig* geometry_copy = nullptr,
                                     const GeometryExportConfig* geometry_export = nullptr,
                                     const VertexFetchLimits* vertex_fetch_limits = nullptr,
                                     const TessellationKernelConfig* tessellation = nullptr);
    [[nodiscard]] Shader::TessellationDataConstantBuffer GetTessellationConstants(
        const Shader::ShaderParams& hull, const Shader::RuntimeInfo& runtime);
    [[nodiscard]] id<MTLComputePipelineState> GetComputePipeline(Variant& variant);
    [[nodiscard]] const Shader::Profile& GetProfile() const {
        return shaders.GetProfile();
    }

private:
    struct Key {
        u64 hash;
        u64 code_hash;
        Shader::HwStage hw_stage;
        Shader::SwStage sw_stage;

        bool operator==(const Key&) const = default;
    };

    struct KeyHash {
        size_t operator()(const Key& key) const noexcept;
    };

    using Variants = std::vector<std::unique_ptr<Variant>>;

    struct Program {
        struct TessellationAnalysis {
            Shader::RuntimeInfo runtime;
            TessellationConstantBinding binding;
        };

        u64 code_hash{};
        std::vector<u32> code;
        Variants variants;
        std::vector<TessellationAnalysis> tessellation;
        Variant* last_used{};
    };

    Program& GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                        const Shader::ShaderParams& params);

    const Instance& instance;
    BufferCache& buffer_cache;
    ShaderCompiler shaders;
    PipelineCompiler pipelines;
    std::unordered_map<Key, std::vector<Program>, KeyHash> programs;
};

} // namespace Metal
