// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <bitset>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_metal/metal_fmask.h"
#include "video_core/renderer_metal/metal_geometry_copy.h"
#include "video_core/renderer_metal/metal_geometry_export.h"
#include "video_core/renderer_metal/metal_geometry_mesh.h"
#include "video_core/renderer_metal/metal_geometry_resources.h"
#include "video_core/renderer_metal/metal_geometry_shader.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_tessellation_kernel.h"

namespace Metal {

enum class ShaderResourceType {
    Buffer,
    Texture,
    Sampler,
};

struct ShaderResourceBinding {
    ShaderResourceType type;
    u32 descriptor_set;
    u32 binding;
    u32 index;
    u32 count;
    bool used{};
};

struct TranslatedShader {
    Shader::SwStage stage{Shader::SwStage::Compute};
    std::string source;
    std::string entry_point;
    std::vector<ShaderResourceBinding> resources;
    u32 buffer_count{};
    u32 texture_count{};
    u32 sampler_count{};
    bool uses_push_data{};
    std::array<u32, 3> mesh_workgroup_size{};
    std::vector<GeometryMeshVarying> geometry_inputs;
    bool geometry_inputs_supported{true};
};

struct SamplerLodUsage {
    bool bias{};
    bool clamp{};
    bool anisotropic{};
    bool native_bias{};
};

struct ImageMinimumLodState {
    u32 minimum{};
    u32 base_level{};
    u32 levels{};

    static ImageMinimumLodState FromImage(const AmdGpu::Image& image,
                                          const Shader::ImageResource& resource) {
        if (!image.min_lod) {
            return {};
        }
        return {u32(image.min_lod), u32(image.base_level), image.NumViewLevels(resource.is_array)};
    }

    bool operator==(const ImageMinimumLodState&) const = default;
};

struct ImageAliasState {
    std::bitset<Shader::NUM_IMAGES> eligible;
    std::bitset<Shader::NUM_IMAGES> srgb_images;
    std::array<u32, Shader::NUM_IMAGES> targets{};

    static ImageAliasState FromInfo(const Shader::Info& info,
                                    std::bitset<Shader::NUM_IMAGES> eligible) {
        if (info.images.size() > Shader::NUM_IMAGES) {
            throw std::runtime_error("Metal image aliases exceed native capacity");
        }
        ImageAliasState state{.eligible = eligible};
        for (u32 index = 0; index < state.targets.size(); ++index) {
            state.targets[index] = index;
            if (index >= info.images.size() || !eligible.test(index)) {
                continue;
            }
            const auto& resource = info.images[index];
            const auto sharp = resource.GetSharp(info);
            if (!sharp.Address() || resource.is_depth || resource.is_atomic ||
                resource.mip_fallback_mode != Shader::MipStorageFallbackMode::None ||
                sharp.GetViewType(resource.is_array) == AmdGpu::ImageType::Color2DMsaa ||
                sharp.GetViewType(resource.is_array) == AmdGpu::ImageType::Color2DMsaaArray) {
                continue;
            }
            state.srgb_images[index] = sharp.GetNumberFmt() == AmdGpu::NumberFormat::Srgb;
            for (u32 previous = 0; previous < index; ++previous) {
                const auto& candidate = info.images[previous];
                if (eligible.test(previous) && candidate.is_depth == resource.is_depth &&
                    candidate.is_atomic == resource.is_atomic &&
                    candidate.is_array == resource.is_array &&
                    candidate.is_r128 == resource.is_r128 &&
                    candidate.post_op == resource.post_op &&
                    candidate.mip_fallback_mode == resource.mip_fallback_mode &&
                    std::bit_cast<std::array<u32, 8>>(candidate.GetSharp(info)) ==
                        std::bit_cast<std::array<u32, 8>>(sharp)) {
                    state.targets[index] = state.targets[previous];
                    break;
                }
            }
        }
        return state;
    }

    bool operator==(const ImageAliasState&) const = default;
};

struct ShaderMetadata {
    std::array<SamplerLodUsage, Shader::NUM_SAMPLERS> sampler_lod{};
    std::bitset<Shader::NUM_BUFFERS> unbound_buffers;
    std::bitset<Shader::NUM_BUFFERS> guarded_buffers;
    std::bitset<Shader::NUM_IMAGES> single_mip_gathers;
    std::bitset<Shader::NUM_IMAGES> base_mip_gathers;
    std::bitset<Shader::NUM_IMAGES> lowered_minimum_lod_images;
    std::bitset<Shader::NUM_IMAGES> minimum_lod_dependencies;
    std::bitset<Shader::NUM_IMAGES> descriptor_query_images;
    std::bitset<Shader::NUM_IMAGES> query_only_images;
    std::bitset<Shader::NUM_IMAGES> fmask_dependencies;
    std::array<FmaskImageState, Shader::NUM_IMAGES> fmask_images{};
    std::array<ImageMinimumLodState, Shader::NUM_IMAGES> image_minimum_lod{};
    std::bitset<Shader::NUM_SAMPLERS> post_clamp_samplers;
    std::optional<GeometryKernelBindings> geometry;
    std::optional<GeometryCopyResult> geometry_copy;
    std::optional<GeometryExportBindings> geometry_export;
    GeometryBufferAccesses geometry_buffers;
    std::shared_ptr<const GeometryBufferAddressProgram> guest_buffer_bindings;
    std::bitset<Shader::NUM_BUFFERS> paged_buffers;
    ImageAliasState image_aliases;
    std::optional<TessellationKernelBindings> tessellation;
};

struct PreparedShader : ShaderMetadata {
    std::vector<u32> spirv;
};

struct CompiledShader : ShaderMetadata {
    TranslatedShader translated;
    std::shared_future<id<MTLLibrary>> library;
};

struct VertexFetchLimits {
    u32 vertex_count{};
    u32 instance_count{};
    u32 vertex_offset{};
    u32 step_rate_0{};
    u32 step_rate_1{};
    bool indexed{};
    bool indirect{};
};

class ShaderCompiler {
public:
    static constexpr u32 PushDataBufferIndex = 0;
    static constexpr u32 DescriptorBufferIndex = 1;
    static constexpr u32 BufferAddressWordOffset = Shader::NUM_BUFFERS;

    ShaderCompiler(const Instance& instance, u32 sparse_page_shift);

    [[nodiscard]] const Shader::Profile& GetProfile() const {
        return profile;
    }

    [[nodiscard]] static TranslatedShader TranslateSpirv(std::span<const u32> spirv,
                                                         bool flip_vertex_y = false,
                                                         bool sample_rate_shading = false);
    [[nodiscard]] static std::bitset<Shader::NUM_BUFFERS> UnboundBuffers(const Shader::Info& info);
    static void ConfigureVertexFetch(Shader::Info& info, const VertexFetchLimits* limits);
    [[nodiscard]] static PreparedShader PrepareGuest(
        Shader::Pools& pools, const Shader::Profile& profile, bool native_sampler_lod_bias,
        bool native_query_texture_lod, Shader::Info& info, Shader::RuntimeInfo runtime,
        std::span<const u32> code, Shader::Backend::Bindings& binding, bool flip_vertex_y = false,
        bool sample_rate_shading = false, const GeometryKernelConfig* geometry = nullptr,
        const GeometryCopyConfig* geometry_copy = nullptr,
        const GeometryExportConfig* geometry_export = nullptr,
        const VertexFetchLimits* vertex_fetch_limits = nullptr,
        const TessellationKernelConfig* tessellation = nullptr);
    [[nodiscard]] CompiledShader CompileGuest(
        Shader::Info& info, Shader::RuntimeInfo runtime, std::span<const u32> code,
        Shader::Backend::Bindings& binding, std::string_view name, bool flip_vertex_y = false,
        bool sample_rate_shading = false, const GeometryKernelConfig* geometry = nullptr,
        const GeometryCopyConfig* geometry_copy = nullptr,
        const GeometryExportConfig* geometry_export = nullptr,
        const VertexFetchLimits* vertex_fetch_limits = nullptr,
        const TessellationKernelConfig* tessellation = nullptr);

private:
    Shader::Profile profile;
    bool native_sampler_lod_bias;
    bool native_query_texture_lod;
    Shader::Pools pools;
    std::mutex translation_mutex;
    PipelineCompiler compiler;
};

} // namespace Metal
