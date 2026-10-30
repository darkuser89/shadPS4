// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_metal/metal_geometry_address.h"
#include "video_core/renderer_metal/metal_geometry_output.h"
#include "video_core/renderer_metal/metal_geometry_resources.h"

namespace Metal {

struct GeometryCopyInput {
    std::array<u32, 16> system_sgprs;
    std::array<u32, 4> vgprs;
    u32 invocation;
    u32 vertex;
    std::array<float, 4> viewport;
};
static_assert(sizeof(GeometryCopyInput) == 104);

struct GeometryCopyWave {
    std::array<u32, 16> system_sgprs;
    std::array<u32, 3> vgprs;
    std::array<float, 4> viewport;
};
static_assert(sizeof(GeometryCopyWave) == 92);

struct GeometryCopyParameters {
    u32 work_items;
    u32 invocations;
    u32 max_vertices;
    u32 components;
    u32 ring_words;
    u32 base_word_offset;

    bool operator==(const GeometryCopyParameters&) const = default;
};
static_assert(sizeof(GeometryCopyParameters) == 24);

[[nodiscard]] inline GeometryCopyParameters GeometryCopyWorkLayout(u32 invocations,
                                                                   u32 max_vertices, u32 components,
                                                                   u32 ring_bytes,
                                                                   u32 base_word_offset) {
    const u64 work_items = u64{invocations} * max_vertices;
    if (!invocations || invocations % 64 || !max_vertices || !components ||
        u64{max_vertices} * components >= (1u << 12) || !ring_bytes || ring_bytes % sizeof(u32) ||
        base_word_offset >= ring_bytes / sizeof(u32) ||
        work_items > std::numeric_limits<u32>::max() ||
        work_items * (sizeof(GeometryCopyInput) / sizeof(u32)) >
            u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Invalid native geometry copy work layout");
    }
    return {u32(work_items), invocations, max_vertices, components, u32(ring_bytes / sizeof(u32)),
            base_word_offset};
}

struct GeometryCopyConfig {
    GeometryAssemblyParameters emissions;
    u32 work_items;
    u32 stream;
    u32 shared_memory_size{};
    bool compact_workgroups{};

    bool operator==(const GeometryCopyConfig&) const = default;

    [[nodiscard]] u64 InputSizeBytes() const {
        return RegisterSizeBytes() + (compact_workgroups ? u64{work_items / 64} * sizeof(u32) : 0);
    }

    [[nodiscard]] u64 RegisterSizeBytes() const {
        return u64{work_items} * sizeof(GeometryCopyInput);
    }
};

struct GeometryCopyResult {
    GeometryAssemblyParameters payload;
    std::vector<GeometryOutputAttribute> attributes;
    u32 inputs;
    u32 emissions;
    u32 output;
    u32 first_binding{};
    std::shared_ptr<const GeometryBufferAddressOrigins> address_origins;
    std::shared_ptr<const GeometryBufferAddressProgram> address_program;
    u64 input_sources{};
    std::array<u32, 2> viewport_extent{};

    [[nodiscard]] u32 InputBinding() const {
        return first_binding + inputs;
    }

    [[nodiscard]] u32 EmissionBinding() const {
        return first_binding + emissions;
    }

    [[nodiscard]] u32 OutputBinding() const {
        return first_binding + output;
    }
};

class GeometryCopyTranslation {
public:
    GeometryCopyTranslation(const GeometryCopyConfig& config, const Shader::RuntimeInfo& runtime,
                            const Shader::Profile& profile);

    [[nodiscard]] Shader::TranslationHooks Hooks();
    void LowerWorkgroups(Shader::IR::Program& program) const;
    [[nodiscard]] const GeometryCopyResult& Result() const {
        return result;
    }

private:
    GeometryCopyConfig config;
    Shader::HwVertexRuntimeInfo vertex_runtime;
    u32 num_user_data;
    u32 viewport_width;
    u32 viewport_height;
    GeometryCopyResult result{};
    std::vector<Shader::IR::VirtualReg> values;
    Shader::IR::U32 input_base;
    Shader::IR::U32 output_row;
    Shader::IR::U32 ordinal;
    Shader::IR::U1 output_valid;
    GeometryClipPlanes clip_planes;
};

} // namespace Metal
