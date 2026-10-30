// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_metal/metal_geometry_address.h"
#include "video_core/renderer_metal/metal_geometry_layout.h"
#include "video_core/renderer_metal/metal_geometry_resources.h"

namespace Metal {

struct GeometryKernelWorkItem {
    u32 ring_offset;
    u32 wave_id;
    std::array<u32, 6> vertices;
    u32 primitive_id;
    u32 invocation_id;
    u32 active{1};
};
static_assert(sizeof(GeometryKernelWorkItem) == 44);

struct GeometryKernelInput {
    u32 ring_offset;
    u32 wave_id;
    std::array<u32, 6> vertex_offsets;
    u32 primitive_id;
    u32 invocation_id;
    u32 active{1};
};
static_assert(sizeof(GeometryKernelInput) == 44);

struct GeometryKernelConfig {
    GeometryAssemblyParameters emissions;
    u32 shared_memory_size{};
    bool compact_workgroups{};

    bool operator==(const GeometryKernelConfig&) const = default;

    [[nodiscard]] u64 InputSizeBytes() const {
        return RegisterSizeBytes() +
               (compact_workgroups ? u64{emissions.invocations / 64} * sizeof(u32) : 0);
    }

    [[nodiscard]] u64 RegisterSizeBytes() const {
        return u64{emissions.invocations} * sizeof(GeometryKernelInput);
    }
};

struct GeometryKernelBindings {
    u32 inputs;
    u32 emissions;
    u32 first_binding{};
    std::shared_ptr<const GeometryBufferAddressOrigins> address_origins;
    std::shared_ptr<const GeometryBufferAddressProgram> address_program;
    u64 input_sources{};

    [[nodiscard]] u32 InputBinding() const {
        return first_binding + inputs;
    }

    [[nodiscard]] u32 EmissionBinding() const {
        return first_binding + emissions;
    }
};

class GeometryKernelTranslation {
public:
    GeometryKernelTranslation(const GeometryKernelConfig& config,
                              const Shader::RuntimeInfo& runtime, const Shader::Profile& profile);

    [[nodiscard]] Shader::TranslationHooks Hooks();
    void LowerWorkgroups(Shader::IR::Program& program) const;
    [[nodiscard]] GeometryKernelBindings Bindings() const {
        return bindings;
    }

private:
    GeometryKernelConfig config;
    u32 num_user_data;
    GeometryKernelBindings bindings{};
};

} // namespace Metal
