// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_metal/metal_geometry_export.h"
#include "video_core/renderer_metal/metal_tessellation_control.h"
#include "video_core/renderer_metal/metal_tessellation_raster.h"
#include "video_core/renderer_metal/metal_tessellation_scalar.h"

namespace Metal {

[[nodiscard]] bool SynchronizeTessellationLds(Shader::IR::Program& program);
void LowerTessellationWaves(Shader::IR::Program& program, u32 threads);

struct TessellationKernelConfig {
    u32 workgroups;
    u32 threads_per_group;
    u32 lds_bytes;
    std::array<u32, 9> constants;
    bool compact_evaluation{};
    bool raster_output{};

    bool operator==(const TessellationKernelConfig&) const = default;

    [[nodiscard]] TessellationKernelConfig Evaluation() const {
        auto result = *this;
        result.threads_per_group = 64;
        return result;
    }

    [[nodiscard]] u64 InputSizeBytes() const {
        return RegisterSizeBytes() + (compact_evaluation ? u64{workgroups} * sizeof(u32) : 0) +
               (raster_output ? 4 * sizeof(float) : 0);
    }

    [[nodiscard]] u64 RegisterSizeBytes() const {
        return u64{workgroups} * threads_per_group * sizeof(GeometryExportInput);
    }

    [[nodiscard]] u64 LdsSizeBytes() const {
        return u64{workgroups} * lds_bytes;
    }
};

struct TessellationKernelBindings {
    GeometryExportBindings registers;
    std::optional<u32> lds;
    std::optional<u32> errors;
    std::optional<TessellationRasterBindings> raster;
    std::shared_ptr<const GeometryBufferAddressProgram> scalar_reads;
};

class TessellationKernelTranslation {
public:
    TessellationKernelTranslation(const TessellationKernelConfig& config,
                                  const Shader::RuntimeInfo& runtime,
                                  const Shader::Profile& profile);

    [[nodiscard]] Shader::TranslationHooks Hooks();
    [[nodiscard]] Shader::IR::Program Translate(
        std::span<const u32> code, Shader::Pools& pools, Shader::Info& info,
        Shader::RuntimeInfo& translated_runtime, const Shader::Profile& translation_profile,
        const Shader::TranslationHooks& hooks,
        void (*lower_resources)(Shader::IR::Program&) = nullptr,
        bool (*patch_buffer)(Shader::IR::Inst&, Shader::Info&) = nullptr,
        Shader::ImageReadConverter convert_image_read = nullptr,
        Shader::ImageSampleConverter convert_image_sample = nullptr);
    void LowerEvaluationGroups(Shader::IR::Program& program) const;
    [[nodiscard]] TessellationKernelBindings Bindings() const {
        return bindings;
    }

private:
    TessellationKernelConfig config;
    Shader::RuntimeInfo runtime;
    Shader::Profile profile;
    TessellationKernelBindings bindings;
    std::optional<TessellationRasterTranslation> raster;
    TessellationControlFlow control_flow;
    TessellationScalarReads scalar_constants;
};

} // namespace Metal
