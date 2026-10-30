// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_metal/metal_geometry_output.h"

namespace Metal {

struct TessellationRasterBindings {
    u32 output;
    u32 vertex_words;
    std::vector<GeometryOutputAttribute> attributes;
};

class TessellationRasterTranslation {
public:
    TessellationRasterTranslation(const Shader::RuntimeInfo& runtime,
                                  const Shader::Profile& profile);
    void Collect(Shader::IR::Program& program);
    void Lower(Shader::IR::Program& program, const Shader::IR::U32& invocation,
               const Shader::IR::U32& input_base, u32 inputs, u32 viewport_word);
    void Finish(Shader::IR::Program& program);
    [[nodiscard]] const TessellationRasterBindings& Bindings() const {
        return bindings;
    }

private:
    Shader::HwVertexRuntimeInfo vertex_runtime;
    u32 viewport_width;
    u32 viewport_height;
    TessellationRasterBindings bindings{};
    std::vector<Shader::IR::VirtualReg> values;
    Shader::IR::U32 row;
    Shader::IR::U1 active;
    u32 input_buffer{};
    u32 viewport_word{};
    GeometryClipPlanes clip_planes;
    bool dynamic_viewport{};
};

} // namespace Metal
