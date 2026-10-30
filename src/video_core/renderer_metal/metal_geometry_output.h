// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <optional>
#include <span>
#include <vector>
#include "shader_recompiler/ir/attribute.h"
#include "shader_recompiler/ir/value.h"
#include "video_core/renderer_metal/metal_geometry_layout.h"

namespace Shader::IR {
struct Program;
class IREmitter;
} // namespace Shader::IR

namespace Metal {

struct GeometryOutputAttribute {
    Shader::IR::Attribute attribute;
    u32 component;
    u32 stream;
    u32 word;
};

class GeometryClipPlanes {
public:
    void Collect(std::map<std::pair<Shader::IR::Attribute, u32>, u32>& outputs, u32 mask);
    void Bind(Shader::IR::Program& program);
    void Evaluate(Shader::IR::IREmitter& ir, std::span<const GeometryOutputAttribute> attributes,
                  std::span<Shader::IR::U32> values) const;

private:
    std::array<std::optional<u32>, 6> slots{};
    std::optional<u32> buffer;
};

[[nodiscard]] u32 GeometryInitialActiveCount(u32 work_items, u32 requested);

void InitializeGeometryExec(Shader::IR::Program& program, Shader::IR::IREmitter& ir,
                            Shader::IR::Attribute active_attribute);

void LowerGeometryWorkgroupMap(Shader::IR::Program& program, u32 input_buffer, u32 mapping_word);

void LowerGeometryWaveOperations(Shader::IR::Program& program);

void LowerGeometryOutputs(Shader::IR::Program& program,
                          const GeometryAssemblyParameters& parameters,
                          std::span<const GeometryOutputAttribute> attributes, u32 output_buffer,
                          const Shader::IR::U32& bounded_invocation);

} // namespace Metal
