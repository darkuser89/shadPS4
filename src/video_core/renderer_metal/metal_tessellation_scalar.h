// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "shader_recompiler/ir/passes/resource_pass.h"
#include "shader_recompiler/ir/program.h"
#include "video_core/renderer_metal/metal_geometry_address.h"

namespace Metal {

class TessellationScalarReads {
public:
    void Prepare(Shader::IR::Program& program,
                 const Shader::Optimization::ResourceDiscoveryList& resources);
    void Lower(Shader::IR::Program& program);
    [[nodiscard]] std::shared_ptr<const GeometryBufferAddressProgram> LiveReads() const;

private:
    GeometryBufferAddressProgram addresses;
    std::vector<Shader::IR::Value> locations;
    std::vector<Shader::IR::Value> predicates;
    std::vector<const Shader::IR::Inst*> reads;
};

[[nodiscard]] std::vector<GeometryBufferByteInterval> ScalarReadRanges(
    const GeometryBufferAddressProgram& program, std::span<const GeometryAddressInput> inputs,
    std::span<const u32> user_data, u64 byte_limit);

} // namespace Metal
