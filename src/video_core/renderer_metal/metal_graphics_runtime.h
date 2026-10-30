// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/runtime_info.h"

namespace AmdGpu {
union Regs;
}
namespace Shader {
struct Profile;
}
namespace Metal {

[[nodiscard]] Shader::RuntimeInfo BuildGraphicsRuntime(const AmdGpu::Regs& regs,
                                                       Shader::HwStage stage,
                                                       const Shader::Profile& profile);
[[nodiscard]] Shader::RuntimeInfo BuildTessellationRuntime(
    const AmdGpu::Regs& regs, Shader::HwStage stage,
    const Shader::TessellationDataConstantBuffer& constants);

} // namespace Metal
