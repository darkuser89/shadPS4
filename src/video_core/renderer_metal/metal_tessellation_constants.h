// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include "shader_recompiler/runtime_info.h"

namespace Shader {
struct Profile;
}

namespace Metal {

struct TessellationConstantBinding {
    u32 pointer_register;
    u32 dword_offset;

    bool operator==(const TessellationConstantBinding&) const = default;
};

[[nodiscard]] TessellationConstantBinding InspectTessellationConstantBinding(
    std::span<const u32> code, const Shader::RuntimeInfo& runtime, const Shader::Profile& profile);

[[nodiscard]] Shader::TessellationDataConstantBuffer ReadTessellationConstants(
    const TessellationConstantBinding& binding, std::span<const u32> user_data,
    const std::function<void(VAddr, u64)>& prepare_memory_read);

} // namespace Metal
