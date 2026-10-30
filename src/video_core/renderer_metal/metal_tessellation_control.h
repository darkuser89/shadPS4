// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>
#include "shader_recompiler/ir/program.h"

namespace Metal {

class TessellationControlFlow {
public:
    void CollectBarriers(const Shader::IR::Program& program);
    void Lower(Shader::IR::Program& program, u32 threads);

private:
    std::unordered_set<const Shader::IR::Inst*> barriers;
    std::vector<std::unique_ptr<Shader::IR::Block>> blocks;
};

} // namespace Metal
