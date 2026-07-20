// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/types.h"
#include "shader_recompiler/profile.h"

std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst);
std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst, const Shader::Profile& profile);
std::vector<u32> EmitGroupAnyToSpirv(const Shader::Profile& profile);
