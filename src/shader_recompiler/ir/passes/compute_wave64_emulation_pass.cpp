// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include <span>

#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {

namespace {

bool HasCrossLaneOperation(const IR::Program& program) {
    for (const IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::ReadFirstLane:
            case IR::Opcode::ReadLane:
            case IR::Opcode::Ballot:
            case IR::Opcode::BallotFindLsb:
            case IR::Opcode::GroupAny:
                return true;
            default:
                break;
            }
        }
    }
    return false;
}

bool HasStraightLineControlFlow(const IR::Program& program) {
    if (program.syntax_list.empty() ||
        program.syntax_list.back().type != IR::AbstractSyntaxNode::Type::Return) {
        return false;
    }

    const auto body = std::span{program.syntax_list}.first(program.syntax_list.size() - 1);
    return std::ranges::all_of(body, [](const IR::AbstractSyntaxNode& node) {
        return node.type == IR::AbstractSyntaxNode::Type::Block;
    });
}

} // Anonymous namespace

void ComputeWave64EmulationPass(IR::Program& program, const RuntimeInfo& runtime_info,
                                const Profile& profile) {
    auto& info = program.info;
    if (info.stage != Stage::Compute || !profile.needs_compute_wave64_emulation) {
        return;
    }

    const auto& workgroup_size = runtime_info.cs_info.workgroup_size;
    const u32 num_invocations = workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
    if (num_invocations != 64 || !HasStraightLineControlFlow(program) ||
        !HasCrossLaneOperation(program)) {
        return;
    }

    // The exchange array is native workgroup memory even when guest shared memory was lowered to
    // a storage buffer. Account only for guest memory that is still emitted as workgroup memory.
    constexpr u32 Wave64ScratchSize = 64 * sizeof(u32);
    const u32 guest_workgroup_memory =
        info.shared_types == IR::Type::Void ? 0 : runtime_info.cs_info.shared_memory_size;
    if (profile.max_shared_memory_size < Wave64ScratchSize ||
        guest_workgroup_memory > profile.max_shared_memory_size - Wave64ScratchSize) {
        return;
    }

    // Workgroup exchange uses control barriers. Restrict it to one complete guest wave with no
    // conditional control flow so every invocation is guaranteed to reach every barrier.
    info.emulate_compute_wave64_cross_lane = true;
}

} // namespace Shader::Optimization
