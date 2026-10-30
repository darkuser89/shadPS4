// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "video_core/renderer_metal/metal_tessellation_kernel.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

bool NeedsWaveExchange(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::Ballot:
    case IR::Opcode::ReadFirstLane:
    case IR::Opcode::ReadLane:
    case IR::Opcode::Shuffle:
    case IR::Opcode::ShuffleXor:
    case IR::Opcode::QuadBroadcast:
    case IR::Opcode::GroupAny:
        return true;
    default:
        return false;
    }
}

bool IsUniform(const IR::Value& value, std::unordered_map<const IR::Inst*, bool>& known) {
    if (value.IsImmediate()) {
        return true;
    }
    if (value.IsEmpty()) {
        return false;
    }
    const auto* inst = value.Inst();
    const auto [entry, inserted] = known.try_emplace(inst, false);
    if (!inserted) {
        return entry->second;
    }
    const auto opcode = inst->GetOpcode();
    if (opcode == IR::Opcode::GetUserData) {
        return known[inst] = true;
    }
    if (opcode == IR::Opcode::GetAttributeU32) {
        return known[inst] = inst->Arg(0).Attribute() == IR::Attribute::WorkgroupId ||
                             inst->Arg(0).Attribute() == IR::Attribute::WorkgroupIndex;
    }
    const bool pure = (opcode >= IR::Opcode::IAdd32 && opcode <= IR::Opcode::UGreaterThanEqual64 &&
                       opcode != IR::Opcode::MaskedBitCount32) ||
                      (opcode >= IR::Opcode::LogicalOr && opcode <= IR::Opcode::LogicalNot) ||
                      opcode == IR::Opcode::SelectU32 || opcode == IR::Opcode::SelectU64 ||
                      opcode == IR::Opcode::ConditionRef;
    if (!pure) {
        return false;
    }
    for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
        if (!IsUniform(inst->Arg(arg), known)) {
            return false;
        }
    }
    return known[inst] = true;
}

IR::U32 FirstLane(IR::IREmitter& ir, const IR::U64& mask) {
    const auto halves = ir.UnpackUint2x32(mask);
    const IR::U32 low{ir.CompositeExtract(halves, 0)};
    const IR::U32 high{ir.CompositeExtract(halves, 1)};
    return IR::U32{ir.Select(ir.INotEqual(low, ir.Imm32(0u)), ir.FindILsb(low),
                             ir.Select(ir.INotEqual(high, ir.Imm32(0u)),
                                       ir.IAdd(ir.Imm32(32u), ir.FindILsb(high)), ir.Imm32(~0u)))};
}

} // namespace

bool SynchronizeTessellationLds(IR::Program& program) {
    std::unordered_map<const IR::Inst*, bool> known;
    std::unordered_set<const IR::Block*> uniform;
    std::vector<bool> conditions;
    u32 divergence{};
    bool loops{};
    bool early_return{};
    bool synchronization{};
    bool scheduled{};
    using Type = IR::AbstractSyntaxNode::Type;
    for (const auto& node : program.syntax_list) {
        switch (node.type) {
        case Type::If: {
            const bool divergent = !IsUniform(node.data.if_node.cond, known);
            conditions.push_back(divergent);
            divergence += divergent;
            break;
        }
        case Type::EndIf:
            if (conditions.empty()) {
                throw std::runtime_error("Invalid Metal tessellation conditional scope");
            }
            divergence -= conditions.back();
            conditions.pop_back();
            break;
        case Type::Loop:
        case Type::Repeat:
        case Type::Break:
            loops = true;
            break;
        case Type::Return:
        case Type::Unreachable:
            early_return |= !conditions.empty();
            break;
        case Type::Block:
            if (!divergence) {
                uniform.insert(node.data.block);
            }
            break;
        }
    }
    if (!conditions.empty()) {
        throw std::runtime_error("Unbalanced Metal tessellation conditional scope");
    }
    for (auto* block : program.blocks) {
        for (const auto& inst : *block) {
            if (!Shader::Optimization::IsSharedMemoryInstruction(inst) &&
                inst.GetOpcode() != IR::Opcode::Barrier && !NeedsWaveExchange(inst.GetOpcode())) {
                continue;
            }
            synchronization = true;
            if (!uniform.contains(block)) {
                scheduled = true;
            }
        }
    }
    if (synchronization && (loops || early_return)) {
        scheduled = true;
    }
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : *block) {
            if (Shader::Optimization::IsSharedMemoryInstruction(inst)) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
        }
    }
    return scheduled;
}

void LowerTessellationWaves(IR::Program& program, u32 threads) {
    std::vector<IR::Inst*> instructions;
    bool scratch{};
    for (auto* block : program.blocks) {
        for (auto& inst : *block) {
            const auto opcode = inst.GetOpcode();
            if (NeedsWaveExchange(opcode) || opcode == IR::Opcode::LaneId ||
                opcode == IR::Opcode::MaskedBitCount32 || opcode == IR::Opcode::BallotFindLsb) {
                instructions.push_back(&inst);
                scratch |= NeedsWaveExchange(opcode);
            }
        }
    }
    if (threads < 64 || threads > 256 || threads % 64 ||
        (program.info.shared_memory_scratch_size &&
         program.info.shared_memory_scratch_size != threads * sizeof(u32))) {
        throw std::runtime_error("Invalid Metal tessellation wave scratch layout");
    }
    if (scratch) {
        program.info.shared_memory_scratch_size = threads * sizeof(u32);
    }
    for (auto* inst : instructions) {
        IR::IREmitter ir{*inst->GetParent(), IR::Block::InstructionList::s_iterator_to(*inst)};
        const auto local = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
        const auto lane = ir.BitwiseAnd(local, ir.Imm32(63u));
        const auto base = ir.BitwiseAnd(local, ir.Imm32(~63u));
        const auto offset = ir.ShiftLeftLogical(local, ir.Imm32(2u));
        const auto ballot = [&](const IR::U1& predicate) {
            const IR::U32 low{ir.CompositeExtract(ir.UnpackUint2x32(ir.Ballot(predicate)), 0)};
            ir.WriteShared(32, low, offset);
            ir.Barrier();
            const IR::U32 lo{ir.LoadShared(32, false, ir.ShiftLeftLogical(base, ir.Imm32(2u)))};
            const IR::U32 hi{ir.LoadShared(
                32, false, ir.ShiftLeftLogical(ir.IAdd(base, ir.Imm32(32u)), ir.Imm32(2u)))};
            ir.Barrier();
            return ir.PackUint2x32(ir.CompositeConstruct(lo, hi));
        };
        const auto read = [&](const IR::U32& value, const IR::U32& target) {
            ir.WriteShared(32, value, offset);
            ir.Barrier();
            const IR::U32 result{ir.LoadShared(
                32, false,
                ir.ShiftLeftLogical(ir.IAdd(base, ir.BitwiseAnd(target, ir.Imm32(63u))),
                                    ir.Imm32(2u)))};
            ir.Barrier();
            return result;
        };
        switch (inst->GetOpcode()) {
        case IR::Opcode::LaneId:
            inst->ReplaceUsesWithAndRemove(lane);
            break;
        case IR::Opcode::Ballot:
            inst->ReplaceUsesWithAndRemove(ballot(IR::U1{inst->Arg(0)}));
            break;
        case IR::Opcode::GroupAny:
            inst->ReplaceUsesWithAndRemove(
                ir.INotEqual(ballot(IR::U1{inst->Arg(0)}), ir.Imm64(u64{0})));
            break;
        case IR::Opcode::ReadFirstLane: {
            const auto first = FirstLane(ir, ballot(ir.GetExec()));
            inst->ReplaceUsesWithAndRemove(read(IR::U32{inst->Arg(0)}, first));
            break;
        }
        case IR::Opcode::ReadLane:
        case IR::Opcode::Shuffle:
            inst->ReplaceUsesWithAndRemove(read(IR::U32{inst->Arg(0)}, IR::U32{inst->Arg(1)}));
            break;
        case IR::Opcode::ShuffleXor:
            inst->ReplaceUsesWithAndRemove(
                read(IR::U32{inst->Arg(0)}, ir.BitwiseXor(lane, IR::U32{inst->Arg(1)})));
            break;
        case IR::Opcode::QuadBroadcast:
            inst->ReplaceUsesWithAndRemove(
                read(IR::U32{inst->Arg(0)},
                     ir.BitwiseOr(ir.BitwiseAnd(lane, ir.Imm32(~3u)),
                                  ir.BitwiseAnd(IR::U32{inst->Arg(1)}, ir.Imm32(3u)))));
            break;
        case IR::Opcode::MaskedBitCount32: {
            const auto mask =
                ir.ISub(ir.ShiftLeftLogical(ir.Imm64(u64{1}), lane), ir.Imm64(u64{1}));
            const IR::U32 half{
                ir.CompositeExtract(ir.UnpackUint2x32(mask), inst->Arg(2).U1() ? 1u : 0u)};
            inst->ReplaceUsesWithAndRemove(ir.IAdd(
                ir.BitCount(ir.BitwiseAnd(IR::U32{inst->Arg(0)}, half)), IR::U32{inst->Arg(1)}));
            break;
        }
        case IR::Opcode::BallotFindLsb:
            inst->ReplaceUsesWithAndRemove(FirstLane(ir, IR::U64{inst->Arg(0)}));
            break;
        default:
            throw std::runtime_error("Unexpected Metal tessellation wave operation");
        }
    }
}

} // namespace Metal
