// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

Id SubgroupScope(EmitContext& ctx) {
    return ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup));
}

void Wave64Barrier(EmitContext& ctx) {
    const Id scope = ctx.ConstU32(static_cast<u32>(spv::Scope::Workgroup));
    const Id semantics = ctx.ConstU32(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                                       spv::MemorySemanticsMask::WorkgroupMemory));
    ctx.OpControlBarrier(scope, scope, semantics);
}

Id Wave64ScratchPointer(EmitContext& ctx, Id lane) {
    return ctx.OpAccessChain(ctx.wave64_scratch_u32, ctx.wave64_scratch, lane);
}

Id Wave64ReadLane(EmitContext& ctx, Id value, Id lane) {
    const Id local_index = ctx.OpLoad(ctx.U32[1], ctx.local_invocation_index);
    ctx.OpStore(Wave64ScratchPointer(ctx, local_index), value);
    Wave64Barrier(ctx);
    const Id result = ctx.OpLoad(
        ctx.U32[1],
        Wave64ScratchPointer(ctx, ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.ConstU32(63U))));
    // Prevent a following exchange from overwriting a value before the other half-wave read it.
    Wave64Barrier(ctx);
    return result;
}

Id Wave64Ballot(EmitContext& ctx, Id bit) {
    const Id local_index = ctx.OpLoad(ctx.U32[1], ctx.local_invocation_index);
    const Id clear_label = ctx.OpLabel();
    const Id clear_merge_label = ctx.OpLabel();
    const Id must_clear = ctx.OpULessThan(ctx.U1[1], local_index, ctx.ConstU32(2U));
    ctx.OpSelectionMerge(clear_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(must_clear, clear_label, clear_merge_label);

    ctx.AddLabel(clear_label);
    ctx.OpStore(Wave64ScratchPointer(ctx, local_index), ctx.u32_zero_value);
    ctx.OpBranch(clear_merge_label);

    ctx.AddLabel(clear_merge_label);
    Wave64Barrier(ctx);

    const Id active_label = ctx.OpLabel();
    const Id active_merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(active_merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(bit, active_label, active_merge_label);

    ctx.AddLabel(active_label);
    const Id half = ctx.OpShiftRightLogical(ctx.U32[1], local_index, ctx.ConstU32(5U));
    const Id bit_index = ctx.OpBitwiseAnd(ctx.U32[1], local_index, ctx.ConstU32(31U));
    const Id bit_mask = ctx.OpShiftLeftLogical(ctx.U32[1], ctx.u32_one_value, bit_index);
    const Id scope = ctx.ConstU32(static_cast<u32>(spv::Scope::Workgroup));
    ctx.OpAtomicOr(ctx.U32[1], Wave64ScratchPointer(ctx, half), scope, ctx.u32_zero_value,
                   bit_mask);
    ctx.OpBranch(active_merge_label);

    ctx.AddLabel(active_merge_label);
    Wave64Barrier(ctx);
    const Id low = ctx.OpLoad(ctx.U32[1], Wave64ScratchPointer(ctx, ctx.u32_zero_value));
    const Id high = ctx.OpLoad(ctx.U32[1], Wave64ScratchPointer(ctx, ctx.u32_one_value));
    // Prevent the next exchange from clearing the words before all lanes have read them.
    Wave64Barrier(ctx);
    return ctx.OpCompositeConstruct(ctx.U32[4], low, high, ctx.u32_zero_value, ctx.u32_zero_value);
}

} // Anonymous namespace

Id EmitWarpId(EmitContext& ctx) {
    UNREACHABLE();
}

Id EmitLaneId(EmitContext& ctx) {
    if (ctx.info.l_stage == LogicalStage::Compute && ctx.profile.needs_compute_wave64_emulation) {
        const Id local_index = ctx.OpLoad(ctx.U32[1], ctx.local_invocation_index);
        return ctx.OpBitwiseAnd(ctx.U32[1], local_index, ctx.ConstU32(63U));
    }
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id EmitQuadShuffle(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitReadFirstLane(EmitContext& ctx, Id value) {
    if (ctx.info.emulate_compute_wave64_cross_lane) {
        return Wave64ReadLane(ctx, value, ctx.u32_zero_value);
    }
    return ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitReadLane(EmitContext& ctx, Id value, Id lane) {
    if (ctx.info.emulate_compute_wave64_cross_lane) {
        return Wave64ReadLane(ctx, value, lane);
    }
    return ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
}

Id EmitWriteLane(EmitContext& ctx, Id value, Id write_value, u32 lane) {
    const Id is_target_lane = ctx.OpIEqual(ctx.U1[1], EmitLaneId(ctx), ctx.ConstU32(lane & 63U));
    return ctx.OpSelect(ctx.U32[1], is_target_lane, write_value, value);
}

Id EmitBallot(EmitContext& ctx, Id bit) {
    if (ctx.info.emulate_compute_wave64_cross_lane) {
        return Wave64Ballot(ctx, bit);
    }
    return ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit);
}

Id EmitBallotFindLsb(EmitContext& ctx, Id mask) {
    if (ctx.info.emulate_compute_wave64_cross_lane) {
        const Id low = ctx.OpCompositeExtract(ctx.U32[1], mask, 0U);
        const Id high = ctx.OpCompositeExtract(ctx.U32[1], mask, 1U);
        const Id low_valid = ctx.OpINotEqual(ctx.U1[1], low, ctx.u32_zero_value);
        const Id high_valid = ctx.OpINotEqual(ctx.U1[1], high, ctx.u32_zero_value);
        const Id high_lsb =
            ctx.OpIAdd(ctx.U32[1], ctx.OpFindILsb(ctx.U32[1], high), ctx.ConstU32(32U));
        const Id no_low_result = ctx.OpSelect(ctx.U32[1], high_valid, high_lsb, ctx.ConstU32(~0U));
        return ctx.OpSelect(ctx.U32[1], low_valid, ctx.OpFindILsb(ctx.U32[1], low), no_low_result);
    }
    return ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), mask);
}

Id EmitGroupAny(EmitContext& ctx, Id bit) {
    if (ctx.info.emulate_compute_wave64_cross_lane) {
        const Id ballot = Wave64Ballot(ctx, bit);
        const Id low = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
        const Id high = ctx.OpCompositeExtract(ctx.U32[1], ballot, 1U);
        return ctx.OpINotEqual(ctx.U1[1], ctx.OpBitwiseOr(ctx.U32[1], low, high),
                               ctx.u32_zero_value);
    }
    if (!ctx.profile.SupportsSubgroup(SubgroupFeature::Vote) &&
        ctx.profile.SupportsSubgroup(SubgroupFeature::Ballot)) {
        const Id ballot = ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit);
        Id bits = ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
        for (u32 index = 1; index < 4; ++index) {
            bits = ctx.OpBitwiseOr(ctx.U32[1], bits,
                                   ctx.OpCompositeExtract(ctx.U32[1], ballot, index));
        }
        return ctx.OpINotEqual(ctx.U1[1], bits, ctx.u32_zero_value);
    }
    return ctx.OpGroupNonUniformAny(ctx.U1[1], SubgroupScope(ctx), bit);
}

} // namespace Shader::Backend::SPIRV
