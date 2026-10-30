// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/opcodes.h"

namespace Shader::Backend::SPIRV {

namespace {
using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;

std::pair<Id, Id> AtomicArgs(EmitContext& ctx) {
    const Id scope{ctx.ConstU32(static_cast<u32>(spv::Scope::Device))};
    const Id semantics{ctx.u32_zero_value};
    return {scope, semantics};
}

enum class FloatAtomicOperation {
    Min,
    Max,
    CompareSwap,
};

Id FloatAtomicCompare(EmitContext& ctx, Id left, Id right, FloatAtomicOperation operation,
                      u32 bit_size = 32) {
    const Id type = bit_size == 64 ? ctx.U64 : ctx.U32[1];
    const Id zero = bit_size == 64 ? ctx.u64_zero_value : ctx.u32_zero_value;
    const Id magnitude_mask =
        bit_size == 64 ? ctx.Constant(type, 0x7fffffffffffffffULL) : ctx.ConstU32(0x7fffffffu);
    const Id left_magnitude = ctx.OpBitwiseAnd(type, left, magnitude_mask);
    const Id right_magnitude = ctx.OpBitwiseAnd(type, right, magnitude_mask);
    const Id ordered = ctx.OpLogicalAnd(
        ctx.U1[1],
        ctx.OpULessThanEqual(ctx.U1[1], left_magnitude,
                             (bit_size == 64 ? ctx.Constant(type, 0x7ff0000000000000ULL)
                                             : ctx.ConstU32(0x7f800000u))),
        ctx.OpULessThanEqual(ctx.U1[1], right_magnitude,
                             (bit_size == 64 ? ctx.Constant(type, 0x7ff0000000000000ULL)
                                             : ctx.ConstU32(0x7f800000u))));
    const Id both_zero =
        ctx.OpIEqual(ctx.U1[1], ctx.OpBitwiseOr(type, left_magnitude, right_magnitude), zero);
    if (operation == FloatAtomicOperation::CompareSwap) {
        return ctx.OpLogicalAnd(
            ctx.U1[1], ordered,
            ctx.OpLogicalOr(ctx.U1[1], both_zero, ctx.OpIEqual(ctx.U1[1], left, right)));
    }
    const Id sign_mask =
        bit_size == 64 ? ctx.Constant(type, 0x8000000000000000ULL) : ctx.ConstU32(0x80000000u);
    const auto order_key = [&](Id bits) {
        const Id negative =
            ctx.OpINotEqual(ctx.U1[1], ctx.OpBitwiseAnd(type, bits, sign_mask), zero);
        return ctx.OpSelect(type, negative, ctx.OpNot(type, bits),
                            ctx.OpBitwiseXor(type, bits, sign_mask));
    };
    const Id left_key = order_key(left);
    const Id right_key = order_key(right);
    const Id compared = operation == FloatAtomicOperation::Max
                            ? ctx.OpUGreaterThan(ctx.U1[1], left_key, right_key)
                            : ctx.OpULessThan(ctx.U1[1], left_key, right_key);
    return ctx.OpLogicalAnd(
        ctx.U1[1], ordered,
        ctx.OpLogicalAnd(ctx.U1[1], ctx.OpLogicalNot(ctx.U1[1], both_zero), compared));
}

Id FloatAtomicU32(EmitContext& ctx, Id pointer, Id atomic_type, Id value,
                  FloatAtomicOperation operation, Id compare_value = {}) {
    const auto [scope, semantics] = AtomicArgs(ctx);
    const Id operand = ctx.OpBitcast(ctx.U32[1], value);
    const Id stored_operand = ctx.OpBitcast(atomic_type, operand);
    const Id loop = ctx.OpLabel();
    const Id continuation = ctx.OpLabel();
    const Id merge = ctx.OpLabel();
    ctx.OpBranch(loop);
    ctx.AddLabel(loop);
    const Id original = ctx.OpAtomicLoad(atomic_type, pointer, scope, semantics);
    const Id original_bits = ctx.OpBitcast(ctx.U32[1], original);
    const Id replace = operation == FloatAtomicOperation::CompareSwap
                           ? FloatAtomicCompare(ctx, original_bits, compare_value, operation)
                           : FloatAtomicCompare(ctx, operand, original_bits, operation);
    const Id replacement = ctx.OpSelect(atomic_type, replace, stored_operand, original);
    const Id observed = ctx.OpAtomicCompareExchange(atomic_type, pointer, scope, semantics,
                                                    semantics, replacement, original);
    const Id complete = ctx.OpIEqual(ctx.U1[1], observed, original);
    ctx.OpLoopMerge(merge, continuation, spv::LoopControlMask::MaskNone);
    ctx.OpBranchConditional(complete, merge, continuation);
    ctx.AddLabel(continuation);
    ctx.OpBranch(loop);
    ctx.AddLabel(merge);
    return ctx.OpBitcast(ctx.F32[1], original);
}

Id FloatCmpSwapU32(EmitContext& ctx, Id pointer, Id value, Id compare, bool is_shared = true) {
    return ctx.WithSharedMemoryLock(
        [&] {
            const Id original =
                FloatAtomicU32(ctx, pointer, ctx.U32[1], ctx.OpBitcast(ctx.F32[1], value),
                               FloatAtomicOperation::CompareSwap, compare);
            return ctx.OpBitcast(ctx.U32[1], original);
        },
        is_shared);
}

Id BufferFloatAtomic(EmitContext& ctx, u32 handle, Id address, Id value,
                     FloatAtomicOperation operation, Id compare_value = {}) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id pointer = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    return FloatAtomicU32(ctx, pointer, ctx.U32[1], value, operation, compare_value);
}

Id BoundedAtomicU32(EmitContext& ctx, Id pointer, Id bound, bool decrement,
                    bool signed_pointer = false) {
    const auto [scope, semantics] = AtomicArgs(ctx);
    const Id atomic_type = signed_pointer ? ctx.S32[1] : ctx.U32[1];
    if (bound.value == ctx.u32_zero_value.value) {
        const Id zero = signed_pointer ? ctx.ConstS32(0) : ctx.u32_zero_value;
        const Id original = ctx.OpAtomicExchange(atomic_type, pointer, scope, semantics, zero);
        return signed_pointer ? ctx.OpBitcast(ctx.U32[1], original) : original;
    }
    const Id loop = ctx.OpLabel();
    const Id continuation = ctx.OpLabel();
    const Id merge = ctx.OpLabel();
    ctx.OpBranch(loop);
    ctx.AddLabel(loop);
    const Id loaded = ctx.OpAtomicLoad(atomic_type, pointer, scope, semantics);
    const Id original = signed_pointer ? ctx.OpBitcast(ctx.U32[1], loaded) : loaded;
    const Id wrapped =
        decrement
            ? ctx.OpLogicalOr(ctx.U1[1], ctx.OpIEqual(ctx.U1[1], original, ctx.u32_zero_value),
                              ctx.OpUGreaterThan(ctx.U1[1], original, bound))
            : ctx.OpUGreaterThanEqual(ctx.U1[1], original, bound);
    const Id stepped = decrement ? ctx.OpISub(ctx.U32[1], original, ctx.ConstU32(1u))
                                 : ctx.OpIAdd(ctx.U32[1], original, ctx.ConstU32(1u));
    const Id replacement =
        ctx.OpSelect(ctx.U32[1], wrapped, decrement ? bound : ctx.u32_zero_value, stepped);
    const Id stored = signed_pointer ? ctx.OpBitcast(ctx.S32[1], replacement) : replacement;
    const Id observed = ctx.OpAtomicCompareExchange(atomic_type, pointer, scope, semantics,
                                                    semantics, stored, loaded);
    const Id complete = ctx.OpIEqual(ctx.U1[1], observed, loaded);
    ctx.OpLoopMerge(merge, continuation, spv::LoopControlMask::MaskNone);
    ctx.OpBranchConditional(complete, merge, continuation);
    ctx.AddLabel(continuation);
    ctx.OpBranch(loop);
    ctx.AddLabel(merge);
    return original;
}

Id GuestPageCounterU32(EmitContext& ctx, Id pointer, Id bound, bool decrement) {
    const auto [scope, semantics] = AtomicArgs(ctx);
    const Id zero_label = ctx.OpLabel();
    const Id nonzero_label = ctx.OpLabel();
    const Id full_label = ctx.OpLabel();
    const Id bounded_label = ctx.OpLabel();
    const Id nonzero_merge = ctx.OpLabel();
    const Id merge = ctx.OpLabel();
    const Id zero_bound = ctx.OpIEqual(ctx.U1[1], bound, ctx.u32_zero_value);
    ctx.OpSelectionMerge(merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(zero_bound, zero_label, nonzero_label);
    ctx.AddLabel(zero_label);
    const Id zero_result =
        ctx.OpAtomicExchange(ctx.U32[1], pointer, scope, semantics, ctx.u32_zero_value);
    ctx.OpBranch(merge);
    ctx.AddLabel(nonzero_label);
    const Id full_bound = ctx.OpIEqual(ctx.U1[1], bound, ctx.ConstU32(~0u));
    ctx.OpSelectionMerge(nonzero_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(full_bound, full_label, bounded_label);
    ctx.AddLabel(full_label);
    const Id full_result = decrement
                               ? ctx.OpAtomicIDecrement(ctx.U32[1], pointer, scope, semantics)
                               : ctx.OpAtomicIIncrement(ctx.U32[1], pointer, scope, semantics);
    ctx.OpBranch(nonzero_merge);
    ctx.AddLabel(bounded_label);
    const Id bounded_result = BoundedAtomicU32(ctx, pointer, bound, decrement);
    const Id bounded_result_label = ctx.last_label;
    ctx.OpBranch(nonzero_merge);
    ctx.AddLabel(nonzero_merge);
    const Id nonzero_result =
        ctx.OpPhi(ctx.U32[1], full_result, full_label, bounded_result, bounded_result_label);
    ctx.OpBranch(merge);
    ctx.AddLabel(merge);
    return ctx.OpPhi(ctx.U32[1], zero_result, zero_label, nonzero_result, nonzero_merge);
}

enum class DsAtomicOperation {
    ReverseSubtract,
    MaskOr,
    Wrap,
};

Id DsAtomicU32(EmitContext& ctx, Id pointer, Id data0, Id data1, DsAtomicOperation operation) {
    const auto [scope, semantics] = AtomicArgs(ctx);
    if (operation == DsAtomicOperation::MaskOr) {
        if (data0.value == ctx.u32_zero_value.value) {
            return ctx.OpAtomicOr(ctx.U32[1], pointer, scope, semantics, data1);
        }
        if (data0.value == ctx.ConstU32(~0u).value) {
            return ctx.OpAtomicExchange(ctx.U32[1], pointer, scope, semantics, data1);
        }
    }
    const Id loop = ctx.OpLabel();
    const Id continuation = ctx.OpLabel();
    const Id merge = ctx.OpLabel();
    ctx.OpBranch(loop);
    ctx.AddLabel(loop);
    const Id original = ctx.OpAtomicLoad(ctx.U32[1], pointer, scope, semantics);
    const Id replacement = [&] {
        switch (operation) {
        case DsAtomicOperation::ReverseSubtract:
            return ctx.OpISub(ctx.U32[1], data0, original);
        case DsAtomicOperation::MaskOr:
            return ctx.OpBitwiseOr(
                ctx.U32[1], ctx.OpBitwiseAnd(ctx.U32[1], original, ctx.OpNot(ctx.U32[1], data0)),
                data1);
        case DsAtomicOperation::Wrap:
            return ctx.OpSelect(ctx.U32[1], ctx.OpUGreaterThanEqual(ctx.U1[1], original, data0),
                                ctx.OpISub(ctx.U32[1], original, data0),
                                ctx.OpIAdd(ctx.U32[1], original, data1));
        default:
            UNREACHABLE();
        }
    }();
    const Id observed = ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope, semantics,
                                                    semantics, replacement, original);
    const Id complete = ctx.OpIEqual(ctx.U1[1], observed, original);
    ctx.OpLoopMerge(merge, continuation, spv::LoopControlMask::MaskNone);
    ctx.OpBranchConditional(complete, merge, continuation);
    ctx.AddLabel(continuation);
    ctx.OpBranch(loop);
    ctx.AddLabel(merge);
    return original;
}

Id SharedDsAtomic(EmitContext& ctx, Id offset, Id data0, Id data1, DsAtomicOperation operation) {
    const Id index = ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(2u));
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
    return ctx.WithSharedMemoryLock(
        [&] { return DsAtomicU32(ctx, pointer, data0, data1, operation); });
}

Id BufferDsAtomic(EmitContext& ctx, u32 handle, Id address, Id data0, Id data1,
                  DsAtomicOperation operation) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id pointer = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    return ctx.WithSharedMemoryLock(
        [&] { return DsAtomicU32(ctx, pointer, data0, data1, operation); },
        buffer.buffer_type == BufferType::SharedMemory);
}

Id SharedAtomicU32(EmitContext& ctx, Id offset, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.WithSharedMemoryLock(
        [&] { return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics, value); });
}

Id SharedAtomicU32IncDec(EmitContext& ctx, Id offset, Id bound, bool decrement,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index)};
    if (bound.value != ctx.ConstU32(~0u).value) {
        return ctx.WithSharedMemoryLock(
            [&] { return BoundedAtomicU32(ctx, pointer, bound, decrement); });
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.WithSharedMemoryLock(
        [&] { return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics); });
}

Id SoftwareSharedAtomicU64(EmitContext& ctx, Id pointer, Id value,
                           Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    return ctx.WithSharedMemoryLock([&] {
        const Id original = ctx.OpLoad(ctx.U64, pointer);
        const Id replacement = [&] {
            if (atomic_func == &Sirit::Module::OpAtomicExchange) {
                return value;
            }
            if (atomic_func == &Sirit::Module::OpAtomicIAdd) {
                return ctx.OpIAdd(ctx.U64, original, value);
            }
            if (atomic_func == &Sirit::Module::OpAtomicISub) {
                return ctx.OpISub(ctx.U64, original, value);
            }
            if (atomic_func == &Sirit::Module::OpAtomicAnd) {
                return ctx.OpBitwiseAnd(ctx.U64, original, value);
            }
            if (atomic_func == &Sirit::Module::OpAtomicOr) {
                return ctx.OpBitwiseOr(ctx.U64, original, value);
            }
            if (atomic_func == &Sirit::Module::OpAtomicXor) {
                return ctx.OpBitwiseXor(ctx.U64, original, value);
            }
            if (atomic_func != &Sirit::Module::OpAtomicUMin &&
                atomic_func != &Sirit::Module::OpAtomicSMin &&
                atomic_func != &Sirit::Module::OpAtomicUMax &&
                atomic_func != &Sirit::Module::OpAtomicSMax) {
                UNREACHABLE();
            }
            const bool minimum = atomic_func == &Sirit::Module::OpAtomicUMin ||
                                 atomic_func == &Sirit::Module::OpAtomicSMin;
            const bool signed_value = atomic_func == &Sirit::Module::OpAtomicSMin ||
                                      atomic_func == &Sirit::Module::OpAtomicSMax;
            const Id replace = signed_value
                                   ? (minimum ? ctx.OpSLessThan(ctx.U1[1], value, original)
                                              : ctx.OpSGreaterThan(ctx.U1[1], value, original))
                                   : (minimum ? ctx.OpULessThan(ctx.U1[1], value, original)
                                              : ctx.OpUGreaterThan(ctx.U1[1], value, original));
            return ctx.OpSelect(ctx.U64, replace, value, original);
        }();
        ctx.OpStore(pointer, replacement);
        return original;
    });
}

Id SoftwareSharedCmpSwapU64(EmitContext& ctx, Id pointer, Id value, Id compare,
                            bool is_float = false) {
    return ctx.WithSharedMemoryLock([&] {
        const Id original = ctx.OpLoad(ctx.U64, pointer);
        const Id replace = is_float ? FloatAtomicCompare(ctx, original, compare,
                                                         FloatAtomicOperation::CompareSwap, 64)
                                    : ctx.OpIEqual(ctx.U1[1], original, compare);
        ctx.OpStore(pointer, ctx.OpSelect(ctx.U64, replace, value, original));
        return original;
    });
}

Id SoftwareSharedDsAtomicU64(EmitContext& ctx, Id pointer, Id data0, Id data1,
                             DsAtomicOperation operation) {
    ASSERT(ctx.info.uses_shared_memory_lock);
    return ctx.WithSharedMemoryLock([&] {
        const Id original = ctx.OpLoad(ctx.U64, pointer);
        const Id replacement = [&] {
            switch (operation) {
            case DsAtomicOperation::ReverseSubtract:
                return ctx.OpISub(ctx.U64, data0, original);
            case DsAtomicOperation::MaskOr:
                return ctx.OpBitwiseOr(
                    ctx.U64, ctx.OpBitwiseAnd(ctx.U64, original, ctx.OpNot(ctx.U64, data0)), data1);
            default:
                UNREACHABLE();
            }
        }();
        ctx.OpStore(pointer, replacement);
        return original;
    });
}

Id SoftwareSharedCounterU64(EmitContext& ctx, Id pointer, Id bound, bool decrement) {
    ASSERT(ctx.info.uses_shared_memory_lock);
    return ctx.WithSharedMemoryLock([&] {
        const Id original = ctx.OpLoad(ctx.U64, pointer);
        const Id wrapped =
            decrement
                ? ctx.OpLogicalOr(ctx.U1[1], ctx.OpIEqual(ctx.U1[1], original, ctx.u64_zero_value),
                                  ctx.OpUGreaterThan(ctx.U1[1], original, bound))
                : ctx.OpUGreaterThanEqual(ctx.U1[1], original, bound);
        const Id stepped = decrement ? ctx.OpISub(ctx.U64, original, ctx.u64_one_value)
                                     : ctx.OpIAdd(ctx.U64, original, ctx.u64_one_value);
        const Id replacement =
            ctx.OpSelect(ctx.U64, wrapped, decrement ? bound : ctx.u64_zero_value, stepped);
        ctx.OpStore(pointer, replacement);
        return original;
    });
}

void SharedMemorySourceAtomic(EmitContext& ctx, Id destination, Id source, u32 bit_size,
                              u32 operation) {
    ASSERT(ctx.info.uses_shared_memory_lock);
    const Id type = bit_size == 64 ? ctx.U64 : ctx.U32[1];
    const Id zero = bit_size == 64 ? ctx.u64_zero_value : ctx.u32_zero_value;
    const Id one = bit_size == 64 ? ctx.u64_one_value : ctx.ConstU32(1u);
    ctx.WithSharedMemoryLock([&] {
        const Id original = ctx.OpLoad(type, destination);
        const Id operand = ctx.OpLoad(type, source);
        const Id replacement = [&] {
            switch (static_cast<IR::SharedMemoryOperation>(operation)) {
            case IR::SharedMemoryOperation::Add:
                return ctx.OpIAdd(type, original, operand);
            case IR::SharedMemoryOperation::Subtract:
                return ctx.OpISub(type, original, operand);
            case IR::SharedMemoryOperation::ReverseSubtract:
                return ctx.OpISub(type, operand, original);
            case IR::SharedMemoryOperation::Increment:
                return ctx.OpSelect(type, ctx.OpUGreaterThanEqual(ctx.U1[1], original, operand),
                                    zero, ctx.OpIAdd(type, original, one));
            case IR::SharedMemoryOperation::Decrement:
                return ctx.OpSelect(
                    type,
                    ctx.OpLogicalOr(ctx.U1[1], ctx.OpIEqual(ctx.U1[1], original, zero),
                                    ctx.OpUGreaterThan(ctx.U1[1], original, operand)),
                    operand, ctx.OpISub(type, original, one));
            case IR::SharedMemoryOperation::SignedMin:
                return ctx.OpSelect(type, ctx.OpSLessThan(ctx.U1[1], operand, original), operand,
                                    original);
            case IR::SharedMemoryOperation::SignedMax:
                return ctx.OpSelect(type, ctx.OpSGreaterThan(ctx.U1[1], operand, original), operand,
                                    original);
            case IR::SharedMemoryOperation::UnsignedMin:
                return ctx.OpSelect(type, ctx.OpULessThan(ctx.U1[1], operand, original), operand,
                                    original);
            case IR::SharedMemoryOperation::UnsignedMax:
                return ctx.OpSelect(type, ctx.OpUGreaterThan(ctx.U1[1], operand, original), operand,
                                    original);
            case IR::SharedMemoryOperation::And:
                return ctx.OpBitwiseAnd(type, original, operand);
            case IR::SharedMemoryOperation::Or:
                return ctx.OpBitwiseOr(type, original, operand);
            case IR::SharedMemoryOperation::Xor:
                return ctx.OpBitwiseXor(type, original, operand);
            case IR::SharedMemoryOperation::Write:
                return operand;
            case IR::SharedMemoryOperation::FloatMin:
                return ctx.OpSelect(
                    type,
                    FloatAtomicCompare(ctx, operand, original, FloatAtomicOperation::Min, bit_size),
                    operand, original);
            case IR::SharedMemoryOperation::FloatMax:
                return ctx.OpSelect(
                    type,
                    FloatAtomicCompare(ctx, operand, original, FloatAtomicOperation::Max, bit_size),
                    operand, original);
            default:
                UNREACHABLE();
            }
        }();
        ctx.OpStore(destination, replacement);
        return Id{};
    });
}

void SharedSourceAtomic(EmitContext& ctx, Id address, Id source, u32 bit_size, u32 operation) {
    const Id shift = ctx.ConstU32(bit_size == 64 ? 3u : 2u);
    const Id pointer_type = bit_size == 64 ? ctx.shared_u64 : ctx.shared_u32;
    const Id memory = bit_size == 64 ? ctx.shared_memory_u64 : ctx.shared_memory_u32;
    const Id destination = ctx.EmitSharedMemoryAccess(
        pointer_type, memory, ctx.OpShiftRightLogical(ctx.U32[1], address, shift));
    const Id operand = ctx.EmitSharedMemoryAccess(
        pointer_type, memory, ctx.OpShiftRightLogical(ctx.U32[1], source, shift));
    SharedMemorySourceAtomic(ctx, destination, operand, bit_size, operation);
}

void BufferSourceAtomic(EmitContext& ctx, u32 handle, Id address, Id source, u32 bit_size,
                        u32 operation) {
    const auto& buffer = ctx.buffers[handle];
    ASSERT(buffer.buffer_type == BufferType::SharedMemory);
    const Id offset = buffer.Offset(bit_size == 64 ? PointerSize::B64 : PointerSize::B32);
    if (Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
        source = ctx.OpIAdd(ctx.U32[1], source, offset);
    }
    const auto [memory, pointer_type] =
        buffer.Alias(bit_size == 64 ? PointerType::U64 : PointerType::U32);
    const Id destination = ctx.OpAccessChain(pointer_type, memory, ctx.u32_zero_value, address);
    const Id operand = ctx.OpAccessChain(pointer_type, memory, ctx.u32_zero_value, source);
    SharedMemorySourceAtomic(ctx, destination, operand, bit_size, operation);
}

Id SharedU64Pointer(EmitContext& ctx, Id offset) {
    const Id index = ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(3u));
    return ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index);
}

Id BufferSharedU64Pointer(EmitContext& ctx, u32 handle, Id address) {
    const auto& buffer = ctx.buffers[handle];
    ASSERT(buffer.buffer_type == BufferType::SharedMemory);
    if (const Id offset = buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U64);
    return ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
}

Id SharedAtomicU64(EmitContext& ctx, Id offset, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)};
    if (ctx.info.uses_shared_memory_lock) {
        return SoftwareSharedAtomicU64(ctx, pointer, value, atomic_func);
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, pointer, scope, semantics, value);
}

Id SharedAtomicU64IncDec(EmitContext& ctx, Id offset,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)};
    if (ctx.info.uses_shared_memory_lock) {
        const auto operation = atomic_func == &Sirit::Module::OpAtomicIIncrement
                                   ? &Sirit::Module::OpAtomicIAdd
                                   : &Sirit::Module::OpAtomicISub;
        return SoftwareSharedAtomicU64(ctx, pointer, ctx.u64_one_value, operation);
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, pointer, scope, semantics);
}

Id SharedAtomicU32CmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.WithSharedMemoryLock([&] {
        return ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope, semantics, semantics, value,
                                           cmp_value);
    });
}

Id SharedAtomicU64CmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)};
    if (ctx.info.uses_shared_memory_lock) {
        return SoftwareSharedCmpSwapU64(ctx, pointer, value, cmp_value);
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(ctx.U64, pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

template <bool is_float = false>
Id BufferAtomicU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    const Id type = is_float ? ctx.F32[1] : ctx.U32[1];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(is_float ? PointerType::F32 : PointerType::U32);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.WithSharedMemoryLock(
        [&] { return (ctx.*atomic_func)(type, ptr, scope, semantics, value); },
        buffer.buffer_type == BufferType::SharedMemory);
}

Id BufferAtomicU32IncDec(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id bound,
                         bool decrement, Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    if (bound.value != ctx.ConstU32(~0u).value) {
        return ctx.WithSharedMemoryLock(
            [&] { return BoundedAtomicU32(ctx, ptr, bound, decrement); },
            buffer.buffer_type == BufferType::SharedMemory);
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.WithSharedMemoryLock(
        [&] { return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics); },
        buffer.buffer_type == BufferType::SharedMemory);
}

Id BufferAtomicU32CmpSwap(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          Id cmp_value,
                          Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.WithSharedMemoryLock(
        [&] {
            return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics, semantics, value,
                                      cmp_value);
        },
        buffer.buffer_type == BufferType::SharedMemory);
}

Id BufferAtomicU64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U64);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    if (ctx.info.uses_shared_memory_lock && buffer.buffer_type == BufferType::SharedMemory) {
        return SoftwareSharedAtomicU64(ctx, ptr, value, atomic_func);
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, ptr, scope, semantics, value);
}

Id FixImageAtomicCoords(EmitContext& ctx, Id coords, AmdGpu::ImageType image_type) {
    switch (image_type) {
    case AmdGpu::ImageType::Color1D: {
        // Lowered to 2D with height 1
        const auto x = coords;
        return ctx.OpCompositeConstruct(ctx.U32[2], x, ctx.u32_zero_value);
    }
    case AmdGpu::ImageType::Color1DArray: {
        // Lowered to 2D array with height 1
        const auto x = ctx.OpCompositeExtract(ctx.U32[1], coords, 0U);
        const auto slice = ctx.OpCompositeExtract(ctx.U32[1], coords, 1U);
        return ctx.OpCompositeConstruct(ctx.U32[3], x, ctx.u32_zero_value, slice);
    }
    default:
        return coords;
    }
}

Id ImageAtomicMinMax(EmitContext& ctx, Id pointer, Id atomic_type, Id value, bool signed_operation,
                     bool maximum) {
    const auto [scope, semantics] = AtomicArgs(ctx);
    const bool signed_pointer = atomic_type.value == ctx.S32[1].value;
    const Id comparison_type = signed_operation ? ctx.S32[1] : ctx.U32[1];
    const Id operand = signed_operation ? ctx.OpBitcast(ctx.S32[1], value) : value;
    const Id stored_operand = signed_pointer ? ctx.OpBitcast(ctx.S32[1], value) : value;
    const Id loop = ctx.OpLabel();
    const Id continuation = ctx.OpLabel();
    const Id merge = ctx.OpLabel();
    ctx.OpBranch(loop);
    ctx.AddLabel(loop);
    const Id original = ctx.OpAtomicLoad(atomic_type, pointer, scope, semantics);
    const Id compared = ctx.OpBitcast(comparison_type, original);
    const Id replace = signed_operation
                           ? (maximum ? ctx.OpSGreaterThan(ctx.U1[1], operand, compared)
                                      : ctx.OpSLessThan(ctx.U1[1], operand, compared))
                           : (maximum ? ctx.OpUGreaterThan(ctx.U1[1], operand, compared)
                                      : ctx.OpULessThan(ctx.U1[1], operand, compared));
    const Id replacement = ctx.OpSelect(atomic_type, replace, stored_operand, original);
    const Id observed = ctx.OpAtomicCompareExchange(atomic_type, pointer, scope, semantics,
                                                    semantics, replacement, original);
    const Id complete = ctx.OpIEqual(ctx.U1[1], observed, original);
    ctx.OpLoopMerge(merge, continuation, spv::LoopControlMask::MaskNone);
    ctx.OpBranchConditional(complete, merge, continuation);
    ctx.AddLabel(continuation);
    ctx.OpBranch(loop);
    ctx.AddLabel(merge);
    return signed_pointer ? ctx.OpBitcast(ctx.U32[1], original) : original;
}

Id ImageAtomicU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                  Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id atomic_type =
        ctx.profile.needs_typed_image_atomics ? (*texture.data_types)[1] : ctx.U32[1];
    const bool signed_pointer = atomic_type.value == ctx.S32[1].value;
    ASSERT(atomic_type.value == ctx.U32[1].value || signed_pointer);
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer_type = ctx.TypePointer(spv::StorageClass::Image, atomic_type);
    const Id pointer =
        ctx.OpImageTexelPointer(pointer_type, texture.id, fixed_coords, ctx.ConstU32(0U));
    if (ctx.profile.needs_typed_image_atomics) {
        const bool signed_operation = atomic_func == &Sirit::Module::OpAtomicSMin ||
                                      atomic_func == &Sirit::Module::OpAtomicSMax;
        const bool unsigned_operation = atomic_func == &Sirit::Module::OpAtomicUMin ||
                                        atomic_func == &Sirit::Module::OpAtomicUMax;
        if ((signed_operation && !signed_pointer) || (unsigned_operation && signed_pointer)) {
            const bool maximum = atomic_func == &Sirit::Module::OpAtomicSMax ||
                                 atomic_func == &Sirit::Module::OpAtomicUMax;
            return ImageAtomicMinMax(ctx, pointer, atomic_type, value, signed_operation, maximum);
        }
    }
    const auto [scope, semantics]{AtomicArgs(ctx)};
    const Id operand = signed_pointer ? ctx.OpBitcast(ctx.S32[1], value) : value;
    const Id original = (ctx.*atomic_func)(atomic_type, pointer, scope, semantics, operand);
    return signed_pointer ? ctx.OpBitcast(ctx.U32[1], original) : original;
}

Id ImageAtomicF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                  Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer{
        ctx.OpImageTexelPointer(ctx.image_f32, texture.id, fixed_coords, ctx.ConstU32(0U))};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.F32[1], pointer, scope, semantics, value);
}

Id ImageFloatAtomic(EmitContext& ctx, u32 handle, Id coords, Id value,
                    FloatAtomicOperation operation, Id compare_value = {}) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id atomic_type = (*texture.data_types)[1];
    ASSERT(atomic_type.value == ctx.U32[1].value || atomic_type.value == ctx.S32[1].value);
    const Id fixed_coords = FixImageAtomicCoords(ctx, coords, texture.view_type);
    const Id pointer_type = ctx.TypePointer(spv::StorageClass::Image, atomic_type);
    const Id pointer =
        ctx.OpImageTexelPointer(pointer_type, texture.id, fixed_coords, ctx.ConstU32(0U));
    return FloatAtomicU32(ctx, pointer, atomic_type, value, operation, compare_value);
}

Id ImageAtomicU32IncDec(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id bound,
                        bool decrement) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id atomic_type = (*texture.data_types)[1];
    const bool signed_pointer = atomic_type.value == ctx.S32[1].value;
    ASSERT(atomic_type.value == ctx.U32[1].value || signed_pointer);
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer_type = ctx.TypePointer(spv::StorageClass::Image, atomic_type);
    const Id pointer =
        ctx.OpImageTexelPointer(pointer_type, texture.id, fixed_coords, ctx.ConstU32(0U));
    if (bound.value == ctx.ConstU32(~0u).value) {
        const auto [scope, semantics] = AtomicArgs(ctx);
        const Id one = signed_pointer ? ctx.ConstS32(1) : ctx.ConstU32(1u);
        const Id original = decrement
                                ? ctx.OpAtomicISub(atomic_type, pointer, scope, semantics, one)
                                : ctx.OpAtomicIAdd(atomic_type, pointer, scope, semantics, one);
        return signed_pointer ? ctx.OpBitcast(ctx.U32[1], original) : original;
    }
    return BoundedAtomicU32(ctx, pointer, bound, decrement, signed_pointer);
}

Id ImageAtomicU32CmpSwap(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                         Id cmp_value,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id, Id, Id)) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id atomic_type =
        ctx.profile.needs_typed_image_atomics ? (*texture.data_types)[1] : ctx.U32[1];
    const bool signed_pointer = atomic_type.value == ctx.S32[1].value;
    ASSERT(atomic_type.value == ctx.U32[1].value || signed_pointer);
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer_type = ctx.TypePointer(spv::StorageClass::Image, atomic_type);
    const Id pointer =
        ctx.OpImageTexelPointer(pointer_type, texture.id, fixed_coords, ctx.ConstU32(0U));
    const auto [scope, semantics]{AtomicArgs(ctx)};
    const Id operand = signed_pointer ? ctx.OpBitcast(ctx.S32[1], value) : value;
    const Id compared = signed_pointer ? ctx.OpBitcast(ctx.S32[1], cmp_value) : cmp_value;
    const Id original =
        (ctx.*atomic_func)(atomic_type, pointer, scope, semantics, semantics, operand, compared);
    return signed_pointer ? ctx.OpBitcast(ctx.U32[1], original) : original;
}
} // Anonymous namespace

Id EmitContext::DefineGuestPageAtomic(IR::Opcode opcode, bool coherent) {
    const auto operation =
        OpFunction(U32[1], spv::FunctionControlMask::MaskNone,
                   TypeFunction(U32[1], physical_pointer_type_u32, U32[1], U32[1]));
    const auto pointer = OpFunctionParameter(physical_pointer_type_u32);
    const auto value = OpFunctionParameter(U32[1]);
    const auto compare = OpFunctionParameter(U32[1]);
    Decorate(pointer, spv::Decoration::AliasedPointer);
    if (coherent) {
        Decorate(pointer, spv::Decoration::Coherent);
    }
    const std::string name =
        std::string{IR::NameOf(opcode)} + (coherent ? "_guest_page_coherent" : "_guest_page");
    Name(operation, name + "_pointer");
    AddLabel();
    const auto scope = ConstU32(static_cast<u32>(spv::Scope::Device));
    Id result{};
    switch (opcode) {
    case IR::Opcode::BufferAtomicIAdd32:
        result = OpAtomicIAdd(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicISub32:
        result = OpAtomicISub(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicSMin32:
        result = OpAtomicSMin(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicUMin32:
        result = OpAtomicUMin(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicSMax32:
        result = OpAtomicSMax(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicUMax32:
        result = OpAtomicUMax(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicAnd32:
        result = OpAtomicAnd(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicOr32:
        result = OpAtomicOr(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicXor32:
        result = OpAtomicXor(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicSwap32:
        result = OpAtomicExchange(U32[1], pointer, scope, u32_zero_value, value);
        break;
    case IR::Opcode::BufferAtomicCmpSwap32:
        result = OpAtomicCompareExchange(U32[1], pointer, scope, u32_zero_value, u32_zero_value,
                                         value, compare);
        break;
    case IR::Opcode::BufferAtomicInc32:
    case IR::Opcode::BufferAtomicDec32:
        result =
            GuestPageCounterU32(*this, pointer, value, opcode == IR::Opcode::BufferAtomicDec32);
        break;
    case IR::Opcode::BufferAtomicFMin32:
    case IR::Opcode::BufferAtomicFMax32:
        result = OpBitcast(U32[1], FloatAtomicU32(*this, pointer, U32[1], OpBitcast(F32[1], value),
                                                  opcode == IR::Opcode::BufferAtomicFMin32
                                                      ? FloatAtomicOperation::Min
                                                      : FloatAtomicOperation::Max));
        break;
    case IR::Opcode::BufferAtomicFCmpSwapRaw32:
    case IR::Opcode::BufferAtomicFCmpSwap32:
        result = FloatCmpSwapU32(*this, pointer, value, compare, false);
        break;
    default:
        throw std::runtime_error("Unsupported native guest page atomic");
    }
    OpReturnValue(result);
    OpFunctionEnd();
    const auto function = OpFunction(U32[1], spv::FunctionControlMask::MaskNone,
                                     TypeFunction(U32[1], U64, U32[1], U32[1]));
    const auto address = OpFunctionParameter(U64);
    const auto operand = OpFunctionParameter(U32[1]);
    const auto comparison = OpFunctionParameter(U32[1]);
    Name(function, name);
    const auto entry = AddLabel();
    const auto available = OpLabel();
    const auto merge = OpLabel();
    const auto gpu_address = OpFunctionCall(U64, guest_page_pointer, address);
    const auto valid = OpINotEqual(U1[1], gpu_address, u64_zero_value);
    OpSelectionMerge(merge, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(valid, available, merge);
    AddLabel(available);
    const auto original =
        OpFunctionCall(U32[1], operation, OpConvertUToPtr(physical_pointer_type_u32, gpu_address),
                       operand, comparison);
    OpBranch(merge);
    AddLabel(merge);
    OpReturnValue(OpPhi(U32[1], u32_zero_value, entry, original, available));
    OpFunctionEnd();
    return function;
}

Id EmitSharedAtomicIAdd32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitSharedAtomicIAdd64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitSharedAtomicUMax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitSharedAtomicUMax64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitSharedAtomicSMax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitSharedAtomicSMax64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitSharedAtomicUMin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitSharedAtomicUMin64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitSharedAtomicSMin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitSharedAtomicSMin64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitSharedAtomicAnd32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitSharedAtomicAnd64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitSharedAtomicOr32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicOr);
}

Id EmitSharedAtomicOr64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicOr);
}

Id EmitSharedAtomicXor32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicXor);
}

Id EmitSharedAtomicXor64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicXor);
}

Id EmitSharedAtomicISub32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicISub);
}

Id EmitSharedAtomicISub64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicISub);
}

Id EmitSharedAtomicRSub32(EmitContext& ctx, Id offset, Id value) {
    return SharedDsAtomic(ctx, offset, value, {}, DsAtomicOperation::ReverseSubtract);
}

Id EmitSharedAtomicMaskOr32(EmitContext& ctx, Id offset, Id mask, Id value) {
    return SharedDsAtomic(ctx, offset, mask, value, DsAtomicOperation::MaskOr);
}

Id EmitSharedAtomicWrap32(EmitContext& ctx, Id offset, Id subtract, Id add) {
    return SharedDsAtomic(ctx, offset, subtract, add, DsAtomicOperation::Wrap);
}

Id EmitSharedAtomicExchange32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitBufferAtomicRSub32(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    return BufferDsAtomic(ctx, handle, address, value, {}, DsAtomicOperation::ReverseSubtract);
}

Id EmitBufferAtomicMaskOr32(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id mask,
                            Id value) {
    return BufferDsAtomic(ctx, handle, address, mask, value, DsAtomicOperation::MaskOr);
}

Id EmitBufferAtomicWrap32(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id subtract,
                          Id add) {
    return BufferDsAtomic(ctx, handle, address, subtract, add, DsAtomicOperation::Wrap);
}

Id EmitSharedAtomicRSub64(EmitContext& ctx, Id offset, Id value) {
    return SoftwareSharedDsAtomicU64(ctx, SharedU64Pointer(ctx, offset), value, ctx.u64_zero_value,
                                     DsAtomicOperation::ReverseSubtract);
}

Id EmitBufferAtomicRSub64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    return SoftwareSharedDsAtomicU64(ctx, BufferSharedU64Pointer(ctx, handle, address), value,
                                     ctx.u64_zero_value, DsAtomicOperation::ReverseSubtract);
}

Id EmitSharedAtomicMaskOr64(EmitContext& ctx, Id offset, Id mask, Id value) {
    return SoftwareSharedDsAtomicU64(ctx, SharedU64Pointer(ctx, offset), mask, value,
                                     DsAtomicOperation::MaskOr);
}

Id EmitBufferAtomicMaskOr64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id mask,
                            Id value) {
    return SoftwareSharedDsAtomicU64(ctx, BufferSharedU64Pointer(ctx, handle, address), mask, value,
                                     DsAtomicOperation::MaskOr);
}

Id EmitSharedAtomicIncBound64(EmitContext& ctx, Id offset, Id bound) {
    return SoftwareSharedCounterU64(ctx, SharedU64Pointer(ctx, offset), bound, false);
}

Id EmitBufferAtomicInc64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id bound) {
    return SoftwareSharedCounterU64(ctx, BufferSharedU64Pointer(ctx, handle, address), bound,
                                    false);
}

Id EmitSharedAtomicDecBound64(EmitContext& ctx, Id offset, Id bound) {
    return SoftwareSharedCounterU64(ctx, SharedU64Pointer(ctx, offset), bound, true);
}

Id EmitBufferAtomicDec64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id bound) {
    return SoftwareSharedCounterU64(ctx, BufferSharedU64Pointer(ctx, handle, address), bound, true);
}

void EmitSharedAtomicSrc2_32(EmitContext& ctx, Id address, Id source, u32 operation) {
    SharedSourceAtomic(ctx, address, source, 32, operation);
}

void EmitBufferAtomicSrc2_32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id source,
                             u32 operation) {
    BufferSourceAtomic(ctx, handle, address, source, 32, operation);
}

void EmitSharedAtomicSrc2_64(EmitContext& ctx, Id address, Id source, u32 operation) {
    SharedSourceAtomic(ctx, address, source, 64, operation);
}

void EmitBufferAtomicSrc2_64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id source,
                             u32 operation) {
    BufferSourceAtomic(ctx, handle, address, source, 64, operation);
}

Id EmitSharedAtomicExchange64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitBufferAtomicISub64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicISub);
}

Id EmitBufferAtomicAnd64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitBufferAtomicOr64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicOr);
}

Id EmitBufferAtomicXor64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicXor);
}

Id EmitBufferAtomicSwap64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitSharedAtomicFCmpSwap32(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id index = ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.ConstU32(2u));
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
    return FloatCmpSwapU32(ctx, pointer, value, cmp_value);
}

Id EmitSharedAtomicFCmpSwap64(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    ASSERT(ctx.info.uses_shared_memory_lock);
    return SoftwareSharedCmpSwapU64(ctx, SharedU64Pointer(ctx, offset), value, cmp_value, true);
}

Id EmitSharedAtomicCmpSwap32(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicU32CmpSwap(ctx, offset, value, cmp_value);
}

Id EmitSharedAtomicCmpSwap64(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicU64CmpSwap(ctx, offset, value, cmp_value);
}

Id EmitSharedAtomicInc32(EmitContext& ctx, Id offset, Id bound) {
    return SharedAtomicU32IncDec(ctx, offset, bound, false, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicInc64(EmitContext& ctx, Id offset) {
    return SharedAtomicU64IncDec(ctx, offset, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicDec32(EmitContext& ctx, Id offset, Id bound) {
    return SharedAtomicU32IncDec(ctx, offset, bound, true, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitSharedAtomicDec64(EmitContext& ctx, Id offset) {
    return SharedAtomicU64IncDec(ctx, offset, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicIAdd32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitBufferAtomicIAdd64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitBufferAtomicISub32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicISub32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicISub);
}

Id EmitBufferAtomicSMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicSMin32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitBufferAtomicSMin64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitBufferAtomicUMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicUMin32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitBufferAtomicUMin64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitBufferAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.OpBitcast(ctx.F32[1], ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicFMin32,
                                                                 handle, ctx.Def(guest_address),
                                                                 ctx.OpBitcast(ctx.U32[1], value),
                                                                 ctx.u32_zero_value));
    }
    if (ctx.profile.needs_software_float_atomics) {
        return BufferFloatAtomic(ctx, handle, address, value, FloatAtomicOperation::Min);
    }
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicU32<true>(ctx, inst, handle, address, value,
                                     &Sirit::Module::OpAtomicFMin);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    // FIXME this needs control flow because it currently executes both atomics
    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitBufferAtomicUMax32(ctx, inst, handle, address, u32_value, {})),
        EmitBitCastF32U32(ctx, EmitBufferAtomicSMin32(ctx, inst, handle, address, u32_value, {})));

    return result;
}

Id EmitBufferAtomicSMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicSMax32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitBufferAtomicSMax64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitBufferAtomicUMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicUMax32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitBufferAtomicUMax64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitBufferAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.OpBitcast(ctx.F32[1], ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicFMax32,
                                                                 handle, ctx.Def(guest_address),
                                                                 ctx.OpBitcast(ctx.U32[1], value),
                                                                 ctx.u32_zero_value));
    }
    if (ctx.profile.needs_software_float_atomics) {
        return BufferFloatAtomic(ctx, handle, address, value, FloatAtomicOperation::Max);
    }
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicU32<true>(ctx, inst, handle, address, value,
                                     &Sirit::Module::OpAtomicFMax);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    // FIXME this needs control flow because it currently executes both atomics
    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitBufferAtomicUMin32(ctx, inst, handle, address, u32_value, {})),
        EmitBitCastF32U32(ctx, EmitBufferAtomicSMax32(ctx, inst, handle, address, u32_value, {})));

    return result;
}

Id EmitBufferAtomicInc32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id bound,
                         const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicInc32, handle,
                                       ctx.Def(guest_address), bound, ctx.u32_zero_value);
    }
    return BufferAtomicU32IncDec(ctx, inst, handle, address, bound, false,
                                 &Sirit::Module::OpAtomicIIncrement);
}

Id EmitBufferAtomicDec32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id bound,
                         const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicDec32, handle,
                                       ctx.Def(guest_address), bound, ctx.u32_zero_value);
    }
    return BufferAtomicU32IncDec(ctx, inst, handle, address, bound, true,
                                 &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicAnd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                         const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicAnd32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitBufferAtomicOr32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                        const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicOr32, handle, ctx.Def(guest_address),
                                       value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicOr);
}

Id EmitBufferAtomicXor32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                         const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicXor32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicXor);
}

Id EmitBufferAtomicSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicSwap32, handle,
                                       ctx.Def(guest_address), value, ctx.u32_zero_value);
    }
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitBufferAtomicFCmpSwapRaw32(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value,
                                 Id cmp_value, const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicFCmpSwapRaw32, handle,
                                       ctx.Def(guest_address), value, cmp_value);
    }
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id pointer = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    return FloatCmpSwapU32(ctx, pointer, value, cmp_value,
                           buffer.buffer_type == BufferType::SharedMemory);
}

Id EmitBufferAtomicFCmpSwapRaw64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value,
                                 Id cmp_value) {
    ASSERT(ctx.info.uses_shared_memory_lock);
    return SoftwareSharedCmpSwapU64(ctx, BufferSharedU64Pointer(ctx, handle, address), value,
                                    cmp_value, true);
}

Id EmitBufferAtomicCmpSwap64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value,
                             Id cmp_value) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U64);
    const Id pointer = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    if (ctx.info.uses_shared_memory_lock && buffer.buffer_type == BufferType::SharedMemory) {
        return SoftwareSharedCmpSwapU64(ctx, pointer, value, cmp_value);
    }
    const auto [scope, semantics] = AtomicArgs(ctx);
    return ctx.OpAtomicCompareExchange(ctx.U64, pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

Id EmitBufferAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                             Id cmp_value, const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicCmpSwap32, handle,
                                       ctx.Def(guest_address), value, cmp_value);
    }
    return BufferAtomicU32CmpSwap(ctx, inst, handle, address, value, cmp_value,
                                  &Sirit::Module::OpAtomicCompareExchange);
}

Id EmitBufferAtomicFCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                              Id cmp_value, const IR::Value& guest_address) {
    if (!guest_address.IsEmpty()) {
        return ctx.OpBitcast(ctx.F32[1],
                             ctx.EmitGuestPageAtomic(IR::Opcode::BufferAtomicFCmpSwap32, handle,
                                                     ctx.Def(guest_address),
                                                     ctx.OpBitcast(ctx.U32[1], value),
                                                     ctx.OpBitcast(ctx.U32[1], cmp_value)));
    }
    if (ctx.profile.needs_software_float_atomics) {
        return BufferFloatAtomic(ctx, handle, address, value, FloatAtomicOperation::CompareSwap,
                                 ctx.OpBitcast(ctx.U32[1], cmp_value));
    }
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto u32_cmp = ctx.OpBitcast(ctx.U32[1], cmp_value);
    const auto result = BufferAtomicU32CmpSwap(ctx, inst, handle, address, u32_value, u32_cmp,
                                               &Sirit::Module::OpAtomicCompareExchange);
    return ctx.OpBitcast(ctx.F32[1], result);
}

Id EmitImageAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitImageAtomicSMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitImageAtomicUMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitImageAtomicSMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitImageAtomicUMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitImageAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.needs_software_float_atomics) {
        return ImageFloatAtomic(ctx, handle, coords, value, FloatAtomicOperation::Max);
    }
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        return ImageAtomicF32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicFMax);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitImageAtomicUMin32(ctx, inst, handle, coords, u32_value)),
        EmitBitCastF32U32(ctx, EmitImageAtomicSMax32(ctx, inst, handle, coords, u32_value)));

    return result;
}

Id EmitImageAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.needs_software_float_atomics) {
        return ImageFloatAtomic(ctx, handle, coords, value, FloatAtomicOperation::Min);
    }
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        return ImageAtomicF32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicFMin);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitImageAtomicUMax32(ctx, inst, handle, coords, u32_value)),
        EmitBitCastF32U32(ctx, EmitImageAtomicSMin32(ctx, inst, handle, coords, u32_value)));

    return result;
}

Id EmitImageAtomicInc32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (!ctx.profile.needs_bounded_atomic_counters) {
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
    return ImageAtomicU32IncDec(ctx, inst, handle, coords, value, false);
}

Id EmitImageAtomicDec32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (!ctx.profile.needs_bounded_atomic_counters) {
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
    return ImageAtomicU32IncDec(ctx, inst, handle, coords, value, true);
}

Id EmitImageAtomicAnd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitImageAtomicOr32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicOr);
}

Id EmitImageAtomicXor32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicXor);
}

Id EmitImageAtomicExchange32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitImageAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                            Id cmp_value) {
    return ImageAtomicU32CmpSwap(ctx, inst, handle, coords, value, cmp_value,
                                 &Sirit::Module::OpAtomicCompareExchange);
}

Id EmitImageAtomicFCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                             Id cmp_value) {
    ASSERT(ctx.profile.needs_software_float_atomics);
    return ImageFloatAtomic(ctx, handle, coords, value, FloatAtomicOperation::CompareSwap,
                            ctx.OpBitcast(ctx.U32[1], cmp_value));
}

Id EmitDataAppend(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitDataConsume(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

} // namespace Shader::Backend::SPIRV
