// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Gcn {

void Translator::EmitDataShare(const GcnInst& inst) {
    if (profile.needs_software_float_atomics && profile.needs_software_shared_int64_atomics) {
        switch (inst.opcode) {
        case Opcode::DS_CMPST_F32:
            return DS_CMPST(32, false, inst, true);
        case Opcode::DS_CMPST_RTN_F32:
            return DS_CMPST(32, true, inst, true);
        case Opcode::DS_CMPST_F64:
            return DS_CMPST(64, false, inst, true);
        case Opcode::DS_CMPST_RTN_F64:
            return DS_CMPST(64, true, inst, true);
        default:
            break;
        }
    }
    if (profile.needs_software_shared_int64_atomics) {
        switch (inst.opcode) {
        case Opcode::DS_ADD_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Add);
        case Opcode::DS_SUB_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Subtract);
        case Opcode::DS_RSUB_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::ReverseSubtract);
        case Opcode::DS_INC_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Increment);
        case Opcode::DS_DEC_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Decrement);
        case Opcode::DS_MIN_SRC2_I32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::SignedMin);
        case Opcode::DS_MAX_SRC2_I32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::SignedMax);
        case Opcode::DS_MIN_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::UnsignedMin);
        case Opcode::DS_MAX_SRC2_U32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::UnsignedMax);
        case Opcode::DS_AND_SRC2_B32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::And);
        case Opcode::DS_OR_SRC2_B32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Or);
        case Opcode::DS_XOR_SRC2_B32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Xor);
        case Opcode::DS_WRITE_SRC2_B32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::Write);
        case Opcode::DS_MIN_SRC2_F32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::FloatMin);
        case Opcode::DS_MAX_SRC2_F32:
            return DS_SRC2(inst, 32, IR::SharedMemoryOperation::FloatMax);
        case Opcode::DS_ADD_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Add);
        case Opcode::DS_SUB_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Subtract);
        case Opcode::DS_RSUB_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::ReverseSubtract);
        case Opcode::DS_INC_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Increment);
        case Opcode::DS_DEC_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Decrement);
        case Opcode::DS_MIN_SRC2_I64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::SignedMin);
        case Opcode::DS_MAX_SRC2_I64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::SignedMax);
        case Opcode::DS_MIN_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::UnsignedMin);
        case Opcode::DS_MAX_SRC2_U64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::UnsignedMax);
        case Opcode::DS_AND_SRC2_B64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::And);
        case Opcode::DS_OR_SRC2_B64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Or);
        case Opcode::DS_XOR_SRC2_B64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Xor);
        case Opcode::DS_WRITE_SRC2_B64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::Write);
        case Opcode::DS_MIN_SRC2_F64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::FloatMin);
        case Opcode::DS_MAX_SRC2_F64:
            return DS_SRC2(inst, 64, IR::SharedMemoryOperation::FloatMax);
        case Opcode::DS_ADD_RTN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Add, true);
        case Opcode::DS_SUB_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Sub, false);
        case Opcode::DS_SUB_RTN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Sub, true);
        case Opcode::DS_INC_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Inc, false);
        case Opcode::DS_INC_RTN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Inc, true);
        case Opcode::DS_DEC_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Dec, false);
        case Opcode::DS_DEC_RTN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Dec, true);
        case Opcode::DS_MIN_I64:
            return DS_OP<IR::U64>(inst, AtomicOp::Smin, false);
        case Opcode::DS_MIN_RTN_I64:
            return DS_OP<IR::U64>(inst, AtomicOp::Smin, true);
        case Opcode::DS_MAX_I64:
            return DS_OP<IR::U64>(inst, AtomicOp::Smax, false);
        case Opcode::DS_MAX_RTN_I64:
            return DS_OP<IR::U64>(inst, AtomicOp::Smax, true);
        case Opcode::DS_MIN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Umin, false);
        case Opcode::DS_MIN_RTN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Umin, true);
        case Opcode::DS_MAX_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Umax, false);
        case Opcode::DS_MAX_RTN_U64:
            return DS_OP<IR::U64>(inst, AtomicOp::Umax, true);
        case Opcode::DS_AND_B64:
            return DS_OP<IR::U64>(inst, AtomicOp::And, false);
        case Opcode::DS_AND_RTN_B64:
            return DS_OP<IR::U64>(inst, AtomicOp::And, true);
        case Opcode::DS_OR_B64:
            return DS_OP<IR::U64>(inst, AtomicOp::Or, false);
        case Opcode::DS_OR_RTN_B64:
            return DS_OP<IR::U64>(inst, AtomicOp::Or, true);
        case Opcode::DS_XOR_B64:
            return DS_OP<IR::U64>(inst, AtomicOp::Xor, false);
        case Opcode::DS_XOR_RTN_B64:
            return DS_OP<IR::U64>(inst, AtomicOp::Xor, true);
        case Opcode::DS_RSUB_U64:
            return DS_RMW64(inst, false);
        case Opcode::DS_RSUB_RTN_U64:
            return DS_RMW64(inst, true);
        case Opcode::DS_MSKOR_B64:
            return DS_RMW64(inst, false);
        case Opcode::DS_MSKOR_RTN_B64:
            return DS_RMW64(inst, true);
        case Opcode::DS_WRXCHG_RTN_B64:
            return DS_WRXCHG64(inst, false, false);
        case Opcode::DS_WRXCHG2_RTN_B64:
            return DS_WRXCHG64(inst, true, false);
        case Opcode::DS_WRXCHG2ST64_RTN_B64:
            return DS_WRXCHG64(inst, true, true);
        default:
            break;
        }
    }
    if (profile.needs_extended_ds_atomics) {
        switch (inst.opcode) {
        case Opcode::DS_RSUB_U32:
            return DS_RMW32(inst, false);
        case Opcode::DS_RSUB_RTN_U32:
        case Opcode::DS_MSKOR_RTN_B32:
        case Opcode::DS_WRAP_RTN_B32:
            return DS_RMW32(inst, true);
        case Opcode::DS_MSKOR_B32:
            return DS_RMW32(inst, false);
        case Opcode::DS_MIN_RTN_I32:
            return DS_OP(inst, AtomicOp::Smin, true);
        case Opcode::DS_MAX_RTN_I32:
            return DS_OP(inst, AtomicOp::Smax, true);
        case Opcode::DS_WRXCHG_RTN_B32:
            return DS_WRXCHG32(inst, false, false);
        case Opcode::DS_WRXCHG2_RTN_B32:
            return DS_WRXCHG32(inst, true, false);
        case Opcode::DS_WRXCHG2ST64_RTN_B32:
            return DS_WRXCHG32(inst, true, true);
        default:
            break;
        }
    }
    switch (inst.opcode) {
        // DS
    case Opcode::DS_ADD_U32:
        return DS_OP(inst, AtomicOp::Add, false);
    case Opcode::DS_ADD_U64:
        return DS_OP<IR::U64>(inst, AtomicOp::Add, false);
    case Opcode::DS_SUB_U32:
        return DS_OP(inst, AtomicOp::Sub, false);
    case Opcode::DS_INC_U32:
        return DS_OP(inst, AtomicOp::Inc, false);
    case Opcode::DS_DEC_U32:
        return DS_OP(inst, AtomicOp::Dec, false);
    case Opcode::DS_MIN_I32:
        return DS_OP(inst, AtomicOp::Smin, false);
    case Opcode::DS_MAX_I32:
        return DS_OP(inst, AtomicOp::Smax, false);
    case Opcode::DS_MIN_U32:
        return DS_OP(inst, AtomicOp::Umin, false);
    case Opcode::DS_MAX_U32:
        return DS_OP(inst, AtomicOp::Umax, false);
    case Opcode::DS_AND_B32:
        return DS_OP(inst, AtomicOp::And, false);
    case Opcode::DS_OR_B32:
        return DS_OP(inst, AtomicOp::Or, false);
    case Opcode::DS_XOR_B32:
        return DS_OP(inst, AtomicOp::Xor, false);
    case Opcode::DS_WRITE_B32:
        return DS_WRITE(32, false, false, false, inst);
    case Opcode::DS_WRITE2_B32:
        return DS_WRITE(32, false, true, false, inst);
    case Opcode::DS_WRITE2ST64_B32:
        return DS_WRITE(32, false, true, true, inst);
    case Opcode::DS_ADD_RTN_U32:
        return DS_OP(inst, AtomicOp::Add, true);
    case Opcode::DS_SUB_RTN_U32:
        return DS_OP(inst, AtomicOp::Sub, true);
    case Opcode::DS_INC_RTN_U32:
        if (!profile.needs_bounded_atomic_counters) {
            return LogMissingOpcode(inst);
        }
        return DS_OP(inst, AtomicOp::Inc, true);
    case Opcode::DS_DEC_RTN_U32:
        if (!profile.needs_bounded_atomic_counters) {
            return LogMissingOpcode(inst);
        }
        return DS_OP(inst, AtomicOp::Dec, true);
    case Opcode::DS_MIN_RTN_U32:
        return DS_OP(inst, AtomicOp::Umin, true);
    case Opcode::DS_MAX_RTN_U32:
        return DS_OP(inst, AtomicOp::Umax, true);
    case Opcode::DS_AND_RTN_B32:
        return DS_OP(inst, AtomicOp::And, true);
    case Opcode::DS_OR_RTN_B32:
        return DS_OP(inst, AtomicOp::Or, true);
    case Opcode::DS_XOR_RTN_B32:
        return DS_OP(inst, AtomicOp::Xor, true);
    case Opcode::DS_CMPST_B32:
        return DS_CMPST(32, false, inst);
    case Opcode::DS_CMPST_RTN_B32:
        return DS_CMPST(32, true, inst);
    case Opcode::DS_CMPST_B64:
        return DS_CMPST(64, false, inst);
    case Opcode::DS_CMPST_RTN_B64:
        return DS_CMPST(64, true, inst);
    case Opcode::DS_SWIZZLE_B32:
        return DS_SWIZZLE_B32(inst);
    case Opcode::DS_READ_B32:
        return DS_READ(32, false, false, false, inst);
    case Opcode::DS_READ2_B32:
        return DS_READ(32, false, true, false, inst);
    case Opcode::DS_READ2ST64_B32:
        return DS_READ(32, false, true, true, inst);
    case Opcode::DS_READ_U16:
        return DS_READ(16, false, false, false, inst);
    case Opcode::DS_CONSUME:
        return DS_CONSUME(inst);
    case Opcode::DS_APPEND:
        return DS_APPEND(inst);
    case Opcode::DS_WRITE_B16:
        return DS_WRITE(16, false, false, false, inst);
    case Opcode::DS_WRITE_B64:
        return DS_WRITE(64, false, false, false, inst);
    case Opcode::DS_WRITE2_B64:
        return DS_WRITE(64, false, true, false, inst);
    case Opcode::DS_WRITE2ST64_B64:
        return DS_WRITE(64, false, true, true, inst);
    case Opcode::DS_READ_B64:
        return DS_READ(64, false, false, false, inst);
    case Opcode::DS_READ2_B64:
        return DS_READ(64, false, true, false, inst);
    case Opcode::DS_READ2ST64_B64:
        return DS_READ(64, false, true, true, inst);
    default:
        LogMissingOpcode(inst);
    }
}

// VOP2

void Translator::V_READFIRSTLANE_B32(const GcnInst& inst) {
    const IR::U32 value{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.ReadFirstLane(value));
}

void Translator::V_READLANE_B32(const GcnInst& inst) {
    ASSERT(inst.src[0].field == OperandField::VectorGPR);
    const IR::U32 value{GetSrc(inst.src[0])};
    const IR::U32 lane{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ReadLane(value, lane));
}

void Translator::V_WRITELANE_B32(const GcnInst& inst) {
    const IR::VectorReg dst{inst.dst[0].code};
    const IR::U32 value{GetSrc(inst.src[0])};
    const IR::U32 lane{GetSrc(inst.src[1])};
    const IR::U32 old_value{GetSrc(inst.dst[0])};
    ir.SetVectorReg(dst, ir.WriteLane(old_value, value, lane));
}

// DS

template <typename T>
void Translator::DS_OP(const GcnInst& inst, AtomicOp op, bool rtn) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{GetSrc(inst.src[0])};
    const T data = [&] {
        if ((op == AtomicOp::Inc || op == AtomicOp::Dec) &&
            !profile.needs_bounded_atomic_counters) {
            return T{};
        }
        if constexpr (std::is_same_v<T, IR::U32>) {
            return GetSrc(inst.src[1]);
        } else {
            return GetSrc64(inst.src[1]);
        }
    }();
    const IR::U32 offset =
        ir.Imm32((u32(inst.control.ds.offset1) << 8u) + u32(inst.control.ds.offset0));
    const IR::U32 addr_offset = ir.IAdd(addr, offset);
    const T original_val = [&] -> T {
        switch (op) {
        case AtomicOp::Add:
            return ir.SharedAtomicIAdd(addr_offset, data, is_gds);
        case AtomicOp::Umin:
            return ir.SharedAtomicIMin(addr_offset, data, false, is_gds);
        case AtomicOp::Smin:
            return ir.SharedAtomicIMin(addr_offset, data, true, is_gds);
        case AtomicOp::Umax:
            return ir.SharedAtomicIMax(addr_offset, data, false, is_gds);
        case AtomicOp::Smax:
            return ir.SharedAtomicIMax(addr_offset, data, true, is_gds);
        case AtomicOp::And:
            return ir.SharedAtomicAnd(addr_offset, data, is_gds);
        case AtomicOp::Or:
            return ir.SharedAtomicOr(addr_offset, data, is_gds);
        case AtomicOp::Xor:
            return ir.SharedAtomicXor(addr_offset, data, is_gds);
        case AtomicOp::Sub:
            return ir.SharedAtomicISub(addr_offset, data, is_gds);
        case AtomicOp::Inc:
            if constexpr (std::is_same_v<T, IR::U32>) {
                if (profile.needs_bounded_atomic_counters) {
                    return ir.SharedAtomicInc(addr_offset, data, is_gds);
                }
            } else {
                if (profile.needs_software_shared_int64_atomics) {
                    return ir.SharedAtomicInc(addr_offset, data, is_gds);
                }
            }
            return ir.SharedAtomicInc<T>(addr_offset, is_gds);
        case AtomicOp::Dec:
            if constexpr (std::is_same_v<T, IR::U32>) {
                if (profile.needs_bounded_atomic_counters) {
                    return ir.SharedAtomicDec(addr_offset, data, is_gds);
                }
            } else {
                if (profile.needs_software_shared_int64_atomics) {
                    return ir.SharedAtomicDec(addr_offset, data, is_gds);
                }
            }
            return ir.SharedAtomicDec<T>(addr_offset, is_gds);
        default:
            UNREACHABLE();
        }
    }();
    if (rtn) {
        if constexpr (std::is_same_v<T, IR::U32>) {
            SetDst(inst.dst[0], original_val);
        } else {
            SetDst64(inst.dst[0], original_val);
        }
    }
}

void Translator::DS_SRC2(const GcnInst& inst, u32 bit_size, IR::SharedMemoryOperation operation) {
    const auto& ds = inst.control.ds;
    const IR::U32 address{GetSrc(inst.src[0])};
    const IR::U32 displacement =
        (ds.offset1 & 0x80) != 0
            ? IR::U32{ir.ShiftRightArithmetic(address, ir.Imm32(17u))}
            : ir.Imm32(static_cast<u32>(
                  static_cast<s32>(((u32(ds.offset1) << 8u) | u32(ds.offset0)) ^ 0x4000u) -
                  0x4000));
    const IR::U32 source = ir.IAdd(address, ir.ShiftLeftLogical(displacement, ir.Imm32(2u)));
    ir.SharedAtomicSrc2(bit_size, address, source, operation, ds.gds);
}

void Translator::DS_RMW32(const GcnInst& inst, bool rtn) {
    const auto& ds = inst.control.ds;
    const IR::U32 address =
        ir.IAdd(IR::U32{GetSrc(inst.src[0])}, ir.Imm32((u32(ds.offset1) << 8u) + u32(ds.offset0)));
    const IR::U32 data0{GetSrc(inst.src[1])};
    const IR::U32 original = [&] {
        switch (inst.opcode) {
        case Opcode::DS_RSUB_U32:
        case Opcode::DS_RSUB_RTN_U32:
            return ir.SharedAtomicRSub(address, data0, ds.gds);
        case Opcode::DS_MSKOR_B32:
        case Opcode::DS_MSKOR_RTN_B32:
            return ir.SharedAtomicMaskOr(address, data0, IR::U32{GetSrc(inst.src[2])}, ds.gds);
        case Opcode::DS_WRAP_RTN_B32:
            return ir.SharedAtomicWrap(address, data0, IR::U32{GetSrc(inst.src[2])}, ds.gds);
        default:
            UNREACHABLE();
        }
    }();
    if (rtn) {
        SetDst(inst.dst[0], original);
    }
}

void Translator::DS_WRXCHG32(const GcnInst& inst, bool pair, bool stride64) {
    const auto& ds = inst.control.ds;
    const IR::U32 address{GetSrc(inst.src[0])};
    const IR::U32 data0{GetSrc(inst.src[1])};
    const IR::U32 data1 = pair ? IR::U32{GetSrc(inst.src[2])} : IR::U32{};
    const u32 offset0 =
        pair ? u32(ds.offset0) * (stride64 ? 256u : 4u) : (u32(ds.offset1) << 8u) + u32(ds.offset0);
    const IR::U32 original0 =
        ir.SharedAtomicExchange(ir.IAdd(address, ir.Imm32(offset0)), data0, ds.gds);
    if (pair) {
        const u32 offset1 = u32(ds.offset1) * (stride64 ? 256u : 4u);
        const IR::U32 original1 =
            ir.SharedAtomicExchange(ir.IAdd(address, ir.Imm32(offset1)), data1, ds.gds);
        const IR::VectorReg dst{inst.dst[0].code};
        ir.SetVectorReg(dst, original0);
        ir.SetVectorReg(dst + 1, original1);
    } else {
        SetDst(inst.dst[0], original0);
    }
}

void Translator::DS_RMW64(const GcnInst& inst, bool rtn) {
    const auto& ds = inst.control.ds;
    const IR::U32 address =
        ir.IAdd(IR::U32{GetSrc(inst.src[0])}, ir.Imm32((u32(ds.offset1) << 8u) + u32(ds.offset0)));
    const IR::U64 data0{GetSrc64(inst.src[1])};
    const IR::U64 original = [&] {
        switch (inst.opcode) {
        case Opcode::DS_RSUB_U64:
        case Opcode::DS_RSUB_RTN_U64:
            return ir.SharedAtomicRSub(address, data0, ds.gds);
        case Opcode::DS_MSKOR_B64:
        case Opcode::DS_MSKOR_RTN_B64:
            return ir.SharedAtomicMaskOr(address, data0, IR::U64{GetSrc64(inst.src[2])}, ds.gds);
        default:
            UNREACHABLE();
        }
    }();
    if (rtn) {
        SetDst64(inst.dst[0], original);
    }
}

void Translator::DS_WRXCHG64(const GcnInst& inst, bool pair, bool stride64) {
    const auto& ds = inst.control.ds;
    const IR::U32 address{GetSrc(inst.src[0])};
    const IR::U64 data0{GetSrc64(inst.src[1])};
    const IR::U64 data1 = pair ? IR::U64{GetSrc64(inst.src[2])} : IR::U64{};
    const u32 offset0 =
        pair ? u32(ds.offset0) * (stride64 ? 512u : 8u) : (u32(ds.offset1) << 8u) + u32(ds.offset0);
    const IR::U64 original0 =
        ir.SharedAtomicExchange(ir.IAdd(address, ir.Imm32(offset0)), data0, ds.gds);
    if (pair) {
        const u32 offset1 = u32(ds.offset1) * (stride64 ? 512u : 8u);
        const IR::U64 original1 =
            ir.SharedAtomicExchange(ir.IAdd(address, ir.Imm32(offset1)), data1, ds.gds);
        const IR::VectorReg dst{inst.dst[0].code};
        const auto result0 = ir.UnpackUint2x32(original0);
        const auto result1 = ir.UnpackUint2x32(original1);
        ir.SetVectorReg(dst, IR::U32{ir.CompositeExtract(result0, 0)});
        ir.SetVectorReg(dst + 1, IR::U32{ir.CompositeExtract(result0, 1)});
        ir.SetVectorReg(dst + 2, IR::U32{ir.CompositeExtract(result1, 0)});
        ir.SetVectorReg(dst + 3, IR::U32{ir.CompositeExtract(result1, 1)});
    } else {
        SetDst64(inst.dst[0], original0);
    }
}

void Translator::DS_CMPST(int bit_size, bool rtn, const GcnInst& inst, bool is_float) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{GetSrc(inst.src[0])};
    const IR::U32 offset =
        ir.Imm32((u32(inst.control.ds.offset1) << 8u) + u32(inst.control.ds.offset0));
    const IR::U32 addr_offset = ir.IAdd(addr, offset);
    if (bit_size == 64) {
        const IR::U64 cmp_value{GetSrc64(inst.src[1])};
        const IR::U64 write_value{GetSrc64(inst.src[2])};
        const IR::U64 original_val =
            is_float ? ir.SharedAtomicFCmpSwap(addr_offset, write_value, cmp_value, is_gds)
                     : ir.SharedAtomicCmpSwap(addr_offset, write_value, cmp_value, is_gds);
        if (rtn) {
            SetDst64(inst.dst[0], original_val);
        }
    } else {
        const IR::U32 cmp_value{GetSrc(inst.src[1])};
        const IR::U32 write_value{GetSrc(inst.src[2])};
        const IR::U32 original_val =
            is_float ? ir.SharedAtomicFCmpSwap(addr_offset, write_value, cmp_value, is_gds)
                     : ir.SharedAtomicCmpSwap(addr_offset, write_value, cmp_value, is_gds);
        if (rtn) {
            SetDst(inst.dst[0], original_val);
        }
    }
}

void Translator::DS_WRITE(int bit_size, bool is_signed, bool is_pair, bool stride64,
                          const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{ir.GetVectorReg(IR::VectorReg(inst.src[0].code))};
    const IR::VectorReg data0{inst.src[1].code};
    const IR::VectorReg data1{inst.src[2].code};
    const u32 offset = (inst.control.ds.offset1 << 8u) + inst.control.ds.offset0;
    if (info.hw_stage == HwStage::Fragment) {
        ASSERT_MSG(!is_pair && bit_size == 32 && offset % 256 == 0,
                   "Unexpected shared memory offset alignment: {}", offset);
        ir.SetVectorReg(GetScratchVgpr(offset), ir.GetVectorReg(data0));
        return;
    }
    if (is_pair) {
        const u32 adj = (bit_size == 32 ? 4 : 8) * (stride64 ? 64 : 1);
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset0 * adj)));
        if (bit_size == 64) {
            ir.WriteShared(64,
                           ir.PackUint2x32(ir.CompositeConstruct(ir.GetVectorReg(data0),
                                                                 ir.GetVectorReg(data0 + 1))),
                           addr0, is_gds);
        } else if (bit_size == 32) {
            ir.WriteShared(32, ir.GetVectorReg(data0), addr0, is_gds);
        } else if (bit_size == 16) {
            ir.WriteShared(16, ir.UConvert(16, ir.GetVectorReg(data0)), addr0, is_gds);
        }
        const IR::U32 addr1 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset1 * adj)));
        if (bit_size == 64) {
            ir.WriteShared(64,
                           ir.PackUint2x32(ir.CompositeConstruct(ir.GetVectorReg(data1),
                                                                 ir.GetVectorReg(data1 + 1))),
                           addr1, is_gds);
        } else if (bit_size == 32) {
            ir.WriteShared(32, ir.GetVectorReg(data1), addr1, is_gds);
        } else if (bit_size == 16) {
            ir.WriteShared(16, ir.UConvert(16, ir.GetVectorReg(data1)), addr1, is_gds);
        }
    } else {
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(offset));
        if (bit_size == 64) {
            const IR::Value data =
                ir.CompositeConstruct(ir.GetVectorReg(data0), ir.GetVectorReg(data0 + 1));
            ir.WriteShared(bit_size, ir.PackUint2x32(data), addr0, is_gds);
        } else if (bit_size == 32) {
            ir.WriteShared(bit_size, ir.GetVectorReg(data0), addr0, is_gds);
        } else if (bit_size == 16) {
            ir.WriteShared(bit_size, ir.UConvert(16, ir.GetVectorReg(data0)), addr0, is_gds);
        }
    }
}

void Translator::DS_READ(int bit_size, bool is_signed, bool is_pair, bool stride64,
                         const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{ir.GetVectorReg(IR::VectorReg(inst.src[0].code))};
    IR::VectorReg dst_reg{inst.dst[0].code};
    const u32 offset = (inst.control.ds.offset1 << 8u) + inst.control.ds.offset0;
    if (info.hw_stage == HwStage::Fragment) {
        ASSERT_MSG(!is_pair && bit_size == 32 && offset % 256 == 0,
                   "Unexpected shared memory offset alignment: {}", offset);
        ir.SetVectorReg(dst_reg, ir.GetVectorReg(GetScratchVgpr(offset)));
        return;
    }
    if (is_pair) {
        // Pair loads are either 32 or 64-bit
        const u32 adj = (bit_size == 32 ? 4 : 8) * (stride64 ? 64 : 1);
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset0 * adj)));
        const IR::Value data0 = ir.LoadShared(bit_size, is_signed, addr0, is_gds);
        if (bit_size == 64) {
            const auto vector = ir.UnpackUint2x32(IR::U64{data0});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 0)});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 1)});
        } else if (bit_size == 32) {
            ir.SetVectorReg(dst_reg++, IR::U32{data0});
        } else if (bit_size == 16) {
            ir.SetVectorReg(dst_reg++, IR::U32{ir.UConvert(32, IR::U16{data0})});
        }
        const IR::U32 addr1 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset1 * adj)));
        const IR::Value data1 = ir.LoadShared(bit_size, is_signed, addr1, is_gds);
        if (bit_size == 64) {
            const auto vector = ir.UnpackUint2x32(IR::U64{data1});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 0)});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 1)});
        } else if (bit_size == 32) {
            ir.SetVectorReg(dst_reg++, IR::U32{data1});
        } else if (bit_size == 16) {
            ir.SetVectorReg(dst_reg++, IR::U32{ir.UConvert(32, IR::U16{data1})});
        }
    } else {
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(offset));
        const IR::Value data = ir.LoadShared(bit_size, is_signed, addr0, is_gds);
        if (bit_size == 64) {
            const auto vector = ir.UnpackUint2x32(IR::U64{data});
            ir.SetVectorReg(dst_reg, IR::U32{ir.CompositeExtract(vector, 0)});
            ir.SetVectorReg(dst_reg + 1, IR::U32{ir.CompositeExtract(vector, 1)});
        } else if (bit_size == 32) {
            ir.SetVectorReg(dst_reg, IR::U32{data});
        } else if (bit_size == 16) {
            ir.SetVectorReg(dst_reg++, IR::U32{ir.UConvert(32, IR::U16{data})});
        }
    }
}

void Translator::DS_SWIZZLE_B32(const GcnInst& inst) {
    const u8 offset0 = inst.control.ds.offset0;
    const u8 offset1 = inst.control.ds.offset1;
    const IR::U32 src{GetSrc(inst.src[0])};
    if (offset1 & 0x80) {
        if (offset0 == 0x0 || offset0 == 0x55 || offset0 == 0xAA || offset0 == 0xFF) {
            SetDst(inst.dst[0], ir.QuadBroadcast(src, ir.Imm32(offset0 & 3)));
        } else {
            const IR::U32 lane_id = ir.LaneId();
            const IR::U32 id_in_quad = ir.BitwiseAnd(lane_id, ir.Imm32(3));
            const IR::U32 base = ir.ShiftLeftLogical(id_in_quad, ir.Imm32(1));
            const IR::U32 sel = ir.BitFieldExtract(ir.Imm32(offset0), base, ir.Imm32(2));
            const IR::U32 quad_base = ir.BitwiseAnd(lane_id, ir.Imm32(~3u));
            SetDst(inst.dst[0], ir.Shuffle(src, ir.BitwiseOr(quad_base, sel)));
        }
    } else {
        const u8 and_mask = (offset0 & 0x1f) | (~u8{0} << 5);
        const u8 or_mask = (offset0 >> 5) | ((offset1 & 0x3) << 3);
        const u8 xor_mask = offset1 >> 2;
        if (or_mask == 0u && and_mask == 255u) {
            SetDst(inst.dst[0], ir.ShuffleXor(src, ir.Imm32(xor_mask)));
        } else {
            const IR::U32 lane_id = ir.LaneId();
            const IR::U32 local = ir.BitwiseAnd(lane_id, ir.Imm32(31u));
            const IR::U32 half = ir.BitwiseAnd(lane_id, ir.Imm32(~31u));
            const IR::U32 index = ir.BitwiseXor(
                ir.BitwiseOr(ir.BitwiseAnd(local, ir.Imm32(and_mask)), ir.Imm32(or_mask)),
                ir.Imm32(xor_mask));
            SetDst(inst.dst[0], ir.Shuffle(src, ir.BitwiseOr(index, half)));
        }
    }
}

void Translator::DS_APPEND(const GcnInst& inst) {
    const u32 inst_offset = (u32(inst.control.ds.offset1) << 8u) + inst.control.ds.offset0;
    const IR::U32 base = ir.BitFieldExtract(ir.GetM0(), ir.Imm32(16), ir.Imm32(16));
    const IR::U32 gds_offset = ir.IAdd(base, ir.Imm32(inst_offset));
    const IR::U32 prev = ir.DataAppend(ir.ShiftRightLogical(gds_offset, ir.Imm32(2u)));
    SetDst(inst.dst[0], prev);
}

void Translator::DS_CONSUME(const GcnInst& inst) {
    const u32 inst_offset = (u32(inst.control.ds.offset1) << 8u) + inst.control.ds.offset0;
    const IR::U32 base = ir.BitFieldExtract(ir.GetM0(), ir.Imm32(16), ir.Imm32(16));
    const IR::U32 gds_offset = ir.IAdd(base, ir.Imm32(inst_offset));
    const IR::U32 prev = ir.DataConsume(ir.ShiftRightLogical(gds_offset, ir.Imm32(2u)));
    SetDst(inst.dst[0], prev);
}

} // namespace Shader::Gcn
