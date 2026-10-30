// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/profile.h"

namespace Shader::Gcn {

static constexpr u32 SQ_SRC_LITERAL = 0xFF;

void Translator::EmitScalarMemory(const GcnInst& inst) {
    switch (inst.opcode) {
        // SMRD
    case Opcode::S_LOAD_DWORD:
        return S_LOAD_DWORD(1, inst);
    case Opcode::S_LOAD_DWORDX2:
        return S_LOAD_DWORD(2, inst);
    case Opcode::S_LOAD_DWORDX4:
        return S_LOAD_DWORD(4, inst);
    case Opcode::S_LOAD_DWORDX8:
        return S_LOAD_DWORD(8, inst);
    case Opcode::S_LOAD_DWORDX16:
        return S_LOAD_DWORD(16, inst);
    case Opcode::S_BUFFER_LOAD_DWORD:
        return S_BUFFER_LOAD_DWORD(1, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX2:
        return S_BUFFER_LOAD_DWORD(2, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX4:
        return S_BUFFER_LOAD_DWORD(4, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX8:
        return S_BUFFER_LOAD_DWORD(8, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX16:
        return S_BUFFER_LOAD_DWORD(16, inst);
    case Opcode::S_MEMTIME:
        return S_MEMTIME(inst);
    default:
        LogMissingOpcode(inst);
    }
}

// SMRD

IR::U32 Translator::ScalarMemoryDwordOffset(const GcnInst& inst, IR::ScalarReg sbase) {
    const auto& smrd = inst.control.smrd;
    if (smrd.imm) {
        return ir.Imm32(smrd.offset);
    }
    if (smrd.offset == SQ_SRC_LITERAL) {
        return ir.Imm32(inst.src[1].code);
    }
    const auto byte_offset = ir.GetScalarReg(IR::ScalarReg(smrd.offset));
    const IR::U32 dword_offset = ir.ShiftRightLogical(byte_offset, ir.Imm32(2));
    if (!profile.needs_wide_buffer_byte_offsets) {
        return dword_offset;
    }
    const auto low_bits = ir.IAdd(ir.BitwiseAnd(ir.GetScalarReg(sbase), ir.Imm32(3)),
                                  ir.BitwiseAnd(byte_offset, ir.Imm32(3)));
    return ir.IAdd(dword_offset, ir.ShiftRightLogical(low_bits, ir.Imm32(2)));
}

void Translator::S_LOAD_DWORD(int num_dwords, const GcnInst& inst) {
    const IR::ScalarReg sbase{inst.src[0].code * 2};
    const IR::U32 dword_offset = ScalarMemoryDwordOffset(inst, sbase);
    const IR::Value base =
        ir.CompositeConstruct(ir.GetScalarReg(sbase), ir.GetScalarReg(sbase + 1));
    IR::ScalarReg dst_reg{inst.dst[0].code};
    for (u32 i = 0; i < num_dwords; i++) {
        IR::U32 index = ir.IAdd(dword_offset, ir.Imm32(i));
        ir.SetScalarReg(dst_reg + i, ir.ReadConst(base, index));
    }
}

void Translator::S_BUFFER_LOAD_DWORD(int num_dwords, const GcnInst& inst) {
    const IR::ScalarReg sbase{inst.src[0].code * 2};
    const IR::U32 dword_offset = ScalarMemoryDwordOffset(inst, sbase);

    IR::BufferInstInfo buffer_info{};
    buffer_info.pc.Assign(pc);

    const IR::Value vsharp =
        ir.CompositeConstruct(ir.GetScalarReg(sbase), ir.GetScalarReg(sbase + 1),
                              ir.GetScalarReg(sbase + 2), ir.GetScalarReg(sbase + 3));
    IR::ScalarReg dst_reg{inst.dst[0].code};
    for (u32 i = 0; i < num_dwords; i++) {
        const IR::U32 index = ir.IAdd(dword_offset, ir.Imm32(i));
        ir.SetScalarReg(dst_reg + i, ir.ReadConstBuffer(vsharp, index, buffer_info));
    }
}

void Translator::S_MEMTIME(const GcnInst& inst) {
    SetDst64(inst.dst[0], ir.Memtime());
}

} // namespace Shader::Gcn
