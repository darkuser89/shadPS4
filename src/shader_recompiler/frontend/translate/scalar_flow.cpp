// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/opcodes.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/profile.h"

namespace Shader::Gcn {

void Translator::EmitFlowControl(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_BARRIER:
        return S_BARRIER();
    case Opcode::S_WAITCNT:
        return S_WAITCNT(inst);
    case Opcode::S_TTRACEDATA:
        LOG_WARNING(Render_Vulkan, "S_TTRACEDATA instruction!");
        return;
    case Opcode::S_SETPRIO:
        LOG_WARNING(Render_Vulkan, "S_SETPRIO instruction!");
        return;
    case Opcode::S_TRAP:
        LOG_WARNING(Render_Vulkan, "S_TRAP instruction!");
        return;
    case Opcode::S_GETPC_B64:
        return S_GETPC_B64(inst);
    case Opcode::S_SETPC_B64:
    case Opcode::S_NOP:
    case Opcode::S_ENDPGM:
    case Opcode::S_CBRANCH_EXECZ:
    case Opcode::S_CBRANCH_SCC0:
    case Opcode::S_CBRANCH_SCC1:
    case Opcode::S_CBRANCH_VCCNZ:
    case Opcode::S_CBRANCH_VCCZ:
    case Opcode::S_CBRANCH_EXECNZ:
    case Opcode::S_BRANCH:
        return;
    case Opcode::S_SENDMSG:
        S_SENDMSG(inst);
        return;
    default:
        UNREACHABLE();
    }
}

void Translator::S_BARRIER() {
    ir.Barrier(
        profile.needs_guest_compute_barriers &&
                (info.sw_stage == SwStage::Compute ||
                 (profile.needs_geometry_output_lowering && info.sw_stage == SwStage::Geometry) ||
                 (profile.needs_geometry_copy_lowering && info.sw_stage == SwStage::Vertex) ||
                 (profile.needs_geometry_export_lowering && info.hw_stage == HwStage::Export))
            ? IR::MemoryClass::All
            : IR::MemoryClass::None);
}

void Translator::S_WAITCNT(const GcnInst& inst) {
    if (!profile.needs_guest_compute_barriers ||
        (info.sw_stage != SwStage::Compute &&
         !(profile.needs_geometry_output_lowering && info.sw_stage == SwStage::Geometry) &&
         !(profile.needs_geometry_copy_lowering && info.sw_stage == SwStage::Vertex) &&
         !(profile.needs_geometry_export_lowering && info.hw_stage == HwStage::Export))) {
        return;
    }
    const u16 counts = static_cast<u16>(inst.control.sopp.simm);
    IR::MemoryClass memory = IR::MemoryClass::None;
    if ((counts & 0xf) != 0xf) {
        memory |= IR::MemoryClass::Buffer | IR::MemoryClass::Image;
    }
    if (((counts >> 4) & 0x7) != 0x7) {
        memory |= IR::MemoryClass::Buffer | IR::MemoryClass::Image | IR::MemoryClass::Gds;
    }
    if (((counts >> 8) & 0xf) != 0xf) {
        memory |= IR::MemoryClass::Shared | IR::MemoryClass::Gds;
    }
    if (memory != IR::MemoryClass::None) {
        ir.DeviceMemoryBarrier(memory);
    }
}

void Translator::S_GETPC_B64(const GcnInst& inst) {
    const IR::ScalarReg dst{inst.dst[0].code};
    ir.SetScalarReg(dst, ir.GetPcLo(ir.Imm32(pc)));
    ir.SetScalarReg(dst + 1, ir.Imm32(0));
}

void Translator::S_SENDMSG(const GcnInst& inst) {
    if (profile.needs_geometry_output_lowering) {
        const u32 message = static_cast<u16>(inst.control.sopp.simm);
        const auto type = static_cast<SendMsgSimm::Message>(message & 0xf);
        if (type == SendMsgSimm::Message::Gs || type == SendMsgSimm::Message::GsDone) {
            const u32 stream = (message >> 8) & 3;
            if (message & 0x20) {
                if (profile.needs_geometry_message_predication) {
                    ir.EmitVertex(stream, ir.GetExec());
                } else {
                    ir.EmitVertex(stream);
                }
            }
            if (message & 0x10) {
                if (profile.needs_geometry_message_predication) {
                    ir.EmitPrimitive(stream, ir.GetExec());
                } else {
                    ir.EmitPrimitive(stream);
                }
            }
            return;
        }
    }
    const auto& simm = reinterpret_cast<const SendMsgSimm&>(inst.control.sopp.simm);
    switch (simm.msg) {
    case SendMsgSimm::Message::Gs: {
        switch (simm.op) {
        case SendMsgSimm::GsOp::Nop:
            break;
        case SendMsgSimm::GsOp::Cut:
            ir.EmitPrimitive();
            break;
        case SendMsgSimm::GsOp::Emit:
            ir.EmitVertex();
            break;
        default:
            UNREACHABLE();
        }
        break;
    }
    case SendMsgSimm::Message::GsDone: {
        break;
    }
    default:
        UNREACHABLE();
    }
}

} // namespace Shader::Gcn
