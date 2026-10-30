// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <boost/container/flat_map.hpp>
#include <queue>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#include "common/arch.h"
#include "common/decoder.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "core/emulator_settings.h"
#include "core/signals.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/passes/resource_pass.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/srt_gvn_table.h"
#include "shader_recompiler/ir/value.h"

#if defined(__APPLE__) && defined(ARCH_X86_64)
#include <sys/ucontext.h>
#endif

#ifdef ARCH_X86_64

using namespace Xbyak::util;

static constexpr size_t SrtCodePoolSize = 32_MB;
static Xbyak::CodeGenerator g_srt_codegen(SrtCodePoolSize);
static const u8* g_srt_codegen_start = nullptr;

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    const auto func_addr = (PFN_SrtWalker)g_srt_codegen.getCurr();
    g_srt_codegen.db(ptr, size);
    g_srt_codegen.ready();
    return func_addr;
}

} // namespace Shader

namespace {

struct NativeSrtRange;
static std::atomic<NativeSrtRange*> g_native_srt_ranges{};
static std::atomic<u32> g_native_srt_readers{};
static std::mutex g_native_srt_mutex;
static std::once_flag g_srt_signal_registration;
static_assert(std::atomic<NativeSrtRange*>::is_always_lock_free);
static_assert(std::atomic<u32>::is_always_lock_free);

struct NativeSrtRange {
    NativeSrtRange(const u8* code, size_t size_)
        : start{reinterpret_cast<uintptr_t>(code)}, size{size_} {
        std::scoped_lock lock{g_native_srt_mutex};
        next.store(g_native_srt_ranges.load());
        g_native_srt_ranges.store(this);
    }

    ~NativeSrtRange() {
        std::scoped_lock lock{g_native_srt_mutex};
        auto* link = &g_native_srt_ranges;
        while (link->load() != this) {
            link = &link->load()->next;
        }
        link->store(next.load());
        while (g_native_srt_readers.load() != 0) {
            std::this_thread::yield();
        }
    }

    const uintptr_t start;
    const size_t size;
    std::atomic<NativeSrtRange*> next{};
};

struct NativeSrtWalkerCode {
    Xbyak::CodeGenerator code{4096, Xbyak::AutoGrow};
    std::optional<NativeSrtRange> range;
};

static void DumpSrtProgram(const Shader::Info& info, const u8* code, size_t codesize) {
    using namespace Common::FS;

    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}_{:#018x}.srtprogram.txt", info.hw_stage, info.pgm_hash);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create, FileType::TextFile};

    u64 address = reinterpret_cast<u64>(code);
    u64 code_end = address + codesize;
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = ZYAN_STATUS_SUCCESS;
    while (address < code_end && ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                                     instruction, operands, reinterpret_cast<void*>(address)))) {
        std::string s =
            Common::Decoder::Instance()->disassembleInst(instruction, operands, address);
        s += "\n";
        file.WriteString(s);
        address += instruction.length;
    }
}

static bool IsSrtWalkerCode(const void* code) {
    const auto address = reinterpret_cast<uintptr_t>(code);
    if (g_srt_codegen_start && address >= reinterpret_cast<uintptr_t>(g_srt_codegen_start) &&
        address < reinterpret_cast<uintptr_t>(g_srt_codegen.getCurr())) {
        return true;
    }
    g_native_srt_readers.fetch_add(1);
    bool found{};
    for (auto* range = g_native_srt_ranges.load(); range; range = range->next.load()) {
        if (address >= range->start && address - range->start < range->size) {
            found = true;
            break;
        }
    }
    g_native_srt_readers.fetch_sub(1);
    return found;
}

static bool SrtWalkerSignalHandler(void* context, void* fault_address) {
    const void* code = Common::GetRip(context);
    if (!IsSrtWalkerCode(code)) {
        return false;
    }

    // Patch instruction to zero register
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = Common::Decoder::Instance()->decodeInstruction(instruction, operands,
                                                                       const_cast<void*>(code), 15);

    ASSERT(ZYAN_SUCCESS(status) && instruction.mnemonic == ZYDIS_MNEMONIC_MOV &&
           operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
           operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY);

#ifdef __APPLE__
    if (*static_cast<const u8*>(code) == 0x2E) {
        auto& registers = static_cast<ucontext_t*>(context)->uc_mcontext->__ss;
        switch (operands[0].reg.value) {
        case ZYDIS_REGISTER_RDI:
            registers.__rdi = 0;
            break;
        case ZYDIS_REGISTER_R10D:
            registers.__r10 = 0;
            break;
        default:
            return false;
        }
        registers.__rip += instruction.length;
        return true;
    }
#endif

    size_t len = instruction.length;
    const size_t patch_size = 3;
    u8* code_patch = const_cast<u8*>(reinterpret_cast<const u8*>(code));

    // We can only encounter rdi or r10d as the first operand in a
    // fault memory access for SRT walker.
    switch (operands[0].reg.value) {
    case ZYDIS_REGISTER_RDI:
        // mov rdi, [rdi + (off_dw << 2)] -> xor rdi, rdi
        code_patch[0] = 0x48;
        code_patch[1] = 0x31;
        code_patch[2] = 0xFF;
        break;
    case ZYDIS_REGISTER_R10D:
        // mov r10d, [rdi + (off_dw << 2)] -> xor r10d, r10d
        code_patch[0] = 0x45;
        code_patch[1] = 0x31;
        code_patch[2] = 0xD2;
        break;
    default:
        UNREACHABLE_MSG("Unsupported register for SRT walker patch");
        return false;
    }

    // Fill nops
    memset(code_patch + patch_size, 0x90, len - patch_size);

    LOG_WARNING(Render_Recompiler, "Patched SRT walker at {}, fault address {}", code,
                fault_address);

    return true;
}

using namespace Shader;

struct PassInfo {
    struct PointerListCompare {
        bool operator()(const IR::Value& a, const IR::Value& b) const {
            // Keep order with the following criteria:
            // 1. If immediate, sort by immediate value
            // 2. If not immediate, sort by instruction order
            // 3. Immediates come first, then instructions
            if (a.IsImmediate() && b.IsImmediate()) {
                return a.U32() < b.U32();
            } else if (!a.IsImmediate() && !b.IsImmediate()) {
                auto a_inst = a.Inst();
                auto b_inst = b.Inst();
                auto a_block = a_inst->GetParent();
                auto b_block = b_inst->GetParent();
                if (a_block != b_block) {
                    return a_block < b_block; // Sort by block address
                }

                auto it = a_block->Instructions().iterator_to(*a_inst);
                auto end = a_block->Instructions().end();
                for (; it != end; ++it) {
                    if (&*it == b_inst) {
                        return true;
                    }
                }
                return false;
            } else {
                return a.IsImmediate(); // Immediates come first
            }
        }
    };

    // map offset to inst
    using PtrUserList =
        boost::container::flat_map<IR::Value, Shader::IR::Inst*, PointerListCompare>;

    Optimization::SrtGvnTable gvn_table;
    // keys are GetUserData, ReadConst or ReadConstBuffer instructions that are used as pointers
    std::unordered_map<IR::Inst*, PtrUserList> pointer_uses;
    // GetUserData instructions corresponding to sgpr_base of SRT roots
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> srt_roots;

    // pick a single inst for a given value number
    std::unordered_map<u32, IR::Inst*> vn_to_inst;

    // Bumped during codegen to assign offsets to readconsts
    u32 dst_off_dw;
    std::unordered_map<const IR::Inst*, u32> buffer_flatbuf_offsets;
    std::unordered_set<const IR::Inst*> sharp_sources;
    std::unordered_map<const IR::Inst*, SharpLocation> sharp_locations;
    bool prepare_memory_read{};
    u32 pointer_depth{};
    Xbyak::Label memory_read_exit;

    PtrUserList* GetUsesAsPointer(IR::Inst* inst) {
        auto it = pointer_uses.find(inst);
        if (it != pointer_uses.end()) {
            return &it->second;
        }
        return nullptr;
    }

    // Return a single instruction that this instruction is identical to, according
    // to value number
    // The "original" is arbitrary. Here it's the first instruction found for a given value number
    IR::Inst* DeduplicateInstruction(IR::Inst* inst) {
        auto it = vn_to_inst.try_emplace(gvn_table.GetValueNumber(inst), inst);
        return it.first->second;
    }
};
} // namespace

namespace Shader::Optimization {

namespace {

static inline u32 GetFlatbufOffset(const IR::Inst* inst, const PassInfo& pass_info) {
    if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
        const auto inst_info = inst->Flags<IR::BufferInstInfo>();
        if (pass_info.prepare_memory_read && inst_info.flatbuf_off_dw == 0) {
            const auto it = pass_info.buffer_flatbuf_offsets.find(inst);
            return it == pass_info.buffer_flatbuf_offsets.end() ? 0 : it->second;
        }
        return inst_info.flatbuf_off_dw;
    }
    if (inst->GetOpcode() == IR::Opcode::ReadConst) {
        if (pass_info.prepare_memory_read) {
            return inst->Flags<u32>();
        }
        return inst->Flags<u16>();
    }
    UNREACHABLE_MSG("Instruction not supported");
}

static inline void SetFlatbufOffset(IR::Inst* inst, u32 offset, PassInfo& pass_info) {
    if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
        auto inst_info = inst->Flags<IR::BufferInstInfo>();
        if (pass_info.prepare_memory_read) {
            if (offset >= UNKNOWN_LOCATION) {
                pass_info.buffer_flatbuf_offsets[inst] = offset;
                inst_info.flatbuf_off_dw.Assign(0);
            } else {
                pass_info.buffer_flatbuf_offsets.erase(inst);
                inst_info.flatbuf_off_dw.Assign(offset);
            }
            inst->SetFlags(inst_info);
            return;
        }
        inst_info.flatbuf_off_dw.Assign(offset);
        inst->SetFlags(inst_info);
        return;
    }
    if (inst->GetOpcode() == IR::Opcode::ReadConst) {
        if (pass_info.prepare_memory_read) {
            inst->SetFlags(offset);
        } else {
            inst->SetFlags(static_cast<u16>(offset));
        }
        return;
    }
    UNREACHABLE_MSG("Instruction not supported");
}

static bool ComputeOffset(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                          const IR::Value& off_dw);

#define ABORT_ON_FAILURE(expr)                                                                     \
    if (!(expr)) {                                                                                 \
        return false;                                                                              \
    }

#define POP_ABORT_ON_FAILURE(expr)                                                                 \
    if (!(expr)) {                                                                                 \
        c.add(rsp, 8);                                                                             \
        return false;                                                                              \
    }

static bool EmitComputeOffsetIAdd32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.add(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.add(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.add(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetISub32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.neg(reg);
        c.add(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.sub(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.neg(reg);
        c.add(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetIMul32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.imul(reg, reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.imul(reg, reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.imul(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetShiftLeftLogical32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                                PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        if (!pass_info.prepare_memory_read) {
            c.shl(reg, inst->Arg(0).U32());
        }
        c.mov(ecx, reg);
        c.mov(reg, inst->Arg(0).U32());
        c.shl(reg, cl);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.shl(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.pop(rcx);
        c.shl(reg, cl);
    }
    return true;
}

static bool EmitComputeOffsetShiftRightLogical32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                                 PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, reg);
        c.mov(reg, inst->Arg(0).U32());
        c.shr(reg, cl);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.shr(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.pop(rcx);
        c.shr(reg, cl);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseAnd32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                          PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.and_(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.and_(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.and_(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseOr32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                         PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.or_(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.or_(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.or_(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseXor32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                          PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.xor_(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.xor_(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.xor_(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseNot32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                          PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
    c.not_(reg);
    return true;
}

static bool EmitComputeOffsetUMin32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, inst->Arg(0).U32());
        c.cmp(reg, ecx);
        c.cmova(reg, ecx);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.mov(ecx, inst->Arg(1).U32());
        c.cmp(reg, ecx);
        c.cmova(reg, ecx);
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.cmp(reg, dword[rsp]);
        c.cmova(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetUMax32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, inst->Arg(0).U32());
        c.cmp(reg, ecx);
        c.cmovb(reg, ecx);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.mov(ecx, inst->Arg(1).U32());
        c.cmp(reg, ecx);
        c.cmovb(reg, ecx);
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.cmp(reg, dword[rsp]);
        c.cmovb(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitFieldUExtract(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                              PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        c.mov(reg, inst->Arg(0).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
    }
    bool in_stack = false;
    if (inst->Arg(1).IsImmediate()) {
        c.shr(reg, inst->Arg(1).U32());
    } else {
        c.push(reg.cvt64());
        in_stack = true;
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, reg);
        c.shr(dword[rsp], cl);
    }
    if (inst->Arg(2).IsImmediate()) {
        if (in_stack) {
            c.pop(reg.cvt64());
        }
        if (!pass_info.prepare_memory_read || inst->Arg(2).U32() != 32) {
            c.and_(reg, (1U << inst->Arg(2).U32()) - 1);
        }
    } else {
        if (!in_stack) {
            c.push(reg.cvt64());
        }
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(2)));
        c.mov(ecx, reg);
        c.mov(edx, 1);
        c.shl(edx, cl);
        c.dec(edx);
        if (pass_info.prepare_memory_read) {
            c.cmp(ecx, 32);
            c.mov(eax, 0xFFFFFFFFu);
            c.cmove(edx, eax);
        }
        c.and_(dword[rsp], edx);
        c.pop(reg.cvt64());
    }
    return true;
}

static bool IsAllowedOffsetInstruction(const IR::Inst* inst) {
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
    case IR::Opcode::ReadConst:
    case IR::Opcode::ReadConstBuffer:
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::IMul32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::UMin32:
    case IR::Opcode::UMax32:
    case IR::Opcode::BitFieldUExtract:
        return true;
    default:
        return false;
    }
}

static bool ComputeOffset(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                          const IR::Value& off_dw) {
    ASSERT(reg != ecx && reg != edx);
    auto inst = off_dw.Inst();
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        c.mov(reg, ptr[rsi + (static_cast<u32>(inst->Arg(0).ScalarReg()) << 2)]);
        return true;
    case IR::Opcode::ReadConst:
    case IR::Opcode::ReadConstBuffer:
        if (u32 offset = GetFlatbufOffset(pass_info.DeduplicateInstruction(inst), pass_info);
            offset != 0) {
            c.mov(reg, ptr[rsi + (offset << 2)]);
            return true;
        }
        return false;
    case IR::Opcode::IAdd32:
        ABORT_ON_FAILURE(EmitComputeOffsetIAdd32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::ISub32:
        ABORT_ON_FAILURE(EmitComputeOffsetISub32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::IMul32:
        ABORT_ON_FAILURE(EmitComputeOffsetIMul32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::ShiftLeftLogical32:
        ABORT_ON_FAILURE(EmitComputeOffsetShiftLeftLogical32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::ShiftRightLogical32:
        ABORT_ON_FAILURE(EmitComputeOffsetShiftRightLogical32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseAnd32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseAnd32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseOr32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseOr32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseXor32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseXor32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseNot32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseNot32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::UMin32:
        ABORT_ON_FAILURE(EmitComputeOffsetUMin32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::UMax32:
        ABORT_ON_FAILURE(EmitComputeOffsetUMax32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitFieldUExtract:
        ABORT_ON_FAILURE(EmitComputeOffsetBitFieldUExtract(c, reg, pass_info, inst));
        return true;
    default:
        LOG_ERROR(Render_Recompiler, "Unexpected instruction for offset computation, {}",
                  magic_enum::enum_name(inst->GetOpcode()));
        return false;
    }
}

static void PrepareMemoryRead(Xbyak::CodeGenerator& c, PassInfo& pass_info,
                              const Xbyak::Address& address, u64 size) {
    if (!pass_info.prepare_memory_read) {
        return;
    }
    c.push(rdi);
    c.push(rsi);
    c.push(r10);
    c.lea(rsi, address);
    c.mov(rdi, r13);
    c.mov(rdx, size);
    c.mov(r11, rsp);
    c.and_(rsp, -16);
    c.sub(rsp, 16);
    c.mov(ptr[rsp], r11);
    c.mov(rax, reinterpret_cast<uintptr_t>(&SrtMemoryRead::Prepare));
    c.call(rax);
    c.mov(rsp, ptr[rsp]);
    c.pop(r10);
    c.pop(rsi);
    c.pop(rdi);
    c.test(al, al);
    c.jz(pass_info.memory_read_exit, Xbyak::CodeGenerator::T_NEAR);
}

static Xbyak::Address ImmediateSrtAddress(Xbyak::CodeGenerator& c, const PassInfo& pass_info,
                                          u32 offset_dw) {
    if (pass_info.prepare_memory_read) {
        const u64 offset_bytes = u64{offset_dw} << 2;
        if (offset_bytes >= 0x80000000u) {
            c.mov(r10, offset_bytes);
            return ptr[rdi + r10];
        }
    }
    return ptr[rdi + (offset_dw << 2)];
}

static void EmitSrtRead(Xbyak::CodeGenerator& c, const PassInfo& pass_info,
                        const Xbyak::Reg& destination, const Xbyak::Address& address) {
#ifdef __APPLE__
    if (pass_info.prepare_memory_read) {
        // The ignored CS override marks native reads for recovery without code patching.
        c.db(0x2E);
    }
#endif
    c.mov(destination, address);
}

static inline bool PushPtr(Xbyak::CodeGenerator& c, PassInfo& pass_info, const IR::Value& off_dw) {
    c.push(rdi);
    if (off_dw.IsImmediate()) {
        const auto address = ImmediateSrtAddress(c, pass_info, off_dw.U32());
        if (pass_info.pointer_depth) {
            PrepareMemoryRead(c, pass_info, address, sizeof(u64));
        }
        EmitSrtRead(c, pass_info, rdi, address);
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, r10d, pass_info, off_dw));
        if (pass_info.prepare_memory_read) {
            c.shl(r10, 2);
        } else {
            c.shl(r10d, 2);
            c.mov(r10d, r10d);
        }
        if (pass_info.pointer_depth) {
            PrepareMemoryRead(c, pass_info, ptr[rdi + r10], sizeof(u64));
        }
        EmitSrtRead(c, pass_info, rdi, ptr[rdi + r10]);
    }
    c.mov(r10, pass_info.prepare_memory_read ? 0xFFFFFFFFFFFCULL : 0xFFFFFFFFFFFFULL);
    c.and_(rdi, r10);
    return true;
}

static inline void PopPtr(Xbyak::CodeGenerator& c) {
    c.pop(rdi);
}

static void VisitPointer(const IR::Value& off_dw, IR::Inst* subtree, PassInfo& pass_info,
                         Xbyak::CodeGenerator& c) {
    if ((subtree->GetOpcode() == IR::Opcode::ReadConst ||
         subtree->GetOpcode() == IR::Opcode::ReadConstBuffer) &&
        GetFlatbufOffset(subtree, pass_info) == 0) {
        return;
    }

    if (!PushPtr(c, pass_info, off_dw)) {
        if (pass_info.prepare_memory_read) {
            throw std::runtime_error("Failed to compute Metal SRT pointer offset");
        }
        LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
        return;
    }
    ++pass_info.pointer_depth;
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    // First copy all the src data from this tree level
    // That way, all data that was contiguous in the guest SRT is also contiguous in the
    // flattened buffer.
    // TODO src and dst are contiguous. Optimize with wider loads/stores
    // TODO if this subtree is dynamically indexed, don't compact it (keep it sparse)
    u64 prepared_reads{};
    for (auto it = use_list->begin(); it != use_list->end(); ++it) {
        if (pass_info.prepare_memory_read && pass_info.dst_off_dw >= 0x20000000u) {
            throw std::runtime_error("Metal SRT exceeds the host destination range");
        }
        const auto& [src_off_dw, use] = *it;
        if (src_off_dw.IsImmediate()) {
            const auto address = ImmediateSrtAddress(c, pass_info, src_off_dw.U32());
            if (!prepared_reads) {
                u64 read_size = sizeof(u32);
                if (pass_info.prepare_memory_read) {
                    const u64 byte_offset = u64{src_off_dw.U32()} << 2;
                    for (auto next = std::next(it); next != use_list->end(); ++next) {
                        if (!next->first.IsImmediate() ||
                            (u64{next->first.U32()} << 2) != byte_offset + read_size) {
                            break;
                        }
                        read_size += sizeof(u32);
                    }
                }
                PrepareMemoryRead(c, pass_info, address, read_size);
                prepared_reads = read_size / sizeof(u32);
            }
            --prepared_reads;
            EmitSrtRead(c, pass_info, r10d, address);
        } else {
            if (!ComputeOffset(c, r10d, pass_info, src_off_dw)) {
                if (pass_info.prepare_memory_read) {
                    throw std::runtime_error("Failed to compute Metal SRT data offset");
                }
                LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
                continue;
            }
            if (pass_info.prepare_memory_read) {
                c.shl(r10, 2);
            } else {
                c.shl(r10d, 2);
                c.mov(r10d, r10d);
            }
            PrepareMemoryRead(c, pass_info, ptr[rdi + r10], sizeof(u32));
            EmitSrtRead(c, pass_info, r10d, dword[rdi + r10]);
        }
        c.mov(ptr[rsi + (pass_info.dst_off_dw << 2)], r10d);
        if (pass_info.prepare_memory_read) {
            if (const auto alias = pass_info.sharp_locations.find(use);
                alias != pass_info.sharp_locations.end()) {
                c.mov(ptr[rsi + (u32{alias->second} << 2)], r10d);
            }
        }
        SetFlatbufOffset(use, pass_info.dst_off_dw, pass_info);
        pass_info.dst_off_dw = pass_info.prepare_memory_read
                                   ? pass_info.dst_off_dw + 1
                                   : static_cast<u16>(pass_info.dst_off_dw + 1);
    }

    // Then visit any children used as pointers
    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            VisitPointer(src_off_dw, use, pass_info, c);
        }
    }

    --pass_info.pointer_depth;
    PopPtr(c);
}

template <typename F>
static void VisitSharpSources(const ResourceDiscoveryList& resources, F&& visit) {
    for (const auto& resource : resources) {
        for (const auto& sharp : resource.sharps) {
            for (u32 i = 0; i < sharp.num_dwords; ++i) {
                if (auto* source = sharp.dwords[i].TryInst()) {
                    visit(source);
                }
            }
            if (sharp.post_op == SharpFetchPostOp::DisableAnisoIfSingleLod) {
                if (auto* source = sharp.post_op_data.lod_prod.TryInst()) {
                    visit(source);
                }
            }
        }
    }
}

static void ReserveSharpLocations(IR::Inst* subtree, PassInfo& pass_info,
                                  std::unordered_set<const IR::Inst*>& visited) {
    if (!visited.insert(subtree).second) {
        return;
    }
    auto* uses = pass_info.GetUsesAsPointer(subtree);
    ASSERT(uses);
    for (const auto& [offset, use] : *uses) {
        if (pass_info.sharp_sources.contains(use) && !pass_info.sharp_locations.contains(use)) {
            const auto location = NUM_USER_DATA_REGS + pass_info.sharp_locations.size();
            if (location >= UNKNOWN_LOCATION) {
                throw std::runtime_error("Metal SRT exceeds the descriptor alias range");
            }
            pass_info.sharp_locations.emplace(use, static_cast<SharpLocation>(location));
        }
    }
    for (const auto& [offset, use] : *uses) {
        if (pass_info.GetUsesAsPointer(use)) {
            ReserveSharpLocations(use, pass_info, visited);
        }
    }
}

static void PrepareSrtLayout(Info& info, PassInfo& pass_info) {
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    ASSERT(pass_info.dst_off_dw == info.srt_info.flattened_bufsize_dw);
    if (!pass_info.prepare_memory_read) {
        return;
    }
    u64 read_count{};
    for (const auto& [pointer, uses] : pass_info.pointer_uses) {
        read_count += uses.size();
    }
    if (read_count <= UNKNOWN_LOCATION - NUM_USER_DATA_REGS) {
        return;
    }
    std::unordered_set<const IR::Inst*> visited;
    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        ReserveSharpLocations(root, pass_info, visited);
    }
    pass_info.dst_off_dw += static_cast<u32>(pass_info.sharp_locations.size());
}

static SharpLocation NativeSharpLocation(const IR::Inst* source, const PassInfo& pass_info) {
    const auto offset = GetFlatbufOffset(source, pass_info);
    if (!offset) {
        throw std::runtime_error("Metal descriptor source was not flattened");
    }
    const auto alias = pass_info.sharp_locations.find(source);
    const u32 location = alias != pass_info.sharp_locations.end() ? alias->second : offset;
    if (location >= UNKNOWN_LOCATION) {
        throw std::runtime_error("Metal descriptor source exceeds the descriptor index range");
    }
    return static_cast<SharpLocation>(location);
}

static void GenerateSrtProgram(Info& info, PassInfo& pass_info) {
    if (pass_info.srt_roots.empty()) {
        return;
    }

    pass_info.prepare_memory_read = bool(info.prepare_memory_read);
    PrepareSrtLayout(info, pass_info);
    std::shared_ptr<NativeSrtWalkerCode> native_codegen;
    if (pass_info.prepare_memory_read) {
        native_codegen = std::make_shared<NativeSrtWalkerCode>();
    }
    Xbyak::CodeGenerator& c = native_codegen ? native_codegen->code : g_srt_codegen;

    // Register the signal handler for SRT walker, if not already registered
    std::call_once(g_srt_signal_registration, [] {
        g_srt_codegen_start = g_srt_codegen.getCurr();
        auto* signals = Core::Signals::Instance();
        // Call after the memory invalidation handler
        constexpr u32 priority = 1;
        signals->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    });

    const size_t code_offset = c.getSize();
    if (pass_info.prepare_memory_read) {
        c.push(r12);
        c.push(r13);
        c.mov(r12, rsp);
        c.mov(r13, rdx);
    } else {
        info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
    }
    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        VisitPointer(IR::Value(static_cast<u32>(sgpr_base)), root, pass_info, c);
    }

    if (pass_info.prepare_memory_read) {
        for (const auto* source : pass_info.sharp_sources) {
            NativeSharpLocation(source, pass_info);
        }
        c.L(pass_info.memory_read_exit);
        c.mov(rsp, r12);
        c.pop(r13);
        c.pop(r12);
    }
    c.ret();
    if (pass_info.prepare_memory_read) {
        c.ready(Xbyak::CodeGenerator::PROTECT_RE);
    } else {
        c.ready();
    }

    const size_t code_size = c.getSize() - code_offset;
    const auto* code_start = c.getCode() + code_offset;
    if (!pass_info.prepare_memory_read) {
        info.srt_info.walker_func_size = code_size;
    }

    if (EmulatorSettings.IsDumpShaders()) {
        DumpSrtProgram(info, code_start, code_size);
    }

    if (native_codegen) {
        native_codegen->range.emplace(code_start, code_size);
        info.walker_code_owner = native_codegen;
        info.walker_with_memory_read = c.getCode<PFN_SrtWalkerWithMemoryRead>();
        info.walker_code = {code_start, code_size};
    }
    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
}

static bool IsReadConstSource(const IR::Value base) {
    auto* inst = base.TryInst();
    if (!inst) {
        return false;
    }
    return inst->GetOpcode() == IR::Opcode::GetUserData ||
           inst->GetOpcode() == IR::Opcode::ReadConst ||
           inst->GetOpcode() == IR::Opcode::ReadConstBuffer;
}

void SimplifyReadConstAddressAdd(IR::Inst& inst, bool native) {
    // This handles the following pattern by combining the addition with the offset
    // s_add_u32    s12, s2, <offset>
    // s_addc_u32   s13, s3, 0
    // s_load_dword s16, s[12:13], 0x0

    if (inst.GetOpcode() != IR::Opcode::ReadConst) {
        return;
    }
    IR::Inst* addr = inst.Arg(0).Inst();
    IR::Inst* hi = addr->Arg(1).TryInst();
    IR::Inst* lo = addr->Arg(0).TryInst();
    if (!hi || !lo) {
        return;
    }

    if (lo->GetOpcode() != IR::Opcode::IAdd32 || hi->GetOpcode() != IR::Opcode::IAdd32) {
        return;
    }

    IR::Value add_offset;
    IR::Value base;
    if (base = lo->Arg(0); IsReadConstSource(base)) {
        add_offset = lo->Arg(1);
    } else if (base = lo->Arg(1); IsReadConstSource(base)) {
        add_offset = lo->Arg(0);
    } else {
        return;
    }

    IR::Inst* sel = hi->Arg(1).TryInst();
    if (!sel || sel->GetOpcode() != IR::Opcode::SelectU32) {
        return;
    }

    IR::Value sel_true = sel->Arg(1);
    IR::Value sel_false = sel->Arg(2);
    if (!sel_true.IsImmediate() || !sel_false.IsImmediate() || sel_true.U32() != 1 ||
        sel_false.U32() != 0) {
        return;
    }

    addr->SetArg(0, base);
    addr->SetArg(1, hi->Arg(0));
    for (auto [user, operand] : addr->Uses()) {
        ASSERT(user->GetOpcode() == IR::Opcode::ReadConst && operand == 0);
        IR::IREmitter ir{*user->GetParent(), IR::Block::InstructionList::s_iterator_to(*user)};
        IR::U32 offset = IR::U32{user->Arg(1)};
        IR::U32 dw_add_offset =
            add_offset.IsImmediate()
                ? ir.Imm32(add_offset.U32() >> 2u)
                : IR::U32{ir.ShiftRightLogical(IR::U32{add_offset}, ir.Imm32(2u))};
        if (native) {
            const auto low_bits = ir.IAdd(ir.BitwiseAnd(IR::U32{base}, ir.Imm32(3)),
                                         ir.BitwiseAnd(IR::U32{add_offset}, ir.Imm32(3)));
            dw_add_offset = ir.IAdd(dw_add_offset, ir.ShiftRightLogical(low_bits, ir.Imm32(2)));
        }
        if (offset.IsImmediate() && dw_add_offset.IsImmediate()) {
            user->SetArg(1, ir.Imm32(offset.U32() + dw_add_offset.U32()));
        } else {
            user->SetArg(1, ir.IAdd(offset, dw_add_offset));
        }
    }
}

} // Anonymous namespace

void FlattenExtendedUserdataPass(IR::Program& program, const ResourceDiscoveryList& resources) {
    auto& post_order = program.post_order_blocks;
    PassInfo pass_info;
    program.info.flatbuf_sharp_locations.clear();
    if (program.info.prepare_memory_read) {
        VisitSharpSources(resources, [](IR::Inst* source) {
            if (source->GetOpcode() == IR::Opcode::ReadConstBuffer) {
                auto flags = source->Flags<IR::BufferInstInfo>();
                flags.sharp_source.Assign(1u);
                source->SetFlags(flags);
            }
        });
    }

    // traverse at end and assign offsets to duplicate readconsts, using
    // vn_to_inst as the source
    boost::container::small_vector<IR::Inst*, 32> all_readconsts, visited;
    std::queue<IR::Inst*> queue;

    for (auto it = post_order.rbegin(); it != post_order.rend(); it++) {
        IR::Block* block{*it};
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::ReadConst &&
                inst.GetOpcode() != IR::Opcode::ReadConstBuffer) {
                continue;
            }

            if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
                // Only flatten ReadConstBuffer if it was marked as a sharp source in the
                // resource discovery pass
                auto inst_info = inst.Flags<IR::BufferInstInfo>();
                if (!inst_info.sharp_source) {
                    continue;
                }
            }

            SimplifyReadConstAddressAdd(inst, bool(program.info.prepare_memory_read));

            all_readconsts.push_back(&inst);

            auto offset = inst.Arg(1);
            if (offset.IsImmediate()) {
                continue;
            }

            queue.push(offset.Inst());
            while (!queue.empty()) {
                IR::Inst* inst{queue.front()};
                queue.pop();

                if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
                    auto buffer_inst_info = inst->Flags<IR::BufferInstInfo>();
                    if (!buffer_inst_info.sharp_source.Value()) {
                        all_readconsts.push_back(inst);
                        auto offset = inst->Arg(1);
                        if (!offset.IsImmediate()) {
                            auto arg_inst = offset.Inst();
                            if (std::ranges::find(visited, arg_inst) == visited.end()) {
                                visited.push_back(arg_inst);
                                queue.push(arg_inst);
                            }
                        }
                    }
                    continue;
                }
                if (inst->GetOpcode() == IR::Opcode::GetUserData ||
                    inst->GetOpcode() == IR::Opcode::ReadConst ||
                    !IsAllowedOffsetInstruction(inst)) {
                    continue;
                }
                for (size_t arg = inst->NumArgs(); arg--;) {
                    const IR::Value arg_value = inst->Arg(arg);
                    if (arg_value.IsImmediate()) {
                        continue;
                    }
                    IR::Inst* arg_inst = arg_value.Inst();
                    if (std::ranges::find(visited, arg_inst) == visited.end()) {
                        visited.push_back(arg_inst);
                        queue.push(arg_inst);
                    }
                }
            }
        }
    }

    for (auto inst : all_readconsts) {
        if (pass_info.DeduplicateInstruction(inst) != inst) {
            // This is a duplicate of a readconst we've already visited
            continue;
        }

        IR::Inst* base = inst->Arg(0).Inst();
        if (auto* inst = base->Arg(0).TryInst();
            inst && inst->GetOpcode() == IR::Opcode::ReadFirstLane) {
            continue;
        }
        ASSERT_MSG(IsReadConstSource(base->Arg(0)), "ReadConst base low not from constant memory");
        ASSERT_MSG(IsReadConstSource(base->Arg(1)), "ReadConst base high not from constant memory");

        IR::Inst* ptr_lo = base->Arg(0).Inst();
        ptr_lo = pass_info.DeduplicateInstruction(ptr_lo);

        auto ptr_uses_kv = pass_info.pointer_uses.try_emplace(ptr_lo, PassInfo::PtrUserList{});
        PassInfo::PtrUserList& user_list = ptr_uses_kv.first->second;

        user_list[inst->Arg(1)] = inst;

        if (ptr_lo->GetOpcode() == IR::Opcode::GetUserData) {
            IR::ScalarReg ud_reg = ptr_lo->Arg(0).ScalarReg();
            pass_info.srt_roots[ud_reg] = ptr_lo;
        }
    }

    if (program.info.prepare_memory_read) {
        VisitSharpSources(resources, [&](IR::Inst* source) {
            if (source->GetOpcode() != IR::Opcode::GetUserData) {
                pass_info.sharp_sources.insert(pass_info.DeduplicateInstruction(source));
            }
        });
    }

    GenerateSrtProgram(program.info, pass_info);

    // Assign offsets to duplicate readconsts
    for (IR::Inst* readconst : all_readconsts) {
        ASSERT(pass_info.vn_to_inst.contains(pass_info.gvn_table.GetValueNumber(readconst)));
        IR::Inst* original = pass_info.DeduplicateInstruction(readconst);
        SetFlatbufOffset(readconst, GetFlatbufOffset(original, pass_info), pass_info);
    }

    if (!pass_info.sharp_locations.empty()) {
        VisitSharpSources(resources, [&](IR::Inst* source) {
            if (source->GetOpcode() != IR::Opcode::GetUserData) {
                program.info.flatbuf_sharp_locations.emplace(
                    source,
                    NativeSharpLocation(pass_info.DeduplicateInstruction(source), pass_info));
            }
        });
    }

    program.info.RefreshFlatBuf();
}

} // namespace Shader::Optimization

#else

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    UNREACHABLE_MSG("RegisterWalkerCode unimplemented for target architecture.");
}

namespace Optimization {

void FlattenExtendedUserdataPass(IR::Program& program, const ResourceDiscoveryList& resources) {
    UNREACHABLE_MSG("FlattenExtendedUserdataPass unimplemented for target architecture.");
}

} // namespace Optimization

} // namespace Shader

#endif
