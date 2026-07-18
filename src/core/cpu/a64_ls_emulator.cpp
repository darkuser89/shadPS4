// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstring>

#include "core/cpu/a64_ls_emulator.h"

namespace Core::CPU {

namespace {

u64 ReadReg(const u64* x, u32 n, bool sp_for_31) {
    if (n == 31) {
        return sp_for_31 ? x[31] : 0;
    }
    return x[n];
}

void WriteReg(u64* x, u32 n, u64 value) {
    if (n != 31) {
        x[n] = value;
    }
}

s64 SignExtend(u64 value, u32 bits) {
    const u64 mask = u64{1} << (bits - 1);
    return static_cast<s64>((value ^ mask) - mask);
}

void CopyFromMemory(void* dst, u64 src, size_t n) {
    std::memcpy(dst, reinterpret_cast<const void*>(src), n);
    // The JIT's TSO forms carry acquire semantics on loads.
    std::atomic_thread_fence(std::memory_order_acquire);
}

void CopyToMemory(u64 dst, const void* src, size_t n) {
    // The JIT's TSO forms carry release semantics on stores.
    std::atomic_thread_fence(std::memory_order_release);
    std::memcpy(reinterpret_cast<void*>(dst), src, n);
}

u64 LoadZx(u64 addr, u32 bytes) {
    u64 value = 0;
    CopyFromMemory(&value, addr, bytes);
    return value;
}

void StoreVal(u64 addr, u64 value, u32 bytes) {
    CopyToMemory(addr, &value, bytes);
}

// Load with optional sign extension into a 32/64-bit destination, per the
// LDRS* `opc` conventions.
u64 LoadWithOpc(u64 addr, u32 bytes, u32 opc) {
    const u64 raw = LoadZx(addr, bytes);
    switch (opc) {
    case 1: // zero-extending load
        return raw;
    case 2: // sign-extend into 64-bit
        return static_cast<u64>(SignExtend(raw, bytes * 8));
    case 3: // sign-extend into 32-bit (upper 32 bits zero)
        return static_cast<u64>(SignExtend(raw, bytes * 8)) & 0xFFFF'FFFFULL;
    default:
        return raw;
    }
}

} // namespace

bool EmulateA64LoadStore(u32 instr, u64 target_addr, u64* x, unsigned char* v) {
    const u32 rt = instr & 0x1F;
    const u32 rn = (instr >> 5) & 0x1F;

    // --- Load/store register, unsigned scaled 12-bit immediate -------------
    // size(2) 111 V 01 opc(2) imm12 Rn Rt
    if ((instr & 0x3B000000) == 0x39000000) {
        const u32 size = instr >> 30;
        const bool simd = (instr >> 26) & 1;
        const u32 opc = (instr >> 22) & 3;
        if (simd) {
            const u32 scale = ((opc & 2) << 1) | size;
            if (scale > 4) {
                return false;
            }
            const u32 bytes = 1u << scale;
            const bool is_load = opc & 1;
            if (is_load) {
                unsigned char tmp[16]{};
                CopyFromMemory(tmp, target_addr, bytes);
                std::memset(v + rt * 16, 0, 16);
                std::memcpy(v + rt * 16, tmp, bytes);
            } else {
                CopyToMemory(target_addr, v + rt * 16, bytes);
            }
            return true;
        }
        if (size == 3 && opc == 2) {
            return true; // PRFM: no architectural effect
        }
        if (opc == 0) {
            StoreVal(target_addr, ReadReg(x, rt, false), 1u << size);
            return true;
        }
        if (opc == 1 || ((opc == 2 || opc == 3) && size <= 2)) {
            WriteReg(x, rt, LoadWithOpc(target_addr, 1u << size, opc));
            return true;
        }
        return false;
    }

    // --- Load/store register, unscaled 9-bit immediate (LDUR/STUR) --------
    // size(2) 111 V 00 opc(2) 0 imm9 00 Rn Rt
    if ((instr & 0x3B200C00) == 0x38000000) {
        const u32 size = instr >> 30;
        const bool simd = (instr >> 26) & 1;
        const u32 opc = (instr >> 22) & 3;
        if (simd) {
            const u32 scale = ((opc & 2) << 1) | size;
            if (scale > 4) {
                return false;
            }
            const u32 bytes = 1u << scale;
            if (opc & 1) {
                unsigned char tmp[16]{};
                CopyFromMemory(tmp, target_addr, bytes);
                std::memset(v + rt * 16, 0, 16);
                std::memcpy(v + rt * 16, tmp, bytes);
            } else {
                CopyToMemory(target_addr, v + rt * 16, bytes);
            }
            return true;
        }
        if (opc == 0) {
            StoreVal(target_addr, ReadReg(x, rt, false), 1u << size);
            return true;
        }
        if (opc == 1 || ((opc == 2 || opc == 3) && size <= 2)) {
            WriteReg(x, rt, LoadWithOpc(target_addr, 1u << size, opc));
            return true;
        }
        return false;
    }

    // --- LDAPUR/STLUR (LRCPC2, unscaled 9-bit immediate) -------------------
    // size(2) 011001 opc(2) 0 imm9 00 Rn Rt
    if ((instr & 0x3F200C00) == 0x19000000) {
        const u32 size = instr >> 30;
        const u32 opc = (instr >> 22) & 3;
        if (opc == 0) {
            StoreVal(target_addr, ReadReg(x, rt, false), 1u << size);
            return true;
        }
        if (opc == 1 || ((opc == 2 || opc == 3) && size <= 2)) {
            WriteReg(x, rt, LoadWithOpc(target_addr, 1u << size, opc));
            return true;
        }
        return false;
    }

    // --- Load/store register, register offset ------------------------------
    // size(2) 111 V 00 opc(2) 1 Rm option(3) S 10 Rn Rt
    if ((instr & 0x3B200C00) == 0x38200800) {
        const u32 size = instr >> 30;
        const bool simd = (instr >> 26) & 1;
        const u32 opc = (instr >> 22) & 3;
        if (simd) {
            const u32 scale = ((opc & 2) << 1) | size;
            if (scale > 4) {
                return false;
            }
            const u32 bytes = 1u << scale;
            if (opc & 1) {
                unsigned char tmp[16]{};
                CopyFromMemory(tmp, target_addr, bytes);
                std::memset(v + rt * 16, 0, 16);
                std::memcpy(v + rt * 16, tmp, bytes);
            } else {
                CopyToMemory(target_addr, v + rt * 16, bytes);
            }
            return true;
        }
        if (size == 3 && opc == 2) {
            return true; // PRFM (register offset)
        }
        if (opc == 0) {
            StoreVal(target_addr, ReadReg(x, rt, false), 1u << size);
            return true;
        }
        if (opc == 1 || ((opc == 2 || opc == 3) && size <= 2)) {
            WriteReg(x, rt, LoadWithOpc(target_addr, 1u << size, opc));
            return true;
        }
        return false;
    }

    // --- LDAR/LDAPR/STLR (register, no offset) -----------------------------
    if ((instr & 0x3FFFFC00) == 0x089FFC00) { // STLR
        const u32 size = instr >> 30;
        StoreVal(target_addr, ReadReg(x, rt, false), 1u << size);
        return true;
    }
    if ((instr & 0x3FFFFC00) == 0x08DFFC00) { // LDAR
        const u32 size = instr >> 30;
        WriteReg(x, rt, LoadZx(target_addr, 1u << size));
        return true;
    }
    if ((instr & 0x3FFFFC00) == 0x38BFC000) { // LDAPR
        const u32 size = instr >> 30;
        WriteReg(x, rt, LoadZx(target_addr, 1u << size));
        return true;
    }

    // --- Load/store pair, signed 7-bit scaled offset, no writeback ---------
    // opc(2) 101 V 010 L imm7 Rt2 Rn Rt
    if ((instr & 0x3B800000) == 0x29000000) {
        const u32 opc = instr >> 30;
        const bool simd = (instr >> 26) & 1;
        const bool is_load = (instr >> 22) & 1;
        const u32 rt2 = (instr >> 10) & 0x1F;
        u32 scale;
        if (simd) {
            if (opc > 2) {
                return false;
            }
            scale = 2 + opc; // S/D/Q
        } else {
            if (opc == 1) {
                // LDPSW
                if (!is_load) {
                    return false;
                }
                const u64 lo = LoadZx(target_addr, 4);
                const u64 hi = LoadZx(target_addr + 4, 4);
                WriteReg(x, rt, static_cast<u64>(SignExtend(lo, 32)));
                WriteReg(x, rt2, static_cast<u64>(SignExtend(hi, 32)));
                return true;
            }
            if (opc == 3) {
                return false;
            }
            scale = 2 + (opc >> 1); // W/X
        }
        const u32 bytes = 1u << scale;
        if (simd) {
            if (is_load) {
                unsigned char tmp[32]{};
                CopyFromMemory(tmp, target_addr, bytes * 2);
                std::memset(v + rt * 16, 0, 16);
                std::memset(v + rt2 * 16, 0, 16);
                std::memcpy(v + rt * 16, tmp, bytes);
                std::memcpy(v + rt2 * 16, tmp + bytes, bytes);
            } else {
                unsigned char tmp[32]{};
                std::memcpy(tmp, v + rt * 16, bytes);
                std::memcpy(tmp + bytes, v + rt2 * 16, bytes);
                CopyToMemory(target_addr, tmp, bytes * 2);
            }
            return true;
        }
        if (is_load) {
            WriteReg(x, rt, LoadZx(target_addr, bytes));
            WriteReg(x, rt2, LoadZx(target_addr + bytes, bytes));
        } else {
            u64 vals[2] = {ReadReg(x, rt, false), ReadReg(x, rt2, false)};
            if (bytes == 4) {
                u32 w[2] = {static_cast<u32>(vals[0]), static_cast<u32>(vals[1])};
                CopyToMemory(target_addr, w, 8);
            } else {
                CopyToMemory(target_addr, vals, 16);
            }
        }
        return true;
    }

    // --- LSE atomic memory operations --------------------------------------
    // size(2) 111 0 00 A R 1 Rs o3 opc(3) 00 Rn Rt
    if ((instr & 0x3B200C00) == 0x38200000) {
        const u32 size = instr >> 30;
        const u32 bytes = 1u << size;
        const u32 rs = (instr >> 16) & 0x1F;
        const u32 o3 = (instr >> 15) & 1;
        const u32 opc = (instr >> 12) & 7;
        const u64 operand = ReadReg(x, rs, false);
        const u64 old = LoadZx(target_addr, bytes);
        u64 result;
        if (o3 == 1 && opc == 0) {
            result = operand; // SWP
        } else if (o3 == 0) {
            switch (opc) {
            case 0: // LDADD
                result = old + operand;
                break;
            case 1: // LDCLR
                result = old & ~operand;
                break;
            case 2: // LDEOR
                result = old ^ operand;
                break;
            case 3: // LDSET
                result = old | operand;
                break;
            case 4: // LDSMAX
                result = static_cast<u64>(
                    std::max(SignExtend(old, bytes * 8), SignExtend(operand, bytes * 8)));
                break;
            case 5: // LDSMIN
                result = static_cast<u64>(
                    std::min(SignExtend(old, bytes * 8), SignExtend(operand, bytes * 8)));
                break;
            case 6: // LDUMAX
                result = std::max(old, operand);
                break;
            case 7: // LDUMIN
                result = std::min(old, operand);
                break;
            default:
                return false;
            }
        } else {
            return false;
        }
        StoreVal(target_addr, result, bytes);
        WriteReg(x, rt, old);
        return true;
    }

    // --- CAS / CASA / CASL / CASAL -----------------------------------------
    // size(2) 001000 1 L 1 Rs o0 11111 Rn Rt
    if ((instr & 0x3FA07C00) == 0x08A07C00) {
        const u32 size = instr >> 30;
        const u32 bytes = 1u << size;
        const u32 rs = (instr >> 16) & 0x1F;
        const u64 mask = bytes == 8 ? ~u64{0} : ((u64{1} << (bytes * 8)) - 1);
        const u64 expected = ReadReg(x, rs, false) & mask;
        const u64 desired = ReadReg(x, rt, false) & mask;
        const u64 old = LoadZx(target_addr, bytes);
        if (old == expected) {
            StoreVal(target_addr, desired, bytes);
        }
        WriteReg(x, rs, old);
        return true;
    }

    // --- AdvSIMD load/store multiple structures (no writeback) -------------
    // 0 Q 0011000 L 000000 opcode size Rn Rt
    if ((instr & 0xBFBF0000) == 0x0C000000) {
        const bool q = (instr >> 30) & 1;
        const bool is_load = (instr >> 22) & 1;
        const u32 opcode = (instr >> 12) & 0xF;
        const u32 size = (instr >> 10) & 3;
        u32 reg_count;
        switch (opcode) {
        case 0x0: // LD/ST4
            reg_count = 4;
            break;
        case 0x2: // LD/ST1 x4
            reg_count = 4;
            break;
        case 0x4: // LD/ST3
            reg_count = 3;
            break;
        case 0x6: // LD/ST1 x3
            reg_count = 3;
            break;
        case 0x7: // LD/ST1 x1
            reg_count = 1;
            break;
        case 0x8: // LD/ST2
            reg_count = 2;
            break;
        case 0xA: // LD/ST1 x2
            reg_count = 2;
            break;
        default:
            return false;
        }
        const u32 lane_bytes = q ? 16 : 8;
        const bool deinterleave = (opcode == 0x0 || opcode == 0x4 || opcode == 0x8);
        const u32 element_bytes = 1u << size;
        if (!deinterleave) {
            // Contiguous: registers Rt, Rt+1, ... each holds one full vector.
            for (u32 r = 0; r < reg_count; ++r) {
                const u32 vr = (rt + r) & 31;
                if (is_load) {
                    std::memset(v + vr * 16, 0, 16);
                    CopyFromMemory(v + vr * 16, target_addr + r * lane_bytes, lane_bytes);
                } else {
                    CopyToMemory(target_addr + r * lane_bytes, v + vr * 16, lane_bytes);
                }
            }
        } else {
            // De-interleaved by element across reg_count registers.
            const u32 elements = lane_bytes / element_bytes;
            u64 addr = target_addr;
            if (is_load) {
                for (u32 r = 0; r < reg_count; ++r) {
                    std::memset(v + ((rt + r) & 31) * 16, 0, 16);
                }
            }
            for (u32 e = 0; e < elements; ++e) {
                for (u32 r = 0; r < reg_count; ++r) {
                    unsigned char* lane = v + ((rt + r) & 31) * 16 + e * element_bytes;
                    if (is_load) {
                        CopyFromMemory(lane, addr, element_bytes);
                    } else {
                        CopyToMemory(addr, lane, element_bytes);
                    }
                    addr += element_bytes;
                }
            }
        }
        return true;
    }

    // --- AdvSIMD load/store single structure (no writeback) ---------------
    // 0 Q 0011010 L R opcode S size Rn Rt   (R selects LD1..LD4 group)
    if ((instr & 0xBF9F0000) == 0x0D000000) {
        const bool is_load = (instr >> 22) & 1;
        const bool r_bit = (instr >> 21) & 1;
        const u32 opcode = (instr >> 13) & 7;
        const u32 s_bit = (instr >> 12) & 1;
        const u32 size = (instr >> 10) & 3;
        const bool q = (instr >> 30) & 1;

        // LD1R/LD2R/LD3R/LD4R replicate (opcode 110/111, load only).
        if (is_load && (opcode == 6 || opcode == 7)) {
            const u32 reg_count = (opcode == 6 ? (r_bit ? 2 : 1) : (r_bit ? 4 : 3));
            const u32 element_bytes = 1u << size;
            const u32 lanes = (q ? 16 : 8) / element_bytes;
            for (u32 r = 0; r < reg_count; ++r) {
                const u32 vr = (rt + r) & 31;
                std::memset(v + vr * 16, 0, 16);
                unsigned char elem[8]{};
                CopyFromMemory(elem, target_addr + r * element_bytes, element_bytes);
                for (u32 l = 0; l < lanes; ++l) {
                    std::memcpy(v + vr * 16 + l * element_bytes, elem, element_bytes);
                }
            }
            return true;
        }

        // Single-lane LD1/ST1..LD4/ST4. opcode high bits pick element size,
        // the low bit plus S/size encode the lane index.
        u32 element_bytes;
        u32 index;
        switch (opcode & 6) {
        case 0: // 8-bit
            element_bytes = 1;
            index = (q << 3) | (s_bit << 2) | size;
            break;
        case 2: // 16-bit
            element_bytes = 2;
            index = (q << 2) | (s_bit << 1) | (size >> 1);
            break;
        case 4: // 32-bit (size==0) or 64-bit (size==1, S==0)
            if ((size & 1) == 0) {
                element_bytes = 4;
                index = (q << 1) | s_bit;
            } else {
                element_bytes = 8;
                index = q;
            }
            break;
        default:
            return false;
        }
        const u32 reg_count = (opcode & 1) ? ((r_bit ? 4 : 3)) : ((r_bit ? 2 : 1));
        u64 addr = target_addr;
        for (u32 r = 0; r < reg_count; ++r) {
            unsigned char* lane = v + ((rt + r) & 31) * 16 + index * element_bytes;
            if (is_load) {
                CopyFromMemory(lane, addr, element_bytes);
            } else {
                CopyToMemory(addr, lane, element_bytes);
            }
            addr += element_bytes;
        }
        return true;
    }

    return false;
}

} // namespace Core::CPU
