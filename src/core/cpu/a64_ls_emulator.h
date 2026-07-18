// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::CPU {

// Software emulation of a single A64 load/store-class instruction against a
// substituted target address. Used to service guest accesses to canonical PS4
// fixed addresses that fall inside the macOS GPU carveout: the faulting
// instruction is decoded and its memory access performed at the relocated
// address instead, then execution resumes after the instruction.
//
// `x` points at 32 u64 slots: x0..x30 followed by SP. Register 31 is treated
// as SP for base registers and as XZR for data registers, per the A64 rules.
// `v` points at the 32 NEON registers (16 bytes each).
//
// Returns true when the instruction was recognized and fully emulated (the
// caller must then advance the host PC by 4). Unrecognized or unsupported
// encodings (writeback forms, exclusives) return false and leave all state
// untouched.
bool EmulateA64LoadStore(u32 instr, u64 target_addr, u64* x, unsigned char* v);

} // namespace Core::CPU
