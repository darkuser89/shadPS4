// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// FEXCore-based x86-64 -> ARM64 CPU backend for shadPS4 on Apple Silicon.
//
// This replaces the "run guest x86-64 code directly under Rosetta 2" model with
// native ARM64 execution via FEXCore's JIT. It is only built/used on ARCH_ARM64;
// x86-64 hosts continue to execute guest code directly.
//
// Build wiring, scalar HLE thunks, unresolved Aerolib fallbacks, per-thread guest
// state, and guest callbacks are integrated. Raw PS4 syscalls and full guest
// signal delivery still require dedicated translation in the frontend.

#include "common/types.h"

#include <span>

namespace Core {

struct EntryParams;

namespace CPU {

struct GuestRegisterContext {
    u64 rax{};
    u64 rbx{};
    u64 rcx{};
    u64 rdx{};
    u64 rsi{};
    u64 rdi{};
    u64 rbp{};
    u64 rsp{};
    u64 r8{};
    u64 r9{};
    u64 r10{};
    u64 r11{};
    u64 r12{};
    u64 r13{};
    u64 r14{};
    u64 r15{};
    u64 rip{};
    u64 fs{};
};

/// Owns the process-wide FEXCore context and runs guest threads through it.
class FexBackend {
public:
    static FexBackend& Instance();

    /// Lazily create + InitCore the FEXCore context (idempotent).
    /// Returns false if FEXCore could not be initialized.
    bool Initialize();

    /// Set up a guest thread at `entry_addr` mirroring the SysV/PS4 kernel entry
    /// layout (params on the guest stack, RDI=params, RSI=exit_func) and run it
    /// through FEXCore until it exits. Does not return for the main thread.
    [[noreturn]] void RunMainThread(VAddr entry_addr, EntryParams* params, void* exit_func);

    /// Call a guest x86-64 function (SysV ABI: RDI=a0, RSI=a1, RDX=a2) through
    /// FEXCore and return its result (RAX). Used for module_start / ctors that
    /// return, unlike the never-returning main thread. Uses a HLT return sentinel.
    u64 CallGuestFunction(VAddr entry_addr, u64 a0, u64 a1, u64 a2);

    /// Run a PS4 pthread entry function on the current native host thread using
    /// its actual guest stack. Each host thread receives an independent FEX
    /// thread state so translated guest threads can execute concurrently.
    u64 RunGuestThread(VAddr entry_addr, u64 arg, VAddr guest_stack_base,
                       u64 guest_stack_size);

    /// Return the x86 stack currently visible to translated guest code.  The
    /// main PS4 thread needs this because FEX cannot share its native ARM call
    /// stack with the emulated x86 stack.
    bool GetCurrentGuestStackBounds(VAddr& base, u64& size) const;

    /// Queue an asynchronous guest signal from a native Darwin signal frame.
    /// This is the signal-safe half of delivery: it only snapshots fixed TLS
    /// state and redirects the raw PC to FEX's pause trampoline.
    bool QueueGuestSignal(VAddr handler, u64 orbis_signum, s32 native_signum,
                          VAddr fault_address, void* native_context) noexcept;

    /// Run the queued guest handler after FEX's pause trampoline has spilled
    /// its live register allocation. Called by Ps4SyscallHandler::SleepThread,
    /// outside the native POSIX signal handler.
    void DispatchPendingGuestSignal();

    /// Snapshot the x86 register state currently held in the FEX CPU frame and
    /// invoke a guest signal handler without exposing an ARM ucontext to it.
    bool CaptureGuestContext(void* native_context, GuestRegisterContext& context);
    u64 InvokeGuestSignalHandler(VAddr handler, u64 signum, void* guest_context);

    /// Re-enter guest code from a synchronous native HLE callback while
    /// preserving the currently paused FEX thread state.
    u64 CallGuestCallback(VAddr entry_addr, std::span<const u64> args);
    u64 CallGuestCallback(VAddr entry_addr, u64 a0 = 0, u64 a1 = 0, u64 a2 = 0, u64 a3 = 0,
                          u64 a4 = 0, u64 a5 = 0);

    /// Clear FEX's live call/return prediction stack after a guest fiber swaps
    /// to an unrelated x86 stack. The architectural guest stack remains intact.
    void ResetCurrentCallRetStack();

    /// True when an address belongs to shadPS4-managed guest memory rather than
    /// a native host function.
    bool IsGuestAddress(VAddr address) const;

    /// Signal-dispatch hooks used to identify faults originating in generated
    /// FEX code and report the corresponding x86-64 guest RIP.
    bool HandleAccessViolation(void* context, void* fault_address);
    bool HandleIllegalInstruction(void* context);

private:
    FexBackend();
    ~FexBackend();

    struct Impl;
    Impl* impl;
};

} // namespace CPU
} // namespace Core
