// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "fiber.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <unordered_map>

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "core/cpu/fex_backend.h"
#include "core/memory.h"
#include "core/cpu/fex_hle.h"
#include "core/libraries/fiber/fiber_error.h"
#include "core/libraries/libs.h"
#include "core/tls.h"

namespace Libraries::Fiber {
namespace {

constexpr u32 FiberSignature0 = 0xdef1649c;
constexpr u32 FiberSignature1 = 0xb37592a0;
constexpr u32 FiberOptSignature = 0xbb40e64d;
constexpr u64 FiberStackSignature = 0x7149f2ca7149f2ca;
constexpr u64 FiberStackSizeCheck = 0xdeadbeefdeadbeef;
constexpr size_t NumGuestGprs = 16;

std::atomic<u32> context_size_check{};
constexpr size_t NumGuestXmmQwords = 16 * 4; // CPUState AVX layout: 4 qwords per register

struct SavedGuestContext {
    std::array<u64, NumGuestGprs> gprs{};
    std::array<u64, NumGuestXmmQwords> xmm{};
    u32 mxcsr{};
    u64* resume_arg{};
    bool valid{};
};

struct FiberThreadState {
    SavedGuestContext thread_context{};
    OrbisFiberContext api_context{};
    OrbisFiber* current{};
    bool running{};
};

thread_local FiberThreadState fiber_state{};

// A suspended fiber's context belongs to the FIBER, not to the thread that
// suspended it: sceFiber explicitly allows resuming a fiber from a different
// thread (the game's audio job system round-robins one fiber across its worker
// gang). Keeping these in a thread_local map made a cross-thread resume
// silently restart the fiber from its entry on the still-live fiber stack,
// corrupting it. Guarded by a global lock; the fiber state machine (Idle->Run
// CAS) already prevents one fiber from running on two threads at once.
std::mutex g_fiber_contexts_lock;
std::unordered_map<OrbisFiber*, SavedGuestContext> g_fiber_contexts;

void StoreFiberContext(OrbisFiber* fiber, const SavedGuestContext& context) {
    std::scoped_lock lk{g_fiber_contexts_lock};
    g_fiber_contexts[fiber] = context;
}

// Removes and returns the fiber's suspended context. Consuming semantics: a
// context can only be resumed once; suspension stores a fresh one.
bool TakeFiberContext(OrbisFiber* fiber, SavedGuestContext& out) {
    std::scoped_lock lk{g_fiber_contexts_lock};
    const auto it = g_fiber_contexts.find(fiber);
    if (it == g_fiber_contexts.end() || !it->second.valid) {
        return false;
    }
    out = it->second;
    g_fiber_contexts.erase(it);
    return true;
}

void EraseFiberContext(OrbisFiber* fiber) {
    std::scoped_lock lk{g_fiber_contexts_lock};
    g_fiber_contexts.erase(fiber);
}

bool IsValidFiber(const OrbisFiber* fiber) {
    return fiber && fiber->magic_start == FiberSignature0 && fiber->magic_end == FiberSignature1;
}

void InitializeStack(OrbisFiber* fiber) {
    if (!fiber->addr_context) {
        return;
    }
    auto* const begin = static_cast<u64*>(fiber->addr_context);
    *begin = FiberStackSignature;
    if (fiber->flags & FiberFlags::ContextSizeCheck) {
        std::fill(begin + 1, reinterpret_cast<u64*>(fiber->context_end), FiberStackSizeCheck);
    }
}

s32 AttachContext(OrbisFiber* fiber, void* address, u64 size) {
    if (!address || !size) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (reinterpret_cast<u64>(address) & 15) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (size < ORBIS_FIBER_CONTEXT_MINIMUM_SIZE) {
        return ORBIS_FIBER_ERROR_RANGE;
    }
    if ((size & 15) || fiber->addr_context) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    fiber->addr_context = address;
    fiber->size_context = size;
    fiber->context_start = address;
    fiber->context_end = static_cast<u8*>(address) + size;
    InitializeStack(fiber);
    return ORBIS_OK;
}

void SaveCurrent(SavedGuestContext& context, u64* resume_arg) {
    auto* const gprs = Core::CPU::g_fex_guest_gregs;
    ASSERT_MSG(gprs, "Saving a fiber without an active FEX guest context");
    std::copy_n(gprs, context.gprs.size(), context.gprs.begin());
    if (Core::CPU::g_fex_guest_xmm) {
        std::copy_n(Core::CPU::g_fex_guest_xmm, context.xmm.size(), context.xmm.begin());
    }
    if (Core::CPU::g_fex_guest_mxcsr) {
        context.mxcsr = *Core::CPU::g_fex_guest_mxcsr;
    }
    context.resume_arg = resume_arg;
    context.valid = true;
}

void Activate(SavedGuestContext& context, u64 arg_on_resume) {
    ASSERT_MSG(context.valid, "Activating an uninitialized FEX fiber context");
    if (context.resume_arg) {
        *context.resume_arg = arg_on_resume;
        context.resume_arg = nullptr;
    }
    std::copy(context.gprs.begin(), context.gprs.end(), Core::CPU::g_fex_guest_gregs);
    if (Core::CPU::g_fex_guest_xmm) {
        std::copy(context.xmm.begin(), context.xmm.end(), Core::CPU::g_fex_guest_xmm);
    }
    if (Core::CPU::g_fex_guest_mxcsr && context.mxcsr != 0) {
        *Core::CPU::g_fex_guest_mxcsr = context.mxcsr;
    }
    Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RAX] = ORBIS_OK;
    // Diagnostic: after this HLE returns, the thunk pops [rsp] and resumes
    // there. A non-executable resume target silently kills the guest thread in
    // FEX, so validate it here where the fiber context is still identifiable.
    const u64 resume_rsp = Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RSP];
    auto* memory = Core::Memory::Instance();
    u64 resume_rip = 0;
    bool executable = false;
    if (memory->IsValidMapping(resume_rsp, sizeof(u64))) {
        resume_rip = *reinterpret_cast<const u64*>(resume_rsp);
        void* start{};
        void* end{};
        u32 prot{};
        if (memory->QueryProtection(resume_rip, &start, &end, &prot) == 0) {
            executable = True(static_cast<Core::MemoryProt>(prot) & Core::MemoryProt::CpuExec);
        }
    }
    if (!executable && fiber_state.current) {
        LOG_ERROR(Lib_Fiber,
                  "Fiber resume target is not executable: thread='{}' rsp={:#x} rip={:#x} "
                  "(current fiber={} '{}')",
                  Common::GetCurrentThreadName(), resume_rsp, resume_rip,
                  fmt::ptr(fiber_state.current), fiber_state.current->name);
        Common::Log::Flush();
    }
}

std::atomic<u64> g_fiber_trace_count{0};

void TraceFiberApi(const char* api, const OrbisFiber* fiber) {
    const u64 n = g_fiber_trace_count.fetch_add(1, std::memory_order_relaxed);
    // Fibers cycle at frame/audio rate; keep only a startup window plus a
    // sparse sample so fiber activity stays visible without flooding the log.
    if (n < 128 || n % 8192 == 0) {
        LOG_INFO(Lib_Fiber, "[{}] {}: thread='{}' fiber={} '{}' entry={} state={}", n, api,
                 Common::GetCurrentThreadName(), fmt::ptr(fiber), fiber ? fiber->name : "-",
                 fiber ? fmt::ptr(reinterpret_cast<const void*>(fiber->entry)) : nullptr,
                 fiber ? static_cast<u32>(fiber->state.load()) : 0u);
    }
}

s32 FiberEntryReturned();

VAddr EntryReturnStub() {
    static const VAddr stub =
        Core::CPU::EmitHleStub(&Core::CPU::HleThunkT<FiberEntryReturned>::thunk);
    return stub;
}

void Start(OrbisFiber* fiber, u64 arg_on_run_to) {
    SavedGuestContext context{};

    u64 stack_top = fiber->addr_context
                        ? reinterpret_cast<u64>(fiber->addr_context) + fiber->size_context
                        : Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RSP];
    stack_top &= ~u64{15};

    // ThunkOp performs a guest pop after the native HLE returns. Put the fiber
    // entry in that pop slot and leave a normal return address for the entry.
    const u64 function_rsp = stack_top - sizeof(u64);
    const u64 pop_slot = function_rsp - sizeof(u64);
    *reinterpret_cast<u64*>(function_rsp) = EntryReturnStub();
    *reinterpret_cast<u64*>(pop_slot) = reinterpret_cast<u64>(fiber->entry);

    context.gprs.fill(0);
    context.gprs[Core::CPU::FEX_RSP] = pop_slot;
    context.gprs[Core::CPU::FEX_RDI] = fiber->arg_on_initialize;
    context.gprs[Core::CPU::FEX_RSI] = arg_on_run_to;
    context.valid = true;
    Activate(context, arg_on_run_to);
}

void SetCurrent(OrbisFiber* fiber) {
    fiber_state.current = fiber;
    fiber_state.api_context.current_fiber = fiber;
    if (auto* tcb = Core::GetTcbBase()) {
        tcb->tcb_fiber = fiber ? &fiber_state.api_context : nullptr;
    }
}

void ResumeFiber(OrbisFiber* fiber, u64 arg_on_run_to) {
    SavedGuestContext context{};
    if (TakeFiberContext(fiber, context)) {
        Activate(context, arg_on_run_to);
    } else {
        Start(fiber, arg_on_run_to);
    }
}

s32 ReturnToThread(u64 arg_on_return, u64* arg_on_run, bool entry_returned) {
    TraceFiberApi(entry_returned ? "EntryReturned" : "ReturnToThread", fiber_state.current);
    if (!fiber_state.running || !fiber_state.current) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    OrbisFiber* const current = fiber_state.current;
    if (!entry_returned && current->addr_context) {
        SavedGuestContext context{};
        SaveCurrent(context, arg_on_run);
        StoreFiberContext(current, context);
    } else {
        EraseFiberContext(current);
    }
    current->state = FiberState::Idle;

    SetCurrent(nullptr);
    fiber_state.running = false;
    Activate(fiber_state.thread_context, arg_on_return);
    return ORBIS_OK;
}

s32 FiberEntryReturned() {
    LOG_ERROR(Lib_Fiber, "Fiber entry returned without sceFiberReturnToThread");
    return ReturnToThread(0, nullptr, true);
}

} // namespace

s32 PS4_SYSV_ABI sceFiberInitializeImpl(OrbisFiber* fiber, const char* name, OrbisFiberEntry entry,
                                        u64 arg_on_initialize, void* addr_context, u64 size_context,
                                        const OrbisFiberOptParam* opt_param, u32 flags,
                                        u32 build_ver) {
    if (!fiber || !name || !entry) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((reinterpret_cast<u64>(fiber) & 7) || (reinterpret_cast<u64>(addr_context) & 15) ||
        (opt_param && (reinterpret_cast<u64>(opt_param) & 7))) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (size_context && size_context < ORBIS_FIBER_CONTEXT_MINIMUM_SIZE) {
        return ORBIS_FIBER_ERROR_RANGE;
    }
    if ((size_context & 15) || ((!addr_context) != (!size_context))) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (opt_param && opt_param->magic != FiberOptSignature) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    u32 user_flags = flags;
    if (build_ver >= Common::ElfInfo::FW_350) {
        user_flags |= FiberFlags::SetFpuRegs;
    }
    if (context_size_check.load()) {
        user_flags |= FiberFlags::ContextSizeCheck;
    }

    std::memset(fiber->name, 0, sizeof(fiber->name));
    std::strncpy(fiber->name, name, ORBIS_FIBER_MAX_NAME_LENGTH);
    fiber->entry = entry;
    fiber->arg_on_initialize = arg_on_initialize;
    fiber->addr_context = addr_context;
    fiber->size_context = size_context;
    fiber->context = nullptr;
    fiber->flags = user_flags;
    fiber->context_start = addr_context;
    fiber->context_end = addr_context ? static_cast<u8*>(addr_context) + size_context : nullptr;
    fiber->magic_start = FiberSignature0;
    fiber->magic_end = FiberSignature1;
    fiber->state = FiberState::Idle;
    InitializeStack(fiber);
    EraseFiberContext(fiber);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberInitialize(OrbisFiber* fiber, const char* name, OrbisFiberEntry entry,
                                    u64 arg_on_initialize, void* addr_context, u64 size_context,
                                    const OrbisFiberOptParam* opt_param, u32 build_ver) {
    return sceFiberInitializeImpl(fiber, name, entry, arg_on_initialize, addr_context, size_context,
                                  opt_param, 0, build_ver);
}

s32 PS4_SYSV_ABI sceFiberOptParamInitialize(OrbisFiberOptParam* opt_param) {
    if (!opt_param) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if (reinterpret_cast<u64>(opt_param) & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    opt_param->magic = FiberOptSignature;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberFinalize(OrbisFiber* fiber) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if (reinterpret_cast<u64>(fiber) & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (!IsValidFiber(fiber)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    FiberState expected = FiberState::Idle;
    if (!fiber->state.compare_exchange_strong(expected, FiberState::Terminated)) {
        return ORBIS_FIBER_ERROR_STATE;
    }
    EraseFiberContext(fiber);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberRunImpl(OrbisFiber* fiber, void* addr_context, u64 size_context,
                                 u64 arg_on_run_to, u64* arg_on_return) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((reinterpret_cast<u64>(fiber) & 7) || (reinterpret_cast<u64>(addr_context) & 15)) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (!IsValidFiber(fiber)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    TraceFiberApi("Run", fiber);
    if (fiber_state.running) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }
    if (addr_context || size_context) {
        const s32 result = AttachContext(fiber, addr_context, size_context);
        if (result < 0) {
            return result;
        }
    }
    FiberState expected = FiberState::Idle;
    if (!fiber->state.compare_exchange_strong(expected, FiberState::Run)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

    SaveCurrent(fiber_state.thread_context, arg_on_return);
    fiber_state.running = true;
    SetCurrent(fiber);
    ResumeFiber(fiber, arg_on_run_to);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberRun(OrbisFiber* fiber, u64 arg_on_run_to, u64* arg_on_return) {
    return sceFiberRunImpl(fiber, nullptr, 0, arg_on_run_to, arg_on_return);
}

s32 PS4_SYSV_ABI sceFiberSwitchImpl(OrbisFiber* fiber, void* addr_context, u64 size_context,
                                    u64 arg_on_run_to, u64* arg_on_run) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((reinterpret_cast<u64>(fiber) & 7) || (reinterpret_cast<u64>(addr_context) & 15)) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (!IsValidFiber(fiber)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    TraceFiberApi("Switch", fiber);
    if (!fiber_state.running || !fiber_state.current) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }
    if (addr_context || size_context) {
        const s32 result = AttachContext(fiber, addr_context, size_context);
        if (result < 0) {
            return result;
        }
    }
    FiberState expected = FiberState::Idle;
    if (!fiber->state.compare_exchange_strong(expected, FiberState::Run)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

    OrbisFiber* const previous = fiber_state.current;
    if (previous->addr_context) {
        SavedGuestContext context{};
        SaveCurrent(context, arg_on_run);
        StoreFiberContext(previous, context);
    } else {
        EraseFiberContext(previous);
    }
    previous->state = FiberState::Idle;

    SetCurrent(fiber);
    ResumeFiber(fiber, arg_on_run_to);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberSwitch(OrbisFiber* fiber, u64 arg_on_run_to, u64* arg_on_run) {
    return sceFiberSwitchImpl(fiber, nullptr, 0, arg_on_run_to, arg_on_run);
}

s32 PS4_SYSV_ABI sceFiberGetSelf(OrbisFiber** fiber) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if (!fiber_state.running || !fiber_state.current) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }
    *fiber = fiber_state.current;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberReturnToThread(u64 arg_on_return, u64* arg_on_run) {
    return ReturnToThread(arg_on_return, arg_on_run, false);
}

s32 PS4_SYSV_ABI sceFiberGetInfo(OrbisFiber* fiber, OrbisFiberInfo* info) {
    if (!fiber || !info) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((reinterpret_cast<u64>(fiber) & 7) || (reinterpret_cast<u64>(info) & 7)) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (!IsValidFiber(fiber) || info->size != sizeof(*info)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    info->entry = fiber->entry;
    info->arg_on_initialize = fiber->arg_on_initialize;
    info->addr_context = fiber->addr_context;
    info->size_context = fiber->size_context;
    std::strncpy(info->name, fiber->name, ORBIS_FIBER_MAX_NAME_LENGTH);
    info->size_context_margin = ~u64{};
    if ((fiber->flags & FiberFlags::ContextSizeCheck) && fiber->addr_context) {
        auto* begin = static_cast<u64*>(fiber->context_start);
        auto* cursor = begin + 1;
        auto* const end = static_cast<u64*>(fiber->context_end);
        while (cursor < end && *cursor == FiberStackSizeCheck) {
            ++cursor;
        }
        info->size_context_margin =
            reinterpret_cast<u64>(cursor) - reinterpret_cast<u64>(begin + 1);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberStartContextSizeCheck(u32 flags) {
    if (flags) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    u32 expected{};
    return context_size_check.compare_exchange_strong(expected, 1) ? ORBIS_OK
                                                                   : ORBIS_FIBER_ERROR_STATE;
}

s32 PS4_SYSV_ABI sceFiberStopContextSizeCheck() {
    u32 expected{1};
    return context_size_check.compare_exchange_strong(expected, 0) ? ORBIS_OK
                                                                   : ORBIS_FIBER_ERROR_STATE;
}

s32 PS4_SYSV_ABI sceFiberRename(OrbisFiber* fiber, const char* name) {
    if (!fiber || !name) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if (reinterpret_cast<u64>(fiber) & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (!IsValidFiber(fiber)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    std::strncpy(fiber->name, name, ORBIS_FIBER_MAX_NAME_LENGTH);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberGetThreadFramePointerAddress(u64* frame_pointer) {
    if (!frame_pointer) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if (!fiber_state.running) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }
    *frame_pointer = Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RBP];
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("hVYD7Ou2pCQ", "libSceFiber", 1, "libSceFiber", sceFiberInitialize);
    LIB_FUNCTION("7+OJIpko9RY", "libSceFiber", 1, "libSceFiber", sceFiberInitializeImpl);
    LIB_FUNCTION("asjUJJ+aa8s", "libSceFiber", 1, "libSceFiber", sceFiberOptParamInitialize);
    LIB_FUNCTION("JeNX5F-NzQU", "libSceFiber", 1, "libSceFiber", sceFiberFinalize);
    LIB_FUNCTION("a0LLrZWac0M", "libSceFiber", 1, "libSceFiber", sceFiberRun);
    LIB_FUNCTION("PFT2S-tJ7Uk", "libSceFiber", 1, "libSceFiber", sceFiberSwitch);
    LIB_FUNCTION("p+zLIOg27zU", "libSceFiber", 1, "libSceFiber", sceFiberGetSelf);
    LIB_FUNCTION("B0ZX2hx9DMw", "libSceFiber", 1, "libSceFiber", sceFiberReturnToThread);
    LIB_FUNCTION("avfGJ94g36Q", "libSceFiber", 1, "libSceFiber", sceFiberRunImpl);
    LIB_FUNCTION("ZqhZFuzKT6U", "libSceFiber", 1, "libSceFiber", sceFiberSwitchImpl);
    LIB_FUNCTION("uq2Y5BFz0PE", "libSceFiber", 1, "libSceFiber", sceFiberGetInfo);
    LIB_FUNCTION("Lcqty+QNWFc", "libSceFiber", 1, "libSceFiber", sceFiberStartContextSizeCheck);
    LIB_FUNCTION("Kj4nXMpnM8Y", "libSceFiber", 1, "libSceFiber", sceFiberStopContextSizeCheck);
    LIB_FUNCTION("JzyT91ucGDc", "libSceFiber", 1, "libSceFiber", sceFiberRename);
    LIB_FUNCTION("0dy4JtMUcMQ", "libSceFiber", 1, "libSceFiber",
                 sceFiberGetThreadFramePointerAddress);
}

} // namespace Libraries::Fiber
