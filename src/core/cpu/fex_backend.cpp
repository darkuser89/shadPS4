// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FEXCore CPU backend implementation. See fex_backend.h for status/scope.
// Every setup step here is validated in the standalone FEXCore macOS harness.

#include <csignal>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/cpu/a64_ls_emulator.h"
#include "core/cpu/fex_backend.h"
#include "core/cpu/fex_thunks.h" // g_guest_frame, FexThunkHandler
#include "core/libraries/kernel/threads/exception.h"
#include "core/linker.h" // EntryParams
#include "core/memory.h"
#include "core/signals.h"
#include "core/tls.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>
#include <sys/mman.h>
#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#endif
#include <unistd.h>

// FEXCore embedding API. Requires libFEXCore.a on the include/link path (TODO: build wiring).
#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/HostFeatures.h>
#include <FEXCore/Core/SignalDelegator.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Debug/InternalThreadState.h>
#include <FEXCore/HLE/SyscallHandler.h>
#include <FEXCore/Utils/ArchHelpers/Arm64.h>
#include <FEXCore/fextl/memory.h>

namespace Core::CPU {

// Published to the FEXCore-free thunk marshalling (fex_hle.h).
thread_local uint64_t* g_fex_guest_gregs = nullptr;
thread_local uint64_t* g_fex_guest_xmm = nullptr;
thread_local uint32_t* g_fex_guest_mxcsr = nullptr;
thread_local const char* g_current_fex_hle_signature = nullptr;
thread_local FEXCore::Core::InternalThreadState* g_current_fex_thread = nullptr;
thread_local VAddr g_current_guest_stack_base = 0;
thread_local u64 g_current_guest_stack_size = 0;

[[noreturn]] void UnsupportedHleAbi(const char* signature) {
    UNREACHABLE_MSG("Unsupported FEX HLE ABI: {}", signature);
}

namespace {

constexpr u64 GuestStackSize = 16 * 1024 * 1024;
constexpr u64 HleStubArenaSize = 64 * 1024 * 1024;

// Guest address of a `0F 3E` (FEX CALLBACKRET) instruction. FEXCore pushes this
// as the return address before running a re-entrant guest callback
// (HandleCallback/ExecuteJITCallback); when the callback returns it hits this
// instruction and FEX cleanly unwinds back to the native thunk. The base
// SignalDelegator returns 0 for this, which would make a callback `ret` jump to
// RIP 0 and fault. Set once in Initialize().
std::atomic<uint64_t> g_callback_ret_addr{0};

#if defined(__APPLE__) && defined(__aarch64__)
constexpr sig_atomic_t PendingSignalIdle = 0;
constexpr sig_atomic_t PendingSignalPreparing = 1;
constexpr sig_atomic_t PendingSignalQueued = 2;
constexpr sig_atomic_t PendingSignalReturning = 3;

using DarwinMcontext = std::remove_pointer_t<mcontext_t>;

// All data touched by SigactionHandler lives in one fixed, trivially initialized
// TLS object. The handler publishes a complete snapshot by setting `phase` to
// PendingSignalQueued last; no allocation, logging, or locking occurs there.
struct PendingGuestSignal {
    volatile sig_atomic_t phase{PendingSignalIdle};
    VAddr handler{};
    u64 orbis_signum{};
    VAddr fault_address{};
    VAddr host_pc{};
    bool was_in_jit{};
    bool resume_state_valid{};
    sigset_t saved_signal_mask{};
    DarwinMcontext saved_mcontext{};
    std::array<std::byte, sizeof(FEXCore::Core::CPUState)> saved_cpu_state{};
    std::array<std::byte, sizeof(FEXCore::Core::CPUState)> resume_cpu_state{};
    Libraries::Kernel::Ucontext orbis_context{};
};

thread_local PendingGuestSignal g_pending_guest_signal{};
#endif

bool FexAccessViolationHandler(void* context, void* fault_address) {
    return FexBackend::Instance().HandleAccessViolation(context, fault_address);
}

bool FexIllegalInstructionHandler(void* context) {
    return FexBackend::Instance().HandleIllegalInstruction(context);
}

bool AppleArmFeature(const char* name) {
    u64 value = 0;
    size_t size = sizeof(value);
    return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}

u32 AppleCacheLineSize() {
    u64 value = 0;
    size_t size = sizeof(value);
    if (::sysctlbyname("hw.cachelinesize", &value, &size, nullptr, 0) == 0 && value != 0 &&
        value <= std::numeric_limits<u32>::max()) {
        return static_cast<u32>(value);
    }
    return 64;
}

// FEX's normal feature discovery reads privileged ARM identification registers, which SIGILL from
// EL0 on macOS. Query Apple's stable hw.optional.arm sysctls instead. Missing keys stay disabled,
// so one binary remains safe on M1 while automatically using newer M-series instructions.
FEXCore::HostFeatures MakeAppleSiliconHostFeatures() {
    FEXCore::HostFeatures HF{};
    HF.DCacheLineSize = AppleCacheLineSize();
    HF.ICacheLineSize = HF.DCacheLineSize;
    HF.SupportsAtomics = AppleArmFeature("hw.optional.arm.FEAT_LSE") ||
                         AppleArmFeature("hw.optional.armv8_1_atomics");
    HF.SupportsRCPC = AppleArmFeature("hw.optional.arm.FEAT_LRCPC");
    HF.SupportsTSOImm9 = AppleArmFeature("hw.optional.arm.FEAT_LRCPC2");
    HF.SupportsAES = AppleArmFeature("hw.optional.arm.FEAT_AES");
    HF.SupportsCRC = AppleArmFeature("hw.optional.arm.FEAT_CRC32") ||
                     AppleArmFeature("hw.optional.armv8_crc32");
    HF.SupportsSHA = AppleArmFeature("hw.optional.arm.FEAT_SHA1") &&
                     AppleArmFeature("hw.optional.arm.FEAT_SHA256");
    HF.SupportsPMULL_128Bit = AppleArmFeature("hw.optional.arm.FEAT_PMULL");
    HF.SupportsFCMA = AppleArmFeature("hw.optional.arm.FEAT_FCMA");
    HF.SupportsFlagM = AppleArmFeature("hw.optional.arm.FEAT_FlagM");
    HF.SupportsFlagM2 = AppleArmFeature("hw.optional.arm.FEAT_FlagM2");
    HF.SupportsFRINTTS = AppleArmFeature("hw.optional.arm.FEAT_FRINTTS");
    HF.SupportsAFP = AppleArmFeature("hw.optional.arm.FEAT_AFP");
    HF.SupportsRPRES = AppleArmFeature("hw.optional.arm.FEAT_RPRES");
    HF.SupportsCSSC = AppleArmFeature("hw.optional.arm.FEAT_CSSC");
    HF.SupportsMOPS = AppleArmFeature("hw.optional.arm.FEAT_MOPS");
    HF.SupportsAVX = true; // PS4 guests require AVX; FEX emulates it
    HF.SupportsAES256 = HF.SupportsAVX && HF.SupportsAES;
    HF.SupportsPreserveAllABI = true;
    HF.Supports3DNow = true;
    HF.SupportsSSE4a = true; // PS4 Jaguar has SSE4a
    HF.CPUMIDRs.push_back(0);
    return HF;
}

// Minimal SyscallHandler. PS4 guest `syscall` instructions must route into
// shadPS4's kernel HLE here. TODO: dispatch Args to Libraries::Kernel.
class Ps4SyscallHandler final : public FEXCore::HLE::SyscallHandler,
                                public FEXCore::Allocator::FEXAllocOperators {
public:
    uint64_t HandleSyscall(FEXCore::Core::CpuStateFrame* Frame,
                           FEXCore::HLE::SyscallArguments* Args) override {
        // TODO: translate to shadPS4 kernel HLE (sceKernel* / raw syscalls).
        UNREACHABLE_MSG("Guest raw syscall not yet routed to PS4 kernel HLE");
        return 0;
    }
    void SleepThread(FEXCore::Context::Context*, FEXCore::Core::CpuStateFrame*) override {
        FexBackend::Instance().DispatchPendingGuestSignal();
    }
    std::optional<FEXCore::ExecutableFileSectionInfo> LookupExecutableFileSection(
        FEXCore::Core::InternalThreadState*, uint64_t) override {
        return std::nullopt;
    }
    FEXCore::HLE::ExecutableRangeInfo QueryGuestExecutableRange(FEXCore::Core::InternalThreadState*,
                                                                uint64_t address) override {
        // HLE stubs and the callback-return/HLT sentinel live in host mappings outside
        // shadPS4's guest VMA map, but contain x86 instructions executed by FEX.
        const auto& arena = GetHleStubArenaState();
        const uint64_t arena_base = reinterpret_cast<uint64_t>(arena.base);
        if (arena.base != MAP_FAILED && address >= arena_base &&
            address < arena_base + HleStubArenaSize) {
            return {arena_base, HleStubArenaSize, true};
        }

        const uint64_t callback_ret = g_callback_ret_addr.load(std::memory_order_acquire);
        if (callback_ret != 0) {
            const uint64_t page_size = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
            const uint64_t sentinel_base = callback_ret & ~(page_size - 1);
            if (address >= sentinel_base && address < sentinel_base + page_size) {
                return {sentinel_base, page_size, true};
            }
        }

        void* start{};
        void* end{};
        u32 raw_prot{};
        if (Core::Memory::Instance()->QueryProtection(address, &start, &end, &raw_prot) != 0) {
            return {0, 0, false};
        }

        const auto prot = static_cast<Core::MemoryProt>(raw_prot);
        if (False(prot & Core::MemoryProt::CpuExec)) {
            return {0, 0, false};
        }
        return {reinterpret_cast<uint64_t>(start),
                reinterpret_cast<uint64_t>(end) - reinterpret_cast<uint64_t>(start),
                True(prot & Core::MemoryProt::CpuWrite)};
    }
};

// SignalDelegator whose TLS registration we drive manually before execution.
// TODO: a real delegator translating macOS signals -> guest exceptions.
class Ps4SignalDelegator final : public FEXCore::SignalDelegator,
                                 public FEXCore::Allocator::FEXAllocOperators {
public:
    void RegisterTLSState(FEXCore::Core::InternalThreadState* Thread) {
        TLSThread = Thread;
    }
    void UninstallTLSState(FEXCore::Core::InternalThreadState*) {
        TLSThread = nullptr;
    }

    uintptr_t GetThunkCallbackRET() const override {
        return g_callback_ret_addr.load(std::memory_order_acquire);
    }

protected:
    FEXCore::Core::InternalThreadState* GetTLSThread() {
        return TLSThread;
    }

private:
    static thread_local FEXCore::Core::InternalThreadState* TLSThread;
};
thread_local FEXCore::Core::InternalThreadState* Ps4SignalDelegator::TLSThread = nullptr;

} // namespace

struct FexBackend::Impl {
    fextl::unique_ptr<FEXCore::Context::Context> ctx;
    fextl::unique_ptr<Ps4SignalDelegator> signals;
    fextl::unique_ptr<Ps4SyscallHandler> syscalls;
    FexThunkHandler thunks;
    std::mutex init_mutex;
    bool initialized = false;

    // One persistent guest thread reused for all guest execution (module_start,
    // ctors, main). Guest stack + a HLT return sentinel so guest `ret` cleanly
    // stops ExecuteThread.
    FEXCore::Core::InternalThreadState* thread = nullptr;
    void* guest_stack_base = nullptr;
    uint64_t guest_stack_top = 0;
    void* sentinel_base = nullptr;
    uint64_t sentinel = 0; // guest addr of an HLT instruction
    void* main_callret_mapping = nullptr;
    size_t main_callret_mapping_size = 0;

    struct CallRetAllocation {
        void* mapping;
        size_t size;
    };

    // Guest code that walks the conventional x86-64 RBP chain expects the
    // outermost frame to contain both a null previous-frame pointer and a null
    // return address. A translated thread has no native x86 caller to provide
    // that frame, so install the ABI boundary explicitly. The actual entry
    // return address remains immediately below this frame.
    static VAddr InstallRootFrame(VAddr aligned_stack_top) {
        ASSERT((aligned_stack_top & 0xF) == 0);
        const VAddr root_frame = aligned_stack_top - 2 * sizeof(u64);
        auto* words = reinterpret_cast<u64*>(root_frame);
        words[0] = 0;
        words[1] = 0;
        return root_frame;
    }

    // Configure a freshly created guest thread's segment/mode/return-stack state.
    // These four items are all mandatory and were each isolated in the harness.
    CallRetAllocation SetupThread(FEXCore::Core::InternalThreadState* thread, VAddr rip) {
        auto* frame = thread->CurrentFrame;

        // GDT/LDT with a 64-bit (Long mode) code segment. FEX derives 64-bit-ness
        // from CS.L, NOT from CONFIG_IS64BIT_MODE; without L=1 it decodes as 32-bit
        // and truncates the guest RIP to 32 bits (fatal once code lives >4GB).
        static thread_local FEXCore::Core::CPUState::gdt_segment gdt[32]{};
        frame->State.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_GDT] = &gdt[0];
        frame->State.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_LDT] = &gdt[0];
        frame->State.cs_idx = FEXCore::Core::CPUState::DEFAULT_USER_CS << 3;
        auto* cs = FEXCore::Core::CPUState::GetSegmentFromIndex(frame->State, frame->State.cs_idx);
        FEXCore::Core::CPUState::SetGDTBase(cs, 0);
        FEXCore::Core::CPUState::SetGDTLimit(cs, 0xF'FFFFU);
        cs->L = 1; // 64-bit
        cs->D = 0;
        frame->State.cs_cached = FEXCore::Core::CPUState::CalculateGDTBase(*cs);
        frame->State.rip = rip;

        // Call-return shadow stack (return prediction, REG_CALLRET_SP). FEXCore
        // does not allocate this; the embedder must. Map RW directly — on macOS a
        // PROT_NONE mmap caps max-protection and a later mprotect->RW gives SIGBUS.
        constexpr size_t CRSize = FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE;
        const size_t host_page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        const size_t mapping_size = CRSize + 2 * host_page_size;
        void* cr = ::mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_MSG(cr != MAP_FAILED, "Failed to allocate FEX call-return stack");
        thread->CallRetStackBase =
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(cr) + host_page_size);
        frame->State.callret_sp = reinterpret_cast<uint64_t>(thread->CallRetStackBase) + CRSize / 4;
        frame->State.fs_cached = reinterpret_cast<uint64_t>(Core::GetTcbBase());
        return {cr, mapping_size};
    }

    void BeginExecution(FEXCore::Core::InternalThreadState* guest_thread, VAddr stack_base,
                        u64 stack_size) {
        auto& state = guest_thread->CurrentFrame->State;
        state.fs_cached = reinterpret_cast<uint64_t>(Core::GetTcbBase());
#if defined(__APPLE__) && defined(__aarch64__)
        // Resolve the pending-signal TLS before a POSIX signal can interrupt
        // this guest thread.  Do not reset it here: signal delivery temporarily
        // executes an isolated callback thread and must retain the interrupted
        // thread's queued snapshot across both BeginExecution calls.
        (void)g_pending_guest_signal.phase;
#endif
        signals->RegisterTLSState(guest_thread);
        g_fex_guest_gregs = state.gregs;
        g_fex_guest_xmm = &state.xmm.avx.data[0][0];
        g_fex_guest_mxcsr = &state.mxcsr;
        g_current_fex_thread = guest_thread;
        g_current_guest_stack_base = stack_base;
        g_current_guest_stack_size = stack_size;
    }

    void EndExecution(FEXCore::Core::InternalThreadState* guest_thread) {
        signals->UninstallTLSState(guest_thread);
        g_fex_guest_gregs = nullptr;
        g_fex_guest_xmm = nullptr;
        g_fex_guest_mxcsr = nullptr;
        g_current_fex_thread = nullptr;
        g_current_guest_stack_base = 0;
        g_current_guest_stack_size = 0;
    }

    static void ResetCallRetStack(FEXCore::Core::InternalThreadState* guest_thread) {
        guest_thread->CurrentFrame->State.callret_sp =
            reinterpret_cast<uint64_t>(guest_thread->CallRetStackBase) +
            FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE / 4;
    }
};

FexBackend::FexBackend() : impl(new Impl()) {}
FexBackend::~FexBackend() {
    if (impl->ctx && impl->thread) {
        impl->ctx->DestroyThread(impl->thread);
    }
    if (impl->main_callret_mapping) {
        ::munmap(impl->main_callret_mapping, impl->main_callret_mapping_size);
    }
    if (impl->guest_stack_base) {
        ::munmap(impl->guest_stack_base, 16 * 1024 * 1024);
    }
    if (impl->sentinel_base) {
        ::munmap(impl->sentinel_base, static_cast<size_t>(::sysconf(_SC_PAGESIZE)));
    }
    delete impl;
}

FexBackend& FexBackend::Instance() {
    static FexBackend instance;
    return instance;
}

bool FexBackend::Initialize() {
    std::scoped_lock lock{impl->init_mutex};
    if (impl->initialized) {
        return true;
    }

    // NB: do NOT call FEXCore::Allocator::SetupHooks() — that installs FEX's
    // Linux 64-bit OS allocator (guest VA reservation), which crashes on macOS
    // and which shadPS4 does not need (shadPS4 owns guest memory itself).
    FEXCore::Config::Initialize();
    FEXCore::Config::Load();
    FEXCore::Config::Set(FEXCore::Config::CONFIG_IS64BIT_MODE, "1");
    FEXCore::Config::Set(FEXCore::Config::CONFIG_TSOENABLED, "1");
    // x86-TSO orders vector (SSE/AVX) and rep-string stores exactly like scalar
    // stores. With these left unordered, a consumer thread can observe a
    // scalar publish flag before the vector/memcpy-written payload it guards
    // and read stale job data (observed: Phyre worker gangs interpreting a
    // stale job slot as a shutdown command and exiting, deadlocking the gang).
    FEXCore::Config::Set(FEXCore::Config::CONFIG_VECTORTSOENABLED, "1");
    FEXCore::Config::Set(FEXCore::Config::CONFIG_MEMCPYSETTSOENABLED, "1");
    // Unaligned atomics crossing a 16-byte/cacheline boundary tear without the
    // strict in-process split-lock (real x86 bus-locks them). Torn pointer
    // reads from lock-free job queues are exactly the kind of corruption this
    // produces.
    FEXCore::Config::Set(FEXCore::Config::CONFIG_STRICTINPROCESSSPLITLOCKS, "1");
    // Escape hatches for performance experiments; correctness requires "1".
    if (const char* vector_tso = std::getenv("FEX_VECTORTSO")) {
        FEXCore::Config::Set(FEXCore::Config::CONFIG_VECTORTSOENABLED, vector_tso);
        LOG_INFO(Core_Linker, "FEXCore: FEX_VECTORTSO={}", vector_tso);
    }
    if (const char* memcpy_tso = std::getenv("FEX_MEMCPYSETTSO")) {
        FEXCore::Config::Set(FEXCore::Config::CONFIG_MEMCPYSETTSOENABLED, memcpy_tso);
        LOG_INFO(Core_Linker, "FEXCore: FEX_MEMCPYSETTSO={}", memcpy_tso);
    }

    // The embedded FEXCore setup does not install FEX's normal environment
    // configuration layer. Honour the two block-formation controls explicitly
    // so JIT regressions can be isolated without rebuilding shadPS4.
    if (const char* max_inst = std::getenv("FEX_MAXINST")) {
        FEXCore::Config::Set(FEXCore::Config::CONFIG_MAXINST, max_inst);
        LOG_INFO(Core_Linker, "FEXCore: FEX_MAXINST={}", max_inst);
    }
    if (const char* multiblock = std::getenv("FEX_MULTIBLOCK")) {
        FEXCore::Config::Set(FEXCore::Config::CONFIG_MULTIBLOCK, multiblock);
        LOG_INFO(Core_Linker, "FEXCore: FEX_MULTIBLOCK={}", multiblock);
    }

    auto features = MakeAppleSiliconHostFeatures();
    LOG_INFO(Core_Linker,
             "FEXCore Apple host: cache line {} B, LSE={}, LRCPC={}, LRCPC2={}, AFP={}, "
             "RPRES={}, CSSC={}, MOPS={}",
             features.DCacheLineSize, features.SupportsAtomics, features.SupportsRCPC,
             features.SupportsTSOImm9, features.SupportsAFP, features.SupportsRPRES,
             features.SupportsCSSC, features.SupportsMOPS);
    impl->ctx = FEXCore::Context::Context::CreateNewContext(features);
    if (!impl->ctx) {
        LOG_ERROR(Core_Linker, "FEXCore: CreateNewContext failed");
        return false;
    }

    impl->signals = fextl::make_unique<Ps4SignalDelegator>();
    impl->syscalls = fextl::make_unique<Ps4SyscallHandler>();
    impl->ctx->SetSignalDelegator(impl->signals.get());
    impl->ctx->SetSyscallHandler(impl->syscalls.get());
#if defined(__APPLE__) && defined(__aarch64__)
    // macOS reserves the PS4's canonical 0x10-0x6f billion user interval for
    // Apple GPU mappings. Keep guest pointers canonical and translate only
    // FEX JIT data accesses to shadPS4's linear relocated mapping. This avoids
    // handling every load/store through SIGSEGV.
    auto* memory = Core::Memory::Instance();
    const uint64_t relocated_start = memory->GuestMemoryRemapBase();
    const int64_t relocation_delta = static_cast<int64_t>(
        relocated_start - Core::CanonicalGuestRemapStart);
    impl->ctx->SetGuestMemoryAddressRemap(Core::CanonicalGuestRemapStart,
                                         Core::CanonicalGuestRemapEnd, relocation_delta);
    LOG_INFO(Core_Linker, "FEXCore canonical guest remap: {:#x}-{:#x} -> {:#x}-{:#x}",
             Core::CanonicalGuestRemapStart, Core::CanonicalGuestRemapEnd, relocated_start,
             relocated_start + Core::CanonicalGuestRemapSize);
#endif
    impl->ctx->SetThunkHandler(&impl->thunks); // HLE guest->native boundary (0F 3F stubs)
    impl->ctx->SetHardwareTSOSupport(false);
    LOG_INFO(Core_Linker, "FEXCore memory model: software TSO (macOS barrier mode)");
    impl->ctx->EnableExitOnHLT();              // guest `ret` -> HLT sentinel -> stop

    if (!impl->ctx->InitCore()) {
        LOG_ERROR(Core_Linker, "FEXCore: InitCore failed");
        return false;
    }

    // Run after shadPS4's memory-tracking handlers. These hooks currently add
    // guest-RIP diagnostics; handled memory faults remain owned by their
    // existing subsystem-specific handlers.
    constexpr u32 SignalPriority = 100;
    Core::Signals::Instance()->RegisterAccessViolationHandler(FexAccessViolationHandler,
                                                              SignalPriority);
    Core::Signals::Instance()->RegisterIllegalInstructionHandler(FexIllegalInstructionHandler,
                                                                 SignalPriority);

    // Persistent guest thread + stack + HLT return sentinel.
    impl->guest_stack_base =
        ::mmap(nullptr, GuestStackSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_MSG(impl->guest_stack_base != MAP_FAILED, "Failed to allocate guest stack");
    impl->guest_stack_top =
        (reinterpret_cast<uint64_t>(impl->guest_stack_base) + GuestStackSize - 256) & ~uint64_t(15);

    const size_t host_page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    impl->sentinel_base =
        ::mmap(nullptr, host_page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_MSG(impl->sentinel_base != MAP_FAILED, "Failed to allocate HLT sentinel");
    *reinterpret_cast<uint8_t*>(impl->sentinel_base) = 0xF4; // hlt
    impl->sentinel = reinterpret_cast<uint64_t>(impl->sentinel_base);

    // Callback-return trampoline in the same executable page: a `0F 3E`
    // (FEX CALLBACKRET) instruction. Ps4SignalDelegator::GetThunkCallbackRET()
    // hands this to FEXCore so a re-entrant guest callback's `ret` unwinds
    // cleanly instead of jumping to RIP 0.
    auto* sentinel_bytes = reinterpret_cast<uint8_t*>(impl->sentinel_base);
    sentinel_bytes[8] = 0x0F;
    sentinel_bytes[9] = 0x3E;
    g_callback_ret_addr.store(reinterpret_cast<uint64_t>(impl->sentinel_base) + 8,
                              std::memory_order_release);

    impl->thread = impl->ctx->CreateThread(0, impl->guest_stack_top);
    ASSERT_MSG(impl->thread, "FEXCore CreateThread failed");
    const auto callret = impl->SetupThread(impl->thread, 0);
    impl->main_callret_mapping = callret.mapping;
    impl->main_callret_mapping_size = callret.size;

    impl->initialized = true;
    LOG_INFO(Core_Linker,
             "FEXCore CPU backend initialized (native ARM64, no Rosetta); guest stack "
             "{:#x}..{:#x}, HLT sentinel {:#x}",
             reinterpret_cast<u64>(impl->guest_stack_base), impl->guest_stack_top, impl->sentinel);
    return true;
}

u64 FexBackend::CallGuestFunction(VAddr entry_addr, u64 a0, u64 a1, u64 a2) {
    // A guest module may be loaded and started synchronously from an HLE call
    // while another guest frame is paused (for example sceKernelLoadStartModule).
    // Re-enter that frame through FEX's callback trampoline; starting a second
    // top-level ExecuteThread here would overwrite and then clear the outer TLS
    // register pointers before its HLE thunk can write the return value.
    if (g_current_fex_thread) {
        return CallGuestCallback(entry_addr, a0, a1, a2);
    }
    if (!Initialize()) {
        UNREACHABLE_MSG("Failed to initialize FEXCore for guest function {:#x}", entry_addr);
    }
    auto& s = impl->thread->CurrentFrame->State;

    // Push the HLT sentinel as the return address, then set args (SysV) and RIP.
    const VAddr root_frame = impl->InstallRootFrame(impl->guest_stack_top);
    uint64_t rsp = root_frame;
    rsp -= 8;
    *reinterpret_cast<uint64_t*>(rsp) = impl->sentinel;
    s.gregs[FEXCore::X86State::REG_RSP] = rsp;
    s.gregs[FEXCore::X86State::REG_RBP] = root_frame;
    s.gregs[FEXCore::X86State::REG_RDI] = a0;
    s.gregs[FEXCore::X86State::REG_RSI] = a1;
    s.gregs[FEXCore::X86State::REG_RDX] = a2;
    s.rip = entry_addr;
    impl->ResetCallRetStack(impl->thread);

    impl->BeginExecution(impl->thread, reinterpret_cast<VAddr>(impl->guest_stack_base),
                         GuestStackSize);
    impl->ctx->ExecuteThread(impl->thread); // runs entry; ret -> sentinel HLT -> returns here
    impl->EndExecution(impl->thread);
    return s.gregs[FEXCore::X86State::REG_RAX];
}

u64 FexBackend::RunGuestThread(VAddr entry_addr, u64 arg, VAddr guest_stack_base,
                               u64 guest_stack_size) {
    if (!Initialize()) {
        UNREACHABLE_MSG("Failed to initialize FEXCore for guest thread {:#x}", entry_addr);
    }

    const VAddr stack_top = (guest_stack_base + guest_stack_size) & ~uint64_t(15);
    const VAddr root_frame = impl->InstallRootFrame(stack_top);
    uint64_t rsp = root_frame;
    rsp -= 8;
    *reinterpret_cast<uint64_t*>(rsp) = impl->sentinel;

    auto* guest_thread = impl->ctx->CreateThread(entry_addr, rsp);
    ASSERT_MSG(guest_thread, "FEXCore CreateThread failed for guest pthread");
    const auto callret = impl->SetupThread(guest_thread, entry_addr);
    auto& state = guest_thread->CurrentFrame->State;
    state.gregs[FEXCore::X86State::REG_RSP] = rsp;
    state.gregs[FEXCore::X86State::REG_RBP] = root_frame;
    // pthread HLE receives pointer arguments in their translated host form so
    // native code can dereference them.  The x86-64 thread entry is guest code
    // again and must observe the canonical PS4 pointer value.
    state.gregs[FEXCore::X86State::REG_RDI] = CanonicalizeGuestPointer(arg);
    impl->ResetCallRetStack(guest_thread);

    impl->BeginExecution(guest_thread, guest_stack_base, guest_stack_size);
    impl->ctx->ExecuteThread(guest_thread);
    const u64 result = state.gregs[FEXCore::X86State::REG_RAX];
    // A guest thread must only stop by returning to the HLT sentinel. Any other
    // rip means the dispatcher exited spuriously (callback-ret sentinel, stop
    // request, unhandled fault) and the "return value" below is garbage - the
    // caller would treat a still-live worker as cleanly exited.
    if (state.rip < impl->sentinel || state.rip > impl->sentinel + 1) {
        LOG_ERROR(Core_Linker,
                  "RunGuestThread: thread '{}' stopped at rip={:#x} instead of the return "
                  "sentinel {:#x} (entry={:#x}, rax={:#x}, rsp={:#x})",
                  Common::GetCurrentThreadName(), state.rip, impl->sentinel, entry_addr, result,
                  state.gregs[FEXCore::X86State::REG_RSP]);
        // Full crime-scene dump: guest GPRs, stack window and code bytes around
        // rip, captured before anything is torn down.
        static constexpr std::array<const char*, 16> GprNames{
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
            "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
        for (int i = 0; i < 16; ++i) {
            LOG_ERROR(Core_Linker, "  {} = {:#018x}", GprNames[i], state.gregs[i]);
        }
        auto* memory = Core::Memory::Instance();
        const auto dump_range = [&](const char* label, u64 addr, u64 count_qwords) {
            if (!memory->IsValidMapping(addr) ||
                !memory->IsValidMapping(addr + count_qwords * 8 - 1)) {
                LOG_ERROR(Core_Linker, "  {}: {:#x} not mapped", label, addr);
                return;
            }
            for (u64 i = 0; i < count_qwords; i += 4) {
                const auto* p = reinterpret_cast<const u64*>(addr + i * 8);
                LOG_ERROR(Core_Linker, "  {} {:#x}: {:016x} {:016x} {:016x} {:016x}", label,
                          addr + i * 8, p[0], p[1], p[2], p[3]);
            }
        };
        dump_range("stack", state.gregs[FEXCore::X86State::REG_RSP], 48);
        dump_range("code", (state.rip - 0x80) & ~7ull, 32);
        // Pending guest call chain from FEX's call-return shadow stack
        // (entries: {guest return address, host block}, pushed downwards).
        const u64 cr_top = reinterpret_cast<u64>(guest_thread->CallRetStackBase) +
                           FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE / 4;
        u64 cr_sp = state.callret_sp;
        for (int i = 0; cr_sp < cr_top && i < 24; cr_sp += 16, ++i) {
            const auto* e = reinterpret_cast<const u64*>(cr_sp);
            if (e[0] || e[1]) {
                LOG_ERROR(Core_Linker, "  callret[{}] guest_ret={:#x} host={:#x}", i, e[0], e[1]);
            }
        }
        Common::Log::Flush();
    }
    impl->EndExecution(guest_thread);
    impl->ctx->DestroyThread(guest_thread);
    ::munmap(callret.mapping, callret.size);
    return result;
}

u64 FexBackend::CallGuestCallback(VAddr entry_addr, u64 a0, u64 a1, u64 a2, u64 a3, u64 a4,
                                  u64 a5) {
    const std::array<u64, 6> args{a0, a1, a2, a3, a4, a5};
    return CallGuestCallback(entry_addr, args);
}

u64 FexBackend::CallGuestCallback(VAddr entry_addr, std::span<const u64> args) {
    if (!Initialize()) {
        UNREACHABLE_MSG("Failed to initialize FEXCore for guest callback {:#x}", entry_addr);
    }

    static constexpr std::array<int, 6> ArgRegs{
        FEXCore::X86State::REG_RDI, FEXCore::X86State::REG_RSI, FEXCore::X86State::REG_RDX,
        FEXCore::X86State::REG_RCX, FEXCore::X86State::REG_R8,  FEXCore::X86State::REG_R9};

    if (g_current_fex_thread) {
        auto* frame = g_current_fex_thread->CurrentFrame;
        auto& state = frame->State;
        for (size_t i = 0; i < std::min(args.size(), ArgRegs.size()); ++i) {
            state.gregs[ArgRegs[i]] = args[i];
        }
        const size_t stack_arg_count =
            args.size() > ArgRegs.size() ? args.size() - ArgRegs.size() : 0;
        // FEX reserves 16 bytes below the paused RSP and places CALLBACKRET in the lower slot.
        // At guest entry RSP therefore equals old RSP-16, so the first x86-64 SysV overflow
        // argument belongs in the upper slot at callback RSP+8 (old RSP-8).
        auto* callback_stack_args =
            reinterpret_cast<u64*>(state.gregs[FEXCore::X86State::REG_RSP] - sizeof(u64));
        std::vector<u64> saved_stack(callback_stack_args, callback_stack_args + stack_arg_count);
        for (size_t i = 0; i < stack_arg_count; ++i) {
            callback_stack_args[i] = args[ArgRegs.size() + i];
        }
        // A callback entered from a marked native thunk is real guest JIT code,
        // so temporarily clear the outer thunk's "state already spilled"
        // marker.  This lets an asynchronous signal spill the callback's live
        // registers normally.  Restore the marker before returning to the
        // still-active native thunk.
        const u64 outer_in_syscall_info = frame->InSyscallInfo;
        frame->InSyscallInfo = 0;
        impl->ctx->HandleCallback(g_current_fex_thread, entry_addr);
        frame->InSyscallInfo = outer_in_syscall_info;
        std::copy(saved_stack.begin(), saved_stack.end(), callback_stack_args);
        return state.gregs[FEXCore::X86State::REG_RAX];
    }

    // Asynchronous HLE callbacks have no paused guest frame to re-enter. Give
    // them a temporary FEX state and stack on the current native callback thread.
    constexpr size_t CallbackStackSize = 2 * 1024 * 1024;
    void* callback_stack = ::mmap(nullptr, CallbackStackSize, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_MSG(callback_stack != MAP_FAILED,
               "Failed to allocate stack for asynchronous guest callback");
    const size_t stack_arg_count = args.size() > ArgRegs.size() ? args.size() - ArgRegs.size() : 0;
    const uint64_t stack_top =
        (reinterpret_cast<uint64_t>(callback_stack) + CallbackStackSize) & ~uint64_t(15);
    const VAddr root_frame = impl->InstallRootFrame(stack_top);
    uint64_t rsp = root_frame - (stack_arg_count + 1) * sizeof(u64);
    if ((rsp & 15) != 8) {
        rsp -= sizeof(u64);
    }
    *reinterpret_cast<u64*>(rsp) = impl->sentinel;
    auto* callback_stack_args = reinterpret_cast<u64*>(rsp + sizeof(u64));
    for (size_t i = 0; i < stack_arg_count; ++i) {
        callback_stack_args[i] = args[ArgRegs.size() + i];
    }

    auto* callback_thread = impl->ctx->CreateThread(entry_addr, rsp);
    ASSERT_MSG(callback_thread, "FEXCore CreateThread failed for guest callback");
    const auto callret = impl->SetupThread(callback_thread, entry_addr);
    auto& state = callback_thread->CurrentFrame->State;
    state.gregs[FEXCore::X86State::REG_RSP] = rsp;
    state.gregs[FEXCore::X86State::REG_RBP] = root_frame;
    for (size_t i = 0; i < std::min(args.size(), ArgRegs.size()); ++i) {
        state.gregs[ArgRegs[i]] = args[i];
    }
    impl->ResetCallRetStack(callback_thread);

    impl->BeginExecution(callback_thread, reinterpret_cast<VAddr>(callback_stack),
                         CallbackStackSize);
    impl->ctx->ExecuteThread(callback_thread);
    const u64 result = state.gregs[FEXCore::X86State::REG_RAX];
    impl->EndExecution(callback_thread);

    impl->ctx->DestroyThread(callback_thread);
    ::munmap(callret.mapping, callret.size);
    ::munmap(callback_stack, CallbackStackSize);
    return result;
}

void FexBackend::ResetCurrentCallRetStack() {
    ASSERT_MSG(g_current_fex_thread, "FEX call/return stack reset without an active guest thread");
    impl->ResetCallRetStack(g_current_fex_thread);
}

bool FexBackend::IsGuestAddress(VAddr address) const {
    if (Core::Memory::Instance()->IsValidMapping(address)) {
        return true;
    }
    const auto& arena = GetHleStubArenaState();
    const VAddr arena_base = reinterpret_cast<VAddr>(arena.base);
    return arena.base != MAP_FAILED && address >= arena_base && address < arena_base + arena.off;
}

bool FexBackend::GetCurrentGuestStackBounds(VAddr& base, u64& size) const {
    if (!g_current_fex_thread || !g_current_guest_stack_base || !g_current_guest_stack_size) {
        return false;
    }
    base = g_current_guest_stack_base;
    size = g_current_guest_stack_size;
    return true;
}

bool FexBackend::QueueGuestSignal(VAddr handler, u64 orbis_signum, s32 native_signum,
                                  VAddr fault_address, void* native_context) noexcept {
#if defined(__APPLE__) && defined(__aarch64__)
    if (!impl->initialized || !g_current_fex_thread || !handler || !native_context) {
        return false;
    }

    auto& pending = g_pending_guest_signal;
    if (pending.phase != PendingSignalIdle) {
        return false;
    }
    pending.phase = PendingSignalPreparing;

    auto* raw_context = static_cast<ucontext_t*>(native_context);
    if (!raw_context->uc_mcontext) {
        pending.phase = PendingSignalIdle;
        return false;
    }

    auto* frame = g_current_fex_thread->CurrentFrame;
    auto& state = frame->State;
    auto& host = raw_context->uc_mcontext->__ss;
    const u64 host_pc = host.__pc;
    const auto& config = impl->signals->GetConfig();
    const bool was_in_jit = impl->ctx->IsAddressInCodeBuffer(g_current_fex_thread, host_pc);

    pending.handler = handler;
    pending.orbis_signum = orbis_signum;
    pending.fault_address = fault_address;
    pending.host_pc = host_pc;
    pending.was_in_jit = was_in_jit;
    pending.resume_state_valid = false;
    pending.saved_signal_mask = raw_context->uc_sigmask;
    std::memcpy(&pending.saved_mcontext, raw_context->uc_mcontext, sizeof(pending.saved_mcontext));
    std::memcpy(pending.saved_cpu_state.data(), &state, sizeof(state));

    // Keep the native delivery signal blocked through the pause callback. The
    // complete original mask is restored at PauseReturn below.
    sigaddset(&raw_context->uc_sigmask, native_signum);

    host.__x[28] = reinterpret_cast<u64>(frame); // FEX STATE register
    // A JIT PC does not always mean the fixed guest registers are live.  HLE
    // thunks and syscalls spill CPUState before their native call and mark the
    // refill interval in InSyscallInfo.  Spilling again in that interval would
    // overwrite valid guest state with caller-saved host scratch registers.
    const bool state_already_spilled = frame->InSyscallInfo != 0;
    host.__pc = was_in_jit && !state_already_spilled ? config.ThreadPauseHandlerAddressSpillSRA
                                                     : config.ThreadPauseHandlerAddress;

    // Prevent FEX from reclaiming an interrupted code buffer until the saved
    // host PC has been restored by the PauseReturn SIGILL.
    ++frame->SignalHandlerRefCounter;
    pending.phase = PendingSignalQueued;
    return true;
#else
    (void)handler;
    (void)orbis_signum;
    (void)native_signum;
    (void)fault_address;
    (void)native_context;
    return false;
#endif
}

void FexBackend::DispatchPendingGuestSignal() {
#if defined(__APPLE__) && defined(__aarch64__)
    auto& pending = g_pending_guest_signal;
    if (pending.phase != PendingSignalQueued || !g_current_fex_thread) {
        return;
    }

    auto& state = g_current_fex_thread->CurrentFrame->State;
    if (pending.was_in_jit) {
        state.rip = impl->ctx->RestoreRIPFromHostPC(g_current_fex_thread, pending.host_pc);
    }

    GuestRegisterContext guest{};
    if (!CaptureGuestContext(nullptr, guest)) {
        // SleepThread can only be reached with a current FEX frame. Still let
        // the pause trampoline reach its restoration HLT if the invariant is
        // ever broken instead of leaving the thread parked permanently.
        pending.phase = PendingSignalReturning;
        return;
    }

    auto& ctx = pending.orbis_context;
    ctx = {};
    ctx.uc_mcontext.mc_r8 = guest.r8;
    ctx.uc_mcontext.mc_r9 = guest.r9;
    ctx.uc_mcontext.mc_r10 = guest.r10;
    ctx.uc_mcontext.mc_r11 = guest.r11;
    ctx.uc_mcontext.mc_r12 = guest.r12;
    ctx.uc_mcontext.mc_r13 = guest.r13;
    ctx.uc_mcontext.mc_r14 = guest.r14;
    ctx.uc_mcontext.mc_r15 = guest.r15;
    ctx.uc_mcontext.mc_rdi = guest.rdi;
    ctx.uc_mcontext.mc_rsi = guest.rsi;
    ctx.uc_mcontext.mc_rbp = guest.rbp;
    ctx.uc_mcontext.mc_rbx = guest.rbx;
    ctx.uc_mcontext.mc_rdx = guest.rdx;
    ctx.uc_mcontext.mc_rax = guest.rax;
    ctx.uc_mcontext.mc_rcx = guest.rcx;
    ctx.uc_mcontext.mc_rsp = guest.rsp;
    ctx.uc_mcontext.mc_fsbase = guest.fs;
    ctx.uc_mcontext.mc_rip = guest.rip;
    ctx.uc_mcontext.mc_addr = pending.fault_address;

    // The pause trampoline has now spilled the live JIT registers into CPUState.
    // Keep this architectural snapshot as the safe resume point in case the
    // signal callback rotates the shared JIT code-buffer generation.
    std::memcpy(pending.resume_cpu_state.data(), &state, sizeof(state));
    pending.resume_state_valid = true;

    InvokeGuestSignalHandler(pending.handler, pending.orbis_signum, &ctx);
    pending.phase = PendingSignalReturning;
#endif
}

bool FexBackend::CaptureGuestContext(void* native_context, GuestRegisterContext& context) {
    if (!impl->initialized || !g_current_fex_thread) {
        return false;
    }

    auto& state = g_current_fex_thread->CurrentFrame->State;
#if defined(__APPLE__) && defined(__aarch64__)
    if (native_context) {
        const u64 host_pc = reinterpret_cast<u64>(Common::GetRip(native_context));
        if (impl->ctx->IsAddressInCodeBuffer(g_current_fex_thread, host_pc)) {
            // Translated blocks keep the architectural GPRs in FEX's fixed ARM
            // register allocation. Import them before entering a guest callback;
            // State.gregs is only authoritative at dispatcher/HLE boundaries.
            const auto& host = static_cast<ucontext_t*>(native_context)->uc_mcontext->__ss;
            state.gregs[FEXCore::X86State::REG_RAX] = host.__x[4];
            state.gregs[FEXCore::X86State::REG_RCX] = host.__x[7];
            state.gregs[FEXCore::X86State::REG_RDX] = host.__x[5];
            state.gregs[FEXCore::X86State::REG_RBX] = host.__x[6];
            state.gregs[FEXCore::X86State::REG_RSP] = host.__x[8];
            state.gregs[FEXCore::X86State::REG_RBP] = host.__x[9];
            state.gregs[FEXCore::X86State::REG_RSI] = host.__x[10];
            state.gregs[FEXCore::X86State::REG_RDI] = host.__x[11];
            state.gregs[FEXCore::X86State::REG_R8] = host.__x[12];
            state.gregs[FEXCore::X86State::REG_R9] = host.__x[13];
            state.gregs[FEXCore::X86State::REG_R10] = host.__x[14];
            state.gregs[FEXCore::X86State::REG_R11] = host.__x[15];
            state.gregs[FEXCore::X86State::REG_R12] = host.__x[16];
            state.gregs[FEXCore::X86State::REG_R13] = host.__x[17];
            state.gregs[FEXCore::X86State::REG_R14] = host.__x[19];
            state.gregs[FEXCore::X86State::REG_R15] = host.__fp;
            state.rip = impl->ctx->RestoreRIPFromHostPC(g_current_fex_thread, host_pc);
        }
    }
#else
    (void)native_context;
#endif

    context.rax = state.gregs[FEXCore::X86State::REG_RAX];
    context.rbx = state.gregs[FEXCore::X86State::REG_RBX];
    context.rcx = state.gregs[FEXCore::X86State::REG_RCX];
    context.rdx = state.gregs[FEXCore::X86State::REG_RDX];
    context.rsi = state.gregs[FEXCore::X86State::REG_RSI];
    context.rdi = state.gregs[FEXCore::X86State::REG_RDI];
    context.rbp = state.gregs[FEXCore::X86State::REG_RBP];
    context.rsp = state.gregs[FEXCore::X86State::REG_RSP];
    context.r8 = state.gregs[FEXCore::X86State::REG_R8];
    context.r9 = state.gregs[FEXCore::X86State::REG_R9];
    context.r10 = state.gregs[FEXCore::X86State::REG_R10];
    context.r11 = state.gregs[FEXCore::X86State::REG_R11];
    context.r12 = state.gregs[FEXCore::X86State::REG_R12];
    context.r13 = state.gregs[FEXCore::X86State::REG_R13];
    context.r14 = state.gregs[FEXCore::X86State::REG_R14];
    context.r15 = state.gregs[FEXCore::X86State::REG_R15];
    context.rip = state.rip;
    context.fs = state.fs_cached;
    return true;
}

u64 FexBackend::InvokeGuestSignalHandler(VAddr handler, u64 signum, void* guest_context) {
    ASSERT_MSG(g_current_fex_thread, "Guest signal delivered without an active FEX thread");
    // A GC signal commonly interrupts a worker inside a native pthread wait.
    // Re-entering that already-paused FEX state through HandleCallback mutates
    // its guest stack and call/return shadow stack while a native HLE call is
    // still active. Run the small signal callback in an isolated FEX frame on
    // the same host thread instead; its FS base still resolves to this thread's
    // PS4 TLS, while the supplied Ucontext describes the interrupted registers.
    auto* interrupted_thread = g_current_fex_thread;
    const VAddr interrupted_stack_base = g_current_guest_stack_base;
    const u64 interrupted_stack_size = g_current_guest_stack_size;

    impl->signals->UninstallTLSState(interrupted_thread);
    g_fex_guest_gregs = nullptr;
    g_fex_guest_xmm = nullptr;
    g_fex_guest_mxcsr = nullptr;
    g_current_fex_thread = nullptr;
    g_current_guest_stack_base = 0;
    g_current_guest_stack_size = 0;

    const u64 result = CallGuestCallback(handler, signum, reinterpret_cast<u64>(guest_context));
    impl->BeginExecution(interrupted_thread, interrupted_stack_base, interrupted_stack_size);
    return result;
}

bool FexBackend::HandleAccessViolation(void* context, void* fault_address) {
    const u64 host_pc = reinterpret_cast<u64>(Common::GetRip(context));
    if (!impl->initialized) {
        LOG_CRITICAL(Core_Linker,
                     "FEX access violation rejected before initialization: host PC {:#x}, "
                     "address {}",
                     host_pc, fmt::ptr(fault_address));
        return false;
    }
    if (!g_current_fex_thread) {
        LOG_CRITICAL(Core_Linker,
                     "FEX access violation has no active guest thread: host PC {:#x}, address {}",
                     host_pc, fmt::ptr(fault_address));
        return false;
    }
    const auto& state = g_current_fex_thread->CurrentFrame->State;
    const bool in_reported_code_buffer =
        impl->ctx->IsAddressInCodeBuffer(g_current_fex_thread, host_pc);
    // RestoreRIPFromHostPC validates shared code-buffer generations as well as the generation
    // currently attached to this thread. A reconstructed RIP different from the saved dispatcher
    // RIP therefore proves that the faulting PC belongs to valid translated guest code.
    const u64 guest_rip = impl->ctx->RestoreRIPFromHostPC(g_current_fex_thread, host_pc);
    const bool in_valid_jit_code = in_reported_code_buffer || guest_rip != state.rip;
#if defined(__APPLE__) && defined(__aarch64__)
    // Guest accesses to canonical PS4 fixed addresses that fall inside the macOS
    // GPU carveout (0x1000000000-0x6FFFFFFFFF, hard-reserved in every process)
    // cannot be backed by any host mapping. Titles that dereference the
    // constants they passed to fixed sceKernelMemoryPool*/MapMemory calls,
    // instead of the relocated address returned to them, fault here. Emulate
    // the single faulting load/store against the relocated address that the
    // memory manager actually mapped — the same linear shift used by
    // MemoryManager::TranslateCanonicalGuestAddress.
    const auto fault_va = reinterpret_cast<u64>(fault_address);
    if (in_valid_jit_code && fault_va >= Core::CanonicalGuestRemapStart &&
        fault_va < Core::CanonicalGuestRemapEnd) {
        auto* memory = Core::Memory::Instance();
        const u64 relocated = memory->TranslateCanonicalGuestAddress(fault_va);
        if (relocated != fault_va && memory->IsValidMapping(relocated, 1)) {
            auto* raw_context = static_cast<ucontext_t*>(context);
            auto& host = raw_context->uc_mcontext->__ss;
            const u32 instr = *reinterpret_cast<const u32*>(host.__pc);
            std::array<u64, 32> arm_gprs{};
            std::copy(std::begin(host.__x), std::end(host.__x), arm_gprs.begin());
            arm_gprs[29] = host.__fp;
            arm_gprs[30] = host.__lr;
            arm_gprs[31] = host.__sp;
            auto* neon = reinterpret_cast<unsigned char*>(
                &raw_context->uc_mcontext->__ns.__v[0]);
            if (Core::CPU::EmulateA64LoadStore(instr, relocated, arm_gprs.data(), neon)) {
                std::copy_n(arm_gprs.begin(), std::size(host.__x), std::begin(host.__x));
                host.__fp = arm_gprs[29];
                host.__lr = arm_gprs[30];
                host.__sp = arm_gprs[31];
                host.__pc += 4;
                static std::atomic<u64> carveout_hits{0};
                const u64 hit = carveout_hits.fetch_add(1, std::memory_order_relaxed);
                if (hit < 8 || hit % 65536 == 0) {
                    LOG_WARNING(Core_Linker,
                                "Emulated carveout guest access #{}: canonical {:#x} -> {:#x} "
                                "(instr {:#010x}, guest RIP {:#x})",
                                hit, fault_va, relocated, instr, guest_rip);
                }
                return true;
            }
            LOG_CRITICAL(Core_Linker,
                         "Carveout guest access to {:#x} uses an unsupported load/store encoding "
                         "{:#010x} (guest RIP {:#x}); cannot emulate",
                         fault_va, instr, guest_rip);
        }
    }
    if (Common::IsAlignmentError(context) && in_valid_jit_code) {
        auto* raw_context = static_cast<ucontext_t*>(context);
        auto& host = raw_context->uc_mcontext->__ss;
        std::array<u64, 32> arm_gprs{};
        std::copy(std::begin(host.__x), std::end(host.__x), arm_gprs.begin());
        arm_gprs[29] = host.__fp;
        arm_gprs[30] = host.__lr;
        arm_gprs[31] = host.__sp;

        const auto result = FEXCore::ArchHelpers::Arm64::HandleUnalignedAccess(
            g_current_fex_thread, FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::HalfBarrier,
            host_pc, arm_gprs.data());
        // Diagnostic: unaligned-atomic backpatches lose atomicity guarantees for
        // subsequent aligned accesses through the same code site. Make every
        // occurrence visible (throttled) so torn-value bugs can be correlated.
        static std::atomic<u64> unaligned_hits{0};
        const u64 hit = unaligned_hits.fetch_add(1, std::memory_order_relaxed);
        if (hit < 16 || hit % 1024 == 0) {
            LOG_WARNING(Core_Linker,
                        "Unaligned guest atomic #{} at host PC {:#x} (guest RIP {:#x}, addr {}, "
                        "patched={})",
                        hit, host_pc, guest_rip, fmt::ptr(fault_address), result.has_value());
        }
        if (result) {
            std::copy_n(arm_gprs.begin(), std::size(host.__x), std::begin(host.__x));
            host.__fp = arm_gprs[29];
            host.__lr = arm_gprs[30];
            host.__sp = arm_gprs[31];
            host.__pc += *result;
            return true;
        }
    }
#endif
    if (!in_reported_code_buffer && guest_rip == state.rip) {
        LOG_CRITICAL(Core_Linker,
                     "FEX access violation outside current code buffer: host PC {:#x}, address {}, "
                     "saved guest RIP {:#x}, InSyscallInfo={:#x}",
                     host_pc, fmt::ptr(fault_address), state.rip,
                     g_current_fex_thread->CurrentFrame->InSyscallInfo);
        return false;
    }
    if (!in_reported_code_buffer) {
        LOG_WARNING(Core_Linker,
                    "FEX reconstructed guest RIP {:#x} from a shared code-buffer generation "
                    "not reported as current (host PC {:#x})",
                    guest_rip, host_pc);
    }
    const auto* code = reinterpret_cast<const u8*>(guest_rip);
    LOG_CRITICAL(Core_Linker,
                 "FEX guest memory fault: guest RIP {:#x}, host PC {:#x}, address {}; bytes "
                 "{:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} "
                 "{:02x} {:02x} {:02x} {:02x} {:02x} {:02x}; RAX={:#x} RBX={:#x} RCX={:#x} "
                 "RDX={:#x} RSI={:#x} RDI={:#x} RBP={:#x} RSP={:#x} R8={:#x} R9={:#x} "
                 "R10={:#x} R11={:#x} R12={:#x} R13={:#x} R14={:#x} R15={:#x} FS={:#x}",
                 guest_rip, host_pc, fmt::ptr(fault_address), code[0], code[1], code[2], code[3],
                 code[4], code[5], code[6], code[7], code[8], code[9], code[10], code[11], code[12],
                 code[13], code[14], code[15], state.gregs[FEXCore::X86State::REG_RAX],
                 state.gregs[FEXCore::X86State::REG_RBX], state.gregs[FEXCore::X86State::REG_RCX],
                 state.gregs[FEXCore::X86State::REG_RDX], state.gregs[FEXCore::X86State::REG_RSI],
                 state.gregs[FEXCore::X86State::REG_RDI], state.gregs[FEXCore::X86State::REG_RBP],
                 state.gregs[FEXCore::X86State::REG_RSP], state.gregs[FEXCore::X86State::REG_R8],
                 state.gregs[FEXCore::X86State::REG_R9], state.gregs[FEXCore::X86State::REG_R10],
                 state.gregs[FEXCore::X86State::REG_R11], state.gregs[FEXCore::X86State::REG_R12],
                 state.gregs[FEXCore::X86State::REG_R13], state.gregs[FEXCore::X86State::REG_R14],
                 state.gregs[FEXCore::X86State::REG_R15], state.fs_cached);
#if defined(__APPLE__) && defined(__aarch64__)
    // FEX keeps x86 GPRs in fixed ARM registers while translated code is running. State.gregs is
    // only synchronized at runtime boundaries, so recover the live values from the signal frame.
    const auto& host = static_cast<ucontext_t*>(context)->uc_mcontext->__ss;
    LOG_CRITICAL(Core_Linker,
                 "FEX live guest GPRs: RAX={:#x} RBX={:#x} RCX={:#x} RDX={:#x} RSI={:#x} "
                 "RDI={:#x} RBP={:#x} RSP={:#x} R8={:#x} R9={:#x} R10={:#x} R11={:#x} "
                 "R12={:#x} R13={:#x} R14={:#x} R15={:#x}",
                 host.__x[4], host.__x[6], host.__x[7], host.__x[5], host.__x[10], host.__x[11],
                 host.__x[9], host.__x[8], host.__x[12], host.__x[13], host.__x[14], host.__x[15],
                 host.__x[16], host.__x[17], host.__x[19], host.__fp);

    const u64 live_rbx = host.__x[6];
    if (live_rbx >= g_current_guest_stack_base &&
        live_rbx + 0x60 <= g_current_guest_stack_base + g_current_guest_stack_size) {
        const auto* locals = reinterpret_cast<const u64*>(live_rbx);
        LOG_CRITICAL(Core_Linker,
                     "FEX guest locals at RBX: +00={:#x} +08={:#x} +10={:#x} +18={:#x} "
                     "+20={:#x} +28={:#x} +30={:#x} +38={:#x} +40={:#x} +48={:#x} "
                     "+50={:#x} +58={:#x}",
                     locals[0], locals[1], locals[2], locals[3], locals[4], locals[5], locals[6],
                     locals[7], locals[8], locals[9], locals[10], locals[11]);
    }

    // Guest frame locals are often the only way to reconstruct the source of a bad table base.
    // Keep this strictly inside the known guest stack so diagnostics cannot cause a second fault.
    const u64 live_rbp = host.__x[9];
    if (live_rbp >= g_current_guest_stack_base + 0x50 &&
        live_rbp + 0x30 <= g_current_guest_stack_base + g_current_guest_stack_size) {
        const auto* locals = reinterpret_cast<const u64*>(live_rbp - 0x50);
        LOG_CRITICAL(Core_Linker,
                     "FEX guest stack around RBP: -50={:#x} -48={:#x} -40={:#x} -38={:#x} "
                     "-30={:#x} -28={:#x} -20={:#x} -18={:#x} -10={:#x} -08={:#x} "
                     "+00={:#x} +08={:#x} +10={:#x} +18={:#x} +20={:#x} +28={:#x}",
                     locals[0], locals[1], locals[2], locals[3], locals[4], locals[5], locals[6],
                     locals[7], locals[8], locals[9], locals[10], locals[11], locals[12],
                     locals[13], locals[14], locals[15]);
    }

    const auto* host_code = reinterpret_cast<const u32*>(host_pc);
    LOG_CRITICAL(Core_Linker,
                 "FEX host code around fault: -10={:08x} -0c={:08x} -08={:08x} -04={:08x} "
                 "+00={:08x} +04={:08x} +08={:08x} +0c={:08x} +10={:08x} +14={:08x}",
                 host_code[-4], host_code[-3], host_code[-2], host_code[-1], host_code[0],
                 host_code[1], host_code[2], host_code[3], host_code[4], host_code[5]);

    // A carveout access that reached here means the memory manager had no
    // relocated mapping for it (the pool was never reserved, or the encoding is
    // unsupported); the specific reason was already logged above.
    if (reinterpret_cast<u64>(fault_address) >= Core::CanonicalGuestRemapStart &&
        reinterpret_cast<u64>(fault_address) < Core::CanonicalGuestRemapEnd) {
        LOG_CRITICAL(Core_Linker,
                     "Unhandled guest access to canonical PS4 address {:#x} in the macOS GPU "
                     "carveout (0x1000000000-0x6FFFFFFFFF); see INTEGRATION.md (CUSA32809).",
                     reinterpret_cast<u64>(fault_address));
    }
#endif
    return false;
}

bool FexBackend::HandleIllegalInstruction(void* context) {
    if (!impl->initialized || !g_current_fex_thread) {
        return false;
    }
    const u64 host_pc = reinterpret_cast<u64>(Common::GetRip(context));
#if defined(__APPLE__) && defined(__aarch64__)
    auto& pending = g_pending_guest_signal;
    const auto& signal_config = impl->signals->GetConfig();
    if (host_pc == signal_config.PauseReturnInstruction &&
        pending.phase == PendingSignalReturning) {
        auto* raw_context = static_cast<ucontext_t*>(context);
        auto* frame = g_current_fex_thread->CurrentFrame;

        const bool resume_host_pc_is_current =
            !pending.was_in_jit ||
            impl->ctx->IsAddressInCurrentCodeBuffer(g_current_fex_thread, pending.host_pc);

        if (!resume_host_pc_is_current && pending.resume_state_valid) {
            // A guest signal callback may compile enough code to rotate FEX's
            // shared code buffer. The interrupted generation is retained while
            // the handler runs, but resuming it directly after dropping the
            // signal reference would leave host return addresses pointing into
            // reclaimable memory. Restore the spilled guest state and re-enter
            // through the current dispatcher generation instead.
            std::memcpy(&frame->State, pending.resume_cpu_state.data(), sizeof(frame->State));
            impl->ResetCallRetStack(g_current_fex_thread);

            auto& host = raw_context->uc_mcontext->__ss;
            host.__x[1] = 0; // do not request a single-instruction dispatcher entry
            host.__x[28] = reinterpret_cast<u64>(frame);
            host.__pc = frame->Pointers.DispatcherLoopTopFillSRA;
            raw_context->uc_sigmask = pending.saved_signal_mask;
            --frame->SignalHandlerRefCounter;
            pending.phase = PendingSignalIdle;
            return true;
        }

        std::memcpy(&frame->State, pending.saved_cpu_state.data(), sizeof(frame->State));
        std::memcpy(raw_context->uc_mcontext, &pending.saved_mcontext,
                    sizeof(pending.saved_mcontext));
        raw_context->uc_sigmask = pending.saved_signal_mask;
        --frame->SignalHandlerRefCounter;
        pending.phase = PendingSignalIdle;
        return true;
    }
#endif
    if (!impl->ctx->IsAddressInCodeBuffer(g_current_fex_thread, host_pc)) {
        const auto& config = impl->signals->GetConfig();
        if (host_pc >= config.DispatcherBegin && host_pc < config.DispatcherEnd) {
            const auto& state = g_current_fex_thread->CurrentFrame->State;
            const u64 guest_rip = state.rip;
            const auto* code = reinterpret_cast<const u8*>(guest_rip);
            const auto guest_sigill =
                g_current_fex_thread->CurrentFrame->Pointers.GuestSignal_SIGILL;
            const auto& arena = GetHleStubArenaState();
            const u64 arena_base = reinterpret_cast<u64>(arena.base);
            const u64 rsp = state.gregs[FEXCore::X86State::REG_RSP];
            const auto* stack = reinterpret_cast<const u64*>(rsp);
            LOG_CRITICAL(
                Core_Linker,
                "FEX dispatcher SIGILL: guest RIP {:#x}, host PC {:#x} (dispatcher +{:#x}); "
                "guest-sigill={:#x}, bytes {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} "
                "{:02x}; HLE arena={:#x}..{:#x} (offset {:#x}); RSP={:#x}, stack={:#x} {:#x} "
                "{:#x} {:#x}; pause={:#x}, stop={:#x}",
                guest_rip, host_pc, host_pc - config.DispatcherBegin, guest_sigill, code[0],
                code[1], code[2], code[3], code[4], code[5], code[6], code[7], arena_base,
                arena_base + arena.off, guest_rip - arena_base, rsp, stack[0], stack[1], stack[2],
                stack[3], config.PauseReturnInstruction, config.ThreadStopHandlerAddress);
        }
        return false;
    }
    const u64 guest_rip = impl->ctx->RestoreRIPFromHostPC(g_current_fex_thread, host_pc);
    const auto* code = reinterpret_cast<const u8*>(guest_rip);
    LOG_CRITICAL(Core_Linker,
                 "FEX guest illegal instruction: guest RIP {:#x}, bytes {:02x} {:02x} {:02x} "
                 "{:02x} {:02x} {:02x} {:02x} {:02x}, host PC {:#x}",
                 guest_rip, code[0], code[1], code[2], code[3], code[4], code[5], code[6], code[7],
                 host_pc);
    return false;
}

void FexBackend::RunMainThread(VAddr entry_addr, EntryParams* params, void* exit_func) {
    if (!Initialize()) {
        UNREACHABLE_MSG("Failed to initialize FEXCore for guest main {:#x}", entry_addr);
    }
    auto& s = impl->thread->CurrentFrame->State;

    // Mirror the PS4 kernel entry stack layout the x86 path builds by hand.
    const VAddr root_frame = impl->InstallRootFrame(impl->guest_stack_top);
    uint64_t rsp = root_frame;
    rsp -= 8; // videoout_basic expects the stack misaligned, as in the x86 path
    const uint64_t* pw = reinterpret_cast<const uint64_t*>(params);
    rsp -= 8;
    *reinterpret_cast<uint64_t*>(rsp) = pw[1];
    rsp -= 8;
    *reinterpret_cast<uint64_t*>(rsp) = pw[0];

    s.gregs[FEXCore::X86State::REG_RDI] = reinterpret_cast<uint64_t>(params);
    s.gregs[FEXCore::X86State::REG_RSI] = reinterpret_cast<uint64_t>(exit_func);
    s.gregs[FEXCore::X86State::REG_RSP] = rsp;
    s.gregs[FEXCore::X86State::REG_RBP] = root_frame;
    s.rip = entry_addr;
    impl->ResetCallRetStack(impl->thread);

    LOG_INFO(Core_Linker, "FEXCore: entering guest main at {:#x} (RSP {:#x})", entry_addr, rsp);
    impl->BeginExecution(impl->thread, reinterpret_cast<VAddr>(impl->guest_stack_base),
                         GuestStackSize);
    impl->ctx->ExecuteThread(impl->thread); // runs the game until it exits
    impl->EndExecution(impl->thread);

    UNREACHABLE_MSG("Guest main thread returned from FEXCore ExecuteThread");
}

} // namespace Core::CPU
