// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// FEXCore-FREE HLE thunk pieces. Kept free of FEXCore headers so tls.h (which
// pulls this in via HOST_CALL) doesn't drag FEXCore into every translation unit.
//
// Mechanism (validated in the FEXCore harness): guest x86 reaches a native ARM64
// function through FEX's thunk opcode `0F 3F` + a 32-byte field holding the host
// thunk pointer. A per-function marshalling thunk reads the guest SysV argument
// registers (via the raw gregs pointer fex_backend publishes) and calls the real
// HLE function with its true signature, writing the result to RAX.

#include <bit>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>
#include <sys/mman.h>

#include "common/types.h"

namespace Core::CPU {

// Raw pointer to the executing guest thread's gregs[16], published by fex_backend
// before ExecuteThread. FEXCore-type-free on purpose.
extern thread_local uint64_t* g_fex_guest_gregs;
extern thread_local uint64_t* g_fex_guest_xmm;
extern thread_local uint32_t* g_fex_guest_mxcsr;
// Set while a guest import is executing as native ARM64 code. This lets the
// process signal handler name the HLE call if the native implementation faults.
extern thread_local const char* g_current_fex_hle_signature;

[[noreturn]] void UnsupportedHleAbi(const char* signature);

// Maps a canonical PS4 fixed guest pointer (one the guest passed to a fixed
// sceKernelMemoryPool*/MapMemory call) that falls inside the macOS GPU carveout
// onto the address the memory manager actually mapped. Identity for any pointer
// outside the carveout. Applied only to pointer/reference HLE arguments so that
// native HLE code dereferences the live mapping instead of the unmapped
// canonical address. Defined in memory.cpp; a no-op on non-ARM64 hosts.
u64 TranslateCanonicalGuestPointer(u64 guest_ptr);

// gregs[] indices (FEXCore X86State order).
enum : int {
    FEX_RAX = 0,
    FEX_RCX = 1,
    FEX_RDX = 2,
    FEX_RSP = 4,
    FEX_RBP = 5,
    FEX_RSI = 6,
    FEX_RDI = 7,
    FEX_R8 = 8,
    FEX_R9 = 9
};
inline constexpr int kFexSysvArg[6] = {FEX_RDI, FEX_RSI, FEX_RDX, FEX_RCX, FEX_R8, FEX_R9};

template <class T>
using HleBareT = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T>
inline constexpr bool kHleGpr = std::is_reference_v<T> || std::is_integral_v<HleBareT<T>> ||
                                std::is_pointer_v<HleBareT<T>> || std::is_enum_v<HleBareT<T>>;

template <class T>
inline constexpr bool kHleSse = std::is_floating_point_v<HleBareT<T>>;

template <class T>
inline constexpr bool kSupportedHleArg =
    (kHleGpr<T> || kHleSse<T>) && (std::is_reference_v<T> || sizeof(HleBareT<T>) <= sizeof(u64));

template <class T>
inline constexpr bool kSupportedHleRet = [] {
    if constexpr (std::is_void_v<T>) {
        return true;
    } else {
        return kSupportedHleArg<T>;
    }
}();

struct HleArgCursor {
    size_t gpr_index{};
    size_t xmm_index{};
    size_t stack_index{};

    u64 ReadGpr() {
        if (gpr_index < std::size(kFexSysvArg)) {
            return g_fex_guest_gregs[kFexSysvArg[gpr_index++]];
        }
        ++gpr_index;
        const auto* stack = reinterpret_cast<const u64*>(g_fex_guest_gregs[FEX_RSP] + sizeof(u64));
        return stack[stack_index++];
    }

    u64 ReadXmm() {
        if (xmm_index < 8) {
            // CPUState::XMMRegs::AVX stores four qwords per architectural register.
            return g_fex_guest_xmm[(xmm_index++) * 4];
        }
        ++xmm_index;
        const auto* stack = reinterpret_cast<const u64*>(g_fex_guest_gregs[FEX_RSP] + sizeof(u64));
        return stack[stack_index++];
    }
};

template <class T>
decltype(auto) ReadHleArg(HleArgCursor& cursor) {
    using Bare = HleBareT<T>;
    if constexpr (kHleSse<T>) {
        const u64 bits = cursor.ReadXmm();
        if constexpr (sizeof(Bare) == sizeof(u32)) {
            return std::bit_cast<Bare>(static_cast<u32>(bits));
        } else {
            return std::bit_cast<Bare>(bits);
        }
    } else {
        const u64 bits = cursor.ReadGpr();
        if constexpr (std::is_reference_v<T>) {
            return static_cast<T>(*reinterpret_cast<std::add_pointer_t<std::remove_reference_t<T>>>(
                TranslateCanonicalGuestPointer(bits)));
        } else if constexpr (std::is_pointer_v<Bare>) {
            return reinterpret_cast<Bare>(TranslateCanonicalGuestPointer(bits));
        } else {
            return static_cast<Bare>(bits);
        }
    }
}

template <class Ret>
void WriteHleReturn(Ret&& value) {
    using Bare = HleBareT<Ret>;
    if constexpr (kHleSse<Ret>) {
        if constexpr (sizeof(Bare) == sizeof(u32)) {
            const u32 bits = std::bit_cast<u32>(static_cast<Bare>(value));
            std::memcpy(&g_fex_guest_xmm[0], &bits, sizeof(bits));
        } else {
            const u64 bits = std::bit_cast<u64>(static_cast<Bare>(value));
            g_fex_guest_xmm[0] = bits;
        }
    } else if constexpr (std::is_reference_v<Ret>) {
        g_fex_guest_gregs[FEX_RAX] = reinterpret_cast<u64>(&value);
    } else if constexpr (std::is_pointer_v<Bare>) {
        g_fex_guest_gregs[FEX_RAX] = reinterpret_cast<u64>(value);
    } else {
        g_fex_guest_gregs[FEX_RAX] = static_cast<u64>(value);
    }
}

// Per-function marshalling thunk for scalar SysV arguments. Integer and pointer
// arguments use the six GPR argument registers, float/double use XMM0..7, and
// overflow arguments are read from the guest stack. Aggregate-by-value ABI
// classification remains intentionally rejected with an explicit diagnostic.
template <auto Fn>
struct HleThunkT;

template <class Ret, class... Args, PS4_SYSV_ABI Ret (*Fn)(Args...)>
struct HleThunkT<Fn> {
    static void thunk(void* /*x0 unused; full state via g_fex_guest_gregs*/) {
        const char* previous_signature = g_current_fex_hle_signature;
        g_current_fex_hle_signature = __PRETTY_FUNCTION__;
        struct SignatureScope {
            const char* previous;
            ~SignatureScope() {
                g_current_fex_hle_signature = previous;
            }
        } signature_scope{previous_signature};

        if constexpr ((kSupportedHleArg<Args> && ...) && kSupportedHleRet<Ret>) {
            HleArgCursor cursor{};
            std::tuple<Args...> args{ReadHleArg<Args>(cursor)...};
            if constexpr (std::is_void_v<Ret>) {
                std::apply(Fn, args);
            } else {
                decltype(auto) result = std::apply(Fn, args);
                WriteHleReturn<Ret>(std::forward<Ret>(result));
            }
        } else {
            UnsupportedHleAbi(__PRETTY_FUNCTION__);
        }
    }
};

struct HleStubArenaState {
    uint8_t* base = static_cast<uint8_t*>(::mmap(nullptr, 64 * 1024 * 1024, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    size_t off = 0;
};

inline HleStubArenaState& GetHleStubArenaState() {
    static HleStubArenaState state;
    return state;
}

// Bump-allocator for executable guest memory holding `0F 3F` stubs. Single-threaded
// at HLE registration time.
inline uint8_t* StubArenaAlloc(size_t n) {
    auto& arena = GetHleStubArenaState();
    constexpr size_t ArenaSize = 64 * 1024 * 1024;
    const size_t aligned_n = (n + 15) & ~size_t(15);
    if (arena.base == MAP_FAILED || aligned_n > ArenaSize - arena.off) {
        UnsupportedHleAbi("FEX HLE stub arena allocation failed or overflowed");
    }
    uint8_t* p = arena.base + arena.off;
    arena.off += aligned_n;
    return p;
}

// Emit a `0F 3F` stub whose 32-byte field carries `thunk`; return its guest address.
// FEX implements this as a complete thunk call: it invokes the host function and
// pops the x86 return address itself.
inline VAddr EmitHleStub(void (*thunk)(void*)) {
    uint8_t* c = StubArenaAlloc(34);
    c[0] = 0x0F;
    c[1] = 0x3F;
    std::memset(c + 2, 0, 32);
    std::memcpy(c + 2, &thunk, sizeof(thunk));
    return reinterpret_cast<VAddr>(c);
}

} // namespace Core::CPU
