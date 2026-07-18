// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "common/arch.h"
#include "core/aerolib/aerolib.h"
#include "core/aerolib/stubs.h"
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
#include "core/cpu/fex_hle.h"
#endif

namespace Core::AeroLib {

// Helper to provide stub implementations for missing functions
//
// This works by pre-compiling generic stub functions ("slots"), and then
// on lookup, setting up the nid_entry they are matched with
//
// If it runs out of stubs with name information, it will return
// a default implementation without function name details

constexpr u32 MAX_STUBS = 8192;

u64 UnresolvedStub() {
    LOG_ERROR(Core, "Returning zero to {}", __builtin_return_address(0));
    return 0;
}

static u64 UnknownStub() {
    LOG_ERROR(Core, "Returning zero to {}", __builtin_return_address(0));
    return 0;
}

static const NidEntry* stub_nids[MAX_STUBS];
static std::string stub_nids_unknown[MAX_STUBS];

static u64 CommonStub(int stub_index, void* addr) {
    auto entry = stub_nids[stub_index];
    if (entry) {
        LOG_ERROR(Core, "Stub: {} (nid: {}) called, returning zero to {}", entry->name, entry->nid,
                  addr);
    } else {
        LOG_ERROR(Core, "Stub: Unknown (nid: {}) called, returning zero to {}",
                  stub_nids_unknown[stub_index], addr);
    }
    return 0;
}

#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
// Aerolib fallbacks are selected dynamically, so they cannot use the normal
// compile-time HleThunkT wrapper. Each tiny guest stub loads its slot index in
// EDI and then enters this shared FEX thunk. FEX spills the guest registers
// before calling us and pops the guest return address after we return.
static void CommonFexStub(void*) {
    const u64 index = Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RDI];
    const u64 rsp = Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RSP];
    const auto guest_return = *reinterpret_cast<void* const*>(rsp);
    Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RAX] =
        index < MAX_STUBS ? CommonStub(static_cast<int>(index), guest_return) : UnknownStub();
}

static u64 MakeFexStub(u32 index) {
    // mov edi, imm32; 0f 3f; 32-byte FEX thunk field
    auto* code = Core::CPU::StubArenaAlloc(39);
    code[0] = 0xBF;
    std::memcpy(code + 1, &index, sizeof(index));
    code[5] = 0x0F;
    code[6] = 0x3F;
    std::memset(code + 7, 0, 32);
    auto* thunk = &CommonFexStub;
    std::memcpy(code + 7, &thunk, sizeof(thunk));
    return reinterpret_cast<u64>(code);
}
#endif

template <int stub_index>
static u64 CommonStubTemplate() {
    return CommonStub(stub_index, __builtin_return_address(0));
}

template <size_t... Is>
consteval auto MakeStubArray(std::index_sequence<Is...>) {
    return std::array<u64 (*)(), sizeof...(Is)>{&CommonStubTemplate<Is>...};
}

constexpr auto stub_handlers = MakeStubArray(std::make_index_sequence<MAX_STUBS>{});
static u32 UsedStubEntries;

u64 GetStub(const char* nid) {
    if (UsedStubEntries >= MAX_STUBS) {
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
        return MakeFexStub(MAX_STUBS);
#else
        return (u64)&UnknownStub;
#endif
    }

    const auto entry = FindByNid(nid);
    if (!entry) {
        stub_nids_unknown[UsedStubEntries] = nid;
    } else {
        stub_nids[UsedStubEntries] = entry;
    }

    const u32 index = UsedStubEntries++;
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
    return MakeFexStub(index);
#else
    return reinterpret_cast<u64>(stub_handlers[index]);
#endif
}

} // namespace Core::AeroLib
