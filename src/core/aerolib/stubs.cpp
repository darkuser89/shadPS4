// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "common/arch.h"
#include "core/aerolib/aerolib.h"
#include "core/aerolib/stubs.h"
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
#include <deque>
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
// compile-time HleThunkT wrapper. Keep stable metadata for every generated
// guest stub instead of sharing the fixed-size native template table.
struct FexStubMetadata {
    const NidEntry* entry{};
    std::string unknown_nid;
};

static std::deque<FexStubMetadata> fex_stub_metadata;

static void CommonFexStub(void* metadata_ptr) {
    const auto& metadata = *static_cast<const FexStubMetadata*>(metadata_ptr);
    const u64 rsp = Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RSP];
    const auto guest_return = *reinterpret_cast<void* const*>(rsp);
    if (metadata.entry) {
        LOG_ERROR(Core, "Stub: {} (nid: {}) called, returning zero to {}", metadata.entry->name,
                  metadata.entry->nid, guest_return);
    } else {
        LOG_ERROR(Core, "Stub: Unknown (nid: {}) called, returning zero to {}",
                  metadata.unknown_nid, guest_return);
    }
    Core::CPU::g_fex_guest_gregs[Core::CPU::FEX_RAX] = 0;
}

static u64 MakeFexStub(const FexStubMetadata* metadata) {
    // movabs rdi, metadata; 0f 3f; 32-byte FEX thunk field
    auto* code = Core::CPU::StubArenaAlloc(44);
    code[0] = 0x48;
    code[1] = 0xBF;
    std::memcpy(code + 2, &metadata, sizeof(metadata));
    code[10] = 0x0F;
    code[11] = 0x3F;
    std::memset(code + 12, 0, 32);
    auto* thunk = &CommonFexStub;
    std::memcpy(code + 12, &thunk, sizeof(thunk));
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
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
    auto& metadata = fex_stub_metadata.emplace_back();
    metadata.entry = FindByNid(nid);
    if (!metadata.entry) {
        metadata.unknown_nid = nid;
    }
    return MakeFexStub(&metadata);
#else
    if (UsedStubEntries >= MAX_STUBS) {
        return (u64)&UnknownStub;
    }

    const auto entry = FindByNid(nid);
    if (!entry) {
        stub_nids_unknown[UsedStubEntries] = nid;
    } else {
        stub_nids[UsedStubEntries] = entry;
    }

    const u32 index = UsedStubEntries++;
    return reinterpret_cast<u64>(stub_handlers[index]);
#endif
}

} // namespace Core::AeroLib
