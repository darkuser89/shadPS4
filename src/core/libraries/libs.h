// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/arch.h"
#include "core/loader/elf.h"
#include "core/loader/symbols_resolver.h"
#include "core/tls.h"
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
#include "core/cpu/fex_hle.h"
#endif

// Address to register for an HLE import. On x86-64 the guest calls the native
// function directly (its own ABI). On ARM64/FEXCore the guest can't, so register
// a `0F 3F` thunk-stub address that routes through the per-function marshalling
// thunk instead. `function` is a compile-time template arg, so HleThunkT<function>
// knows the exact signature.
template <auto Function>
inline u64 GetHleImportAddr() {
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
    return static_cast<u64>(Core::CPU::EmitHleStub(&Core::CPU::HleThunkT<Function>::thunk));
#else
    return reinterpret_cast<u64>(Core::HostCallWrapperImpl<Function>::wrap);
#endif
}

#define LIB_FUNCTION(nid, lib, libversion, mod, function)                                          \
    do {                                                                                           \
        Core::Loader::SymbolResolver sr{};                                                         \
        sr.name = nid;                                                                             \
        sr.library = lib;                                                                          \
        sr.library_version = libversion;                                                           \
        sr.module = mod;                                                                           \
        sr.type = Core::Loader::SymbolType::Function;                                              \
        sym->AddSymbol(sr, GetHleImportAddr<function>());                                          \
    } while (0)

#define LIB_OBJ(nid, lib, libversion, mod, obj)                                                    \
    do {                                                                                           \
        Core::Loader::SymbolResolver sr{};                                                         \
        sr.name = nid;                                                                             \
        sr.library = lib;                                                                          \
        sr.library_version = libversion;                                                           \
        sr.module = mod;                                                                           \
        sr.type = Core::Loader::SymbolType::Object;                                                \
        sym->AddSymbol(sr, reinterpret_cast<u64>(obj));                                            \
    } while (0)

namespace Libraries {

void InitHLELibs(Core::Loader::SymbolsResolver* sym);

} // namespace Libraries
