// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// FEXCore-dependent side of the HLE thunk boundary (only fex_backend includes
// this). The FEXCore-free marshalling pieces live in fex_hle.h (pulled in by
// tls.h via HOST_CALL). See fex_hle.h for the mechanism.
//
// aerolib wiring (NO Linker::Relocate change needed): on ARM64, HOST_CALL(func)
// (= HostCallWrapperImpl<func>, core/tls.h) returns a `0F 3F` stub emitted by
// EmitHleStub(&HleThunkT<func>::thunk) instead of the native wrap(). The existing
// LIB_FUNCTION registrations then register the stub address, which Relocate writes
// into the import slot unchanged. FexThunkHandler reads the host thunk pointer
// back out of the stub (pointer-in-hash — no per-name hashing).

#include <cstring>

#include "core/cpu/fex_hle.h"

#include <FEXCore/Core/Thunks.h>
#include <FEXCore/IR/IR.h>

namespace Core::CPU {

class FexThunkHandler final : public FEXCore::ThunkHandler {
public:
    FEXCore::ThunkedFunction* LookupThunk(const FEXCore::IR::SHA256Sum& field) override {
        FEXCore::ThunkedFunction* fn;
        std::memcpy(&fn, field.data, sizeof(fn));
        return fn;
    }
};

} // namespace Core::CPU
