// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <exception>
#include <functional>
#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct SrtMemoryRead {
    const std::function<void(VAddr, u64)>& prepare;
    std::exception_ptr error;

    static bool PS4_SYSV_ABI Prepare(SrtMemoryRead* context, VAddr address, u64 size) noexcept {
        try {
            context->prepare(address, size);
            return true;
        } catch (...) {
            context->error = std::current_exception();
            return false;
        }
    }
};

using PFN_SrtWalkerWithMemoryRead = void PS4_SYSV_ABI (*)(const u32*, u32*, SrtMemoryRead*);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
