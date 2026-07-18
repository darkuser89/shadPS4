// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#include "common/arch.h"
#include "common/types.h"
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
#include "core/cpu/fex_backend.h"
#include "core/cpu/fex_hle.h"
#endif

namespace Core::CPU {

template <class T>
inline constexpr bool kGuestCallbackScalar =
    std::is_integral_v<std::remove_cvref_t<T>> || std::is_enum_v<std::remove_cvref_t<T>> ||
    std::is_pointer_v<std::remove_cvref_t<T>>;

template <class T>
u64 PackGuestCallbackArg(T&& value) {
    using Bare = std::remove_cvref_t<T>;
    if constexpr (std::is_pointer_v<Bare>) {
        return reinterpret_cast<u64>(value);
    } else if constexpr (std::is_enum_v<Bare>) {
        return static_cast<u64>(static_cast<std::underlying_type_t<Bare>>(value));
    } else {
        return static_cast<u64>(value);
    }
}

/// Invoke a callback that may either be a native shadPS4 helper or an x86-64
/// function supplied by the guest. The latter is routed through FEXCore instead
/// of being executed as ARM64. Scalar arguments beyond the six SysV GPR slots
/// are placed on the guest stack by FexBackend.
template <class Fn, class... Args>
decltype(auto) InvokeGuestOrHost(Fn fn, Args&&... args) {
    using Ret = std::invoke_result_t<Fn, Args...>;
#if defined(ARCH_ARM64) && defined(SHAD_ENABLE_FEX)
    auto& fex = FexBackend::Instance();
    const VAddr callback_addr = reinterpret_cast<VAddr>(fn);
    if (fex.IsGuestAddress(callback_addr)) {
        if constexpr ((kGuestCallbackScalar<Args> && ...) &&
                      (std::is_void_v<Ret> || kGuestCallbackScalar<Ret>)) {
            std::array<u64, sizeof...(Args)> packed{};
            size_t i = 0;
            ((packed[i++] = PackGuestCallbackArg(std::forward<Args>(args))), ...);
            const u64 raw = fex.CallGuestCallback(callback_addr, packed);
            if constexpr (std::is_void_v<Ret>) {
                return;
            } else if constexpr (std::is_pointer_v<std::remove_cvref_t<Ret>>) {
                return reinterpret_cast<Ret>(raw);
            } else {
                return static_cast<Ret>(raw);
            }
        } else {
            UnsupportedHleAbi(__PRETTY_FUNCTION__);
        }
    }
#endif
    return std::invoke(fn, std::forward<Args>(args)...);
}

} // namespace Core::CPU
