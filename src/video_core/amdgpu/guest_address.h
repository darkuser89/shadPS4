// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>

#include "common/types.h"

namespace AmdGpu {

// Resolves a PS4 virtual address to the corresponding host-visible mapping. This is an identity
// operation on hosts where the guest address space can be mapped at its canonical addresses.
VAddr ResolveGuestAddress(VAddr address);

template <typename T = VAddr>
T ResolveGuestAddressAs(VAddr address) {
    return std::bit_cast<T>(ResolveGuestAddress(address));
}

} // namespace AmdGpu
