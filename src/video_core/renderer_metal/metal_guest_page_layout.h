// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include "common/types.h"

namespace Metal {

class GuestPageTableLayout {
public:
    static constexpr u32 AddressBits = 40;
    static constexpr u32 PageBits = 14;
    static constexpr u32 LeafBits = 10;
    static constexpr u64 AddressSpaceSize = u64{1} << AddressBits;
    static constexpr u64 PageSize = u64{1} << PageBits;
    static constexpr u32 LeafEntries = u32{1} << LeafBits;
    static constexpr u32 RootEntries = u32{1} << (AddressBits - PageBits - LeafBits);
    using Leaf = std::array<u64, LeafEntries>;

    static constexpr u32 RootIndex(VAddr address) {
        return address >> (PageBits + LeafBits);
    }

    static constexpr u32 LeafIndex(VAddr address) {
        return (address >> PageBits) & (LeafEntries - 1);
    }

    void Map(VAddr address, u64 size, u64 gpu_address) {
        if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address ||
            address % PageSize || size % PageSize || !gpu_address || gpu_address % sizeof(u64) ||
            size - 1 > std::numeric_limits<u64>::max() - gpu_address) {
            throw std::runtime_error("Invalid native guest page mapping");
        }
        for (u64 offset = 0; offset < size; offset += PageSize) {
            const auto it = leaves.find(RootIndex(address + offset));
            if (it != leaves.end()) {
                const u64 previous = it->second[LeafIndex(address + offset)];
                if (previous && previous != gpu_address + offset) {
                    throw std::runtime_error("Conflicting native guest page mapping");
                }
            }
        }
        for (u64 offset = 0; offset < size; offset += PageSize) {
            leaves[RootIndex(address + offset)][LeafIndex(address + offset)] = gpu_address + offset;
        }
    }

    [[nodiscard]] const std::map<u32, Leaf>& Leaves() const noexcept {
        return leaves;
    }

private:
    std::map<u32, Leaf> leaves;
};

} // namespace Metal
