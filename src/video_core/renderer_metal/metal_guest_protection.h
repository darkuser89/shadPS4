// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include "common/alignment.h"
#include "common/multi_level_page_table.h"
#include "core/memory.h"

namespace Metal {

class GuestMemoryProtection {
public:
    static constexpr u64 PageSize = 4096;
    static constexpr u64 AddressSpaceSize = u64{1} << 40;

    [[nodiscard]] static constexpr Core::MemoryPermission CpuPermissions(u32 flags) {
        const auto prot = static_cast<Core::MemoryProt>(flags);
        Core::MemoryPermission permissions{};
        if (True(prot & Core::MemoryProt::CpuReadWrite)) {
            permissions |= Core::MemoryPermission::Read;
        }
        if (True(prot & Core::MemoryProt::CpuWrite)) {
            permissions |= Core::MemoryPermission::Write;
        }
        if (True(prot & Core::MemoryProt::CpuExec)) {
            permissions |= Core::MemoryPermission::Execute;
        }
        return permissions;
    }

    [[nodiscard]] static constexpr Core::MemoryPermission Combine(Core::MemoryPermission guest,
                                                                  Core::MemoryPermission watches) {
        if (False(watches & Core::MemoryPermission::Read)) {
            return Core::MemoryPermission::None;
        }
        return (guest & watches & Core::MemoryPermission::ReadWrite) |
               (guest & Core::MemoryPermission::Execute);
    }

    [[nodiscard]] std::unique_lock<std::shared_mutex> LockPageRegistration() {
        return std::unique_lock{registration_mutex};
    }

    [[nodiscard]] std::shared_lock<std::shared_mutex> LockRegisteredPages() const {
        return std::shared_lock{registration_mutex};
    }

    void Reserve(VAddr address, u64 size) {
        if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address) {
            throw std::runtime_error("Invalid native guest protection registration");
        }
        std::unique_lock lock{mutex};
        pages.reserve(address / PageSize, (address + size - 1) / PageSize);
    }

    template <typename WatchPermissions, typename Protect>
    void Set(VAddr address, u64 size, Core::MemoryPermission permissions,
             WatchPermissions&& watches, Protect&& protect) {
        ValidateRange(address, size);
        std::unique_lock lock{mutex};
        for (u64 page = address / PageSize; page < (address + size) / PageSize; ++page) {
            auto* entry = pages.find(page);
            if (!entry) {
                throw std::runtime_error("Native guest protection requires registered pages");
            }
        }
        for (u64 page = address / PageSize; page < (address + size) / PageSize; ++page) {
            pages[page] = static_cast<u8>(permissions);
        }
        Apply(
            address, size, [&](u64 page) { return Combine(permissions, watches(page)); }, protect);
    }

    template <typename Protect>
    void ApplyWatches(VAddr address, u64 size, Core::MemoryPermission watches, Protect&& protect) {
        ValidateRange(address, size);
        std::shared_lock lock{mutex};
        Apply(address, size, [&](u64 page) { return Combine(Get(page), watches); }, protect);
    }

    [[nodiscard]] bool Allows(VAddr address, u64 size, Core::MemoryPermission access) const {
        if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address) {
            return false;
        }
        std::shared_lock lock{mutex};
        for (u64 page = address / PageSize; page <= (address + size - 1) / PageSize; ++page) {
            if ((Get(page) & access) != access) {
                return false;
            }
        }
        return true;
    }

private:
    static void ValidateRange(VAddr address, u64 size) {
        if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address ||
            address % PageSize || size % PageSize) {
            throw std::runtime_error("Invalid native guest protection range");
        }
    }

    template <typename Permissions, typename Protect>
    static void Apply(VAddr address, u64 size, Permissions&& permissions, Protect&& protect) {
        VAddr begin = address;
        auto current = permissions(address / PageSize);
        for (VAddr next = address + PageSize; next < address + size; next += PageSize) {
            const auto value = permissions(next / PageSize);
            if (value != current) {
                protect(begin, next - begin, current);
                begin = next;
                current = value;
            }
        }
        protect(begin, address + size - begin, current);
    }

    [[nodiscard]] Core::MemoryPermission Get(u64 page) const {
        const auto* entry = pages.find(page);
        return entry ? static_cast<Core::MemoryPermission>(*entry) : Core::MemoryPermission::None;
    }

    struct Traits {
        using Entry = u8;
        static constexpr size_t ADDRESS_SPACE_BITS = 40;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = 12;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<Traits> pages;
    mutable std::shared_mutex mutex;
    mutable std::shared_mutex registration_mutex;
};

} // namespace Metal
