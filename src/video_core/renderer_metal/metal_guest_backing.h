// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <stdexcept>
#include <system_error>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include "common/types.h"

namespace Metal {

class GuestBacking {
public:
    explicit GuestBacking(u64 size) : size{size} {
        if (size == 0 || size % PageSize != 0 ||
            size > std::numeric_limits<size_t>::max() - PageSize) {
            throw std::invalid_argument("Invalid Metal guest backing size");
        }
        allocation = mmap(nullptr, size + PageSize, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (allocation == MAP_FAILED) {
            throw std::system_error(errno, std::generic_category(),
                                    "Failed to allocate Metal guest backing");
        }
        const auto address = reinterpret_cast<uintptr_t>(allocation);
        base = reinterpret_cast<u8*>((address + PageSize - 1) & ~(PageSize - 1));
    }

    GuestBacking(const GuestBacking&) = delete;
    GuestBacking& operator=(const GuestBacking&) = delete;

    ~GuestBacking() {
        munmap(allocation, size + PageSize);
    }

    [[nodiscard]] u8* Base() const noexcept {
        return base;
    }

    void* Map(VAddr address, PAddr offset, u64 length, int protection) {
        if (address == 0 || address % PageSize != 0 || offset % PageSize != 0 || length == 0 ||
            length % PageSize != 0 || offset > size || length > size - offset ||
            length > std::numeric_limits<VAddr>::max() - address ||
            (protection & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) != 0) {
            throw std::invalid_argument("Invalid Metal guest backing mapping");
        }
        const auto allocation_address = reinterpret_cast<VAddr>(allocation);
        if (address < allocation_address + size + PageSize &&
            allocation_address < address + length) {
            throw std::invalid_argument("Metal guest mapping overlaps canonical backing");
        }
        mach_vm_address_t target = address;
        vm_prot_t current{};
        vm_prot_t maximum{};
        const auto result =
            mach_vm_remap(mach_task_self(), &target, length, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                          mach_task_self(), reinterpret_cast<mach_vm_address_t>(base + offset),
                          FALSE, &current, &maximum, VM_INHERIT_COPY);
        if (result != KERN_SUCCESS || target != address || (maximum & protection) != protection) {
            throw std::runtime_error("Failed to alias Metal guest backing");
        }
        auto* mapped = reinterpret_cast<void*>(address);
        if (mprotect(mapped, length, protection) != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "Failed to protect Metal guest backing mapping");
        }
        return mapped;
    }

private:
    static constexpr u64 PageSize = 16_KB;
    const u64 size;
    void* allocation{};
    u8* base{};
};

} // namespace Metal
