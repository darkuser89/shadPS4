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

class GuestWriteAlias {
public:
    GuestWriteAlias(VAddr address, u64 size) {
        constexpr u64 PageSize = 16_KB;
        const auto offset = address % PageSize;
        if (address == 0 || size == 0 || size > std::numeric_limits<VAddr>::max() - address ||
            size > std::numeric_limits<u64>::max() - offset - (PageSize - 1)) {
            throw std::invalid_argument("Invalid Metal guest write alias range");
        }
        length = (size + offset + PageSize - 1) & ~(PageSize - 1);
        vm_prot_t current{};
        vm_prot_t maximum{};
        const auto result = mach_vm_remap(mach_task_self(), &mapping, length, 0, VM_FLAGS_ANYWHERE,
                                          mach_task_self(), address - offset, FALSE, &current,
                                          &maximum, VM_INHERIT_COPY);
        if (result != KERN_SUCCESS) {
            throw std::runtime_error("Failed to create Metal guest write alias");
        }
        if ((maximum & (VM_PROT_READ | VM_PROT_WRITE)) != (VM_PROT_READ | VM_PROT_WRITE)) {
            mach_vm_deallocate(mach_task_self(), mapping, length);
            throw std::runtime_error("Metal guest write alias is not writable");
        }
        if (mprotect(reinterpret_cast<void*>(mapping), length, PROT_READ | PROT_WRITE) != 0) {
            const auto error = errno;
            mach_vm_deallocate(mach_task_self(), mapping, length);
            throw std::system_error(error, std::generic_category(),
                                    "Failed to protect Metal guest write alias");
        }
        data = reinterpret_cast<u8*>(mapping) + offset;
    }

    GuestWriteAlias(const GuestWriteAlias&) = delete;
    GuestWriteAlias& operator=(const GuestWriteAlias&) = delete;

    ~GuestWriteAlias() {
        mach_vm_deallocate(mach_task_self(), mapping, length);
    }

    [[nodiscard]] u8* Data() const noexcept {
        return data;
    }

private:
    mach_vm_address_t mapping{};
    mach_vm_size_t length{};
    u8* data{};
};

} // namespace Metal
