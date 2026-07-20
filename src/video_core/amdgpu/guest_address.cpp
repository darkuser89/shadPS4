// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/memory.h"
#include "video_core/amdgpu/guest_address.h"

namespace AmdGpu {

VAddr ResolveGuestAddress(VAddr address) {
    return Core::Memory::Instance()->TranslateCanonicalGuestAddress(address);
}

} // namespace AmdGpu
