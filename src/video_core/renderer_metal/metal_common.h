// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#if !defined(__APPLE__)
#error "The Metal 4 renderer requires macOS"
#endif

#include <Metal/Metal.h>

#include "common/types.h"

namespace Metal {

inline const char* ErrorDescription(NSError* error) {
    return error ? error.localizedDescription.UTF8String : "unknown Metal error";
}

} // namespace Metal
