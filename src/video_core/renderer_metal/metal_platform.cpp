// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <sys/sysctl.h>

#include "common/assert.h"
#include "video_core/renderer_metal/metal_platform.h"

namespace Metal {

void RequireAppleSilicon() {
#if !defined(__aarch64__)
    int translated{};
    size_t size = sizeof(translated);
    if (sysctlbyname("sysctl.proc_translated", &translated, &size, nullptr, 0) != 0 ||
        translated != 1) {
        UNREACHABLE_MSG("The macOS renderer requires Apple Silicon; x86_64 builds require Rosetta");
    }
#endif
}

} // namespace Metal
