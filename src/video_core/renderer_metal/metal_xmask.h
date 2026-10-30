// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "video_core/renderer_metal/metal_tiling.h"
namespace Metal {
struct XmaskLayout {
    enum class Kind { Cmask, Htile };
    static XmaskLayout Create(const TilingLayout& surface, Kind kind, bool linear);
    [[nodiscard]] u64 BitAddress(u32 tx, u32 ty, u32 layer) const;
    [[nodiscard]] u64 SliceBytes() const {
        return u64{width} * height * element_bits / 512;
    }
    u32 pipes;
    u32 macro_width;
    u32 macro_height;
    u32 width;
    u32 height;
    AmdGpu::PipeConfig pipe;
    bool linear;
    u32 element_bits;
    u32 cache_bits;
};
} // namespace Metal
