// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include <stdexcept>
#include "common/alignment.h"
#include "video_core/renderer_metal/metal_xmask.h"
namespace Metal {
XmaskLayout XmaskLayout::Create(const TilingLayout& surface, Kind kind, bool linear) {
    const auto pipe = surface.alt_tile ? AmdGpu::GetAltPipeConfig(surface.tile_mode)
                                       : AmdGpu::GetPipeConfig(surface.tile_mode);
    u32 pipes{};
    switch (pipe) {
    case AmdGpu::PipeConfig::P2:
        pipes = 2;
        break;
    case AmdGpu::PipeConfig::P8_32x32_8x16:
    case AmdGpu::PipeConfig::P8_32x32_16x16:
        pipes = 8;
        break;
    case AmdGpu::PipeConfig::P16_32x32_8x16:
    case AmdGpu::PipeConfig::P16_32x32_16x16:
        pipes = 16;
        break;
    default:
        throw std::runtime_error("Metal metadata pipe configuration requires lowering");
    }
    const u32 element_bits = kind == Kind::Cmask ? 4 : 32;
    const u32 cache_bits = kind == Kind::Cmask ? 1024 : 16384;
    u32 width = cache_bits / element_bits, height = 1;
    while (width > height * 2 * pipes && !(width & 1)) {
        width /= 2;
        height *= 2;
    }
    const u32 linear_alignment = pipes == 2 ? 32 : 64;
    const u32 macro_width = linear ? linear_alignment : width * 8;
    const u32 macro_height = linear ? linear_alignment : height * 8 * pipes;
    const u32 aligned_width = Common::AlignUp(u32(surface.mips[0].pitch), macro_width);
    u32 aligned_height = Common::AlignUp(u32(surface.mips[0].height), macro_height);
    while (kind == Kind::Cmask &&
           (u64{aligned_width} * aligned_height * element_bits / 512) % (pipes * 256)) {
        aligned_height += macro_height;
    }
    return {pipes, macro_width, macro_height, aligned_width, aligned_height,
            pipe,  linear,      element_bits, cache_bits};
}

u64 XmaskLayout::BitAddress(u32 tx, u32 ty, u32 layer) const {
    const auto& layout = *this;
    const u32 x0 = tx & 1, x1 = (tx >> 1) & 1, x2 = (tx >> 2) & 1, x3 = (tx >> 3) & 1;
    const u32 y0 = ty & 1, y1 = (ty >> 1) & 1, y2 = (ty >> 2) & 1, y3 = (ty >> 3) & 1;
    u32 pipe{}, element{}, shift{}, index_bits{};
    switch (layout.pipe) {
    case AmdGpu::PipeConfig::P2:
        pipe = x0 ^ y0;
        element = (x1 << 2) | ((x1 ^ y1) << 1) | (x1 ^ y0);
        shift = index_bits = 3;
        break;
    case AmdGpu::PipeConfig::P8_32x32_8x16:
        pipe = (x1 ^ y0 ^ x2) | ((x0 ^ y1) << 1) | ((x2 ^ y2) << 2);
        element = (x1 << 1) | (x1 ^ y1);
        shift = 1;
        index_bits = 2;
        break;
    case AmdGpu::PipeConfig::P8_32x32_16x16:
        pipe = (x0 ^ y0 ^ x1) | ((x1 ^ y1) << 1) | ((x2 ^ y2) << 2);
        element = (x1 << 1) | (x1 ^ y0);
        shift = 1;
        index_bits = 2;
        break;
    case AmdGpu::PipeConfig::P16_32x32_8x16:
        pipe = (x1 ^ y0) | ((x0 ^ y1) << 1) | ((x2 ^ y3) << 2) | ((x3 ^ y2) << 3);
        element = (x1 << 1) | (x1 ^ y1);
        index_bits = 2;
        break;
    case AmdGpu::PipeConfig::P16_32x32_16x16:
        pipe = (x0 ^ y0 ^ x1) | ((x1 ^ y1) << 1) | ((x2 ^ y3) << 2) | ((x3 ^ y2) << 3);
        element = (x1 << 1) | (x1 ^ y0);
        index_bits = 2;
        break;
    default:
        throw std::runtime_error("Invalid Metal metadata pipe layout");
    }
    u64 macro_bits{}, micro{};
    if (layout.linear) {
        micro = (tx / 4 + u64{ty / 4} * (layout.width / 32)) << shift;
        macro_bits =
            u64{layer} * (u64{layout.width} * layout.height / 64 / layout.pipes) * element_bits;
    } else {
        const u32 macro_x = layout.macro_width / 8;
        const u32 macro_y = layout.macro_height / 8;
        const u32 pitch = layout.width / layout.macro_width;
        const u64 macro = tx / macro_x + u64{ty / macro_y} * pitch +
                          u64{layer} * pitch * (layout.height / layout.macro_height);
        micro = u64{(tx % macro_x) / 4 + (ty % macro_y) / 4 * (macro_x / 4)} << shift;
        macro_bits = macro * cache_bits;
    }
    micro = (micro & ~((u64{1} << index_bits) - 1)) + element;
    const u64 bits = macro_bits + micro * element_bits;
    return ((bits % 2048) + pipe * 2048 + bits / 2048 * 2048 * layout.pipes);
}

} // namespace Metal
