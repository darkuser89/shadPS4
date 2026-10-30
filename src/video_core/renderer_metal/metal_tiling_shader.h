// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Metal {

inline constexpr std::string_view TilingShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint Samples [[function_constant(0)]];
constant uint MicroMode [[function_constant(1)]];
constant uint Thickness [[function_constant(2)]];
constant uint IsTiler [[function_constant(3)]];
constant uint ArrayMode [[function_constant(4)]];
constant uint PipeConfig [[function_constant(5)]];
constant uint BankWidth [[function_constant(6)]];
constant uint BankHeight [[function_constant(7)]];
constant uint Banks [[function_constant(8)]];
constant uint BankBits [[function_constant(9)]];
constant uint TileSplit [[function_constant(10)]];
constant uint MacroAspect [[function_constant(11)]];
constant uint IsMacro [[function_constant(12)]];
constant uint Bits [[function_constant(13)]];

struct MipInfo {
    uint size;
    ushort pitch;
    ushort height;
    uint offset;
};

struct Parameters {
    uint bank_swizzle;
    uint micro_tiled_mips;
    uint num_mips;
    uint element_count;
    MipInfo mips[16];
};

uint Bit(uint value, uint position) {
    return (value >> position) & 1;
}

uint PixelIndex(uint x, uint y, uint z, uint thickness) {
    uint x0 = Bit(x, 0), x1 = Bit(x, 1), x2 = Bit(x, 2);
    uint y0 = Bit(y, 0), y1 = Bit(y, 1), y2 = Bit(y, 2);
    uint z0 = Bit(z, 0), z1 = Bit(z, 1), z2 = Bit(z, 2);
    uint p[9] = {};
    if (MicroMode == 0) {
        if (Bits == 8) {
            p[0] = x0; p[1] = x1; p[2] = x2; p[3] = y1; p[4] = y0; p[5] = y2;
        } else if (Bits == 16) {
            p[0] = x0; p[1] = x1; p[2] = x2; p[3] = y0; p[4] = y1; p[5] = y2;
        } else if (Bits == 32) {
            p[0] = x0; p[1] = x1; p[2] = y0; p[3] = x2; p[4] = y1; p[5] = y2;
        } else if (Bits == 64) {
            p[0] = x0; p[1] = y0; p[2] = x1; p[3] = x2; p[4] = y1; p[5] = y2;
        } else if (Bits == 128) {
            p[0] = y0; p[1] = x0; p[2] = x1; p[3] = x2; p[4] = y1; p[5] = y2;
        }
    } else if (MicroMode == 1 || MicroMode == 2) {
        p[0] = x0; p[1] = y0; p[2] = x1; p[3] = y1; p[4] = x2; p[5] = y2;
    } else {
        if (Bits == 8 || Bits == 16) {
            p[0] = x0; p[1] = y0; p[2] = x1; p[3] = y1; p[4] = z0; p[5] = z1;
        } else if (Bits == 32) {
            p[0] = x0; p[1] = y0; p[2] = x1; p[3] = z0; p[4] = y1; p[5] = z1;
        } else {
            p[0] = x0; p[1] = y0; p[2] = z0; p[3] = x1; p[4] = y1; p[5] = z1;
        }
        p[6] = x2; p[7] = y2;
        p[8] = thickness == 8 ? z2 : 0;
    }
    uint index = 0;
    for (uint i = 0; i < 9; ++i) {
        index |= p[i] << i;
    }
    return index;
}

uint Pipes() {
    return PipeConfig == 0 ? 2 : (PipeConfig >= 16 ? 16 : 8);
}

uint PipeBits() {
    return PipeConfig == 0 ? 1 : (PipeConfig >= 16 ? 4 : 3);
}

uint Pipe(uint x, uint y, uint slice) {
    uint x3 = Bit(x, 3), x4 = Bit(x, 4), x5 = Bit(x, 5), x6 = Bit(x, 6);
    uint y3 = Bit(y, 3), y4 = Bit(y, 4), y5 = Bit(y, 5), y6 = Bit(y, 6);
    uint pipe = 0;
    if (PipeConfig == 0) {
        pipe = x3 ^ y3;
    } else if (PipeConfig == 10) {
        pipe = (x4 ^ y3 ^ x5) | ((x3 ^ y4) << 1) | ((x5 ^ y5) << 2);
    } else if (PipeConfig == 12) {
        pipe = (x3 ^ y3 ^ x4) | ((x4 ^ y4) << 1) | ((x5 ^ y5) << 2);
    } else if (PipeConfig == 16) {
        pipe = (x4 ^ y3) | ((x3 ^ y4) << 1) | ((x5 ^ y6) << 2) | ((x6 ^ y5) << 3);
    } else if (PipeConfig == 17) {
        pipe = (x3 ^ y3 ^ x4) | ((x4 ^ y4) << 1) | ((x5 ^ y6) << 2) | ((x6 ^ y5) << 3);
    }
    uint rotation = 0;
    if (ArrayMode == 12 || ArrayMode == 13 || ArrayMode == 14) {
        rotation = max(1u, Pipes() / 2 - 1) * (slice / Thickness);
    }
    return pipe ^ (rotation & (Pipes() - 1));
}

uint Bank(uint x, uint y, uint slice, uint split_slice, constant Parameters& info) {
    uint tx = x / (8 * BankWidth * Pipes());
    uint ty = y / (8 * BankHeight);
    uint x3 = Bit(tx, 0), x4 = Bit(tx, 1), x5 = Bit(tx, 2), x6 = Bit(tx, 3);
    uint y3 = Bit(ty, 0), y4 = Bit(ty, 1), y5 = Bit(ty, 2), y6 = Bit(ty, 3);
    uint bank = 0;
    if (Banks == 16) {
        bank = (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) | ((x6 ^ y3) << 3);
    } else if (Banks == 8) {
        bank = (x3 ^ y5) | ((x4 ^ y4 ^ y5) << 1) | ((x5 ^ y3) << 2);
    } else if (Banks == 4) {
        bank = (x3 ^ y4) | ((x4 ^ y3) << 1);
    } else if (Banks == 2) {
        bank = x3 ^ y3;
    }
    uint slice_rotation = 0;
    if (ArrayMode == 4 || ArrayMode == 7 || ArrayMode == 8) {
        slice_rotation = (Banks / 2 - 1) * (slice / Thickness);
    } else if (ArrayMode == 12 || ArrayMode == 13 || ArrayMode == 14) {
        slice_rotation = max(1u, Pipes() / 2 - 1) * (slice / Thickness) / Pipes();
    }
    uint split_rotation = 0;
    if (ArrayMode == 4 || ArrayMode == 12 || ArrayMode == 6 || ArrayMode == 11) {
        split_rotation = (Banks / 2 + 1) * split_slice;
    }
    return (bank ^ (info.bank_swizzle + slice_rotation) ^ split_rotation) & (Banks - 1);
}

ulong MicroAddress(uint x, uint y, uint slice, uint pitch, uint height,
                   uint thickness, uint sample) {
    ulong tile_bytes = ulong(64) * thickness * (Bits / 8) * Samples;
    ulong slice_bytes = ulong(pitch) * height * thickness * (Bits / 8) * Samples;
    ulong tile_offset = (ulong(y / 8) * (pitch / 8) + x / 8) * tile_bytes;
    ulong pixel_index = PixelIndex(x, y, slice, thickness);
    ulong element_offset = MicroMode == 2
        ? (pixel_index * Samples + sample) * (Bits / 8)
        : pixel_index * (Bits / 8) + sample * tile_bytes / Samples;
    return (slice / thickness) * slice_bytes + tile_offset + element_offset;
}

ulong MacroAddress(uint x, uint y, uint slice, uint pitch, uint height,
                   uint sample, constant Parameters& info) {
    ulong tile_bytes = ulong(64) * Thickness * (Bits / 8) * Samples;
    ulong pixel_index = PixelIndex(x, y, slice, Thickness);
    ulong element_offset = MicroMode == 2
        ? (pixel_index * Samples + sample) * (Bits / 8)
        : pixel_index * (Bits / 8) + sample * tile_bytes / Samples;
    uint splits = 1, split_slice = 0;
    if (tile_bytes > TileSplit && Thickness == 1) {
        splits = uint(tile_bytes / TileSplit);
        split_slice = uint(element_offset / TileSplit);
        element_offset %= TileSplit;
        tile_bytes = TileSplit;
    }
    uint macro_pitch = 8 * BankWidth * Pipes() * MacroAspect;
    uint macro_height = 8 * BankHeight * Banks / MacroAspect;
    ulong macro_bytes = tile_bytes * (macro_pitch / 8) * (macro_height / 8) / (Pipes() * Banks);
    uint macros_per_row = pitch / macro_pitch;
    ulong macro_offset = (ulong(y / macro_height) * macros_per_row + x / macro_pitch) * macro_bytes;
    ulong slice_bytes = ulong(macros_per_row) * (height / macro_height) * macro_bytes;
    ulong slice_offset = slice_bytes * (split_slice + ulong(splits) * (slice / Thickness));
    uint tile_index = ((y / 8) % BankHeight) * BankWidth + ((x / 8) / Pipes()) % BankWidth;
    ulong total = slice_offset + macro_offset + element_offset + tile_index * tile_bytes;
    if (ArrayMode == 5 || ArrayMode == 6 || ArrayMode == 9 || ArrayMode == 10 ||
        ArrayMode == 11 || ArrayMode == 15) {
        x %= macro_pitch;
        y %= macro_height;
    }
    return (total & 255) | (ulong(Pipe(x, y, slice)) << 8) |
           (ulong(Bank(x, y, slice, split_slice, info)) << (8 + PipeBits())) |
           ((total >> 8) << (8 + PipeBits() + BankBits));
}

template <typename T>
void CopyElement(device uchar* destination, device uchar* source,
                 ulong destination_offset, ulong source_offset) {
    *reinterpret_cast<device T*>(destination + destination_offset) =
        *reinterpret_cast<device T*>(source + source_offset);
}

kernel void transform_tiles(device uchar* tiled [[buffer(0)]],
                            device uchar* linear [[buffer(1)]],
                            constant Parameters& info [[buffer(2)]],
                            device atomic_uint& errors [[buffer(3)]],
                            uint global_index [[thread_position_in_grid]]) {
    if (global_index >= info.element_count) {
        return;
    }
    uint element = global_index;
    uint mip = 0;
    for (; mip < info.num_mips; ++mip) {
        uint count = info.mips[mip].size / (Bits / 8);
        if (element < count) {
            break;
        }
        element -= count;
    }
    if (mip == info.num_mips) {
        atomic_fetch_or_explicit(&errors, 1u, memory_order_relaxed);
        return;
    }
    uint sample = element % Samples;
    uint pixel = element / Samples;
    uint pitch = info.mips[mip].pitch;
    uint height = info.mips[mip].height;
    uint x = pixel % pitch;
    uint y = (pixel / pitch) % height;
    uint slice = pixel / (pitch * height);
    bool micro = !IsMacro || (info.micro_tiled_mips & (1u << mip)) != 0;
    uint thickness = IsMacro && micro ? min(Thickness, 4u) : Thickness;
    ulong relative_offset = micro
        ? MicroAddress(x, y, slice, pitch, height, thickness, sample)
        : MacroAddress(x, y, slice, pitch, height, sample, info);
    if (relative_offset > info.mips[mip].size - Bits / 8) {
        atomic_fetch_or_explicit(&errors, 2u, memory_order_relaxed);
        return;
    }
    ulong tiled_offset = info.mips[mip].offset + relative_offset;
    ulong linear_offset = ulong(global_index) * (Bits / 8);
    device uchar* destination = IsTiler ? tiled : linear;
    device uchar* source = IsTiler ? linear : tiled;
    ulong destination_offset = IsTiler ? tiled_offset : linear_offset;
    ulong source_offset = IsTiler ? linear_offset : tiled_offset;
    if (Bits == 8) {
        CopyElement<uchar>(destination, source, destination_offset, source_offset);
    } else if (Bits == 16) {
        CopyElement<ushort>(destination, source, destination_offset, source_offset);
    } else if (Bits == 32) {
        CopyElement<uint>(destination, source, destination_offset, source_offset);
    } else if (Bits == 64) {
        CopyElement<uint2>(destination, source, destination_offset, source_offset);
    } else if (Bits == 96) {
        CopyElement<packed_uint3>(destination, source, destination_offset, source_offset);
    } else {
        CopyElement<uint4>(destination, source, destination_offset, source_offset);
    }
}
)MSL";

} // namespace Metal
