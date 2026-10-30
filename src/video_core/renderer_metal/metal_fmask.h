// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <stdexcept>
#include "video_core/amdgpu/resource.h"

namespace Metal {

struct FmaskFormat {
    u32 bits;
    u32 samples;
    u32 fragments;
    u32 index_bits;
    AmdGpu::DataFormat storage_format;
};

[[nodiscard]] inline FmaskFormat GetFmaskFormat(AmdGpu::DataFormat format) {
    using AmdGpu::DataFormat;
    static constexpr std::array formats{
        FmaskFormat{8, 2, 1, 1, DataFormat::Format8},
        FmaskFormat{8, 4, 1, 1, DataFormat::Format8},
        FmaskFormat{8, 8, 1, 1, DataFormat::Format8},
        FmaskFormat{8, 2, 2, 1, DataFormat::Format8},
        FmaskFormat{8, 4, 2, 2, DataFormat::Format8},
        FmaskFormat{8, 4, 4, 2, DataFormat::Format8},
        FmaskFormat{16, 16, 1, 1, DataFormat::Format16},
        FmaskFormat{16, 8, 2, 2, DataFormat::Format16},
        FmaskFormat{32, 16, 2, 2, DataFormat::Format32},
        FmaskFormat{32, 8, 4, 4, DataFormat::Format32},
        FmaskFormat{32, 8, 8, 4, DataFormat::Format32},
        FmaskFormat{64, 16, 4, 4, DataFormat::Format32_32},
        FmaskFormat{64, 16, 8, 4, DataFormat::Format32_32},
    };
    if (!AmdGpu::IsFmask(format)) {
        throw std::runtime_error("Invalid Metal Fmask format");
    }
    return formats[u32(format) - u32(DataFormat::FormatFmask8_S2_F1)];
}

[[nodiscard]] inline AmdGpu::Image PhysicalFmaskImage(AmdGpu::Image image) {
    const auto format = GetFmaskFormat(image.GetDataFmt());
    const auto type = image.GetType();
    if (image.GetNumberFmt() != AmdGpu::NumberFormat::Uint ||
        (type != AmdGpu::ImageType::Color2D && type != AmdGpu::ImageType::Color2DArray) ||
        image.base_level || image.last_level || image.min_lod || image.compression_en) {
        throw std::runtime_error(
            "Metal Fmask view requires an uncompressed single-level Uint surface");
    }
    image.data_format = u64(format.storage_format);
    image.num_format = u64(AmdGpu::NumberFormat::Uint);
    image.dst_sel_x = u64(AmdGpu::CompSwizzle::Red);
    image.dst_sel_y = u64(AmdGpu::CompSwizzle::Green);
    image.dst_sel_z = u64(AmdGpu::CompSwizzle::Blue);
    image.dst_sel_w = u64(AmdGpu::CompSwizzle::Alpha);
    return image;
}

struct FmaskImageState {
    AmdGpu::DataFormat format{AmdGpu::DataFormat::FormatInvalid};
    AmdGpu::CompMapping mapping{AmdGpu::IdentityMapping};

    [[nodiscard]] static FmaskImageState FromImage(const AmdGpu::Image& image) {
        return AmdGpu::IsFmask(image.GetDataFmt())
                   ? FmaskImageState{image.GetDataFmt(), image.DstSelect()}
                   : FmaskImageState{};
    }

    bool operator==(const FmaskImageState&) const = default;
};

} // namespace Metal
