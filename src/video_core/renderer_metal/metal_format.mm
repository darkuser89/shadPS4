// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <stdexcept>
#include <fmt/format.h>
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_metal/metal_format.h"

namespace Metal {

namespace {

struct SurfaceFormats {
    AmdGpu::DataFormat data;
    std::array<MTLPixelFormat, 6> formats;
};

constexpr auto Invalid = MTLPixelFormatInvalid;
constexpr std::array Formats{
    SurfaceFormats{AmdGpu::DataFormat::Format8,
                   {MTLPixelFormatR8Unorm, MTLPixelFormatR8Snorm, MTLPixelFormatR8Uint,
                    MTLPixelFormatR8Sint, Invalid, MTLPixelFormatR8Unorm_sRGB}},
    SurfaceFormats{AmdGpu::DataFormat::Format16,
                   {MTLPixelFormatR16Unorm, MTLPixelFormatR16Snorm, MTLPixelFormatR16Uint,
                    MTLPixelFormatR16Sint, MTLPixelFormatR16Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format8_8,
                   {MTLPixelFormatRG8Unorm, MTLPixelFormatRG8Snorm, MTLPixelFormatRG8Uint,
                    MTLPixelFormatRG8Sint, Invalid, MTLPixelFormatRG8Unorm_sRGB}},
    SurfaceFormats{AmdGpu::DataFormat::Format32,
                   {Invalid, Invalid, MTLPixelFormatR32Uint, MTLPixelFormatR32Sint,
                    MTLPixelFormatR32Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format16_16,
                   {MTLPixelFormatRG16Unorm, MTLPixelFormatRG16Snorm, MTLPixelFormatRG16Uint,
                    MTLPixelFormatRG16Sint, MTLPixelFormatRG16Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format10_11_11,
                   {Invalid, Invalid, Invalid, Invalid, MTLPixelFormatRG11B10Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format2_10_10_10,
                   {MTLPixelFormatRGB10A2Unorm, Invalid, MTLPixelFormatRGB10A2Uint, Invalid,
                    Invalid, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format8_8_8_8,
                   {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Snorm, MTLPixelFormatRGBA8Uint,
                    MTLPixelFormatRGBA8Sint, Invalid, MTLPixelFormatRGBA8Unorm_sRGB}},
    SurfaceFormats{AmdGpu::DataFormat::Format32_32,
                   {Invalid, Invalid, MTLPixelFormatRG32Uint, MTLPixelFormatRG32Sint,
                    MTLPixelFormatRG32Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format16_16_16_16,
                   {MTLPixelFormatRGBA16Unorm, MTLPixelFormatRGBA16Snorm, MTLPixelFormatRGBA16Uint,
                    MTLPixelFormatRGBA16Sint, MTLPixelFormatRGBA16Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format32_32_32_32,
                   {Invalid, Invalid, MTLPixelFormatRGBA32Uint, MTLPixelFormatRGBA32Sint,
                    MTLPixelFormatRGBA32Float, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::Format5_9_9_9,
                   {Invalid, Invalid, Invalid, Invalid, MTLPixelFormatRGB9E5Float, Invalid}},
    SurfaceFormats{
        AmdGpu::DataFormat::FormatBc1,
        {MTLPixelFormatBC1_RGBA, Invalid, Invalid, Invalid, Invalid, MTLPixelFormatBC1_RGBA_sRGB}},
    SurfaceFormats{
        AmdGpu::DataFormat::FormatBc2,
        {MTLPixelFormatBC2_RGBA, Invalid, Invalid, Invalid, Invalid, MTLPixelFormatBC2_RGBA_sRGB}},
    SurfaceFormats{
        AmdGpu::DataFormat::FormatBc3,
        {MTLPixelFormatBC3_RGBA, Invalid, Invalid, Invalid, Invalid, MTLPixelFormatBC3_RGBA_sRGB}},
    SurfaceFormats{
        AmdGpu::DataFormat::FormatBc4,
        {MTLPixelFormatBC4_RUnorm, MTLPixelFormatBC4_RSnorm, Invalid, Invalid, Invalid, Invalid}},
    SurfaceFormats{
        AmdGpu::DataFormat::FormatBc5,
        {MTLPixelFormatBC5_RGUnorm, MTLPixelFormatBC5_RGSnorm, Invalid, Invalid, Invalid, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::FormatBc6,
                   {MTLPixelFormatBC6H_RGBUfloat, MTLPixelFormatBC6H_RGBFloat, Invalid, Invalid,
                    Invalid, Invalid}},
    SurfaceFormats{AmdGpu::DataFormat::FormatBc7,
                   {MTLPixelFormatBC7_RGBAUnorm, Invalid, Invalid, Invalid, Invalid,
                    MTLPixelFormatBC7_RGBAUnorm_sRGB}},
};

constexpr auto FormatTable = [] {
    std::array<std::array<MTLPixelFormat, 6>, AmdGpu::NUM_DATA_FMTS> table{};
    for (const auto& entry : Formats) {
        table[u32(entry.data)] = entry.formats;
    }
    return table;
}();

MTLTextureSwizzle Swizzle(AmdGpu::CompSwizzle swizzle) {
    switch (swizzle) {
    case AmdGpu::CompSwizzle::Zero:
        return MTLTextureSwizzleZero;
    case AmdGpu::CompSwizzle::One:
        return MTLTextureSwizzleOne;
    case AmdGpu::CompSwizzle::Red:
        return MTLTextureSwizzleRed;
    case AmdGpu::CompSwizzle::Green:
        return MTLTextureSwizzleGreen;
    case AmdGpu::CompSwizzle::Blue:
        return MTLTextureSwizzleBlue;
    case AmdGpu::CompSwizzle::Alpha:
        return MTLTextureSwizzleAlpha;
    default:
        throw std::runtime_error("Invalid Metal guest component selection");
    }
}

} // namespace

MTLVertexFormat VertexFormat(const AmdGpu::Buffer& buffer) {
    using Data = AmdGpu::DataFormat;
    using Number = AmdGpu::NumberFormat;
    const auto data = buffer.GetDataFmt();
    const auto number = buffer.GetNumberFmt();
    const auto select = [&](const auto& formats) {
        switch (number) {
        case Number::Uint:
            return formats[0];
        case Number::Sint:
            return formats[1];
        case Number::Unorm:
            return formats[2];
        case Number::Snorm:
            return formats[3];
        case Number::Float:
            return formats[4];
        default:
            return MTLVertexFormatInvalid;
        }
    };
    using Formats = std::array<MTLVertexFormat, 5>;
    MTLVertexFormat format = MTLVertexFormatInvalid;
    switch (data) {
    case Data::Format8:
        format = select(Formats{MTLVertexFormatUChar, MTLVertexFormatChar,
                                MTLVertexFormatUCharNormalized, MTLVertexFormatCharNormalized,
                                MTLVertexFormatInvalid});
        break;
    case Data::Format8_8:
        format = select(Formats{MTLVertexFormatUChar2, MTLVertexFormatChar2,
                                MTLVertexFormatUChar2Normalized, MTLVertexFormatChar2Normalized,
                                MTLVertexFormatInvalid});
        break;
    case Data::Format8_8_8_8:
        format = select(Formats{MTLVertexFormatUChar4, MTLVertexFormatChar4,
                                MTLVertexFormatUChar4Normalized, MTLVertexFormatChar4Normalized,
                                MTLVertexFormatInvalid});
        break;
    case Data::Format16:
        format = select(Formats{MTLVertexFormatUShort, MTLVertexFormatShort,
                                MTLVertexFormatUShortNormalized, MTLVertexFormatShortNormalized,
                                MTLVertexFormatHalf});
        break;
    case Data::Format16_16:
        format = select(Formats{MTLVertexFormatUShort2, MTLVertexFormatShort2,
                                MTLVertexFormatUShort2Normalized, MTLVertexFormatShort2Normalized,
                                MTLVertexFormatHalf2});
        break;
    case Data::Format16_16_16_16:
        format = select(Formats{MTLVertexFormatUShort4, MTLVertexFormatShort4,
                                MTLVertexFormatUShort4Normalized, MTLVertexFormatShort4Normalized,
                                MTLVertexFormatHalf4});
        break;
    case Data::Format32:
        format = select(Formats{MTLVertexFormatUInt, MTLVertexFormatInt, MTLVertexFormatInvalid,
                                MTLVertexFormatInvalid, MTLVertexFormatFloat});
        break;
    case Data::Format32_32:
        format = select(Formats{MTLVertexFormatUInt2, MTLVertexFormatInt2, MTLVertexFormatInvalid,
                                MTLVertexFormatInvalid, MTLVertexFormatFloat2});
        break;
    case Data::Format32_32_32:
        format = select(Formats{MTLVertexFormatUInt3, MTLVertexFormatInt3, MTLVertexFormatInvalid,
                                MTLVertexFormatInvalid, MTLVertexFormatFloat3});
        break;
    case Data::Format32_32_32_32:
        format = select(Formats{MTLVertexFormatUInt4, MTLVertexFormatInt4, MTLVertexFormatInvalid,
                                MTLVertexFormatInvalid, MTLVertexFormatFloat4});
        break;
    case Data::Format2_10_10_10:
        if (number == Number::Unorm)
            format = MTLVertexFormatUInt1010102Normalized;
        if (number == Number::Snorm)
            format = MTLVertexFormatInt1010102Normalized;
        break;
    case Data::Format10_11_11:
        if (number == Number::Float)
            format = MTLVertexFormatFloatRG11B10;
        break;
    case Data::Format5_9_9_9:
        if (number == Number::Float)
            format = MTLVertexFormatFloatRGB9E5;
        break;
    default:
        break;
    }
    return format;
}

MTLPixelFormat SurfaceFormat(AmdGpu::DataFormat data, AmdGpu::NumberFormat number) {
    const auto normalized_number = AmdGpu::RemapNumberFormat(number, data);
    const auto normalized_data = AmdGpu::RemapDataFormat(data);
    size_t column{};
    switch (normalized_number) {
    case AmdGpu::NumberFormat::Unorm:
        column = 0;
        break;
    case AmdGpu::NumberFormat::Snorm:
        column = 1;
        break;
    case AmdGpu::NumberFormat::Uint:
        column = 2;
        break;
    case AmdGpu::NumberFormat::Sint:
        column = 3;
        break;
    case AmdGpu::NumberFormat::Float:
        column = 4;
        break;
    case AmdGpu::NumberFormat::Srgb:
        column = 5;
        break;
    default:
        throw std::runtime_error("Metal guest surface requires number-format conversion");
    }
    if (u32(normalized_data) < FormatTable.size() &&
        FormatTable[u32(normalized_data)][column] != Invalid) {
        return FormatTable[u32(normalized_data)][column];
    }
    throw std::runtime_error(
        fmt::format("Metal guest surface requires format lowering: {}:{}", u32(data), u32(number)));
}

MTLPixelFormat DepthFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR16Unorm:
        return MTLPixelFormatDepth16Unorm;
    case MTLPixelFormatR32Float:
        return MTLPixelFormatDepth32Float;
    case MTLPixelFormatR8Uint:
        return MTLPixelFormatStencil8;
    default:
        throw std::runtime_error("Unsupported Metal guest depth surface format");
    }
}

MTLPixelFormat StorageFormat(MTLPixelFormat format) {
    switch (LinearFormat(format)) {
    case MTLPixelFormatA8Unorm:
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
        return MTLPixelFormatR8Uint;
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatB5G6R5Unorm:
    case MTLPixelFormatA1BGR5Unorm:
    case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm:
        return MTLPixelFormatR16Uint;
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatBGR10A2Unorm:
        return MTLPixelFormatR32Uint;
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA16Float:
        return MTLPixelFormatRG32Uint;
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
    case MTLPixelFormatRGBA32Float:
        return MTLPixelFormatRGBA32Uint;
    default:
        return format;
    }
}

MTLPixelFormat LinearFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Unorm_sRGB:
        return MTLPixelFormatR8Unorm;
    case MTLPixelFormatRG8Unorm_sRGB:
        return MTLPixelFormatRG8Unorm;
    case MTLPixelFormatRGBA8Unorm_sRGB:
        return MTLPixelFormatRGBA8Unorm;
    case MTLPixelFormatBGRA8Unorm_sRGB:
        return MTLPixelFormatBGRA8Unorm;
    case MTLPixelFormatBC1_RGBA_sRGB:
        return MTLPixelFormatBC1_RGBA;
    case MTLPixelFormatBC2_RGBA_sRGB:
        return MTLPixelFormatBC2_RGBA;
    case MTLPixelFormatBC3_RGBA_sRGB:
        return MTLPixelFormatBC3_RGBA;
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
        return MTLPixelFormatBC7_RGBAUnorm;
    default:
        return format;
    }
}

MTLTextureSwizzleChannels ComponentMapping(AmdGpu::CompMapping mapping) {
    return MTLTextureSwizzleChannelsMake(Swizzle(mapping.r), Swizzle(mapping.g), Swizzle(mapping.b),
                                         Swizzle(mapping.a));
}

} // namespace Metal
