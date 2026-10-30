// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>
#include "common/hash.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_sampler.h"

namespace Metal {

namespace {

MTLSamplerAddressMode AddressMode(AmdGpu::ClampMode mode) {
    switch (mode) {
    case AmdGpu::ClampMode::Wrap:
        return MTLSamplerAddressModeRepeat;
    case AmdGpu::ClampMode::Mirror:
        return MTLSamplerAddressModeMirrorRepeat;
    case AmdGpu::ClampMode::ClampLastTexel:
        return MTLSamplerAddressModeClampToEdge;
    case AmdGpu::ClampMode::MirrorOnceLastTexel:
        return MTLSamplerAddressModeMirrorClampToEdge;
    case AmdGpu::ClampMode::ClampBorder:
        return MTLSamplerAddressModeClampToBorderColor;
    default:
        throw std::runtime_error("Metal guest sampler address mode requires shader lowering");
    }
}

MTLSamplerMinMagFilter Filter(AmdGpu::Filter filter) {
    switch (filter) {
    case AmdGpu::Filter::Point:
    case AmdGpu::Filter::AnisoPoint:
        return MTLSamplerMinMagFilterNearest;
    case AmdGpu::Filter::Bilinear:
    case AmdGpu::Filter::AnisoLinear:
        return MTLSamplerMinMagFilterLinear;
    default:
        throw std::runtime_error("Invalid Metal guest sampler filter");
    }
}

MTLSamplerBorderColor BorderColor(AmdGpu::BorderColor color) {
    switch (color) {
    case AmdGpu::BorderColor::TransparentBlack:
        return MTLSamplerBorderColorTransparentBlack;
    case AmdGpu::BorderColor::OpaqueBlack:
        return MTLSamplerBorderColorOpaqueBlack;
    case AmdGpu::BorderColor::White:
        return MTLSamplerBorderColorOpaqueWhite;
    default:
        throw std::runtime_error("Metal custom guest border color requires shader lowering");
    }
}

} // namespace

SamplerCache::SamplerCache(const Instance& instance_) : instance{instance_} {}

size_t SamplerCache::KeyHash::operator()(const Key& key) const noexcept {
    return HashCombine(HashCombine(key.raw0, key.raw1), key.is_depth);
}

id<MTLSamplerState> SamplerCache::ObtainSampler(const AmdGpu::Sampler& sampler, bool is_depth,
                                                bool native_bias) {
    if (!sampler.Valid() || sampler.MinLod() > sampler.MaxLod()) {
        throw std::runtime_error("Metal guest sampler LOD requires shader lowering");
    }
    if (native_bias && (!instance.SupportsSamplerLodBias() ||
                        !NativeSamplerLodBiasCompatible(sampler.LodBias()))) {
        throw std::runtime_error("Metal native sampler LOD bias is unsupported or loses precision");
    }
    const bool anisotropic = HasSamplerAnisotropy(sampler);
    const bool uses_border = sampler.clamp_x == AmdGpu::ClampMode::ClampBorder ||
                             sampler.clamp_y == AmdGpu::ClampMode::ClampBorder ||
                             sampler.clamp_z == AmdGpu::ClampMode::ClampBorder;
    constexpr u64 StateMask0 = 0x1ff | (u64{3} << 29) | (u64{0xffffff} << 32);
    constexpr u64 StateMask1 = (u64{5} << 20) | (u64{3} << 26);
    const Key key{sampler.raw0 & (StateMask0 | (anisotropic ? u64{7} << 9 : 0) |
                                  (is_depth ? u64{7} << 12 : 0)),
                  sampler.raw1 & (StateMask1 | (uses_border ? u64{3} << 62 : 0) |
                                  (native_bias ? 0x3fffu : 0u)),
                  is_depth};
    if (auto it = samplers.find(key); it != samplers.end()) {
        return it->second;
    }
    auto descriptor = [MTLSamplerDescriptor new];
    descriptor.minFilter = Filter(sampler.xy_min_filter);
    descriptor.magFilter = Filter(sampler.xy_mag_filter);
    descriptor.mipFilter = static_cast<MTLSamplerMipFilter>(sampler.mip_filter.Value());
    descriptor.sAddressMode = AddressMode(sampler.clamp_x);
    descriptor.tAddressMode = AddressMode(sampler.clamp_y);
    descriptor.rAddressMode = AddressMode(sampler.clamp_z);
    if (uses_border) {
        descriptor.borderColor = BorderColor(sampler.border_color_type);
    }
    descriptor.normalizedCoordinates = YES;
    descriptor.lodMinClamp = sampler.MinLod();
    descriptor.lodMaxClamp = sampler.MaxLod();
    if (native_bias) {
        descriptor.lodBias = sampler.LodBias();
    }
    descriptor.maxAnisotropy = anisotropic ? sampler.MaxAniso() : 1;
    descriptor.compareFunction =
        is_depth ? static_cast<MTLCompareFunction>(sampler.depth_compare_func.Value())
                 : MTLCompareFunctionNever;
    descriptor.supportArgumentBuffers = YES;
    if (sampler.filter_mode != AmdGpu::FilterMode::Blend) {
        if (![instance.GetDevice() supportsFamily:MTLGPUFamilyApple10]) {
            throw std::runtime_error(
                "Metal guest sampler reduction requires native shader lowering on this GPU");
        }
        const bool all_linear = descriptor.minFilter == MTLSamplerMinMagFilterLinear &&
                                descriptor.magFilter == MTLSamplerMinMagFilterLinear &&
                                descriptor.mipFilter == MTLSamplerMipFilterLinear;
        const bool all_nearest = descriptor.minFilter == MTLSamplerMinMagFilterNearest &&
                                 descriptor.magFilter == MTLSamplerMinMagFilterNearest &&
                                 descriptor.mipFilter != MTLSamplerMipFilterLinear;
        if ((!all_linear && !all_nearest) || is_depth) {
            throw std::runtime_error("Metal guest sampler reduction requires shader lowering");
        }
        descriptor.reductionMode = sampler.filter_mode == AmdGpu::FilterMode::Min
                                       ? MTLSamplerReductionModeMinimum
                                       : MTLSamplerReductionModeMaximum;
    }
    id<MTLSamplerState> result = instance.CreateSampler(descriptor);
    samplers.emplace(key, result);
    return result;
}

} // namespace Metal
