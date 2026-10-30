// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cmath>
#include <unordered_map>
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

class Instance;

[[nodiscard]] inline bool HasSamplerAnisotropy(const AmdGpu::Sampler& sampler) noexcept {
    return sampler.max_aniso.Value() != AmdGpu::AnisoRatio::One &&
           (AmdGpu::IsAnisoFilter(sampler.xy_min_filter) ||
            AmdGpu::IsAnisoFilter(sampler.xy_mag_filter));
}

[[nodiscard]] inline bool NativeSamplerLodBiasCompatible(float bias) noexcept {
    return bias >= -16.f && bias <= 15.984375f && std::trunc(bias * 64.f) == bias * 64.f;
}

class SamplerCache {
public:
    explicit SamplerCache(const Instance& instance);
    [[nodiscard]] id<MTLSamplerState> ObtainSampler(const AmdGpu::Sampler& sampler, bool is_depth,
                                                    bool native_bias = false);

private:
    struct Key {
        u64 raw0;
        u64 raw1;
        bool is_depth;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept;
    };

    const Instance& instance;
    std::unordered_map<Key, id<MTLSamplerState>, KeyHash> samplers;
};

} // namespace Metal
