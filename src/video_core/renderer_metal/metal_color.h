// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include "common/types.h"

namespace Metal {

constexpr float PresentationNitsPerUnit = 100.f;

enum class ScreenshotTransfer {
    Encoded,
    Linear,
    Pq,
};

inline float PqComponentNits(float encoded) {
    constexpr double m1 = 2610.0 / 16384.0;
    constexpr double m2 = 2523.0 / 32.0;
    constexpr double c1 = 3424.0 / 4096.0;
    constexpr double c2 = 2413.0 / 128.0;
    constexpr double c3 = 2392.0 / 128.0;
    const double power = std::pow(std::clamp(double{encoded}, 0.0, 1.0), 1.0 / m2);
    return float(10000.0 * std::pow(std::max(power - c1, 0.0) / (c2 - c3 * power), 1.0 / m1));
}

inline std::array<float, 3> LinearBt2020ToSrgb(const std::array<float, 3>& rgb) {
    return {1.660491002f * rgb[0] - 0.587641139f * rgb[1] - 0.072849863f * rgb[2],
            -0.124550475f * rgb[0] + 1.132899897f * rgb[1] - 0.008349422f * rgb[2],
            -0.018150763f * rgb[0] - 0.100578898f * rgb[1] + 1.118729661f * rgb[2]};
}

inline u8 ScreenshotComponent(float value, bool linear) {
    value = std::isnan(value) ? 0.f : std::clamp(value, 0.f, 1.f);
    if (linear) {
        value =
            value <= 0.0031308f ? 12.92f * value : 1.055f * std::pow(value, 1.f / 2.4f) - 0.055f;
    }
    return u8(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
}

} // namespace Metal
