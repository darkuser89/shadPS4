// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include "video_core/amdgpu/regs_shader.h"

namespace Metal {

template <typename PrepareMemoryRead>
Shader::ShaderParams ReadShaderParams(
    VAddr address, std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data,
    PrepareMemoryRead&& prepare_memory_read) {
    const auto prepare = [&](u64 offset, u64 size) {
        if (!address || offset > std::numeric_limits<VAddr>::max() - address ||
            size > std::numeric_limits<VAddr>::max() - address - offset) {
            throw std::runtime_error("Invalid Metal guest shader address");
        }
        prepare_memory_read(address + offset, size);
    };
    const auto read_info = [&](u64 offset) {
        prepare(offset, sizeof(AmdGpu::BinaryInfo));
        AmdGpu::BinaryInfo info;
        std::memcpy(&info, reinterpret_cast<const void*>(address + offset), sizeof(info));
        return info;
    };
    std::array<u32, 2> header;
    prepare(0, sizeof(header));
    std::memcpy(header.data(), reinterpret_cast<const void*>(address), sizeof(header));
    AmdGpu::BinaryInfo info{};
    if (header[0] == 0xBEEB03FF) {
        info = read_info(u64{(header[1] + 1) * 2} * sizeof(u32));
    }
    if (!info.Valid()) {
        constexpr u64 SearchSize = 0x4000 * sizeof(u32);
        constexpr u64 SignatureSize = AmdGpu::BinaryInfo::signature_ref.size();
        constexpr u64 CpuPageSize = 4096;
        u64 prepared_end{};
        for (u64 offset = 0; offset < SearchSize; offset += sizeof(u32)) {
            if (offset + SignatureSize > prepared_end) {
                const u64 remaining = CpuPageSize - (address + offset) % CpuPageSize;
                const u64 size = std::min(std::max(remaining, SignatureSize),
                                          SearchSize + SignatureSize - sizeof(u32) - offset);
                prepare(offset, size);
                prepared_end = offset + size;
            }
            if (std::memcmp(reinterpret_cast<const void*>(address + offset),
                            AmdGpu::BinaryInfo::signature_ref.data(), SignatureSize) != 0) {
                continue;
            }
            info = read_info(offset);
            if (info.Valid()) {
                break;
            }
        }
    }
    const u64 code_words = info.length / sizeof(u32);
    if (!info.Valid() || !code_words) {
        throw std::runtime_error("Invalid Metal guest shader binary metadata");
    }
    prepare(0, code_words * sizeof(u32));
    return {user_data, {reinterpret_cast<const u32*>(address), code_words}, info.shader_hash};
}

} // namespace Metal
