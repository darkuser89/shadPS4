// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include "shader_recompiler/frontend/decode.h"

namespace Metal {

inline VAddr FetchShaderAddress(std::span<const u32> user_data, u32 sgpr_base) {
    if (sgpr_base >= user_data.size() || user_data.size() - sgpr_base < 2) {
        throw std::runtime_error("Invalid Metal fetch shader pointer registers");
    }
    VAddr address;
    std::memcpy(&address, user_data.data() + sgpr_base, sizeof(address));
    if (!address || address % sizeof(u32)) {
        throw std::runtime_error("Invalid Metal fetch shader pointer");
    }
    return address;
}

template <typename PrepareMemoryRead>
void PrepareFetchShaderRead(VAddr address, u64 offset, u64 size,
                            PrepareMemoryRead&& prepare_memory_read) {
    if (offset > std::numeric_limits<VAddr>::max() - address ||
        size > std::numeric_limits<VAddr>::max() - address - offset) {
        throw std::runtime_error("Invalid Metal fetch shader range");
    }
    prepare_memory_read(address + offset, size);
}

template <typename PrepareMemoryRead>
std::vector<u32> ReadFetchShader(std::span<const u32> user_data, u32 sgpr_base,
                                 PrepareMemoryRead&& prepare_memory_read) {
    using namespace Shader::Gcn;
    const VAddr address = FetchShaderAddress(user_data, sgpr_base);
    constexpr u64 CpuPageSize = 4096;
    u64 prepared_end{};
    const auto read = [&](u64 offset, std::span<u32> words) {
        if (offset + words.size_bytes() > prepared_end) {
            const u64 remaining = CpuPageSize - (address + offset) % CpuPageSize;
            const u64 size = std::max<u64>(remaining, words.size_bytes());
            PrepareFetchShaderRead(address, offset, size, prepare_memory_read);
            prepared_end = offset + size;
        }
        std::memcpy(words.data(), reinterpret_cast<const void*>(address + offset),
                    words.size_bytes());
    };
    std::vector<u32> code;
    GcnDecodeContext decoder;
    while (true) {
        const u64 offset = code.size() * sizeof(u32);
        alignas(u64) std::array<u32, 2> words{};
        read(offset, std::span{words}.first(1));
        const auto encoding = GetInstructionEncoding(words[0]);
        if (encoding == InstEncoding::ILLEGAL) {
            throw std::runtime_error("Invalid Metal fetch shader encoding");
        }
        const auto operand = static_cast<OperandField>(words[0] & 0xff);
        if (encoding == InstEncoding::SOP1 &&
            OpcodeSOP1((words[0] >> 8) & 0xff) == OpcodeSOP1::S_SETPC_B64 &&
            operand != OperandField::LiteralConst && operand != OperandField::Sdwa &&
            operand != OperandField::Dpp) {
            code.push_back(words[0]);
            return code;
        }
        read(offset, words);
        GcnCodeSlice slice{words.data(), words.data() + words.size(), true};
        const auto instruction = decoder.decodeInstruction(slice);
        if (!instruction.length || instruction.length > sizeof(words) ||
            instruction.length % sizeof(u32)) {
            throw std::runtime_error("Invalid Metal fetch shader instruction length");
        }
        code.insert(code.end(), words.begin(), words.begin() + instruction.length / sizeof(u32));
        if (instruction.opcode == Opcode::S_SETPC_B64) {
            return code;
        }
        if (instruction.opcode == Opcode::S_ENDPGM) {
            throw std::runtime_error("Metal fetch shader ended without returning");
        }
    }
}

template <typename PrepareMemoryRead>
bool FetchShaderMatches(std::span<const u32> user_data, u32 sgpr_base,
                        std::span<const u32> expected, PrepareMemoryRead&& prepare_memory_read) {
    const VAddr address = FetchShaderAddress(user_data, sgpr_base);
    constexpr u64 CpuPageSize = 4096;
    for (u64 offset = 0; offset < expected.size_bytes();) {
        const u64 remaining = CpuPageSize - (address + offset) % CpuPageSize;
        const u64 size = std::min<u64>(remaining, expected.size_bytes() - offset);
        PrepareFetchShaderRead(address, offset, size, prepare_memory_read);
        if (std::memcmp(reinterpret_cast<const u8*>(expected.data()) + offset,
                        reinterpret_cast<const void*>(address + offset), size) != 0) {
            return false;
        }
        offset += size;
    }
    return true;
}

} // namespace Metal
