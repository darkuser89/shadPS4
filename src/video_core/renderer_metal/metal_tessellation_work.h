// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>
#include "video_core/renderer_metal/metal_tessellation_kernel.h"

namespace AmdGpu {
union Regs;
enum class IndexSwapMode : u32;
} // namespace AmdGpu

namespace Metal {

struct TessellationPatchFactors {
    std::array<u32, 4> outer;
    std::array<u32, 2> inner;
    u32 instance;
    u32 patch;
};
static_assert(sizeof(TessellationPatchFactors) == 32);

struct TessellationPatchGroup {
    u32 instance;
    u32 first_patch;
    u32 patch_count;
    u32 local_invocations;
    u32 hull_invocations;
};

class TessellationWorkLayout {
public:
    TessellationWorkLayout(const AmdGpu::Regs& regs,
                           const Shader::TessellationDataConstantBuffer& constants);

    [[nodiscard]] const TessellationKernelConfig& KernelConfig() const {
        return config;
    }
    [[nodiscard]] TessellationPatchGroup Group(u32 index) const;
    [[nodiscard]] TessellationWorkLayout Slice(u32 first, u32 count) const;
    [[nodiscard]] u64 PatchCount(u32 first, u32 count) const;
    [[nodiscard]] u32 FirstGroup() const {
        return first_group;
    }
    bool operator==(const TessellationWorkLayout&) const = default;

private:
    TessellationKernelConfig config{};
    u32 patches_per_instance{};
    u32 groups_per_instance{};
    u32 patches_per_group{};
    u32 input_control_points{};
    u32 hull_threads_per_patch{};
    u32 first_group{};
};

struct TessellationGroupOffsets {
    u32 factor_buffer;
    std::optional<u32> offchip_lds;
};

struct TessellationOffchipLayout {
    u32 block_bytes;
    u32 group_count;

    [[nodiscard]] u64 SizeBytes() const {
        return u64{block_bytes} * group_count;
    }
};

[[nodiscard]] TessellationOffchipLayout BuildTessellationOffchipLayout(const AmdGpu::Regs& regs,
                                                                       bool neo_mode);

struct TessellationRegisterInputs {
    std::vector<GeometryExportInput> local;
    std::vector<GeometryExportInput> hull;
};

struct TessellationIndexRange {
    u64 address;
    u32 count;
    u32 element_size;
    u32 source_offset{};
    u32 source_padding{};

    [[nodiscard]] u64 SizeBytes() const {
        return source_offset + u64{count} * element_size + source_padding;
    }
};

[[nodiscard]] TessellationIndexRange BuildTessellationIndexRange(const AmdGpu::Regs& regs,
                                                                 u32 first_index,
                                                                 u64 max_buffer_bytes,
                                                                 bool neo_mode = false);

void DecodeTessellationIndices(std::span<u32> indices, const TessellationIndexRange& range,
                               AmdGpu::IndexSwapMode swap_mode);

struct TessellationEvaluationPoint {
    u32 u_bits;
    u32 v_bits;
    u32 relative_patch;
};

struct TessellationEvaluationWave {
    std::array<u32, 16> system_sgprs;
    std::span<const TessellationEvaluationPoint> points;
};

struct TessellationExportRingLayout {
    u32 wave_bytes;
    u64 size;
};

void ValidateTessellationEvaluationState(const AmdGpu::Regs& regs, bool raster);

[[nodiscard]] std::vector<std::array<u32, 16>> BuildTessellationEvaluationRegisters(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::span<const u32> offchip_offsets, bool neo_mode);

[[nodiscard]] TessellationExportRingLayout BuildTessellationExportRingLayout(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout);

[[nodiscard]] std::vector<TessellationEvaluationWave> BuildTessellationExportWaves(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::span<const std::span<const TessellationEvaluationPoint>> points,
    std::span<const u32> offchip_offsets, bool neo_mode);

[[nodiscard]] std::vector<GeometryExportInput> BuildTessellationEvaluationInputs(
    const TessellationWorkLayout& layout, std::span<const TessellationEvaluationWave> waves);

[[nodiscard]] std::vector<TessellationGroupOffsets> BuildTessellationGroupOffsets(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout, bool neo_mode);

[[nodiscard]] std::vector<TessellationWorkLayout> SplitTessellationWork(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout, u64 max_buffer_bytes,
    bool neo_mode);

[[nodiscard]] TessellationRegisterInputs BuildTessellationRegisterInputs(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::optional<std::span<const u32>> decoded_indices,
    std::span<const TessellationGroupOffsets> group_offsets, bool neo_mode = false);

} // namespace Metal
