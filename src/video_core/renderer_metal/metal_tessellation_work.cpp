// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_geometry_work.h"
#include "video_core/renderer_metal/metal_tessellation_work.h"

namespace Metal {
namespace {

struct RingParameters {
    u64 factor_bytes;
    u32 components;
    u32 offchip_block_bytes;
    u32 max_groups;
};

bool SupportedIndexSwap(AmdGpu::IndexSwapMode mode) {
    return mode == AmdGpu::IndexSwapMode::None || mode == AmdGpu::IndexSwapMode::Swap16 ||
           mode == AmdGpu::IndexSwapMode::Swap32 || mode == AmdGpu::IndexSwapMode::SwapWord;
}

RingParameters ReadRingParameters(const AmdGpu::Regs& regs, bool neo_mode) {
    if (u32(regs.tess_config.type) > u32(AmdGpu::TessellationType::Quad)) {
        throw std::runtime_error("Metal tessellation ring allocation requires a valid domain");
    }
    constexpr u32 FactorRingSizeRegister = 0x30938 / sizeof(u32);
    const auto layout = BuildTessellationOffchipLayout(regs, neo_mode);
    const bool offchip = regs.hs_program.settings.oc_lds_en;
    return {u64{regs.reg_array[FactorRingSizeRegister] & 0xffff} * sizeof(u32),
            2 + 2 * u32(regs.tess_config.type), offchip ? layout.block_bytes : 0,
            offchip ? layout.group_count : std::numeric_limits<u32>::max()};
}

} // namespace

TessellationOffchipLayout BuildTessellationOffchipLayout(const AmdGpu::Regs& regs, bool neo_mode) {
    constexpr u32 OffchipParameterRegister = 0x3093c / sizeof(u32);
    const u32 parameters = regs.reg_array[OffchipParameterRegister];
    return {u32((8192 * sizeof(u32)) >> ((parameters >> 9) & 3)),
            (parameters & 0x1ff) + u32(neo_mode)};
}

TessellationIndexRange BuildTessellationIndexRange(const AmdGpu::Regs& regs, u32 first_index,
                                                   u64 max_buffer_bytes, bool neo_mode) {
    const auto type = regs.index_buffer_type.index_type;
    const u32 element_size = type == AmdGpu::IndexType::Index16 ? sizeof(u16) : sizeof(u32);
    if ((type != AmdGpu::IndexType::Index16 && type != AmdGpu::IndexType::Index32) ||
        !SupportedIndexSwap(regs.index_buffer_type.swap_mode) ||
        RasterRestartEnabled(regs, true, neo_mode)) {
        throw std::runtime_error("Metal tessellation index state requires additional lowering");
    }
    if (u64{regs.num_indices} * sizeof(u32) > max_buffer_bytes) {
        throw std::runtime_error("Metal tessellation decoded indices exceed native storage limits");
    }
    const u32 count = first_index < regs.max_index_size
                          ? std::min(regs.num_indices, regs.max_index_size - first_index)
                          : 0;
    if (!count) {
        return {0, 0, element_size};
    }
    const u64 address = regs.index_base_address.Address() + u64{first_index} * element_size;
    const u64 bytes = u64{count} * element_size;
    constexpr u64 AddressSpaceSize = u64{1} << 40;
    if (address >= AddressSpaceSize || address % element_size ||
        bytes > AddressSpaceSize - address || bytes > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Metal tessellation index range exceeds native storage limits");
    }
    const bool paired = element_size == sizeof(u16) &&
                        (regs.index_buffer_type.swap_mode == AmdGpu::IndexSwapMode::Swap32 ||
                         regs.index_buffer_type.swap_mode == AmdGpu::IndexSwapMode::SwapWord);
    const u32 offset = paired ? address & 3 : 0;
    const u32 padding = paired ? (4 - ((offset + bytes) & 3)) & 3 : 0;
    const TessellationIndexRange result{address - offset, count, element_size, offset, padding};
    if (result.SizeBytes() > AddressSpaceSize - result.address ||
        result.SizeBytes() > std::numeric_limits<u32>::max() ||
        result.SizeBytes() > u64{regs.num_indices} * sizeof(u32)) {
        throw std::runtime_error("Metal tessellation swapped indices exceed source storage");
    }
    return result;
}

void DecodeTessellationIndices(std::span<u32> indices, const TessellationIndexRange& range,
                               AmdGpu::IndexSwapMode swap_mode) {
    if (range.count > indices.size() ||
        (range.element_size != sizeof(u16) && range.element_size != sizeof(u32)) ||
        !SupportedIndexSwap(swap_mode) || range.SizeBytes() > indices.size_bytes()) {
        throw std::runtime_error("Invalid Metal tessellation index decoding contract");
    }
    const bool paired =
        range.element_size == sizeof(u16) && (swap_mode == AmdGpu::IndexSwapMode::Swap32 ||
                                              swap_mode == AmdGpu::IndexSwapMode::SwapWord);
    if (paired && range.count) {
        const u64 end = range.source_offset + u64{range.count} * sizeof(u16);
        if (range.address % sizeof(u32) || range.source_offset > sizeof(u16) ||
            range.source_offset % sizeof(u16) || range.source_padding != ((4 - (end & 3)) & 3)) {
            throw std::runtime_error("Invalid Metal tessellation paired index source");
        }
    } else if (range.source_offset || range.source_padding) {
        throw std::runtime_error("Invalid Metal tessellation index source padding");
    }
    if (range.element_size == sizeof(u16)) {
        const auto* data = reinterpret_cast<const u8*>(indices.data());
        for (size_t index = range.count; index-- > 0;) {
            u16 value;
            const u64 offset = range.source_offset + index * sizeof(u16);
            std::memcpy(&value, data + (paired ? offset ^ 2 : offset), sizeof(value));
            indices[index] = swap_mode == AmdGpu::IndexSwapMode::Swap16 ||
                                     swap_mode == AmdGpu::IndexSwapMode::Swap32
                                 ? std::byteswap(value)
                                 : value;
        }
    } else if (swap_mode != AmdGpu::IndexSwapMode::None) {
        for (auto& value : indices.first(range.count)) {
            if (swap_mode == AmdGpu::IndexSwapMode::Swap32) {
                value = std::byteswap(value);
            } else if (swap_mode == AmdGpu::IndexSwapMode::SwapWord) {
                value = std::rotl(value, 16);
            } else {
                value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
            }
        }
    }
    std::ranges::fill(indices.subspan(range.count), 0u);
}

TessellationWorkLayout::TessellationWorkLayout(
    const AmdGpu::Regs& regs, const Shader::TessellationDataConstantBuffer& constants)
    : patches_per_group{regs.ls_hs_config.num_patches},
      input_control_points{regs.ls_hs_config.hs_input_control_points},
      hull_threads_per_patch{regs.ls_hs_config.hs_output_control_points} {
    if ((regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHs &&
         regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs) ||
        regs.primitive_type != AmdGpu::PrimitiveType::PatchPrimitive || !patches_per_group ||
        constants.num_patches != patches_per_group || !input_control_points ||
        input_control_points > 32 || !hull_threads_per_patch || hull_threads_per_patch > 32) {
        throw std::runtime_error("Invalid Metal tessellation patch configuration");
    }
    const u32 invocations =
        patches_per_group * std::max(input_control_points, hull_threads_per_patch);
    if (invocations > 256) {
        throw std::runtime_error("Metal tessellation patch group exceeds the guest thread limit");
    }
    if (regs.num_indices % input_control_points) {
        throw std::runtime_error("Metal tessellation draw has an incomplete input patch");
    }
    constexpr u32 LsResources2Register = 0xb52c / sizeof(u32);
    const u32 lds_units = (regs.reg_array[LsResources2Register] >> 7) & 0x1ff;
    if (!lds_units || lds_units > 128) {
        throw std::runtime_error("Metal tessellation LDS allocation is outside the guest range");
    }
    const u32 lds_bytes = lds_units * 128 * sizeof(u32);
    const u64 input_end = u64{patches_per_group} * input_control_points * constants.ls_stride;
    const u64 output_end =
        u64{constants.hs_output_base} + u64{patches_per_group} * constants.patch_output_size;
    const u64 patch_end =
        u64{constants.patch_const_base} + u64{patches_per_group} * constants.patch_const_size;
    if (std::max({input_end, output_end, patch_end}) > lds_bytes) {
        throw std::runtime_error("Metal tessellation constants exceed the allocated guest LDS");
    }
    patches_per_instance = regs.num_indices / input_control_points;
    groups_per_instance =
        u32((u64{patches_per_instance} + patches_per_group - 1) / patches_per_group);
    const u64 groups = u64{groups_per_instance} * regs.num_instances.NumInstances();
    if (groups > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Metal tessellation workgroup count exceeds capacity");
    }
    config = {u32(groups), (invocations + 63) & ~63u, lds_bytes,
              std::bit_cast<std::array<u32, 9>>(constants)};
}

TessellationPatchGroup TessellationWorkLayout::Group(u32 index) const {
    if (index >= config.workgroups) {
        throw std::runtime_error("Metal tessellation patch group is outside the draw");
    }
    index += first_group;
    const u32 first_patch = (index % groups_per_instance) * patches_per_group;
    const u32 count = std::min(patches_per_group, patches_per_instance - first_patch);
    return {index / groups_per_instance, first_patch, count, count * input_control_points,
            count * hull_threads_per_patch};
}

TessellationWorkLayout TessellationWorkLayout::Slice(u32 first, u32 count) const {
    if (first > config.workgroups || count > config.workgroups - first) {
        throw std::runtime_error("Metal tessellation work slice is outside its parent");
    }
    auto result = *this;
    result.first_group += first;
    result.config.workgroups = count;
    return result;
}

u64 TessellationWorkLayout::PatchCount(u32 first, u32 count) const {
    if (first > config.workgroups || count > config.workgroups - first) {
        throw std::runtime_error("Metal tessellation patch count is outside its work slice");
    }
    if (!count) {
        return 0;
    }
    const auto prefix = [&](u64 group) {
        return group / groups_per_instance * patches_per_instance +
               std::min<u64>(group % groups_per_instance * patches_per_group, patches_per_instance);
    };
    const u64 begin = u64{first_group} + first;
    return prefix(begin + count) - prefix(begin);
}

std::vector<TessellationWorkLayout> SplitTessellationWork(const AmdGpu::Regs& regs,
                                                          const TessellationWorkLayout& layout,
                                                          u64 max_buffer_bytes, bool neo_mode) {
    const auto& config = layout.KernelConfig();
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    if (layout !=
        TessellationWorkLayout{regs, constants}.Slice(layout.FirstGroup(), config.workgroups)) {
        throw std::runtime_error("Metal tessellation batch plan does not match the draw");
    }
    if (!config.workgroups) {
        return {};
    }
    const auto ring = ReadRingParameters(regs, neo_mode);
    const u64 input_bytes = u64{config.threads_per_group} * sizeof(GeometryExportInput);
    const u64 offchip_limit = ring.offchip_block_bytes ? max_buffer_bytes / ring.offchip_block_bytes
                                                       : std::numeric_limits<u32>::max();
    const u64 max_groups =
        std::min({u64{ring.max_groups}, offchip_limit, max_buffer_bytes / input_bytes,
                  max_buffer_bytes / config.lds_bytes,
                  u64{std::numeric_limits<u32>::max()} * sizeof(u32) / input_bytes,
                  u64{std::numeric_limits<u32>::max()} / config.lds_bytes});
    if (!max_groups) {
        throw std::runtime_error("Metal tessellation has no available batch storage");
    }
    std::vector<TessellationWorkLayout> batches;
    u32 first{};
    while (first < config.workgroups) {
        u32 count{};
        u32 limit = u32(std::min<u64>(max_groups, config.workgroups - first));
        while (count < limit) {
            const u32 candidate = count + u32((u64{limit} - count + 1) / 2);
            const u64 patches = layout.PatchCount(first, candidate);
            const u64 bytes = (patches * ring.components + candidate) * sizeof(u32);
            if (bytes <= std::min(ring.factor_bytes, max_buffer_bytes) &&
                patches <= max_buffer_bytes / sizeof(TessellationPatchFactors)) {
                count = candidate;
            } else {
                limit = candidate - 1;
            }
        }
        if (!count) {
            throw std::runtime_error("Metal tessellation patch group exceeds the factor ring");
        }
        batches.push_back(layout.Slice(first, count));
        first += count;
    }
    return batches;
}

std::vector<TessellationGroupOffsets> BuildTessellationGroupOffsets(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout, bool neo_mode) {
    const auto& config = layout.KernelConfig();
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    if (layout !=
        TessellationWorkLayout{regs, constants}.Slice(layout.FirstGroup(), config.workgroups)) {
        throw std::runtime_error("Metal tessellation ring allocation does not match the draw");
    }
    if (!config.workgroups) {
        return {};
    }
    const auto ring = ReadRingParameters(regs, neo_mode);
    if (config.workgroups > ring.max_groups) {
        throw std::runtime_error("Metal tessellation offchip ring requires batched dispatch");
    }
    const u64 minimum_group_bytes = (u64{ring.components} + 1) * sizeof(u32);
    if (u64{config.workgroups} * minimum_group_bytes > ring.factor_bytes) {
        throw std::runtime_error("Metal tessellation factor ring requires batched dispatch");
    }
    std::vector<TessellationGroupOffsets> offsets;
    offsets.reserve(config.workgroups);
    u64 cursor{};
    for (u32 index = 0; index < config.workgroups; ++index) {
        const auto group = layout.Group(index);
        const u64 size = (u64{group.patch_count} * ring.components + 1) * sizeof(u32);
        if (size > ring.factor_bytes - cursor) {
            throw std::runtime_error("Metal tessellation factor ring requires batched dispatch");
        }
        offsets.push_back({u32(cursor), ring.offchip_block_bytes
                                            ? std::optional<u32>{index * ring.offchip_block_bytes}
                                            : std::nullopt});
        cursor += size;
    }
    return offsets;
}

TessellationRegisterInputs BuildTessellationRegisterInputs(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::optional<std::span<const u32>> decoded_indices,
    std::span<const TessellationGroupOffsets> group_offsets, bool neo_mode) {
    const auto& config = layout.KernelConfig();
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    if (layout !=
            TessellationWorkLayout{regs, constants}.Slice(layout.FirstGroup(), config.workgroups) ||
        group_offsets.size() != config.workgroups ||
        (decoded_indices && decoded_indices->size() != regs.num_indices) ||
        RasterRestartEnabled(regs, decoded_indices.has_value(), neo_mode)) {
        throw std::runtime_error("Metal tessellation register inputs do not match the draw");
    }
    constexpr u32 LsResources2Register = 0xb52c / sizeof(u32);
    constexpr u32 HsResources2Register = 0xb42c / sizeof(u32);
    constexpr u32 UserSgprMask = 0x1f << 1;
    constexpr u32 LdsSizeMask = 0x1ff << 7;
    constexpr u32 OffchipLdsEnable = 1 << 7;
    const u32 ls_resources = regs.reg_array[LsResources2Register];
    const u32 hs_resources = regs.reg_array[HsResources2Register];
    const bool offchip = hs_resources & OffchipLdsEnable;
    if ((ls_resources & ~(UserSgprMask | LdsSizeMask)) ||
        (hs_resources & ~(UserSgprMask | OffchipLdsEnable)) ||
        regs.ls_program.settings.num_user_regs > Shader::NUM_USER_DATA_REGS ||
        regs.hs_program.settings.num_user_regs > Shader::NUM_USER_DATA_REGS) {
        throw std::runtime_error("Metal tessellation system register state requires lowering");
    }
    for (const auto& offsets : group_offsets) {
        if (offsets.offchip_lds.has_value() != offchip) {
            throw std::runtime_error("Metal tessellation offchip input does not match HS state");
        }
    }
    TessellationRegisterInputs result;
    const size_t count = size_t(config.workgroups) * config.threads_per_group;
    result.local.resize(count);
    result.hull.resize(count);
    const u32 input_points = regs.ls_hs_config.hs_input_control_points;
    const u32 hull_threads = regs.ls_hs_config.hs_output_control_points;
    for (u32 group_index = 0; group_index < config.workgroups; ++group_index) {
        const auto group = layout.Group(group_index);
        const auto& offsets = group_offsets[group_index];
        for (u32 lane = 0; lane < config.threads_per_group; ++lane) {
            const size_t index = size_t(group_index) * config.threads_per_group + lane;
            auto& local = result.local[index];
            auto& hull = result.hull[index];
            local.active = lane < group.local_invocations;
            hull.active = lane < group.hull_invocations;
            if (local.active) {
                const u32 input_index = group.first_patch * input_points + lane;
                const u32 vertex_index =
                    decoded_indices ? (*decoded_indices)[input_index] : input_index;
                local.vgprs = {vertex_index + regs.index_offset, lane, group.instance, 0};
            }
            u32 scalar = 0;
            if (offchip) {
                hull.system_sgprs[scalar++] = *offsets.offchip_lds;
            }
            hull.system_sgprs[scalar] = offsets.factor_buffer;
            if (hull.active) {
                const u32 relative_patch = lane / hull_threads;
                const u32 control_point = lane % hull_threads;
                hull.vgprs = {group.first_patch + relative_patch,
                              relative_patch | (control_point << 8), 0, 0};
            }
        }
    }
    return result;
}

void ValidateTessellationEvaluationState(const AmdGpu::Regs& regs, bool raster) {
    const auto& program = raster ? regs.vs_program : regs.es_program;
    u32 resources2;
    std::memcpy(&resources2, reinterpret_cast<const u8*>(&program.settings) + sizeof(u32),
                sizeof(resources2));
    const u32 expected =
        (program.settings.num_user_regs << 1) | (regs.hs_program.settings.oc_lds_en ? 1u << 7 : 0u);
    if (regs.stage_enable.raw !=
            (raster ? AmdGpu::ShaderStageEnable::LsHs : AmdGpu::ShaderStageEnable::LsHsEsGs) ||
        !program.Address<VAddr>() || program.settings.num_user_regs > Shader::NUM_USER_DATA_REGS ||
        resources2 != expected) {
        throw std::runtime_error("Metal TES system-register state requires native lowering");
    }
}

TessellationExportRingLayout BuildTessellationExportRingLayout(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout) {
    ValidateTessellationEvaluationState(regs, false);
    const auto& config = layout.KernelConfig();
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    const auto& mode = regs.vgt_gs_mode;
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs ||
        layout !=
            TessellationWorkLayout{regs, constants}.Slice(layout.FirstGroup(), config.workgroups) ||
        mode.mode != AmdGpu::GsScenario::ScenarioG || mode.onchip || mode.gs_c_pack_en ||
        mode.es_passthru || mode.compute_mode || mode.fast_compute_mode || mode.element_info_en ||
        mode.suppress_cuts || !regs.vgt_esgs_ring_itemsize ||
        regs.vgt_esgs_ring_itemsize >= (1u << 15)) {
        throw std::runtime_error("Metal tessellation export ring state requires lowering");
    }
    const u32 wave_bytes = regs.vgt_esgs_ring_itemsize * 256;
    const u64 size = u64{regs.vgt_esgs_ring_size} * 256;
    if (!size || size > std::numeric_limits<u32>::max() ||
        u64{config.workgroups} * wave_bytes > size) {
        throw std::runtime_error("Metal tessellation export waves exceed their ring capacity");
    }
    return {wave_bytes, size};
}

std::vector<std::array<u32, 16>> BuildTessellationEvaluationRegisters(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::span<const u32> offchip_offsets, bool neo_mode) {
    const bool raster = regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHs;
    ValidateTessellationEvaluationState(regs, raster);
    const auto& config = layout.KernelConfig();
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    const bool offchip = regs.hs_program.settings.oc_lds_en;
    if (layout !=
            TessellationWorkLayout{regs, constants}.Slice(layout.FirstGroup(), config.workgroups) ||
        (offchip ? offchip_offsets.size() != config.workgroups : !offchip_offsets.empty())) {
        throw std::runtime_error("Metal TES register plan differs from its Hull groups");
    }
    const auto ring =
        raster ? TessellationExportRingLayout{} : BuildTessellationExportRingLayout(regs, layout);
    const auto parameters = ReadRingParameters(regs, neo_mode);
    std::vector<std::array<u32, 16>> inputs(config.workgroups);
    for (u32 group = 0; group < config.workgroups; ++group) {
        auto& input = inputs[group];
        if (offchip) {
            const u32 offset = offchip_offsets[group];
            if (offset % parameters.offchip_block_bytes ||
                u64{offset} + parameters.offchip_block_bytes >
                    u64{parameters.max_groups} * parameters.offchip_block_bytes) {
                throw std::runtime_error("Metal TES offchip offset exceeds its ring");
            }
            input[raster ? 1 : 0] = offset;
        }
        if (!raster) {
            input[offchip ? 2 : 0] = group * ring.wave_bytes;
        }
    }
    return inputs;
}

std::vector<TessellationEvaluationWave> BuildTessellationExportWaves(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::span<const std::span<const TessellationEvaluationPoint>> points,
    std::span<const u32> offchip_offsets, bool neo_mode) {
    const auto registers =
        BuildTessellationEvaluationRegisters(regs, layout, offchip_offsets, neo_mode);
    if (points.size() != layout.KernelConfig().workgroups) {
        throw std::runtime_error("Metal tessellation export points do not match the patch groups");
    }
    std::vector<TessellationEvaluationWave> waves;
    waves.reserve(points.size());
    for (u32 index = 0; index < points.size(); ++index) {
        const auto group = layout.Group(index);
        if (points[index].size() > 64 || std::ranges::any_of(points[index], [&](const auto& point) {
                return point.relative_patch >= group.patch_count;
            })) {
            throw std::runtime_error("Metal tessellation export point exceeds its patch group");
        }
        waves.push_back({registers[index], points[index]});
    }
    return waves;
}

std::vector<GeometryExportInput> BuildTessellationEvaluationInputs(
    const TessellationWorkLayout& layout, std::span<const TessellationEvaluationWave> waves) {
    const auto config = layout.KernelConfig().Evaluation();
    if (waves.size() != config.workgroups || config.threads_per_group != 64 ||
        config.InputSizeBytes() / sizeof(u32) > std::numeric_limits<u32>::max() ||
        config.LdsSizeBytes() > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Metal tessellation evaluation inputs exceed kernel bounds");
    }
    for (u32 group_index = 0; group_index < config.workgroups; ++group_index) {
        const auto group = layout.Group(group_index);
        const auto& wave = waves[group_index];
        if (wave.points.size() > config.threads_per_group ||
            std::ranges::any_of(wave.points, [&](const auto& point) {
                return point.relative_patch >= group.patch_count;
            })) {
            throw std::runtime_error("Metal tessellation evaluation point exceeds its patch group");
        }
    }
    std::vector<GeometryExportInput> inputs(size_t(config.workgroups) * config.threads_per_group);
    for (u32 group_index = 0; group_index < config.workgroups; ++group_index) {
        const auto group = layout.Group(group_index);
        const auto& wave = waves[group_index];
        for (u32 lane = 0; lane < config.threads_per_group; ++lane) {
            auto& input = inputs[size_t(group_index) * config.threads_per_group + lane];
            input.system_sgprs = wave.system_sgprs;
            input.active = lane < wave.points.size();
            if (input.active) {
                const auto& point = wave.points[lane];
                input.vgprs = {point.u_bits, point.v_bits, point.relative_patch,
                               group.first_patch + point.relative_patch};
            }
        }
    }
    return inputs;
}

} // namespace Metal
