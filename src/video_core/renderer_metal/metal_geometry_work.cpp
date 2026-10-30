// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_geometry_work.h"
#include "video_core/renderer_metal/metal_indirect_work.h"

namespace Metal {
namespace {

void ValidateWaveRange(u32 items, u32 wave_bytes, u64 capacity) {
    if (!items || items % 64 || !wave_bytes || wave_bytes % 256 ||
        capacity > std::numeric_limits<u32>::max() || u64{items / 64} * wave_bytes > capacity) {
        throw std::runtime_error("Native geometry waves exceed their linear ring capacity");
    }
}

} // namespace

u64 RasterIndexSourceSizeBytes(u32 count, u32 element_size, AmdGpu::IndexSwapMode swap_mode,
                               u32 source_offset) {
    using Swap = AmdGpu::IndexSwapMode;
    const bool paired =
        element_size == sizeof(u16) && (swap_mode == Swap::Swap32 || swap_mode == Swap::SwapWord);
    if ((element_size != 0 && element_size != sizeof(u16) && element_size != sizeof(u32)) ||
        (!element_size && (count || swap_mode != Swap::None)) ||
        (swap_mode != Swap::None && swap_mode != Swap::Swap16 && swap_mode != Swap::Swap32 &&
         swap_mode != Swap::SwapWord) ||
        (source_offset && (!paired || !count || source_offset != sizeof(u16)))) {
        throw std::runtime_error("Invalid native raster index source layout");
    }
    const u64 bytes = u64{count} * element_size;
    return paired && count ? (source_offset + bytes + 3) & ~u64{3} : bytes;
}

RasterIndexRange BuildRasterIndexRange(VAddr base, u32 first_index, u32 count, u32 element_size,
                                       AmdGpu::IndexSwapMode swap_mode) {
    if (element_size != sizeof(u16) && element_size != sizeof(u32)) {
        throw std::runtime_error("Native raster indices require a supported element size");
    }
    static_cast<void>(RasterIndexSourceSizeBytes(count, element_size, swap_mode));
    if (!count)
        return {0, 0, element_size};
    constexpr u64 address_space = u64{1} << 40;
    const u64 offset = u64{first_index} * element_size;
    const u64 bytes = u64{count} * element_size;
    if (base >= address_space || offset >= address_space - base) {
        throw std::runtime_error("Native raster index origin exceeds guest address space");
    }
    const VAddr address = base + offset;
    if (!address || address % element_size || bytes > address_space - address) {
        throw std::runtime_error("Native raster index range exceeds guest address space");
    }
    const bool paired =
        element_size == sizeof(u16) && (swap_mode == AmdGpu::IndexSwapMode::Swap32 ||
                                        swap_mode == AmdGpu::IndexSwapMode::SwapWord);
    const u32 source_offset = paired ? address & 3 : 0;
    const u64 source_bytes =
        RasterIndexSourceSizeBytes(count, element_size, swap_mode, source_offset);
    const VAddr source_address = address - source_offset;
    if (source_bytes > address_space - source_address) {
        throw std::runtime_error("Native raster index source exceeds guest address space");
    }
    return {source_address, count, element_size, source_offset,
            u32(source_bytes - source_offset - bytes)};
}

RasterDrawParameters RasterDrawLayout(u32 vertices, u32 instances, u32 step_rate_0, u32 step_rate_1,
                                      u32 index_width, u32 index_count, u32 index_offset,
                                      AmdGpu::IndexSwapMode swap_mode, u32 first_vertex,
                                      u32 first_instance) {
    using Swap = AmdGpu::IndexSwapMode;
    if (!instances || (index_width != 0 && index_width != 2 && index_width != 4) ||
        index_count > vertices || (!index_width && index_count) ||
        (swap_mode != Swap::None && swap_mode != Swap::Swap16 && swap_mode != Swap::Swap32 &&
         swap_mode != Swap::SwapWord) ||
        (!index_width && swap_mode != Swap::None) ||
        u64{first_vertex} + vertices > std::numeric_limits<u32>::max() ||
        (instances && u64{first_instance} + instances - 1 > std::numeric_limits<u32>::max())) {
        throw std::runtime_error("Invalid native raster vertex input parameters");
    }
    const u64 active = u64{vertices} * instances;
    if (active > std::numeric_limits<u32>::max() - 63) {
        throw std::runtime_error("Native raster vertex invocation count exceeds capacity");
    }
    const u64 work_items = (active + 63) & ~u64{63};
    const u64 words = work_items * (sizeof(GeometryExportInput) / sizeof(u32)) + 4;
    if (words > u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Native raster vertex inputs exceed shader addressing");
    }
    return {vertices,    instances,   u32(work_items), u32(active), step_rate_0,  step_rate_1,
            index_width, index_count, index_offset,    swap_mode,   first_vertex, first_instance};
}

u64 RasterRestartStateSizeBytes(u32 vertex_words) {
    if (!vertex_words || vertex_words > (std::numeric_limits<u32>::max() - 4) / 4) {
        throw std::runtime_error("Native raster restart state exceeds shader addressing");
    }
    return (u64{4} + u64{4} * vertex_words) * sizeof(u32);
}

void ValidateRasterRestartIndex(u32 index_width, u32 restart_index, bool neo_mode) {
    if (index_width != sizeof(u16) && index_width != sizeof(u32)) {
        throw std::runtime_error("Native restart requires a supported index width");
    }
    if (neo_mode && index_width == sizeof(u16) && restart_index > 0xffffu) {
        throw std::runtime_error("Native Neo high restart index requires its comparison rule");
    }
}

bool RasterRestartEnabled(const AmdGpu::Regs& regs, bool indexed, bool neo_mode) {
    if (!indexed || !(regs.enable_primitive_restart & 1)) {
        return false;
    }
    const u32 width = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16   ? 2
                      : regs.index_buffer_type.index_type == AmdGpu::IndexType::Index32 ? 4
                                                                                        : 0;
    ValidateRasterRestartIndex(width, regs.primitive_restart_index, neo_mode);
    return width == sizeof(u32) || regs.primitive_restart_index <= 0xffffu;
}

RasterRestartParameters BuildRasterRestartParameters(const RasterDrawParameters& draw,
                                                     u32 restart_index) {
    if (draw != RasterDrawLayout(draw.vertices, draw.instances, draw.step_rate_0, draw.step_rate_1,
                                 draw.index_width, draw.index_count, draw.index_offset,
                                 draw.swap_mode, draw.first_vertex, draw.first_instance) ||
        !draw.index_width) {
        throw std::runtime_error("Native raster restart requires matching indexed inputs");
    }
    return BuildRasterRestartParameters(draw.work_items, restart_index);
}

RasterRestartParameters BuildRasterRestartParameters(u32 work_items, u32 restart_index) {
    if (work_items % 64) {
        throw std::runtime_error("Native raster restart requires whole workgroups");
    }
    const u32 blocks = work_items / 64;
    return {work_items, blocks, u32((u64{blocks} + 63) / 64), restart_index};
}

u32 RasterDrawBatchCapacity(u32 vertex_words, u64 max_buffer_length, u32 alignment, bool restart,
                            u32 header_bytes) {
    if (!vertex_words || (alignment != 64 && alignment != 192) ||
        (header_bytes != 4 * sizeof(float) && header_bytes != sizeof(GeometryIndirectHeader))) {
        throw std::runtime_error("Invalid native raster capture capacity parameters");
    }
    const u64 budget = std::min(u64{32} << 20, max_buffer_length);
    const bool indirect = header_bytes == sizeof(GeometryIndirectHeader);
    const u64 fixed_bytes = header_bytes +
                            (indirect ? 2 * RasterCaptureStateSizeBytes(vertex_words) +
                                            sizeof(RasterCaptureSnapshotParameters)
                             : restart ? RasterRestartStateSizeBytes(vertex_words)
                                       : 0) +
                            (restart && indirect ? sizeof(RasterRestartDispatchParameters) : 0);
    if (budget <= fixed_bytes) {
        throw std::runtime_error("Native raster batch has no available storage");
    }
    const u64 unaligned = std::min(
        {(budget - fixed_bytes) / (sizeof(GeometryExportInput) + u64{vertex_words} * sizeof(u32) +
                                   sizeof(u32) + (restart ? 9 : 0)),
         (u64{std::numeric_limits<u32>::max()} + 1 - 4) /
             (sizeof(GeometryExportInput) / sizeof(u32)),
         u64{std::numeric_limits<u32>::max()} / vertex_words});
    const u64 capacity = unaligned - unaligned % alignment;
    if (!capacity) {
        throw std::runtime_error("Native raster vertex layout exceeds batch storage");
    }
    return u32(capacity);
}

RasterPrimitiveParameters BuildRasterPrimitiveParameters(AmdGpu::PrimitiveType topology,
                                                         const RasterDrawParameters& draw) {
    if (draw != RasterDrawLayout(draw.vertices, draw.instances, draw.step_rate_0, draw.step_rate_1,
                                 draw.index_width, draw.index_count, draw.index_offset,
                                 draw.swap_mode, draw.first_vertex, draw.first_instance)) {
        throw std::runtime_error("Invalid native raster primitive inputs");
    }
    using Type = AmdGpu::PrimitiveType;
    switch (topology) {
    case Type::RectList:
    case Type::QuadList: {
        const u32 width = topology == Type::RectList ? 3 : 4;
        if (draw.first_vertex % width) {
            throw std::runtime_error("Native raster list starts inside a primitive");
        }
        return {width, width, draw.vertices / width, 0};
    }
    case Type::QuadStrip: {
        const u32 carry = draw.first_vertex ? 2 : 0;
        if (draw.first_vertex % 2 || (carry && draw.instances != 1)) {
            throw std::runtime_error("Native quad strip requires an aligned instance batch");
        }
        const u64 vertices = u64{draw.vertices} + carry;
        return {4, 2, vertices >= 4 ? u32(vertices / 2 - 1) : 0, carry};
    }
    case Type::Polygon: {
        const u32 carry = draw.first_vertex ? 1 : 0;
        if (carry && draw.instances != 1) {
            throw std::runtime_error("Native polygon continuation requires one instance");
        }
        return {3, 1, carry ? draw.vertices : draw.vertices > 2 ? draw.vertices - 2 : 0, carry};
    }
    default:
        throw std::runtime_error("Unsupported native raster primitive topology");
    }
}

RasterPrimitiveIdParameters BuildRasterPrimitiveIds(AmdGpu::PrimitiveType topology,
                                                    const RasterDrawParameters& draw,
                                                    u32 vertices_per_instance, u32 reset_value,
                                                    bool reset_on_instance) {
    const auto primitive = BuildRasterPrimitiveParameters(topology, draw);
    if (topology == AmdGpu::PrimitiveType::Polygon ||
        u64{draw.first_vertex} + draw.vertices > vertices_per_instance ||
        (draw.first_vertex && draw.instances != 1) ||
        (u64{draw.first_vertex} + draw.vertices != vertices_per_instance &&
         draw.vertices % primitive.vertex_step)) {
        throw std::runtime_error("Native raster primitive IDs require complete list/strip batches");
    }
    const bool strip = topology == AmdGpu::PrimitiveType::QuadStrip;
    const u32 total = strip ? (vertices_per_instance >= 4 ? vertices_per_instance / 2 - 1 : 0)
                            : vertices_per_instance / primitive.input_vertices;
    const u32 first = strip ? (draw.first_vertex ? draw.first_vertex / 2 - 1 : 0)
                            : draw.first_vertex / primitive.input_vertices;
    const u32 step = reset_on_instance ? 0 : total;
    return {u32(u64{reset_value} + u64{draw.first_instance} * step + first), step, 1};
}

RasterPrimitiveCounterSettings RasterGuestPrimitiveIds(const AmdGpu::Regs& regs,
                                                       u32 indirect_draws) {
    static_assert(offsetof(AmdGpu::Regs, enable_primitive_id) == 0x28a84);
    constexpr u32 reset_register = 0x28a8c / sizeof(u32);
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::Vs ||
        (regs.primitive_type != AmdGpu::PrimitiveType::QuadList &&
         regs.primitive_type != AmdGpu::PrimitiveType::QuadStrip)) {
        throw std::runtime_error("Native fragment primitive IDs require quad list/strip lowering");
    }
    const u32 control = regs.enable_primitive_id;
    if (!(control & 1) || (control & ~3u)) {
        throw std::runtime_error("Native fragment primitive IDs require enabled known VGT control");
    }
    const u32 reset_on_instance = (control & 2) == 0;
    if (!reset_on_instance && indirect_draws > 1) {
        throw std::runtime_error(
            "Native multi-draw primitive IDs require their packet counter boundary");
    }
    return {regs.reg_array[reset_register], reset_on_instance};
}

void ValidateRasterPrimitiveIdVertexSources(u64 sources) {
    if (sources & (u64{1} << (u32(GeometryAddressSource::ExportVector) + 2))) {
        throw std::runtime_error("Native VS primitive-ID input requires pre-capture lowering");
    }
}

u64 RasterPrimitiveCounterSizeBytes(u32 jobs, bool indirect) {
    if (!jobs)
        throw std::runtime_error("Native raster primitive counter requires nonempty capacity");
    const u64 blocks = (u64{jobs} + 63) / 64;
    return (u64{jobs} + blocks + 2) * sizeof(RasterPrimitiveCounter) +
           (indirect
                ? sizeof(RasterIndirectPrimitiveScanParameters) + sizeof(RasterPrimitiveScanWork)
                : sizeof(RasterPrimitiveScanParameters));
}

void ValidateRasterPrimitiveCounter(const RasterPrimitiveCounterConfig& config,
                                    const RasterDrawParameters& draw,
                                    const RasterDrawParameters* previous) {
    const auto valid = [&](const RasterDrawParameters& p) {
        return p == RasterDrawLayout(p.vertices, p.instances, p.step_rate_0, p.step_rate_1,
                                     p.index_width, p.index_count, p.index_offset, p.swap_mode,
                                     p.first_vertex, p.first_instance) &&
               p.vertices && u64{p.first_vertex} + p.vertices <= config.vertices_per_instance &&
               (p.instances == 1 ||
                (!p.first_vertex && p.vertices == config.vertices_per_instance));
    };
    if (!config.vertices_per_instance || config.reset_on_instance > 1 || !valid(draw) ||
        (!previous && (draw.first_vertex || draw.first_instance)) ||
        (previous &&
         (!valid(*previous) || previous->step_rate_0 != draw.step_rate_0 ||
          previous->step_rate_1 != draw.step_rate_1 || previous->index_width != draw.index_width ||
          previous->index_offset != draw.index_offset || previous->swap_mode != draw.swap_mode ||
          (draw.first_vertex
               ? (previous->instances != 1 || draw.instances != 1 ||
                  previous->first_instance != draw.first_instance ||
                  u64{previous->first_vertex} + previous->vertices != draw.first_vertex)
               : (u64{previous->first_vertex} + previous->vertices !=
                      config.vertices_per_instance ||
                  u64{previous->first_instance} + previous->instances != draw.first_instance))))) {
        throw std::runtime_error("Native raster primitive counter requires adjacent draw batches");
    }
}

u32 RasterDrawCarryOffset(const RasterDrawParameters& previous, const RasterDrawParameters& draw,
                          u32 carry_vertices) {
    if (draw != RasterDrawLayout(draw.vertices, draw.instances, draw.step_rate_0, draw.step_rate_1,
                                 draw.index_width, draw.index_count, draw.index_offset,
                                 draw.swap_mode, draw.first_vertex, draw.first_instance) ||
        previous != RasterDrawLayout(previous.vertices, previous.instances, previous.step_rate_0,
                                     previous.step_rate_1, previous.index_width,
                                     previous.index_count, previous.index_offset,
                                     previous.swap_mode, previous.first_vertex,
                                     previous.first_instance) ||
        (carry_vertices != 1 && carry_vertices != 2) || !draw.first_vertex ||
        draw.first_vertex % carry_vertices || previous.first_vertex % carry_vertices ||
        previous.vertices < carry_vertices || previous.vertices % carry_vertices ||
        previous.instances != 1 || draw.instances != 1 ||
        u64{previous.first_vertex} + previous.vertices != draw.first_vertex ||
        previous.first_instance != draw.first_instance ||
        previous.step_rate_0 != draw.step_rate_0 || previous.step_rate_1 != draw.step_rate_1 ||
        previous.index_width != draw.index_width || previous.index_offset != draw.index_offset ||
        previous.swap_mode != draw.swap_mode) {
        throw std::runtime_error("Native raster continuation requires adjacent capture batches");
    }
    return previous.vertices - carry_vertices;
}

void ValidateRasterDrawAnchor(const RasterDrawParameters& origin,
                              const RasterDrawParameters& draw) {
    if (origin != RasterDrawLayout(origin.vertices, origin.instances, origin.step_rate_0,
                                   origin.step_rate_1, origin.index_width, origin.index_count,
                                   origin.index_offset, origin.swap_mode, origin.first_vertex,
                                   origin.first_instance) ||
        draw != RasterDrawLayout(draw.vertices, draw.instances, draw.step_rate_0, draw.step_rate_1,
                                 draw.index_width, draw.index_count, draw.index_offset,
                                 draw.swap_mode, draw.first_vertex, draw.first_instance) ||
        origin.first_vertex || origin.vertices < 3 || origin.instances != 1 ||
        draw.instances != 1 || origin.vertices > draw.first_vertex ||
        origin.first_instance != draw.first_instance || origin.step_rate_0 != draw.step_rate_0 ||
        origin.step_rate_1 != draw.step_rate_1 || origin.index_width != draw.index_width ||
        origin.index_offset != draw.index_offset || origin.swap_mode != draw.swap_mode) {
        throw std::runtime_error("Native polygon anchor requires the original instance capture");
    }
}

RasterDrawBatchPlan::RasterDrawBatchPlan(u32 vertices_, u32 instances_, u32 capacity_,
                                         u32 step_rate_0, u32 step_rate_1, u32 index_width,
                                         u32 index_count_, u32 index_offset,
                                         AmdGpu::IndexSwapMode swap_mode, u32 primitive_vertices)
    : vertices{vertices_}, instances{instances_}, capacity{capacity_}, index_count{index_count_},
      prototype{RasterDrawLayout(std::min(vertices, capacity), 1, step_rate_0, step_rate_1,
                                 index_width, std::min(index_count, capacity), index_offset,
                                 swap_mode)} {
    if ((!primitive_vertices || primitive_vertices > 4) || !instances || !capacity ||
        capacity % (primitive_vertices == 3 ? 192 : 64) || index_count > vertices ||
        u64{capacity} * (sizeof(GeometryExportInput) / sizeof(u32)) + 4 >
            u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Invalid native raster vertex batch plan");
    }
}

u64 RasterDrawBatchPlan::Count() const {
    if (!vertices) {
        return 0;
    }
    if (vertices <= capacity) {
        const u32 per_batch = capacity / vertices;
        return (u64{instances} + per_batch - 1) / per_batch;
    }
    return ((u64{vertices} + capacity - 1) / capacity) * instances;
}

RasterDrawParameters RasterDrawBatchPlan::Batch(u64 index) const {
    if (index >= Count()) {
        throw std::runtime_error("Native raster vertex batch index is out of bounds");
    }
    if (vertices <= capacity) {
        const u32 per_batch = capacity / vertices;
        const u32 first_instance = u32(index * per_batch);
        return RasterDrawLayout(vertices, std::min(per_batch, instances - first_instance),
                                prototype.step_rate_0, prototype.step_rate_1, prototype.index_width,
                                index_count, prototype.index_offset, prototype.swap_mode, 0,
                                first_instance);
    }
    const u64 per_instance = (u64{vertices} + capacity - 1) / capacity;
    const u32 first = u32(index % per_instance) * capacity;
    const u32 count = std::min(capacity, vertices - first);
    const u32 available = first < index_count ? std::min(count, index_count - first) : 0;
    return RasterDrawLayout(count, 1, prototype.step_rate_0, prototype.step_rate_1,
                            prototype.index_width, available, prototype.index_offset,
                            prototype.swap_mode, first, u32(index / per_instance));
}

std::vector<GeometryAddressInput> BuildRasterInputDomains(
    const RasterDrawParameters& parameters, const GeometryExportWave& system_registers) {
    const auto& p = parameters;
    if (p != RasterDrawLayout(p.vertices, p.instances, p.step_rate_0, p.step_rate_1, p.index_width,
                              p.index_count, p.index_offset, p.swap_mode, p.first_vertex,
                              p.first_instance) ||
        !p.work_items) {
        throw std::runtime_error("Invalid native raster vertex input domains");
    }
    using Attribute = Shader::IR::Attribute;
    std::vector<GeometryAddressInput> inputs;
    for (u32 scalar = 0; scalar < system_registers.system_sgprs.size(); ++scalar) {
        const u32 value = system_registers.system_sgprs[scalar];
        inputs.push_back({Attribute::GeometryExportScalar, scalar, {value, value}});
    }
    const u32 index_max = !p.index_width       ? p.first_vertex + p.vertices - 1
                          : !p.index_count     ? 0
                          : p.index_width == 2 ? 0xffffu
                                               : 0xffffffffu;
    const u64 maximum = u64{index_max} + p.index_offset;
    inputs.push_back({Attribute::GeometryExportVector, 0,
                      maximum > std::numeric_limits<u32>::max()
                          ? GeometryAddressInterval{0, std::numeric_limits<u32>::max()}
                          : GeometryAddressInterval{0, u32(maximum)}});
    inputs.push_back(
        {Attribute::GeometryExportVector,
         1,
         {0, p.step_rate_0 ? (p.first_instance + p.instances - 1) / p.step_rate_0 : 0}});
    inputs.push_back(
        {Attribute::GeometryExportVector,
         2,
         {0, p.step_rate_1 ? (p.first_instance + p.instances - 1) / p.step_rate_1 : 0}});
    inputs.push_back({Attribute::GeometryExportVector, 3, {0, p.first_instance + p.instances - 1}});
    inputs.push_back({Attribute::GeometryInputActive, 0, {0, 1}});
    return inputs;
}

std::vector<GeometryAddressInput> BuildIndirectRasterInputDomains(
    const GeometryExportWave& system_registers, u32 step_rate_0, u32 step_rate_1) {
    using Attribute = Shader::IR::Attribute;
    constexpr u32 maximum = std::numeric_limits<u32>::max();
    std::vector<GeometryAddressInput> inputs;
    for (u32 scalar = 0; scalar < system_registers.system_sgprs.size(); ++scalar) {
        const u32 value = system_registers.system_sgprs[scalar];
        inputs.push_back({Attribute::GeometryExportScalar, scalar, {value, value}});
    }
    inputs.push_back({Attribute::BaseVertex, 0, {0, maximum}});
    inputs.push_back({Attribute::BaseInstance, 0, {0, maximum}});
    inputs.push_back({Attribute::GeometryExportVector, 0, {0, maximum}});
    inputs.push_back(
        {Attribute::GeometryExportVector, 1, {0, step_rate_0 ? maximum / step_rate_0 : 0}});
    inputs.push_back(
        {Attribute::GeometryExportVector, 2, {0, step_rate_1 ? maximum / step_rate_1 : 0}});
    inputs.push_back({Attribute::GeometryExportVector, 3, {0, maximum}});
    inputs.push_back({Attribute::GeometryInputActive, 0, {0, 1}});
    return inputs;
}

GeometryRingLayout BuildGeometryRingLayout(const AmdGpu::Regs& regs,
                                           const GeometryDrawParameters& draw) {
    const auto& mode = regs.vgt_gs_mode;
    const auto& instances = regs.vgt_gs_instance_cnt;
    const bool tessellation = regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs;
    AmdGpu::PrimitiveType topology = regs.primitive_type;
    if (tessellation) {
        switch (regs.tess_config.topology) {
        case AmdGpu::TessellationTopology::Point:
            topology = AmdGpu::PrimitiveType::PointList;
            break;
        case AmdGpu::TessellationTopology::Line:
            topology = AmdGpu::PrimitiveType::LineList;
            break;
        case AmdGpu::TessellationTopology::TriangleCw:
        case AmdGpu::TessellationTopology::TriangleCcw:
            topology = AmdGpu::PrimitiveType::TriangleList;
            break;
        default:
            throw std::runtime_error("Invalid native tessellation geometry input topology");
        }
    }
    if ((!tessellation && regs.stage_enable.raw != AmdGpu::ShaderStageEnable::EsGs) ||
        mode.mode != AmdGpu::GsScenario::ScenarioG || mode.onchip || mode.gs_c_pack_en ||
        mode.es_passthru || mode.compute_mode || mode.fast_compute_mode || mode.element_info_en ||
        mode.suppress_cuts || regs.vgt_strmout_config.streamout_0_en ||
        regs.vgt_strmout_config.streamout_1_en || regs.vgt_strmout_config.streamout_2_en ||
        regs.vgt_strmout_config.streamout_3_en || regs.es_program.settings.scratch_en ||
        regs.gs_program.settings.scratch_en || regs.vs_program.settings.scratch_en ||
        bool(regs.es_program.settings.oc_lds_en) !=
            (tessellation && bool(regs.hs_program.settings.oc_lds_en)) ||
        regs.gs_program.settings.oc_lds_en || regs.vs_program.settings.oc_lds_en ||
        (!tessellation && (draw.input_vertices != regs.num_indices ||
                           draw.instances != regs.num_instances.NumInstances())) ||
        (tessellation && draw.instances != 1) ||
        draw != GeometryDrawLayout(topology, draw.input_vertices, draw.instances,
                                   instances.IsEnabled() ? instances.count : 1,
                                   tessellation ? 0 : regs.vgt_instance_step_rate_0,
                                   tessellation ? 0 : regs.vgt_instance_step_rate_1,
                                   tessellation ? 0 : draw.index_width,
                                   tessellation ? 0 : regs.index_offset) ||
        !regs.vgt_esgs_ring_itemsize || regs.vgt_esgs_ring_itemsize >= (1u << 15) ||
        !regs.vgt_gsvs_ring_itemsize || regs.vgt_gsvs_ring_itemsize >= (1u << 15) ||
        !regs.vgt_gs_max_vert_out || regs.vgt_gs_max_vert_out > (128u << mode.cut_mode)) {
        throw std::runtime_error("Native geometry ring state requires additional VGT lowering");
    }
    GeometryRingLayout layout{draw.export_items,
                              draw.geometry_items,
                              regs.vgt_esgs_ring_itemsize * 256,
                              regs.vgt_gsvs_ring_itemsize * 256,
                              u64{regs.vgt_esgs_ring_size} * 256,
                              u64{regs.vgt_gsvs_ring_size} * 256,
                              {0, regs.vgt_gsvs_ring_offset_1.GetOffset(),
                               regs.vgt_gsvs_ring_offset_2.GetOffset(),
                               regs.vgt_gsvs_ring_offset_3.GetOffset()},
                              {regs.vgt_gs_vert_itemsize[0], regs.vgt_gs_vert_itemsize[1],
                               regs.vgt_gs_vert_itemsize[2], regs.vgt_gs_vert_itemsize[3]},
                              regs.vgt_gs_max_vert_out};
    ValidateWaveRange(layout.export_items, layout.export_wave_bytes, layout.export_capacity);
    ValidateWaveRange(layout.geometry_items, layout.geometry_wave_bytes, layout.geometry_capacity);
    for (u32 stream = 0; stream < layout.stream_components.size(); ++stream) {
        const u64 words = u64{layout.stream_components[stream]} * layout.max_vertices;
        if (words >= (1u << 12) || (words && u64{layout.stream_word_offsets[stream]} + words >
                                                 regs.vgt_gsvs_ring_itemsize)) {
            throw std::runtime_error("Native geometry stream exceeds its GSVS ring item");
        }
    }
    return layout;
}

GeometryWaveInputs BuildGeometryWaveInputs(const GeometryRingLayout& layout,
                                           const GeometryExportWave& export_template,
                                           u32 first_geometry_wave_id) {
    ValidateWaveRange(layout.export_items, layout.export_wave_bytes, layout.export_capacity);
    ValidateWaveRange(layout.geometry_items, layout.geometry_wave_bytes, layout.geometry_capacity);
    const u32 export_waves = layout.export_items / 64;
    const u32 geometry_waves = layout.geometry_items / 64;
    if (geometry_waves - 1 > std::numeric_limits<u32>::max() - first_geometry_wave_id) {
        throw std::runtime_error("Native geometry wave IDs exceed capacity");
    }
    GeometryWaveInputs inputs;
    inputs.exports.resize(export_waves, export_template);
    inputs.geometry.resize(geometry_waves);
    for (u32 wave = 0; wave < export_waves; ++wave) {
        inputs.exports[wave].system_sgprs[0] = wave * layout.export_wave_bytes;
    }
    for (u32 wave = 0; wave < geometry_waves; ++wave) {
        inputs.geometry[wave] = {wave * layout.geometry_wave_bytes, first_geometry_wave_id + wave};
    }
    return inputs;
}

} // namespace Metal
