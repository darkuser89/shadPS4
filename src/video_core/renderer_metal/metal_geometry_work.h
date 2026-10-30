// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include "video_core/amdgpu/regs_vertex.h"
#include "video_core/renderer_metal/metal_geometry_export.h"
#include "video_core/renderer_metal/metal_geometry_layout.h"
#include "video_core/renderer_metal/metal_geometry_shader.h"

namespace AmdGpu {
union Regs;
}

namespace Metal {

struct GeometryExportWave {
    std::array<u32, 16> system_sgprs;
};
static_assert(sizeof(GeometryExportWave) == 64);

struct GeometryKernelWave {
    u32 ring_offset;
    u32 wave_id;
};
static_assert(sizeof(GeometryKernelWave) == 8);

struct RasterDrawParameters {
    u32 vertices;
    u32 instances;
    u32 work_items;
    u32 active_work_items;
    u32 step_rate_0;
    u32 step_rate_1;
    u32 index_width;
    u32 index_count;
    u32 index_offset;
    AmdGpu::IndexSwapMode swap_mode;
    u32 first_vertex;
    u32 first_instance;

    bool operator==(const RasterDrawParameters&) const = default;
};
static_assert(sizeof(RasterDrawParameters) == 48);

struct RasterIndexRange {
    VAddr address;
    u32 count;
    u32 element_size;
    u32 source_offset{};
    u32 source_padding{};

    [[nodiscard]] u64 SizeBytes() const {
        return source_offset + u64{count} * element_size + source_padding;
    }
    [[nodiscard]] VAddr LogicalAddress() const {
        return count ? address + source_offset : 0;
    }
};

[[nodiscard]] RasterIndexRange BuildRasterIndexRange(
    VAddr base, u32 first_index, u32 count, u32 element_size,
    AmdGpu::IndexSwapMode swap_mode = AmdGpu::IndexSwapMode::None);
[[nodiscard]] u64 RasterIndexSourceSizeBytes(u32 count, u32 element_size,
                                             AmdGpu::IndexSwapMode swap_mode,
                                             u32 source_offset = 0);

struct RasterRestartParameters {
    u32 work_items;
    u32 blocks;
    u32 superblocks;
    u32 restart_index;

    [[nodiscard]] u64 RecordSizeBytes() const {
        return u64{work_items} * 2 * sizeof(u32);
    }
    [[nodiscard]] u64 BlockSizeBytes() const {
        return u64{blocks} * sizeof(u32);
    }
    [[nodiscard]] u64 SuperblockSizeBytes() const {
        return u64{superblocks} * sizeof(u32);
    }
    [[nodiscard]] u64 SizeBytes() const {
        return RecordSizeBytes() + BlockSizeBytes() + SuperblockSizeBytes();
    }
};
static_assert(sizeof(RasterRestartParameters) == 16);

struct RasterRestartDispatchParameters {
    RasterRestartParameters scan;
    std::array<std::array<u32, 3>, 4> dispatch;
};
static_assert(sizeof(RasterRestartDispatchParameters) == 64);

[[nodiscard]] u64 RasterRestartStateSizeBytes(u32 vertex_words);
[[nodiscard]] u32 RasterDrawBatchCapacity(u32 vertex_words, u64 max_buffer_length, u32 alignment,
                                          bool restart, u32 header_bytes = 4 * sizeof(float));

void ValidateRasterRestartIndex(u32 index_width, u32 restart_index, bool neo_mode);
[[nodiscard]] bool RasterRestartEnabled(const AmdGpu::Regs& regs, bool indexed, bool neo_mode);
[[nodiscard]] RasterRestartParameters BuildRasterRestartParameters(const RasterDrawParameters& draw,
                                                                   u32 restart_index);
[[nodiscard]] RasterRestartParameters BuildRasterRestartParameters(u32 work_items,
                                                                   u32 restart_index);

struct RasterPrimitiveParameters {
    u32 input_vertices;
    u32 vertex_step;
    u32 primitives_per_instance;
    u32 carry_vertices;
};
static_assert(sizeof(RasterPrimitiveParameters) == 16);

struct RasterPrimitiveIdParameters {
    u32 first;
    u32 instance_step;
    u32 enabled;
    u32 reserved{};
};
static_assert(sizeof(RasterPrimitiveIdParameters) == 16);

struct RasterPrimitiveCounterConfig {
    u32 vertices_per_instance;
    u32 reset_value;
    u32 reset_on_instance;

    bool operator==(const RasterPrimitiveCounterConfig&) const = default;
};
static_assert(sizeof(RasterPrimitiveCounterConfig) == 12);

struct RasterPrimitiveCounterSettings {
    u32 reset_value;
    u32 reset_on_instance;

    bool operator==(const RasterPrimitiveCounterSettings&) const = default;
};
static_assert(sizeof(RasterPrimitiveCounterSettings) == 8);

[[nodiscard]] RasterPrimitiveCounterSettings RasterGuestPrimitiveIds(const AmdGpu::Regs& regs,
                                                                     u32 indirect_draws = 0);
void ValidateRasterPrimitiveIdVertexSources(u64 sources);

struct RasterPrimitiveCounter {
    u32 total;
    u32 suffix;
    u32 instance;
    u32 first_instance;
    u32 valid;
    u32 error;
};
static_assert(sizeof(RasterPrimitiveCounter) == 24);

struct RasterPrimitiveScanParameters {
    GeometryAssemblyParameters mesh;
    u32 reset_value;
    u32 reset_on_instance;
    u32 has_previous;
    u32 reserved{};
};
static_assert(sizeof(RasterPrimitiveScanParameters) == 96);

struct RasterPrimitiveScanWork {
    RasterPrimitiveScanParameters scan;
    std::array<u32, 3> blocks_dispatch;
    std::array<u32, 3> assign_dispatch;
    RasterPrimitiveIdParameters ids;
    RasterPrimitiveCounter previous;
};
static_assert(sizeof(RasterPrimitiveScanWork) == 160);
static_assert(offsetof(RasterPrimitiveScanWork, blocks_dispatch) == 96);
static_assert(offsetof(RasterPrimitiveScanWork, assign_dispatch) == 108);
static_assert(offsetof(RasterPrimitiveScanWork, ids) == 120);
static_assert(offsetof(RasterPrimitiveScanWork, previous) == 136);

struct RasterIndirectPrimitiveScanParameters {
    RasterPrimitiveScanParameters scan;
    u32 capacity;
    u32 source_words;
};
static_assert(sizeof(RasterIndirectPrimitiveScanParameters) == 104);

[[nodiscard]] u64 RasterPrimitiveCounterSizeBytes(u32 jobs, bool indirect);

void ValidateRasterPrimitiveCounter(const RasterPrimitiveCounterConfig& config,
                                    const RasterDrawParameters& draw,
                                    const RasterDrawParameters* previous);

[[nodiscard]] RasterPrimitiveIdParameters BuildRasterPrimitiveIds(AmdGpu::PrimitiveType topology,
                                                                  const RasterDrawParameters& draw,
                                                                  u32 vertices_per_instance,
                                                                  u32 reset_value,
                                                                  bool reset_on_instance);

struct RasterPackParameters {
    GeometryAssemblyParameters mesh;
    u32 first;
    u32 count;
    u32 vertices_per_instance;
    u32 quads_per_instance;
    u32 source_words;
    u32 viewport_word;
    u32 viewport_count;
    u32 last_provoking;
    u32 input_vertices;
    std::array<u32, 2> position_words;
    u32 vertex_step;
    u32 carry_vertices;
};
static_assert(sizeof(RasterPackParameters) == 132);

struct ViewportPackingParameters {
    GeometryAssemblyParameters input;
    GeometryAssemblyParameters output;
    u64 first;
    u32 count;
    u32 slots;
    u32 viewport_word;
    u32 viewport_count;
    u32 last_provoking;
    u32 padding{};
};
static_assert(sizeof(ViewportPackingParameters) == 192);

[[nodiscard]] RasterPrimitiveParameters BuildRasterPrimitiveParameters(
    AmdGpu::PrimitiveType topology, const RasterDrawParameters& draw);
[[nodiscard]] u32 RasterDrawCarryOffset(const RasterDrawParameters& previous,
                                        const RasterDrawParameters& draw, u32 carry_vertices = 2);
void ValidateRasterDrawAnchor(const RasterDrawParameters& origin, const RasterDrawParameters& draw);

[[nodiscard]] RasterDrawParameters RasterDrawLayout(
    u32 vertices, u32 instances, u32 step_rate_0, u32 step_rate_1, u32 index_width, u32 index_count,
    u32 index_offset = 0, AmdGpu::IndexSwapMode swap_mode = AmdGpu::IndexSwapMode::None,
    u32 first_vertex = 0, u32 first_instance = 0);

class RasterDrawBatchPlan {
public:
    RasterDrawBatchPlan(u32 vertices, u32 instances, u32 capacity, u32 step_rate_0, u32 step_rate_1,
                        u32 index_width, u32 index_count, u32 index_offset,
                        AmdGpu::IndexSwapMode swap_mode, u32 primitive_vertices = 4);
    [[nodiscard]] u64 Count() const;
    [[nodiscard]] RasterDrawParameters Batch(u64 index) const;

private:
    u32 vertices;
    u32 instances;
    u32 capacity;
    u32 index_count;
    RasterDrawParameters prototype;
};

[[nodiscard]] std::vector<GeometryAddressInput> BuildRasterInputDomains(
    const RasterDrawParameters& parameters, const GeometryExportWave& system_registers);

[[nodiscard]] std::vector<GeometryAddressInput> BuildIndirectRasterInputDomains(
    const GeometryExportWave& system_registers, u32 step_rate_0, u32 step_rate_1);

struct GeometryDrawParameters {
    u32 input_vertices;
    u32 instances;
    u32 gs_invocations;
    u32 primitives;
    u32 export_items;
    u32 geometry_items;
    u32 export_vertices;
    u32 geometry_invocations;
    u32 step_rate_0;
    u32 step_rate_1;
    u32 topology;
    u32 index_width;
    u32 index_offset;

    bool operator==(const GeometryDrawParameters&) const = default;
};
static_assert(sizeof(GeometryDrawParameters) == 52);

struct GeometryRingLayout {
    u32 export_items;
    u32 geometry_items;
    u32 export_wave_bytes;
    u32 geometry_wave_bytes;
    u64 export_capacity;
    u64 geometry_capacity;
    std::array<u32, 4> stream_word_offsets;
    std::array<u32, 4> stream_components;
    u32 max_vertices;

    bool operator==(const GeometryRingLayout&) const = default;
};

struct GeometryWaveInputs {
    std::vector<GeometryExportWave> exports;
    std::vector<GeometryKernelWave> geometry;
};

[[nodiscard]] GeometryRingLayout BuildGeometryRingLayout(const AmdGpu::Regs& regs,
                                                         const GeometryDrawParameters& draw);
[[nodiscard]] GeometryWaveInputs BuildGeometryWaveInputs(const GeometryRingLayout& layout,
                                                         const GeometryExportWave& export_template,
                                                         u32 first_geometry_wave_id);

[[nodiscard]] inline GeometryDrawParameters GeometryDrawLayout(AmdGpu::PrimitiveType topology,
                                                               u32 count, u32 instances,
                                                               u32 gs_invocations, u32 step_rate_0,
                                                               u32 step_rate_1, u32 index_width,
                                                               u32 index_offset = 0) {
    if (!instances || !gs_invocations || gs_invocations > 63 ||
        (index_width != 0 && index_width != 2 && index_width != 4)) {
        throw std::runtime_error("Invalid native geometry draw parameters");
    }
    u32 primitives{};
    using Type = AmdGpu::PrimitiveType;
    switch (topology) {
    case Type::PointList:
        primitives = count;
        break;
    case Type::LineList:
        primitives = count / 2;
        break;
    case Type::LineStrip:
        primitives = count ? count - 1 : 0;
        break;
    case Type::TriangleList:
        primitives = count / 3;
        break;
    case Type::TriangleStrip:
    case Type::TriangleFan:
        primitives = count > 2 ? count - 2 : 0;
        break;
    case Type::AdjLineList:
        primitives = count / 4;
        break;
    case Type::AdjLineStrip:
        primitives = count > 3 ? count - 3 : 0;
        break;
    case Type::AdjTriangleList:
        primitives = count / 6;
        break;
    case Type::AdjTriangleStrip:
        primitives = count > 4 ? (count - 4) / 2 : 0;
        break;
    default:
        throw std::runtime_error("Native geometry draw topology requires additional lowering");
    }
    const u64 exports = primitives ? u64{count} * instances : 0;
    if (exports > std::numeric_limits<u32>::max() - 63) {
        throw std::runtime_error("Native geometry export count exceeds capacity");
    }
    const u64 invocations = u64{primitives} * instances * gs_invocations;
    if (invocations > std::numeric_limits<u32>::max() - 63) {
        throw std::runtime_error("Native geometry invocation count exceeds capacity");
    }
    const u64 export_items = (exports + 63) & ~u64{63};
    const u64 geometry_items = (invocations + 63) & ~u64{63};
    if (export_items * (sizeof(GeometryExportInput) / sizeof(u32)) >
            u64{std::numeric_limits<u32>::max()} + 1 ||
        geometry_items * (sizeof(GeometryKernelWorkItem) / sizeof(u32)) >
            u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Native geometry input addresses exceed capacity");
    }
    return {count,        instances,         gs_invocations,
            primitives,   u32(export_items), u32(geometry_items),
            u32(exports), u32(invocations),  step_rate_0,
            step_rate_1,  u32(topology),     index_width,
            index_offset};
}

} // namespace Metal
