// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_geometry_mesh.h"
#include "video_core/renderer_metal/metal_indirect_work.h"

namespace Metal {

u64 RasterCaptureStateSizeBytes(u32 vertex_words) {
    if (vertex_words < 5 || vertex_words > (std::numeric_limits<u32>::max() -
                                            sizeof(RasterCaptureSnapshotHeader) / sizeof(u32)) /
                                               4) {
        throw std::runtime_error("Native raster capture state exceeds shader addressing");
    }
    return sizeof(RasterCaptureSnapshotHeader) + u64{4} * vertex_words * sizeof(u32);
}

GeometryAssemblyParameters RasterCaptureMeshLayout(u32 capacity, u32 vertex_words,
                                                   u32 mesh_vertices, u32 primitive_vertices) {
    const u32 output_vertices = primitive_vertices == 1 ? 3 : 6;
    if (!capacity || capacity % (primitive_vertices == 3 ? 192 : 64) || vertex_words < 5 ||
        !primitive_vertices || primitive_vertices > 4 || mesh_vertices < output_vertices ||
        mesh_vertices > 256) {
        throw std::runtime_error("Invalid native raster capture mesh capacity");
    }
    const u32 per_job = mesh_vertices / output_vertices;
    const u32 jobs = u32((u64{capacity} + per_job - 1) / per_job);
    return GeometryAssemblyLayout(jobs, mesh_vertices, vertex_words, mesh_vertices, {2, 0, 0, 0},
                                  1);
}

u64 RasterCaptureMeshSizeBytes(u32 capacity, u32 vertex_words, u32 mesh_vertices,
                               u32 primitive_vertices, bool restart, bool viewport_copy,
                               bool primitive_ids) {
    if (primitive_ids && primitive_vertices == 1)
        throw std::runtime_error("Native polygon primitive IDs require their guest counter rule");
    const auto mesh =
        RasterCaptureMeshLayout(capacity, vertex_words, mesh_vertices, primitive_vertices);
    const auto scan = BuildRasterRestartParameters(capacity, 0);
    return sizeof(GeometryIndirectHeader) +
           u64{capacity} * (sizeof(GeometryExportInput) + u64{vertex_words} * sizeof(u32)) +
           2 * RasterCaptureStateSizeBytes(vertex_words) + sizeof(RasterCaptureSnapshotParameters) +
           sizeof(RasterCapturePackConfig) + sizeof(RasterCapturePackWork) +
           u64{vertex_words} * sizeof(u32) + sizeof(RasterPrimitiveIdParameters) +
           (primitive_ids ? RasterPrimitiveCounterSizeBytes(mesh.jobs, true) : 0) +
           (restart ? scan.SizeBytes() + sizeof(RasterRestartDispatchParameters) : 0) +
           mesh.EmissionSizeBytes() + mesh.PrimitiveSizeBytes() +
           (viewport_copy ? mesh.EmissionSizeBytes() + mesh.PrimitiveSizeBytes() +
                                ViewportPackingFixedSizeBytes()
                          : 0);
}

u64 ViewportPackingFixedSizeBytes() {
    return 2 * sizeof(ViewportPackingParameters) + sizeof(RasterCapturePackWork) +
           sizeof(GeometryMeshViewports);
}

u64 ViewportPackingMinimumSizeBytes(u32 vertex_words, u32 mesh_vertices, u32 topology) {
    if (vertex_words < 5 || topology > 2 || mesh_vertices <= topology)
        throw std::runtime_error("Invalid native viewport packing capacity");
    const auto layout = GeometryAssemblyLayout(1, mesh_vertices, vertex_words, mesh_vertices,
                                               {topology, 0, 0, 0}, 1);
    return layout.EmissionSizeBytes() + layout.PrimitiveSizeBytes() +
           ViewportPackingFixedSizeBytes();
}

u32 RasterCaptureMeshCapacity(u32 vertex_words, u32 mesh_vertices, u32 primitive_vertices,
                              u32 index_width, bool restart, u64 max_buffer_length,
                              bool viewport_copy, bool primitive_ids) {
    if ((index_width != 0 && index_width != 2 && index_width != 4) || (restart && !index_width)) {
        throw std::runtime_error("Invalid native raster capture mesh index mode");
    }
    const u32 alignment = primitive_vertices == 3 ? 192 : 64;
    static_cast<void>(
        RasterCaptureMeshLayout(alignment, vertex_words, mesh_vertices, primitive_vertices));
    const u64 budget = std::min(u64{32} << 20, max_buffer_length);
    const u32 upper = RasterDrawBatchCapacity(vertex_words, max_buffer_length, alignment, restart,
                                              sizeof(GeometryIndirectHeader));
    const auto fits = [&](u32 capacity) {
        return RasterCaptureMeshSizeBytes(capacity, vertex_words, mesh_vertices, primitive_vertices,
                                          restart, viewport_copy, primitive_ids) <= budget;
    };
    u32 low = 0, high = upper / alignment;
    while (low < high) {
        const u32 middle = low + (high - low + 1) / 2;
        if (fits(middle * alignment))
            low = middle;
        else
            high = middle - 1;
    }
    if (!low)
        throw std::runtime_error("Native raster capture mesh exceeds aggregate storage");
    return low * alignment;
}

u32 RasterCaptureSubmissionCapacity(u32 vertex_words, u32 mesh_vertices, u32 primitive_vertices,
                                    u32 index_width, bool restart, u64 max_buffer_length,
                                    bool viewport_copy, bool primitive_ids) {
    const u64 budget = std::min(u64{32} << 20, max_buffer_length);
    const u64 window_bytes = std::min(u64{1} << 20, budget / 8) + sizeof(RasterIndirectParameters) +
                             sizeof(RasterCaptureCursor) + 5 * sizeof(u32);
    const u32 alignment = primitive_vertices == 3 ? 192 : 64;
    const u64 minimum =
        RasterCaptureSubmissionSizeBytes(alignment, vertex_words, mesh_vertices, primitive_vertices,
                                         restart, viewport_copy, primitive_ids);
    if (window_bytes >= budget || minimum > budget - window_bytes)
        throw std::runtime_error("Native raster capture submission cannot hold its minimum work");
    const u64 target = std::min(budget - window_bytes, std::max(u64{4} << 20, minimum));
    constexpr u64 overhead = sizeof(RasterCaptureBatch) + sizeof(RasterCaptureParameters) +
                             sizeof(RasterCaptureInputParameters);
    return RasterCaptureMeshCapacity(vertex_words, mesh_vertices, primitive_vertices, index_width,
                                     restart, target - overhead, viewport_copy, primitive_ids);
}

u64 RasterCaptureSubmissionSizeBytes(u32 capacity, u32 vertex_words, u32 mesh_vertices,
                                     u32 primitive_vertices, bool restart, bool viewport_copy,
                                     bool primitive_ids) {
    return RasterCaptureMeshSizeBytes(capacity, vertex_words, mesh_vertices, primitive_vertices,
                                      restart, viewport_copy, primitive_ids) +
           sizeof(RasterCaptureBatch) + sizeof(RasterCaptureParameters) +
           sizeof(RasterCaptureInputParameters);
}

RasterIndirectSubmissionLayout BuildRasterIndirectSubmissionLayout(u32 max_count, u64 capture_bytes,
                                                                   u64 max_buffer_length) {
    const u64 budget = std::min(u64{32} << 20, max_buffer_length);
    const u32 records = u32(std::min(u64{max_count}, std::min(u64{1} << 20, budget / 8) /
                                                         sizeof(RasterIndirectRecord)));
    const u64 window_bytes = u64{records} * sizeof(RasterIndirectRecord) +
                             sizeof(RasterIndirectParameters) + sizeof(RasterCaptureCursor) +
                             5 * sizeof(u32);
    if (!records || !capture_bytes || window_bytes >= budget ||
        capture_bytes > budget - window_bytes)
        throw std::runtime_error("Native raster indirect submission exceeds aggregate storage");
    return {records, u32(std::min(u64{8}, (budget - window_bytes) / capture_bytes)), capture_bytes,
            window_bytes};
}

RasterIndirectWindowLayout BuildRasterIndirectWindow(u32 max_count, u32 first_draw,
                                                     u32 record_count, u32 stride, u32 index_width,
                                                     u32 index_capacity, u32 capture_capacity) {
    const u32 argument_size = index_width ? 20 : 16;
    if (u64{first_draw} + record_count > max_count || stride < argument_size || stride % 4 ||
        (index_width != 0 && index_width != 2 && index_width != 4) ||
        (!index_width && index_capacity) || !capture_capacity || capture_capacity % 64 ||
        u64{capture_capacity} * sizeof(GeometryExportInput) + 4 * sizeof(float) > (u64{32} << 20)) {
        throw std::runtime_error("Invalid native raster indirect argument window");
    }
    const RasterIndirectParameters parameters{
        {max_count, stride / 4, 0, index_capacity, 0, 0, 0, 0},
        first_draw,
        record_count,
        capture_capacity,
        index_width};
    return {parameters, record_count ? u64{record_count - 1} * stride + argument_size : 0,
            u64{record_count} * sizeof(RasterIndirectRecord)};
}

RasterCaptureParameters BuildRasterCaptureParameters(const RasterIndirectParameters& window,
                                                     u32 primitive_vertices, u32 step_rate_0,
                                                     u32 step_rate_1, u32 index_offset,
                                                     AmdGpu::IndexSwapMode swap_mode) {
    using Swap = AmdGpu::IndexSwapMode;
    if (window.source.stride_words > std::numeric_limits<u32>::max() / 4 ||
        window.source.primitive || window.source.paged > 1 ||
        (window.source.has_count > 2 || (window.source.has_count == 2 && !window.source.paged)) ||
        !primitive_vertices || primitive_vertices > 4 ||
        window.capture_capacity % (primitive_vertices == 3 ? 192 : 64) ||
        (swap_mode != Swap::None && swap_mode != Swap::Swap16 && swap_mode != Swap::Swap32 &&
         swap_mode != Swap::SwapWord) ||
        (!window.index_width && swap_mode != Swap::None)) {
        throw std::runtime_error("Invalid native raster capture controller parameters");
    }
    static_cast<void>(BuildRasterIndirectWindow(window.source.max_count, window.first_draw,
                                                window.record_count, window.source.stride_words * 4,
                                                window.index_width, window.source.index_capacity,
                                                window.capture_capacity));
    return {window, primitive_vertices, step_rate_0, step_rate_1, index_offset, swap_mode};
}

RasterCaptureInputLayout BuildRasterCaptureInputLayout(
    const RasterCaptureParameters& capture, VAddr index_address,
    const std::array<float, 4>& viewport, const GeometryExportWave& system_registers,
    std::optional<u32> restart_index, u64 max_buffer_length,
    const RasterIndirectRegisters& indirect_registers) {
    indirect_registers.Validate();
    const auto normalized = BuildRasterCaptureParameters(capture.window, capture.primitive_vertices,
                                                         capture.step_rate_0, capture.step_rate_1,
                                                         capture.index_offset, capture.swap_mode);
    const auto& window = normalized.window;
    const auto indices = window.index_width
                             ? BuildRasterIndexRange(index_address, 0, window.source.index_capacity,
                                                     window.index_width, capture.swap_mode)
                             : RasterIndexRange{};
    const u64 input_bytes =
        u64{window.capture_capacity} * sizeof(GeometryExportInput) + sizeof(GeometryIndirectHeader);
    const u64 restart_bytes = restart_index ? u64{window.capture_capacity} * 2 * sizeof(u32) : 0;
    const auto restart_capacity =
        restart_index ? BuildRasterRestartParameters(window.capture_capacity, *restart_index)
                      : RasterRestartParameters{};
    const u64 scan_bytes = restart_index ? restart_capacity.BlockSizeBytes() +
                                               restart_capacity.SuperblockSizeBytes() +
                                               sizeof(RasterRestartDispatchParameters)
                                         : 0;
    if (capture.reserved || (!window.index_width && (index_address || restart_index)) ||
        (restart_index && window.index_width == 2 && *restart_index > 0xffffu) ||
        input_bytes + restart_bytes + scan_bytes > std::min(u64{32} << 20, max_buffer_length)) {
        throw std::runtime_error("Invalid native raster capture input layout");
    }
    return {{normalized, indices.LogicalAddress(), viewport, system_registers,
             u32(restart_index.has_value()), restart_index.value_or(0),
             indirect_registers.vertex_sgpr_offset, indirect_registers.instance_sgpr_offset},
            input_bytes,
            restart_bytes,
            restart_capacity};
}

} // namespace Metal
