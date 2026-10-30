// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "video_core/renderer_metal/metal_geometry_work.h"

namespace Metal {

struct IndirectReadParameters {
    u32 max_count;
    u32 stride_words;
    u32 primitive;
    u32 index_capacity;
    u64 argument_address;
    u64 count_address;
    u32 paged;
    u32 has_count;
};
static_assert(sizeof(IndirectReadParameters) == 40);

struct RasterIndirectParameters {
    IndirectReadParameters source;
    u32 first_draw;
    u32 record_count;
    u32 capture_capacity;
    u32 index_width;
};
static_assert(sizeof(RasterIndirectParameters) == 56);

struct RasterIndirectRecord {
    u32 draw_id;
    u32 active;
    std::array<u32, 5> arguments;
    u32 index_width;
    u32 index_prefix_count;
    u32 error;
    u64 work_items;
    u64 capture_batches;
    u64 index_byte_offset;
};
static_assert(sizeof(RasterIndirectRecord) == 64);

struct RasterIndirectWindowLayout {
    RasterIndirectParameters parameters;
    u64 argument_bytes;
    u64 record_bytes;
};

struct RasterCaptureParameters {
    RasterIndirectParameters window;
    u32 primitive_vertices;
    u32 step_rate_0;
    u32 step_rate_1;
    u32 index_offset;
    AmdGpu::IndexSwapMode swap_mode;
    u32 reserved{};
};
static_assert(sizeof(RasterCaptureParameters) == 80);

struct RasterCaptureCursor {
    u32 record_index;
    u32 completed;
    u64 batch_index;
};
static_assert(sizeof(RasterCaptureCursor) == 16);

struct RasterCaptureBatch {
    RasterDrawParameters draw;
    u64 index_byte_offset;
    u64 batch_index;
    u32 draw_id;
    u32 record_index;
    u32 first_capture;
    u32 last_capture;
    u32 base_vertex;
    u32 first_instance;
    u32 valid;
    u32 error;
    std::array<u32, 3> dispatch;
    u32 reserved;
};
static_assert(sizeof(RasterCaptureBatch) == 112);

struct RasterCaptureInputParameters {
    RasterCaptureParameters capture;
    u64 index_address;
    std::array<float, 4> viewport;
    GeometryExportWave system_registers;
    u32 restart_enabled;
    u32 restart_index;
    u32 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
};
static_assert(sizeof(RasterCaptureInputParameters) == 184);

struct RasterCaptureSnapshotParameters {
    RasterCaptureInputParameters inputs;
    u32 vertex_words;
    u32 source_words;
    u32 has_previous;
    u32 same_context;
};
static_assert(sizeof(RasterCaptureSnapshotParameters) == 200);

struct RasterCaptureSnapshotHeader {
    RasterCaptureBatch batch;
    u32 length;
    u32 valid;
    u32 error;
    u32 reserved;
};
static_assert(sizeof(RasterCaptureSnapshotHeader) == 128);

[[nodiscard]] u64 RasterCaptureStateSizeBytes(u32 vertex_words);

struct RasterCapturePackConfig {
    RasterCaptureSnapshotParameters snapshot;
    RasterPackParameters pack;
    u32 reserved{};
};
static_assert(sizeof(RasterCapturePackConfig) == 336);

struct RasterCapturePackWork {
    RasterPackParameters pack;
    std::array<u32, 3> dispatch;
    std::array<u32, 3> mesh_dispatch;
    u32 flags;
};
static_assert(sizeof(RasterCapturePackWork) == 160);

[[nodiscard]] GeometryAssemblyParameters RasterCaptureMeshLayout(u32 capacity, u32 vertex_words,
                                                                 u32 mesh_vertices,
                                                                 u32 primitive_vertices);
[[nodiscard]] u32 RasterCaptureMeshCapacity(u32 vertex_words, u32 mesh_vertices,
                                            u32 primitive_vertices, u32 index_width, bool restart,
                                            u64 max_buffer_length, bool viewport_copy = false,
                                            bool primitive_ids = false);
[[nodiscard]] u64 RasterCaptureMeshSizeBytes(u32 capacity, u32 vertex_words, u32 mesh_vertices,
                                             u32 primitive_vertices, bool restart,
                                             bool viewport_copy = false,
                                             bool primitive_ids = false);
[[nodiscard]] u64 ViewportPackingFixedSizeBytes();
[[nodiscard]] u64 ViewportPackingMinimumSizeBytes(u32 vertex_words, u32 mesh_vertices,
                                                  u32 topology);

struct RasterIndirectSubmissionLayout {
    u32 records_per_window;
    u32 captures_per_submission;
    u64 capture_bytes;
    u64 window_bytes;
};

[[nodiscard]] u32 RasterCaptureSubmissionCapacity(u32 vertex_words, u32 mesh_vertices,
                                                  u32 primitive_vertices, u32 index_width,
                                                  bool restart, u64 max_buffer_length,
                                                  bool viewport_copy, bool primitive_ids = false);
[[nodiscard]] u64 RasterCaptureSubmissionSizeBytes(u32 capacity, u32 vertex_words,
                                                   u32 mesh_vertices, u32 primitive_vertices,
                                                   bool restart, bool viewport_copy,
                                                   bool primitive_ids = false);
[[nodiscard]] RasterIndirectSubmissionLayout BuildRasterIndirectSubmissionLayout(
    u32 max_count, u64 capture_bytes, u64 max_buffer_length);

struct RasterCaptureInputLayout {
    RasterCaptureInputParameters parameters;
    u64 input_bytes;
    u64 restart_bytes;
    RasterRestartParameters restart_capacity;
};

[[nodiscard]] RasterIndirectWindowLayout BuildRasterIndirectWindow(u32 max_count, u32 first_draw,
                                                                   u32 record_count, u32 stride,
                                                                   u32 index_width,
                                                                   u32 index_capacity,
                                                                   u32 capture_capacity);

[[nodiscard]] RasterCaptureParameters BuildRasterCaptureParameters(
    const RasterIndirectParameters& window, u32 primitive_vertices, u32 step_rate_0,
    u32 step_rate_1, u32 index_offset, AmdGpu::IndexSwapMode swap_mode);

[[nodiscard]] RasterCaptureInputLayout BuildRasterCaptureInputLayout(
    const RasterCaptureParameters& capture, VAddr index_address,
    const std::array<float, 4>& viewport, const GeometryExportWave& system_registers,
    std::optional<u32> restart_index, u64 max_buffer_length,
    const RasterIndirectRegisters& indirect_registers = {});

} // namespace Metal
