// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include "common/types.h"

namespace Metal {

struct GeometryAssemblyParameters {
    u32 invocations;
    u32 max_vertices;
    u32 vertex_words;
    u32 mesh_vertices;
    std::array<u32, 4> topology;
    std::array<u32, 4> mesh_counts;
    std::array<u32, 4> mesh_offsets;
    u32 stream_stride_words;
    u32 mesh_stride_words;
    u32 meshes_per_invocation;
    u32 jobs;

    bool operator==(const GeometryAssemblyParameters&) const = default;

    [[nodiscard]] u64 EmissionSizeBytes() const {
        const u64 bytes_per_invocation = u64{stream_stride_words} * 4 * sizeof(u32);
        if (!bytes_per_invocation ||
            invocations > std::numeric_limits<u64>::max() / bytes_per_invocation) {
            throw std::runtime_error("Native geometry emission size exceeds capacity");
        }
        return u64{invocations} * bytes_per_invocation;
    }

    [[nodiscard]] u64 PrimitiveSizeBytes() const {
        return u64{jobs} * mesh_stride_words * sizeof(u32);
    }
};
static_assert(sizeof(GeometryAssemblyParameters) == 80);

[[nodiscard]] inline GeometryAssemblyParameters GeometryAssemblyLayout(
    u32 invocations, u32 max_vertices, u32 vertex_words, u32 mesh_vertices,
    const std::array<u32, 4>& topology, u32 active_streams) {
    if (!vertex_words || !mesh_vertices || mesh_vertices > 256 || active_streams > 15) {
        throw std::runtime_error("Invalid native geometry assembly layout");
    }
    const u64 stream_stride = 4 + u64{max_vertices} * vertex_words;
    if (stream_stride > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Native geometry emission stride exceeds capacity");
    }
    GeometryAssemblyParameters parameters{invocations,
                                          max_vertices,
                                          vertex_words,
                                          mesh_vertices,
                                          topology,
                                          {},
                                          {},
                                          u32(stream_stride),
                                          4 + mesh_vertices * 8,
                                          0,
                                          0};
    u64 mesh_count{};
    for (u32 stream = 0; stream < 4; ++stream) {
        parameters.mesh_offsets[stream] = u32(mesh_count);
        if (!(active_streams & (1u << stream))) {
            continue;
        }
        if (topology[stream] > 2 || mesh_vertices <= topology[stream]) {
            throw std::runtime_error("Native geometry topology exceeds mesh vertex capacity");
        }
        const u64 stride = mesh_vertices - topology[stream];
        const u64 count = max_vertices <= mesh_vertices
                              ? 1
                              : 1 + (max_vertices - mesh_vertices + stride - 1) / stride;
        mesh_count += count;
        if (mesh_count > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Native geometry mesh count exceeds capacity");
        }
        parameters.mesh_counts[stream] = u32(count);
    }
    const u64 jobs = u64{invocations} * mesh_count;
    if (jobs > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Native geometry assembly jobs exceed capacity");
    }
    parameters.meshes_per_invocation = u32(mesh_count);
    parameters.jobs = u32(jobs);
    return parameters;
}

} // namespace Metal
