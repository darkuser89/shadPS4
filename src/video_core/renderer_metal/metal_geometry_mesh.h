// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <string>
#include "video_core/renderer_metal/metal_geometry_output.h"

namespace Metal {

struct GeometryMeshVarying {
    u32 parameter;
    u32 components;
    bool flat;
};

struct GeometryMeshSource {
    std::string source;
    u32 vertices;
    u32 primitives;
    u32 output_bytes;
    bool primitive_ids{};
};

struct GeometryMeshViewports {
    u32 count;
    std::array<u32, 3> padding{};
    std::array<std::array<float, 4>, 16> transforms{};
};
static_assert(sizeof(GeometryMeshViewports) == 272);

[[nodiscard]] GeometryMeshSource GenerateGeometryMesh(
    u32 stream, u32 topology, u32 vertex_words, std::span<const GeometryOutputAttribute> attributes,
    std::span<const GeometryMeshVarying> varyings, bool last_provoking_vertex, bool flip_vertex_y,
    const std::array<u32, 2>& viewport_extent = {}, bool primitive_ids = false);

} // namespace Metal
