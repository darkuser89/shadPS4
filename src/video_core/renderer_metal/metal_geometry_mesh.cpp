// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
#include <fmt/format.h>
#include "video_core/renderer_metal/metal_geometry_mesh.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

struct OutputSize {
    u32 size{};
    u32 alignment{1};

    void Add(u32 bytes, u32 align) {
        size = (size + align - 1) / align * align + bytes;
        alignment = std::max(alignment, align);
    }

    u32 Get() const {
        return (size + alignment - 1) / alignment * alignment;
    }
};
} // namespace

GeometryMeshSource GenerateGeometryMesh(u32 stream, u32 topology, u32 vertex_words,
                                        std::span<const GeometryOutputAttribute> attributes,
                                        std::span<const GeometryMeshVarying> varyings,
                                        bool last_provoking_vertex, bool flip_vertex_y,
                                        const std::array<u32, 2>& viewport_extent,
                                        bool primitive_ids) {
    const bool transform_viewport = viewport_extent[0] && viewport_extent[1];
    if (stream >= 4 || topology > 2 || !vertex_words ||
        bool(viewport_extent[0]) != bool(viewport_extent[1])) {
        throw std::runtime_error("Invalid native geometry mesh configuration");
    }
    std::vector<GeometryOutputAttribute> selected;
    u32 clip_count{};
    std::array<u32, 8> cull_words{};
    for (const auto& output : attributes) {
        if (output.stream >= 4 || !output.word || output.word >= vertex_words) {
            throw std::runtime_error("Invalid native geometry mesh payload mapping");
        }
        if (output.stream != stream) {
            continue;
        }
        const bool vector =
            output.attribute == IR::Attribute::Position0 || IR::IsParam(output.attribute);
        const bool distance = output.attribute == IR::Attribute::ClipDistance ||
                              output.attribute == IR::Attribute::CullDistance;
        const bool scalar = output.attribute == IR::Attribute::PointSize ||
                            output.attribute == IR::Attribute::RenderTargetIndex ||
                            output.attribute == IR::Attribute::ViewportIndex;
        const u32 components = vector                                            ? 4u
                               : output.attribute == IR::Attribute::ClipDistance ? 14u
                               : distance                                        ? 8u
                                                                                 : 1u;
        if ((!vector && !distance && !scalar) || output.component >= components) {
            throw std::runtime_error("Invalid native geometry mesh semantic");
        }
        for (const auto& other : selected) {
            if (output.word == other.word ||
                (output.attribute == other.attribute && output.component == other.component)) {
                throw std::runtime_error("Aliased native geometry mesh output");
            }
        }
        selected.push_back(output);
        if (output.attribute == IR::Attribute::ClipDistance) {
            clip_count = std::max(clip_count, output.component + 1);
        }
        if (output.attribute == IR::Attribute::CullDistance) {
            cull_words[output.component] = output.word;
        }
    }
    const auto word = [&](IR::Attribute attribute, u32 component) {
        const auto it = std::ranges::find_if(selected, [&](const auto& output) {
            return output.attribute == attribute && output.component == component;
        });
        return it == selected.end() ? 0u : it->word;
    };
    for (u32 component = 0; component < 4; ++component) {
        if (!word(IR::Attribute::Position0, component)) {
            throw std::runtime_error("Native geometry mesh requires a complete position");
        }
    }
    std::array<bool, 32> locations{};
    for (const auto& varying : varyings) {
        if (varying.parameter >= locations.size() || !varying.components ||
            varying.components > 4 || locations[varying.parameter]) {
            throw std::runtime_error("Invalid native geometry mesh varying");
        }
        locations[varying.parameter] = true;
        for (u32 component = 0; component < varying.components; ++component) {
            if (!word(IR::Attribute::Param0 + varying.parameter, component)) {
                throw std::runtime_error("Unmapped native geometry mesh varying");
            }
        }
    }
    OutputSize vertex_size;
    vertex_size.Add(16, 16);
    OutputSize primitive_size;
    primitive_size.Add(1, 1);
    std::string vertex_fields = "float4 position [[position]];\n";
    std::string primitive_fields = "bool culled [[primitive_culled]];\n";
    if (primitive_ids) {
        primitive_fields += "uint primitive_id [[primitive_id]];\n";
        primitive_size.Add(sizeof(u32), alignof(u32));
    }
    if (clip_count) {
        vertex_fields += fmt::format("float clip [[clip_distance]] [{}];\n", clip_count);
        vertex_size.Add(clip_count * 4, 4);
    }
    const bool point_size = topology == 0 && word(IR::Attribute::PointSize, 0);
    if (point_size) {
        vertex_fields += "float point_size [[point_size]];\n";
        vertex_size.Add(4, 4);
    }
    const auto layer = word(IR::Attribute::RenderTargetIndex, 0);
    const auto viewport = word(IR::Attribute::ViewportIndex, 0);
    if (transform_viewport && !viewport) {
        throw std::runtime_error("Native mesh viewport conversion requires a viewport export");
    }
    if (layer) {
        primitive_fields += "uint layer [[render_target_array_index]];\n";
        primitive_size.Add(4, 4);
    }
    if (viewport) {
        primitive_fields += "uint viewport [[viewport_array_index]];\n";
        primitive_size.Add(4, 4);
    }
    for (const auto& varying : varyings) {
        auto& fields = varying.flat ? primitive_fields : vertex_fields;
        fields += fmt::format("float{} param{} [[user(locn{})]];\n",
                              varying.components == 1 ? "" : std::to_string(varying.components),
                              varying.parameter, varying.parameter);
        auto& size = varying.flat ? primitive_size : vertex_size;
        const u32 bytes = varying.components == 3 ? 16 : varying.components * 4;
        size.Add(bytes, bytes);
    }
    GeometryMeshSource result{};
    result.primitive_ids = primitive_ids;
    for (u32 vertices = 256; vertices > topology; --vertices) {
        const u32 primitives = vertices - topology;
        const u32 bytes = vertices * vertex_size.Get() + primitives * primitive_size.Get() +
                          primitives * (topology + 1) * sizeof(u32);
        if (bytes <= 16384) {
            result.vertices = vertices;
            result.primitives = primitives;
            result.output_bytes = bytes;
            break;
        }
    }
    if (!result.vertices) {
        throw std::runtime_error("Native geometry mesh output exceeds capacity");
    }
    auto& source = result.source;
    source = "#include <metal_stdlib>\nusing namespace metal;\n";
    source += "struct Parameters { uint invocations, max_vertices, vertex_words, mesh_vertices; "
              "uint topology[4], mesh_counts[4], mesh_offsets[4]; "
              "uint stream_stride_words, mesh_stride_words, meshes_per_invocation, jobs; };\n";
    source += "static_assert(sizeof(Parameters) == 80);\n";
    if (transform_viewport) {
        source += "struct Viewports { uint count; uint padding[3]; float transforms[64]; };\n"
                  "static_assert(sizeof(Viewports) == 272);\n";
    }
    source += fmt::format("struct Vertex {{ {} }};\nstruct Primitive {{ {} }};\n", vertex_fields,
                          primitive_fields);
    source += fmt::format("static_assert(sizeof(Vertex) == {} && sizeof(Primitive) == {});\n",
                          vertex_size.Get(), primitive_size.Get());
    source += fmt::format("using Output = mesh<Vertex, Primitive, {}, {}, topology::{}>;\n",
                          result.vertices, result.primitives,
                          topology == 0   ? "point"
                          : topology == 1 ? "line"
                                          : "triangle");
    source +=
        fmt::format(R"metal(
[[mesh]] void metal4_geometry_mesh(device const uint* emissions [[buffer(0)]],
                                  device const uint* primitives [[buffer(1)]],
                                  constant Parameters& p [[buffer(2)]],
                                  {5}
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint2 grid [[threadgroups_per_grid]],
                                  uint lane [[thread_index_in_threadgroup]], Output output) {{
    const ulong linear_job = ulong(group.y) * grid.x + group.x;
    const uint stream = {0};
    const uint per_invocation = p.mesh_counts[stream];
    if (p.mesh_vertices != {1} || p.vertex_words != {2} || p.topology[stream] != {3} ||
        !per_invocation || linear_job >= ulong(p.invocations) * per_invocation) {{
        if (!lane) output.set_primitive_count(0);
        return;
    }}
    const uint invocation = uint(linear_job / per_invocation);
    const uint chunk = uint(linear_job % per_invocation);
    const ulong job = ulong(invocation) * p.meshes_per_invocation + p.mesh_offsets[stream] + chunk;
    device const uint* records = primitives + job * p.mesh_stride_words;
    const uint first = records[0], vertices = records[1], count = records[2];
    if (records[3] || vertices > {1} || count > {4} || first > p.max_vertices ||
        vertices > p.max_vertices - first || (count && !vertices)) {{
        if (!lane) output.set_primitive_count(0);
        return;
    }}
    device const uint* source = emissions + (ulong(invocation) * 4 + stream) * p.stream_stride_words + 4;
    if (!lane) output.set_primitive_count(count);
    for (uint vertex_index = lane; vertex_index < vertices; vertex_index += 32) {{
        device const uint* row = source + (ulong(first) + vertex_index) * p.vertex_words;
        Vertex value{{}};
)metal",
                    stream, result.vertices, vertex_words, topology, result.primitives,
                    transform_viewport ? "constant Viewports& viewport_data [[buffer(3)]]," : "");
    for (u32 component = 0; component < 4; ++component) {
        source += fmt::format("value.position[{}] = as_type<float>(row[{}]);\n", component,
                              word(IR::Attribute::Position0, component));
    }
    if (transform_viewport) {
        source += fmt::format(R"metal(
    device const uint* primitive_record = records + 4 + ulong(vertex_index / {0}) * 8;
    const uint provoking = primitive_record[{1}];
    const uint viewport_index = provoking < vertices
        ? source[(ulong(first) + provoking) * p.vertex_words + {2}] : 0u;
    const uint viewport_word = min(viewport_index, viewport_data.count - 1u) * 4u;
    value.position.x = (value.position.x * viewport_data.transforms[viewport_word + 2u] +
                        viewport_data.transforms[viewport_word]) / {3}.0f - 1.0f;
    value.position.y = (value.position.y * viewport_data.transforms[viewport_word + 3u] +
                        viewport_data.transforms[viewport_word + 1u]) / {4}.0f - 1.0f;
)metal",
                              topology + 1, last_provoking_vertex ? 4 : 3, viewport,
                              viewport_extent[0], viewport_extent[1]);
    }
    if (flip_vertex_y) {
        source += "value.position.y = -value.position.y;\n";
    }
    for (u32 component = 0; component < clip_count; ++component) {
        if (const auto offset = word(IR::Attribute::ClipDistance, component)) {
            source += fmt::format("value.clip[{}] = as_type<float>(row[{}]);\n", component, offset);
        }
    }
    if (point_size) {
        source += fmt::format("value.point_size = as_type<float>(row[{}]);\n",
                              word(IR::Attribute::PointSize, 0));
    }
    const auto write_varying = [&](const GeometryMeshVarying& varying) {
        for (u32 component = 0; component < varying.components; ++component) {
            source += fmt::format("value.param{}{} = as_type<float>(row[{}]);\n", varying.parameter,
                                  varying.components == 1 ? "" : fmt::format("[{}]", component),
                                  word(IR::Attribute::Param0 + varying.parameter, component));
        }
    };
    for (const auto& varying : varyings) {
        if (!varying.flat) {
            write_varying(varying);
        }
    }
    source += "output.set_vertex(vertex_index, value);\n}\n";
    source += fmt::format(R"metal(
    for (uint primitive = lane; primitive < count; primitive += 32) {{
        device const uint* record = records + 4 + ulong(primitive) * 8;
        Primitive value{{}};
        uint indices[{0}];
        bool invalid = record[3] >= vertices || record[4] >= vertices;
        for (uint component = 0; component < {0}; ++component) {{
            indices[component] = record[component];
            invalid |= indices[component] >= vertices;
        }}
        if (!invalid) {{
)metal",
                          topology + 1);
    if (primitive_ids) {
        source += "value.primitive_id = record[5];\n";
    }
    if (layer || viewport ||
        std::ranges::any_of(varyings, [](const auto& varying) { return varying.flat; })) {
        source += fmt::format(
            "device const uint* row = source + (ulong(first) + record[{}]) * p.vertex_words;\n",
            last_provoking_vertex ? 4 : 3);
    }
    if (layer) {
        source += fmt::format("value.layer = row[{}];\n", layer);
    }
    if (viewport) {
        source += fmt::format("value.viewport = row[{}];\n", viewport);
    }
    for (const auto& varying : varyings) {
        if (varying.flat) {
            write_varying(varying);
        }
    }
    for (const auto offset : cull_words) {
        if (offset) {
            source += fmt::format(R"metal(
            bool outside_{0} = true;
            for (uint component = 0; component < {1}; ++component)
                outside_{0} &= as_type<float>(source[(ulong(first) + indices[component]) * p.vertex_words + {0}]) < 0.0f;
            value.culled |= outside_{0};
)metal",
                                  offset, topology + 1);
        }
    }
    source += fmt::format(R"metal(
        }}
        value.culled |= invalid;
        for (uint component = 0; component < {0}; ++component)
            output.set_index(primitive * {0} + component, invalid ? 0u : indices[component]);
        output.set_primitive(primitive, value);
    }}
}}
)metal",
                          topology + 1);
    return result;
}

} // namespace Metal
