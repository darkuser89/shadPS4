// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {

using QuadParameters = RasterPackParameters;
static_assert(sizeof(QuadParameters) == 132);

constexpr std::string_view RasterQuadsSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct MeshParameters {
    uint invocations; uint max_vertices; uint vertex_words; uint mesh_vertices;
    uint topology[4]; uint mesh_counts[4]; uint mesh_offsets[4];
    uint stream_stride_words; uint mesh_stride_words; uint meshes_per_invocation; uint jobs;
};
struct Parameters {
    MeshParameters mesh;
    uint first; uint count; uint vertices_per_instance; uint quads_per_instance;
    uint source_words; uint viewport_word; uint viewport_count; uint last_provoking;
    uint input_vertices; uint position_words[2]; uint vertex_step; uint carry_vertices;
};
static_assert(sizeof(MeshParameters) == 80 && sizeof(Parameters) == 132);
struct PrimitiveIdParameters { uint first; uint instance_step; uint enabled; uint reserved; };
static_assert(sizeof(PrimitiveIdParameters) == 16);
struct RasterRows { device const uint* rows[4]; };
struct RestartRecord { uint begin; uint state; };
uint emit_raster_primitive(thread const RasterRows& source, bool valid, uint local, uint primitive_id,
    uint primitive_instance,
    constant Parameters& p, device const uint* flat_words, device uint* payload,
    device uint* records) {
    const bool polygon = p.vertex_step == 1u;
    const uint output_vertices = polygon ? 3u : 6u;
    uint errors = 0;
        device const uint* provoking = source.rows[polygon ? 0u : p.last_provoking ? p.input_vertices - 1u : 0u];
        if (valid && p.viewport_word)
            valid = provoking[p.viewport_word] < p.viewport_count;
        if (!valid) errors |= 0x4000u;
        uint anchor = 0u;
        if (valid && !polygon && p.input_vertices == 3u) {
            bool found = false;
            for (uint corner = 0; corner < 3; ++corner) {
                device const uint* a = source.rows[corner];
                device const uint* b = source.rows[(corner + 1u) % 3u];
                device const uint* c = source.rows[(corner + 2u) % 3u];
                const float ax = as_type<float>(a[p.position_words[0]]);
                const float ay = as_type<float>(a[p.position_words[1]]);
                const float bx = as_type<float>(b[p.position_words[0]]);
                const float by = as_type<float>(b[p.position_words[1]]);
                const float cx = as_type<float>(c[p.position_words[0]]);
                const float cy = as_type<float>(c[p.position_words[1]]);
                const bool right_angle = (ax == bx && ay == cy) || (ay == by && ax == cx);
                if (right_angle && !found) { anchor = corner; found = true; }
            }
            valid = found;
            if (!found) errors |= 0x8000u;
        }
        const uint next = (anchor + 1u) % 3u;
        const uint previous = (anchor + 2u) % 3u;
        uint corners[6] = {p.input_vertices == 3u ? anchor : 0u,
            p.input_vertices == 3u ? next : 1u, p.input_vertices == 3u ? previous : 2u,
            p.input_vertices == 3u ? previous : 0u, p.input_vertices == 3u ? next : 2u, 3u};
        if (p.vertex_step == 2u) {
            corners[2] = corners[4] = 3u;
            corners[5] = 2u;
        }
        for (uint output_vertex = 0; output_vertex < output_vertices; ++output_vertex) {
            device uint* row = payload + 4u + ulong(local * output_vertices + output_vertex) * p.mesh.vertex_words;
            for (uint word = 0; word < p.mesh.vertex_words; ++word) {
                uint value = 0u;
                if (valid) {
                    if (flat_words[word]) value = provoking[word];
                    else if (!polygon && p.input_vertices == 3u && corners[output_vertex] == 3u) {
                        if (!word) value = 1u;
                        else {
                            const float a = as_type<float>(source.rows[anchor][word]);
                            const float b = as_type<float>(source.rows[next][word]);
                            const float c = as_type<float>(source.rows[previous][word]);
                            value = as_type<uint>((b + c) - a);
                        }
                    } else value = source.rows[corners[output_vertex]][word];
                }
                row[word] = value;
            }
        }
        for (uint triangle = 0; triangle < output_vertices / 3u; ++triangle) {
            const uint start = local * output_vertices + triangle * 3u;
            device uint* record = records + 4u + ulong(local * (output_vertices / 3u) + triangle) * 8u;
            for (uint component = 0; component < 3; ++component)
                record[component] = start + component;
            record[3] = valid ? start : 0xffffffffu;
            record[4] = valid ? start + 2u : 0xffffffffu;
            record[5] = valid ? primitive_id : 0u;
            record[6] = valid ? primitive_instance : 0u;
            record[7] = 0u;
        }
    return errors;
}
kernel void metal4_pack_raster_quads(
    device const uint* vertices [[buffer(0)]], device uint* emissions [[buffer(1)]],
    device uint* primitives [[buffer(2)]], constant Parameters& p [[buffer(3)]],
    device const uint* flat_words [[buffer(4)]], device atomic_uint& validation_errors [[buffer(5)]],
    device const uint* carry [[buffer(6)]], device const uint* origin [[buffer(7)]],
    constant PrimitiveIdParameters& ids [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong job = ulong(group.y) * grid.x + group.x;
    if (job >= p.mesh.jobs) return;
    const bool polygon = p.vertex_step == 1u;
    const uint output_vertices = polygon ? 3u : 6u;
    const uint per_job = p.mesh.mesh_vertices / output_vertices;
    const uint first = uint(job) * per_job;
    const uint count = min(per_job, p.count - first);
    device uint* payload = emissions + job * 4u * p.mesh.stream_stride_words;
    device uint* records = primitives + job * p.mesh.mesh_stride_words;
    uint errors = 0;
    for (uint local = lane; local < count; local += 32u) {
        const uint index = p.first + first + local;
        const uint instance = index / p.quads_per_instance;
        const uint quad = index % p.quads_per_instance;
        RasterRows source;
        bool valid = true;
        for (uint corner = 0; corner < p.input_vertices; ++corner) {
            if (polygon && !corner && p.carry_vertices) source.rows[corner] = origin;
            else {
                const uint source_index = polygon ? (!corner ? 0u : quad + corner - (p.carry_vertices ? 1u : 0u))
                                                   : quad * p.vertex_step + corner;
                if ((!polygon || corner) && source_index < p.carry_vertices)
                    source.rows[corner] = carry + source_index * p.mesh.vertex_words;
                else {
                    const ulong source_vertex = ulong(instance) * p.vertices_per_instance + source_index - p.carry_vertices;
                    const ulong begin = source_vertex * p.mesh.vertex_words;
                    const bool in_bounds = begin + p.mesh.vertex_words <= p.source_words;
                    source.rows[corner] = in_bounds ? vertices + begin : vertices;
                    valid &= in_bounds;
                }
            }
            if (valid) valid &= source.rows[corner][0] == 1u;
        }
        const uint primitive_id = ids.enabled ? ids.first + instance * ids.instance_step + quad : 0u;
        errors |= emit_raster_primitive(source, valid, local, primitive_id, 0u, p, flat_words, payload, records);
    }
    errors = simd_or(errors);
    if (!lane) {
        payload[0] = count * output_vertices;
        payload[1] = payload[2] = payload[3] = 0u;
        records[0] = 0u;
        records[1] = count * output_vertices;
        records[2] = errors ? 0u : count * (output_vertices / 3u);
        records[3] = errors;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}

bool raster_restart_state_valid(device const uint* carry, constant Parameters& p) {
    return !p.carry_vertices || (carry[0] <= p.carry_vertices && !carry[1] && !carry[2] && !carry[3]);
}
device const uint* raster_restart_row(device const uint* vertices, device const uint* carry,
                                      long index, uint base, constant Parameters& p) {
    if (index >= long(base)) {
        const ulong offset = ulong(index) * p.mesh.vertex_words;
        return offset + p.mesh.vertex_words <= p.source_words ? vertices + offset : nullptr;
    }
    if (!base && index >= -3 && index < 0 && p.carry_vertices)
        return carry + 4ul + ulong(4 + index) * p.mesh.vertex_words;
    return nullptr;
}
kernel void metal4_snapshot_raster_restart(
    device const uint* vertices [[buffer(0)]], device uint* state [[buffer(1)]],
    constant Parameters& p [[buffer(3)]], device atomic_uint& validation_errors [[buffer(5)]],
    device const uint* carry [[buffer(6)]], device const RestartRecord* metadata [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint errors = raster_restart_state_valid(carry, p) ? 0u : 0x10000u;
    const uint base = p.count - p.vertices_per_instance;
    const RestartRecord last = metadata[p.count - 1u];
    if ((last.state != 1u && last.state != 2u) || last.begin > p.count ||
        (last.state == 2u && last.begin != p.count) || (last.state == 1u && last.begin >= p.count))
        errors |= 0x10000u;
    const uint begin = max(base, last.begin);
    const bool continuing = p.carry_vertices && !begin;
    const uint length = errors ? 0u : p.count - begin + (continuing ? carry[0] : 0u);
    for (uint word = lane; word < 4u * p.mesh.vertex_words; word += 32u) {
        const uint row = word / p.mesh.vertex_words;
        const uint component = word % p.mesh.vertex_words;
        device const uint* source = nullptr;
        bool used = false;
        if (length) {
            if (!row) {
                used = true;
                source = continuing && carry[0] ? carry + 4u
                    : raster_restart_row(vertices, carry, long(begin), base, p);
            } else if (length >= 4u - row) {
                used = true;
                source = raster_restart_row(vertices, carry, long(p.count) - (4u - row), base, p);
            }
        }
        if (used && !source) errors |= 0x10000u;
        state[4u + word] = source ? source[component] : 0u;
    }
    errors = simd_or(errors);
    if (!lane) {
        state[0] = errors ? 0u : length;
        state[1] = errors;
        state[2] = state[3] = 0u;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}
kernel void metal4_pack_raster_restart(
    device const uint* vertices [[buffer(0)]], device uint* emissions [[buffer(1)]],
    device uint* primitives [[buffer(2)]], constant Parameters& p [[buffer(3)]],
    device const uint* flat_words [[buffer(4)]], device atomic_uint& validation_errors [[buffer(5)]],
    device const uint* carry [[buffer(6)]], device const RestartRecord* metadata [[buffer(8)]],
    constant PrimitiveIdParameters& ids [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong job = ulong(group.y) * grid.x + group.x;
    if (job >= p.mesh.jobs) return;
    const bool polygon = p.vertex_step == 1u;
    const uint output_vertices = polygon ? 3u : 6u;
    const uint per_job = p.mesh.mesh_vertices / output_vertices;
    const uint first = uint(job) * per_job;
    const uint count = min(per_job, p.count - first);
    device uint* payload = emissions + job * 4u * p.mesh.stream_stride_words;
    device uint* records = primitives + job * p.mesh.mesh_stride_words;
    const bool state_valid = raster_restart_state_valid(carry, p);
    uint errors = state_valid ? 0u : 0x10000u;
    uint emitted = 0u;
    for (uint tile = 0; tile < count; tile += 32u) {
        const uint local = tile + lane;
        const uint index = p.first + first + local;
        const uint base = (index / p.vertices_per_instance) * p.vertices_per_instance;
        uint begin = base;
        bool continuing = false;
        bool ending = false;
        if (local < count) {
            const RestartRecord item = metadata[index];
            const bool valid = state_valid && (item.state == 1u || item.state == 2u) &&
                item.begin <= index + 1u && (item.state != 2u || item.begin == index + 1u) &&
                (item.state != 1u || item.begin <= index);
            if (!valid) errors |= 0x10000u;
            begin = max(base, item.begin);
            continuing = p.carry_vertices && !begin;
            if (valid && item.state == 1u) {
                const uint phase = index - begin + (continuing ? carry[0] : 0u);
                ending = polygon ? phase >= 2u : p.vertex_step == 2u ? phase >= 3u && (phase & 1u)
                    : phase % p.input_vertices == p.input_vertices - 1u;
            }
        }
        const uint selected = ending ? 1u : 0u;
        const uint destination = emitted + simd_prefix_exclusive_sum(selected);
        const uint tile_count = simd_sum(selected);
        if (ending) {
            RasterRows source;
            bool valid = true;
            for (uint corner = 0; corner < p.input_vertices; ++corner) {
                if (polygon && !corner)
                    source.rows[corner] = continuing && carry[0] ? carry + 4u
                        : raster_restart_row(vertices, carry, long(begin), base, p);
                else {
                    const long slot = polygon ? long(index) + corner - 2
                        : long(index) - (p.input_vertices - 1u) + corner;
                    source.rows[corner] = raster_restart_row(vertices, carry, slot, base, p);
                }
                valid &= source.rows[corner] && source.rows[corner][0] == 1u;
            }
            const uint primitive_instance = ids.enabled == 2u ? ids.first + index / p.vertices_per_instance : 0u;
            errors |= emit_raster_primitive(source, valid, destination, 0u, primitive_instance, p, flat_words, payload, records);
        }
        emitted += tile_count;
    }
    errors = simd_or(errors);
    if (!lane) {
        payload[0] = emitted * output_vertices;
        payload[1] = payload[2] = payload[3] = 0u;
        records[0] = 0u;
        records[1] = emitted * output_vertices;
        records[2] = errors ? 0u : emitted * (output_vertices / 3u);
        records[3] = errors;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}
struct CapturePackWork { Parameters pack; uint dispatch[3]; uint mesh_dispatch[3]; uint flags; };
static_assert(sizeof(CapturePackWork) == 160);
kernel void metal4_pack_capture(
    device const uint* vertices [[buffer(0)]], device uint* emissions [[buffer(1)]],
    device uint* primitives [[buffer(2)]], constant CapturePackWork& work [[buffer(3)]],
    device const uint* flat_words [[buffer(4)]], device atomic_uint& validation_errors [[buffer(5)]],
    device const uint* previous [[buffer(6)]], device const RestartRecord* metadata [[buffer(8)]],
    constant PrimitiveIdParameters& ids [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constant Parameters& p = work.pack;
    device const uint* carry = previous + 28u;
    const bool restart = (work.flags & 2u) != 0u;
    const ulong job = ulong(group.y) * grid.x + group.x;
    if (job >= p.mesh.jobs) return;
    const bool polygon = p.vertex_step == 1u;
    const uint output_vertices = polygon ? 3u : 6u;
    const uint per_job = p.mesh.mesh_vertices / output_vertices;
    const uint first = uint(job) * per_job;
    const uint count = min(per_job, p.count - first);
    device uint* payload = emissions + job * 4u * p.mesh.stream_stride_words;
    device uint* records = primitives + job * p.mesh.mesh_stride_words;
    const bool state_valid = (work.flags == 1u || work.flags == 3u) &&
        (!p.carry_vertices || (carry[1] == 1u && !carry[2] && !carry[3] && carry[0] <= p.carry_vertices));
    uint errors = state_valid ? 0u : 0x10000u;
    uint emitted = 0u;
    for (uint tile = 0; tile < count; tile += 32u) {
        const uint local = tile + lane;
        const uint index = p.first + first + local;
        const uint base = (index / p.vertices_per_instance) * p.vertices_per_instance;
        uint begin = base;
        bool continuing = false;
        bool ending = false;
        if (local < count) {
            const RestartRecord item = restart ? metadata[index] : RestartRecord{0u, 1u};
            const bool valid = state_valid && (item.state == 1u || item.state == 2u) &&
                item.begin <= index + 1u && (item.state != 2u || item.begin == index + 1u) &&
                (item.state != 1u || item.begin <= index);
            if (!valid) errors |= 0x10000u;
            begin = max(base, item.begin);
            continuing = p.carry_vertices && !begin;
            if (valid && item.state == 1u) {
                const uint phase = index - begin + (continuing ? carry[0] : 0u);
                ending = polygon ? phase >= 2u : p.vertex_step == 2u ? phase >= 3u && (phase & 1u)
                    : phase % p.input_vertices == p.input_vertices - 1u;
            }
        }
        const uint selected = ending ? 1u : 0u;
        const uint destination = emitted + simd_prefix_exclusive_sum(selected);
        const uint tile_count = simd_sum(selected);
        if (ending) {
            RasterRows source;
            bool valid = true;
            for (uint corner = 0; corner < p.input_vertices; ++corner) {
                if (polygon && !corner)
                    source.rows[corner] = continuing && carry[0] ? carry + 4u
                        : raster_restart_row(vertices, carry, long(begin), base, p);
                else {
                    const long slot = polygon ? long(index) + corner - 2
                        : long(index) - (p.input_vertices - 1u) + corner;
                    source.rows[corner] = raster_restart_row(vertices, carry, slot, base, p);
                }
                valid &= source.rows[corner] && source.rows[corner][0] == 1u;
            }
            const uint primitive_instance = ids.enabled == 2u ? ids.first + index / p.vertices_per_instance : 0u;
            errors |= emit_raster_primitive(source, valid, destination, 0u, primitive_instance, p, flat_words, payload, records);
        }
        emitted += tile_count;
    }
    errors = simd_or(errors);
    if (!lane) {
        payload[0] = emitted * output_vertices;
        payload[1] = payload[2] = payload[3] = 0u;
        records[0] = 0u;
        records[1] = emitted * output_vertices;
        records[2] = errors ? 0u : emitted * (output_vertices / 3u);
        records[3] = errors;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}
)metal";

constexpr std::string_view RasterPrimitiveCounterSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct MeshParameters {
    uint invocations; uint max_vertices; uint vertex_words; uint mesh_vertices;
    uint topology[4]; uint mesh_counts[4]; uint mesh_offsets[4];
    uint stream_stride_words; uint mesh_stride_words; uint meshes_per_invocation; uint jobs;
};
struct Parameters { MeshParameters mesh; uint reset_value; uint reset_on_instance; uint has_previous; uint reserved; };
struct Counter { uint total; uint suffix; uint instance; uint first_instance; uint valid; uint error; };
static_assert(sizeof(Parameters) == 96 && sizeof(Counter) == 24);
Counter merge_counters(Counter first, Counter last) {
    const uint errors = first.error | last.error;
    if (!first.valid) { last.error = errors; return last; }
    if (!last.valid) { first.error = errors; return first; }
    return {first.total + last.total,
        first.instance == last.instance ? first.suffix + last.suffix : last.suffix,
        last.instance, first.first_instance, 1u, errors | (first.instance > last.first_instance ? 0x20000u : 0u)};
}
Counter scan_counters(threadgroup Counter* values, Counter value, uint lane) {
    values[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < 64u; stride *= 2u) {
        const Counter own = values[lane];
        const Counter before = lane >= stride ? values[lane - stride] : Counter{};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[lane] = merge_counters(before, own);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return values[lane];
}
kernel void metal4_scan_primitive_jobs(device const uint* primitives [[buffer(0)]],
    device Counter* prefix [[buffer(1)]], device Counter* blocks [[buffer(2)]],
    constant Parameters& p [[buffer(5)]], device atomic_uint& errors [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong block = ulong(group.y) * grid.x + group.x;
    if (block >= (ulong(p.mesh.jobs) + 63u) / 64u) return;
    const ulong job = block * 64u + lane;
    Counter summary{};
    if (job < p.mesh.jobs) {
        device const uint* records = primitives + job * p.mesh.mesh_stride_words;
        const uint count = records[2];
        summary.error = records[3];
        const bool valid = !records[0] && count <= p.mesh.mesh_vertices / 3u &&
            !(count & 1u) && records[1] == count * 3u;
        if (!valid && !summary.error) summary.error |= 0x20000u;
        if (valid && !summary.error) {
            for (uint primitive = 0; primitive < count / 2u; ++primitive) {
                device const uint* record = records + 4u + ulong(primitive) * 16u;
                const uint instance = record[6];
                if ((summary.valid && instance < summary.instance) || record[14] != instance ||
                    record[7] || record[15]) summary.error |= 0x20000u;
                if (!summary.valid) summary.first_instance = instance;
                summary.suffix = summary.valid && instance == summary.instance ? summary.suffix + 1u : 1u;
                summary.instance = instance;
                summary.valid = 1u;
                ++summary.total;
            }
        }
        if (summary.error) atomic_fetch_or_explicit(&errors, summary.error, memory_order_relaxed);
    }
    threadgroup Counter values[64];
    const Counter result = scan_counters(values, summary, lane);
    if (job < p.mesh.jobs) prefix[job] = result;
    if (lane == 63u) blocks[block] = result;
}
kernel void metal4_scan_primitive_blocks(device Counter* blocks [[buffer(2)]],
    device const Counter& previous [[buffer(3)]], device Counter& next [[buffer(4)]],
    constant Parameters& p [[buffer(5)]], device atomic_uint& errors [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    if (group.x || group.y) return;
    const ulong count = (ulong(p.mesh.jobs) + 63u) / 64u;
    threadgroup Counter values[64];
    Counter running{};
    for (ulong first = 0; first < count; first += 64u) {
        const ulong item = first + lane;
        const Counter value = item < count ? blocks[item] : Counter{};
        const Counter prefix = scan_counters(values, value, lane);
        if (item < count) blocks[item] = merge_counters(running, prefix);
        running = merge_counters(running, values[63]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!lane) {
        const Counter before = p.has_previous ? previous : Counter{};
        next = merge_counters(before, running);
        if (p.reserved || p.reset_on_instance > 1u || p.has_previous > 1u ||
            before.valid > 1u || (!before.valid && (before.total || before.suffix || before.instance || before.first_instance)))
            next.error |= 0x20000u;
        if (next.error) atomic_fetch_or_explicit(&errors, next.error, memory_order_relaxed);
    }
}
kernel void metal4_assign_primitive_ids(device uint* primitives [[buffer(0)]],
    device const Counter* prefix [[buffer(1)]], device const Counter* blocks [[buffer(2)]],
    device const Counter& previous [[buffer(3)]], constant Parameters& p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong job = ulong(group.y) * grid.x + group.x;
    if (job >= p.mesh.jobs) return;
    Counter before = p.has_previous ? previous : Counter{};
    const ulong block = job / 64u;
    if (block) before = merge_counters(before, blocks[block - 1u]);
    if (job % 64u) before = merge_counters(before, prefix[job - 1u]);
    device uint* records = primitives + job * p.mesh.mesh_stride_words;
    if (prefix[job].error || records[3] || records[2] > p.mesh.mesh_vertices / 3u || (records[2] & 1u)) return;
    const uint count = records[2] / 2u;
    for (uint primitive = lane; primitive < count; primitive += 32u) {
        device uint* record = records + 4u + ulong(primitive) * 16u;
        uint ordinal = before.total + primitive;
        if (p.reset_on_instance) {
            uint first = 0u, last = primitive;
            while (first < last) {
                const uint middle = first + (last - first) / 2u;
                if (records[4u + ulong(middle) * 16u + 6u] < record[6]) first = middle + 1u;
                else last = middle;
            }
            ordinal = primitive - first + (before.valid && before.instance == record[6] ? before.suffix : 0u);
        }
        const uint id = before.error ? 0u : p.reset_value + ordinal;
        record[5] = record[13] = id;
    }
}
struct PrimitiveIdParameters { uint first; uint instance_step; uint enabled; uint reserved; };
struct ScanWork { Parameters scan; uint blocks_dispatch[3]; uint assign_dispatch[3]; PrimitiveIdParameters ids; Counter previous; };
struct CaptureDraw {
    uint vertices; uint instances; uint work_items; uint active_work_items;
    uint step_rate_0; uint step_rate_1; uint index_width; uint index_count;
    uint index_offset; uint swap_mode; uint first_vertex; uint first_instance;
};
struct CaptureBatch {
    CaptureDraw draw; ulong index_byte_offset; ulong batch_index;
    uint draw_id; uint record_index; uint first_capture; uint last_capture;
    uint base_vertex; uint first_instance; uint valid; uint error;
    uint dispatch[3]; uint reserved;
};
struct CapturePackParameters {
    MeshParameters mesh;
    uint first; uint count; uint vertices_per_instance; uint quads_per_instance;
    uint source_words; uint viewport_word; uint viewport_count; uint last_provoking;
    uint input_vertices; uint position_words[2]; uint vertex_step; uint carry_vertices;
};
struct CapturePackWork { CapturePackParameters pack; uint dispatch[3]; uint mesh_dispatch[3]; uint flags; };
struct IndirectParameters { Parameters scan; uint capacity; uint source_words; };
static_assert(sizeof(ScanWork) == 160 && sizeof(CaptureBatch) == 112 && sizeof(CapturePackWork) == 160 && sizeof(IndirectParameters) == 104);
bool counter_mesh_compatible(MeshParameters a, MeshParameters b) {
    if (!a.jobs || a.jobs > b.jobs || a.invocations != a.jobs ||
        a.max_vertices != b.max_vertices || a.vertex_words != b.vertex_words ||
        a.mesh_vertices != b.mesh_vertices || a.stream_stride_words != b.stream_stride_words ||
        a.mesh_stride_words != b.mesh_stride_words || a.meshes_per_invocation != b.meshes_per_invocation)
        return false;
    for (uint stream = 0u; stream < 4u; ++stream)
        if (a.topology[stream] != b.topology[stream] || a.mesh_counts[stream] != b.mesh_counts[stream] ||
            a.mesh_offsets[stream] != b.mesh_offsets[stream]) return false;
    return true;
}
kernel void metal4_prepare_primitive_scan(device CapturePackWork& control [[buffer(0)]],
    device const CaptureBatch& batch [[buffer(1)]], device const Counter& previous [[buffer(3)]],
    device ScanWork& work [[buffer(4)]],
    constant IndirectParameters& settings [[buffer(5)]], device atomic_uint& errors [[buffer(6)]]) {
    constant Parameters& p = settings.scan;
    ScanWork result{};
    uint failure = 0u;
    if (control.flags) {
        const MeshParameters mesh = control.pack.mesh;
        const uint per_job = mesh.mesh_vertices / 6u;
        if ((control.flags != 1u && control.flags != 3u) || p.reserved || p.reset_on_instance > 1u ||
            p.has_previous > 1u || batch.valid != 1u || batch.error || batch.reserved ||
            batch.first_capture > 1u || batch.last_capture > 1u || !batch.draw.vertices ||
            !batch.draw.instances || ulong(batch.draw.vertices) * batch.draw.instances != batch.draw.active_work_items ||
            ulong(batch.draw.first_instance) + batch.draw.instances > 0x100000000ul ||
            !counter_mesh_compatible(mesh, p.mesh) || mesh.vertex_words < 5u || mesh.topology[0] != 2u ||
            !settings.capacity || ulong(settings.capacity) * mesh.vertex_words != settings.source_words ||
            control.pack.source_words != settings.source_words || batch.draw.work_items > settings.capacity ||
            batch.draw.work_items != ((ulong(batch.draw.active_work_items) + 63u) & ~63ul) ||
            mesh.mesh_vertices > 256u || !per_job || control.pack.first ||
            control.pack.vertex_step < 2u || control.pack.vertex_step > 4u ||
            settings.capacity % (control.pack.vertex_step == 3u ? 192u : 64u) ||
            p.mesh.invocations != p.mesh.jobs || (ulong(settings.capacity) + per_job - 1u) / per_job != p.mesh.jobs ||
            control.pack.input_vertices != (control.pack.vertex_step == 2u ? 4u : control.pack.vertex_step) ||
            control.pack.count != batch.draw.active_work_items || control.pack.count > settings.capacity || control.pack.count > 0xffffffffu - per_job + 1u ||
            (control.pack.count + per_job - 1u) / per_job != mesh.jobs ||
            control.pack.vertices_per_instance != batch.draw.vertices ||
            control.pack.quads_per_instance != batch.draw.instances || control.pack.carry_vertices != batch.draw.first_vertex)
            failure = 0x20000u;
        if (!failure) {
            const uint width = min(mesh.jobs, 65535u);
            for (uint component = 0u; component < 3u; ++component) {
                const uint expected = component == 0u ? width : component == 1u
                    ? uint((ulong(mesh.jobs) + width - 1u) / width) : 1u;
                if (control.dispatch[component] != expected || control.mesh_dispatch[component] != expected)
                    failure = 0x20000u;
            }
        }
        if (!failure && ((batch.first_capture && (batch.draw.first_vertex || batch.draw.first_instance)) ||
            (!batch.first_capture && !p.has_previous))) failure = 0x10000u;
        Counter before{};
        if (!failure && !batch.first_capture) {
            before = previous;
            if (before.error) failure = before.error;
            else if (before.valid > 1u || (before.valid && (before.first_instance > before.instance ||
                before.instance > batch.draw.first_instance)) ||
                (!before.valid && (before.total || before.suffix || before.instance || before.first_instance)))
                failure = 0x20000u;
        }
        if (!failure) {
            result.scan = p;
            result.scan.mesh = mesh;
            result.scan.has_previous = batch.first_capture ? 0u : 1u;
            result.previous = before;
            result.ids = {batch.draw.first_instance, 0u, 2u, 0u};
            const uint groups[2] = {uint((ulong(mesh.jobs) + 63u) / 64u), mesh.jobs};
            for (uint stage = 0u; stage < 2u; ++stage) {
                const uint width = min(groups[stage], 65535u);
                for (uint component = 0u; component < 3u; ++component) {
                    const uint value = component == 0u ? width : component == 1u ? (groups[stage] + width - 1u) / width : 1u;
                    if (stage) result.assign_dispatch[component] = value;
                    else result.blocks_dispatch[component] = value;
                }
            }
        }
    } else {
        device const uchar* bytes = reinterpret_cast<device const uchar*>(&batch);
        for (uint byte = 0u; byte < sizeof(CaptureBatch); ++byte)
            if (bytes[byte]) failure = batch.error ? batch.error : 0x20000u;
        control = {};
    }
    if (failure) {
        control = {};
        result = {};
        result.scan.has_previous = 1u;
        result.previous.error = failure;
        atomic_fetch_or_explicit(&errors, failure, memory_order_relaxed);
    }
    work = result;
}
)metal";

} // namespace

GeometryAssembly::IndirectRasterState GeometryAssembly::SnapshotIndirectRasterVertices(
    const PreparedRasterVertices& prepared, const IndirectRasterState* previous) {
    if (!prepared.kernel.submitted || !prepared.indirect_inputs || !prepared.indirect_shader ||
        prepared.vertices.owner != this || (previous && previous->owner != this)) {
        throw std::runtime_error("Native indirect raster snapshot requires its submitted capture");
    }
    const auto& inputs = *prepared.indirect_inputs;
    if (!inputs.capture.owner || &inputs.capture.owner->scheduler != &scheduler ||
        &inputs.capture.owner->instance != &instance ||
        bool(prepared.indirect_restart) != bool(inputs.layout.parameters.restart_enabled)) {
        throw std::runtime_error("Native indirect raster snapshot has a different restart context");
    }
    const auto& layout = prepared.vertices.GetLayout();
    const bool same_layout = previous && previous->shader == prepared.indirect_shader &&
                             previous->user_data == prepared.indirect_user_data &&
                             previous->layout.vertex_words == layout.vertex_words &&
                             std::ranges::equal(previous->layout.attributes, layout.attributes,
                                                [](const auto& a, const auto& b) {
                                                    return a.attribute == b.attribute &&
                                                           a.component == b.component &&
                                                           a.stream == b.stream && a.word == b.word;
                                                });
    auto state = inputs.capture.owner->SnapshotRasterCapture(
        inputs, prepared.vertices.storage, layout.vertex_words,
        previous ? &previous->state : nullptr, same_layout);
    return {*this, std::move(state), layout, prepared.indirect_shader, prepared.indirect_user_data};
}

id<MTLComputePipelineState> GeometryAssembly::RasterQuadsPipeline(u32 stage) {
    if (stage >= raster_quads_pipelines.size()) {
        throw std::runtime_error("Invalid native raster packing stage");
    }
    if (!raster_quads_pipelines[stage]) {
        auto library = compiler.CompileLibrary(RasterQuadsSource, "shadPS4:RasterQuads");
        const std::array names{@"metal4_pack_raster_quads", @"metal4_pack_raster_restart",
                               @"metal4_snapshot_raster_restart", @"metal4_pack_capture"};
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = names[stage];
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 32;
        raster_quads_pipelines[stage] = compiler.CompileComputePipeline(descriptor);
    }
    return raster_quads_pipelines[stage];
}

id<MTLComputePipelineState> GeometryAssembly::RasterPrimitiveCounterPipeline(u32 stage) {
    if (stage >= raster_primitive_counter_pipelines.size())
        throw std::runtime_error("Invalid native raster primitive counter stage");
    if (!raster_primitive_counter_pipelines[stage]) {
        auto library =
            compiler.CompileLibrary(RasterPrimitiveCounterSource, "shadPS4:RasterPrimitiveIds");
        const std::array names{@"metal4_scan_primitive_jobs", @"metal4_scan_primitive_blocks",
                               @"metal4_assign_primitive_ids", @"metal4_prepare_primitive_scan"};
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = names[stage];
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        const u32 threads = stage == 3 ? 1 : stage == 2 ? 32 : 64;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(threads, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = threads;
        raster_primitive_counter_pipelines[stage] = compiler.CompileComputePipeline(descriptor);
    }
    return raster_primitive_counter_pipelines[stage];
}

std::shared_ptr<StagingBufferPool::Allocation> GeometryAssembly::PrepareIndirectPrimitiveScan(
    const GeometryAssemblyParameters& mesh, u32 capacity,
    const RasterPrimitiveCounterSettings& config, const IndirectDraws::RasterCapturePack& control,
    const IndirectDraws::RasterCapture& capture, const BufferRange& errors,
    const std::shared_ptr<StagingBufferPool::Allocation>& previous) {
    const auto device = instance.GetDevice();
    const auto batch = capture.GetRange();
    const auto pack = control.GetRange();
    if (!capture.owner || &capture.owner->instance != &instance ||
        &capture.owner->scheduler != &scheduler || config.reset_on_instance > 1 || !capacity ||
        !mesh.jobs || u64{capacity} * mesh.vertex_words > std::numeric_limits<u32>::max() ||
        batch.size != sizeof(RasterCaptureBatch) || pack.size != sizeof(RasterCapturePackWork) ||
        batch.buffer.device != device || pack.buffer.device != device ||
        (previous && (previous->SizeBytes() != sizeof(RasterPrimitiveCounter) ||
                      previous->GetBuffer().device != device)))
        throw std::runtime_error("Invalid native indirect primitive counter context");
    const RasterIndirectPrimitiveScanParameters data{
        {mesh, config.reset_value, config.reset_on_instance, previous != nullptr},
        capacity,
        u32(u64{capacity} * mesh.vertex_words)};
    auto uniforms = staging.Allocate(sizeof(data));
    std::memcpy(uniforms.UploadData().data(), &data, sizeof(data));
    auto work = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(sizeof(RasterPrimitiveScanWork), alignof(u32)));
    ResourceBindings bindings{instance, 7, 0, 0};
    bindings.SetBuffer(0, pack.buffer, pack.offset);
    bindings.SetBuffer(1, batch.buffer, batch.offset);
    bindings.SetBuffer(3, previous ? previous->GetBuffer() : uniforms.GetBuffer(),
                       previous ? previous->GetOffset() : uniforms.GetOffset());
    bindings.SetBuffer(4, work->GetBuffer(), work->GetOffset());
    bindings.SetBuffer(5, uniforms.GetBuffer(), uniforms.GetOffset());
    bindings.SetBuffer(6, errors.buffer, errors.offset);
    control.Retain(bindings);
    capture.Retain(bindings);
    if (previous)
        bindings.RetainUploadAllocation(previous);
    bindings.RetainUploadAllocation(work);
    bindings.RetainUploadAllocation(std::move(uniforms));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(RasterPrimitiveCounterPipeline(3), bindings);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return work;
}

std::shared_ptr<StagingBufferPool::Allocation> GeometryAssembly::AssignRasterPrimitiveIds(
    Result& result, const RasterPrimitiveCounterSettings& config, const BufferRange& errors,
    const std::shared_ptr<StagingBufferPool::Allocation>& previous,
    const std::shared_ptr<StagingBufferPool::Allocation>& scan_work) {
    const auto& mesh = result.emissions.GetParameters();
    const auto device = instance.GetDevice();
    const u32 blocks = u32((u64{mesh.jobs} + 63) / 64);
    const u64 prefix_bytes = u64{mesh.jobs} * sizeof(RasterPrimitiveCounter);
    if (!mesh.jobs || bool(result.indirect) != bool(scan_work) || (scan_work && previous) ||
        !result.primitives || config.reset_on_instance > 1 ||
        prefix_bytes > device.maxBufferLength ||
        result.primitives->SizeBytes() < mesh.PrimitiveSizeBytes() ||
        (previous && (previous->SizeBytes() != sizeof(RasterPrimitiveCounter) ||
                      previous->GetBuffer().device != device)) ||
        (scan_work && (scan_work->SizeBytes() != sizeof(RasterPrimitiveScanWork) ||
                       scan_work->GetBuffer().device != device)))
        throw std::runtime_error("Invalid native raster primitive counter buffers");
    auto prefix = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(prefix_bytes, alignof(u32)));
    auto block_prefix = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(u64{blocks} * sizeof(RasterPrimitiveCounter), alignof(u32)));
    auto next = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(sizeof(RasterPrimitiveCounter), alignof(u32)));
    auto parameters = scan_work;
    if (!parameters) {
        const RasterPrimitiveScanParameters data{mesh, config.reset_value, config.reset_on_instance,
                                                 previous != nullptr};
        parameters =
            std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(sizeof(data)));
        std::memcpy(parameters->UploadData().data(), &data, sizeof(data));
    }
    ResourceBindings bindings{instance, 7, 0, 0};
    bindings.SetBuffer(0, result.primitives->GetBuffer(), result.primitives->GetOffset());
    bindings.SetBuffer(1, prefix->GetBuffer(), prefix->GetOffset());
    bindings.SetBuffer(2, block_prefix->GetBuffer(), block_prefix->GetOffset());
    bindings.SetBuffer(3,
                       scan_work  ? scan_work->GetBuffer()
                       : previous ? previous->GetBuffer()
                                  : parameters->GetBuffer(),
                       scan_work
                           ? scan_work->GetOffset() + offsetof(RasterPrimitiveScanWork, previous)
                       : previous ? previous->GetOffset()
                                  : parameters->GetOffset());
    bindings.SetBuffer(4, next->GetBuffer(), next->GetOffset());
    bindings.SetBuffer(5, parameters->GetBuffer(), parameters->GetOffset());
    bindings.SetBuffer(6, errors.buffer, errors.offset);
    bindings.RetainUploadAllocation(result.primitives);
    bindings.RetainUploadAllocation(prefix);
    bindings.RetainUploadAllocation(block_prefix);
    bindings.RetainUploadAllocation(next);
    bindings.RetainUploadAllocation(parameters);
    if (previous)
        bindings.RetainUploadAllocation(previous);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    for (u32 stage = 0; stage < 3; ++stage) {
        commands.BindCompute(RasterPrimitiveCounterPipeline(stage), bindings);
        if (scan_work && stage != 1) {
            const u64 offset = stage == 0 ? offsetof(RasterPrimitiveScanWork, blocks_dispatch)
                                          : offsetof(RasterPrimitiveScanWork, assign_dispatch);
            commands.DispatchIndirect(
                {scan_work->GetBuffer(), scan_work->GetOffset() + offset, 3 * sizeof(u32)},
                MTLSizeMake(stage == 0 ? 64 : 32, 1, 1));
        } else {
            const u32 groups = stage == 0 ? blocks : stage == 1 ? 1 : mesh.jobs;
            const u32 width = std::min(groups, 65535u);
            commands.Dispatch(MTLSizeMake(width, (u64{groups} + width - 1) / width, 1),
                              MTLSizeMake(stage == 2 ? 32 : 64, 1, 1));
        }
        scheduler.ProducerBarrier(MTLStageDispatch, stage == 2 ? MTLStageAll : MTLStageDispatch);
    }
    if (scan_work)
        result.retained_bytes += RasterPrimitiveCounterSizeBytes(mesh.jobs, true);
    result.primitive_ids = true;
    return next;
}

GeometryAssembly::PackedIndirectRaster GeometryAssembly::AssembleIndirectRasterQuads(
    const PreparedRasterVertices& prepared, const MeshShader& mesh,
    const IndirectRasterState* previous, u32 viewport_count,
    const RasterPrimitiveCounterSettings* primitive_counter) {
    if (!prepared.indirect_inputs || !mesh.mesh || !mesh.shader || mesh.stream ||
        mesh.topology != 2 || prepared.vertices.owner != this ||
        bool(primitive_counter) != mesh.mesh->primitive_ids ||
        (primitive_counter && primitive_counter->reset_on_instance > 1) ||
        mesh.vertex_words != prepared.vertices.GetLayout().vertex_words ||
        !std::ranges::equal(mesh.attributes, prepared.vertices.GetLayout().attributes,
                            [](const auto& a, const auto& b) {
                                return a.attribute == b.attribute && a.component == b.component &&
                                       a.stream == b.stream && a.word == b.word;
                            }))
        throw std::runtime_error("Native indirect raster pack requires its matching mesh");
    const auto& inputs = *prepared.indirect_inputs;
    const auto& context = inputs.layout.parameters;
    const u32 primitive = context.capture.primitive_vertices;
    if (primitive_counter && primitive == 1)
        throw std::runtime_error("Native polygon primitive IDs require their guest counter rule");
    const u32 words = mesh.vertex_words;
    const u32 output_vertices = primitive == 1 ? 3 : 6;
    if (mesh.mesh->vertices < output_vertices ||
        mesh.mesh->primitives < (mesh.mesh->vertices / output_vertices) * (output_vertices / 3) ||
        context.capture.window.capture_capacity >
            RasterCaptureMeshCapacity(words, mesh.mesh->vertices, primitive,
                                      context.capture.window.index_width, context.restart_enabled,
                                      instance.GetDevice().maxBufferLength,
                                      mesh.viewport_extent[0] != 0, primitive_counter != nullptr))
        throw std::runtime_error("Native indirect raster pack exceeds its aggregate budget");
    RasterCapturePackConfig config{};
    config.pack.mesh = RasterCaptureMeshLayout(context.capture.window.capture_capacity, words,
                                               mesh.mesh->vertices, primitive);
    config.pack.count = context.capture.window.capture_capacity;
    config.pack.source_words = u32(u64{config.pack.count} * words);
    config.pack.viewport_count = viewport_count;
    config.pack.last_provoking = mesh.last_provoking_vertex;
    config.pack.input_vertices = primitive == 1 ? 3 : primitive == 2 ? 4 : primitive;
    config.pack.vertex_step = primitive;
    std::vector<u32> flat_words(words);
    for (const auto& attribute : mesh.attributes) {
        if (!attribute.word || attribute.word >= words || attribute.stream)
            throw std::runtime_error("Invalid native indirect raster attribute layout");
        bool flat = attribute.attribute == Shader::IR::Attribute::RenderTargetIndex ||
                    attribute.attribute == Shader::IR::Attribute::ViewportIndex;
        if (mesh.fragment)
            flat |= std::ranges::any_of(
                mesh.fragment->shader.translated.geometry_inputs, [&](const auto& varying) {
                    return varying.flat &&
                           attribute.attribute ==
                               Shader::IR::Attribute::Param0 + varying.parameter &&
                           attribute.component < varying.components;
                });
        flat_words[attribute.word] = flat;
        if (attribute.attribute == Shader::IR::Attribute::Position0 && attribute.component < 2)
            config.pack.position_words[attribute.component] = attribute.word;
        if (attribute.attribute == Shader::IR::Attribute::ViewportIndex) {
            if (attribute.component || !viewport_count || viewport_count > 16)
                throw std::runtime_error("Invalid native indirect raster viewport selection");
            config.pack.viewport_word = attribute.word;
        }
    }
    auto state = SnapshotIndirectRasterVertices(prepared, previous);
    const bool same_layout =
        previous && previous->shader == state.shader && previous->user_data == state.user_data &&
        previous->layout.vertex_words == state.layout.vertex_words &&
        std::ranges::equal(previous->layout.attributes, state.layout.attributes,
                           [](const auto& a, const auto& b) {
                               return a.attribute == b.attribute && a.component == b.component &&
                                      a.stream == b.stream && a.word == b.word;
                           });
    auto control = inputs.capture.owner->PrepareRasterPack(
        inputs, state.state, previous ? &previous->state : nullptr, config, same_layout);
    const auto errors = inputs.capture.owner->PrepareErrorBuffer();
    const bool counter_compatible = primitive_counter && control.uses_previous && previous &&
                                    previous->counter && previous->counter_settings &&
                                    *previous->counter_settings == *primitive_counter;
    std::shared_ptr<StagingBufferPool::Allocation> scan_work;
    if (primitive_counter)
        scan_work = PrepareIndirectPrimitiveScan(
            config.pack.mesh, context.capture.window.capture_capacity, *primitive_counter, control,
            inputs.capture, errors, counter_compatible ? previous->counter : nullptr);
    const auto& layout = config.pack.mesh;
    Emissions emissions{std::make_shared<StagingBufferPool::Allocation>(
                            staging.AllocatePrivate(layout.EmissionSizeBytes(), alignof(u32))),
                        layout};
    auto primitives = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(layout.PrimitiveSizeBytes(), alignof(u32)));
    auto flat = staging.Allocate(u64{words} * sizeof(u32) + sizeof(RasterPrimitiveIdParameters),
                                 alignof(u32));
    std::memcpy(flat.UploadData().data(), flat_words.data(), u64{words} * sizeof(u32));
    std::memset(flat.UploadData().data() + u64{words} * sizeof(u32), 0,
                sizeof(RasterPrimitiveIdParameters));
    const auto source = prepared.vertices.GetRange();
    const auto payload = emissions.GetRange();
    const auto work = control.GetRange();
    const auto before = control.uses_previous ? previous->GetRange() : work;
    const auto metadata = context.restart_enabled ? inputs.RestartRecords() : work;
    ResourceBindings bindings{instance, 10, 0, 0};
    bindings.SetBuffer(0, source.buffer, source.offset);
    bindings.SetBuffer(1, payload.buffer, payload.offset);
    bindings.SetBuffer(2, primitives->GetBuffer(), primitives->GetOffset());
    bindings.SetBuffer(3, work.buffer, work.offset);
    bindings.SetBuffer(4, flat.GetBuffer(), flat.GetOffset());
    bindings.SetBuffer(5, errors.buffer, errors.offset);
    bindings.SetBuffer(6, before.buffer, before.offset);
    bindings.SetBuffer(8, metadata.buffer, metadata.offset);
    bindings.SetBuffer(9, scan_work ? scan_work->GetBuffer() : flat.GetBuffer(),
                       scan_work ? scan_work->GetOffset() + offsetof(RasterPrimitiveScanWork, ids)
                                 : flat.GetOffset() + u64{words} * sizeof(u32));
    if (scan_work)
        bindings.RetainUploadAllocation(scan_work);
    prepared.vertices.Retain(bindings);
    inputs.Retain(bindings);
    state.Retain(bindings);
    if (control.uses_previous)
        previous->Retain(bindings);
    if (prepared.indirect_restart)
        prepared.indirect_restart->Retain(bindings);
    emissions.Retain(bindings);
    control.Retain(bindings);
    bindings.RetainUploadAllocation(primitives);
    bindings.RetainUploadAllocation(std::move(flat));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(RasterQuadsPipeline(3), bindings);
    commands.DispatchIndirect(control.DispatchArguments(), MTLSizeMake(32, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    const u64 retained =
        RasterCaptureMeshSizeBytes(context.capture.window.capture_capacity, words,
                                   mesh.mesh->vertices, primitive, context.restart_enabled);
    Result result{std::move(emissions), std::move(primitives), control.storage, retained,
                  mesh.mesh->vertices / (mesh.topology + 1)};
    if (primitive_counter) {
        state.counter = AssignRasterPrimitiveIds(result, *primitive_counter, errors, {}, scan_work);
        state.counter_settings = *primitive_counter;
    }
    return {std::move(result), std::move(state)};
}

std::optional<GeometryAssembly::RasterRestartState> GeometryAssembly::AssembleRasterQuads(
    const RasterVertices& vertices, const RasterDrawParameters& draw, const MeshShader& mesh,
    const BufferRange& errors, const std::function<void(Result&&)>& consume, u32 viewport_count,
    AmdGpu::PrimitiveType topology, const RasterDrawContinuation* previous,
    const RasterDrawContinuation* origin, const RasterRestartWork* restart,
    const RasterRestartState* restart_previous, const RasterPrimitiveIdParameters* primitive_ids,
    const RasterPrimitiveCounterConfig* primitive_counter,
    const RasterRestartState* counter_previous) {
    const auto device = instance.GetDevice();
    const auto source = vertices.GetRange();
    const auto& attributes = vertices.GetLayout().attributes;
    const u32 words = vertices.GetLayout().vertex_words;
    const u64 source_words = u64{vertices.WorkItems()} * words;
    const auto primitive = BuildRasterPrimitiveParameters(topology, draw);
    if (bool(primitive_ids || primitive_counter) != bool(mesh.mesh && mesh.mesh->primitive_ids) ||
        (primitive_ids && (restart || topology == AmdGpu::PrimitiveType::Polygon ||
                           primitive_ids->enabled != 1 || primitive_ids->reserved)) ||
        (primitive_counter &&
         (!restart || primitive_ids || topology == AmdGpu::PrimitiveType::Polygon)) ||
        (counter_previous && !primitive_counter)) {
        throw std::runtime_error("Native raster primitive IDs require their matching producer");
    }
    if (primitive_counter) {
        ValidateRasterPrimitiveCounter(*primitive_counter, draw,
                                       counter_previous ? &counter_previous->draw : nullptr);
        if (counter_previous &&
            (counter_previous->owner != this || !counter_previous->counter ||
             !counter_previous->counter_config ||
             *counter_previous->counter_config != *primitive_counter ||
             counter_previous->counter_topology != topology ||
             counter_previous->counter_restart_index != restart->parameters.restart_index)) {
            throw std::runtime_error(
                "Native raster primitive counter has a different draw context");
        }
    }
    const u32 input_vertices = primitive.input_vertices;
    const bool polygon = topology == AmdGpu::PrimitiveType::Polygon;
    const u32 output_vertices = polygon ? 3 : 6;
    const u64 quads =
        restart ? draw.active_work_items : u64{primitive.primitives_per_instance} * draw.instances;
    if ((input_vertices != 3 && input_vertices != 4) || !consume || vertices.owner != this ||
        (restart ? (previous || origin) : bool(previous) != bool(primitive.carry_vertices)) ||
        (!restart && bool(origin) != (polygon && bool(primitive.carry_vertices))) ||
        bool(restart_previous) != (restart && bool(draw.first_vertex)) ||
        vertices.WorkItems() != draw.work_items ||
        draw != RasterDrawLayout(draw.vertices, draw.instances, draw.step_rate_0, draw.step_rate_1,
                                 draw.index_width, draw.index_count, draw.index_offset,
                                 draw.swap_mode, draw.first_vertex, draw.first_instance) ||
        !quads || quads > std::numeric_limits<u32>::max() || !mesh.mesh || !mesh.shader ||
        mesh.stream || mesh.topology != 2 || mesh.vertex_words != words || words < 5 ||
        mesh.mesh->vertices < output_vertices || mesh.mesh->vertices > 256 ||
        mesh.mesh->primitives < (mesh.mesh->vertices / output_vertices) * (output_vertices / 3) ||
        !std::ranges::equal(
            mesh.attributes, attributes,
            [](const auto& a, const auto& b) {
                return a.attribute == b.attribute && a.component == b.component &&
                       a.stream == b.stream && a.word == b.word;
            }) ||
        source_words > std::numeric_limits<u32>::max() || !source.buffer ||
        source.buffer.device != device || source.offset % sizeof(u32) ||
        source.offset > source.buffer.length || source.size < source_words * sizeof(u32) ||
        source.size > source.buffer.length - source.offset || !errors.buffer ||
        errors.buffer.device != device || errors.offset % sizeof(u32) ||
        errors.offset > errors.buffer.length || errors.size < sizeof(u32) ||
        errors.size > errors.buffer.length - errors.offset ||
        (source.buffer == errors.buffer && source.offset < errors.offset + sizeof(u32) &&
         errors.offset < source.offset + source.size)) {
        throw std::runtime_error("Invalid native raster quad payload or mesh interface");
    }
    const auto captured_range = [&](const RasterDrawContinuation& captured, u32 first, u32 count) {
        const auto& prior = captured.vertices;
        auto range = prior.GetRange();
        const u64 used = u64{captured.parameters.vertices} * words * sizeof(u32);
        if (prior.owner != this || prior.WorkItems() != captured.parameters.work_items ||
            u64{first} + count > captured.parameters.vertices ||
            prior.GetLayout().vertex_words != words ||
            !std::ranges::equal(prior.GetLayout().attributes, attributes,
                                [](const auto& a, const auto& b) {
                                    return a.attribute == b.attribute &&
                                           a.component == b.component && a.stream == b.stream &&
                                           a.word == b.word;
                                }) ||
            !range.buffer || range.buffer.device != device || range.offset % sizeof(u32) ||
            range.offset > range.buffer.length || range.size < used ||
            range.size > range.buffer.length - range.offset ||
            (range.buffer == errors.buffer && range.offset < errors.offset + sizeof(u32) &&
             errors.offset < range.offset + range.size)) {
            throw std::runtime_error("Invalid native raster continuation payload");
        }
        range.offset += u64{first} * words * sizeof(u32);
        range.size = u64{count} * words * sizeof(u32);
        return range;
    };
    BufferRange carry = source;
    BufferRange anchor = source;
    if (previous) {
        const u32 first =
            RasterDrawCarryOffset(previous->parameters, draw, primitive.carry_vertices);
        carry = captured_range(*previous, first, primitive.carry_vertices);
    }
    if (origin) {
        ValidateRasterDrawAnchor(origin->parameters, draw);
        anchor = captured_range(*origin, 0, 1);
    }
    BufferRange restart_records{};
    const auto valid_restart_range = [&](const BufferRange& range, u64 expected) {
        return range.buffer && range.buffer.device == device && range.offset % sizeof(u32) == 0 &&
               range.offset <= range.buffer.length && range.size >= expected &&
               range.size <= range.buffer.length - range.offset &&
               !(range.buffer == errors.buffer && range.offset < errors.offset + sizeof(u32) &&
                 errors.offset < range.offset + range.size);
    };
    if (restart) {
        const auto expected = BuildRasterRestartParameters(draw, restart->parameters.restart_index);
        restart_records = restart->Records();
        if (restart->owner != this || restart->draw != draw ||
            restart->parameters.work_items != expected.work_items ||
            restart->parameters.blocks != expected.blocks ||
            restart->parameters.superblocks != expected.superblocks ||
            !valid_restart_range(restart_records, expected.RecordSizeBytes())) {
            throw std::runtime_error("Native raster restart metadata differs from its capture");
        }
    }
    if (restart_previous) {
        static_cast<void>(RasterDrawCarryOffset(restart_previous->draw, draw, 1));
        carry = restart_previous->GetRange();
        if (restart_previous->owner != this || restart_previous->layout.vertex_words != words ||
            !std::ranges::equal(restart_previous->layout.attributes, attributes,
                                [](const auto& a, const auto& b) {
                                    return a.attribute == b.attribute &&
                                           a.component == b.component && a.stream == b.stream &&
                                           a.word == b.word;
                                }) ||
            !valid_restart_range(carry, RasterRestartStateSizeBytes(words))) {
            throw std::runtime_error("Invalid native raster restart continuation state");
        }
    }
    u32 viewport_word{};
    std::array<u32, 2> position_words{};
    std::vector<u32> flat_words(words);
    for (const auto& attribute : attributes) {
        if (!attribute.word || attribute.word >= words || attribute.stream) {
            throw std::runtime_error("Invalid native raster quad attribute layout");
        }
        bool flat = attribute.attribute == Shader::IR::Attribute::RenderTargetIndex ||
                    attribute.attribute == Shader::IR::Attribute::ViewportIndex;
        if (mesh.fragment) {
            flat |= std::ranges::any_of(
                mesh.fragment->shader.translated.geometry_inputs, [&](const auto& varying) {
                    return varying.flat &&
                           attribute.attribute ==
                               Shader::IR::Attribute::Param0 + varying.parameter &&
                           attribute.component < varying.components;
                });
        }
        flat_words[attribute.word] = flat;
        if (attribute.attribute == Shader::IR::Attribute::Position0 && attribute.component < 2) {
            position_words[attribute.component] = attribute.word;
        }
        if (attribute.attribute == Shader::IR::Attribute::ViewportIndex) {
            if (attribute.component || !viewport_count || viewport_count > 16) {
                throw std::runtime_error("Invalid native raster quad viewport selection");
            }
            viewport_word = attribute.word;
        }
    }
    if (!polygon && input_vertices == 3 && (!position_words[0] || !position_words[1])) {
        throw std::runtime_error("Native rect reconstruction requires position coordinates");
    }
    const auto layout =
        GeometryAssemblyLayout(1, mesh.mesh->vertices, words, mesh.mesh->vertices, {2, 0, 0, 0}, 1);
    const u64 counter_bytes = primitive_counter ? sizeof(RasterPrimitiveCounter) : 0;
    const u64 fixed_counter_bytes = primitive_counter ? sizeof(RasterPrimitiveScanParameters) +
                                                            2 * sizeof(RasterPrimitiveCounter)
                                                      : 0;
    const u64 capacity = std::min(
        {u64{device.maxBufferLength} / layout.EmissionSizeBytes(),
         u64{device.maxBufferLength} / layout.PrimitiveSizeBytes(),
         u64{device.maxBufferLength} / std::max(u64{1}, counter_bytes),
         ((u64{32} << 20) - fixed_counter_bytes) /
             (layout.EmissionSizeBytes() + layout.PrimitiveSizeBytes() + 2 * counter_bytes),
         u64{std::numeric_limits<u32>::max()}});
    if (!capacity) {
        throw std::runtime_error("Native raster quad mesh exceeds batch storage");
    }
    auto flat_upload = std::make_shared<StagingBufferPool::Allocation>(
        staging.Allocate(u64{words} * sizeof(u32), alignof(u32)));
    std::memcpy(flat_upload->UploadData().data(), flat_words.data(), u64{words} * sizeof(u32));
    const RasterPrimitiveIdParameters id_parameters =
        primitive_counter ? RasterPrimitiveIdParameters{draw.first_instance, 0, 2}
        : primitive_ids   ? *primitive_ids
                          : RasterPrimitiveIdParameters{};
    std::shared_ptr<StagingBufferPool::Allocation> counter =
        counter_previous ? counter_previous->counter : nullptr;
    const u32 per_job = mesh.mesh->vertices / output_vertices;
    std::optional<RasterRestartState> next_state;
    if (restart) {
        const u64 bytes = RasterRestartStateSizeBytes(words);
        if (bytes > device.maxBufferLength) {
            throw std::runtime_error("Native raster restart state exceeds device capacity");
        }
        auto allocation = std::make_shared<StagingBufferPool::Allocation>(
            staging.AllocatePrivate(bytes, alignof(u32)));
        next_state = RasterRestartState{*this, draw, vertices.GetLayout(), allocation};
        const QuadParameters data{layout,
                                  0,
                                  draw.active_work_items,
                                  draw.vertices,
                                  draw.instances,
                                  u32(source_words),
                                  viewport_word,
                                  viewport_count,
                                  mesh.last_provoking_vertex,
                                  input_vertices,
                                  position_words,
                                  primitive.vertex_step,
                                  draw.first_vertex};
        auto uniforms = staging.Allocate(sizeof(data));
        std::memcpy(uniforms.UploadData().data(), &data, sizeof(data));
        const auto output = next_state->GetRange();
        ResourceBindings snapshot{instance, 9, 0, 0};
        snapshot.SetBuffer(0, source.buffer, source.offset);
        snapshot.SetBuffer(1, output.buffer, output.offset);
        snapshot.SetBuffer(3, uniforms.GetBuffer(), uniforms.GetOffset());
        snapshot.SetBuffer(5, errors.buffer, errors.offset);
        snapshot.SetBuffer(6, carry.buffer, carry.offset);
        snapshot.SetBuffer(8, restart_records.buffer, restart_records.offset);
        vertices.Retain(snapshot);
        restart->Retain(snapshot);
        if (restart_previous)
            restart_previous->Retain(snapshot);
        next_state->Retain(snapshot);
        snapshot.RetainUploadAllocation(std::move(uniforms));
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
        commands.BindCompute(RasterQuadsPipeline(2), snapshot);
        commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    }
    const auto pipeline = RasterQuadsPipeline(restart ? 1 : 0);
    for (u64 first = 0; first < quads;) {
        const u32 count = u32(std::min(quads - first, capacity * per_job));
        const u32 jobs = u32((u64{count} + per_job - 1) / per_job);
        auto emissions =
            Allocate(jobs, mesh.mesh->vertices, words, mesh.mesh->vertices, {2, 0, 0, 0}, 1);
        const auto& parameters = emissions.GetParameters();
        const QuadParameters data{parameters,
                                  u32(first),
                                  count,
                                  draw.vertices,
                                  restart ? draw.instances : primitive.primitives_per_instance,
                                  u32(source_words),
                                  viewport_word,
                                  viewport_count,
                                  mesh.last_provoking_vertex,
                                  input_vertices,
                                  position_words,
                                  primitive.vertex_step,
                                  restart ? draw.first_vertex : primitive.carry_vertices};
        auto uniforms = staging.Allocate(sizeof(data) + sizeof(id_parameters));
        std::memcpy(uniforms.UploadData().data(), &data, sizeof(data));
        std::memcpy(uniforms.UploadData().data() + sizeof(data), &id_parameters,
                    sizeof(id_parameters));
        auto output = std::make_shared<StagingBufferPool::Allocation>(
            staging.AllocatePrivate(parameters.PrimitiveSizeBytes(), alignof(u32)));
        const auto payload = emissions.GetRange();
        ResourceBindings bindings{instance, 10, 0, 0};
        bindings.SetBuffer(0, source.buffer, source.offset);
        bindings.SetBuffer(1, payload.buffer, payload.offset);
        bindings.SetBuffer(2, output->GetBuffer(), output->GetOffset());
        bindings.SetBuffer(3, uniforms.GetBuffer(), uniforms.GetOffset());
        bindings.SetBuffer(4, flat_upload->GetBuffer(), flat_upload->GetOffset());
        bindings.SetBuffer(5, errors.buffer, errors.offset);
        bindings.SetBuffer(6, carry.buffer, carry.offset);
        if (previous)
            previous->vertices.Retain(bindings);
        bindings.SetBuffer(7, anchor.buffer, anchor.offset);
        if (origin)
            origin->vertices.Retain(bindings);
        if (restart) {
            bindings.SetBuffer(8, restart_records.buffer, restart_records.offset);
            restart->Retain(bindings);
            if (restart_previous)
                restart_previous->Retain(bindings);
        }
        bindings.SetBuffer(9, uniforms.GetBuffer(), uniforms.GetOffset() + sizeof(data));
        vertices.Retain(bindings);
        emissions.Retain(bindings);
        bindings.RetainUploadAllocation(output);
        bindings.RetainUploadAllocation(flat_upload);
        bindings.RetainUploadAllocation(std::move(uniforms));
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
        commands.BindCompute(pipeline, bindings);
        const u32 width = std::min(jobs, 65535u);
        commands.Dispatch(MTLSizeMake(width, (u64{jobs} + width - 1) / width, 1),
                          MTLSizeMake(32, 1, 1));
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
        Result result{std::move(emissions), std::move(output), {}, 0, 0, primitive_ids != nullptr};
        if (primitive_counter)
            counter = AssignRasterPrimitiveIds(
                result, {primitive_counter->reset_value, primitive_counter->reset_on_instance},
                errors, counter);
        consume(std::move(result));
        first += count;
    }
    if (primitive_counter) {
        next_state->counter = std::move(counter);
        next_state->counter_config = *primitive_counter;
        next_state->counter_topology = topology;
        next_state->counter_restart_index = restart->parameters.restart_index;
    }
    return next_state;
}

} // namespace Metal
