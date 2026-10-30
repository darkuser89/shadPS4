// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_guest_page_table.h"
#include "video_core/renderer_metal/metal_indirect.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
constexpr std::string_view IndirectSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint max_count; uint stride_words; uint primitive; uint index_capacity;
    ulong argument_address; ulong count_address; uint paged; uint has_count;
};
static_assert(sizeof(Parameters) == 40);
struct ExecutionRange { uint location; uint length; };
struct CommandBinding { command_buffer commands [[id(0)]]; };
static_assert(sizeof(CommandBinding) == 8);
static_assert(sizeof(ExecutionRange) == 8);
static_assert(uint(primitive_type::point) == 0 && uint(primitive_type::line) == 1 &&
              uint(primitive_type::line_strip) == 2 && uint(primitive_type::triangle) == 3 &&
              uint(primitive_type::triangle_strip) == 4);

bool read_word(device const ulong* root, ulong address, thread uint& value) {
    if (address >= (1ul << 40) || (address & 3ul)) return false;
    ulong leaf_address = root[address >> 24];
    if (!leaf_address) return false;
    device const ulong* leaf = reinterpret_cast<device const ulong*>(leaf_address);
    ulong page = leaf[(address >> 14) & 1023ul];
    if (!page) return false;
    value = *reinterpret_cast<device const uint*>(page + (address & 16383ul));
    return true;
}
uint active_draws(device const uint* count, device const ulong* root,
                  constant Parameters& parameters, device atomic_uint& errors) {
    if (atomic_load_explicit(&errors, memory_order_relaxed) & 64u) return 0;
    if (!parameters.has_count) return parameters.max_count;
    uint value;
    if (parameters.paged) {
        if (!read_word(root, parameters.count_address, value)) {
            atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
            return 0;
        }
    } else value = count[0];
    return min(value, parameters.max_count);
}
template<uint N>
bool read_arguments(device const uint* arguments, device const ulong* root,
                    constant Parameters& parameters, device atomic_uint& errors,
                    uint draw, thread uint (&args)[N]) {
    ulong offset = ulong(draw) * parameters.stride_words;
    for (uint i = 0; i < N; ++i) {
        if (parameters.paged) {
            if (!read_word(root, parameters.argument_address + (offset + i) * 4ul, args[i])) {
                atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
                return false;
            }
        } else args[i] = arguments[offset + i];
    }
    return true;
}
kernel void metal4_encode_draws(device const uint* arguments [[buffer(0)]],
                               device const uint* count [[buffer(1)]],
                               constant CommandBinding& binding [[buffer(2)]],
                               device ExecutionRange& range [[buffer(3)]],
                               constant Parameters& parameters [[buffer(4)]],
                               device atomic_uint& errors [[buffer(6)]],
                               device const ulong* root [[buffer(7)]],
                               uint draw [[thread_position_in_grid]]) {
    if (draw >= parameters.max_count) return;
    uint active = active_draws(count, root, parameters, errors);
    if (draw == 0) range = {0, active};
    render_command command(binding.commands, draw);
    command.reset();
    if (draw >= active) return;
    uint args[4];
    if (!read_arguments(arguments, root, parameters, errors, draw, args)) return;
    command.draw_primitives(primitive_type(parameters.primitive), args[2], args[0], args[1], args[3]);
}

template<typename T>
void encode_indexed(device const uint* arguments, device const uint* count,
                    command_buffer commands, device ExecutionRange& range,
                    constant Parameters& parameters, device const T* indices,
                    device atomic_uint& errors, device const ulong* root, uint draw) {
    if (draw >= parameters.max_count) return;
    uint active = active_draws(count, root, parameters, errors);
    if (draw == 0) range = {0, active};
    render_command command(commands, draw);
    command.reset();
    if (draw >= active) return;
    uint args[5];
    if (!read_arguments(arguments, root, parameters, errors, draw, args)) return;
    if (!args[0] || !args[1]) return;
    if (args[2] > parameters.index_capacity || args[0] > parameters.index_capacity - args[2]) {
        atomic_fetch_or_explicit(&errors, 1u, memory_order_relaxed);
        return;
    }
    command.draw_indexed_primitives(primitive_type(parameters.primitive), args[0],
                                    indices + args[2], args[1], args[3], args[4]);
}
kernel void metal4_encode_indexed16(device const uint* arguments [[buffer(0)]],
                                   device const uint* count [[buffer(1)]],
                                   constant CommandBinding& binding [[buffer(2)]],
                                   device ExecutionRange& range [[buffer(3)]],
                                   constant Parameters& parameters [[buffer(4)]],
                                   device const ushort* indices [[buffer(5)]],
                                   device atomic_uint& errors [[buffer(6)]],
                                   device const ulong* root [[buffer(7)]],
                                   uint draw [[thread_position_in_grid]]) {
    encode_indexed(arguments, count, binding.commands, range, parameters, indices, errors, root, draw);
}
kernel void metal4_encode_indexed32(device const uint* arguments [[buffer(0)]],
                                   device const uint* count [[buffer(1)]],
                                   constant CommandBinding& binding [[buffer(2)]],
                                   device ExecutionRange& range [[buffer(3)]],
                                   constant Parameters& parameters [[buffer(4)]],
                                   device const uint* indices [[buffer(5)]],
                                   device atomic_uint& errors [[buffer(6)]],
                                   device const ulong* root [[buffer(7)]],
                                   uint draw [[thread_position_in_grid]]) {
    encode_indexed(arguments, count, binding.commands, range, parameters, indices, errors, root, draw);
}
struct RasterParameters {
    Parameters source;
    uint first_draw; uint record_count; uint capture_capacity; uint index_width;
};
struct RasterRecord {
    uint draw_id; uint active; uint arguments[5]; uint index_width;
    uint index_prefix_count; uint error;
    ulong work_items; ulong capture_batches; ulong index_byte_offset;
};
static_assert(sizeof(RasterParameters) == 56);
static_assert(sizeof(RasterRecord) == 64);
kernel void metal4_prepare_raster_window(device const uint* arguments [[buffer(0)]],
                                        device const uint* count [[buffer(1)]],
                                        device uint& source_complete [[buffer(2)]],
                                        device RasterRecord* records [[buffer(3)]],
                                        constant RasterParameters& parameters [[buffer(4)]],
                                        device atomic_uint& errors [[buffer(6)]],
                                        device const ulong* root [[buffer(7)]],
                                        uint local_draw [[thread_position_in_grid]]) {
    if (local_draw >= parameters.record_count) return;
    if (!local_draw) source_complete = 0u;
    RasterRecord result = {};
    result.draw_id = parameters.first_draw + local_draw;
    result.index_width = parameters.index_width;
    uint active = parameters.source.max_count;
    if (parameters.source.has_count) {
        if (parameters.source.paged && parameters.source.has_count != 2u) {
            if (!read_word(root, parameters.source.count_address, active)) {
                if (!local_draw) source_complete = 1u;
                result.error = 64u;
                atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
                records[local_draw] = result;
                return;
            }
        } else active = count[0];
    }
    active = min(active, parameters.source.max_count);
    if (!local_draw)
        source_complete = uint(ulong(parameters.first_draw) + parameters.record_count >= active);
    if (result.draw_id >= active) {
        records[local_draw] = result;
        return;
    }
    uint args[5] = {};
    bool valid;
    if (parameters.index_width) {
        valid = read_arguments(arguments, root, parameters.source, errors, local_draw, args);
    } else {
        uint nonindexed[4];
        valid = read_arguments(arguments, root, parameters.source, errors, local_draw, nonindexed);
        if (valid) for (uint i = 0; i < 4; ++i) args[i] = nonindexed[i];
    }
    if (!valid) {
        result.error = 64u;
        records[local_draw] = result;
        return;
    }
    for (uint i = 0; i < 5; ++i) result.arguments[i] = args[i];
    result.active = uint(args[0] != 0 && args[1] != 0);
    result.work_items = ulong(args[0]) * args[1];
    if (result.active) {
        if (args[0] <= parameters.capture_capacity) {
            uint instances = parameters.capture_capacity / args[0];
            result.capture_batches = (ulong(args[1]) + instances - 1) / instances;
        } else {
            ulong batches = (ulong(args[0]) + parameters.capture_capacity - 1) /
                            parameters.capture_capacity;
            result.capture_batches = batches * args[1];
        }
    }
    if (parameters.index_width) {
        result.index_byte_offset = ulong(args[2]) * parameters.index_width;
        if (args[2] < parameters.source.index_capacity)
            result.index_prefix_count = min(args[0], parameters.source.index_capacity - args[2]);
    }
    records[local_draw] = result;
}
struct CaptureParameters {
    RasterParameters window;
    uint primitive_vertices; uint step_rate_0; uint step_rate_1;
    uint index_offset; uint swap_mode; uint reserved;
};
struct CaptureCursor { uint record_index; uint completed; ulong batch_index; };
struct CaptureDraw {
    uint vertices; uint instances; uint work_items; uint active_work_items;
    uint step_rate_0; uint step_rate_1; uint index_width; uint index_count;
    uint index_offset; uint swap_mode; uint first_vertex; uint first_instance;
};
struct CaptureBatch {
    CaptureDraw draw;
    ulong index_byte_offset; ulong batch_index;
    uint draw_id; uint record_index; uint first_capture; uint last_capture;
    uint base_vertex; uint first_instance; uint valid; uint error;
    uint dispatch[3]; uint reserved;
};
static_assert(sizeof(CaptureParameters) == 80 && sizeof(CaptureCursor) == 16);
static_assert(sizeof(CaptureDraw) == 48 && sizeof(CaptureBatch) == 112);
void capture_fault(device CaptureCursor& cursor, thread CaptureBatch& batch,
                   device atomic_uint& errors, uint records, uint error) {
    batch.error = error;
    cursor = CaptureCursor{records, 1u, 0ul};
    atomic_fetch_or_explicit(&errors, error, memory_order_relaxed);
}
kernel void metal4_next_raster_capture(device const RasterRecord* records [[buffer(0)]],
                                      device CaptureCursor& cursor [[buffer(1)]],
                                      device CaptureBatch& batch [[buffer(3)]],
                                      constant CaptureParameters& p [[buffer(4)]],
                                      device atomic_uint& errors [[buffer(6)]],
                                      uint invocation [[thread_position_in_grid]]) {
    if (invocation) return;
    CaptureBatch result = {};
    const uint count = p.window.record_count;
    const uint capacity = p.window.capture_capacity;
    if (cursor.completed > 1u || cursor.record_index > count ||
        (cursor.completed && (cursor.record_index != count || cursor.batch_index)) ||
        (cursor.record_index == count && cursor.batch_index)) {
        capture_fault(cursor, result, errors, count, 0x20000u);
        batch = result;
        return;
    }
    while (cursor.record_index < count) {
        const RasterRecord record = records[cursor.record_index];
        result.record_index = cursor.record_index;
        result.draw_id = p.window.first_draw + cursor.record_index;
        result.batch_index = cursor.batch_index;
        const uint vertices = record.arguments[0];
        const uint instances = record.arguments[1];
        const uint indexed = p.window.index_width;
        const uint prefix = indexed && record.arguments[2] < p.window.source.index_capacity
            ? min(vertices, p.window.source.index_capacity - record.arguments[2]) : 0u;
        const ulong index_offset = indexed ? ulong(record.arguments[2]) * indexed : 0ul;
        ulong batches = 0ul;
        if (vertices && instances) {
            if (vertices <= capacity) {
                const uint per_batch = capacity / vertices;
                batches = (ulong(instances) + per_batch - 1u) / per_batch;
            } else {
                batches = ((ulong(vertices) + capacity - 1u) / capacity) * instances;
            }
        }
        if (record.draw_id != result.draw_id || record.index_width != indexed ||
            record.error != (record.error & 64u) || record.active != uint(vertices && instances) ||
            record.work_items != ulong(vertices) * instances || record.capture_batches != batches ||
            record.index_prefix_count != prefix || record.index_byte_offset != index_offset ||
            (!indexed && record.arguments[4])) {
            capture_fault(cursor, result, errors, count, 0x20000u);
            batch = result;
            return;
        }
        if (record.error) {
            capture_fault(cursor, result, errors, count, record.error);
            batch = result;
            return;
        }
        if (!record.active) {
            if (cursor.batch_index) {
                capture_fault(cursor, result, errors, count, 0x20000u);
                batch = result;
                return;
            }
            ++cursor.record_index;
            continue;
        }
        if (cursor.batch_index >= batches) {
            capture_fault(cursor, result, errors, count, 0x20000u);
            batch = result;
            return;
        }
        CaptureDraw draw = {};
        if (vertices <= capacity) {
            const uint per_batch = capacity / vertices;
            draw.vertices = vertices;
            draw.first_instance = uint(cursor.batch_index * per_batch);
            draw.instances = min(per_batch, instances - draw.first_instance);
            draw.index_count = prefix;
        } else {
            const ulong per_instance = (ulong(vertices) + capacity - 1u) / capacity;
            draw.first_vertex = uint(cursor.batch_index % per_instance * capacity);
            draw.first_instance = uint(cursor.batch_index / per_instance);
            draw.vertices = min(capacity, vertices - draw.first_vertex);
            draw.instances = 1u;
            draw.index_count = draw.first_vertex < prefix ? min(draw.vertices, prefix - draw.first_vertex) : 0u;
        }
        draw.active_work_items = draw.vertices * draw.instances;
        draw.work_items = (draw.active_work_items + 63u) & ~63u;
        draw.step_rate_0 = p.step_rate_0;
        draw.step_rate_1 = p.step_rate_1;
        draw.index_width = indexed;
        draw.index_offset = p.index_offset;
        draw.swap_mode = p.swap_mode;
        result.draw = draw;
        result.index_byte_offset = index_offset + ulong(draw.first_vertex) * indexed;
        result.first_capture = uint(cursor.batch_index == 0ul);
        result.last_capture = uint(cursor.batch_index + 1ul == batches);
        result.base_vertex = record.arguments[indexed ? 3u : 2u];
        result.first_instance = record.arguments[indexed ? 4u : 3u];
        result.valid = 1u;
        result.dispatch[0] = draw.work_items / 64u;
        result.dispatch[1] = result.dispatch[2] = 1u;
        if (result.last_capture) {
            ++cursor.record_index;
            cursor.batch_index = 0ul;
        } else ++cursor.batch_index;
        cursor.completed = uint(cursor.record_index == count);
        batch = result;
        return;
    }
    cursor = CaptureCursor{count, 1u, 0ul};
    batch = CaptureBatch{};
}
struct CaptureInputParameters {
    CaptureParameters capture;
    ulong index_address;
    float viewport[4]; uint system_sgprs[16];
    uint restart_enabled; uint restart_index;
    uint vertex_sgpr_offset; uint instance_sgpr_offset;
};
struct CaptureInput { uint system_sgprs[16]; uint vgprs[4]; uint active; };
struct CaptureRestart { uint begin; uint state; };
static_assert(sizeof(CaptureInputParameters) == 184);
static_assert(sizeof(CaptureInput) == 84 && sizeof(CaptureRestart) == 8);

bool capture_input_valid(device const CaptureBatch& batch, constant CaptureInputParameters& p) {
    const CaptureDraw draw = batch.draw;
    const CaptureParameters capture = p.capture;
    const RasterParameters window = capture.window;
    const ulong active = ulong(draw.vertices) * draw.instances;
    if (batch.valid != 1u || batch.error || batch.reserved ||
        batch.record_index >= window.record_count ||
        batch.draw_id != window.first_draw + batch.record_index ||
        batch.first_capture > 1u || batch.last_capture > 1u ||
        batch.first_capture != uint(batch.batch_index == 0ul) ||
        !active || active > window.capture_capacity ||
        draw.active_work_items != active || draw.work_items != ((active + 63ul) & ~63ul) ||
        draw.work_items > window.capture_capacity || draw.index_count > draw.vertices ||
        draw.index_width != window.index_width || draw.step_rate_0 != capture.step_rate_0 ||
        draw.step_rate_1 != capture.step_rate_1 || draw.index_offset != capture.index_offset ||
        draw.swap_mode != capture.swap_mode ||
        ulong(draw.first_vertex) + draw.vertices > 0xfffffffful ||
        ulong(draw.first_instance) + draw.instances > (1ul << 32) ||
        batch.dispatch[0] != draw.work_items / 64u || batch.dispatch[1] != 1u || batch.dispatch[2] != 1u)
        return false;
    if (!draw.index_width) return !draw.index_count && !batch.index_byte_offset;
    const ulong bytes = ulong(window.source.index_capacity) * draw.index_width;
    return batch.index_byte_offset % draw.index_width == 0ul &&
        (!draw.index_count || (batch.index_byte_offset <= bytes &&
          ulong(draw.index_count) * draw.index_width <= bytes - batch.index_byte_offset));
}
bool capture_read_index(device const ulong* root, ulong address, uint width, thread uint& value) {
    if (address >= (1ul << 40) || address % width || width > (1ul << 40) - address) return false;
    uint word;
    if (!read_word(root, address & ~3ul, word)) return false;
    value = width == 2u ? (word >> (uint(address & 2ul) * 8u)) & 0xffffu : word;
    return true;
}
bool capture_batch_empty(device const CaptureBatch& batch) {
    device const uchar* bytes = reinterpret_cast<device const uchar*>(&batch);
    for (uint byte = 0u; byte < sizeof(CaptureBatch); ++byte)
        if (bytes[byte] != 0u) return false;
    return true;
}
struct CaptureRestartWork {
    uint work_items; uint blocks; uint superblocks; uint restart_index;
    uint dispatch[4][3];
};
static_assert(sizeof(CaptureRestartWork) == 64);
kernel void metal4_prepare_capture_restart(device const CaptureBatch& batch [[buffer(0)]],
                                           constant CaptureInputParameters& p [[buffer(2)]],
                                           device CaptureRestartWork& work [[buffer(4)]],
                                           device atomic_uint& errors [[buffer(6)]]) {
    CaptureRestartWork result = {};
    if (!capture_batch_empty(batch)) {
        if (p.restart_enabled != 1u || !p.capture.window.index_width ||
            (p.capture.window.index_width == 2u && p.restart_index > 0xffffu) ||
            !capture_input_valid(batch, p)) {
            atomic_fetch_or_explicit(&errors, batch.error ? batch.error : 0x40000u, memory_order_relaxed);
        } else {
            result.work_items = batch.draw.work_items;
            result.blocks = result.work_items / 64u;
            result.superblocks = (result.blocks + 63u) / 64u;
            result.restart_index = p.restart_index;
            for (uint stage = 0u; stage < 4u; ++stage) {
                result.dispatch[stage][0] = stage == 1u ? result.superblocks :
                                             stage == 2u ? 1u : result.blocks;
                result.dispatch[stage][1] = result.dispatch[stage][2] = 1u;
            }
        }
    }
    work = result;
}
struct CaptureSnapshotParameters {
    CaptureInputParameters inputs;
    uint vertex_words; uint source_words; uint has_previous; uint same_context;
};
struct CaptureSnapshotHeader {
    CaptureBatch batch;
    uint length; uint valid; uint error; uint reserved;
};
static_assert(sizeof(CaptureSnapshotParameters) == 200 && sizeof(CaptureSnapshotHeader) == 128);
bool capture_snapshot_contiguous(device const CaptureBatch& batch,
                                  device const CaptureSnapshotHeader& previous,
                                  constant CaptureSnapshotParameters& p) {
    const CaptureBatch before = previous.batch;
    if (previous.valid != 1u || previous.error || previous.reserved ||
        !capture_input_valid(previous.batch, p.inputs) || before.last_capture ||
        before.batch_index == 0xfffffffffffffffful || batch.batch_index != before.batch_index + 1ul ||
        batch.record_index != before.record_index || batch.draw_id != before.draw_id ||
        batch.base_vertex != before.base_vertex || batch.first_instance != before.first_instance ||
        previous.length > ulong(before.draw.first_vertex) + before.draw.vertices ||
        (!p.inputs.restart_enabled &&
         previous.length != ulong(before.draw.first_vertex) + before.draw.vertices))
        return false;
    if (!batch.draw.first_vertex)
        return ulong(before.draw.first_instance) + before.draw.instances == batch.draw.first_instance;
    return before.draw.instances == 1u && batch.draw.instances == 1u &&
        before.draw.first_instance == batch.draw.first_instance &&
        ulong(before.draw.first_vertex) + before.draw.vertices == batch.draw.first_vertex;
}
device const uint* capture_snapshot_row(device const uint* vertices,
    device const uint* previous, long index, uint base, bool continuing,
    constant CaptureSnapshotParameters& p) {
    if (index >= long(base)) {
        const ulong first = ulong(index) * p.vertex_words;
        return first + p.vertex_words <= p.source_words ? vertices + first : nullptr;
    }
    if (continuing && !base && index >= -3 && index < 0)
        return previous + 32ul + ulong(4 + index) * p.vertex_words;
    return nullptr;
}
kernel void metal4_snapshot_capture(device const CaptureBatch& batch [[buffer(0)]],
                                    device const uint* vertices [[buffer(1)]],
                                    constant CaptureSnapshotParameters& p [[buffer(2)]],
                                    device uint* state [[buffer(3)]],
                                    device const uint* previous [[buffer(4)]],
                                    device const CaptureRestart* metadata [[buffer(5)]],
                                    device atomic_uint& validation_errors [[buffer(6)]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const bool empty = capture_batch_empty(batch);
    uint errors = 0u;
    CaptureSnapshotHeader result = {};
    uint begin = 0u, base = 0u, count = 0u, length = 0u;
    bool continuing = false;
    if (!empty) {
        result.batch = batch;
        if (p.vertex_words < 5u || p.has_previous > 1u || p.same_context > 1u ||
            p.inputs.restart_enabled > 1u || !capture_input_valid(batch, p.inputs) ||
            (batch.first_capture && (batch.draw.first_vertex || batch.draw.first_instance)) ||
            ulong(batch.draw.work_items) * p.vertex_words > p.source_words) {
            errors = batch.error ? batch.error : 0x40000u;
        } else {
            device const CaptureSnapshotHeader& before =
                *reinterpret_cast<device const CaptureSnapshotHeader*>(previous);
            if (!batch.first_capture && (!p.has_previous || !p.same_context ||
                                          !capture_snapshot_contiguous(batch, before, p)))
                errors |= 0x10000u;
            count = batch.draw.active_work_items;
            base = count - batch.draw.vertices;
            begin = base;
            if (p.inputs.restart_enabled) {
                const CaptureRestart last = metadata[count - 1u];
                if ((last.state != 1u && last.state != 2u) || last.begin > count ||
                    (last.state == 2u && last.begin != count) ||
                    (last.state == 1u && last.begin >= count))
                    errors |= 0x10000u;
                begin = max(base, last.begin);
            }
            continuing = !errors && !base && !begin && batch.draw.first_vertex != 0u;
            const ulong total = ulong(count - min(begin, count)) + (continuing ? before.length : 0u);
            if (total > 0xfffffffful) errors |= 0x10000u;
            length = errors ? 0u : uint(total);
        }
    }
    for (uint word = lane; word < 4u * p.vertex_words; word += 32u) {
        const uint row = word / p.vertex_words, component = word % p.vertex_words;
        device const uint* source = nullptr;
        bool used = false;
        if (length) {
            if (!row) {
                used = true;
                device const CaptureSnapshotHeader& before =
                    *reinterpret_cast<device const CaptureSnapshotHeader*>(previous);
                source = continuing && before.length ? previous + 32u
                    : capture_snapshot_row(vertices, previous, long(begin), base, continuing, p);
            } else if (length >= 4u - row) {
                used = true;
                source = capture_snapshot_row(vertices, previous, long(count) - (4u - row),
                                               base, continuing, p);
            }
        }
        if (used && (!source || source[0] != 1u)) errors |= 0x10000u;
        state[32u + word] = source ? source[component] : 0u;
    }
    errors = simd_or(errors);
    if (errors)
        for (uint word = lane; word < 4u * p.vertex_words; word += 32u) state[32u + word] = 0u;
    if (!lane) {
        result.length = errors ? 0u : length;
        result.valid = uint(!empty && !errors);
        result.error = errors;
        *reinterpret_cast<device CaptureSnapshotHeader*>(state) = result;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors, memory_order_relaxed);
    }
}
struct CaptureMeshParameters {
    uint invocations; uint max_vertices; uint vertex_words; uint mesh_vertices;
    uint topology[4]; uint mesh_counts[4]; uint mesh_offsets[4];
    uint stream_stride_words; uint mesh_stride_words; uint meshes_per_invocation; uint jobs;
};
struct CapturePackParameters {
    CaptureMeshParameters mesh;
    uint first; uint count; uint vertices_per_instance; uint quads_per_instance;
    uint source_words; uint viewport_word; uint viewport_count; uint last_provoking;
    uint input_vertices; uint position_words[2]; uint vertex_step; uint carry_vertices;
};
struct CapturePackConfig { CaptureSnapshotParameters snapshot; CapturePackParameters pack; uint reserved; };
struct CapturePackWork { CapturePackParameters pack; uint dispatch[3]; uint mesh_dispatch[3]; uint flags; };
static_assert(sizeof(CaptureMeshParameters) == 80 && sizeof(CapturePackParameters) == 132);
static_assert(sizeof(CapturePackConfig) == 336 && sizeof(CapturePackWork) == 160);
bool capture_same_batch(device const CaptureBatch& a, device const CaptureBatch& b) {
    device const uchar* left = reinterpret_cast<device const uchar*>(&a);
    device const uchar* right = reinterpret_cast<device const uchar*>(&b);
    for (uint byte = 0u; byte < sizeof(CaptureBatch); ++byte)
        if (left[byte] != right[byte]) return false;
    return true;
}
kernel void metal4_prepare_capture_pack(device const CaptureBatch& batch [[buffer(0)]],
    device const CaptureSnapshotHeader& current [[buffer(1)]],
    device const CaptureSnapshotHeader& previous [[buffer(2)]],
    device CapturePackWork& work [[buffer(3)]], constant CapturePackConfig& p [[buffer(4)]],
    device atomic_uint& errors [[buffer(6)]]) {
    CapturePackWork result = {};
    if (!capture_batch_empty(batch)) {
        uint failure = batch.error ? batch.error : current.error;
        const uint primitive = p.snapshot.inputs.capture.primitive_vertices;
        const uint output_vertices = primitive == 1u ? 3u : 6u;
        const uint per_job = p.pack.mesh.mesh_vertices / output_vertices;
        if (!failure && (p.reserved || !primitive || primitive > 4u || !per_job ||
            p.pack.mesh.mesh_vertices > 256u || p.snapshot.vertex_words < 5u ||
            p.snapshot.has_previous > 1u || p.snapshot.same_context > 1u ||
            p.snapshot.inputs.restart_enabled > 1u || p.pack.first ||
            p.pack.vertex_step != primitive ||
            p.pack.input_vertices != (primitive == 1u ? 3u : primitive == 2u ? 4u : primitive) ||
            current.valid != 1u || current.reserved || !capture_same_batch(batch, current.batch) ||
            !capture_input_valid(batch, p.snapshot.inputs) ||
            p.pack.mesh.vertex_words != p.snapshot.vertex_words ||
            p.pack.source_words != p.snapshot.source_words ||
            batch.draw.active_work_items > p.pack.count ||
            ulong(batch.draw.work_items) * p.snapshot.vertex_words > p.snapshot.source_words))
            failure = 0x40000u;
        if (!failure && !batch.first_capture && (!p.snapshot.has_previous ||
            !p.snapshot.same_context || !capture_snapshot_contiguous(batch, previous, p.snapshot)))
            failure = 0x10000u;
        if (!failure) {
            const uint count = batch.draw.active_work_items;
            const uint jobs = (count + per_job - 1u) / per_job;
            if (jobs > p.pack.mesh.jobs) failure = 0x40000u;
            else {
                result.pack = p.pack;
                result.pack.count = count;
                result.pack.vertices_per_instance = batch.draw.vertices;
                result.pack.quads_per_instance = batch.draw.instances;
                result.pack.carry_vertices = batch.draw.first_vertex;
                result.pack.mesh.invocations = result.pack.mesh.jobs = jobs;
                const uint width = min(jobs, 65535u);
                for (uint component = 0u; component < 3u; ++component) {
                    const uint value = component == 0u ? width : component == 1u ? (jobs + width - 1u) / width : 1u;
                    result.dispatch[component] = result.mesh_dispatch[component] = value;
                }
                result.flags = 1u | (p.snapshot.inputs.restart_enabled << 1u);
            }
        }
        if (failure) atomic_fetch_or_explicit(&errors, failure, memory_order_relaxed);
    }
    work = result;
}
kernel void metal4_prepare_capture_inputs(device const CaptureBatch& batch [[buffer(0)]],
                                         device uint* storage [[buffer(1)]],
                                         constant CaptureInputParameters& p [[buffer(2)]],
                                         device CaptureRestart* restart [[buffer(3)]],
                                         device atomic_uint& errors [[buffer(6)]],
                                         device const ulong* root [[buffer(7)]],
                                         uint item [[thread_position_in_grid]]) {
    if (item >= p.capture.window.capture_capacity) return;
    const bool valid = capture_input_valid(batch, p);
    if (valid && item >= batch.draw.work_items) return;
    if (!valid) atomic_fetch_or_explicit(&errors, batch.error ? batch.error : 0x40000u, memory_order_relaxed);
    CaptureInput result = {};
    for (uint scalar = 0; scalar < 16u; ++scalar) result.system_sgprs[scalar] = p.system_sgprs[scalar];
    uint state = 0u;
    if (valid && item < batch.draw.active_work_items) {
        const CaptureDraw draw = batch.draw;
        const uint instance = draw.first_instance + item / draw.vertices +
                              (p.instance_sgpr_offset ? 0u : batch.first_instance);
        const uint index = item % draw.vertices;
        uint value = draw.index_width ? 0u : draw.first_vertex + index;
        bool mapped = true;
        if (draw.index_width && index < draw.index_count) {
            ulong address = p.index_address + batch.index_byte_offset + ulong(index) * draw.index_width;
            if (draw.index_width == 2u && draw.swap_mode >= 2u) address ^= 2ul;
            mapped = capture_read_index(root, address, draw.index_width, value);
            if (mapped) {
                if (draw.swap_mode == 1u || (draw.index_width == 2u && draw.swap_mode == 2u))
                    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
                else if (draw.swap_mode == 2u && draw.index_width == 4u)
                    value = ((value & 0xffu) << 24) | ((value & 0xff00u) << 8) |
                            ((value >> 8) & 0xff00u) | (value >> 24);
                else if (draw.swap_mode == 3u && draw.index_width == 4u) value = (value << 16) | (value >> 16);
            } else atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
        }
        if (mapped) {
            const bool marker = p.restart_enabled && value == p.restart_index;
            state = marker ? 2u : 1u;
            result.vgprs[0] = marker ? 0u : value + (p.vertex_sgpr_offset ? 0u : batch.base_vertex);
            result.vgprs[1] = draw.step_rate_0 ? instance / draw.step_rate_0 : 0u;
            result.vgprs[2] = draw.step_rate_1 ? instance / draw.step_rate_1 : 0u;
            result.vgprs[3] = instance;
            result.active = marker ? 0u : 1u;
        }
    }
    device CaptureInput* inputs = reinterpret_cast<device CaptureInput*>(storage + 6u);
    inputs[item] = result;
    if (p.restart_enabled) restart[item] = CaptureRestart{0u, state};
    if (!item) {
        for (uint component = 0; component < 4u; ++component)
            storage[component] = as_type<uint>(p.viewport[component]);
        storage[4] = batch.base_vertex;
        storage[5] = batch.first_instance;
    }
}
)metal";
}

IndirectDraws::IndirectDraws(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

id<MTLIndirectCommandBuffer> IndirectDraws::Obtain(u32 max_count, bool indexed) {
    auto& slot = slots[indexed];
    if (slot.leased) {
        throw std::runtime_error("Unconsumed native Metal indirect commands");
    }
    if (slot.commands && slot.capacity >= max_count) {
        slot.leased = true;
        scheduler.UseResource(slot.commands);
        return slot.commands;
    }
    auto descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes =
        indexed ? MTLIndirectCommandTypeDrawIndexed : MTLIndirectCommandTypeDraw;
    descriptor.inheritPipelineState = YES;
    descriptor.inheritBuffers = YES;
    const u32 capacity = std::bit_ceil(max_count);
    auto buffer =
        [instance.GetDevice() newIndirectCommandBufferWithDescriptor:descriptor
                                                     maxCommandCount:capacity
                                                             options:MTLResourceStorageModePrivate];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate native Metal indirect command buffer");
    }
    buffer.label = @"shadPS4:NativeIndirectDraws";
    slot = {buffer, capacity, true};
    scheduler.UseResource(buffer);
    return buffer;
}

void IndirectDraws::ReleaseAfterDraw(const Batch& batch) {
    for (auto& slot : slots) {
        if (slot.commands != batch.commands)
            continue;
        if (!slot.leased) {
            throw std::runtime_error("Invalid native Metal indirect command release");
        }
        slot.leased = false;
        return;
    }
    throw std::runtime_error("Unknown native Metal indirect command release");
}

id<MTLComputePipelineState> IndirectDraws::Pipeline(bool indexed, MTLIndexType index_type) {
    const u32 key = !indexed ? 0 : (index_type == MTLIndexTypeUInt16 ? 1 : 2);
    if (const auto found = pipelines.find(key); found != pipelines.end()) {
        return found->second;
    }
    if (!library) {
        library = compiler.CompileLibrary(IndirectSource, "shadPS4:IndirectDrawEncoding");
    }
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = !indexed ? @"metal4_encode_draws"
                             : (index_type == MTLIndexTypeUInt16 ? @"metal4_encode_indexed16"
                                                                 : @"metal4_encode_indexed32");
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
    auto pipeline = compiler.CompileComputePipeline(descriptor);
    pipelines.emplace(key, pipeline);
    return pipeline;
}

BufferRange IndirectDraws::PrepareErrorBuffer() {
    if (errors.empty() || errors.back().tick != scheduler.CurrentTick()) {
        auto allocation = staging.AllocateReadback(sizeof(u32), alignof(u32));
        commands.FillBuffer(
            {allocation.GetBuffer(), allocation.GetOffset(), allocation.SizeBytes()}, 0);
        errors.push_back({std::move(allocation), scheduler.CurrentTick()});
    }
    const auto& allocation = errors.back().allocation;
    return {allocation.GetBuffer(), allocation.GetOffset(), allocation.SizeBytes()};
}

id<MTLComputePipelineState> IndirectDraws::RasterPipeline(bool capture) {
    const u32 key = capture ? 4 : 3;
    if (const auto found = pipelines.find(key); found != pipelines.end()) {
        return found->second;
    }
    if (!library) {
        library = compiler.CompileLibrary(IndirectSource, "shadPS4:IndirectDrawEncoding");
    }
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = capture ? @"metal4_next_raster_capture" : @"metal4_prepare_raster_window";
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(capture ? 1 : 64, 1, 1);
    descriptor.maxTotalThreadsPerThreadgroup = capture ? 1 : 64;
    auto pipeline = compiler.CompileComputePipeline(descriptor);
    pipelines.emplace(key, pipeline);
    return pipeline;
}

id<MTLComputePipelineState> IndirectDraws::RasterInputPipeline(bool restart) {
    const u32 key = restart ? 6 : 5;
    if (const auto found = pipelines.find(key); found != pipelines.end()) {
        return found->second;
    }
    if (!library) {
        library = compiler.CompileLibrary(IndirectSource, "shadPS4:IndirectDrawEncoding");
    }
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = restart ? @"metal4_prepare_capture_restart" : @"metal4_prepare_capture_inputs";
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(restart ? 1 : 64, 1, 1);
    descriptor.maxTotalThreadsPerThreadgroup = restart ? 1 : 64;
    auto pipeline = compiler.CompileComputePipeline(descriptor);
    pipelines.emplace(key, pipeline);
    return pipeline;
}

id<MTLComputePipelineState> IndirectDraws::RasterSnapshotPipeline() {
    constexpr u32 key = 7;
    if (const auto found = pipelines.find(key); found != pipelines.end())
        return found->second;
    if (!library)
        library = compiler.CompileLibrary(IndirectSource, "shadPS4:IndirectDrawEncoding");
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"metal4_snapshot_capture";
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
    descriptor.maxTotalThreadsPerThreadgroup = 32;
    auto pipeline = compiler.CompileComputePipeline(descriptor);
    pipelines.emplace(key, pipeline);
    return pipeline;
}

id<MTLComputePipelineState> IndirectDraws::RasterPackPipeline() {
    constexpr u32 key = 8;
    if (const auto found = pipelines.find(key); found != pipelines.end())
        return found->second;
    if (!library)
        library = compiler.CompileLibrary(IndirectSource, "shadPS4:IndirectDrawEncoding");
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"metal4_prepare_capture_pack";
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    descriptor.maxTotalThreadsPerThreadgroup = 1;
    auto pipeline = compiler.CompileComputePipeline(descriptor);
    pipelines.emplace(key, pipeline);
    return pipeline;
}

IndirectDraws::RasterCapturePack IndirectDraws::PrepareRasterPack(
    const RasterCaptureInputs& inputs, const RasterCaptureState& current,
    const RasterCaptureState* previous, RasterCapturePackConfig config, bool same_layout) {
    const auto& p = inputs.layout.parameters;
    const u32 words = config.pack.mesh.vertex_words;
    const u32 capacity = p.capture.window.capture_capacity;
    const auto expected = RasterCaptureMeshLayout(capacity, words, config.pack.mesh.mesh_vertices,
                                                  p.capture.primitive_vertices);
    const auto valid = [&](const BufferRange& range, u64 bytes, u64 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() && range.size == bytes &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    if (inputs.capture.owner != this || current.owner != this ||
        current.sequence != inputs.capture.sequence || current.vertex_words != words ||
        std::memcmp(&current.context, &p, sizeof(p)) || config.reserved ||
        config.pack.mesh != expected || config.pack.first || config.pack.count != capacity ||
        config.pack.vertex_step != p.capture.primitive_vertices ||
        config.pack.input_vertices != (p.capture.primitive_vertices == 1 ? 3u
                                       : p.capture.primitive_vertices == 2
                                           ? 4u
                                           : p.capture.primitive_vertices) ||
        config.pack.source_words != u64{capacity} * words || config.pack.last_provoking > 1 ||
        (config.pack.viewport_word &&
         (!config.pack.viewport_count || config.pack.viewport_count > 16 ||
          config.pack.viewport_word >= words)) ||
        config.pack.position_words[0] >= words || config.pack.position_words[1] >= words ||
        (p.capture.primitive_vertices == 3 &&
         (!config.pack.position_words[0] || !config.pack.position_words[1])) ||
        capacity > RasterCaptureMeshCapacity(words, expected.mesh_vertices,
                                             p.capture.primitive_vertices,
                                             p.capture.window.index_width, p.restart_enabled,
                                             instance.GetDevice().maxBufferLength) ||
        !valid(current.GetRange(), RasterCaptureStateSizeBytes(words),
               alignof(RasterCaptureSnapshotHeader)) ||
        (previous && previous->owner != this)) {
        throw std::runtime_error(
            "Native raster capture pack differs from its snapshot or capacity");
    }
    const bool compatible =
        previous && same_layout && previous->sequence == inputs.capture.sequence &&
        previous->vertex_words == words && std::memcmp(&previous->context, &p, sizeof(p)) == 0;
    if (compatible && !valid(previous->GetRange(), RasterCaptureStateSizeBytes(words),
                             alignof(RasterCaptureSnapshotHeader)))
        throw std::runtime_error("Native raster capture pack has invalid previous storage");
    config.snapshot = {p, words, config.pack.source_words, u32(previous != nullptr),
                       u32(compatible)};
    auto upload = staging.Allocate(sizeof(config), alignof(RasterCapturePackConfig));
    std::memcpy(upload.UploadData().data(), &config, sizeof(config));
    auto storage = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(sizeof(RasterCapturePackWork), alignof(RasterCapturePackWork)));
    RasterCapturePack result{storage, compatible};
    ResourceBindings bindings{instance, 7, 0, 0};
    const auto batch = inputs.capture.GetRange();
    const auto state = current.GetRange();
    const auto before =
        compatible ? previous->GetRange()
                   : BufferRange{upload.GetBuffer(), upload.GetOffset(), upload.SizeBytes()};
    bindings.SetBuffer(0, batch.buffer, batch.offset);
    bindings.SetBuffer(1, state.buffer, state.offset);
    bindings.SetBuffer(2, before.buffer, before.offset);
    bindings.SetBuffer(3, storage->GetBuffer(), storage->GetOffset());
    bindings.SetBuffer(4, upload.GetBuffer(), upload.GetOffset());
    const auto errors = PrepareErrorBuffer();
    bindings.SetBuffer(6, errors.buffer, errors.offset);
    inputs.Retain(bindings);
    current.Retain(bindings);
    if (compatible)
        previous->Retain(bindings);
    result.Retain(bindings);
    bindings.RetainUploadAllocation(std::move(upload));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(RasterPackPipeline(), bindings);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return result;
}

IndirectDraws::RasterCaptureState IndirectDraws::SnapshotRasterCapture(
    const RasterCaptureInputs& inputs, std::shared_ptr<StagingBufferPool::Allocation> vertices,
    u32 vertex_words, const RasterCaptureState* previous, bool same_layout) {
    const auto& p = inputs.layout.parameters;
    const u64 state_bytes = RasterCaptureStateSizeBytes(vertex_words);
    const u64 words = u64{p.capture.window.capture_capacity} * vertex_words;
    const auto valid = [&](const BufferRange& range, u64 bytes, u64 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() && range.size == bytes &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    const u64 restart_bytes = p.restart_enabled ? inputs.layout.restart_capacity.SizeBytes() +
                                                      sizeof(RasterRestartDispatchParameters)
                                                : 0;
    if (inputs.capture.owner != this || !inputs.capture.sequence || !vertices ||
        words > std::numeric_limits<u32>::max() ||
        !valid({vertices->GetBuffer(), vertices->GetOffset(), vertices->SizeBytes()},
               words * sizeof(u32), alignof(u32)) ||
        !valid(inputs.capture.GetRange(), sizeof(RasterCaptureBatch),
               alignof(RasterCaptureBatch)) ||
        inputs.layout.input_bytes + words * sizeof(u32) + 2 * state_bytes +
                sizeof(RasterCaptureSnapshotParameters) + restart_bytes >
            std::min(u64{32} << 20, u64{instance.GetDevice().maxBufferLength}) ||
        (previous && previous->owner != this)) {
        throw std::runtime_error("Native raster snapshot differs from its capture or capacity");
    }
    const bool same_context = previous && same_layout &&
                              previous->sequence == inputs.capture.sequence &&
                              previous->vertex_words == vertex_words &&
                              std::memcmp(&previous->context, &p, sizeof(p)) == 0;
    if ((same_context &&
         !valid(previous->GetRange(), state_bytes, alignof(RasterCaptureSnapshotHeader))) ||
        (p.restart_enabled &&
         !valid(inputs.RestartRecords(), inputs.layout.restart_bytes, alignof(u32)))) {
        throw std::runtime_error("Native raster snapshot continuation storage is invalid");
    }
    const auto pipeline = RasterSnapshotPipeline();
    auto storage = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(state_bytes, alignof(RasterCaptureSnapshotHeader)));
    RasterCaptureState result{*this, inputs, vertex_words, storage};
    const RasterCaptureSnapshotParameters parameters{p, vertex_words, u32(words),
                                                     u32(previous != nullptr), u32(same_context)};
    auto upload = staging.Allocate(sizeof(parameters), alignof(RasterCaptureSnapshotParameters));
    std::memcpy(upload.UploadData().data(), &parameters, sizeof(parameters));
    ResourceBindings bindings{instance, 7, 0, 0};
    const auto batch = inputs.capture.GetRange();
    bindings.SetBuffer(0, batch.buffer, batch.offset);
    bindings.SetBuffer(1, vertices->GetBuffer(), vertices->GetOffset());
    bindings.SetBuffer(2, upload.GetBuffer(), upload.GetOffset());
    bindings.SetBuffer(3, storage->GetBuffer(), storage->GetOffset());
    const auto before =
        same_context ? previous->GetRange()
                     : BufferRange{upload.GetBuffer(), upload.GetOffset(), upload.SizeBytes()};
    bindings.SetBuffer(4, before.buffer, before.offset);
    const auto metadata = p.restart_enabled ? inputs.RestartRecords() : before;
    bindings.SetBuffer(5, metadata.buffer, metadata.offset);
    const auto errors = PrepareErrorBuffer();
    bindings.SetBuffer(6, errors.buffer, errors.offset);
    inputs.Retain(bindings);
    result.Retain(bindings);
    if (same_context)
        previous->Retain(bindings);
    bindings.RetainUploadAllocation(vertices);
    bindings.RetainUploadAllocation(std::move(upload));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(pipeline, bindings);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return result;
}

IndirectDraws::RasterCount IndirectDraws::SnapshotRasterCount(const BufferRange& count) {
    if (!count.buffer || count.buffer.device != instance.GetDevice() ||
        count.offset % sizeof(u32) || count.size != sizeof(u32) ||
        count.offset > count.buffer.length || count.size > count.buffer.length - count.offset) {
        throw std::runtime_error("Native raster count snapshot requires its mapped count word");
    }
    auto storage = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(sizeof(u32), alignof(u32)));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.CopyBuffer(count, {storage->GetBuffer(), storage->GetOffset(), storage->SizeBytes()},
                        sizeof(u32));
    scheduler.RetainLifetime(storage);
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageDispatch);
    return {*this, std::move(storage)};
}

IndirectDraws::RasterWindow IndirectDraws::PrepareRasterWindow(
    const BufferRange& window_arguments, const BufferRange& count, u32 max_count, u32 first_draw,
    u32 record_count, u32 stride, u32 index_width, u32 index_capacity, u32 capture_capacity) {
    const auto layout = BuildRasterIndirectWindow(max_count, first_draw, record_count, stride,
                                                  index_width, index_capacity, capture_capacity);
    return PrepareRasterWindowData(window_arguments, count, layout, nullptr, 0, 0);
}

IndirectDraws::RasterWindow IndirectDraws::PrepareRasterWindowPaged(
    const GuestPageTableSnapshot& pages, VAddr window_argument_address, VAddr count_address,
    u32 max_count, u32 first_draw, u32 record_count, u32 stride, u32 index_width,
    u32 index_capacity, u32 capture_capacity, const RasterCount* count_snapshot) {
    const auto layout = BuildRasterIndirectWindow(max_count, first_draw, record_count, stride,
                                                  index_width, index_capacity, capture_capacity);
    return PrepareRasterWindowData({}, {}, layout, &pages, window_argument_address, count_address,
                                   count_snapshot);
}

IndirectDraws::RasterWindow IndirectDraws::PrepareRasterWindowData(
    const BufferRange& window_arguments, const BufferRange& count,
    const RasterIndirectWindowLayout& layout, const GuestPageTableSnapshot* pages,
    VAddr window_argument_address, VAddr count_address, const RasterCount* count_snapshot) {
    auto parameters = layout.parameters;
    if (!parameters.record_count) {
        return {*this, parameters, nullptr, nullptr};
    }
    const auto valid = [&](const BufferRange& range, u64 bytes, u64 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() && range.size >= bytes &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    const auto valid_address = [](VAddr address, u64 bytes) {
        return address < GuestPageTableLayout::AddressSpaceSize && address % 4 == 0 &&
               bytes <= GuestPageTableLayout::AddressSpaceSize - address;
    };
    const auto root = pages ? pages->GetRoot() : BufferRange{};
    const BufferRange frozen_count{
        count_snapshot && count_snapshot->storage ? count_snapshot->storage->GetBuffer() : nil,
        count_snapshot && count_snapshot->storage ? count_snapshot->storage->GetOffset() : 0,
        count_snapshot && count_snapshot->storage ? count_snapshot->storage->SizeBytes() : 0};
    if ((count_snapshot &&
         (!pages || !count_address || count_snapshot->owner != this ||
          !valid(frozen_count, sizeof(u32), alignof(u32)) || frozen_count.size != sizeof(u32))) ||
        layout.record_bytes > std::min(u64{32} << 20, u64{instance.GetDevice().maxBufferLength}) ||
        (pages ? (!valid_address(window_argument_address, layout.argument_bytes) ||
                  (count_address && !valid_address(count_address, sizeof(u32))) ||
                  !valid(root, u64{GuestPageTableLayout::RootEntries} * sizeof(u64), alignof(u64)))
               : (!valid(window_arguments, layout.argument_bytes, alignof(u32)) ||
                  (count.buffer && !valid(count, sizeof(u32), alignof(u32)))))) {
        throw std::runtime_error("Invalid native raster indirect argument storage");
    }
    parameters.source.argument_address = window_argument_address;
    parameters.source.count_address = count_address;
    parameters.source.paged = pages != nullptr;
    parameters.source.has_count = count_snapshot ? 2u
                                  : pages        ? u32(count_address != 0)
                                                 : u32(count.buffer != nil);
    const auto pipeline = RasterPipeline();
    auto records = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(layout.record_bytes, alignof(RasterIndirectRecord)));
    auto source_complete = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(sizeof(u32), alignof(u32)));
    auto upload = staging.Allocate(sizeof(parameters));
    std::memcpy(upload.UploadData().data(), &parameters, sizeof(parameters));
    ResourceBindings bindings{instance, 8, 0, 0};
    bindings.SetBuffer(0, pages ? upload.GetBuffer() : window_arguments.buffer,
                       pages ? upload.GetOffset() : window_arguments.offset);
    bindings.SetBuffer(1,
                       count_snapshot ? frozen_count.buffer
                       : count.buffer ? count.buffer
                                      : upload.GetBuffer(),
                       count_snapshot ? frozen_count.offset
                       : count.buffer ? count.offset
                                      : upload.GetOffset());
    bindings.SetBuffer(2, source_complete->GetBuffer(), source_complete->GetOffset());
    bindings.SetBuffer(3, records->GetBuffer(), records->GetOffset());
    bindings.SetBuffer(4, upload.GetBuffer(), upload.GetOffset());
    const auto error = PrepareErrorBuffer();
    bindings.SetBuffer(6, error.buffer, error.offset);
    bindings.SetBuffer(7, pages ? root.buffer : upload.GetBuffer(),
                       pages ? root.offset : upload.GetOffset());
    if (pages)
        pages->Retain(bindings);
    if (count_snapshot)
        bindings.RetainUploadAllocation(count_snapshot->storage);
    bindings.RetainUploadAllocation(std::move(upload));
    bindings.RetainUploadAllocation(records);
    bindings.RetainUploadAllocation(source_complete);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch | MTLStageBlit);
    commands.BindCompute(pipeline, bindings);
    scheduler.EncoderBarrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake((u64{parameters.record_count} + 63) / 64, 1, 1),
                      MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {*this, parameters, std::move(records), std::move(source_complete)};
}

IndirectDraws::RasterController IndirectDraws::CreateRasterController(
    const RasterWindow& window, u32 primitive_vertices, u32 step_rate_0, u32 step_rate_1,
    u32 index_offset, AmdGpu::IndexSwapMode swap_mode) {
    if (window.owner != this) {
        throw std::runtime_error("Native raster controller requires its own argument window");
    }
    const auto parameters = BuildRasterCaptureParameters(
        window.parameters, primitive_vertices, step_rate_0, step_rate_1, index_offset, swap_mode);
    auto cursor = std::make_shared<StagingBufferPool::Allocation>(
        staging.Allocate(sizeof(RasterCaptureCursor), alignof(RasterCaptureCursor)));
    commands.FillBuffer({cursor->GetBuffer(), cursor->GetOffset(), cursor->SizeBytes()}, 0);
    return {*this, window, parameters, std::move(cursor)};
}

IndirectDraws::RasterCapture IndirectDraws::NextRasterCapture(const RasterController& controller) {
    const auto valid = [&](const BufferRange& range, u64 bytes, u64 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() && range.size == bytes &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    if (controller.owner != this || controller.window.owner != this || !controller.cursor ||
        !valid({controller.cursor->GetBuffer(), controller.cursor->GetOffset(),
                controller.cursor->SizeBytes()},
               sizeof(RasterCaptureCursor), alignof(RasterCaptureCursor))) {
        throw std::runtime_error("Invalid native raster capture cursor ownership or storage");
    }
    const auto records = controller.window.Records();
    const auto& parameters = controller.parameters;
    if (parameters.window.record_count &&
        !valid(records, u64{parameters.window.record_count} * sizeof(RasterIndirectRecord),
               alignof(RasterIndirectRecord))) {
        throw std::runtime_error("Invalid native raster capture argument records");
    }
    const auto pipeline = RasterPipeline(true);
    auto batch = std::make_shared<StagingBufferPool::Allocation>(
        staging.Allocate(sizeof(RasterCaptureBatch), alignof(RasterCaptureBatch)));
    auto upload = staging.Allocate(sizeof(parameters));
    std::memcpy(upload.UploadData().data(), &parameters, sizeof(parameters));
    ResourceBindings bindings{instance, 7, 0, 0};
    bindings.SetBuffer(0, records.buffer ? records.buffer : upload.GetBuffer(),
                       records.buffer ? records.offset : upload.GetOffset());
    bindings.SetBuffer(1, controller.cursor->GetBuffer(), controller.cursor->GetOffset());
    bindings.SetBuffer(3, batch->GetBuffer(), batch->GetOffset());
    bindings.SetBuffer(4, upload.GetBuffer(), upload.GetOffset());
    const auto error = PrepareErrorBuffer();
    bindings.SetBuffer(6, error.buffer, error.offset);
    controller.window.Retain(bindings);
    bindings.RetainUploadAllocation(controller.cursor);
    bindings.RetainUploadAllocation(std::move(upload));
    bindings.RetainUploadAllocation(batch);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch | MTLStageBlit);
    commands.BindCompute(pipeline, bindings);
    scheduler.EncoderBarrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {*this, parameters, std::move(batch), controller.cursor};
}

u32 IndirectDraws::WaitForRasterCaptures(const RasterController& controller) {
    if (controller.owner != this || !controller.cursor || !controller.window.source_complete) {
        throw std::runtime_error("Native raster completion requires its own retained controller");
    }
    auto completion = staging.AllocateReadback(2 * sizeof(u32), alignof(u32));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.CopyBuffer({controller.cursor->GetBuffer(),
                         controller.cursor->GetOffset() + offsetof(RasterCaptureCursor, completed),
                         sizeof(u32)},
                        {completion.GetBuffer(), completion.GetOffset(), sizeof(u32)}, sizeof(u32));
    commands.CopyBuffer({controller.window.source_complete->GetBuffer(),
                         controller.window.source_complete->GetOffset(), sizeof(u32)},
                        {completion.GetBuffer(), completion.GetOffset() + sizeof(u32), sizeof(u32)},
                        sizeof(u32));
    scheduler.RetainLifetime(controller.cursor);
    scheduler.RetainLifetime(controller.window.source_complete);
    scheduler.Finish();
    CheckErrors();
    std::array<u32, 2> flags{};
    std::memcpy(flags.data(), completion.ReadbackData().data(), sizeof(flags));
    if (flags[0] > 1 || flags[1] > 1) {
        throw std::runtime_error("Native raster capture completion flags are invalid");
    }
    return flags[0] ? 1u | (flags[1] << 1) : 0u;
}

bool IndirectDraws::RunRasterCaptures(const RasterController& controller,
                                      u32 captures_per_submission,
                                      const std::function<void(const RasterCapture&)>& consume) {
    if (controller.owner != this || !controller.window.parameters.record_count ||
        !captures_per_submission || captures_per_submission > 8 || !consume) {
        throw std::runtime_error("Invalid native raster capture submission");
    }
    for (;;) {
        for (u32 capture = 0; capture < captures_per_submission; ++capture) {
            consume(NextRasterCapture(controller));
        }
        const u32 completion = WaitForRasterCaptures(controller);
        if (completion)
            return (completion & 2) != 0;
    }
}

IndirectDraws::RasterCaptureInputs IndirectDraws::PrepareRasterCaptureInputs(
    const RasterCapture& capture, const GuestPageTableSnapshot* pages, VAddr index_address,
    const std::array<float, 4>& viewport, const GeometryExportWave& system_registers,
    std::optional<u32> restart_index, const RasterIndirectRegisters& indirect_registers) {
    const auto valid = [&](const BufferRange& range, u64 bytes, u64 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() && range.size >= bytes &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    if (capture.owner != this || !capture.batch ||
        !valid(capture.GetRange(), sizeof(RasterCaptureBatch), alignof(RasterCaptureBatch))) {
        throw std::runtime_error("Native raster inputs require their own captured argument batch");
    }
    const auto layout = BuildRasterCaptureInputLayout(
        capture.parameters, index_address, viewport, system_registers, restart_index,
        instance.GetDevice().maxBufferLength, indirect_registers);
    const bool indexed = layout.parameters.capture.window.index_width != 0;
    const auto root = pages && indexed ? pages->GetRoot() : BufferRange{};
    if (indexed &&
        !valid(root, u64{GuestPageTableLayout::RootEntries} * sizeof(u64), alignof(u64))) {
        throw std::runtime_error("Native raster capture indices require a valid page table");
    }
    const auto pipeline = RasterInputPipeline();
    auto registers = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(layout.input_bytes, alignof(u32)));
    auto restart = layout.restart_bytes
                       ? std::make_shared<StagingBufferPool::Allocation>(
                             staging.AllocatePrivate(layout.restart_bytes, alignof(u32)))
                       : nullptr;
    auto restart_work =
        restart ? std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
                      sizeof(RasterRestartDispatchParameters), alignof(u32)))
                : nullptr;
    auto upload = staging.Allocate(sizeof(layout.parameters));
    std::memcpy(upload.UploadData().data(), &layout.parameters, sizeof(layout.parameters));
    ResourceBindings bindings{instance, 8, 0, 0};
    bindings.SetBuffer(0, capture.batch->GetBuffer(), capture.batch->GetOffset());
    bindings.SetBuffer(1, registers->GetBuffer(), registers->GetOffset());
    bindings.SetBuffer(2, upload.GetBuffer(), upload.GetOffset());
    bindings.SetBuffer(3, restart ? restart->GetBuffer() : upload.GetBuffer(),
                       restart ? restart->GetOffset() : upload.GetOffset());
    if (restart_work)
        bindings.SetBuffer(4, restart_work->GetBuffer(), restart_work->GetOffset());
    const auto error = PrepareErrorBuffer();
    bindings.SetBuffer(6, error.buffer, error.offset);
    bindings.SetBuffer(7, indexed ? root.buffer : upload.GetBuffer(),
                       indexed ? root.offset : upload.GetOffset());
    if (indexed)
        pages->Retain(bindings);
    capture.Retain(bindings);
    bindings.RetainUploadAllocation(registers);
    if (restart)
        bindings.RetainUploadAllocation(restart);
    if (restart_work)
        bindings.RetainUploadAllocation(restart_work);
    bindings.RetainUploadAllocation(std::move(upload));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch | MTLStageBlit);
    if (restart_work) {
        commands.BindCompute(RasterInputPipeline(true), bindings);
        scheduler.EncoderBarrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
        commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    }
    commands.BindCompute(pipeline, bindings);
    scheduler.EncoderBarrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
    commands.DispatchIndirect(capture.DispatchArguments(), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {capture, layout, std::move(registers), std::move(restart), std::move(restart_work)};
}

IndirectDraws::Batch IndirectDraws::Encode(MTLPrimitiveType primitive, const BufferRange& arguments,
                                           const BufferRange& count, u32 max_count, u32 stride,
                                           const BufferRange& indices, MTLIndexType index_type,
                                           bool indexed) {
    return EncodeCommands(primitive, arguments, count, max_count, stride, indices, index_type,
                          indexed, nullptr, 0, 0);
}

IndirectDraws::Batch IndirectDraws::EncodePaged(MTLPrimitiveType primitive,
                                                const GuestPageTableSnapshot& pages,
                                                VAddr argument_address, VAddr count_address,
                                                u32 max_count, u32 stride,
                                                const BufferRange& indices, MTLIndexType index_type,
                                                bool indexed) {
    return EncodeCommands(primitive, {}, {}, max_count, stride, indices, index_type, indexed,
                          &pages, argument_address, count_address);
}

IndirectDraws::Batch IndirectDraws::EncodeCommands(
    MTLPrimitiveType primitive, const BufferRange& arguments, const BufferRange& count,
    u32 max_count, u32 stride, const BufferRange& indices, MTLIndexType index_type, bool indexed,
    const GuestPageTableSnapshot* pages, VAddr argument_address, VAddr count_address) {
    const u32 argument_size = indexed ? sizeof(MTLDrawIndexedPrimitivesIndirectArguments)
                                      : sizeof(MTLDrawPrimitivesIndirectArguments);
    const auto valid = [&](const BufferRange& range, u64 size, u64 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() && range.size >= size &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    const auto valid_address = [](VAddr address, u64 size) {
        return address < GuestPageTableLayout::AddressSpaceSize && address % 4 == 0 &&
               size <= GuestPageTableLayout::AddressSpaceSize - address;
    };
    const u64 argument_bytes = max_count ? u64{stride} * (max_count - 1) + argument_size : 0;
    const BufferRange root = pages ? pages->GetRoot() : BufferRange{};
    if (!max_count || max_count > (u32{1} << 31) || stride < argument_size || stride % 4 ||
        (pages ? (!valid_address(argument_address, argument_bytes) ||
                  (count_address && !valid_address(count_address, sizeof(u32))) ||
                  !valid(root, u64{GuestPageTableLayout::RootEntries} * sizeof(u64), alignof(u64)))
               : (!valid(arguments, argument_bytes, 4) ||
                  (count.buffer && !valid(count, sizeof(u32), 4)))) ||
        (indexed && index_type != MTLIndexTypeUInt16 && index_type != MTLIndexTypeUInt32)) {
        throw std::runtime_error("Invalid native Metal indirect command encoding parameters");
    }
    const u32 element_size = index_type == MTLIndexTypeUInt16 ? 2 : 4;
    if (indexed && (!valid(indices, element_size, element_size) || indices.size % element_size ||
                    indices.size / element_size > std::numeric_limits<u32>::max())) {
        throw std::runtime_error("Invalid native Metal indirect index buffer");
    }
    static_assert(GuestPageTableLayout::AddressBits == 40 && GuestPageTableLayout::PageBits == 14 &&
                  GuestPageTableLayout::LeafBits == 10);
    static_assert(sizeof(MTLIndirectCommandBufferExecutionRange) == 8);
    const IndirectReadParameters parameters{max_count,
                                            stride / 4,
                                            u32(primitive),
                                            indexed ? u32(indices.size / element_size) : 0,
                                            argument_address,
                                            count_address,
                                            u32(pages != nullptr),
                                            u32(pages ? count_address != 0 : count.buffer != nil)};
    auto parameter_data = staging.Allocate(sizeof(parameters));
    std::memcpy(parameter_data.UploadData().data(), &parameters, sizeof(parameters));
    auto range = staging.Allocate(sizeof(MTLIndirectCommandBufferExecutionRange), alignof(u32));
    Batch batch{Obtain(max_count, indexed),
                {range.GetBuffer(), range.GetOffset(), range.SizeBytes()}};
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch | MTLStageBlit);
    ResourceBindings bindings{instance, 8, 0, 0};
    bindings.SetBuffer(0, pages ? parameter_data.GetBuffer() : arguments.buffer,
                       pages ? parameter_data.GetOffset() : arguments.offset);
    bindings.SetBuffer(1, count.buffer ? count.buffer : parameter_data.GetBuffer(),
                       count.buffer ? count.offset : parameter_data.GetOffset());
    static_assert(sizeof(MTLResourceID) == 8);
    const auto command_id = batch.commands.gpuResourceID;
    auto command_binding = staging.Allocate(sizeof(command_id));
    std::memcpy(command_binding.UploadData().data(), &command_id, sizeof(command_id));
    bindings.SetBuffer(2, command_binding.GetBuffer(), command_binding.GetOffset());
    const std::array<id<MTLResource>, 1> indirect_resources{batch.commands};
    bindings.SetIndirectResources(indirect_resources);
    bindings.RetainUploadAllocation(std::move(command_binding));
    bindings.SetBuffer(3, range.GetBuffer(), range.GetOffset());
    bindings.SetBuffer(4, parameter_data.GetBuffer(), parameter_data.GetOffset());
    if (indexed) {
        bindings.SetBuffer(5, indices.buffer, indices.offset);
    }
    const auto error = PrepareErrorBuffer();
    bindings.SetBuffer(6, error.buffer, error.offset);
    bindings.SetBuffer(7, pages ? root.buffer : parameter_data.GetBuffer(),
                       pages ? root.offset : parameter_data.GetOffset());
    if (pages)
        pages->Retain(bindings);
    bindings.RetainUploadAllocation(std::move(parameter_data));
    bindings.RetainUploadAllocation(std::move(range));
    commands.BindCompute(Pipeline(indexed, index_type), bindings);
    scheduler.EncoderBarrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake((u64{max_count} + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return batch;
}

void IndirectDraws::TrackErrorUse(const BufferRange& range) {
    const auto found = std::ranges::find_if(errors, [&](const auto& record) {
        return record.allocation.GetBuffer() == range.buffer &&
               record.allocation.GetOffset() == range.offset && range.size >= sizeof(u32);
    });
    if (found == errors.end()) {
        throw std::runtime_error("Metal error usage requires a retained error record");
    }
    found->tick = std::max(found->tick, scheduler.CurrentTick());
}

void IndirectDraws::CheckErrors() {
    while (!errors.empty() && scheduler.IsFree(errors.front().tick) &&
           errors.front().allocation.IsReady()) {
        u32 flags{};
        std::memcpy(&flags, errors.front().allocation.ReadbackData().data(), sizeof(flags));
        errors.pop_front();
        if (flags & 0x40000) {
            throw std::runtime_error("Native Metal capture input has an invalid argument batch");
        }
        if (flags & 0x20000) {
            throw std::runtime_error(
                "Native Metal raster capture cursor or argument record is invalid");
        }
        if (flags & 0x2000) {
            throw std::runtime_error("Native Metal viewport packing has invalid geometry records");
        }
        if (flags & 0x1000) {
            throw std::runtime_error(
                "Native Metal primitive viewport index exceeds active viewports");
        }
        if (flags & 0x400) {
            throw std::runtime_error("Native Metal direct TES has an invalid primitive marker");
        }
        if (flags & 0x800) {
            throw std::runtime_error("Native Metal direct TES vertex is outside its active patch");
        }
        if (flags & 0x100) {
            throw std::runtime_error(
                "Native Metal copy work references an invalid invocation or vertex");
        }
        if (flags & 0x200) {
            throw std::runtime_error("Native Metal copy source exceeds GS vertex capacity");
        }
        if (flags & 64) {
            throw std::runtime_error("Native Metal guest input references an unmapped page");
        }
        if (flags & 32) {
            throw std::runtime_error("Native Metal geometry has an invalid primitive sequence");
        }
        if (flags & 16) {
            throw std::runtime_error("Native Metal geometry emission exceeds vertex capacity");
        }
        if (flags & 8) {
            throw std::runtime_error("Native Metal geometry copy input exceeds ring bounds");
        }
        if (flags & 4) {
            throw std::runtime_error("Native Metal geometry worklist exceeds export vertex bounds");
        }
        if (flags & 2) {
            throw std::runtime_error(
                "Native Metal custom restart requires full-range index lowering");
        }
        if (flags) {
            throw std::runtime_error(
                "Native Metal indirect draw exceeds guest index buffer bounds");
        }
    }
}

} // namespace Metal
