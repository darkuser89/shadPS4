// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_guest_page_table.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_primitive.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
constexpr std::string_view PrimitiveSource = R"metal(
#include <metal_stdlib>
using namespace metal;
constant bool primitive_restart [[function_constant(0)]];
struct Parameters { uint primitives; uint topology; uint index_width; uint last_vertex; ulong index_address; };
struct IndirectParameters {
    uint max_count; uint stride_words; uint topology; uint index_width;
    uint last_vertex; uint source_capacity; uint output_stride;
    uint restart_index;
    ulong argument_address; ulong count_address; uint paged; uint has_count;
    ulong index_address; uint paged_indices; uint padding;
};
static_assert(sizeof(Parameters) == 24);
static_assert(sizeof(IndirectParameters) == 72);
uint primitive_count(uint topology, uint count) {
    if (topology == 1) return count;
    if (topology == 2) return count / 2;
    if (topology == 3) return count ? count - 1 : 0;
    if (topology == 18) return count > 1 ? count : 0;
    if (topology == 4) return count / 3;
    return count > 2 ? count - 2 : 0;
}
uint primitive_width(uint topology) {
    return topology == 1 ? 1 : topology < 4 || topology == 18 ? 2 : 3;
}
uint3 ordered_vertices(uint topology, uint primitive, bool last_vertex, uint count) {
    uint3 vertices;
    if (topology == 1) {
        vertices = uint3(primitive, 0, 0);
    } else if (topology == 18) {
        vertices = uint3(primitive, primitive + 1 == count ? 0 : primitive + 1, 0);
    } else if (topology == 2 || topology == 3) {
        vertices.x = topology == 2 ? primitive * 2 : primitive;
        vertices.y = vertices.x + 1;
        vertices.z = 0;
    } else if (topology == 4) {
        vertices = uint3(primitive * 3, primitive * 3 + 1, primitive * 3 + 2);
        if (last_vertex) vertices = vertices.zxy;
    } else if (topology == 5) {
        vertices = uint3(primitive + 1, primitive + 2, 0);
        if (last_vertex) vertices = vertices.yzx;
    } else {
        vertices = uint3(primitive, primitive + 1 + (primitive & 1),
                          primitive + 2 - (primitive & 1));
        if (last_vertex) vertices = (primitive & 1) ? vertices.yzx : vertices.zxy;
    }
    return vertices;
}
uint read_index(device const uchar* source, uint element, uint width) {
    if (width == 2) return reinterpret_cast<device const ushort*>(source)[element];
    if (width == 4) return reinterpret_cast<device const uint*>(source)[element];
    return element;
}
uint scan_inclusive(threadgroup uint* scan, uint value, uint lane, bool maximum) {
    scan[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < 64; offset *= 2) {
        uint own = scan[lane];
        uint previous = lane >= offset ? scan[lane - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[lane] = maximum ? max(own, previous) : own + previous;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scan[lane];
}
kernel void metal4_assemble_primitives(device const uchar* source [[buffer(0)]],
                                      device uint* indices [[buffer(1)]],
                                      constant Parameters& parameters [[buffer(2)]],
                                      uint primitive [[thread_position_in_grid]]) {
    if (primitive >= parameters.primitives) return;
    uint width = primitive_width(parameters.topology);
    uint3 vertices = ordered_vertices(parameters.topology, primitive, parameters.last_vertex,
                                       parameters.primitives);
    for (uint element = 0; element < width; ++element) {
        indices[ulong(primitive) * width + element] =
            read_index(source, vertices[element], parameters.index_width);
    }
}
bool read_argument_word(device const ulong* root, ulong address, thread uint& value) {
    if (address >= (1ul << 40) || (address & 3ul)) return false;
    ulong leaf_address = root[address >> 24];
    if (!leaf_address) return false;
    device const ulong* leaf = reinterpret_cast<device const ulong*>(leaf_address);
    ulong page = leaf[(address >> 14) & 1023ul];
    if (!page) return false;
    value = *reinterpret_cast<device const uint*>(page + (address & 16383ul));
    return true;
}
uint read_paged_index(device const ulong* root, ulong index_address, uint element, uint width,
                      device atomic_uint& errors) {
    ulong address = index_address + ulong(element) * width;
    uint word;
    if (!read_argument_word(root, address & ~3ul, word)) {
        atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
        return 0;
    }
    return width == 2 ? (word >> uint((address & 2ul) * 8ul)) & 65535u : word;
}
uint read_source_index(device const uchar* source, uint element,
                       constant IndirectParameters& parameters, device const ulong* root,
                       device atomic_uint& errors) {
    if (!parameters.paged_indices) return read_index(source, element, parameters.index_width);
    return read_paged_index(root, parameters.index_address, element, parameters.index_width, errors);
}
kernel void metal4_assemble_paged_primitives(device uint* indices [[buffer(1)]],
                                            constant Parameters& parameters [[buffer(2)]],
                                            device const ulong* root [[buffer(3)]],
                                            device atomic_uint& errors [[buffer(4)]],
                                            uint primitive [[thread_position_in_grid]]) {
    if (primitive >= parameters.primitives) return;
    uint width = primitive_width(parameters.topology);
    uint3 vertices = ordered_vertices(parameters.topology, primitive, parameters.last_vertex,
                                       parameters.primitives);
    for (uint element = 0; element < width; ++element) {
        indices[ulong(primitive) * width + element] =
            read_paged_index(root, parameters.index_address, vertices[element],
                             parameters.index_width, errors);
    }
}
kernel void metal4_finalize_assembled_draw(device uint* arguments [[buffer(0)]],
                                           constant uint* parameters [[buffer(1)]],
                                           device atomic_uint& errors [[buffer(2)]],
                                           uint lane [[thread_position_in_grid]]) {
    if (lane != 0) return;
    for (uint word = 0; word < 5; ++word) arguments[word] = parameters[word];
    if (atomic_load_explicit(&errors, memory_order_relaxed) & 64u) arguments[0] = 0;
}
kernel void metal4_assemble_indexed_indirect(device const uchar* source [[buffer(0)]],
                                           device uint* indices [[buffer(1)]],
                                           constant IndirectParameters& parameters [[buffer(2)]],
                                           device const uint* arguments [[buffer(3)]],
                                           device uint* output_arguments [[buffer(4)]],
                                           device const uint* count [[buffer(5)]],
                                           device atomic_uint& errors [[buffer(6)]],
                                           device const ulong* root [[buffer(7)]],
                                           device uint& output_count [[buffer(8)]],
                                           uint draw [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint segments[64];
    threadgroup uint emissions[64];
    threadgroup uint args[6];
    if (draw >= parameters.max_count) return;
    device uint* output = output_arguments + ulong(draw) * 5;
    if (lane == 0) {
        for (uint word = 0; word < 5; ++word) output[word] = 0;
        uint active = parameters.max_count;
        bool valid = true;
        if (parameters.has_count) {
            if (parameters.paged) valid = read_argument_word(root, parameters.count_address, active);
            else active = count[0];
        }
        if (!valid) {
            active = 0;
            atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
        }
        active = min(active, parameters.max_count);
        if (draw == 0) output_count = active;
        args[5] = draw < active;
        if (args[5]) {
            ulong offset = ulong(draw) * parameters.stride_words;
            for (uint word = 0; word < 5; ++word) {
                if (parameters.paged) {
                    uint value;
                    if (!read_argument_word(root, parameters.argument_address + (offset + word) * 4ul,
                                            value)) {
                        args[5] = 0;
                        atomic_fetch_or_explicit(&errors, 64u, memory_order_relaxed);
                        break;
                    }
                    args[word] = value;
                } else args[word] = arguments[offset + word];
            }
            if (args[5]) {
                if (!args[0] || !args[1]) args[5] = 0;
                else if (args[2] > parameters.source_capacity ||
                         args[0] > parameters.source_capacity - args[2]) {
                    args[5] = 0;
                    atomic_fetch_or_explicit(&errors, 1u, memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!args[5]) return;
    uint primitives = primitive_count(parameters.topology, args[0]);
    uint width = primitive_width(parameters.topology);
    uint start = draw * parameters.output_stride;
    if (primitive_restart) {
        uint carried_segment = 0;
        uint written = 0;
        for (ulong block = 0; block < args[0]; block += 64) {
            uint position = uint(block) + lane;
            bool valid = position < args[0];
            uint value = valid ? read_source_index(source, args[2] + position, parameters, root, errors) : 0;
            if (parameters.index_width == 4 && parameters.restart_index != 0xffffffffu) {
                scan_inclusive(segments, uint(valid && value == 0xffffffffu), lane, true);
                bool reserved_index = segments[63] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (reserved_index) {
                    if (lane == 0) atomic_fetch_or_explicit(&errors, 2u, memory_order_relaxed);
                    return;
                }
            }
            bool restart = valid && value == parameters.restart_index;
            uint segment = scan_inclusive(segments, restart ? position + 1 : carried_segment,
                                          lane, true);
            uint local = position >= segment ? position - segment : 0;
            if (parameters.topology == 18) {
                uint previous_segment = lane ? segments[lane - 1] : carried_segment;
                bool edge = valid && !restart && local >= 1;
                bool close = valid && (restart ? position - previous_segment >= 2 :
                                                 position == args[0] - 1 && local >= 1);
                uint emitted = uint(edge) + uint(close);
                uint offset = scan_inclusive(emissions, emitted, lane, false);
                if (emitted) {
                    ulong output_base = ulong(start) + ulong(written + offset - emitted) * 2;
                    if (edge) {
                        indices[output_base++] = read_source_index(
                            source, args[2] + position - 1, parameters, root, errors);
                        indices[output_base++] = value;
                    }
                    if (close) {
                        indices[output_base++] = restart ? read_source_index(
                            source, args[2] + position - 1, parameters, root, errors) : value;
                        indices[output_base] = read_source_index(
                            source, args[2] + (restart ? previous_segment : segment),
                            parameters, root, errors);
                    }
                }
                carried_segment = segments[63];
                written += emissions[63];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }
            bool emit = valid && !restart &&
                (parameters.topology == 1 ? true :
                 parameters.topology == 2 ? (local % 2 == 1) :
                 parameters.topology == 3 ? (local >= 1) :
                 parameters.topology == 4 ? (local % 3 == 2) : (local >= 2));
            uint offset = scan_inclusive(emissions, uint(emit), lane, false);
            if (emit) {
                uint primitive = parameters.topology == 1 ? local :
                                 parameters.topology == 2 ? local / 2 :
                                 parameters.topology == 3 ? local - 1 :
                                 parameters.topology == 4 ? local / 3 : local - 2;
                uint3 vertices = ordered_vertices(parameters.topology, primitive,
                                                   parameters.last_vertex, args[0]);
                for (uint element = 0; element < width; ++element) {
                    indices[ulong(start) + ulong(written + offset - 1) * width + element] =
                        read_source_index(source, args[2] + segment + vertices[element],
                                          parameters, root, errors);
                }
            }
            carried_segment = segments[63];
            written += emissions[63];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            output[0] = written * width;
            output[1] = args[1];
            output[2] = start;
            output[3] = args[3];
            output[4] = args[4];
        }
        return;
    }
    if (lane == 0) {
        output[0] = primitives * width;
        output[1] = args[1];
        output[2] = start;
        output[3] = args[3];
        output[4] = args[4];
    }
    for (uint primitive = lane; primitive < primitives; primitive += 64) {
        uint3 vertices = ordered_vertices(parameters.topology, primitive, parameters.last_vertex,
                                           args[0]);
        for (uint element = 0; element < width; ++element) {
            indices[ulong(start) + ulong(primitive) * width + element] =
                read_source_index(source, args[2] + vertices[element], parameters, root, errors);
        }
    }
}
)metal";

u32 PrimitiveCount(AmdGpu::PrimitiveType topology, u32 count) {
    using Type = AmdGpu::PrimitiveType;
    switch (topology) {
    case Type::PointList:
        return count;
    case Type::LineList:
        return count / 2;
    case Type::LineStrip:
        return count ? count - 1 : 0;
    case Type::LineLoop:
        return count > 1 ? count : 0;
    case Type::TriangleList:
        return count / 3;
    case Type::TriangleStrip:
    case Type::TriangleFan:
        return count > 2 ? count - 2 : 0;
    default:
        throw std::runtime_error("Unsupported native Metal primitive assembly");
    }
}

u32 PrimitiveWidth(AmdGpu::PrimitiveType topology, bool last_vertex) {
    if (topology == AmdGpu::PrimitiveType::PointList)
        return 1;
    const bool line = topology == AmdGpu::PrimitiveType::LineList ||
                      topology == AmdGpu::PrimitiveType::LineStrip ||
                      topology == AmdGpu::PrimitiveType::LineLoop;
    if (line && last_vertex)
        throw std::runtime_error(
            "Metal last provoking line attributes require native shader lowering");
    return line ? 2 : 3;
}

MTLPrimitiveType AssembledPrimitive(u32 width) {
    return width == 1   ? MTLPrimitiveTypePoint
           : width == 2 ? MTLPrimitiveTypeLine
                        : MTLPrimitiveTypeTriangle;
}

static_assert(u32(AmdGpu::PrimitiveType::PointList) == 1);
static_assert(u32(AmdGpu::PrimitiveType::LineList) == 2);
static_assert(u32(AmdGpu::PrimitiveType::LineStrip) == 3);
static_assert(u32(AmdGpu::PrimitiveType::TriangleList) == 4);
static_assert(u32(AmdGpu::PrimitiveType::TriangleFan) == 5);
static_assert(u32(AmdGpu::PrimitiveType::TriangleStrip) == 6);
static_assert(u32(AmdGpu::PrimitiveType::LineLoop) == 18);
} // namespace

PrimitiveAssembly::PrimitiveAssembly(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

id<MTLBuffer> PrimitiveAssembly::Obtain(u64 size, bool arguments) {
    auto& slot = slots[arguments];
    if (slot.leased) {
        throw std::runtime_error("Unconsumed native Metal primitive assembly output");
    }
    if (!slot.buffer || slot.buffer.length < size) {
        const u64 capacity =
            std::min<u64>(std::bit_ceil(size), instance.GetDevice().maxBufferLength);
        slot.buffer = instance.CreateBuffer(capacity, MTLResourceStorageModePrivate);
        slot.buffer.label = arguments ? @"shadPS4:PrimitiveArguments" : @"shadPS4:PrimitiveIndices";
    }
    slot.leased = true;
    scheduler.UseResource(slot.buffer);
    return slot.buffer;
}

void PrimitiveAssembly::ReleaseAfterDraw(const BufferRange& indices, const BufferRange& arguments) {
    const std::array ranges{indices, arguments};
    for (u32 i = 0; i < ranges.size(); ++i) {
        if (!ranges[i].buffer)
            continue;
        if (!slots[i].leased || slots[i].buffer != ranges[i].buffer) {
            throw std::runtime_error("Invalid native Metal primitive output release");
        }
        slots[i].leased = false;
    }
}

id<MTLComputePipelineState> PrimitiveAssembly::Pipeline(bool indirect, bool restart, bool paged) {
    auto& pipeline = pipelines[paged ? 3 : (indirect ? (restart ? 2 : 1) : 0)];
    if (!pipeline) {
        auto library = compiler.CompileLibrary(PrimitiveSource, "shadPS4:PrimitiveAssembly");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = paged      ? @"metal4_assemble_paged_primitives"
                        : indirect ? @"metal4_assemble_indexed_indirect"
                                   : @"metal4_assemble_primitives";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        if (indirect) {
            auto constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&restart type:MTLDataTypeBool atIndex:0];
            auto specialized = [MTL4SpecializedFunctionDescriptor new];
            specialized.functionDescriptor = function;
            specialized.constantValues = constants;
            descriptor.computeFunctionDescriptor = specialized;
        } else {
            descriptor.computeFunctionDescriptor = function;
        }
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return pipeline;
}

PrimitiveAssembly::Result PrimitiveAssembly::Assemble(AmdGpu::PrimitiveType topology, u32 count,
                                                      const BufferRange& indices,
                                                      MTLIndexType index_type, bool last_vertex) {
    return AssemblePrimitives(topology, count, indices, index_type, last_vertex, nullptr, 0, 0, {});
}

PrimitiveAssembly::Result PrimitiveAssembly::AssemblePaged(
    AmdGpu::PrimitiveType topology, u32 count, const GuestPageTableSnapshot& pages,
    VAddr index_address, u64 index_size, MTLIndexType index_type, bool last_vertex,
    const BufferRange& errors) {
    return AssemblePrimitives(topology, count, {}, index_type, last_vertex, &pages, index_address,
                              index_size, errors);
}

PrimitiveAssembly::Result PrimitiveAssembly::AssemblePrimitives(
    AmdGpu::PrimitiveType topology, u32 count, const BufferRange& indices, MTLIndexType index_type,
    bool last_vertex, const GuestPageTableSnapshot* pages, VAddr index_address, u64 index_size,
    const BufferRange& errors) {
    const u32 primitive_count = PrimitiveCount(topology, count);
    const u32 vertices = PrimitiveWidth(topology, last_vertex);
    const u64 output_count = u64{primitive_count} * vertices;
    const auto primitive = AssembledPrimitive(vertices);
    if (!output_count)
        return {{}, 0, primitive};
    if (output_count > std::numeric_limits<u32>::max() ||
        output_count * sizeof(u32) > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Native Metal primitive indices exceed device capacity");
    }
    const u32 index_width =
        !pages && !indices.buffer ? 0 : (index_type == MTLIndexTypeUInt16 ? 2 : 4);
    const auto root = pages ? pages->GetRoot() : BufferRange{};
    const auto valid = [&](const BufferRange& range, u64 minimum, u32 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() &&
               range.size >= minimum && range.offset % alignment == 0 &&
               range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    if (pages &&
        ((index_type != MTLIndexTypeUInt16 && index_type != MTLIndexTypeUInt32) ||
         index_address >= GuestPageTableLayout::AddressSpaceSize || index_address % index_width ||
         index_size % index_width || index_size < u64{count} * index_width ||
         index_size > GuestPageTableLayout::AddressSpaceSize - index_address ||
         !valid(root, u64{GuestPageTableLayout::RootEntries} * sizeof(u64), alignof(u64)) ||
         !valid(errors, sizeof(u32), alignof(u32)))) {
        throw std::runtime_error("Invalid native Metal paged primitive source indices");
    }
    if (!pages && indices.buffer &&
        ((index_type != MTLIndexTypeUInt16 && index_type != MTLIndexTypeUInt32) ||
         indices.buffer.device != instance.GetDevice() || indices.offset % index_width ||
         indices.size < u64{count} * index_width || indices.offset > indices.buffer.length ||
         indices.size > indices.buffer.length - indices.offset)) {
        throw std::runtime_error("Invalid native Metal primitive source indices");
    }
    struct Parameters {
        u32 primitives;
        u32 topology;
        u32 index_width;
        u32 last_vertex;
        u64 index_address;
    };
    static_assert(sizeof(Parameters) == 24);
    const Parameters parameters{primitive_count, u32(topology), index_width, u32(last_vertex),
                                index_address};
    auto parameter_data = staging.Allocate(sizeof(parameters));
    std::memcpy(parameter_data.UploadData().data(), &parameters, sizeof(parameters));
    auto output = Obtain(output_count * sizeof(u32));
    ResourceBindings bindings{instance, pages ? 5u : 3u, 0, 0};
    bindings.SetBuffer(0, indices.buffer ? indices.buffer : parameter_data.GetBuffer(),
                       indices.buffer ? indices.offset : parameter_data.GetOffset());
    bindings.SetBuffer(1, output);
    bindings.SetBuffer(2, parameter_data.GetBuffer(), parameter_data.GetOffset());
    if (pages) {
        bindings.SetBuffer(3, root.buffer, root.offset);
        bindings.SetBuffer(4, errors.buffer, errors.offset);
        pages->Retain(bindings);
    }
    bindings.RetainUploadAllocation(std::move(parameter_data));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(Pipeline(false, false, pages != nullptr), bindings);
    commands.Dispatch(MTLSizeMake((u64{primitive_count} + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {{output, 0, output_count * sizeof(u32)}, u32(output_count), primitive};
}

id<MTLComputePipelineState> PrimitiveAssembly::DrawArgumentPipeline() {
    if (!draw_argument_pipeline) {
        auto library = compiler.CompileLibrary(PrimitiveSource, "shadPS4:PrimitiveAssembly");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_finalize_assembled_draw";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        draw_argument_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return draw_argument_pipeline;
}

BufferRange PrimitiveAssembly::FinalizeDrawArguments(u32 index_count, u32 instance_count,
                                                     s32 base_vertex, u32 first_instance,
                                                     const BufferRange& errors) {
    if (!index_count) {
        return {};
    }
    if (!errors.buffer || errors.buffer.device != instance.GetDevice() ||
        errors.offset % alignof(u32) || errors.size < sizeof(u32) ||
        errors.offset > errors.buffer.length ||
        errors.size > errors.buffer.length - errors.offset) {
        throw std::runtime_error("Invalid native Metal primitive error buffer");
    }
    const MTLDrawIndexedPrimitivesIndirectArguments parameters{index_count, instance_count, 0,
                                                               base_vertex, first_instance};
    static_assert(sizeof(parameters) == 5 * sizeof(u32));
    auto parameter_data = staging.Allocate(sizeof(parameters));
    std::memcpy(parameter_data.UploadData().data(), &parameters, sizeof(parameters));
    auto output = Obtain(sizeof(parameters), true);
    ResourceBindings bindings{instance, 3, 0, 0};
    bindings.SetBuffer(0, output);
    bindings.SetBuffer(1, parameter_data.GetBuffer(), parameter_data.GetOffset());
    bindings.SetBuffer(2, errors.buffer, errors.offset);
    bindings.RetainUploadAllocation(std::move(parameter_data));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(DrawArgumentPipeline(), bindings);
    commands.Dispatch(MTLSizeMake(1, 1, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {output, 0, sizeof(parameters)};
}

PrimitiveAssembly::IndirectResult PrimitiveAssembly::AssembleIndexedIndirect(
    AmdGpu::PrimitiveType topology, const BufferRange& indices, MTLIndexType index_type,
    const BufferRange& arguments, const BufferRange& count, u32 max_count, u32 stride,
    bool last_vertex, const BufferRange& errors, std::optional<u32> restart_index) {
    return AssembleIndexedCommands(topology, indices, index_type, arguments, count, max_count,
                                   stride, last_vertex, errors, restart_index, nullptr, 0, 0, 0, 0);
}

PrimitiveAssembly::IndirectResult PrimitiveAssembly::AssembleIndexedPaged(
    AmdGpu::PrimitiveType topology, VAddr index_address, u64 index_size, MTLIndexType index_type,
    const GuestPageTableSnapshot& pages, VAddr argument_address, VAddr count_address, u32 max_count,
    u32 stride, bool last_vertex, const BufferRange& errors, std::optional<u32> restart_index) {
    return AssembleIndexedCommands(topology, {}, index_type, {}, {}, max_count, stride, last_vertex,
                                   errors, restart_index, &pages, argument_address, count_address,
                                   index_address, index_size);
}

PrimitiveAssembly::IndirectResult PrimitiveAssembly::AssembleIndexedCommands(
    AmdGpu::PrimitiveType topology, const BufferRange& indices, MTLIndexType index_type,
    const BufferRange& arguments, const BufferRange& count, u32 max_count, u32 stride,
    bool last_vertex, const BufferRange& errors, std::optional<u32> restart_index,
    const GuestPageTableSnapshot* pages, VAddr argument_address, VAddr count_address,
    VAddr index_address, u64 index_size) {
    const auto valid = [&](const BufferRange& range, u64 minimum, u32 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() &&
               range.offset % alignment == 0 && range.size >= minimum &&
               range.offset <= range.buffer.length &&
               range.size <= range.buffer.length - range.offset;
    };
    const auto valid_address = [](VAddr address, u64 size, u32 alignment = 4) {
        return address < GuestPageTableLayout::AddressSpaceSize && address % alignment == 0 &&
               size <= GuestPageTableLayout::AddressSpaceSize - address;
    };
    const u64 input_bytes = max_count ? u64{stride} * (max_count - 1) +
                                            sizeof(MTLDrawIndexedPrimitivesIndirectArguments)
                                      : 0;
    const auto root = pages ? pages->GetRoot() : BufferRange{};
    const bool paged_arguments = pages && !arguments.buffer;
    const bool paged_indices = pages && !indices.buffer;
    if (!max_count || (index_type != MTLIndexTypeUInt16 && index_type != MTLIndexTypeUInt32) ||
        stride < sizeof(MTLDrawIndexedPrimitivesIndirectArguments) || stride % 4 ||
        (paged_arguments ? (!valid_address(argument_address, input_bytes) ||
                            (count_address && !valid_address(count_address, sizeof(u32))))
                         : (!valid(arguments, input_bytes, 4) ||
                            (count.buffer && !valid(count, sizeof(u32), 4)))) ||
        (pages &&
         !valid(root, u64{GuestPageTableLayout::RootEntries} * sizeof(u64), alignof(u64))) ||
        !valid(errors, sizeof(u32), 4)) {
        throw std::runtime_error("Invalid native Metal indirect primitive arguments");
    }
    const u32 index_width = index_type == MTLIndexTypeUInt16 ? 2 : 4;
    const u64 source_size = paged_indices ? index_size : indices.size;
    if ((paged_indices ? (!index_size || !valid_address(index_address, index_size, index_width))
                       : !valid(indices, index_width, index_width)) ||
        source_size % index_width || source_size / index_width > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Invalid native Metal indirect primitive source indices");
    }
    const u32 capacity = u32(source_size / index_width);
    const u32 width = PrimitiveWidth(topology, last_vertex);
    const u64 output_stride = u64{PrimitiveCount(topology, capacity)} * width;
    if (output_stride > std::numeric_limits<u32>::max() / max_count) {
        throw std::runtime_error("Native Metal indirect primitive index offsets exceed 32 bits");
    }
    const u64 output_count = output_stride * max_count;
    const u64 argument_bytes = u64{max_count} * sizeof(MTLDrawIndexedPrimitivesIndirectArguments);
    const u64 argument_storage = argument_bytes + sizeof(u32);
    const u64 index_bytes = std::max<u64>(output_count, 1) * sizeof(u32);
    if (output_count > std::numeric_limits<u32>::max() ||
        index_bytes > instance.GetDevice().maxBufferLength ||
        argument_storage > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Native Metal indirect primitive storage exceeds device capacity");
    }
    struct Parameters {
        u32 max_count;
        u32 stride_words;
        u32 topology;
        u32 index_width;
        u32 last_vertex;
        u32 source_capacity;
        u32 output_stride;
        u32 restart_index;
        u64 argument_address;
        u64 count_address;
        u32 paged;
        u32 has_count;
        u64 index_address;
        u32 paged_indices;
        u32 padding;
    };
    static_assert(sizeof(Parameters) == 72);
    static_assert(GuestPageTableLayout::AddressBits == 40 && GuestPageTableLayout::PageBits == 14 &&
                  GuestPageTableLayout::LeafBits == 10);
    const Parameters parameters{max_count,
                                stride / 4,
                                u32(topology),
                                index_width,
                                u32(last_vertex),
                                capacity,
                                u32(output_stride),
                                restart_index.value_or(0),
                                argument_address,
                                count_address,
                                u32(paged_arguments),
                                u32(paged_arguments ? count_address != 0 : count.buffer != nil),
                                index_address,
                                u32(paged_indices),
                                0};
    auto parameter_data = staging.Allocate(sizeof(parameters));
    std::memcpy(parameter_data.UploadData().data(), &parameters, sizeof(parameters));
    auto output_indices = Obtain(index_bytes);
    auto output_arguments = Obtain(argument_storage, true);
    ResourceBindings bindings{instance, 9, 0, 0};
    bindings.SetBuffer(0, paged_indices ? parameter_data.GetBuffer() : indices.buffer,
                       paged_indices ? parameter_data.GetOffset() : indices.offset);
    bindings.SetBuffer(1, output_indices);
    bindings.SetBuffer(2, parameter_data.GetBuffer(), parameter_data.GetOffset());
    bindings.SetBuffer(3, paged_arguments ? parameter_data.GetBuffer() : arguments.buffer,
                       paged_arguments ? parameter_data.GetOffset() : arguments.offset);
    bindings.SetBuffer(4, output_arguments);
    bindings.SetBuffer(5, count.buffer ? count.buffer : parameter_data.GetBuffer(),
                       count.buffer ? count.offset : parameter_data.GetOffset());
    bindings.SetBuffer(6, errors.buffer, errors.offset);
    bindings.SetBuffer(7, pages ? root.buffer : parameter_data.GetBuffer(),
                       pages ? root.offset : parameter_data.GetOffset());
    bindings.SetBuffer(8, output_arguments, argument_bytes);
    if (pages)
        pages->Retain(bindings);
    bindings.RetainUploadAllocation(std::move(parameter_data));
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(Pipeline(true, restart_index.has_value()), bindings);
    commands.Dispatch(MTLSizeMake(max_count, 1, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {{output_indices, 0, index_bytes},
            {output_arguments, 0, argument_bytes},
            {output_arguments, argument_bytes, sizeof(u32)},
            AssembledPrimitive(width)};
}

PrimitiveAssembly::IndirectResult PrimitiveAssembly::AssembleIndexedRestart(
    AmdGpu::PrimitiveType topology, u32 count, u32 instance_count, s32 base_vertex,
    u32 first_instance, const BufferRange& indices, MTLIndexType index_type, bool last_vertex,
    u32 restart_index, const BufferRange& errors) {
    const MTLDrawIndexedPrimitivesIndirectArguments arguments{count, instance_count, 0, base_vertex,
                                                              first_instance};
    auto argument_data = staging.Allocate(sizeof(arguments));
    std::memcpy(argument_data.UploadData().data(), &arguments, sizeof(arguments));
    return AssembleIndexedIndirect(
        topology, indices, index_type,
        {argument_data.GetBuffer(), argument_data.GetOffset(), sizeof(arguments)}, {}, 1,
        sizeof(arguments), last_vertex, errors, restart_index);
}

PrimitiveAssembly::IndirectResult PrimitiveAssembly::AssembleIndexedRestartPaged(
    AmdGpu::PrimitiveType topology, u32 count, u32 instance_count, s32 base_vertex,
    u32 first_instance, const GuestPageTableSnapshot& pages, VAddr index_address, u64 index_size,
    MTLIndexType index_type, bool last_vertex, u32 restart_index, const BufferRange& errors) {
    const MTLDrawIndexedPrimitivesIndirectArguments arguments{count, instance_count, 0, base_vertex,
                                                              first_instance};
    auto argument_data = staging.Allocate(sizeof(arguments));
    std::memcpy(argument_data.UploadData().data(), &arguments, sizeof(arguments));
    return AssembleIndexedCommands(
        topology, {}, index_type,
        {argument_data.GetBuffer(), argument_data.GetOffset(), sizeof(arguments)}, {}, 1,
        sizeof(arguments), last_vertex, errors, restart_index, &pages, 0, 0, index_address,
        index_size);
}

} // namespace Metal
