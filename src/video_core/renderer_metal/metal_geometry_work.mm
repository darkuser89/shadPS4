// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
struct RasterInputParameters {
    RasterDrawParameters draw;
    std::array<float, 4> viewport;
    GeometryExportWave system_registers;
    u32 dynamic_inputs;
    u32 restart_enabled;
    u32 restart_index;
    u32 index_source_offset;
};
static_assert(sizeof(RasterInputParameters) == 144);
static_assert(offsetof(RasterInputParameters, index_source_offset) == 140);

constexpr std::string_view RasterInputsSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Draw {
    uint vertices; uint instances; uint work_items; uint active_work_items;
    uint step_rate_0; uint step_rate_1; uint index_width; uint index_count;
    uint index_offset; uint swap_mode;
    uint first_vertex; uint first_instance;
};
struct Parameters { Draw draw; float viewport[4]; uint system_sgprs[16]; uint dynamic_inputs; uint restart_enabled; uint restart_index; uint index_source_offset; };
struct Input { uint system_sgprs[16]; uint vgprs[4]; uint active; };
struct RestartRecord { uint begin; uint state; };
static_assert(sizeof(RestartRecord) == 8);
static_assert(sizeof(Draw) == 48 && sizeof(Parameters) == 144 && sizeof(Input) == 84);

kernel void metal4_prepare_raster_inputs(device const uchar* indices [[buffer(0)]],
                                         device uint* storage [[buffer(1)]],
                                         constant Parameters& p [[buffer(2)]],
                                         device RestartRecord* restart [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint2 grid [[threadgroups_per_grid]],
                                         uint local [[thread_index_in_threadgroup]]) {
    const ulong item = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (item >= p.draw.work_items) return;
    device Input* inputs = reinterpret_cast<device Input*>(storage + (p.dynamic_inputs ? 4u : 0u));
    Input result{};
    uint state = 0u;
    for (uint scalar = 0; scalar < 16; ++scalar)
        result.system_sgprs[scalar] = p.system_sgprs[scalar];
    if (item < p.draw.active_work_items) {
        const uint instance = p.draw.first_instance + uint(item) / p.draw.vertices;
        const uint index = uint(item) % p.draw.vertices;
        uint value = p.draw.index_width ? 0 : p.draw.first_vertex + index;
        if (p.draw.index_width && index < p.draw.index_count) {
            ulong offset = ulong(index) * p.draw.index_width;
            if (p.draw.index_width == 2 && p.draw.swap_mode >= 2)
                offset = (offset + p.index_source_offset) ^ 2ul;
            value = p.draw.index_width == 2 ? *reinterpret_cast<device const ushort*>(indices + offset)
                                            : *reinterpret_cast<device const uint*>(indices + offset);
            if (p.draw.swap_mode == 1 || (p.draw.index_width == 2 && p.draw.swap_mode == 2))
                value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
            else if (p.draw.swap_mode == 2 && p.draw.index_width == 4)
                value = ((value & 0xffu) << 24) | ((value & 0xff00u) << 8) |
                        ((value >> 8) & 0xff00u) | (value >> 24);
            else if (p.draw.swap_mode == 3 && p.draw.index_width == 4)
                value = (value << 16) | (value >> 16);
        }
        const bool is_restart = p.restart_enabled && value == p.restart_index;
        state = is_restart ? 2u : 1u;
        result.vgprs[0] = is_restart ? 0u : value + p.draw.index_offset;
        result.vgprs[1] = p.draw.step_rate_0 ? instance / p.draw.step_rate_0 : 0;
        result.vgprs[2] = p.draw.step_rate_1 ? instance / p.draw.step_rate_1 : 0;
        result.vgprs[3] = instance;
        result.active = is_restart ? 0u : 1u;
    }
    inputs[item] = result;
    if (p.restart_enabled) restart[item] = RestartRecord{0u, state};
    if (!item) {
        device uint* viewport = p.dynamic_inputs ? storage
            : reinterpret_cast<device uint*>(inputs + p.draw.work_items);
        for (uint component = 0; component < 4; ++component)
            viewport[component] = as_type<uint>(p.viewport[component]);
    }
}
)metal";

constexpr std::string_view RasterRestartSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters { uint work_items; uint blocks; uint superblocks; uint restart_index; };
struct RestartRecord { uint begin; uint state; };
static_assert(sizeof(Parameters) == 16 && sizeof(RestartRecord) == 8);
uint restart_scan(threadgroup uint* values, uint value, uint lane) {
    values[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 64; stride *= 2) {
        const uint own = values[lane];
        const uint previous = lane >= stride ? values[lane - stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[lane] = max(own, previous);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return values[lane];
}
kernel void metal4_scan_restart_vertices(
    device RestartRecord* records [[buffer(0)]], device uint* blocks [[buffer(1)]],
    constant Parameters& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong block = ulong(group.y) * grid.x + group.x;
    if (block >= p.blocks) return;
    const uint item = uint(block) * 64u + lane;
    threadgroup uint values[64];
    const uint prefix = restart_scan(values, records[item].state == 2u ? item + 1u : 0u, lane);
    records[item].begin = prefix;
    if (lane == 63u) blocks[block] = prefix;
}
kernel void metal4_scan_restart_blocks(
    device uint* blocks [[buffer(1)]],
    device uint* superblocks [[buffer(2)]], constant Parameters& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong superblock = ulong(group.y) * grid.x + group.x;
    if (superblock >= p.superblocks) return;
    const uint item = uint(superblock) * 64u + lane;
    threadgroup uint values[64];
    const uint prefix = restart_scan(values, item < p.blocks ? blocks[item] : 0u, lane);
    if (item < p.blocks) blocks[item] = prefix;
    if (lane == 63u) superblocks[superblock] = prefix;
}
kernel void metal4_scan_restart_superblocks(
    device uint* superblocks [[buffer(2)]], constant Parameters& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    if (group.x || group.y) return;
    threadgroup uint values[64];
    uint previous = 0u;
    for (ulong first = 0; first < p.superblocks; first += 64u) {
        const ulong item = first + lane;
        const uint prefix = restart_scan(values, max(previous, item < p.superblocks ? superblocks[item] : 0u), lane);
        if (item < p.superblocks) superblocks[item] = prefix;
        previous = values[63];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void metal4_resolve_restart_prefix(
    device RestartRecord* records [[buffer(0)]], device uint* blocks [[buffer(1)]],
    device uint* superblocks [[buffer(2)]], constant Parameters& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong block = ulong(group.y) * grid.x + group.x;
    if (block >= p.blocks) return;
    const uint item = uint(block) * 64u + lane;
    const uint superblock = uint(block) / 64u;
    uint prefix = records[item].begin;
    if (block % 64u) prefix = max(prefix, blocks[block - 1u]);
    if (superblock) prefix = max(prefix, superblocks[superblock - 1u]);
    records[item].begin = prefix;
}
)metal";

constexpr std::string_view GeometryWorkSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint input_vertices; uint instances; uint gs_invocations; uint primitives;
    uint export_items; uint geometry_items; uint export_vertices; uint geometry_invocations;
    uint step_rate_0; uint step_rate_1; uint topology; uint index_width;
    uint index_offset;
};
struct ExportWave { uint system_sgprs[16]; };
struct KernelWave { uint ring_offset; uint wave_id; };
struct ExportInput { uint system_sgprs[16]; uint vgprs[4]; uint active; };
struct WorkItem {
    uint ring_offset; uint wave_id; uint vertices[6]; uint primitive_id; uint invocation_id; uint active;
};
static_assert(sizeof(Parameters) == 52);
static_assert(sizeof(ExportWave) == 64 && sizeof(KernelWave) == 8);
static_assert(sizeof(ExportInput) == 84 && sizeof(WorkItem) == 44);

uint geometry_input_vertices(uint topology, uint primitive, uint primitives, thread uint* slots) {
    if (topology == 1) { slots[0] = primitive; return 1; }
    if (topology == 2 || topology == 3) {
        slots[0] = topology == 2 ? primitive * 2 : primitive;
        slots[1] = slots[0] + 1;
        return 2;
    }
    if (topology == 4) {
        for (uint i = 0; i < 3; ++i) slots[i] = primitive * 3 + i;
        return 3;
    }
    if (topology == 5) {
        slots[0] = primitive + 1; slots[1] = primitive + 2; slots[2] = 0;
        return 3;
    }
    if (topology == 6) {
        slots[0] = primitive;
        slots[1] = primitive + 1 + (primitive & 1);
        slots[2] = primitive + 2 - (primitive & 1);
        return 3;
    }
    if (topology == 10 || topology == 11) {
        for (uint i = 0; i < 4; ++i) slots[i] = (topology == 10 ? primitive * 4 : primitive) + i;
        return 4;
    }
    if (topology == 12) {
        for (uint i = 0; i < 6; ++i) slots[i] = primitive * 6 + i;
        return 6;
    }
    uint canonical[6];
    const uint first = primitive * 2;
    canonical[0] = first;
    canonical[3] = first + (primitive + 1 == primitives ? 5 : 6);
    if (!primitive) {
        canonical[1] = 1; canonical[2] = 2; canonical[4] = 4; canonical[5] = 3;
    } else if (primitive & 1) {
        canonical[1] = first + 3; canonical[2] = first + 4;
        canonical[4] = first + 2; canonical[5] = first - 2;
    } else {
        canonical[1] = first - 2; canonical[2] = first + 2;
        canonical[4] = first + 4; canonical[5] = first + 3;
    }
    for (uint i = 0; i < 6; ++i) slots[i] = canonical[(i + ((primitive & 1) ? 2 : 0)) % 6];
    return 6;
}

kernel void metal4_prepare_geometry_work(device const uchar* indices [[buffer(0)]],
                                         device const ExportWave* export_waves [[buffer(1)]],
                                         device const KernelWave* geometry_waves [[buffer(2)]],
                                         device ExportInput* exports [[buffer(3)]],
                                         device WorkItem* work [[buffer(4)]],
                                         constant Parameters& p [[buffer(5)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint2 grid [[threadgroups_per_grid]],
                                         uint local [[thread_index_in_threadgroup]]) {
    const ulong item = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (item < p.export_items) {
        ExportInput result{};
        for (uint i = 0; i < 16; ++i) result.system_sgprs[i] = export_waves[item / 64].system_sgprs[i];
        if (item < p.export_vertices) {
            result.active = 1;
            const uint instance = uint(item) / p.input_vertices;
            const uint index = uint(item) % p.input_vertices;
            result.vgprs[0] = p.index_width == 2 ? reinterpret_cast<device const ushort*>(indices)[index] :
                              p.index_width == 4 ? reinterpret_cast<device const uint*>(indices)[index] : index;
            result.vgprs[0] += p.index_offset;
            result.vgprs[1] = p.step_rate_0 ? instance / p.step_rate_0 : 0;
            result.vgprs[2] = p.step_rate_1 ? instance / p.step_rate_1 : 0;
            result.vgprs[3] = instance;
        }
        exports[item] = result;
    }
    if (item < p.geometry_items) {
        WorkItem result{};
        const KernelWave wave = geometry_waves[item / 64];
        result.ring_offset = wave.ring_offset;
        result.wave_id = wave.wave_id;
        if (item < p.geometry_invocations) {
            result.active = 1;
            const uint primitive_instance = uint(item) / p.gs_invocations;
            const uint instance = primitive_instance / p.primitives;
            const uint primitive = primitive_instance % p.primitives;
            uint slots[6]{};
            const uint width = geometry_input_vertices(p.topology, primitive, p.primitives, slots);
            for (uint i = 0; i < width; ++i) result.vertices[i] = instance * p.input_vertices + slots[i];
            result.primitive_id = primitive;
            result.invocation_id = uint(item) % p.gs_invocations;
        }
        work[item] = result;
    }
}
)metal";
} // namespace

id<MTLComputePipelineState> GeometryAssembly::RasterInputsPipeline() {
    if (!raster_inputs_pipeline) {
        auto library = compiler.CompileLibrary(RasterInputsSource, "shadPS4:RasterInputs");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_prepare_raster_inputs";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        raster_inputs_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return raster_inputs_pipeline;
}

id<MTLComputePipelineState> GeometryAssembly::RasterRestartPipeline(u32 stage) {
    if (stage >= raster_restart_pipelines.size()) {
        throw std::runtime_error("Invalid native raster restart scan stage");
    }
    if (!raster_restart_pipelines[stage]) {
        auto library = compiler.CompileLibrary(RasterRestartSource, "shadPS4:RasterRestart");
        const std::array names{@"metal4_scan_restart_vertices", @"metal4_scan_restart_blocks",
                               @"metal4_scan_restart_superblocks",
                               @"metal4_resolve_restart_prefix"};
        for (u32 index = 0; index < names.size(); ++index) {
            auto function = [MTL4LibraryFunctionDescriptor new];
            function.library = library;
            function.name = names[index];
            auto descriptor = [MTL4ComputePipelineDescriptor new];
            descriptor.computeFunctionDescriptor = function;
            descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
            descriptor.maxTotalThreadsPerThreadgroup = 64;
            raster_restart_pipelines[index] = compiler.CompileComputePipeline(descriptor);
        }
    }
    return raster_restart_pipelines[stage];
}

GeometryAssembly::RasterDrawInputs GeometryAssembly::PrepareRasterInputs(
    const RasterDrawParameters& parameters, const BufferRange& indices,
    const std::array<float, 4>& viewport, const GeometryExportWave& system_registers,
    bool dynamic_inputs, std::optional<u32> restart_index, u32 index_source_offset) {
    const auto& p = parameters;
    if (p != RasterDrawLayout(p.vertices, p.instances, p.step_rate_0, p.step_rate_1, p.index_width,
                              p.index_count, p.index_offset, p.swap_mode, p.first_vertex,
                              p.first_instance)) {
        throw std::runtime_error("Invalid native raster input preparation layout");
    }
    const auto restart_layout =
        restart_index ? BuildRasterRestartParameters(p, *restart_index) : RasterRestartParameters{};
    if (!p.work_items) {
        return {{}, p, system_registers, dynamic_inputs, std::nullopt};
    }
    const GeometryExportConfig config{p.work_items, 0, true};
    const u64 input_bytes = config.InputSizeBytes();
    const u64 index_bytes =
        RasterIndexSourceSizeBytes(p.index_count, p.index_width, p.swap_mode, index_source_offset);
    const u32 index_alignment = p.index_width == 2 && p.swap_mode >= AmdGpu::IndexSwapMode::Swap32
                                    ? sizeof(u32)
                                    : p.index_width;
    if (input_bytes > instance.GetDevice().maxBufferLength ||
        (restart_index && (restart_layout.SizeBytes() > instance.GetDevice().maxBufferLength ||
                           input_bytes + restart_layout.SizeBytes() > (u64{32} << 20))) ||
        (index_bytes &&
         (!indices.buffer || indices.buffer.device != instance.GetDevice() ||
          indices.offset % index_alignment || indices.offset > indices.buffer.length ||
          indices.size < index_bytes || indices.size > indices.buffer.length - indices.offset))) {
        throw std::runtime_error("Native raster input ranges exceed device storage");
    }
    const auto compute_pipeline = RasterInputsPipeline();
    const RasterInputParameters data{p,
                                     viewport,
                                     system_registers,
                                     dynamic_inputs,
                                     restart_index.has_value(),
                                     restart_index.value_or(0),
                                     index_source_offset};
    auto upload = staging.Allocate(sizeof(data));
    std::memcpy(upload.UploadData().data(), &data, sizeof(data));
    auto registers = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(input_bytes, alignof(u32)));
    std::optional<RasterRestartWork> restart;
    if (restart_index) {
        restart = RasterRestartWork{
            *this,
            p,
            restart_layout,
            std::make_shared<StagingBufferPool::Allocation>(
                staging.AllocatePrivate(restart_layout.RecordSizeBytes(), alignof(u32))),
            std::make_shared<StagingBufferPool::Allocation>(
                staging.AllocatePrivate(restart_layout.BlockSizeBytes(), alignof(u32))),
            std::make_shared<StagingBufferPool::Allocation>(
                staging.AllocatePrivate(restart_layout.SuperblockSizeBytes(), alignof(u32)))};
    }
    ResourceBindings bindings{instance, 4, 0, 0};
    bindings.SetBuffer(0, index_bytes ? indices.buffer : upload.GetBuffer(),
                       index_bytes ? indices.offset : upload.GetOffset());
    bindings.SetBuffer(1, registers->GetBuffer(), registers->GetOffset());
    bindings.SetBuffer(2, upload.GetBuffer(), upload.GetOffset());
    const auto records = restart ? restart->Records()
                                 : BufferRange{registers->GetBuffer(), registers->GetOffset(),
                                               registers->SizeBytes()};
    bindings.SetBuffer(3, records.buffer, records.offset);
    if (restart)
        restart->Retain(bindings);
    bindings.RetainUploadAllocation(std::move(upload));
    bindings.RetainUploadAllocation(registers);
    if (index_bytes) {
        scheduler.UseResource(indices.buffer);
    }
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(compute_pipeline, bindings);
    const u32 groups = p.work_items / 64;
    const u32 width = std::min(groups, 65535u);
    const u32 height = u32((u64{groups} + width - 1) / width);
    commands.Dispatch(MTLSizeMake(width, height, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    if (restart) {
        auto parameters = std::make_shared<StagingBufferPool::Allocation>(
            staging.Allocate(sizeof(restart_layout)));
        std::memcpy(parameters->UploadData().data(), &restart_layout, sizeof(restart_layout));
        const auto blocks = restart->Blocks();
        const auto superblocks = restart->Superblocks();
        for (u32 stage = 0; stage < 4; ++stage) {
            ResourceBindings prefix{instance, 4, 0, 0};
            prefix.SetBuffer(0, records.buffer, records.offset);
            prefix.SetBuffer(1, blocks.buffer, blocks.offset);
            prefix.SetBuffer(2, superblocks.buffer, superblocks.offset);
            prefix.SetBuffer(3, parameters->GetBuffer(), parameters->GetOffset());
            prefix.RetainUploadAllocation(parameters);
            restart->Retain(prefix);
            scheduler.EndEncoding();
            scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
            commands.BindCompute(RasterRestartPipeline(stage), prefix);
            const u32 groups = stage == 1   ? restart_layout.superblocks
                               : stage == 2 ? 1u
                                            : restart_layout.blocks;
            const u32 width = std::min(groups, 65535u);
            commands.Dispatch(MTLSizeMake(width, (u64{groups} + width - 1) / width, 1),
                              MTLSizeMake(64, 1, 1));
            scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
        }
    }
    return {std::move(registers), p, system_registers, dynamic_inputs, std::move(restart)};
}

GeometryAssembly::IndirectRasterRestartWork GeometryAssembly::ScanIndirectRasterRestart(
    const IndirectDraws::RasterCaptureInputs& inputs) {
    const auto& layout = inputs.layout;
    const auto& p = layout.parameters;
    const auto expected =
        BuildRasterRestartParameters(p.capture.window.capture_capacity, p.restart_index);
    const auto& capacity = layout.restart_capacity;
    const auto valid = [&](const BufferRange& range, u64 bytes) {
        return range.buffer && range.buffer.device == instance.GetDevice() &&
               range.offset % alignof(u32) == 0 && range.offset <= range.buffer.length &&
               range.size == bytes && range.size <= range.buffer.length - range.offset;
    };
    if (!inputs.capture.owner || &inputs.capture.owner->scheduler != &scheduler ||
        &inputs.capture.owner->instance != &instance || p.restart_enabled != 1 ||
        !p.capture.window.index_width || !inputs.restart_work ||
        capacity.work_items != expected.work_items || capacity.blocks != expected.blocks ||
        capacity.superblocks != expected.superblocks ||
        capacity.restart_index != expected.restart_index ||
        !valid(inputs.RestartRecords(), expected.RecordSizeBytes()) ||
        !valid(inputs.RestartParameters(), sizeof(RasterRestartParameters)) ||
        !valid({inputs.restart_work->GetBuffer(), inputs.restart_work->GetOffset(),
                inputs.restart_work->SizeBytes()},
               sizeof(RasterRestartDispatchParameters)) ||
        layout.input_bytes + expected.SizeBytes() + sizeof(RasterRestartDispatchParameters) >
            std::min(u64{32} << 20, u64{instance.GetDevice().maxBufferLength})) {
        throw std::runtime_error("Native indirect raster restart differs from its capture");
    }
    auto blocks = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(expected.BlockSizeBytes(), alignof(u32)));
    auto superblocks = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(expected.SuperblockSizeBytes(), alignof(u32)));
    IndirectRasterRestartWork work{*this, inputs, std::move(blocks), std::move(superblocks)};
    for (u32 stage = 0; stage < 4; ++stage) {
        ResourceBindings bindings{instance, 4, 0, 0};
        const std::array ranges{work.Records(), work.Blocks(), work.Superblocks(),
                                work.Parameters()};
        for (u32 binding = 0; binding < ranges.size(); ++binding)
            bindings.SetBuffer(binding, ranges[binding].buffer, ranges[binding].offset);
        work.Retain(bindings);
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
        commands.BindCompute(RasterRestartPipeline(stage), bindings);
        commands.DispatchIndirect(inputs.RestartDispatchArguments(stage), MTLSizeMake(64, 1, 1));
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    }
    return work;
}

GeometryAssembly::PreparedRasterVertices GeometryAssembly::PrepareRasterVertices(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    const RasterDrawInputs& inputs, bool neo_mode,
    std::span<const GuestBufferExtent> guest_extents) {
    if (!shader.geometry_export || !inputs.registers ||
        shader.geometry_export->dynamic_work_items != inputs.dynamic_inputs ||
        shader.geometry_export->ResolveWorkItems(inputs.parameters.work_items) !=
            inputs.parameters.work_items) {
        throw std::runtime_error("Native raster input worklist differs from its vertex kernel");
    }
    const auto domains = BuildRasterInputDomains(inputs.parameters, inputs.system_registers);
    return PrepareRasterVertices(shader_cache, resources, shader, inputs.registers, neo_mode,
                                 inputs.parameters.active_work_items, guest_extents, domains);
}

id<MTLComputePipelineState> GeometryAssembly::WorkPipeline() {
    if (!work_pipeline) {
        auto library = compiler.CompileLibrary(GeometryWorkSource, "shadPS4:GeometryWork");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_prepare_geometry_work";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        work_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return work_pipeline;
}

GeometryAssembly::DrawInputs GeometryAssembly::PrepareDrawInputs(
    const GeometryDrawParameters& parameters, const BufferRange& indices,
    const GeometryRingLayout& rings, const GeometryExportWave& export_template,
    u32 first_geometry_wave_id) {
    if (rings.export_items != parameters.export_items ||
        rings.geometry_items != parameters.geometry_items ||
        parameters != GeometryDrawLayout(AmdGpu::PrimitiveType(parameters.topology),
                                         parameters.input_vertices, parameters.instances,
                                         parameters.gs_invocations, parameters.step_rate_0,
                                         parameters.step_rate_1, parameters.index_width,
                                         parameters.index_offset) ||
        u64{parameters.export_items} * sizeof(GeometryExportInput) >
            instance.GetDevice().maxBufferLength ||
        u64{parameters.geometry_items} * sizeof(GeometryKernelWorkItem) >
            instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Invalid native geometry wave draw layout or device capacity");
    }
    const auto inputs = BuildGeometryWaveInputs(rings, export_template, first_geometry_wave_id);
    const auto upload = [&](const auto& waves) {
        const auto bytes = std::as_bytes(std::span{waves});
        if (bytes.empty() || bytes.size() > instance.GetDevice().maxBufferLength) {
            throw std::runtime_error("Native geometry wave upload exceeds device capacity");
        }
        auto allocation =
            std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(bytes.size(), 16));
        std::memcpy(allocation->UploadData().data(), bytes.data(), bytes.size());
        return allocation;
    };
    auto exports = upload(inputs.exports);
    auto geometry = upload(inputs.geometry);
    const BufferRange export_range{exports->GetBuffer(), exports->GetOffset(),
                                   exports->SizeBytes()};
    const BufferRange geometry_range{geometry->GetBuffer(), geometry->GetOffset(),
                                     geometry->SizeBytes()};
    auto result = PrepareDrawInputs(parameters, indices, export_range, geometry_range);
    result.export_waves = std::move(exports);
    result.geometry_waves = std::move(geometry);
    return result;
}

GeometryAssembly::DrawInputs GeometryAssembly::PrepareDrawInputs(
    const GeometryDrawParameters& parameters, const BufferRange& indices,
    const BufferRange& export_waves, const BufferRange& geometry_waves) {
    const auto& p = parameters;
    if (p != GeometryDrawLayout(AmdGpu::PrimitiveType(p.topology), p.input_vertices, p.instances,
                                p.gs_invocations, p.step_rate_0, p.step_rate_1, p.index_width,
                                p.index_offset)) {
        throw std::runtime_error("Invalid native geometry draw layout");
    }
    if (!p.geometry_items) {
        return {{}, {}, p};
    }
    const auto valid = [&](const BufferRange& range, u64 bytes, u32 alignment) {
        return range.buffer && range.buffer.device == instance.GetDevice() &&
               range.offset % alignment == 0 && range.offset <= range.buffer.length &&
               range.size >= bytes && range.size <= range.buffer.length - range.offset;
    };
    const u64 export_bytes = u64{p.export_items} * sizeof(GeometryExportInput);
    const u64 work_bytes = u64{p.geometry_items} * sizeof(GeometryKernelWorkItem);
    if ((p.index_width && !valid(indices, u64{p.input_vertices} * p.index_width, p.index_width)) ||
        !valid(export_waves, u64{p.export_items / 64} * sizeof(GeometryExportWave), sizeof(u32)) ||
        !valid(geometry_waves, u64{p.geometry_items / 64} * sizeof(GeometryKernelWave),
               sizeof(u32)) ||
        export_bytes > instance.GetDevice().maxBufferLength ||
        work_bytes > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Invalid native geometry draw input ranges");
    }
    const auto compute_pipeline = WorkPipeline();
    if (p.index_width)
        scheduler.UseResource(indices.buffer);
    scheduler.UseResource(export_waves.buffer);
    scheduler.UseResource(geometry_waves.buffer);
    auto parameter_data = staging.Allocate(sizeof(p));
    std::memcpy(parameter_data.UploadData().data(), &p, sizeof(p));
    auto exports = std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(export_bytes));
    auto work = std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(work_bytes));
    ResourceBindings bindings{instance, 6, 0, 0};
    bindings.SetBuffer(0, p.index_width ? indices.buffer : parameter_data.GetBuffer(),
                       p.index_width ? indices.offset : parameter_data.GetOffset());
    bindings.SetBuffer(1, export_waves.buffer, export_waves.offset);
    bindings.SetBuffer(2, geometry_waves.buffer, geometry_waves.offset);
    bindings.SetBuffer(3, exports->GetBuffer(), exports->GetOffset());
    bindings.SetBuffer(4, work->GetBuffer(), work->GetOffset());
    bindings.SetBuffer(5, parameter_data.GetBuffer(), parameter_data.GetOffset());
    bindings.RetainUploadAllocation(std::move(parameter_data));
    bindings.RetainUploadAllocation(exports);
    bindings.RetainUploadAllocation(work);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(compute_pipeline, bindings);
    const u32 groups = std::max(p.export_items, p.geometry_items) / 64;
    const u32 width = std::min(groups, 65535u);
    const u32 height = u32((u64{groups} + width - 1) / width);
    commands.Dispatch(MTLSizeMake(width, height, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {std::move(exports), std::move(work), p};
}

void GeometryAssembly::Export(ShaderCache& shader_cache, GuestResources& resources,
                              ShaderCache::Variant& shader, const DrawInputs& inputs, bool neo_mode,
                              std::span<const GuestBufferExtent> guest_extents) {
    if (!shader.geometry_export || !inputs.exports ||
        shader.geometry_export->work_items != inputs.parameters.export_items) {
        throw std::runtime_error("Native ES worklist differs from its compiled kernel");
    }
    Export(shader_cache, resources, shader, inputs.exports, neo_mode,
           inputs.parameters.export_vertices, guest_extents);
}

GeometryAssembly::Emissions GeometryAssembly::Generate(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    const DrawInputs& inputs, u32 export_vertex_words, const BufferRange& errors, bool neo_mode,
    std::span<const GuestBufferExtent> guest_extents) {
    if (!shader.geometry || !inputs.work ||
        shader.geometry->emissions.invocations != inputs.parameters.geometry_items) {
        throw std::runtime_error("Native GS worklist differs from its compiled kernel");
    }
    const BufferRange work{inputs.work->GetBuffer(), inputs.work->GetOffset(),
                           inputs.work->SizeBytes()};
    return Generate(shader_cache, resources, shader, work, inputs.parameters.export_items,
                    export_vertex_words, errors, neo_mode, inputs.parameters.geometry_invocations,
                    guest_extents);
}

} // namespace Metal
