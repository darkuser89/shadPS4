// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
constexpr std::string_view GeometryInputsSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct WorkItem {
    uint ring_offset; uint wave_id; uint vertices[6]; uint primitive_id; uint invocation_id; uint active;
};
struct Input {
    uint ring_offset; uint wave_id; uint vertex_offsets[6]; uint primitive_id; uint invocation_id; uint active;
};
struct Parameters { uint invocations; uint export_vertices; uint export_vertex_words; };
static_assert(sizeof(WorkItem) == 44);
static_assert(sizeof(Input) == 44);
static_assert(sizeof(Parameters) == 12);
kernel void metal4_prepare_geometry_inputs(device const WorkItem* work [[buffer(0)]],
                                           device Input* inputs [[buffer(1)]],
                                           constant Parameters& p [[buffer(2)]],
                                           device atomic_uint& errors [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint2 grid [[threadgroups_per_grid]],
                                           uint local [[thread_index_in_threadgroup]]) {
    const ulong item = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (item >= p.invocations) return;
    const WorkItem source = work[item];
    bool valid = source.active <= 1;
    if (source.active)
        for (uint slot = 0; slot < 6; ++slot) valid &= source.vertices[slot] < p.export_vertices;
    Input result{};
    if (valid && source.active) {
        result.ring_offset = source.ring_offset;
        result.wave_id = source.wave_id;
        result.primitive_id = source.primitive_id;
        result.invocation_id = source.invocation_id;
        result.active = 1;
        for (uint slot = 0; slot < 6; ++slot) {
            const uint index = source.vertices[slot];
            result.vertex_offsets[slot] = (index / 64) * p.export_vertex_words * 64 + index % 64;
        }
    } else if (!valid) {
        atomic_fetch_or_explicit(&errors, 4u, memory_order_relaxed);
    }
    inputs[item] = result;
}
)metal";

constexpr std::string_view GeometrySource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint invocations; uint max_vertices; uint vertex_words; uint mesh_vertices;
    uint topology[4]; uint mesh_counts[4]; uint mesh_offsets[4];
    uint stream_stride_words; uint mesh_stride_words; uint meshes_per_invocation; uint jobs;
};
static_assert(sizeof(Parameters) == 80);
struct Primitive {
    uint3 vertices;
    uint first_vertex;
    uint last_vertex;
    uint valid;
    uint error;
};
Primitive geometry_primitive(device const uint* source, uint vertex_words,
                             uint global_vertex, uint local_vertex, uint topology) {
    Primitive result{};
    const uint ordinal = source[4 + ulong(global_vertex) * vertex_words];
    const bool sequence_valid = ordinal <= global_vertex &&
        (!ordinal || (global_vertex && source[4 + ulong(global_vertex - 1) * vertex_words] == ordinal - 1));
    if (!sequence_valid) {
        result.error = 2;
        return result;
    }
    if (local_vertex < topology || ordinal < topology) return result;
    if (topology == 2 && source[4 + ulong(global_vertex - 2) * vertex_words] != ordinal - 2) {
        result.error = 2;
        return result;
    }
    result.first_vertex = local_vertex - topology;
    result.last_vertex = local_vertex;
    result.vertices = uint3(local_vertex);
    if (topology == 1) result.vertices = uint3(local_vertex - 1, local_vertex, local_vertex);
    if (topology == 2)
        result.vertices = uint3(local_vertex - 2, local_vertex - 1 + (ordinal & 1), local_vertex - (ordinal & 1));
    result.valid = 1;
    return result;
}
kernel void metal4_assemble_geometry(device const uint* emissions [[buffer(0)]],
                                     device uint* primitives [[buffer(1)]],
                                     constant Parameters& p [[buffer(2)]],
                                     device atomic_uint& validation_errors [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint2 grid [[threadgroups_per_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const ulong linear_job = ulong(group.y) * grid.x + group.x;
    if (linear_job >= p.jobs) return;
    const uint job = uint(linear_job);
    const uint invocation = job / p.meshes_per_invocation;
    const uint mesh_index = job % p.meshes_per_invocation;
    uint stream = 0;
    while (stream < 3 && mesh_index >= p.mesh_offsets[stream] + p.mesh_counts[stream]) ++stream;
    const uint chunk = mesh_index - p.mesh_offsets[stream];
    const uint topology = p.topology[stream];
    device const uint* source = emissions + (ulong(invocation) * 4 + stream) * p.stream_stride_words;
    device uint* destination = primitives + ulong(job) * p.mesh_stride_words;
    const uint first_vertex = chunk * (p.mesh_vertices - topology);
    const uint raw_count = source[0];
    const uint count = raw_count <= p.max_vertices ? raw_count : 0;
    const uint vertices = first_vertex < count ? min(p.mesh_vertices, count - first_vertex) : 0;
    uint primitive_count = 0;
    uint errors = raw_count > p.max_vertices ? 1 : 0;
    for (uint base = 0; base < p.mesh_vertices; base += 32) {
        const uint local_vertex = base + lane;
        Primitive primitive{};
        if (local_vertex < vertices)
            primitive = geometry_primitive(source, p.vertex_words, first_vertex + local_vertex, local_vertex, topology);
        const uint prefix = simd_prefix_exclusive_sum(primitive.valid);
        if (primitive.valid) {
            device uint* output = destination + 4 + (primitive_count + prefix) * 8;
            output[0] = primitive.vertices.x;
            output[1] = primitive.vertices.y;
            output[2] = primitive.vertices.z;
            output[3] = primitive.first_vertex;
            output[4] = primitive.last_vertex;
            output[5] = output[6] = output[7] = 0;
        }
        primitive_count += simd_sum(primitive.valid);
        errors |= simd_or(primitive.error);
    }
    if (!lane) {
        destination[0] = first_vertex;
        destination[1] = vertices;
        destination[2] = errors ? 0 : primitive_count;
        destination[3] = errors;
        if (errors) atomic_fetch_or_explicit(&validation_errors, errors << 4, memory_order_relaxed);
    }
}
)metal";

constexpr std::string_view GeometryCopyHeadersSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint invocations;
    uint source_stride;
    uint destination_stride;
    uint stream;
};
static_assert(sizeof(Parameters) == 16);
kernel void metal4_copy_geometry_headers(device const uint* source [[buffer(0)]],
                                        device uint* destination [[buffer(1)]],
                                        constant Parameters& p [[buffer(2)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint2 grid [[threadgroups_per_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    const ulong slot = ulong(group.y) * grid.x + group.x;
    if (slot >= ulong(p.invocations) * 4 || lane >= 4) return;
    destination[slot * p.destination_stride + lane] =
        (slot & 3) == p.stream ? source[slot * p.source_stride + lane] : 0;
}
)metal";
} // namespace

GeometryAssembly::GeometryAssembly(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

id<MTLComputePipelineState> GeometryAssembly::InputsPipeline() {
    if (!inputs_pipeline) {
        auto library = compiler.CompileLibrary(GeometryInputsSource, "shadPS4:GeometryInputs");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_prepare_geometry_inputs";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        inputs_pipeline = compiler.CompileComputePipeline(descriptor);
    }
    return inputs_pipeline;
}

std::shared_ptr<StagingBufferPool::Allocation> GeometryAssembly::PrepareKernelInputs(
    const BufferRange& work, u32 invocations, u32 export_vertices, u32 export_vertex_words,
    const BufferRange& errors) {
    const auto valid = [&](const BufferRange& range, u64 bytes) {
        return range.buffer && range.buffer.device == instance.GetDevice() &&
               range.offset % sizeof(u32) == 0 && range.offset <= range.buffer.length &&
               range.size >= bytes && range.size <= range.buffer.length - range.offset;
    };
    const u64 input_bytes = u64{invocations} * sizeof(GeometryKernelInput);
    const u64 work_bytes = u64{invocations} * sizeof(GeometryKernelWorkItem);
    if (!invocations || invocations % 64 || !export_vertices || export_vertices % 64 ||
        !export_vertex_words ||
        u64{export_vertices} * export_vertex_words >
            std::numeric_limits<u32>::max() / sizeof(u32) ||
        input_bytes / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1 ||
        input_bytes > instance.GetDevice().maxBufferLength || !valid(work, work_bytes) ||
        !valid(errors, sizeof(u32)) ||
        (work.buffer == errors.buffer && work.offset < errors.offset + sizeof(u32) &&
         errors.offset < work.offset + work_bytes)) {
        throw std::runtime_error("Invalid native geometry input worklist or capacity");
    }
    struct Parameters {
        u32 invocations;
        u32 export_vertices;
        u32 export_vertex_words;
    };
    static_assert(sizeof(Parameters) == 12);
    const Parameters parameters{invocations, export_vertices, export_vertex_words};
    const auto compute_pipeline = InputsPipeline();
    scheduler.UseResource(work.buffer);
    scheduler.UseResource(errors.buffer);
    auto parameter_data = staging.Allocate(sizeof(parameters));
    std::memcpy(parameter_data.UploadData().data(), &parameters, sizeof(parameters));
    auto output = std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(input_bytes));
    ResourceBindings bindings{instance, 4, 0, 0};
    bindings.SetBuffer(0, work.buffer, work.offset);
    bindings.SetBuffer(1, output->GetBuffer(), output->GetOffset());
    bindings.SetBuffer(2, parameter_data.GetBuffer(), parameter_data.GetOffset());
    bindings.SetBuffer(3, errors.buffer, errors.offset);
    bindings.RetainUploadAllocation(std::move(parameter_data));
    bindings.RetainUploadAllocation(output);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(compute_pipeline, bindings);
    const u32 groups = invocations / 64;
    const u32 width = std::min(groups, 65535u);
    const u32 height = u32((u64{groups} + width - 1) / width);
    commands.Dispatch(MTLSizeMake(width, height, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return output;
}

id<MTLComputePipelineState> GeometryAssembly::Pipeline() {
    if (!pipeline) {
        auto library = compiler.CompileLibrary(GeometrySource, "shadPS4:GeometryAssembly");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_assemble_geometry";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 32;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
        auto compiled = compiler.CompileComputePipeline(descriptor);
        if (compiled.threadExecutionWidth != 32) {
            throw std::runtime_error("Native geometry assembly requires a 32-lane SIMD group");
        }
        pipeline = compiled;
    }
    return pipeline;
}

GeometryAssembly::Emissions GeometryAssembly::Allocate(u32 invocations, u32 max_vertices,
                                                       u32 vertex_words, u32 mesh_vertices,
                                                       const std::array<u32, 4>& topology,
                                                       u32 active_streams) {
    const auto parameters = GeometryAssemblyLayout(invocations, max_vertices, vertex_words,
                                                   mesh_vertices, topology, active_streams);
    if (!parameters.jobs) {
        return {{}, parameters};
    }
    if (parameters.EmissionSizeBytes() > instance.GetDevice().maxBufferLength ||
        parameters.PrimitiveSizeBytes() > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Native geometry allocation exceeds device capacity");
    }
    return {std::make_shared<StagingBufferPool::Allocation>(
                staging.Allocate(parameters.EmissionSizeBytes())),
            parameters};
}

id<MTLComputePipelineState> GeometryAssembly::CopyHeadersPipeline() {
    if (!copy_headers_pipeline) {
        auto library =
            compiler.CompileLibrary(GeometryCopyHeadersSource, "shadPS4:GeometryCopyHeaders");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"metal4_copy_geometry_headers";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 32;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
        auto compiled = compiler.CompileComputePipeline(descriptor);
        if (compiled.threadExecutionWidth != 32) {
            throw std::runtime_error("Native geometry header copy requires 32-lane SIMD groups");
        }
        copy_headers_pipeline = compiled;
    }
    return copy_headers_pipeline;
}

void GeometryAssembly::CopyHeaders(const Emissions& source, const Emissions& destination,
                                   u32 stream) {
    const auto& parameters = source.GetParameters();
    const auto& payload = destination.GetParameters();
    const std::array data{parameters.invocations, parameters.stream_stride_words,
                          payload.stream_stride_words, stream};
    const auto header_pipeline = CopyHeadersPipeline();
    auto upload = staging.Allocate(sizeof(data));
    std::memcpy(upload.UploadData().data(), data.data(), sizeof(data));
    ResourceBindings bindings{instance, 3, 0, 0};
    const auto input = source.GetRange();
    const auto output = destination.GetRange();
    bindings.SetBuffer(0, input.buffer, input.offset);
    bindings.SetBuffer(1, output.buffer, output.offset);
    bindings.SetBuffer(2, upload.GetBuffer(), upload.GetOffset());
    bindings.RetainUploadAllocation(std::move(upload));
    source.Retain(bindings);
    destination.Retain(bindings);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(header_pipeline, bindings);
    const u64 slots = u64{parameters.invocations} * 4;
    const u32 width = u32(std::min(slots, u64{65535}));
    const u32 height = u32((slots + width - 1) / width);
    commands.Dispatch(MTLSizeMake(width, height, 1), MTLSizeMake(32, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
}

GuestResources::Stage& GeometryAssembly::PreparedKernel::GetResources() const {
    if (!stage) {
        throw std::runtime_error("Native geometry kernel has no prepared resources");
    }
    return *stage;
}

GeometryAssembly::PreparedKernel GeometryAssembly::PrepareKernel(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    std::span<const GuestResources::GeometryBuffer> buffers, u32 work_items, bool neo_mode,
    u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
    std::span<const GeometryAddressInput> geometry_inputs) {
    const auto active_count = GeometryInitialActiveCount(work_items, initial_active_work_items);
    const auto compute_pipeline = shader_cache.GetComputePipeline(shader);
    if (!compute_pipeline) {
        throw std::runtime_error("Native geometry kernel requires a compute pipeline");
    }
    auto stage =
        resources.Prepare(shader, neo_mode, buffers, active_count, guest_extents, geometry_inputs);
    return {*this, resources, std::move(stage), compute_pipeline, work_items};
}

void GeometryAssembly::Execute(std::span<PreparedKernel* const> kernels,
                               std::span<const ImageInfo> attachments) {
    GuestResources* resources{};
    for (size_t index = 0; index < kernels.size(); ++index) {
        const auto* kernel = kernels[index];
        if (!kernel || kernel->owner != this || !kernel->resources || !kernel->stage ||
            kernel->submitted || !kernel->pipeline ||
            (resources && resources != kernel->resources) ||
            std::find(kernels.begin(), kernels.begin() + index, kernel) !=
                kernels.begin() + index) {
            throw std::runtime_error("Invalid native geometry dispatch sequence");
        }
        resources = kernel->resources;
        if (kernel->arguments) {
            const auto& args = *kernel->arguments;
            if (!args.GetBuffer() || args.GetBuffer().device != instance.GetDevice() ||
                args.GetOffset() % alignof(u32) || kernel->argument_offset % alignof(u32) ||
                args.GetOffset() > args.GetBuffer().length ||
                args.SizeBytes() > args.GetBuffer().length - args.GetOffset() ||
                kernel->argument_offset > args.SizeBytes() ||
                args.SizeBytes() - kernel->argument_offset <
                    sizeof(MTLDispatchThreadgroupsIndirectArguments)) {
                throw std::runtime_error("Native geometry indirect dispatch arguments are invalid");
            }
        } else if (kernel->argument_offset) {
            throw std::runtime_error("Native geometry dispatch offset requires indirect arguments");
        }
        const std::array stages{kernel->stage.get()};
        resources->ValidateAttachments(stages, attachments);
    }
    for (size_t index = 0; index < kernels.size(); ++index) {
        auto* kernel = kernels[index];
        if (kernel->ring_replay) {
            const auto& replay = *kernel->ring_replay;
            auto companions = replay.companions;
            for (size_t next = index + 1; next < kernels.size(); ++next) {
                auto* stage = kernels[next]->stage.get();
                if (std::ranges::find(companions, stage) == companions.end()) {
                    companions.push_back(stage);
                }
            }
            resources->ReplayGeometryRing(*kernel->stage, replay.origin, replay.source, companions,
                                          attachments);
        }
        const std::array stages{kernel->stage.get()};
        resources->Acquire(stages);
        resources->MarkWrites(stages);
        auto snapshot = resources->Snapshot(*kernel->stage);
        for (auto& input : kernel->retained_inputs) {
            snapshot->RetainUploadAllocation(std::move(input));
        }
        kernel->retained_inputs.clear();
        if (kernel->arguments) {
            snapshot->RetainUploadAllocation(kernel->arguments);
        }
        kernel->submitted = true;
        if (kernel->copy_dispatch) {
            BuildCopyDispatch(*kernel);
        }
        if (kernel->clear_emissions || kernel->clear_raster) {
            scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
            scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
            if (kernel->clear_emissions) {
                commands.FillBuffer(kernel->clear_emissions->GetRange(), 0);
            }
            if (kernel->clear_raster) {
                commands.FillBuffer(*kernel->clear_raster, 0);
            }
        }
        if (kernel->header_copy) {
            CopyHeaders(kernel->header_copy->first, kernel->header_copy->second,
                        kernel->copy_stream);
        }
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
        commands.BindCompute(kernel->pipeline, *snapshot);
        if (kernel->arguments) {
            commands.DispatchIndirect({kernel->arguments->GetBuffer(),
                                       kernel->arguments->GetOffset() + kernel->argument_offset,
                                       kernel->arguments->SizeBytes() - kernel->argument_offset},
                                      MTLSizeMake(64, 1, 1));
        } else {
            commands.Dispatch(MTLSizeMake(kernel->work_items / 64, 1, 1), MTLSizeMake(64, 1, 1));
        }
        scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    }
}

GeometryAssembly::PreparedKernel GeometryAssembly::PrepareExport(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
    u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
    std::span<const GeometryAddressInput> geometry_inputs) {
    if (!shader.geometry_export || !shader.shader.geometry_export || shader.geometry ||
        shader.geometry_copy || shader.geometry_export->raster_output ||
        shader.shader.geometry_export->raster || !inputs) {
        throw std::runtime_error("Native ES dispatch requires a compiled export kernel");
    }
    const std::array buffers{
        GuestResources::GeometryBuffer{shader.shader.geometry_export->inputs, std::move(inputs)}};
    return PrepareKernel(shader_cache, resources, shader, buffers,
                         shader.geometry_export->work_items, neo_mode, initial_active_work_items,
                         guest_extents, geometry_inputs);
}

GeometryAssembly::PreparedRasterVertices GeometryAssembly::PrepareRasterVertices(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
    u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
    std::span<const GeometryAddressInput> geometry_inputs) {
    if (!shader.geometry_export || !shader.geometry_export->raster_output ||
        !shader.shader.geometry_export || !shader.shader.geometry_export->raster ||
        shader.geometry || shader.geometry_copy || shader.tessellation || !inputs) {
        throw std::runtime_error("Native vertex raster dispatch requires its compiled payload");
    }
    const auto& config = *shader.geometry_export;
    const auto& bindings = *shader.shader.geometry_export;
    const auto& layout = *bindings.raster;
    const u32 work_items = config.WorkItemsFromInputBytes(inputs->SizeBytes());
    const u64 bytes = bindings.RasterSizeBytes(config, work_items);
    if (bytes > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Native vertex raster payload exceeds its storage layout");
    }
    auto output = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(bytes, alignof(u32)));
    const std::array buffers{GuestResources::GeometryBuffer{bindings.inputs, std::move(inputs)},
                             GuestResources::GeometryBuffer{layout.output, output}};
    auto kernel = PrepareKernel(shader_cache, resources, shader, buffers, work_items, neo_mode,
                                initial_active_work_items, guest_extents, geometry_inputs);
    RasterVertices vertices{*this, std::move(output), work_items, layout};
    kernel.clear_raster = vertices.GetRange();
    return {std::move(vertices), std::move(kernel)};
}

void GeometryAssembly::Export(ShaderCache& shader_cache, GuestResources& resources,
                              ShaderCache::Variant& shader,
                              std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
                              u32 initial_active_work_items,
                              std::span<const GuestBufferExtent> guest_extents) {
    auto kernel = PrepareExport(shader_cache, resources, shader, std::move(inputs), neo_mode,
                                initial_active_work_items, guest_extents);
    const std::array kernels{&kernel};
    Execute(kernels);
}

GeometryAssembly::PreparedEmissions GeometryAssembly::PrepareGenerate(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
    u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
    std::span<const GeometryAddressInput> geometry_inputs,
    std::shared_ptr<StagingBufferPool::Allocation> arguments) {
    if (!shader.geometry || !shader.shader.geometry || shader.geometry_copy || !inputs ||
        shader.geometry->emissions.vertex_words != 1) {
        throw std::runtime_error("Native GS dispatch requires a compiled geometry kernel");
    }
    const auto& plan = shader.geometry->emissions;
    if (shader.geometry->compact_workgroups != bool(arguments) ||
        (arguments &&
         (!arguments->GetBuffer() || arguments->GetBuffer().device != instance.GetDevice() ||
          arguments->GetOffset() % alignof(u32) ||
          arguments->SizeBytes() < sizeof(MTLDispatchThreadgroupsIndirectArguments) ||
          arguments->GetOffset() > arguments->GetBuffer().length ||
          arguments->SizeBytes() > arguments->GetBuffer().length - arguments->GetOffset()))) {
        throw std::runtime_error("Native GS compact dispatch requires valid indirect arguments");
    }
    const auto active_count =
        GeometryInitialActiveCount(plan.invocations, initial_active_work_items);
    u32 active_streams{};
    for (u32 stream = 0; stream < 4; ++stream) {
        active_streams |= plan.mesh_counts[stream] ? 1u << stream : 0;
    }
    auto output = Allocate(plan.invocations, plan.max_vertices, plan.vertex_words,
                           plan.mesh_vertices, plan.topology, active_streams);
    if (output.GetParameters() != plan) {
        throw std::runtime_error("Native GS dispatch layout differs from its compiled kernel");
    }
    const auto& bindings = *shader.shader.geometry;
    const std::array buffers{GuestResources::GeometryBuffer{bindings.inputs, std::move(inputs)},
                             GuestResources::GeometryBuffer{bindings.emissions, output.storage}};
    auto kernel = PrepareKernel(shader_cache, resources, shader, buffers, plan.invocations,
                                neo_mode, active_count, guest_extents, geometry_inputs);
    if (arguments) {
        kernel.arguments = std::move(arguments);
        kernel.clear_emissions = output;
    }
    return {std::move(output), std::move(kernel)};
}

GeometryAssembly::Emissions GeometryAssembly::Generate(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
    u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents) {
    auto prepared = PrepareGenerate(shader_cache, resources, shader, std::move(inputs), neo_mode,
                                    initial_active_work_items, guest_extents);
    const std::array kernels{&prepared.kernel};
    Execute(kernels);
    return std::move(prepared.emissions);
}

GeometryAssembly::Emissions GeometryAssembly::Generate(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    const BufferRange& work, u32 export_vertices, u32 export_vertex_words,
    const BufferRange& errors, bool neo_mode, u32 initial_active_work_items,
    std::span<const GuestBufferExtent> guest_extents) {
    if (!shader.geometry || !shader.shader.geometry || shader.geometry_copy ||
        shader.geometry->emissions.vertex_words != 1) {
        throw std::runtime_error("Native GS dispatch requires a compiled geometry kernel");
    }
    const u32 invocations = shader.geometry->emissions.invocations;
    const u32 active_count = GeometryInitialActiveCount(invocations, initial_active_work_items);
    auto inputs =
        PrepareKernelInputs(work, invocations, export_vertices, export_vertex_words, errors);
    return Generate(shader_cache, resources, shader, std::move(inputs), neo_mode, active_count,
                    guest_extents);
}

GeometryAssembly::PreparedEmissions GeometryAssembly::PrepareCopy(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    const Emissions& emissions, std::shared_ptr<StagingBufferPool::Allocation> inputs,
    bool neo_mode, u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
    std::span<const GeometryAddressInput> geometry_inputs, const BufferRange& errors) {
    if (!shader.geometry_copy || !shader.shader.geometry_copy || shader.geometry || !inputs ||
        emissions.GetParameters() != shader.geometry_copy->emissions || !emissions.storage) {
        throw std::runtime_error("Native GS copy requires matching metadata and compiled shader");
    }
    const auto& config = *shader.geometry_copy;
    const auto active_count =
        GeometryInitialActiveCount(config.work_items, initial_active_work_items);
    const auto& bindings = *shader.shader.geometry_copy;
    const auto& plan = bindings.payload;
    if (config.compact_workgroups &&
        (!errors.buffer || errors.buffer.device != instance.GetDevice() ||
         errors.offset % sizeof(u32) || errors.offset > errors.buffer.length ||
         errors.size < sizeof(u32) || errors.size > errors.buffer.length - errors.offset ||
         (errors.buffer == inputs->GetBuffer() &&
          errors.offset < inputs->GetOffset() + inputs->SizeBytes() &&
          inputs->GetOffset() < errors.offset + sizeof(u32)) ||
         (errors.buffer == emissions.storage->GetBuffer() &&
          errors.offset < emissions.storage->GetOffset() + emissions.storage->SizeBytes() &&
          emissions.storage->GetOffset() < errors.offset + sizeof(u32)))) {
        throw std::runtime_error("Native compact copy requires a separate bounded error buffer");
    }
    if (config.stream >= 4 || !config.emissions.mesh_counts[config.stream]) {
        throw std::runtime_error("Native GS copy requires an active stream");
    }
    auto payload =
        Allocate(config.emissions.invocations, config.emissions.max_vertices, plan.vertex_words,
                 config.emissions.mesh_vertices, config.emissions.topology, 1u << config.stream);
    if (payload.GetParameters() != plan) {
        throw std::runtime_error("Native GS copy layout differs from its compiled shader");
    }
    const std::array buffers{GuestResources::GeometryBuffer{bindings.inputs, inputs},
                             GuestResources::GeometryBuffer{bindings.emissions, emissions.storage},
                             GuestResources::GeometryBuffer{bindings.output, payload.storage}};
    auto kernel = PrepareKernel(shader_cache, resources, shader, buffers, config.work_items,
                                neo_mode, active_count, guest_extents, geometry_inputs);
    static_cast<void>(CopyHeadersPipeline());
    kernel.header_copy.emplace(emissions, payload);
    kernel.copy_stream = config.stream;
    if (config.compact_workgroups) {
        static_cast<void>(CopyDispatchPipeline());
        kernel.arguments = std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
            sizeof(MTLDispatchThreadgroupsIndirectArguments), alignof(u32)));
        kernel.copy_dispatch.emplace(config, emissions, std::move(inputs), errors);
        kernel.clear_emissions = payload;
    }
    return {std::move(payload), std::move(kernel)};
}

GeometryAssembly::Emissions GeometryAssembly::Copy(
    ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
    const Emissions& emissions, std::shared_ptr<StagingBufferPool::Allocation> inputs,
    bool neo_mode, u32 initial_active_work_items,
    std::span<const GuestBufferExtent> guest_extents) {
    auto prepared = PrepareCopy(shader_cache, resources, shader, emissions, std::move(inputs),
                                neo_mode, initial_active_work_items, guest_extents);
    const std::array kernels{&prepared.kernel};
    Execute(kernels);
    return std::move(prepared.emissions);
}

GeometryAssembly::Result GeometryAssembly::Assemble(const Emissions& emissions,
                                                    const BufferRange& errors) {
    const auto& parameters = emissions.GetParameters();
    if (!parameters.jobs) {
        return {emissions, {}};
    }
    const auto source = emissions.GetRange();
    const u64 source_size = parameters.EmissionSizeBytes();
    const u64 output_size = parameters.PrimitiveSizeBytes();
    if (!source.buffer || source.buffer.device != instance.GetDevice() ||
        source.offset % sizeof(u32) || source.offset > source.buffer.length ||
        source.size < source_size || source.size > source.buffer.length - source.offset ||
        output_size > instance.GetDevice().maxBufferLength || !errors.buffer ||
        errors.buffer.device != instance.GetDevice() || errors.offset % sizeof(u32) ||
        errors.offset > errors.buffer.length || errors.size < sizeof(u32) ||
        errors.size > errors.buffer.length - errors.offset ||
        (source.buffer == errors.buffer && source.offset < errors.offset + sizeof(u32) &&
         errors.offset < source.offset + source_size)) {
        throw std::runtime_error("Invalid native geometry assembly buffer capacity");
    }
    const auto compute_pipeline = Pipeline();
    auto parameter_data = staging.Allocate(sizeof(parameters));
    std::memcpy(parameter_data.UploadData().data(), &parameters, sizeof(parameters));
    auto output = std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(output_size));
    ResourceBindings bindings{instance, 4, 0, 0};
    bindings.SetBuffer(0, source.buffer, source.offset);
    bindings.SetBuffer(1, output->GetBuffer(), output->GetOffset());
    bindings.SetBuffer(2, parameter_data.GetBuffer(), parameter_data.GetOffset());
    scheduler.UseResource(errors.buffer);
    bindings.SetBuffer(3, errors.buffer, errors.offset);
    bindings.RetainUploadAllocation(std::move(parameter_data));
    emissions.Retain(bindings);
    bindings.RetainUploadAllocation(output);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(compute_pipeline, bindings);
    const u32 width = std::min(parameters.jobs, 65535u);
    const u32 height = u32((u64{parameters.jobs} + width - 1) / width);
    commands.Dispatch(MTLSizeMake(width, height, 1), MTLSizeMake(32, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return {emissions, std::move(output)};
}

GeometryAssembly::MeshShader GeometryAssembly::CompileMesh(
    u32 stream, u32 topology, u32 vertex_words, std::span<const GeometryOutputAttribute> attributes,
    const ShaderCache::Variant* fragment, bool last_provoking_vertex, bool flip_vertex_y,
    const std::array<u32, 2>& viewport_extent, bool primitive_ids) {
    if (fragment && (fragment->info.sw_stage != Shader::SwStage::Fragment ||
                     fragment->shader.translated.stage != Shader::SwStage::Fragment ||
                     !fragment->shader.translated.geometry_inputs_supported ||
                     fragment->specialization.runtime_info.hw.fs.clip_distance_emulation)) {
        throw std::runtime_error("Metal geometry fragment interface requires native lowering");
    }
    const std::span<const GeometryMeshVarying> varyings =
        fragment ? fragment->shader.translated.geometry_inputs
                 : std::span<const GeometryMeshVarying>{};
    MeshInterface key{stream,        topology,        vertex_words, last_provoking_vertex,
                      flip_vertex_y, viewport_extent, primitive_ids};
    key.attributes.reserve(attributes.size());
    for (const auto& attribute : attributes) {
        key.attributes.push_back(
            {u64(attribute.attribute), attribute.component, attribute.stream, attribute.word});
    }
    key.varyings.reserve(varyings.size());
    for (const auto& varying : varyings) {
        key.varyings.push_back({varying.parameter, varying.components, u32(varying.flat)});
    }
    if (const auto found = mesh_interfaces.find(key); found != mesh_interfaces.end()) {
        return {found->second.mesh,
                stream,
                topology,
                vertex_words,
                found->second.shader,
                fragment,
                {attributes.begin(), attributes.end()},
                viewport_extent,
                last_provoking_vertex};
    }
    auto mesh = std::make_shared<const GeometryMeshSource>(
        GenerateGeometryMesh(stream, topology, vertex_words, attributes, varyings,
                             last_provoking_vertex, flip_vertex_y, viewport_extent, primitive_ids));
    if (const auto found = mesh_shaders.find(mesh->source); found != mesh_shaders.end()) {
        mesh_interfaces.emplace(std::move(key), CachedMesh{mesh, found->second});
        return {std::move(mesh),
                stream,
                topology,
                vertex_words,
                found->second,
                fragment,
                {attributes.begin(), attributes.end()},
                viewport_extent,
                last_provoking_vertex};
    }
    auto shader = std::make_shared<CompiledShader>();
    shader->translated.stage = Shader::SwStage::Geometry;
    shader->translated.source = mesh->source;
    shader->translated.entry_point = "metal4_geometry_mesh";
    shader->translated.buffer_count = viewport_extent[0] ? 4 : 3;
    shader->translated.mesh_workgroup_size = {32, 1, 1};
    shader->library = compiler.CompileLibraryAsync(mesh->source, "shadPS4:GeometryMesh");
    mesh_shaders.emplace(mesh->source, shader);
    mesh_interfaces.emplace(std::move(key), CachedMesh{mesh, shader});
    return {std::move(mesh),
            stream,
            topology,
            vertex_words,
            std::move(shader),
            fragment,
            {attributes.begin(), attributes.end()},
            viewport_extent,
            last_provoking_vertex};
}

void GeometryAssembly::Draw(const Result& result, const MeshShader& shader,
                            const RenderState& state, const GraphicsState& graphics,
                            GraphicsPipelineCache& pipelines, const ShaderCache::Variant* fragment,
                            const GraphicsPipelineState& pipeline_state,
                            ResourceBindings* fragment_bindings,
                            std::span<const std::array<float, 4>> viewports) {
    const auto& parameters = result.emissions.GetParameters();
    if (shader.fragment != fragment || shader.stream >= 4 || !shader.mesh ||
        parameters.mesh_vertices != shader.mesh->vertices ||
        parameters.vertex_words != shader.vertex_words ||
        parameters.topology[shader.stream] != shader.topology ||
        (shader.mesh->primitive_ids && !result.primitive_ids)) {
        throw std::runtime_error("Incompatible native geometry mesh layout");
    }
    const u64 jobs = u64{parameters.invocations} * parameters.mesh_counts[shader.stream];
    if (!jobs) {
        return;
    }
    const auto mesh_pipeline = pipelines.GetMeshPipeline(shader.shader, fragment, pipeline_state);
    const auto source = result.emissions.GetRange();
    const auto output = result.primitives;
    const auto indirect = result.indirect;
    if (!mesh_pipeline || !source.buffer || !output ||
        source.buffer.device != instance.GetDevice() ||
        output->GetBuffer().device != instance.GetDevice() || source.offset % sizeof(u32) ||
        output->GetOffset() % sizeof(u32) || source.size < parameters.EmissionSizeBytes() ||
        source.offset > source.buffer.length ||
        source.size > source.buffer.length - source.offset ||
        output->SizeBytes() < parameters.PrimitiveSizeBytes() ||
        output->GetOffset() > output->GetBuffer().length ||
        output->SizeBytes() > output->GetBuffer().length - output->GetOffset() ||
        (indirect &&
         (!indirect->GetBuffer() || indirect->GetBuffer().device != instance.GetDevice() ||
          indirect->GetOffset() % alignof(RasterCapturePackWork) ||
          indirect->SizeBytes() != sizeof(RasterCapturePackWork) ||
          indirect->GetOffset() > indirect->GetBuffer().length ||
          indirect->SizeBytes() > indirect->GetBuffer().length - indirect->GetOffset()))) {
        throw std::runtime_error("Invalid native geometry mesh buffers");
    }
    std::optional<StagingBufferPool::Allocation> parameter_data;
    if (!indirect) {
        parameter_data.emplace(staging.Allocate(sizeof(parameters)));
        std::memcpy(parameter_data->UploadData().data(), &parameters, sizeof(parameters));
    }
    const bool transform_viewport = shader.viewport_extent[0] && shader.viewport_extent[1];
    if (transform_viewport &&
        (shader.stream || viewports.empty() || viewports.size() > 16 ||
         viewports.size() != graphics.viewport_count ||
         parameters.max_vertices != parameters.mesh_vertices ||
         parameters.meshes_per_invocation != 1 || parameters.mesh_counts[0] != 1)) {
        throw std::runtime_error("Native direct TES viewport mesh layout is inconsistent");
    }
    ResourceBindings bindings{instance, transform_viewport ? 4u : 3u, 0, 0};
    if (transform_viewport) {
        GeometryMeshViewports data{u32(viewports.size())};
        std::ranges::copy(viewports, data.transforms.begin());
        auto uniforms = staging.Allocate(sizeof(data));
        std::memcpy(uniforms.UploadData().data(), &data, sizeof(data));
        bindings.SetBuffer(3, uniforms.GetBuffer(), uniforms.GetOffset());
        bindings.RetainUploadAllocation(std::move(uniforms));
    }
    bindings.SetBuffer(0, source.buffer, source.offset);
    bindings.SetBuffer(1, output->GetBuffer(), output->GetOffset());
    if (indirect) {
        bindings.SetBuffer(2, indirect->GetBuffer(), indirect->GetOffset());
        bindings.RetainUploadAllocation(indirect);
    } else {
        bindings.SetBuffer(2, parameter_data->GetBuffer(), parameter_data->GetOffset());
        bindings.RetainUploadAllocation(std::move(*parameter_data));
    }
    result.emissions.Retain(bindings);
    bindings.RetainUploadAllocation(output);
    scheduler.EndEncoding();
    const MTLStages render_stages =
        MTLStageMesh | MTLStageFragment | (state.visibility_buffer ? MTLStageTile : 0);
    static_cast<void>(commands.BindMeshGraphics(state, graphics, mesh_pipeline, nullptr, &bindings,
                                                fragment_bindings));
    scheduler.QueueBarrier(MTLStageAll, render_stages);
    const u32 width = u32(std::min(jobs, u64{65535}));
    const u32 height = u32((jobs + width - 1) / width);
    if (indirect)
        commands.DrawMeshIndirect(
            {indirect->GetBuffer(),
             indirect->GetOffset() + offsetof(RasterCapturePackWork, mesh_dispatch),
             3 * sizeof(u32)},
            MTLSizeMake(32, 1, 1));
    else
        commands.DrawMesh(MTLSizeMake(width, height, 1), MTLSizeMake(32, 1, 1));
    scheduler.ProducerBarrier(render_stages, MTLStageAll);
}

} // namespace Metal
