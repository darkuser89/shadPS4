// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_graphics_runtime.h"
#include "video_core/renderer_metal/metal_instance.h"

namespace Metal {
namespace {

constexpr u64 SourceMask(GeometryAddressSource source, u32 components = 1) {
    return ((u64{1} << components) - 1) << u32(source);
}

struct DrawDomains {
    std::vector<GeometryAddressInput> exports;
    std::vector<GeometryAddressInput> geometry;
    std::vector<GeometryAddressInput> copy;
};

DrawDomains GeometryDrawDomains(const GeometryDrawParameters& draw,
                                const GeometryRingLayout& rings) {
    if (!draw.export_items || draw.export_items % 64 || !draw.geometry_items ||
        draw.geometry_items % 64 || !draw.input_vertices || !draw.instances || !draw.primitives ||
        !draw.gs_invocations || rings.export_items != draw.export_items ||
        rings.geometry_items != draw.geometry_items || !rings.stream_components[0] ||
        !rings.max_vertices || !rings.export_wave_bytes || !rings.geometry_wave_bytes ||
        rings.export_wave_bytes % 256 || rings.geometry_wave_bytes % 256 ||
        rings.export_capacity > std::numeric_limits<u32>::max() ||
        rings.geometry_capacity > std::numeric_limits<u32>::max() ||
        u64{draw.export_items / 64} * rings.export_wave_bytes > rings.export_capacity ||
        u64{draw.geometry_items / 64} * rings.geometry_wave_bytes > rings.geometry_capacity ||
        draw != GeometryDrawLayout(AmdGpu::PrimitiveType(draw.topology), draw.input_vertices,
                                   draw.instances, draw.gs_invocations, draw.step_rate_0,
                                   draw.step_rate_1, draw.index_width, draw.index_offset)) {
        throw std::runtime_error("Invalid native geometry draw input domains");
    }
    using Attribute = Shader::IR::Attribute;
    DrawDomains result;
    const auto add = [](auto& inputs, Attribute attribute, u32 component, u64 maximum) {
        if (maximum > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Native geometry input domain exceeds register capacity");
        }
        inputs.push_back({attribute, component, {0, u32(maximum)}});
    };
    for (u32 scalar = 0; scalar < 16; ++scalar) {
        add(result.exports, Attribute::GeometryExportScalar, scalar,
            scalar ? 0 : u64{draw.export_items / 64 - 1} * rings.export_wave_bytes);
        add(result.copy, Attribute::GeometryCopyScalar, scalar, 0);
    }
    const u32 index_max = !draw.index_width       ? draw.input_vertices - 1
                          : draw.index_width == 2 ? 0xffffu
                                                  : 0xffffffffu;
    const u64 shifted_index_max = u64{index_max} + draw.index_offset;
    add(result.exports, Attribute::GeometryExportVector, 0,
        std::min<u64>(shifted_index_max, std::numeric_limits<u32>::max()));
    add(result.exports, Attribute::GeometryExportVector, 1,
        draw.step_rate_0 ? (draw.instances - 1) / draw.step_rate_0 : 0);
    add(result.exports, Attribute::GeometryExportVector, 2,
        draw.step_rate_1 ? (draw.instances - 1) / draw.step_rate_1 : 0);
    add(result.exports, Attribute::GeometryExportVector, 3, draw.instances - 1);
    add(result.exports, Attribute::GeometryInputActive, 0, 1);
    const u64 last_export = u64{draw.export_items / 64 - 1} * (rings.export_wave_bytes / 4) + 63;
    for (u32 vertex = 0; vertex < 6; ++vertex) {
        add(result.geometry, Attribute::GeometryVertexOffset, vertex, last_export);
    }
    const u64 last_geometry = u64{draw.geometry_items / 64 - 1} * rings.geometry_wave_bytes;
    add(result.geometry, Attribute::GeometryRingOffset, 0, last_geometry);
    add(result.geometry, Attribute::GeometryWaveId, 0, draw.geometry_items / 64 - 1);
    add(result.geometry, Attribute::PrimitiveId, 0, draw.primitives - 1);
    add(result.geometry, Attribute::InvocationId, 0, draw.gs_invocations - 1);
    add(result.geometry, Attribute::GeometryInputActive, 0, 1);
    const u64 last_copy = last_geometry / 4 +
                          u64{3} * rings.stream_components[0] * rings.max_vertices * 16 +
                          u64{rings.max_vertices - 1} * 16 + 15;
    add(result.copy, Attribute::GeometryCopyVector, 0, last_copy);
    for (u32 vector = 1; vector < 4; ++vector) {
        add(result.copy, Attribute::GeometryCopyVector, vector, 0);
    }
    add(result.copy, Attribute::GeometryCopyActive, 0, 1);
    return result;
}

void ValidateSystemRegisters(const AmdGpu::ShaderProgram& program) {
    u32 settings{};
    std::memcpy(&settings, reinterpret_cast<const u8*>(&program.settings) + sizeof(u32),
                sizeof(settings));
    if (settings != program.settings.num_user_regs << 1) {
        throw std::runtime_error("Native geometry system register configuration requires lowering");
    }
}

} // namespace

GeometryAssembly::PreparedEmissions GeometryAssembly::PrepareTessellationGenerate(
    ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
    const TessellationGeometry::Inputs& inputs, const GeometryAssemblyParameters& emissions,
    bool neo_mode, std::span<GuestResources::Stage* const> companions,
    std::span<const ImageInfo> attachments) {
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs ||
        !regs.gs_program.Address<VAddr>() || emissions.invocations != inputs.draw.geometry_items ||
        inputs.rings != BuildGeometryRingLayout(regs, inputs.draw) ||
        emissions.max_vertices != inputs.rings.max_vertices || !inputs.exports ||
        inputs.exports->SizeBytes() !=
            u64{inputs.draw.export_items} * regs.vgt_esgs_ring_itemsize * sizeof(u32)) {
        throw std::runtime_error("Native tessellation GS input layout differs from its ring");
    }
    ValidateSystemRegisters(regs.gs_program);
    Shader::Backend::Bindings binding{};
    const GeometryKernelConfig config{emissions, 0, true};
    auto& shader = shader_cache.GetShader(
        Shader::HwStage::Geometry, Shader::SwStage::Geometry,
        shader_cache.GetParams(regs.gs_program.Address<VAddr>(), regs.gs_program.user_data),
        BuildGraphicsRuntime(regs, Shader::HwStage::Geometry, shader_cache.GetProfile()), binding,
        false, false, &config);
    constexpr u64 sources = SourceMask(GeometryAddressSource::RingOffset) |
                            SourceMask(GeometryAddressSource::VertexOffset, 6) |
                            SourceMask(GeometryAddressSource::PrimitiveId) |
                            SourceMask(GeometryAddressSource::InvocationId) |
                            SourceMask(GeometryAddressSource::InputActive);
    if (!shader.shader.geometry || !shader.shader.geometry->address_origins ||
        shader.info.buffers.size() > Shader::NUM_BUFFERS ||
        (shader.shader.geometry->input_sources & ~sources)) {
        throw std::runtime_error("Native tessellation GS requires additional system inputs");
    }
    auto prepared = PrepareGenerate(shader_cache, resources, shader, inputs.registers, neo_mode,
                                    inputs.draw.geometry_invocations, {}, inputs.address_envelope,
                                    inputs.arguments);
    auto& stage = prepared.kernel.GetResources();
    const std::array stages{&stage};
    resources.ValidateAttachments(stages, attachments);
    bool output{};
    for (u32 index = 0; index < shader.info.buffers.size(); ++index) {
        if (!(shader.shader.geometry->address_origins->writes[index] &
              SourceMask(GeometryAddressSource::RingOffset))) {
            continue;
        }
        const auto sharp = ReadGuestBufferSharp(shader.info.buffers[index], shader.info);
        if (output || !sharp.Valid() || !sharp.base_address || sharp.base_address % sizeof(u32) ||
            !sharp.swizzle_enable || !sharp.add_tid_enable || sharp.GetIndexStride() != 16 ||
            sharp.GetElementSize() != 4 ||
            sharp.stride !=
                u64{inputs.rings.stream_components[0]} * inputs.rings.max_vertices * 4) {
            throw std::runtime_error("Native tessellation GS output ring layout requires lowering");
        }
        resources.ValidateGeometryOutputRing(stage, index, sharp.base_address,
                                             inputs.rings.geometry_capacity, attachments);
        output = true;
    }
    if (!output) {
        throw std::runtime_error("Native tessellation GS has no proven output ring");
    }
    prepared.kernel.ring_replay = PreparedKernel::RingReplay{
        inputs.guest_ring, inputs.exports, {companions.begin(), companions.end()}};
    return prepared;
}

GeometryAssembly::CopyMeshPlan GeometryAssembly::PrepareCopyMesh(
    ShaderCache& shader_cache, const AmdGpu::Regs& regs, const GeometryRingLayout& rings,
    const Shader::RuntimeInfo& runtime, u32 invocations, const ShaderCache::Variant* fragment,
    bool last_provoking_vertex, bool flip_vertex_y, bool compact_workgroups) {
    std::array<u32, 4> topology;
    u32 streams{};
    for (u32 stream = 0; stream < 4; ++stream) {
        topology[stream] = u32(runtime.hw.gs.out_primitive[stream]);
        streams |= rings.stream_components[stream] ? 1u << stream : 0;
    }
    auto plan = GeometryAssemblyLayout(invocations, rings.max_vertices, 1, 256, topology, streams);
    const auto copy_work =
        GeometryCopyWorkLayout(plan.invocations, plan.max_vertices, rings.stream_components[0],
                               u32(rings.geometry_capacity), 0);
    GeometryCopyConfig copy_config{plan, copy_work.work_items, 0, 0, compact_workgroups};
    const auto copy_runtime =
        BuildGraphicsRuntime(regs, Shader::HwStage::Vertex, shader_cache.GetProfile());
    const auto copy_params =
        shader_cache.GetParams(regs.vs_program.Address<VAddr>(), regs.vs_program.user_data);
    const auto get_copy = [&]() -> ShaderCache::Variant& {
        Shader::Backend::Bindings binding{};
        return shader_cache.GetShader(Shader::HwStage::Vertex, Shader::SwStage::Vertex, copy_params,
                                      copy_runtime, binding, false, false, nullptr, &copy_config);
    };
    auto* copy_shader = &get_copy();
    auto mesh =
        CompileMesh(0, topology[0], copy_shader->shader.geometry_copy->payload.vertex_words,
                    copy_shader->shader.geometry_copy->attributes, fragment, last_provoking_vertex,
                    flip_vertex_y, copy_shader->shader.geometry_copy->viewport_extent);
    if (plan.mesh_vertices != mesh.mesh->vertices) {
        plan = GeometryAssemblyLayout(invocations, rings.max_vertices, 1, mesh.mesh->vertices,
                                      topology, streams);
        copy_config.emissions = plan;
        copy_shader = &get_copy();
        mesh = CompileMesh(0, topology[0], copy_shader->shader.geometry_copy->payload.vertex_words,
                           copy_shader->shader.geometry_copy->attributes, fragment,
                           last_provoking_vertex, flip_vertex_y,
                           copy_shader->shader.geometry_copy->viewport_extent);
    }
    if (plan.mesh_vertices != mesh.mesh->vertices) {
        throw std::runtime_error("Native geometry mesh layout did not stabilize");
    }
    if (!topology[0] && std::ranges::none_of(copy_shader->shader.geometry_copy->attributes,
                                             [](const auto& attribute) {
                                                 return attribute.attribute ==
                                                        Shader::IR::Attribute::PointSize;
                                             })) {
        throw std::runtime_error("Native geometry points require a point-size export");
    }
    if (mesh.viewport_extent[0]) {
        static_cast<void>(ViewportPackingPipeline());
    }
    return {copy_config, copy_work, copy_shader, std::move(mesh)};
}

GeometryAssembly::PreparedTessellationDraw GeometryAssembly::PrepareTessellationDraw(
    ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
    const TessellationGeometry::Inputs& inputs, const BufferRange& errors, bool neo_mode,
    const ShaderCache::Variant* fragment, bool last_provoking_vertex, bool flip_vertex_y,
    std::span<GuestResources::Stage* const> companions, std::span<const ImageInfo> attachments) {
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs ||
        !regs.vs_program.Address<VAddr>() || regs.vgt_strmout_config.rast_stream ||
        (regs.vgt_strmout_config.use_rast_stream_mask &&
         regs.vgt_strmout_config.rast_stream_mask != 1) ||
        inputs.rings != BuildGeometryRingLayout(regs, inputs.draw) || !inputs.geometry_waves) {
        throw std::runtime_error("Native tessellation copy/mesh state requires lowering");
    }
    ValidateSystemRegisters(regs.vs_program);
    ValidateTessellationEvaluationState(regs, false);
    const auto runtime =
        BuildGraphicsRuntime(regs, Shader::HwStage::Geometry, shader_cache.GetProfile());
    auto plan =
        PrepareCopyMesh(shader_cache, regs, inputs.rings, runtime, inputs.draw.geometry_items,
                        fragment, last_provoking_vertex, flip_vertex_y, true);
    constexpr u64 sources = SourceMask(GeometryAddressSource::CopyVector) |
                            SourceMask(GeometryAddressSource::InputActive);
    if (!plan.shader->shader.geometry_copy ||
        (plan.shader->shader.geometry_copy->input_sources & ~sources)) {
        throw std::runtime_error("Native tessellation copy requires additional system inputs");
    }
    const auto domains = GeometryDrawDomains(inputs.draw, inputs.rings);
    const u64 wave_bytes = u64{plan.work.work_items / 64} * sizeof(GeometryCopyWave);
    if (!wave_bytes || wave_bytes > instance.GetDevice().maxBufferLength ||
        plan.config.InputSizeBytes() > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Native tessellation copy work exceeds device capacity");
    }
    auto waves =
        std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(wave_bytes, alignof(u32)));
    GeometryCopyWave wave{};
    wave.viewport = {regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f,
                     regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f,
                     regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f,
                     regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f};
    for (u32 index = 0; index < plan.work.work_items / 64; ++index) {
        std::memcpy(waves->UploadData().data() + u64{index} * sizeof(wave), &wave, sizeof(wave));
    }
    auto copy_inputs = PrepareCopyInputs(
        plan.work,
        {inputs.geometry_waves->GetBuffer(), inputs.geometry_waves->GetOffset(),
         inputs.geometry_waves->SizeBytes()},
        {waves->GetBuffer(), waves->GetOffset(), waves->SizeBytes()}, errors, true);
    auto geometry =
        PrepareTessellationGenerate(shader_cache, resources, regs, inputs, plan.config.emissions,
                                    neo_mode, companions, attachments);
    auto copy = PrepareCopy(shader_cache, resources, *plan.shader, geometry.emissions,
                            std::move(copy_inputs), neo_mode, 0, {}, domains.copy, errors);
    const std::array copy_stages{&copy.kernel.GetResources()};
    resources.ValidateAttachments(copy_stages, attachments);
    resources.ValidateGeometryCopyRing(geometry.kernel.GetResources(), copy.kernel.GetResources(),
                                       inputs.rings.geometry_capacity, attachments);
    copy.kernel.retained_inputs = {std::move(waves), inputs.geometry_waves};
    return {std::move(geometry), std::move(copy), std::move(plan.mesh)};
}

u32 GeometryAssembly::MaxTessellationPrimitives(ShaderCache& shader_cache, const AmdGpu::Regs& regs,
                                                const ShaderCache::Variant* fragment,
                                                bool last_provoking_vertex, bool flip_vertex_y) {
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHsEsGs ||
        !regs.vs_program.Address<VAddr>() || regs.vgt_strmout_config.rast_stream ||
        (regs.vgt_strmout_config.use_rast_stream_mask &&
         regs.vgt_strmout_config.rast_stream_mask != 1)) {
        throw std::runtime_error("Native tessellation downstream planning state requires lowering");
    }
    ValidateSystemRegisters(regs.vs_program);
    ValidateTessellationEvaluationState(regs, false);
    const auto runtime =
        BuildGraphicsRuntime(regs, Shader::HwStage::Geometry, shader_cache.GetProfile());
    const u32 width = regs.tess_config.topology == AmdGpu::TessellationTopology::Point  ? 1
                      : regs.tess_config.topology == AmdGpu::TessellationTopology::Line ? 2
                                                                                        : 3;
    const auto topology = width == 1   ? AmdGpu::PrimitiveType::PointList
                          : width == 2 ? AmdGpu::PrimitiveType::LineList
                                       : AmdGpu::PrimitiveType::TriangleList;
    const u32 gs_invocations = runtime.hw.gs.num_invocations;
    const auto rings = BuildGeometryRingLayout(
        regs, GeometryDrawLayout(topology, width, 1, gs_invocations, 0, 0, 0));
    auto plan = PrepareCopyMesh(shader_cache, regs, rings, runtime, 64, fragment,
                                last_provoking_vertex, flip_vertex_y, true);
    const auto& payload = plan.shader->shader.geometry_copy->payload;
    const std::array sizes{plan.config.emissions.EmissionSizeBytes(), plan.config.InputSizeBytes(),
                           payload.EmissionSizeBytes(), payload.PrimitiveSizeBytes(),
                           u64{plan.work.work_items / 64} * sizeof(GeometryCopyWave)};
    u64 groups = std::numeric_limits<u32>::max();
    u64 bytes{};
    for (const auto size : sizes) {
        if (!size) {
            throw std::runtime_error("Native tessellation downstream layout has no capacity");
        }
        groups = std::min(groups, instance.GetDevice().maxBufferLength / size);
        bytes += size;
    }
    groups = std::min(groups, std::max<u64>(1, (32 * 1024 * 1024) / bytes));
    const u64 primitives = groups * 64 / gs_invocations;
    if (!primitives) {
        throw std::runtime_error("Native tessellation copy/mesh cannot fit one GS group");
    }
    return u32(std::min<u64>(primitives, (std::numeric_limits<u32>::max() - 63) /
                                             std::max(width, gs_invocations)));
}

GeometryAssembly::PreparedDraw GeometryAssembly::PrepareGuestDraw(
    ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
    const GeometryDrawParameters& parameters, const BufferRange& indices, const BufferRange& errors,
    bool neo_mode, const ShaderCache::Variant* fragment, bool last_provoking_vertex,
    bool flip_vertex_y) {
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::EsGs ||
        !regs.es_program.Address<VAddr>() || !regs.gs_program.Address<VAddr>() ||
        !regs.vs_program.Address<VAddr>() || regs.vgt_strmout_config.rast_stream ||
        (regs.vgt_strmout_config.use_rast_stream_mask &&
         regs.vgt_strmout_config.rast_stream_mask != 1)) {
        throw std::runtime_error(
            "Native geometry draw requires additional stage or stream lowering");
    }
    ValidateSystemRegisters(regs.es_program);
    ValidateSystemRegisters(regs.gs_program);
    ValidateSystemRegisters(regs.vs_program);
    const auto rings = BuildGeometryRingLayout(regs, parameters);
    const auto domains = GeometryDrawDomains(parameters, rings);
    const auto gs_runtime =
        BuildGraphicsRuntime(regs, Shader::HwStage::Geometry, shader_cache.GetProfile());
    auto copy_mesh =
        PrepareCopyMesh(shader_cache, regs, rings, gs_runtime, parameters.geometry_items, fragment,
                        last_provoking_vertex, flip_vertex_y, false);
    const auto& plan = copy_mesh.config.emissions;
    const auto& copy_work = copy_mesh.work;
    auto* copy_shader = copy_mesh.shader;
    Shader::Backend::Bindings export_binding{}, geometry_binding{};
    const GeometryExportConfig export_config{parameters.export_items};
    auto& export_shader = shader_cache.GetShader(
        Shader::HwStage::Export, Shader::SwStage::Vertex,
        shader_cache.GetParams(regs.es_program.Address<VAddr>(), regs.es_program.user_data),
        BuildGraphicsRuntime(regs, Shader::HwStage::Export, shader_cache.GetProfile()),
        export_binding, false, false, nullptr, nullptr, &export_config);
    const GeometryKernelConfig geometry_config{plan};
    auto& geometry_shader = shader_cache.GetShader(
        Shader::HwStage::Geometry, Shader::SwStage::Geometry,
        shader_cache.GetParams(regs.gs_program.Address<VAddr>(), regs.gs_program.user_data),
        gs_runtime, geometry_binding, false, false, &geometry_config);
    constexpr u64 export_sources = SourceMask(GeometryAddressSource::ExportScalar) |
                                   SourceMask(GeometryAddressSource::ExportVector, 4) |
                                   SourceMask(GeometryAddressSource::InputActive);
    constexpr u64 geometry_sources = SourceMask(GeometryAddressSource::RingOffset) |
                                     SourceMask(GeometryAddressSource::VertexOffset, 6) |
                                     SourceMask(GeometryAddressSource::PrimitiveId) |
                                     SourceMask(GeometryAddressSource::InvocationId) |
                                     SourceMask(GeometryAddressSource::InputActive);
    constexpr u64 copy_sources = SourceMask(GeometryAddressSource::CopyVector) |
                                 SourceMask(GeometryAddressSource::InputActive);
    if ((export_shader.shader.geometry_export->input_sources & ~export_sources) ||
        (geometry_shader.shader.geometry->input_sources & ~geometry_sources) ||
        (copy_shader->shader.geometry_copy->input_sources & ~copy_sources)) {
        throw std::runtime_error("Native geometry shader requires additional system inputs");
    }
    ValidateGeometryRingBuffers(
        rings, export_shader.info, *export_shader.shader.geometry_export->address_origins,
        geometry_shader.info, *geometry_shader.shader.geometry->address_origins, copy_shader->info,
        *copy_shader->shader.geometry_copy->address_origins);
    const auto inputs = PrepareDrawInputs(parameters, indices, rings, GeometryExportWave{}, 0);
    const BufferRange work{inputs.work->GetBuffer(), inputs.work->GetOffset(),
                           inputs.work->SizeBytes()};
    auto kernel_inputs =
        PrepareKernelInputs(work, parameters.geometry_items, parameters.export_items,
                            regs.vgt_esgs_ring_itemsize, errors);
    std::vector<GeometryCopyWave> copy_waves(copy_work.work_items / 64);
    for (auto& wave : copy_waves) {
        wave.viewport = {regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f,
                         regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f,
                         regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f,
                         regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f};
    }
    const auto wave_bytes = std::as_bytes(std::span{copy_waves});
    if (wave_bytes.size() > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Native geometry copy waves exceed device capacity");
    }
    auto waves =
        std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(wave_bytes.size(), 16));
    std::memcpy(waves->UploadData().data(), wave_bytes.data(), wave_bytes.size());
    const BufferRange geometry_waves{inputs.geometry_waves->GetBuffer(),
                                     inputs.geometry_waves->GetOffset(),
                                     inputs.geometry_waves->SizeBytes()};
    auto copy_inputs =
        PrepareCopyInputs(copy_work, geometry_waves,
                          {waves->GetBuffer(), waves->GetOffset(), waves->SizeBytes()}, errors);
    auto exports = PrepareExport(shader_cache, resources, export_shader, inputs.exports, neo_mode,
                                 parameters.export_vertices, {}, domains.exports);
    auto geometry =
        PrepareGenerate(shader_cache, resources, geometry_shader, std::move(kernel_inputs),
                        neo_mode, parameters.geometry_invocations, {}, domains.geometry);
    auto copy = PrepareCopy(shader_cache, resources, *copy_shader, geometry.emissions,
                            std::move(copy_inputs), neo_mode, 0, {}, domains.copy);
    exports.retained_inputs = {inputs.export_waves, inputs.geometry_waves, inputs.work,
                               std::move(waves)};
    return {std::move(exports), std::move(geometry), std::move(copy), std::move(copy_mesh.mesh)};
}

ShaderCache::Variant& GeometryAssembly::RasterVertexShader(ShaderCache& shader_cache,
                                                           const AmdGpu::Regs& regs,
                                                           const RasterIndirectRegisters* indirect,
                                                           bool primitive_ids) {
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::Vs ||
        (regs.primitive_type != AmdGpu::PrimitiveType::QuadList &&
         regs.primitive_type != AmdGpu::PrimitiveType::RectList &&
         regs.primitive_type != AmdGpu::PrimitiveType::QuadStrip &&
         regs.primitive_type != AmdGpu::PrimitiveType::Polygon) ||
        !regs.vs_program.Address<VAddr>()) {
        throw std::runtime_error(
            "Native raster shader requires an ordinary rect/quad vertex stage");
    }
    ValidateSystemRegisters(regs.vs_program);
    Shader::Backend::Bindings bindings{};
    const GeometryExportConfig config{0, 0, true, true,
                                      indirect ? std::optional{*indirect} : std::nullopt};
    auto runtime = BuildGraphicsRuntime(regs, Shader::HwStage::Vertex, shader_cache.GetProfile());
    if (indirect) {
        indirect->Validate();
        runtime.sw.vs.vertex_sgpr_offset = indirect->vertex_sgpr_offset;
        runtime.sw.vs.instance_sgpr_offset = indirect->instance_sgpr_offset;
    }
    auto& shader = shader_cache.GetShader(
        Shader::HwStage::Vertex, Shader::SwStage::Vertex,
        shader_cache.GetParams(regs.vs_program.Address<VAddr>(), regs.vs_program.user_data),
        runtime, bindings, false, false, nullptr, nullptr, &config);
    const u64 sources = SourceMask(GeometryAddressSource::ExportScalar, 16) |
                        SourceMask(GeometryAddressSource::ExportVector, 4) |
                        SourceMask(GeometryAddressSource::InputActive) |
                        (indirect ? SourceMask(GeometryAddressSource::BaseVertex) |
                                        SourceMask(GeometryAddressSource::BaseInstance)
                                  : 0);
    if (!shader.shader.geometry_export || !shader.shader.geometry_export->raster ||
        (shader.shader.geometry_export->input_sources & ~sources)) {
        throw std::runtime_error("Native raster vertex shader requires additional system inputs");
    }
    if (primitive_ids || (regs.enable_primitive_id & 1))
        ValidateRasterPrimitiveIdVertexSources(shader.shader.geometry_export->input_sources);
    return shader;
}

u32 GeometryAssembly::MaxRasterBatchVertices(ShaderCache& shader_cache, const AmdGpu::Regs& regs,
                                             bool indexed, bool neo_mode,
                                             const RasterIndirectRegisters* indirect) {
    const auto& shader = RasterVertexShader(shader_cache, regs, indirect);
    const auto& layout = *shader.shader.geometry_export->raster;
    return RasterDrawBatchCapacity(layout.vertex_words, instance.GetDevice().maxBufferLength,
                                   regs.primitive_type == AmdGpu::PrimitiveType::RectList ? 192
                                                                                          : 64,
                                   RasterRestartEnabled(regs, indexed, neo_mode),
                                   u32(shader.geometry_export->HeaderSizeBytes()));
}

GeometryAssembly::PreparedRasterVertices GeometryAssembly::PrepareIndirectRasterVertices(
    ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
    const IndirectDraws::RasterCaptureInputs& inputs, bool neo_mode,
    std::span<const GuestBufferExtent> guest_extents, bool primitive_ids) {
    const auto& p = inputs.layout.parameters;
    const auto& window = p.capture.window;
    const u32 width = regs.primitive_type == AmdGpu::PrimitiveType::RectList    ? 3
                      : regs.primitive_type == AmdGpu::PrimitiveType::QuadList  ? 4
                      : regs.primitive_type == AmdGpu::PrimitiveType::QuadStrip ? 2
                                                                                : 1;
    if (!inputs.capture.owner || &inputs.capture.owner->scheduler != &scheduler ||
        &inputs.capture.owner->instance != &instance || !inputs.registers ||
        p.vertex_sgpr_offset > std::numeric_limits<u16>::max() ||
        p.instance_sgpr_offset > std::numeric_limits<u16>::max() ||
        p.capture.primitive_vertices != width ||
        p.capture.step_rate_0 != regs.vgt_instance_step_rate_0 ||
        p.capture.step_rate_1 != regs.vgt_instance_step_rate_1 ||
        p.capture.index_offset != regs.index_offset ||
        p.restart_enabled != u32(RasterRestartEnabled(regs, window.index_width != 0, neo_mode)) ||
        (p.restart_enabled && p.restart_index != regs.primitive_restart_index) ||
        inputs.registers->GetBuffer().device != instance.GetDevice() || !inputs.capture.batch ||
        inputs.capture.batch->GetBuffer().device != instance.GetDevice()) {
        throw std::runtime_error("Native indirect raster input context differs from its shader");
    }
    if (p.restart_enabled)
        ValidateRasterRestartIndex(window.index_width, p.restart_index, neo_mode);
    const RasterIndirectRegisters indirect{u16(p.vertex_sgpr_offset), u16(p.instance_sgpr_offset)};
    auto& shader = RasterVertexShader(shader_cache, regs, &indirect, primitive_ids);
    const auto& config = *shader.geometry_export;
    const auto& layout = *shader.shader.geometry_export->raster;
    const u32 maximum = RasterDrawBatchCapacity(
        layout.vertex_words, instance.GetDevice().maxBufferLength, width == 3 ? 192 : 64,
        p.restart_enabled, u32(config.HeaderSizeBytes()));
    if (window.capture_capacity > maximum ||
        inputs.registers->SizeBytes() != config.InputSizeBytes(window.capture_capacity)) {
        throw std::runtime_error("Native indirect raster capture exceeds its payload admission");
    }
    const auto domains = BuildIndirectRasterInputDomains(p.system_registers, p.capture.step_rate_0,
                                                         p.capture.step_rate_1);
    auto prepared =
        PrepareRasterVertices(shader_cache, resources, shader, inputs.registers, neo_mode,
                              window.capture_capacity, guest_extents, domains);
    prepared.kernel.arguments = inputs.capture.batch;
    prepared.kernel.argument_offset = offsetof(RasterCaptureBatch, dispatch);
    if (p.restart_enabled)
        prepared.indirect_restart = ScanIndirectRasterRestart(inputs);
    prepared.indirect_inputs = inputs;
    prepared.indirect_shader = &shader;
    if (shader.info.user_data.size() > prepared.indirect_user_data.size()) {
        throw std::runtime_error("Native indirect raster shader exceeds user data capacity");
    }
    std::ranges::copy(shader.info.user_data, prepared.indirect_user_data.begin());
    return prepared;
}

GeometryAssembly::IndirectRasterPlan GeometryAssembly::PrepareIndirectRasterPlan(
    ShaderCache& shader_cache, const AmdGpu::Regs& regs, const RasterIndirectRegisters& indirect,
    u32 index_width, const ShaderCache::Variant* fragment, bool last_provoking_vertex,
    bool flip_vertex_y, bool neo_mode, bool primitive_ids) {
    if (primitive_ids)
        static_cast<void>(RasterGuestPrimitiveIds(regs, 1));
    auto& shader = RasterVertexShader(shader_cache, regs, &indirect, primitive_ids);
    const auto& layout = *shader.shader.geometry_export->raster;
    std::array<u32, 2> viewport_extent{};
    if (regs.IsClipDisabled() && std::ranges::any_of(layout.attributes, [](const auto& output) {
            return output.attribute == Shader::IR::Attribute::ViewportIndex;
        })) {
        viewport_extent = {std::min(shader_cache.GetProfile().max_viewport_width / 2, 8192u),
                           std::min(shader_cache.GetProfile().max_viewport_height / 2, 8192u)};
    }
    auto mesh = CompileMesh(0, 2, layout.vertex_words, layout.attributes, fragment,
                            last_provoking_vertex, flip_vertex_y, viewport_extent, primitive_ids);
    const u32 primitive = regs.primitive_type == AmdGpu::PrimitiveType::RectList    ? 3
                          : regs.primitive_type == AmdGpu::PrimitiveType::QuadList  ? 4
                          : regs.primitive_type == AmdGpu::PrimitiveType::QuadStrip ? 2
                                                                                    : 1;
    const u32 capacity = RasterCaptureSubmissionCapacity(
        layout.vertex_words, mesh.mesh->vertices, primitive, index_width,
        RasterRestartEnabled(regs, index_width != 0, neo_mode),
        instance.GetDevice().maxBufferLength, viewport_extent[0] != 0, primitive_ids);
    return {std::move(mesh), capacity};
}

GeometryAssembly::PreparedRasterDraw GeometryAssembly::PrepareGuestRasterDraw(
    ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
    const RasterDrawParameters& parameters, const BufferRange& indices, bool neo_mode,
    const ShaderCache::Variant* fragment, bool last_provoking_vertex, bool flip_vertex_y,
    u32 index_source_offset) {
    const u32 alignment = regs.primitive_type == AmdGpu::PrimitiveType::RectList ? 192 : 64;
    if (!parameters.work_items || parameters.first_vertex % alignment ||
        u64{parameters.first_vertex} + parameters.vertices > regs.num_indices ||
        (parameters.first_vertex + parameters.vertices != regs.num_indices &&
         parameters.vertices % alignment) ||
        u64{parameters.first_instance} + parameters.instances > regs.num_instances.NumInstances() ||
        parameters.step_rate_0 != regs.vgt_instance_step_rate_0 ||
        parameters.step_rate_1 != regs.vgt_instance_step_rate_1 ||
        parameters.index_offset != regs.index_offset ||
        parameters != RasterDrawLayout(parameters.vertices, parameters.instances,
                                       parameters.step_rate_0, parameters.step_rate_1,
                                       parameters.index_width, parameters.index_count,
                                       parameters.index_offset, parameters.swap_mode,
                                       parameters.first_vertex, parameters.first_instance)) {
        throw std::runtime_error("Native raster draw requires matching ordinary vertex inputs");
    }
    const bool primitive_ids =
        fragment && fragment->info.loads.Get(Shader::IR::Attribute::PrimitiveId);
    if (primitive_ids)
        static_cast<void>(RasterGuestPrimitiveIds(regs));
    auto& shader = RasterVertexShader(shader_cache, regs, nullptr, primitive_ids);
    const auto& config = *shader.geometry_export;
    const auto& layout = *shader.shader.geometry_export->raster;
    const u64 bytes = shader.shader.geometry_export->RasterSizeBytes(config, parameters.work_items);
    const std::optional<u32> restart_index =
        RasterRestartEnabled(regs, parameters.index_width != 0, neo_mode)
            ? std::optional{regs.primitive_restart_index}
            : std::nullopt;
    if (restart_index)
        ValidateRasterRestartIndex(parameters.index_width, *restart_index, neo_mode);
    const u64 restart_bytes =
        restart_index ? BuildRasterRestartParameters(parameters, *restart_index).SizeBytes() +
                            RasterRestartStateSizeBytes(layout.vertex_words)
                      : 0;
    if (bytes + config.InputSizeBytes(parameters.work_items) +
            RasterIndexSourceSizeBytes(parameters.index_count, parameters.index_width,
                                       parameters.swap_mode, index_source_offset) +
            restart_bytes >
        std::min(u64{32} << 20, u64{instance.GetDevice().maxBufferLength})) {
        throw std::runtime_error("Native raster vertex capture requires smaller input batches");
    }
    const bool dynamic_viewport =
        regs.IsClipDisabled() && std::ranges::any_of(layout.attributes, [](const auto& output) {
            return output.attribute == Shader::IR::Attribute::ViewportIndex;
        });
    std::array<u32, 2> viewport_extent{};
    if (dynamic_viewport) {
        viewport_extent = {std::min(shader_cache.GetProfile().max_viewport_width / 2, 8192u),
                           std::min(shader_cache.GetProfile().max_viewport_height / 2, 8192u)};
    }
    auto mesh = CompileMesh(0, 2, layout.vertex_words, layout.attributes, fragment,
                            last_provoking_vertex, flip_vertex_y, viewport_extent, primitive_ids);
    const std::array viewport{
        regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f,
        regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f,
        regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f,
        regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f};
    auto inputs = PrepareRasterInputs(parameters, indices, viewport, {}, true, restart_index,
                                      index_source_offset);
    auto prepared = PrepareRasterVertices(shader_cache, resources, shader, inputs, neo_mode);
    return {std::move(prepared), std::move(mesh), std::move(inputs.restart)};
}

} // namespace Metal
