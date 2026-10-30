// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>
#include "shader_recompiler/profile.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_graphics_runtime.h"

namespace Metal {
namespace {

bool DualSource(AmdGpu::BlendControl::BlendFactor factor) {
    using Factor = AmdGpu::BlendControl::BlendFactor;
    return factor == Factor::Src1Color || factor == Factor::InvSrc1Color ||
           factor == Factor::Src1Alpha || factor == Factor::InvSrc1Alpha;
}

void SetProgramProperties(Shader::RuntimeInfo& runtime, const AmdGpu::ShaderProgram& program) {
    runtime.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
    runtime.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
    runtime.props.fp_round_mode32 = program.settings.fp_round_mode32;
    runtime.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    runtime.props.num_allocated_vgprs = program.NumVgprs();
    runtime.props.num_user_data = program.settings.num_user_regs;
    runtime.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
    runtime.props.dx10_clamp = program.settings.dx10_clamp;
}

u32 MapOutputs(std::array<Shader::OutputMap, 3>& outputs, const AmdGpu::Regs& regs) {
    u32 count{};
    const auto& output = regs.vs_output_control;
    using Output = Shader::Output;
    if (output.vs_out_misc_enable) {
        auto& row = outputs[count++];
        row = {output.use_vtx_point_size ? Output::PointSize : Output::None,
               output.use_vtx_edge_flag
                   ? Output::EdgeFlag
                   : (output.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None),
               output.use_vtx_kill_flag
                   ? Output::KillFlag
                   : (output.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None),
               output.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None};
    }
    for (u32 group = 0; group < 2; ++group) {
        if (!(group ? output.vs_out_ccdist1_enable : output.vs_out_ccdist0_enable))
            continue;
        auto& row = outputs[count++];
        for (u32 component = 0; component < 4; ++component) {
            const u32 index = group * 4 + component;
            row[component] = output.IsClipDistEnabled(index)
                                 ? static_cast<Output>(u32(Output::ClipDist0) + index)
                                 : (output.IsCullDistEnabled(index)
                                        ? static_cast<Output>(u32(Output::CullDist0) + index)
                                        : Output::None);
        }
    }
    return count;
}

} // namespace

Shader::RuntimeInfo BuildGraphicsRuntime(const AmdGpu::Regs& regs, Shader::HwStage stage,
                                         const Shader::Profile& profile) {
    Shader::RuntimeInfo runtime{};
    Shader::SwStage logical_stage;
    switch (stage) {
    case Shader::HwStage::Vertex:
    case Shader::HwStage::Export:
        logical_stage = Shader::SwStage::Vertex;
        break;
    case Shader::HwStage::Geometry:
        logical_stage = Shader::SwStage::Geometry;
        break;
    case Shader::HwStage::Fragment:
        logical_stage = Shader::SwStage::Fragment;
        break;
    default:
        throw std::runtime_error("Metal graphics runtime requires a graphics shader stage");
    }
    runtime.Initialize(stage, logical_stage);
    SetProgramProperties(runtime, *regs.ProgramForStage(u32(stage)));
    if (logical_stage == Shader::SwStage::Vertex) {
        runtime.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        runtime.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        if (stage == Shader::HwStage::Export) {
            runtime.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        } else {
            auto& info = runtime.hw.vs;
            info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
            info.emulate_depth_negative_one_to_one =
                regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
            info.clip_disable = regs.IsClipDisabled();
            info.num_outputs = MapOutputs(info.outputs, regs);
        }
        return runtime;
    }
    if (stage == Shader::HwStage::Geometry) {
        auto& info = runtime.hw.gs;
        info.num_outputs = MapOutputs(info.outputs, regs);
        info.output_vertices = regs.vgt_gs_max_vert_out;
        info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        info.in_primitive = regs.primitive_type;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            switch (regs.tess_config.topology) {
            case AmdGpu::TessellationTopology::Point:
                info.in_primitive = AmdGpu::PrimitiveType::PointList;
                break;
            case AmdGpu::TessellationTopology::Line:
                info.in_primitive = AmdGpu::PrimitiveType::LineList;
                break;
            case AmdGpu::TessellationTopology::TriangleCw:
            case AmdGpu::TessellationTopology::TriangleCcw:
                info.in_primitive = AmdGpu::PrimitiveType::TriangleList;
                break;
            default:
                throw std::runtime_error("Invalid Metal geometry tessellation topology");
            }
        }
        constexpr std::array shifts{0u, 8u, 16u, 22u};
        const u32 output = regs.vgt_gs_out_prim_type.raw;
        for (u32 stream = 0; stream < Shader::GsMaxOutputStreams; ++stream) {
            const u32 shift = output & (1u << 31) ? shifts[stream] : 0;
            info.out_primitive[stream] = AmdGpu::GsOutputPrimitiveType((output >> shift) & 63u);
        }
        info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.mode = regs.vgt_gs_mode.mode;
        return runtime;
    }
    auto& info = runtime.hw.fs;
    info.en_flags = regs.ps_input_ena;
    info.addr_flags = regs.ps_input_addr;
    info.num_inputs = regs.num_interp;
    if (info.num_inputs > info.inputs.size()) {
        throw std::runtime_error("Metal guest fragment input count exceeds the shader ABI");
    }
    info.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
    info.depth_before_shader = regs.depth_shader_control.depth_before_shader;
    info.num_samples =
        regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
            ? regs.aa_config.NumSamples()
            : 1;
    info.z_export_format = regs.z_export_format;
    info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                     ((regs.depth_shader_control.stencil_op_val_export_enable |
                       regs.depth_shader_control.stencil_test_val_export_enable)
                      << 1) |
                     (regs.depth_shader_control.mask_export_enable << 2) |
                     (regs.depth_shader_control.coverage_to_mask_enable << 3);
    const auto& blend = regs.blend_control[0];
    info.dual_source_blending =
        blend.enable && (DualSource(blend.color_src_factor) || DualSource(blend.color_dst_factor) ||
                         (blend.separate_alpha_blend && (DualSource(blend.alpha_src_factor) ||
                                                         DualSource(blend.alpha_dst_factor))));
    info.clip_distance_emulation =
        profile.needs_clip_distance_emulation && (regs.vs_output_control.clip_distance_enable ||
                                                  regs.clipper_control.user_clip_plane_enable);
    for (u32 index = 0; index < info.num_inputs; ++index) {
        const auto& input = regs.ps_inputs[index];
        info.inputs[index] = {u16(input.input_offset), u16(input.use_default),
                              u16(input.flat_shade), u16(input.default_value)};
    }
    for (u32 index = 0; index < info.color_buffers.size(); ++index) {
        const auto& color = regs.color_buffers[index];
        if (!color)
            continue;
        info.color_buffers[index] = {.data_format = color.GetDataFmt(),
                                     .num_format = color.GetNumberFmt(),
                                     .num_conversion = color.GetNumberConversion(),
                                     .export_format = regs.color_export_format.GetFormat(index),
                                     .blend_self_scale = 0,
                                     .swizzle = color.Swizzle()};
    }
    return runtime;
}

Shader::RuntimeInfo BuildTessellationRuntime(
    const AmdGpu::Regs& regs, Shader::HwStage stage,
    const Shader::TessellationDataConstantBuffer& constants) {
    const bool geometry = regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs;
    if ((!geometry && regs.stage_enable.raw != AmdGpu::ShaderStageEnable::LsHs) ||
        u32(regs.tess_config.type) > u32(AmdGpu::TessellationType::Quad) ||
        u32(regs.tess_config.partitioning) > u32(AmdGpu::TessellationPartitioning::FracEven) ||
        u32(regs.tess_config.topology) > u32(AmdGpu::TessellationTopology::TriangleCcw)) {
        throw std::runtime_error("Invalid Metal guest tessellation stage or configuration");
    }
    Shader::SwStage logical_stage;
    switch (stage) {
    case Shader::HwStage::Local:
        logical_stage = Shader::SwStage::Vertex;
        break;
    case Shader::HwStage::Hull:
        logical_stage = Shader::SwStage::TessellationControl;
        break;
    case Shader::HwStage::Vertex:
    case Shader::HwStage::Export:
        if ((stage == Shader::HwStage::Export) != geometry) {
            throw std::runtime_error(
                "Metal tessellation evaluation stage does not match its chain");
        }
        logical_stage = Shader::SwStage::TessellationEval;
        break;
    default:
        throw std::runtime_error("Invalid Metal tessellation shader stage");
    }
    Shader::RuntimeInfo runtime{};
    runtime.Initialize(stage, logical_stage);
    SetProgramProperties(runtime, *regs.ProgramForStage(u32(stage)));
    if (stage == Shader::HwStage::Local) {
        runtime.hw.ls.ls_stride = constants.ls_stride;
        runtime.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        runtime.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
    } else if (stage == Shader::HwStage::Hull) {
        auto& control = runtime.sw.tcs;
        control.hs_output_cp_stride = constants.hs_cp_stride;
        control.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        control.num_threads = regs.ls_hs_config.hs_output_control_points;
        control.tess_type = regs.tess_config.type;
        control.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        control.ls_stride = constants.ls_stride;
        control.hs_output_base = constants.hs_output_base;
    } else {
        auto& evaluation = runtime.sw.tes;
        evaluation.hs_output_cp_stride = constants.hs_cp_stride;
        evaluation.tess_type = regs.tess_config.type;
        evaluation.tess_topology = regs.tess_config.topology;
        evaluation.tess_partitioning = regs.tess_config.partitioning;
        if (stage == Shader::HwStage::Export) {
            runtime.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        } else {
            auto& vertex = runtime.hw.vs;
            vertex.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
            vertex.emulate_depth_negative_one_to_one =
                regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
            vertex.clip_disable = regs.IsClipDisabled();
            vertex.num_outputs = MapOutputs(vertex.outputs, regs);
        }
    }
    return runtime;
}

} // namespace Metal
