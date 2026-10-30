// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "common/alignment.h"
#include "core/libraries/kernel/memory.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_metal/metal_format.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_graphics_runtime.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

static bool ConfigureMultisampling(const AmdGpu::Regs& regs, RenderState& render,
                                   GraphicsPipelineState& pipeline) {
    const u32 samples = regs.aa_config.NumSamples();
    if (samples == 1) {
        return false;
    }
    if (samples > render.sample_positions.size() || !regs.mode_control.msaa_enable ||
        regs.aa_config.msaa_exposed_samples != regs.aa_config.msaa_num_samples) {
        throw std::runtime_error("Metal guest coverage sampling requires native EQAA lowering");
    }
    constexpr u32 DbEqaa = 0xA201;
    constexpr u32 DbAlphaToMask = 0xA2DC;
    constexpr u32 PaScModeControl1 = 0xA293;
    constexpr u32 PaScSampleLocations = 0xA2FE;
    constexpr u32 PaScSampleMasks = 0xA30E;
    const u32 eqaa = regs.reg_array[DbEqaa];
    const bool sample_rate = (regs.reg_array[PaScModeControl1] & (1u << 16)) != 0;
    const u32 iterations = sample_rate ? 1u << ((eqaa >> 4) & 7) : 1;
    if ((eqaa & 7) != regs.aa_config.msaa_num_samples || (eqaa & 0x0F000000) ||
        (iterations != 1 && iterations != samples) ||
        (regs.depth_shader_control.mask_export_enable &&
         ((eqaa >> 8) & 7) != regs.aa_config.msaa_num_samples)) {
        throw std::runtime_error("Metal guest sample iteration requires native EQAA lowering");
    }
    if (((regs.reg_array[DbAlphaToMask] & 1) && !regs.depth_shader_control.alpha_to_mask_disable) ||
        regs.depth_shader_control.coverage_to_mask_enable) {
        throw std::runtime_error("Metal guest alpha/coverage masks require native sample lowering");
    }
    const u32 sample_mask = (1u << samples) - 1;
    for (u32 pixel = 0; pixel < 4; ++pixel) {
        const u32 mask = regs.reg_array[PaScSampleMasks + pixel / 2] >> ((pixel % 2) * 16);
        if ((mask & sample_mask) != sample_mask) {
            throw std::runtime_error("Metal guest raster sample masks require native lowering");
        }
        for (u32 sample = 0; sample < samples; ++sample) {
            const u32 packed =
                regs.reg_array[PaScSampleLocations + pixel * 4 + sample / 4] >> ((sample % 4) * 8);
            const std::array<u8, 2> position{u8((packed & 15) ^ 8), u8(((packed >> 4) & 15) ^ 8)};
            if (pixel && render.sample_positions[sample] != position) {
                throw std::runtime_error(
                    "Metal guest quad sample positions require native raster lowering");
            }
            render.sample_positions[sample] = position;
        }
    }
    render.num_samples = pipeline.sample_count = samples;
    render.custom_sample_positions = true;
    return iterations > 1;
}

static void ValidateMultisampleShader(const AmdGpu::Regs& regs, const Shader::Info& info,
                                      bool sample_rate) {
    if (!sample_rate &&
        (info.loads.Get(Shader::IR::Attribute::SampleIndex) ||
         regs.ps_input_addr.persp_sample_ena || regs.ps_input_addr.linear_sample_ena ||
         std::ranges::any_of(info.fs_interpolation, [](const auto& input) {
             return input.auxiliary == Shader::Qualifier::Sample;
         }))) {
        throw std::runtime_error("Metal guest pixel-rate sample inputs require native lowering");
    }
    if (info.loads.GetAny(Shader::IR::Attribute::FragCoord)) {
        const u32 control = std::bit_cast<u32>(regs.barycentric_control);
        if ((control & (1u << 20)) || ((control >> 16) & 3) != (sample_rate ? 2u : 0u)) {
            throw std::runtime_error(
                "Metal guest multisample fragment positions require native interpolation");
        }
    }
}

static MTLPrimitiveType Primitive(AmdGpu::PrimitiveType type) {
    switch (type) {
    case AmdGpu::PrimitiveType::PointList:
        return MTLPrimitiveTypePoint;
    case AmdGpu::PrimitiveType::LineList:
    case AmdGpu::PrimitiveType::LineLoop:
        return MTLPrimitiveTypeLine;
    case AmdGpu::PrimitiveType::LineStrip:
        return MTLPrimitiveTypeLineStrip;
    case AmdGpu::PrimitiveType::TriangleFan:
    case AmdGpu::PrimitiveType::TriangleList:
    case AmdGpu::PrimitiveType::QuadList:
    case AmdGpu::PrimitiveType::RectList:
    case AmdGpu::PrimitiveType::QuadStrip:
    case AmdGpu::PrimitiveType::Polygon:
        return MTLPrimitiveTypeTriangle;
    case AmdGpu::PrimitiveType::TriangleStrip:
        return MTLPrimitiveTypeTriangleStrip;
    default:
        throw std::runtime_error("Metal guest topology requires native primitive lowering");
    }
}

static MTLBlendFactor BlendFactor(AmdGpu::BlendControl::BlendFactor factor) {
    using Factor = AmdGpu::BlendControl::BlendFactor;
    switch (factor) {
    case Factor::Zero:
        return MTLBlendFactorZero;
    case Factor::One:
        return MTLBlendFactorOne;
    case Factor::SrcColor:
        return MTLBlendFactorSourceColor;
    case Factor::OneMinusSrcColor:
        return MTLBlendFactorOneMinusSourceColor;
    case Factor::SrcAlpha:
        return MTLBlendFactorSourceAlpha;
    case Factor::OneMinusSrcAlpha:
        return MTLBlendFactorOneMinusSourceAlpha;
    case Factor::DstAlpha:
        return MTLBlendFactorDestinationAlpha;
    case Factor::OneMinusDstAlpha:
        return MTLBlendFactorOneMinusDestinationAlpha;
    case Factor::DstColor:
        return MTLBlendFactorDestinationColor;
    case Factor::OneMinusDstColor:
        return MTLBlendFactorOneMinusDestinationColor;
    case Factor::SrcAlphaSaturate:
        return MTLBlendFactorSourceAlphaSaturated;
    case Factor::ConstantColor:
        return MTLBlendFactorBlendColor;
    case Factor::OneMinusConstantColor:
        return MTLBlendFactorOneMinusBlendColor;
    case Factor::ConstantAlpha:
        return MTLBlendFactorBlendAlpha;
    case Factor::OneMinusConstantAlpha:
        return MTLBlendFactorOneMinusBlendAlpha;
    case Factor::Src1Color:
        return MTLBlendFactorSource1Color;
    case Factor::InvSrc1Color:
        return MTLBlendFactorOneMinusSource1Color;
    case Factor::Src1Alpha:
        return MTLBlendFactorSource1Alpha;
    case Factor::InvSrc1Alpha:
        return MTLBlendFactorOneMinusSource1Alpha;
    default:
        throw std::runtime_error("Invalid Metal guest blend factor");
    }
}

static MTLBlendOperation BlendOperation(AmdGpu::BlendControl::BlendFunc operation,
                                        AmdGpu::BlendControl::BlendFactor source,
                                        AmdGpu::BlendControl::BlendFactor destination) {
    using Operation = AmdGpu::BlendControl::BlendFunc;
    using Factor = AmdGpu::BlendControl::BlendFactor;
    if ((operation == Operation::Min || operation == Operation::Max) &&
        (source != Factor::One || destination != Factor::One)) {
        throw std::runtime_error("Metal scaled min/max blending requires native shader lowering");
    }
    switch (operation) {
    case Operation::Add:
        return MTLBlendOperationAdd;
    case Operation::Subtract:
        return MTLBlendOperationSubtract;
    case Operation::ReverseSubtract:
        return MTLBlendOperationReverseSubtract;
    case Operation::Min:
        return MTLBlendOperationMin;
    case Operation::Max:
        return MTLBlendOperationMax;
    default:
        throw std::runtime_error("Invalid Metal guest blend operation");
    }
}

GraphicsRenderer::GraphicsRenderer(const Instance& instance_, Scheduler& scheduler_,
                                   ShaderCache& shader_cache_, GraphicsPipelineCache& pipelines_,
                                   GuestResources& resources_, BufferCache& buffer_cache_,
                                   TextureCache& texture_cache_, AmdGpu::Liverpool& liverpool_)
    : instance{instance_}, scheduler{scheduler_}, shader_cache{shader_cache_},
      pipelines{pipelines_}, resources{resources_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, liverpool{liverpool_}, memory{*Core::Memory::Instance()},
      commands{instance, scheduler}, compiler{instance}, indirect_draws{instance, scheduler},
      primitive_assembly{instance, scheduler}, geometry_assembly{instance, scheduler},
      tessellation{instance, scheduler, buffer_cache, shader_cache, resources},
      tessellation_geometry{instance, scheduler}, visibility{instance, scheduler},
      metadata_coverage{instance, scheduler} {
    depth_fragment.info.hw_stage = Shader::HwStage::Fragment;
    depth_fragment.info.sw_stage = Shader::SwStage::Fragment;
    depth_fragment.specialization.runtime_info.Initialize(Shader::HwStage::Fragment,
                                                          Shader::SwStage::Fragment);
    auto& shader = depth_fragment.shader;
    shader.translated.stage = Shader::SwStage::Fragment;
    shader.translated.entry_point = "metal4_depth_fragment";
    shader.translated.source =
        "#include <metal_stdlib>\nfragment void metal4_depth_fragment() {}\n";
    shader.library =
        compiler.CompileLibraryAsync(shader.translated.source, shader.translated.entry_point);
    coverage_fragment.info.hw_stage = Shader::HwStage::Fragment;
    coverage_fragment.info.sw_stage = Shader::SwStage::Fragment;
    coverage_fragment.info.mrt_mask = 1;
    coverage_fragment.specialization.runtime_info.Initialize(Shader::HwStage::Fragment,
                                                             Shader::SwStage::Fragment);
    auto& coverage = coverage_fragment.shader;
    coverage.translated.stage = Shader::SwStage::Fragment;
    coverage.translated.entry_point = "metal4_metadata_coverage";
    coverage.translated.source =
        "#include <metal_stdlib>\n"
        "fragment metal::float4 metal4_metadata_coverage() { return metal::float4(1.0f); }\n";
    coverage.library =
        compiler.CompileLibraryAsync(coverage.translated.source, coverage.translated.entry_point);
}

Shader::RuntimeInfo GraphicsRenderer::Runtime(Shader::HwStage stage) const {
    return BuildGraphicsRuntime(liverpool.regs, stage, shader_cache.GetProfile());
}

static MTLStencilOperation StencilOperation(AmdGpu::StencilFunc operation) {
    using Op = AmdGpu::StencilFunc;
    switch (operation) {
    case Op::Keep:
        return MTLStencilOperationKeep;
    case Op::Zero:
        return MTLStencilOperationZero;
    case Op::ReplaceTest:
    case Op::ReplaceOp:
        return MTLStencilOperationReplace;
    case Op::AddClamp:
        return MTLStencilOperationIncrementClamp;
    case Op::SubClamp:
        return MTLStencilOperationDecrementClamp;
    case Op::Invert:
        return MTLStencilOperationInvert;
    case Op::AddWrap:
        return MTLStencilOperationIncrementWrap;
    case Op::SubWrap:
        return MTLStencilOperationDecrementWrap;
    default:
        throw std::runtime_error("Metal stencil operation requires native stencil lowering");
    }
}

id<MTLDepthStencilState> GraphicsRenderer::DepthStencil(GraphicsState& graphics) {
    const auto& regs = liverpool.regs;
    const auto& control = regs.depth_control;
    std::array<u32, 15> key{};
    key[0] = control.depth_enable && regs.depth_buffer.DepthValid();
    key[1] = key[0] && control.depth_write_enable && !regs.depth_view.z_read_only;
    key[2] = key[0] ? u32(control.depth_func) : u32(AmdGpu::CompareFunc::Always);
    const bool stencil = control.stencil_enable && regs.depth_buffer.StencilValid();
    const auto face = [&](u32 offset, AmdGpu::CompareFunc compare,
                          const AmdGpu::StencilRefMask& reference,
                          std::array<AmdGpu::StencilFunc, 3> operations, u32& native_reference) {
        const bool use_op =
            std::ranges::find(operations, AmdGpu::StencilFunc::ReplaceOp) != operations.end();
        const bool use_test =
            std::ranges::find(operations, AmdGpu::StencilFunc::ReplaceTest) != operations.end();
        if (use_op && reference.stencil_op_val != reference.stencil_test_val &&
            (use_test ||
             (compare != AmdGpu::CompareFunc::Always && compare != AmdGpu::CompareFunc::Never))) {
            throw std::runtime_error(
                "Metal split stencil references require native stencil lowering");
        }
        native_reference = use_op ? reference.stencil_op_val : reference.stencil_test_val;
        key[offset] = u32(compare);
        for (u32 index = 0; index < 3; ++index) {
            key[offset + index + 1] = StencilOperation(operations[index]);
        }
        key[offset + 4] = reference.stencil_mask;
        key[offset + 5] = regs.depth_view.stencil_read_only ? 0 : reference.stencil_write_mask;
    };
    if (stencil) {
        const auto& ops = regs.stencil_control;
        face(3, control.stencil_ref_func, regs.stencil_ref_front,
             {ops.stencil_fail_front, ops.stencil_zpass_front, ops.stencil_zfail_front},
             graphics.stencil_front_reference);
        if (control.backface_enable) {
            face(9, control.stencil_bf_func, regs.stencil_ref_back,
                 {ops.stencil_fail_back, ops.stencil_zpass_back, ops.stencil_zfail_back},
                 graphics.stencil_back_reference);
        } else {
            std::copy_n(key.begin() + 3, 6, key.begin() + 9);
            graphics.stencil_back_reference = graphics.stencil_front_reference;
        }
    } else {
        key[3] = key[9] = u32(AmdGpu::CompareFunc::Always);
    }
    if (const auto found = depth_states.find(key); found != depth_states.end())
        return found->second;
    MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
    descriptor.depthCompareFunction = static_cast<MTLCompareFunction>(key[2]);
    descriptor.depthWriteEnabled = key[1];
    const auto make_face = [&](u32 offset) {
        MTLStencilDescriptor* result = [MTLStencilDescriptor new];
        result.stencilCompareFunction = static_cast<MTLCompareFunction>(key[offset]);
        result.stencilFailureOperation = static_cast<MTLStencilOperation>(key[offset + 1]);
        result.depthStencilPassOperation = static_cast<MTLStencilOperation>(key[offset + 2]);
        result.depthFailureOperation = static_cast<MTLStencilOperation>(key[offset + 3]);
        result.readMask = key[offset + 4];
        result.writeMask = key[offset + 5];
        return result;
    };
    descriptor.frontFaceStencil = make_face(3);
    descriptor.backFaceStencil = make_face(9);
    auto state = instance.CreateDepthStencilState(descriptor);
    depth_states.emplace(key, state);
    return state;
}

GraphicsState GraphicsRenderer::State(u32 width, u32 height, bool& flip_vertex_y) {
    const auto& regs = liverpool.regs;
    const auto& polygon = regs.polygon_control;
    GraphicsState state;
    state.viewport_count = regs.vs_output_control.use_vtx_viewport_idx ? AmdGpu::NUM_VIEWPORTS : 1;
    std::optional<bool> orientation;
    for (u32 index = 0; index < state.viewport_count; ++index) {
        const auto& vp = regs.viewports[index];
        const auto& control = regs.viewport_control;
        const double xscale = control.xscale_enable ? vp.xscale : 1;
        const double yscale = control.yscale_enable ? vp.yscale : 1;
        const double xoffset = control.xoffset_enable ? vp.xoffset : 0;
        const double yoffset = control.yoffset_enable ? vp.yoffset : 0;
        const double zscale = control.zscale_enable ? vp.zscale : 1;
        const double zoffset = control.zoffset_enable ? vp.zoffset : 0;
        auto& viewport = state.viewports[index];
        viewport.min_depth = regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW
                                 ? zoffset - zscale
                                 : zoffset;
        viewport.max_depth = zoffset + zscale;
        if (!std::isfinite(xscale) || !std::isfinite(yscale) || !std::isfinite(xoffset) ||
            !std::isfinite(yoffset) || !std::isfinite(viewport.min_depth) ||
            !std::isfinite(viewport.max_depth) || viewport.min_depth < 0 ||
            viewport.min_depth > 1 || viewport.max_depth < 0 || viewport.max_depth > 1) {
            throw std::runtime_error("Metal unrestricted guest viewport requires shader lowering");
        }
        if (regs.IsClipDisabled()) {
            viewport.width = viewport.height = 16384;
            orientation = true;
        } else if (xscale == 0 || yscale == 0) {
            viewport.width = viewport.height = 1;
            continue;
        } else {
            if (xscale < 0)
                throw std::runtime_error("Metal reversed X viewport requires shader lowering");
            const bool flipped = yscale > 0;
            if (orientation && *orientation != flipped) {
                throw std::runtime_error(
                    "Metal mixed viewport orientations require shader lowering");
            }
            orientation = flipped;
            viewport.x = xoffset - xscale;
            viewport.y = yoffset - std::abs(yscale);
            viewport.width = xscale * 2;
            viewport.height = std::abs(yscale) * 2;
        }
        const auto& screen = regs.screen_scissor;
        const auto shifted = [&](const auto& scissor, bool x, bool upper) -> s64 {
            const s64 value = upper ? (x ? scissor.top_left_x : scissor.top_left_y)
                                    : (x ? scissor.bottom_right_x : scissor.bottom_right_y);
            return value + (scissor.window_offset_disable
                                ? 0
                                : (x ? regs.window_offset.window_x_offset
                                     : regs.window_offset.window_y_offset));
        };
        s64 left =
            std::max({s64{0}, s64(screen.top_left_x), shifted(regs.window_scissor, true, true),
                      shifted(regs.generic_scissor, true, true)});
        s64 top =
            std::max({s64{0}, s64(screen.top_left_y), shifted(regs.window_scissor, false, true),
                      shifted(regs.generic_scissor, false, true)});
        s64 right = std::min({s64(width), s64(screen.bottom_right_x),
                              shifted(regs.window_scissor, true, false),
                              shifted(regs.generic_scissor, true, false)});
        s64 bottom = std::min({s64(height), s64(screen.bottom_right_y),
                               shifted(regs.window_scissor, false, false),
                               shifted(regs.generic_scissor, false, false)});
        if (regs.mode_control.vport_scissor_enable) {
            const auto& scissor = regs.viewport_scissors[index];
            left = std::max(left, shifted(scissor, true, true));
            top = std::max(top, shifted(scissor, false, true));
            right = std::min(right, shifted(scissor, true, false));
            bottom = std::min(bottom, shifted(scissor, false, false));
        }
        left = std::min(left, s64(width));
        top = std::min(top, s64(height));
        state.scissors[index] = {u32(left), u32(top), u32(std::max(left, right) - left),
                                 u32(std::max(top, bottom) - top)};
    }
    flip_vertex_y = orientation.value_or(false);
    state.cull_mode = polygon.cull_front ? MTLCullModeFront
                                         : (polygon.cull_back ? MTLCullModeBack : MTLCullModeNone);
    state.front_face = polygon.front_face == AmdGpu::FrontFace::Clockwise
                           ? MTLWindingClockwise
                           : MTLWindingCounterClockwise;
    state.depth_clip =
        regs.clipper_control.ZclipEnable() ? MTLDepthClipModeClip : MTLDepthClipModeClamp;
    state.fill_mode = polygon.PolyMode() == AmdGpu::PolygonMode::Line ? MTLTriangleFillModeLines
                                                                      : MTLTriangleFillModeFill;
    if (polygon.NeedsBias()) {
        if (polygon.enable_polygon_offset_front && polygon.enable_polygon_offset_back &&
            (regs.poly_offset.front_offset != regs.poly_offset.back_offset ||
             regs.poly_offset.front_scale != regs.poly_offset.back_scale)) {
            throw std::runtime_error("Metal per-face depth bias requires native raster lowering");
        }
        const bool front = polygon.enable_polygon_offset_front;
        state.depth_bias = {front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
                            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) /
                                16,
                            regs.poly_offset.depth_bias};
    }
    state.blend_constant = regs.blend_constants;
    if (regs.depth_control.depth_bounds_enable) {
        state.depth_bounds = {regs.depth_bounds_min, regs.depth_bounds_max, true};
    }
    state.depth_stencil = DepthStencil(state);
    return state;
}

namespace {
static_assert(sizeof(AmdGpu::DrawIndirectArgs) == sizeof(MTLDrawPrimitivesIndirectArguments));
static_assert(sizeof(AmdGpu::DrawIndexedIndirectArgs) ==
              sizeof(MTLDrawIndexedPrimitivesIndirectArguments));
static_assert(offsetof(AmdGpu::DrawIndirectArgs, vertex_count_per_instance) ==
              offsetof(MTLDrawPrimitivesIndirectArguments, vertexCount));
static_assert(offsetof(AmdGpu::DrawIndirectArgs, instance_count) ==
              offsetof(MTLDrawPrimitivesIndirectArguments, instanceCount));
static_assert(offsetof(AmdGpu::DrawIndirectArgs, start_vertex_location) ==
              offsetof(MTLDrawPrimitivesIndirectArguments, vertexStart));
static_assert(offsetof(AmdGpu::DrawIndirectArgs, start_instance_location) ==
              offsetof(MTLDrawPrimitivesIndirectArguments, baseInstance));
static_assert(offsetof(AmdGpu::DrawIndexedIndirectArgs, index_count_per_instance) ==
              offsetof(MTLDrawIndexedPrimitivesIndirectArguments, indexCount));
static_assert(offsetof(AmdGpu::DrawIndexedIndirectArgs, instance_count) ==
              offsetof(MTLDrawIndexedPrimitivesIndirectArguments, instanceCount));
static_assert(offsetof(AmdGpu::DrawIndexedIndirectArgs, start_index_location) ==
              offsetof(MTLDrawIndexedPrimitivesIndirectArguments, indexStart));
static_assert(offsetof(AmdGpu::DrawIndexedIndirectArgs, base_vertex_location) ==
              offsetof(MTLDrawIndexedPrimitivesIndirectArguments, baseVertex));
static_assert(offsetof(AmdGpu::DrawIndexedIndirectArgs, start_instance_location) ==
              offsetof(MTLDrawIndexedPrimitivesIndirectArguments, baseInstance));

} // namespace

void GraphicsRenderer::CheckIndirectErrors() {
    indirect_draws.CheckErrors();
}

void GraphicsRenderer::DrawIndirect(bool indexed, VAddr address, u32 offset, u32 stride,
                                    u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                                    u16 instance_sgpr_offset, bool neo_mode) {
    if (!max_count)
        return;
    if (address >= (u64{1} << 40) || offset >= (u64{1} << 40) - address) {
        throw std::runtime_error("Invalid Metal guest indirect argument address");
    }
    const IndirectDraw indirect{address + offset,   stride,
                                max_count,          count_address,
                                vertex_sgpr_offset, instance_sgpr_offset};
    Draw(indexed, 0, neo_mode, &indirect);
}

void GraphicsRenderer::Draw(bool indexed, u32 index_offset, bool neo_mode) {
    Draw(indexed, index_offset, neo_mode, nullptr);
}

void GraphicsRenderer::ResolveColor(bool neo_mode) {
    const auto& colors = liverpool.regs.color_buffers;
    for (u32 index = 0; index < 2; ++index) {
        const auto& color = colors[index];
        if (!color || color.info.endian != AmdGpu::ColorBuffer::EndianSwap::None ||
            color.info.dcc_enable || color.attrib.force_dst_alpha_1 ||
            color.attrib.num_samples_log2 != color.attrib.num_fragments_log2 ||
            color.BaseSlice() >= color.NumSlices()) {
            throw std::runtime_error("Metal color resolve requires compatible guest surfaces");
        }
    }
    const u32 slices = colors[0].NumSlices() - colors[0].BaseSlice();
    if (slices != colors[1].NumSlices() - colors[1].BaseSlice() ||
        colors[0].info.comp_swap != colors[1].info.comp_swap) {
        throw std::runtime_error("Metal color resolve requires matching layers and components");
    }
    texture_cache.ResolveColor(
        ImageInfo::FromColor(colors[0], liverpool.last_cb_extent[0], neo_mode),
        colors[0].BaseSlice(),
        ImageInfo::FromColor(colors[1], liverpool.last_cb_extent[1], neo_mode),
        colors[1].BaseSlice(), slices);
}

void GraphicsRenderer::Draw(bool indexed, u32 index_offset, bool neo_mode,
                            const IndirectDraw* indirect) {
    const auto& regs = liverpool.regs;
    if ((!indirect && !regs.num_indices) || regs.primitive_type == AmdGpu::PrimitiveType::None)
        return;
    using ColorMode = AmdGpu::ColorControl::OperationMode;
    using LogicOp = AmdGpu::ColorControl::LogicOp;
    const bool metadata_draw = regs.color_control.mode == ColorMode::EliminateFastClear ||
                               regs.color_control.mode == ColorMode::FmaskDecompress;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        if (indirect) {
            throw std::runtime_error("Metal indirect color resolve requires GPU command lowering");
        }
        ResolveColor(neo_mode);
        return;
    }
    const bool tessellation_draw = regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs;
    const bool raster_tessellation_draw = regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHs;
    const bool raster_list_draw = regs.stage_enable.raw == AmdGpu::ShaderStageEnable::Vs &&
                                  (regs.primitive_type == AmdGpu::PrimitiveType::QuadList ||
                                   regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
                                   regs.primitive_type == AmdGpu::PrimitiveType::QuadStrip ||
                                   regs.primitive_type == AmdGpu::PrimitiveType::Polygon);
    const bool geometry_draw =
        regs.stage_enable.raw == AmdGpu::ShaderStageEnable::EsGs || tessellation_draw;
    if ((!geometry_draw && !raster_tessellation_draw &&
         regs.stage_enable.raw != AmdGpu::ShaderStageEnable::Vs) ||
        regs.depth_control.enable_color_writes_on_depth_fail ||
        regs.depth_control.disable_color_writes_on_depth_pass ||
        regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable || regs.depth_render_control.depth_copy ||
        regs.depth_render_control.stencil_copy || regs.depth_render_control.resummarize_enable ||
        regs.clipper_control.zclip_near_disable != regs.clipper_control.zclip_far_disable ||
        regs.polygon_control.PolyMode() == AmdGpu::PolygonMode::Point ||
        (regs.polygon_control.enable_polygon_mode &&
         regs.polygon_control.polygon_mode_front != regs.polygon_control.polygon_mode_back) ||
        (regs.polygon_control.enable_window_offset &&
         (regs.window_offset.window_x_offset || regs.window_offset.window_y_offset))) {
        throw std::runtime_error(
            "Metal guest draw state requires additional native stage lowering");
    }
    if (regs.depth_control.depth_bounds_enable && !instance.SupportsDepthBounds()) {
        throw std::runtime_error("Metal depth bounds require native lowering on this GPU");
    }
    if (regs.color_control.mode != ColorMode::Normal &&
        regs.color_control.mode != ColorMode::Disable && !metadata_draw) {
        throw std::runtime_error("Metal color metadata operation requires native lowering");
    }
    if (metadata_draw &&
        (geometry_draw || raster_tessellation_draw || raster_list_draw ||
         regs.depth_control.depth_enable || regs.depth_control.stencil_enable ||
         regs.depth_control.depth_bounds_enable ||
         (!(regs.reg_array[0xa001] & 1) && (regs.reg_array[0xa001] & 0x00ffff00)))) {
        throw std::runtime_error("Metal metadata draw stage/depth/query state requires lowering");
    }
    const bool restart = RasterRestartEnabled(regs, indexed, neo_mode);
    if ((geometry_draw || raster_tessellation_draw) && (indirect || restart)) {
        throw std::runtime_error(
            "Native geometry indirect/restart draw requires worklist lowering");
    }
    if (raster_list_draw &&
        (regs.polygon_control.PolyMode() != AmdGpu::PolygonMode::Fill ||
         (regs.vgt_strmout_config.raw & 15) || regs.vgt_strmout_config.rast_stream ||
         (regs.vgt_strmout_config.use_rast_stream_mask &&
          regs.vgt_strmout_config.rast_stream_mask != 1))) {
        throw std::runtime_error("Native rect/quad polygon/streamout state requires lowering");
    }
    const auto geometry_runtime =
        geometry_draw ? Runtime(Shader::HwStage::Geometry) : Shader::RuntimeInfo{};
    const u32 geometry_topology = [&] {
        if (!raster_tessellation_draw) {
            return geometry_draw ? u32(geometry_runtime.hw.gs.out_primitive[0]) : 0u;
        }
        switch (BuildTessellationDomainConfig(regs).tessellation.topology) {
        case AmdGpu::TessellationTopology::Point:
            return 0u;
        case AmdGpu::TessellationTopology::Line:
            return 1u;
        case AmdGpu::TessellationTopology::TriangleCw:
        case AmdGpu::TessellationTopology::TriangleCcw:
            return 2u;
        default:
            throw std::runtime_error("Native direct TES topology requires lowering");
        }
    }();
    if (geometry_draw && geometry_topology > 2) {
        throw std::runtime_error("Invalid native geometry raster topology");
    }
    auto primitive = (geometry_draw || raster_tessellation_draw)
                         ? (geometry_topology == 0   ? MTLPrimitiveTypePoint
                            : geometry_topology == 1 ? MTLPrimitiveTypeLine
                                                     : MTLPrimitiveTypeTriangle)
                         : Primitive(regs.primitive_type);
    const bool strip =
        primitive == MTLPrimitiveTypeLineStrip || primitive == MTLPrimitiveTypeTriangleStrip;
    const u32 restart_index =
        regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 0xffff : 0xffffffff;
    if ((primitive == MTLPrimitiveTypeLine || primitive == MTLPrimitiveTypeLineStrip) &&
        (regs.line_control.Width() != 1.f || regs.mode_control.line_stripple_enable)) {
        throw std::runtime_error("Metal wide/stippled lines require native primitive lowering");
    }
    const bool native_strip_restart =
        restart && strip && regs.primitive_restart_index == restart_index;
    const bool lower_topology = regs.primitive_type == AmdGpu::PrimitiveType::TriangleFan ||
                                regs.primitive_type == AmdGpu::PrimitiveType::LineLoop ||
                                (indexed && strip && !native_strip_restart) || (restart && !strip);
    if (indirect && lower_topology && !indexed) {
        throw std::runtime_error(
            "Metal indirect topology requires native GPU-count primitive assembly");
    }
    const u32 argument_size = indexed ? sizeof(MTLDrawIndexedPrimitivesIndirectArguments)
                                      : sizeof(MTLDrawPrimitivesIndirectArguments);
    u64 argument_bytes{};
    if (indirect) {
        const auto valid_sgpr = [](u16 offset) {
            return !offset || (offset >= 0x4c && offset < 0x4c + 16);
        };
        argument_bytes = u64{indirect->stride} * (indirect->max_count - 1) + argument_size;
        if (!valid_sgpr(indirect->vertex_sgpr_offset) ||
            !valid_sgpr(indirect->instance_sgpr_offset) || indirect->stride < argument_size ||
            indirect->stride % 4 || indirect->address % 4 ||
            argument_bytes > std::numeric_limits<u32>::max() ||
            argument_bytes > (u64{1} << 40) - indirect->address ||
            memory.ClampRangeSize(indirect->address, argument_bytes) != argument_bytes) {
            throw std::runtime_error("Invalid Metal guest indirect draw parameters");
        }
        if (!raster_list_draw)
            buffer_cache.PreparePagedBuffer(indirect->address, argument_bytes);
        if (indirect->count_address) {
            if (indirect->count_address >= (u64{1} << 40) - 3 || indirect->count_address % 4 ||
                memory.ClampRangeSize(indirect->count_address, 4) != 4) {
                throw std::runtime_error("Invalid Metal guest indirect draw count address");
            }
            buffer_cache.PreparePagedBuffer(indirect->count_address, 4);
        }
    }
    RenderState render;
    render.width = render.height = 16384;
    GraphicsPipelineState pipeline_state;
    const bool sample_rate = ConfigureMultisampling(regs, render, pipeline_state);
    if (![instance.GetDevice() supportsTextureSampleCount:render.num_samples]) {
        throw std::runtime_error("Metal guest sample count requires native sample splitting");
    }
    pipeline_state.topology =
        primitive == MTLPrimitiveTypePoint
            ? MTLPrimitiveTopologyClassPoint
            : (primitive == MTLPrimitiveTypeLine || primitive == MTLPrimitiveTypeLineStrip
                   ? MTLPrimitiveTopologyClassLine
                   : MTLPrimitiveTopologyClassTriangle);
    pipeline_state.rasterization =
        !regs.clipper_control.dx_rasterization_kill &&
        !(regs.polygon_control.cull_front && regs.polygon_control.cull_back);
    Shader::Backend::Bindings bindings{};
    auto* fragment = metadata_draw ? &coverage_fragment : &depth_fragment;
    if (!metadata_draw && regs.ps_program.Address<u32*>()) {
        fragment = &shader_cache.GetShader(
            Shader::HwStage::Fragment, Shader::SwStage::Fragment,
            shader_cache.GetParams(regs.ps_program.Address<VAddr>(), regs.ps_program.user_data),
            Runtime(Shader::HwStage::Fragment), bindings, false, sample_rate);
    }
    if (!metadata_draw && render.num_samples > 1) {
        ValidateMultisampleShader(regs, fragment->info, sample_rate);
    }
    struct Target {
        ImageInfo info;
        u32 slot;
        u32 base_slice;
        bool written;
        TextureCache::Image* cached{};
    };
    std::vector<Target> targets;
    std::vector<ImageInfo> attachment_infos;
    std::optional<ImageInfo> metadata_info;
    if (metadata_draw) {
        const auto& color = regs.color_buffers[0];
        if (!color || regs.color_target_mask.GetMask(0) != 15 ||
            regs.color_shader_mask.GetMask(0) != 15 ||
            color.info.endian != AmdGpu::ColorBuffer::EndianSwap::None || color.info.dcc_enable ||
            color.attrib.force_dst_alpha_1 ||
            color.attrib.num_samples_log2 != color.attrib.num_fragments_log2 ||
            (regs.blend_control[0].enable && !color.info.blend_bypass) ||
            (!regs.blend_control[0].disable_rop3 && regs.color_control.rop3 != LogicOp::Copy)) {
            throw std::runtime_error("Metal metadata draw requires an unblended MRT0 surface");
        }
        for (u32 index = 1; index < 8; ++index) {
            if (regs.color_target_mask.GetMask(index) || regs.color_shader_mask.GetMask(index)) {
                throw std::runtime_error("Metal metadata draw supports only MRT0");
            }
        }
        metadata_info = ImageInfo::FromColor(color, liverpool.last_cb_extent[0], neo_mode);
        Cmask::Validate(*metadata_info);
        if (regs.color_control.mode == ColorMode::FmaskDecompress &&
            metadata_info->layout.num_samples == 1) {
            throw std::runtime_error("Metal FMASK decompression requires a multisample surface");
        }
        render.num_layers = regs.vs_output_control.use_vtx_render_target_idx
                                ? metadata_info->layers - metadata_info->cmask_base_slice
                                : 1;
        metadata_coverage.Prepare(*metadata_info, render);
        pipeline_state.colors[0].format = MTLPixelFormatRGBA8Unorm;
        attachment_infos.push_back(*metadata_info);
        buffer_cache.PreparePagedBuffer(metadata_info->guest_address,
                                        metadata_info->layout.guest_size);
    }
    const auto target = [&](ImageInfo info, u32 slot, u32 base_slice, bool written) {
        if (base_slice >= info.layers)
            throw std::runtime_error("Invalid Metal guest attachment slice");
        for (const auto& previous : targets) {
            if (info.guest_address <
                    previous.info.guest_address + previous.info.layout.guest_size &&
                previous.info.guest_address < info.guest_address + info.layout.guest_size) {
                throw std::runtime_error(
                    "Metal overlapping attachments require native alias lowering");
            }
        }
        render.width = std::min(render.width, info.width);
        render.height = std::min(render.height, info.height);
        if (info.layout.num_samples != render.num_samples)
            throw std::runtime_error("Metal mixed-sample attachments require native EQAA lowering");
        if (regs.vs_output_control.use_vtx_render_target_idx) {
            const u32 layers = info.layers - base_slice;
            render.num_layers = targets.empty() ? layers : std::min(render.num_layers, layers);
        }
        buffer_cache.PreparePagedBuffer(info.guest_address, info.layout.guest_size);
        attachment_infos.push_back(info);
        targets.push_back({std::move(info), slot, base_slice, written});
    };
    for (u32 index = 0; index < 8; ++index) {
        const auto& color = regs.color_buffers[index];
        if (metadata_draw || regs.color_control.mode == ColorMode::Disable || !color ||
            !(fragment->info.mrt_mask & (1u << index)) || !regs.color_target_mask.GetMask(index))
            continue;
        const auto& blend = regs.blend_control[index];
        if (!blend.disable_rop3 && regs.color_control.rop3 != LogicOp::Copy &&
            regs.color_control.rop3 != LogicOp::Noop) {
            throw std::runtime_error(
                "Metal guest logical blending requires native fragment lowering");
        }
        if (color.info.endian != AmdGpu::ColorBuffer::EndianSwap::None || color.info.dcc_enable ||
            color.attrib.force_dst_alpha_1) {
            throw std::runtime_error("Metal guest color storage mode requires native lowering");
        }
        auto info = ImageInfo::FromColor(color, liverpool.last_cb_extent[index], neo_mode);
        if (render.num_samples > 1 &&
            color.attrib.num_samples_log2 != color.attrib.num_fragments_log2) {
            throw std::runtime_error("Metal guest color fragments require native EQAA lowering");
        }
        auto& state = pipeline_state.colors[index];
        state.format = info.format;
        state.write_mask = static_cast<MTLColorWriteMask>(
            color.Swizzle().ApplyMask(regs.color_target_mask.GetMask(index)));
        if (!blend.disable_rop3 && regs.color_control.rop3 == LogicOp::Noop)
            state.write_mask = MTLColorWriteMaskNone;
        state.blending = blend.enable && !color.info.blend_bypass;
        if (state.blending) {
            state.source_rgb = BlendFactor(blend.color_src_factor);
            state.destination_rgb = BlendFactor(blend.color_dst_factor);
            state.rgb_operation =
                BlendOperation(blend.color_func, blend.color_src_factor, blend.color_dst_factor);
            const auto source =
                blend.separate_alpha_blend ? blend.alpha_src_factor : blend.color_src_factor;
            const auto destination =
                blend.separate_alpha_blend ? blend.alpha_dst_factor : blend.color_dst_factor;
            state.source_alpha = BlendFactor(source);
            state.destination_alpha = BlendFactor(destination);
            state.alpha_operation =
                BlendOperation(blend.separate_alpha_blend ? blend.alpha_func : blend.color_func,
                               source, destination);
        }
        target(std::move(info), index, color.BaseSlice(),
               state.write_mask != MTLColorWriteMaskNone);
        render.num_color_attachments = index + 1;
    }
    for (u32 plane = 0; plane < 2; ++plane) {
        const auto& depth = regs.depth_buffer;
        const bool enabled =
            plane ? regs.depth_control.stencil_enable && depth.StencilValid()
                  : (regs.depth_control.depth_enable || regs.depth_control.depth_bounds_enable) &&
                        depth.DepthValid();
        if (!enabled)
            continue;
        const bool written = plane ? !regs.depth_view.stencil_read_only &&
                                         (regs.stencil_ref_front.stencil_write_mask ||
                                          (regs.depth_control.backface_enable &&
                                           regs.stencil_ref_back.stencil_write_mask))
                                   : regs.depth_control.depth_enable &&
                                         regs.depth_control.depth_write_enable &&
                                         !regs.depth_view.z_read_only;
        const VAddr read_address = plane ? depth.StencilAddress() : depth.DepthAddress();
        const VAddr write_address = plane ? depth.StencilWriteAddress() : depth.DepthWriteAddress();
        if (written && read_address != write_address) {
            throw std::runtime_error(
                "Metal split depth/stencil storage requires native plane copying");
        }
        target(ImageInfo::FromDepth(
                   depth, regs.depth_view.NumSlices(), liverpool.last_db_extent, written, plane,
                   {regs.depth_htile_data_base.GetAddress(), regs.reg_array[0x028abc / 4],
                    regs.reg_array[0x028044 / 4], regs.depth_view.slice_start, regs.depth_clear,
                    regs.stencil_clear, neo_mode}),
               8 + plane, regs.depth_view.slice_start, written);
    }
    bool flip_y{};
    auto graphics = State(render.width, render.height, flip_y);
    const auto acquire_targets = [&] {
        for (auto& attachment : targets) {
            attachment.cached = &texture_cache.ObtainImage(
                attachment.info, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
            ImageViewInfo view;
            view.format = attachment.info.format;
            view.type = attachment.info.TextureType();
            view.base_slice = attachment.base_slice;
            view.slices = attachment.info.layers - attachment.base_slice;
            const auto texture = texture_cache.ObtainView(*attachment.cached, view);
            RenderAttachment* destination =
                attachment.slot < 8
                    ? &render.color_attachments[attachment.slot]
                    : (attachment.slot == 8
                           ? static_cast<RenderAttachment*>(&render.depth_attachment)
                           : &render.stencil_attachment);
            destination->texture = texture;
        }
    };
    const auto mark_targets = [&] {
        for (auto& attachment : targets) {
            if (attachment.written)
                texture_cache.MarkWritten(*attachment.cached);
        }
    };
    const auto mesh_viewports = [&](const std::array<u32, 2>& extent) {
        std::vector<std::array<float, 4>> result;
        if (extent[0] && extent[1]) {
            static_assert(AmdGpu::NUM_VIEWPORTS == 16);
            result.reserve(graphics.viewport_count);
            for (u32 index = 0; index < graphics.viewport_count; ++index) {
                const auto& viewport = regs.viewports[index];
                result.push_back({regs.viewport_control.xoffset_enable ? viewport.xoffset : 0.f,
                                  regs.viewport_control.yoffset_enable ? viewport.yoffset : 0.f,
                                  regs.viewport_control.xscale_enable ? viewport.xscale : 1.f,
                                  regs.viewport_control.yscale_enable ? viewport.yscale : 1.f});
            }
        }
        return result;
    };
    const auto draw_geometry =
        [&](GeometryAssembly::Result&& assembled, const GeometryAssembly::MeshShader& mesh,
            const BufferRange& errors, std::unique_ptr<GuestResources::Stage> fragment_stage) {
            const auto viewports = mesh_viewports(mesh.viewport_extent);
            const auto consume = [&](GeometryAssembly::Result&& result) {
                indirect_draws.TrackErrorUse(errors);
                if (!fragment_stage) {
                    fragment_stage = resources.Prepare(*fragment, neo_mode);
                    const std::array stages{fragment_stage.get()};
                    resources.ValidateAttachments(stages, attachment_infos);
                }
                const std::array stages{fragment_stage.get()};
                resources.Acquire(stages);
                acquire_targets();
                resources.MarkWrites(stages);
                mark_targets();
                auto snapshot = resources.Snapshot(*fragment_stage);
                geometry_assembly.Draw(result, mesh, render, graphics, pipelines, fragment,
                                       pipeline_state, snapshot.get(), viewports);
                if (fragment_stage->HasWrites()) {
                    scheduler.EndEncoding();
                }
                fragment_stage.reset();
            };
            if (mesh.viewport_extent[0]) {
                geometry_assembly.PackViewportBatches(assembled, mesh, errors,
                                                      graphics.viewport_count, consume);
            } else {
                consume(std::move(assembled));
            }
        };
    if (raster_list_draw) {
        const bool polygon = regs.primitive_type == AmdGpu::PrimitiveType::Polygon;
        const u32 input_vertices =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList || polygon ? 3 : 4;
        if (!indirect && regs.num_indices < input_vertices) {
            return;
        }
        std::optional<RasterPrimitiveCounterSettings> primitive_ids;
        if (fragment->info.loads.Get(Shader::IR::Attribute::PrimitiveId))
            primitive_ids = RasterGuestPrimitiveIds(regs, indirect ? indirect->max_count : 0);
        if (indirect) {
            const u32 index_width =
                !indexed                                                          ? 0u
                : regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u
                : regs.index_buffer_type.index_type == AmdGpu::IndexType::Index32 ? 4u
                                                                                  : 0u;
            const auto swap =
                indexed ? regs.index_buffer_type.swap_mode : AmdGpu::IndexSwapMode::None;
            const auto restart_index =
                restart ? std::optional{regs.primitive_restart_index} : std::nullopt;
            if (restart_index)
                ValidateRasterRestartIndex(index_width, *restart_index, neo_mode);
            const auto indices =
                indexed ? BuildRasterIndexRange(regs.index_base_address.Address(), index_offset,
                                                regs.max_index_size, index_width, swap)
                        : RasterIndexRange{};
            const VAddr index_address = indices.LogicalAddress();
            const VAddr index_source_address = indices.address;
            const u64 index_bytes = indices.SizeBytes();
            if (index_bytes &&
                memory.ClampRangeSize(index_source_address, index_bytes) != index_bytes) {
                throw std::runtime_error(
                    "Native indirect raster indices require mapped guest storage");
            }
            for (const auto& attachment : attachment_infos) {
                const auto overlaps = [&](VAddr address, u64 size) {
                    return buffer_cache.OverlapsGuestMemory(address, size, attachment.guest_address,
                                                            attachment.layout.guest_size);
                };
                if (overlaps(indirect->address, argument_bytes) ||
                    (indirect->count_address && overlaps(indirect->count_address, sizeof(u32))) ||
                    (indexed && overlaps(index_source_address, index_bytes))) {
                    throw std::runtime_error(
                        "Native indirect raster attachment feedback requires lowering");
                }
            }
            if (index_bytes)
                buffer_cache.PreparePagedBuffer(index_source_address, index_bytes);
            const bool last_vertex =
                regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last;
            const RasterIndirectRegisters registers{indirect->vertex_sgpr_offset,
                                                    indirect->instance_sgpr_offset};
            auto plan = geometry_assembly.PrepareIndirectRasterPlan(
                shader_cache, regs, registers, index_width, fragment, last_vertex, flip_y, neo_mode,
                primitive_ids.has_value());
            const u32 primitive_vertices = polygon ? 1u
                                           : regs.primitive_type == AmdGpu::PrimitiveType::QuadStrip
                                               ? 2u
                                               : input_vertices;
            const u64 capture_bytes = RasterCaptureSubmissionSizeBytes(
                plan.capacity, plan.mesh.vertex_words, plan.mesh.mesh->vertices, primitive_vertices,
                restart, plan.mesh.viewport_extent[0] != 0, plan.mesh.mesh->primitive_ids);
            const auto submission = BuildRasterIndirectSubmissionLayout(
                indirect->max_count, capture_bytes, instance.GetDevice().maxBufferLength);
            const std::array viewport{
                regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f,
                regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f,
                regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f,
                regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f};
            std::optional<IndirectDraws::RasterCount> count_snapshot;
            if (indirect->count_address) {
                buffer_cache.ObtainPagedBuffer(indirect->count_address, sizeof(u32), false);
                const auto ranges =
                    buffer_cache.GetPagedBufferRanges(indirect->count_address, sizeof(u32));
                if (ranges.size() != 1)
                    throw std::runtime_error(
                        "Native indirect raster count must fit one guest page");
                count_snapshot = indirect_draws.SnapshotRasterCount(ranges.front());
            }
            visibility.Prepare(render, graphics, regs.reg_array[0xa001]);
            static_cast<void>(
                pipelines.GetMeshPipeline(plan.mesh.shader, fragment, pipeline_state));
            for (u64 first = 0; first < indirect->max_count;) {
                const u32 count =
                    std::min(submission.records_per_window, indirect->max_count - u32(first));
                const auto layout = BuildRasterIndirectWindow(
                    indirect->max_count, u32(first), count, indirect->stride, index_width,
                    indexed ? regs.max_index_size : 0, plan.capacity);
                const VAddr address = indirect->address + first * indirect->stride;
                buffer_cache.PreparePagedBuffer(address, layout.argument_bytes);
                buffer_cache.ObtainPagedBuffer(address, layout.argument_bytes, false);
                const auto pages = buffer_cache.CreatePageSnapshot();
                const auto window = indirect_draws.PrepareRasterWindowPaged(
                    *pages, address, indirect->count_address, indirect->max_count, u32(first),
                    count, indirect->stride, index_width, indexed ? regs.max_index_size : 0,
                    plan.capacity, count_snapshot ? &*count_snapshot : nullptr);
                auto controller = indirect_draws.CreateRasterController(
                    window, primitive_vertices, regs.vgt_instance_step_rate_0,
                    regs.vgt_instance_step_rate_1, regs.index_offset, swap);
                std::optional<GeometryAssembly::IndirectRasterState> previous;
                const bool complete = indirect_draws.RunRasterCaptures(
                    controller, submission.captures_per_submission,
                    [&](const IndirectDraws::RasterCapture& capture) {
                        if (index_bytes)
                            buffer_cache.ObtainPagedBuffer(index_source_address, index_bytes,
                                                           false);
                        const auto inputs_pages = buffer_cache.CreatePageSnapshot();
                        const auto inputs = indirect_draws.PrepareRasterCaptureInputs(
                            capture, indexed ? inputs_pages.get() : nullptr, index_address,
                            viewport, {}, restart_index, registers);
                        auto prepared = geometry_assembly.PrepareIndirectRasterVertices(
                            shader_cache, resources, regs, inputs, neo_mode, {},
                            primitive_ids.has_value());
                        auto fragment_stage = resources.Prepare(*fragment, neo_mode);
                        const std::array stages{fragment_stage.get()};
                        resources.ValidateAttachments(stages, attachment_infos);
                        const std::array kernels{&prepared.kernel};
                        geometry_assembly.Execute(kernels, attachment_infos);
                        auto packed = geometry_assembly.AssembleIndirectRasterQuads(
                            prepared, plan.mesh, previous ? &*previous : nullptr,
                            graphics.viewport_count, primitive_ids ? &*primitive_ids : nullptr);
                        const auto errors = indirect_draws.PrepareErrorBuffer();
                        draw_geometry(std::move(packed.result), plan.mesh, errors,
                                      std::move(fragment_stage));
                        previous = std::move(packed.state);
                    });
                if (complete)
                    break;
                first += count;
            }
            return;
        }
        RasterIndexRange range{};
        auto swap = AmdGpu::IndexSwapMode::None;
        if (indexed) {
            const u32 width = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u
                              : regs.index_buffer_type.index_type == AmdGpu::IndexType::Index32
                                  ? 4u
                                  : 0u;
            const u32 count = index_offset < regs.max_index_size
                                  ? std::min(regs.num_indices, regs.max_index_size - index_offset)
                                  : 0;
            range = BuildRasterIndexRange(regs.index_base_address.Address(), index_offset, count,
                                          width);
            swap = regs.index_buffer_type.swap_mode;
        }
        const u32 capacity =
            geometry_assembly.MaxRasterBatchVertices(shader_cache, regs, indexed, neo_mode);
        const RasterDrawBatchPlan batches{regs.num_indices,
                                          regs.num_instances.NumInstances(),
                                          capacity,
                                          regs.vgt_instance_step_rate_0,
                                          regs.vgt_instance_step_rate_1,
                                          indexed ? range.element_size : 0,
                                          range.count,
                                          regs.index_offset,
                                          swap,
                                          polygon ? 1u
                                          : regs.primitive_type == AmdGpu::PrimitiveType::QuadStrip
                                              ? 2u
                                              : input_vertices};
        const bool last_vertex =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last;
        visibility.Prepare(render, graphics, regs.reg_array[0xa001]);
        std::optional<GeometryAssembly::RasterDrawContinuation> previous;
        std::optional<GeometryAssembly::RasterDrawContinuation> origin;
        std::optional<GeometryAssembly::RasterRestartState> restart_state;
        std::optional<RasterPrimitiveCounterConfig> primitive_counter;
        if (primitive_ids && restart)
            primitive_counter = RasterPrimitiveCounterConfig{
                regs.num_indices, primitive_ids->reset_value, primitive_ids->reset_on_instance};
        for (u64 batch = 0; batch < batches.Count(); ++batch) {
            const auto parameters = batches.Batch(batch);
            const auto primitive = BuildRasterPrimitiveParameters(regs.primitive_type, parameters);
            BufferRange indices{};
            const auto source =
                parameters.index_width
                    ? BuildRasterIndexRange(range.address, parameters.first_vertex,
                                            parameters.index_count, parameters.index_width, swap)
                    : RasterIndexRange{};
            const u64 size = source.SizeBytes();
            if (size) {
                const VAddr address = source.address;
                if (!address) {
                    throw std::runtime_error("Native rect/quad indices require guest storage");
                }
                for (u64 offset = 0; offset < size;) {
                    Libraries::Kernel::OrbisVirtualQueryInfo mapping{};
                    const VAddr current = address + offset;
                    if (memory.VirtualQuery(current, 0, &mapping) != 0 || !mapping.is_committed ||
                        mapping.start > current || mapping.end <= current) {
                        throw std::runtime_error("Native rect/quad indices are not fully mapped");
                    }
                    offset += std::min<u64>(size - offset, mapping.end - current);
                }
                for (const auto& attachment : attachment_infos) {
                    if (buffer_cache.OverlapsGuestMemory(address, size, attachment.guest_address,
                                                         attachment.layout.guest_size)) {
                        throw std::runtime_error(
                            "Native rect/quad index/attachment feedback requires lowering");
                    }
                }
                buffer_cache.PreparePagedBuffer(address, size);
                buffer_cache.ObtainPagedBuffer(address, size, false);
                indices = GatherIndices(address, size);
            }
            auto prepared = geometry_assembly.PrepareGuestRasterDraw(
                shader_cache, resources, regs, parameters, indices, neo_mode, fragment, last_vertex,
                flip_y, source.source_offset);
            if (restart && !prepared.restart) {
                throw std::runtime_error("Native raster restart requires its captured metadata");
            }
            std::unique_ptr<GuestResources::Stage> fragment_stage;
            if (restart || primitive.primitives_per_instance) {
                fragment_stage = resources.Prepare(*fragment, neo_mode);
                const std::array fragment_stages{fragment_stage.get()};
                resources.ValidateAttachments(fragment_stages, attachment_infos);
                static_cast<void>(
                    pipelines.GetMeshPipeline(prepared.mesh.shader, fragment, pipeline_state));
            }
            const std::array kernels{&prepared.vertices.kernel};
            geometry_assembly.Execute(kernels, attachment_infos);
            if (!restart && polygon && !parameters.first_vertex && parameters.instances == 1) {
                origin = GeometryAssembly::RasterDrawContinuation{prepared.vertices.vertices,
                                                                  parameters};
            }
            if (!restart && !primitive.primitives_per_instance) {
                continue;
            }
            const auto errors = indirect_draws.PrepareErrorBuffer();
            std::optional<RasterPrimitiveIdParameters> batch_ids;
            if (primitive_ids && !restart)
                batch_ids = BuildRasterPrimitiveIds(regs.primitive_type, parameters,
                                                    regs.num_indices, primitive_ids->reset_value,
                                                    primitive_ids->reset_on_instance);
            auto next_state = geometry_assembly.AssembleRasterQuads(
                prepared.vertices.vertices, parameters, prepared.mesh, errors,
                [&](GeometryAssembly::Result&& assembled) {
                    draw_geometry(std::move(assembled), prepared.mesh, errors,
                                  std::move(fragment_stage));
                },
                graphics.viewport_count, regs.primitive_type,
                !restart && primitive.carry_vertices && previous ? &*previous : nullptr,
                !restart && primitive.carry_vertices && origin && polygon ? &*origin : nullptr,
                restart ? &*prepared.restart : nullptr,
                restart && parameters.first_vertex && restart_state ? &*restart_state : nullptr,
                batch_ids ? &*batch_ids : nullptr,
                primitive_counter ? &*primitive_counter : nullptr,
                primitive_counter && restart_state ? &*restart_state : nullptr);
            if (restart) {
                if (!next_state) {
                    throw std::runtime_error(
                        "Native raster restart did not produce continuation state");
                }
                restart_state = std::move(next_state);
            } else if (regs.primitive_type == AmdGpu::PrimitiveType::QuadStrip || polygon) {
                previous = GeometryAssembly::RasterDrawContinuation{
                    std::move(prepared.vertices.vertices), parameters};
            }
        }
        return;
    }
    if (raster_tessellation_draw) {
        const auto interface = tessellation.DescribeRaster(regs);
        if (!interface.vertex_words) {
            return;
        }
        if (!geometry_topology &&
            std::ranges::none_of(interface.attributes, [](const auto& attribute) {
                return attribute.attribute == Shader::IR::Attribute::PointSize;
            })) {
            throw std::runtime_error("Native direct TES points require a point-size export");
        }
        const bool last_vertex =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last;
        const bool dynamic_viewport =
            regs.IsClipDisabled() &&
            std::ranges::any_of(interface.attributes, [](const auto& attribute) {
                return attribute.attribute == Shader::IR::Attribute::ViewportIndex;
            });
        std::array<u32, 2> viewport_extent{};
        if (dynamic_viewport) {
            viewport_extent = {std::min(shader_cache.GetProfile().max_viewport_width / 2, 8192u),
                               std::min(shader_cache.GetProfile().max_viewport_height / 2, 8192u)};
        }
        const auto viewports = mesh_viewports(viewport_extent);
        const auto mesh = geometry_assembly.CompileMesh(
            0, geometry_topology, interface.vertex_words, interface.attributes, fragment,
            last_vertex, flip_y, viewport_extent);
        static_cast<void>(pipelines.GetMeshPipeline(mesh.shader, fragment, pipeline_state));
        const u64 group_bytes = u64{64} * interface.vertex_words * sizeof(u32);
        const u64 group_capacity = std::min({u64{instance.GetDevice().maxBufferLength},
                                             u64{std::numeric_limits<u32>::max()}, u64{32} << 20}) /
                                   group_bytes;
        if (!group_capacity) {
            throw std::runtime_error("Native direct TES payload exceeds workgroup capacity");
        }
        visibility.Prepare(render, graphics, regs.reg_array[0xa001]);
        tessellation.RunGuestLocalHullBatches(
            regs, indexed, index_offset, neo_mode,
            [&](TessellationDispatch::Result&& hull) {
                tessellation.RunRasterBatches(
                    regs, hull, neo_mode,
                    [&](TessellationExports::Batch&& batch) {
                        const auto errors = indirect_draws.PrepareErrorBuffer();
                        geometry_assembly.AssembleTessellationRaster(
                            batch, mesh, errors,
                            [&](GeometryAssembly::Result&& assembled) {
                                indirect_draws.TrackErrorUse(errors);
                                auto fragment_stage = resources.Prepare(*fragment, neo_mode);
                                const std::array fragment_stages{fragment_stage.get()};
                                resources.ValidateAttachments(fragment_stages, attachment_infos);
                                resources.Acquire(fragment_stages);
                                acquire_targets();
                                resources.MarkWrites(fragment_stages);
                                mark_targets();
                                auto snapshot = resources.Snapshot(*fragment_stage);
                                geometry_assembly.Draw(assembled, mesh, render, graphics, pipelines,
                                                       fragment, pipeline_state, snapshot.get(),
                                                       viewports);
                                if (fragment_stage->HasWrites()) {
                                    scheduler.EndEncoding();
                                }
                            },
                            graphics.viewport_count);
                    },
                    attachment_infos);
            },
            attachment_infos, u32(group_capacity));
        return;
    }
    if (tessellation_draw) {
        const bool last_vertex =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last;
        const u32 primitive_capacity = geometry_assembly.MaxTessellationPrimitives(
            shader_cache, regs, fragment, last_vertex, flip_y);
        visibility.Prepare(render, graphics, regs.reg_array[0xa001]);
        tessellation.RunGuestLocalHullBatches(
            regs, indexed, index_offset, neo_mode,
            [&](TessellationDispatch::Result&& hull) {
                tessellation.RunEvaluationBatches(
                    regs, hull, neo_mode,
                    [&](TessellationExports::Batch&& exports) {
                        tessellation_geometry.RunBatches(
                            regs, exports,
                            [&](TessellationGeometry::Inputs&& inputs) {
                                auto fragment_stage = resources.Prepare(*fragment, neo_mode);
                                const std::array fragment_stages{fragment_stage.get()};
                                resources.ValidateAttachments(fragment_stages, attachment_infos);
                                const auto errors = indirect_draws.PrepareErrorBuffer();
                                auto prepared = geometry_assembly.PrepareTessellationDraw(
                                    shader_cache, resources, regs, inputs, errors, neo_mode,
                                    fragment, last_vertex, flip_y, fragment_stages,
                                    attachment_infos);
                                static_cast<void>(pipelines.GetMeshPipeline(
                                    prepared.mesh.shader, fragment, pipeline_state));
                                const std::array kernels{&prepared.geometry.kernel,
                                                         &prepared.copy.kernel};
                                geometry_assembly.Execute(kernels, attachment_infos);
                                auto assembled =
                                    geometry_assembly.Assemble(prepared.copy.emissions, errors);
                                draw_geometry(std::move(assembled), prepared.mesh, errors,
                                              std::move(fragment_stage));
                            },
                            primitive_capacity);
                    },
                    attachment_infos);
            },
            attachment_infos);
        return;
    }
    if (geometry_draw) {
        const u32 index_width = !indexed ? 0
                                : regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16
                                    ? 2
                                    : 4;
        const auto parameters = GeometryDrawLayout(
            regs.primitive_type, regs.num_indices, regs.num_instances.NumInstances(),
            geometry_runtime.hw.gs.num_invocations, regs.vgt_instance_step_rate_0,
            regs.vgt_instance_step_rate_1, index_width, regs.index_offset);
        if (!parameters.geometry_items)
            return;
        BufferRange indices{};
        if (indexed) {
            if (regs.index_buffer_type.swap_mode != AmdGpu::IndexSwapMode::None ||
                (regs.index_buffer_type.index_type != AmdGpu::IndexType::Index16 &&
                 regs.index_buffer_type.index_type != AmdGpu::IndexType::Index32) ||
                regs.num_indices > regs.max_index_size) {
                throw std::runtime_error("Native geometry index format requires lowering");
            }
            const VAddr address =
                regs.index_base_address.Address() + u64{index_offset} * index_width;
            const u64 size = u64{regs.num_indices} * index_width;
            if (!address || address >= (u64{1} << 40) || address % index_width ||
                size > (u64{1} << 40) - address || size > std::numeric_limits<u32>::max()) {
                throw std::runtime_error("Invalid native geometry index range");
            }
            for (u64 offset = 0; offset < size;) {
                Libraries::Kernel::OrbisVirtualQueryInfo mapping{};
                const VAddr current = address + offset;
                if (memory.VirtualQuery(current, 0, &mapping) != 0 || !mapping.is_committed ||
                    mapping.start > current || mapping.end <= current) {
                    throw std::runtime_error("Native geometry indices are not fully mapped");
                }
                offset += std::min<u64>(size - offset, mapping.end - current);
            }
            for (const auto& attachment : attachment_infos) {
                if (address < attachment.guest_address + attachment.layout.guest_size &&
                    attachment.guest_address < address + size) {
                    throw std::runtime_error(
                        "Native geometry index/attachment feedback requires lowering");
                }
            }
            buffer_cache.PreparePagedBuffer(address, size);
            buffer_cache.ObtainPagedBuffer(address, size, false);
            indices = GatherIndices(address, size);
        }
        const bool last_vertex =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last;
        const auto errors = indirect_draws.PrepareErrorBuffer();
        auto prepared =
            geometry_assembly.PrepareGuestDraw(shader_cache, resources, regs, parameters, indices,
                                               errors, neo_mode, fragment, last_vertex, flip_y);
        auto fragment_stage = resources.Prepare(*fragment, neo_mode);
        const std::array fragment_stages{fragment_stage.get()};
        resources.ValidateAttachments(fragment_stages, attachment_infos);
        static_cast<void>(
            pipelines.GetMeshPipeline(prepared.mesh.shader, fragment, pipeline_state));
        const std::array kernels{&prepared.exports, &prepared.geometry.kernel,
                                 &prepared.copy.kernel};
        geometry_assembly.Execute(kernels, attachment_infos);
        auto assembled = geometry_assembly.Assemble(prepared.copy.emissions, errors);
        visibility.Prepare(render, graphics, regs.reg_array[0xa001]);
        draw_geometry(std::move(assembled), prepared.mesh, errors, std::move(fragment_stage));
        return;
    }
    if (!regs.vs_program.Address<u32*>())
        throw std::runtime_error("Metal draw requires a vertex shader");
    auto vertex_runtime = Runtime(Shader::HwStage::Vertex);
    if (indirect) {
        vertex_runtime.sw.vs.vertex_sgpr_offset = indirect->vertex_sgpr_offset;
        vertex_runtime.sw.vs.instance_sgpr_offset = indirect->instance_sgpr_offset;
    }
    const VertexFetchLimits vertex_fetch_limits{
        regs.num_indices,
        regs.num_instances.NumInstances(),
        regs.index_offset,
        regs.vgt_instance_step_rate_0,
        regs.vgt_instance_step_rate_1,
        indexed,
        indirect != nullptr,
    };
    auto& vertex = shader_cache.GetShader(
        Shader::HwStage::Vertex, Shader::SwStage::Vertex,
        shader_cache.GetParams(regs.vs_program.Address<VAddr>(), regs.vs_program.user_data),
        vertex_runtime, bindings, flip_y, false, nullptr, nullptr, nullptr, &vertex_fetch_limits);
    const bool flat_inputs =
        std::ranges::any_of(fragment->specialization.runtime_info.hw.fs.inputs,
                            [](const auto& input) { return input.is_flat; }) ||
        std::ranges::any_of(fragment->info.fs_interpolation, [](const auto& input) {
            return input.primary == Shader::Qualifier::Flat;
        });
    const bool last_vertex =
        regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last && flat_inputs;
    if (last_vertex &&
        (primitive == MTLPrimitiveTypeLine || primitive == MTLPrimitiveTypeLineStrip)) {
        throw std::runtime_error(
            "Metal last provoking line attributes require native shader lowering");
    }
    const bool assemble = lower_topology || (last_vertex && primitive != MTLPrimitiveTypePoint);
    const bool segmented = restart && assemble;
    if (indirect && assemble && !indexed) {
        throw std::runtime_error(
            "Metal indirect provoking vertex requires native GPU-count primitive assembly");
    }
    const auto& fetch = vertex.specialization.fetch_shader_data;
    if (primitive == MTLPrimitiveTypePoint &&
        !vertex.info.stores.GetAny(Shader::IR::Attribute::PointSize)) {
        throw std::runtime_error("Metal point size requires a native vertex output");
    }
    const auto offset_value = [&](s8 index) {
        if (index < 0)
            return u32{0};
        if (u32(index) >= vertex.info.user_data.size()) {
            throw std::runtime_error("Invalid Metal guest vertex offset register");
        }
        return vertex.info.user_data[index];
    };
    const u32 base_vertex =
        regs.index_offset ? regs.index_offset : offset_value(fetch.vertex_offset_sgpr);
    const u32 first_instance = offset_value(fetch.instance_offset_sgpr);
    if (assemble && !indexed && base_vertex > u32(std::numeric_limits<s32>::max())) {
        throw std::runtime_error("Metal generated indices require a signed native base vertex");
    }
    std::vector<std::pair<VAddr, u32>> vertex_ranges;
    for (u32 location = 0; location < fetch.attributes.size(); ++location) {
        const auto& attribute = fetch.attributes[location];
        const auto sharp = attribute.GetSharp(vertex.info);
        const u64 requested_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
        const u64 size = memory.ClampRangeSize(sharp.base_address, requested_size);
        if (!sharp.base_address || !size || size > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Metal vertex buffer requires valid guest storage");
        }
        using Step = Shader::Gcn::VertexAttribute::InstanceIdType;
        const auto step = attribute.GetStepRate();
        const u32 divisor = step == Step::OverStepRate0
                                ? regs.vgt_instance_step_rate_0
                                : (step == Step::OverStepRate1 ? regs.vgt_instance_step_rate_1 : 1);
        const auto format = VertexFormat(sharp);
        if (format == MTLVertexFormatInvalid) {
            throw std::runtime_error("Metal vertex data format requires native fetch lowering");
        }
        pipeline_state.attributes.push_back(
            {location, format, GraphicsPipelineState::FirstVertexBufferIndex + location, 0});
        const bool constant = !sharp.GetStride() || (step != Step::None && !divisor);
        const u32 stride =
            sharp.GetStride()
                ? sharp.GetStride()
                : Common::AlignUp(AmdGpu::NumBitsPerBlock(sharp.GetDataFmt()) / 8, 4u);
        pipeline_state.buffers.push_back(
            {stride,
             constant ? MTLVertexStepFunctionConstant
                      : (step == Step::None ? MTLVertexStepFunctionPerVertex
                                            : MTLVertexStepFunctionPerInstance),
             constant ? 0 : divisor});
        vertex_ranges.emplace_back(sharp.base_address, u32(size));
        buffer_cache.PreparePagedBuffer(sharp.base_address, size);
    }
    VAddr index_address{};
    u64 index_size{};
    MTLIndexType index_type = MTLIndexTypeUInt16;
    if (indexed) {
        if (regs.index_buffer_type.swap_mode != AmdGpu::IndexSwapMode::None ||
            (regs.index_buffer_type.index_type != AmdGpu::IndexType::Index16 &&
             regs.index_buffer_type.index_type != AmdGpu::IndexType::Index32)) {
            throw std::runtime_error("Metal index format requires native index lowering");
        }
        index_type = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index32
                         ? MTLIndexTypeUInt32
                         : MTLIndexTypeUInt16;
        const u32 element_size = index_type == MTLIndexTypeUInt32 ? 4 : 2;
        index_address = regs.index_base_address.Address() + u64(index_offset) * element_size;
        const u32 index_count = indirect ? regs.max_index_size : regs.num_indices;
        index_size = u64(index_count) * element_size;
        if (index_address >= (u64{1} << 40) || index_address % element_size ||
            index_size > (u64{1} << 40) - index_address ||
            index_size > std::numeric_limits<u32>::max() ||
            (!indirect && regs.num_indices > regs.max_index_size) || !index_size ||
            memory.ClampRangeSize(index_address, index_size) != index_size) {
            throw std::runtime_error("Metal index buffer requires robust guest index lowering");
        }
        buffer_cache.PreparePagedBuffer(index_address, index_size);
    }
    for (const auto& attachment : attachment_infos) {
        const auto overlaps = [&](VAddr address, u64 size) {
            return address < attachment.guest_address + attachment.layout.guest_size &&
                   attachment.guest_address < address + size;
        };
        if ((indirect && (overlaps(indirect->address, argument_bytes) ||
                          (indirect->count_address && overlaps(indirect->count_address, 4)))) ||
            (indexed && overlaps(index_address, index_size)) ||
            std::ranges::any_of(vertex_ranges, [&](const auto& range) {
                return overlaps(range.first, range.second);
            })) {
            throw std::runtime_error(
                "Metal attachment/vertex feedback requires native alias lowering");
        }
    }
    auto vertex_stage = resources.Prepare(vertex, neo_mode);
    auto fragment_stage = resources.Prepare(*fragment, neo_mode);
    if (metadata_draw && vertex_stage->HasWrites()) {
        throw std::runtime_error("Metal metadata vertex side effects require native lowering");
    }
    std::array stages{vertex_stage.get(), fragment_stage.get()};
    resources.ValidateAttachments(stages, attachment_infos);
    const bool use_batch = segmented || indirect;
    pipeline_state.indirect_commands = use_batch;
    const auto pipeline = pipelines.GetPipeline(vertex, fragment, pipeline_state);
    if (indirect) {
        buffer_cache.ObtainPagedBuffer(indirect->address, argument_bytes, false);
        if (indirect->count_address)
            buffer_cache.ObtainPagedBuffer(indirect->count_address, 4, false);
    }
    if (indexed)
        buffer_cache.ObtainPagedBuffer(index_address, index_size, false);
    for (const auto& [address, size] : vertex_ranges) {
        buffer_cache.ObtainPagedBuffer(address, size, false);
    }
    resources.Acquire(stages);
    BufferRange indices{};
    acquire_targets();
    resources.RefreshPageTables(stages);
    const auto vertex_buffers = GatherVertexBuffers(vertex_ranges);
    if (indexed && !assemble)
        indices = GatherIndices(index_address, index_size);
    BufferRange arguments{};
    u32 draw_count = regs.num_indices;
    if (assemble && !indirect && !segmented) {
        PrimitiveAssembly::Result assembled;
        if (indexed) {
            const auto pages = buffer_cache.CreatePageSnapshot();
            const auto errors = indirect_draws.PrepareErrorBuffer();
            assembled = primitive_assembly.AssemblePaged(regs.primitive_type, draw_count, *pages,
                                                         index_address, index_size, index_type,
                                                         last_vertex, errors);
            arguments = primitive_assembly.FinalizeDrawArguments(
                assembled.count, regs.num_instances.NumInstances(), std::bit_cast<s32>(base_vertex),
                first_instance, errors);
        } else {
            assembled = primitive_assembly.Assemble(regs.primitive_type, draw_count, indices,
                                                    index_type, last_vertex);
        }
        indices = assembled.indices;
        draw_count = assembled.count;
        primitive = assembled.primitive;
        index_type = MTLIndexTypeUInt32;
    }
    IndirectDraws::Batch batch{};
    if (segmented && !indirect) {
        const auto pages = buffer_cache.CreatePageSnapshot();
        const auto assembled = primitive_assembly.AssembleIndexedRestartPaged(
            regs.primitive_type, draw_count, regs.num_instances.NumInstances(),
            std::bit_cast<s32>(base_vertex), first_instance, *pages, index_address, index_size,
            index_type, last_vertex, regs.primitive_restart_index,
            indirect_draws.PrepareErrorBuffer());
        indices = assembled.indices;
        arguments = assembled.arguments;
        primitive = assembled.primitive;
        index_type = MTLIndexTypeUInt32;
        batch = indirect_draws.Encode(primitive, arguments, assembled.count, 1,
                                      sizeof(MTLDrawIndexedPrimitivesIndirectArguments), indices,
                                      index_type, true);
    } else if (indirect && assemble) {
        const auto pages = buffer_cache.CreatePageSnapshot();
        const auto assembled = primitive_assembly.AssembleIndexedPaged(
            regs.primitive_type, index_address, index_size, index_type, *pages, indirect->address,
            indirect->count_address, indirect->max_count, indirect->stride, last_vertex,
            indirect_draws.PrepareErrorBuffer(),
            restart ? std::optional{regs.primitive_restart_index} : std::nullopt);
        indices = assembled.indices;
        arguments = assembled.arguments;
        primitive = assembled.primitive;
        index_type = MTLIndexTypeUInt32;
        batch = indirect_draws.Encode(primitive, arguments, assembled.count, indirect->max_count,
                                      sizeof(MTLDrawIndexedPrimitivesIndirectArguments), indices,
                                      index_type, indexed);
    }
    if (indirect && !assemble) {
        const auto pages = buffer_cache.CreatePageSnapshot();
        batch = indirect_draws.EncodePaged(primitive, *pages, indirect->address,
                                           indirect->count_address, indirect->max_count,
                                           indirect->stride, indices, index_type, indexed);
    }
    resources.MarkWrites(stages);
    mark_targets();
    auto vertex_snapshot = resources.Snapshot(*vertex_stage, vertex_buffers);
    auto fragment_snapshot = resources.Snapshot(*fragment_stage);
    visibility.Prepare(render, graphics, regs.reg_array[0xa001]);
    const MTLStages render_stages =
        MTLStageVertex | MTLStageFragment | (render.visibility_buffer ? MTLStageTile : 0);
    const bool shader_writes = vertex_stage->HasWrites() || fragment_stage->HasWrites();
    if (shader_writes) {
        gathered_indices.valid = false;
        for (auto& cached : gathered_vertices) {
            cached.valid = false;
        }
        scheduler.EndEncoding();
    }
    (void)commands.BindGraphics(render, graphics, pipeline, vertex_snapshot.get(),
                                fragment_snapshot.get());
    scheduler.QueueBarrier(MTLStageAll, render_stages);
    if (use_batch) {
        commands.ExecuteIndirect(batch.commands, batch.range);
    } else if (arguments.buffer) {
        commands.DrawIndexedIndirect(primitive, index_type, indices, arguments);
    } else if (indexed || assemble) {
        commands.DrawIndexed(primitive, index_type, indices, draw_count,
                             regs.num_instances.NumInstances(), std::bit_cast<s32>(base_vertex),
                             first_instance);
    } else {
        commands.Draw(primitive, regs.num_indices, regs.num_instances.NumInstances(), base_vertex,
                      first_instance);
    }
    scheduler.ProducerBarrier(render_stages, MTLStageAll);
    if (shader_writes)
        scheduler.EndEncoding();
    if (use_batch)
        indirect_draws.ReleaseAfterDraw(batch);
    if (assemble)
        primitive_assembly.ReleaseAfterDraw(indices, arguments);
    if (metadata_info) {
        const auto tiles = metadata_coverage.ReadTiles(*metadata_info, render);
        texture_cache.ProcessColorMetadata(*metadata_info,
                                           regs.color_control.mode == ColorMode::EliminateFastClear
                                               ? Cmask::Operation::EliminateFastClear
                                               : Cmask::Operation::DecompressFmask,
                                           tiles);
    }
}

BufferRange GraphicsRenderer::GatherIndices(VAddr address, u64 size) {
    return GatherBuffer(address, size, gathered_indices);
}

BufferRange GraphicsRenderer::GatherBuffer(VAddr address, u64 size, GatheredBuffer& cached) {
    if (!size || size > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Metal draw input exceeds native buffer capacity");
    }
    auto ranges = buffer_cache.GetPagedBufferRanges(address, size);
    size_t range_count{};
    for (const auto& range : ranges) {
        if (range_count && ranges[range_count - 1].buffer == range.buffer &&
            ranges[range_count - 1].offset + ranges[range_count - 1].size == range.offset) {
            ranges[range_count - 1].size += range.size;
        } else {
            ranges[range_count++] = range;
        }
    }
    ranges.resize(range_count);
    if (ranges.size() == 1) {
        return ranges.front();
    }
    const auto version = buffer_cache.GetRegionVersion(address, size);
    const bool cpu_modified = buffer_cache.IsRegionCpuModified(address, size);
    if (cached.valid && !cpu_modified && cached.address == address && cached.size == size &&
        cached.version == version && cached.sources.size() == ranges.size() &&
        std::equal(cached.sources.begin(), cached.sources.end(), ranges.begin(),
                   [](const auto& source, const auto& range) {
                       return source.buffer == range.buffer && source.offset == range.offset &&
                              source.size == range.size;
                   })) {
        scheduler.UseResource(cached.buffer);
        return {cached.buffer, 0, size};
    }
    cached.valid = false;
    auto& scratch = cached.buffer;
    if (!scratch || scratch.length < size) {
        scratch = instance.CreateBuffer(size, MTLResourceStorageModePrivate);
    }
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    u64 offset{};
    for (const auto& range : ranges) {
        commands.CopyBuffer(range, {scratch, offset, range.size}, range.size);
        offset += range.size;
    }
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    cached.address = address;
    cached.size = size;
    cached.version = version;
    cached.sources.clear();
    cached.sources.reserve(ranges.size());
    for (const auto& range : ranges) {
        cached.sources.push_back({range.buffer, range.offset, range.size});
    }
    cached.valid = !cpu_modified;
    return {scratch, 0, size};
}

std::vector<BufferRange> GraphicsRenderer::GatherVertexBuffers(
    std::span<const std::pair<VAddr, u32>> ranges) {
    if (ranges.size() > gathered_vertices.size()) {
        throw std::runtime_error("Metal vertex input exceeds native binding capacity");
    }
    std::vector<BufferRange> result;
    result.reserve(ranges.size());
    for (u32 i = 0; i < ranges.size(); ++i) {
        const auto previous = std::find(ranges.begin(), ranges.begin() + i, ranges[i]);
        if (previous != ranges.begin() + i) {
            result.push_back(result[previous - ranges.begin()]);
            continue;
        }
        const auto& [address, size] = ranges[i];
        result.push_back(GatherBuffer(address, size, gathered_vertices[i]));
    }
    return result;
}

void GraphicsRenderer::DumpPixelCounter(u32 counter, const BufferRange& destination, u32 pairs,
                                        u32 stride) {
    visibility.Dump(counter, destination, pairs, stride);
}

void GraphicsRenderer::ResetPixelCounter(u32 counter) {
    visibility.Reset(counter);
}

} // namespace Metal
