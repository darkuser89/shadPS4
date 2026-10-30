// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_metal/metal_geometry_copy.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

bool ValidOutput(IR::Attribute attribute, u32 component) {
    if (attribute == IR::Attribute::Position0 || IR::IsParam(attribute)) {
        return component < 4;
    }
    if (attribute == IR::Attribute::ClipDistance || attribute == IR::Attribute::CullDistance) {
        return component < 8;
    }
    return component == 0 && (attribute == IR::Attribute::PointSize ||
                              attribute == IR::Attribute::RenderTargetIndex ||
                              attribute == IR::Attribute::ViewportIndex);
}

u32 ActiveStreams(const GeometryAssemblyParameters& parameters) {
    u32 active{};
    for (u32 stream = 0; stream < 4; ++stream) {
        active |= parameters.mesh_counts[stream] ? 1u << stream : 0;
    }
    return active;
}

u32 AddBuffer(Shader::Info& info, Shader::BufferType type, bool written) {
    const auto index = u32(info.buffers.size());
    auto& buffer = info.buffers.emplace_back();
    buffer.buffer_type = type;
    buffer.used_types = IR::Type::U32;
    buffer.is_written = written;
    return index;
}

void PredicateExports(IR::Program& program, IR::IREmitter& initialize) {
    std::map<std::pair<IR::Attribute, u32>, IR::VirtualReg> outputs;
    for (auto* block : program.blocks) {
        for (const IR::Inst& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
                continue;
            }
            const auto key = std::pair{inst.Arg(0).Attribute(), inst.Arg(2).U32()};
            if (outputs.contains(key)) {
                continue;
            }
            if (program.next_reg_index == std::numeric_limits<u32>::max()) {
                throw std::runtime_error("Native geometry copy export register capacity exceeded");
            }
            const IR::VirtualReg value{program.next_reg_index++, IR::Type::U32};
            outputs.emplace(key, value);
            initialize.SetVirtualReg(value, initialize.Imm32(0u));
        }
    }
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const auto output = outputs.at({inst.Arg(0).Attribute(), inst.Arg(2).U32()});
            const IR::U32 value{ir.Select(ir.GetExec(), ir.BitCast<IR::U32>(IR::F32{inst.Arg(1)}),
                                          IR::U32{ir.GetVirtualReg(output)})};
            ir.SetVirtualReg(output, value);
            inst.SetArg(1, ir.BitCast<IR::F32>(value));
        }
    }
}
} // namespace

GeometryCopyTranslation::GeometryCopyTranslation(const GeometryCopyConfig& config_,
                                                 const Shader::RuntimeInfo& runtime,
                                                 const Shader::Profile& profile)
    : config{config_}, vertex_runtime{runtime.hw.vs}, num_user_data{runtime.props.num_user_data},
      viewport_width{std::min(profile.max_viewport_width / 2, 8192u)},
      viewport_height{std::min(profile.max_viewport_height / 2, 8192u)} {
    const auto& plan = config.emissions;
    if (runtime.hw_stage != Shader::HwStage::Vertex ||
        runtime.sw_stage != Shader::SwStage::Vertex || !profile.needs_geometry_copy_lowering ||
        !profile.needs_memory_predication || !profile.needs_vector_register_predication ||
        num_user_data > Shader::NUM_USER_DATA_REGS ||
        (vertex_runtime.user_clip_plane_mask & ~0x3fu) || runtime.sw.vs.tess_emulated_primitive ||
        (vertex_runtime.clip_disable && (!viewport_width || !viewport_height)) ||
        !config.work_items || config.work_items % 64 || config.stream >= 4 ||
        !plan.mesh_counts[config.stream] || !plan.invocations || !plan.max_vertices ||
        plan.vertex_words != 1 || config.shared_memory_size % 4 ||
        config.shared_memory_size > profile.max_shared_memory_size ||
        config.InputSizeBytes() / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1 ||
        plan.EmissionSizeBytes() / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1 ||
        plan != GeometryAssemblyLayout(plan.invocations, plan.max_vertices, 1, plan.mesh_vertices,
                                       plan.topology, ActiveStreams(plan))) {
        throw std::runtime_error("Invalid native geometry copy context");
    }
}

Shader::TranslationHooks GeometryCopyTranslation::Hooks() {
    Shader::TranslationHooks hooks;
    hooks.prologue = [this](IR::Program& program, IR::IREmitter& ir) {
        if (program.info.hw_stage != Shader::HwStage::Vertex ||
            program.info.sw_stage != Shader::SwStage::Vertex) {
            throw std::runtime_error("Native geometry copy requires a vertex shader");
        }
        for (u32 scalar = 0; scalar < num_user_data; ++scalar) {
            const auto reg = IR::ScalarReg(scalar);
            ir.SetScalarReg(reg, ir.GetUserData(reg));
        }
        for (u32 scalar = 0; scalar < GeometryCopyInput{}.system_sgprs.size(); ++scalar) {
            ir.SetScalarReg(IR::ScalarReg(num_user_data + scalar),
                            ir.GetAttributeU32(IR::Attribute::GeometryCopyScalar, scalar));
        }
        for (u32 vector = 0; vector < GeometryCopyInput{}.vgprs.size(); ++vector) {
            ir.SetVectorReg(IR::VectorReg(vector),
                            ir.GetAttributeU32(IR::Attribute::GeometryCopyVector, vector));
        }
        InitializeGeometryExec(program, ir, IR::Attribute::GeometryCopyActive);
        PredicateExports(program, ir);
    };
    hooks.lower_rings = [this](IR::Program& program, Shader::RuntimeInfo& runtime) {
        std::map<std::pair<IR::Attribute, u32>, u32> outputs;
        for (auto* block : program.post_order_blocks) {
            for (const IR::Inst& inst : block->Instructions()) {
                if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
                    continue;
                }
                const auto attribute = inst.Arg(0).Attribute();
                const u32 component = inst.Arg(2).U32();
                if (!ValidOutput(attribute, component)) {
                    throw std::runtime_error("Unsupported native geometry copy export");
                }
                outputs.try_emplace({attribute, component}, 0);
            }
        }
        clip_planes.Collect(outputs, vertex_runtime.user_clip_plane_mask);
        if (outputs.empty() ||
            outputs.size() > std::numeric_limits<u32>::max() - program.next_reg_index) {
            throw std::runtime_error("Native geometry copy output capacity exceeded");
        }
        if (vertex_runtime.clip_disable && outputs.contains({IR::Attribute::ViewportIndex, 0})) {
            result.viewport_extent = {viewport_width, viewport_height};
        }
        if ((vertex_runtime.clip_disable || vertex_runtime.emulate_depth_negative_one_to_one) &&
            std::ranges::any_of(std::array{0u, 1u, 2u, 3u}, [&](u32 component) {
                return !outputs.contains({IR::Attribute::Position0, component});
            })) {
            throw std::runtime_error("Native geometry copy requires complete position exports");
        }
        for (const auto& [output, _] : outputs) {
            result.attributes.push_back(
                {output.first, output.second, config.stream, u32(result.attributes.size() + 1)});
            values.push_back(IR::VirtualReg{program.next_reg_index++, IR::Type::U32});
        }
        const auto& plan = config.emissions;
        result.payload =
            GeometryAssemblyLayout(plan.invocations, plan.max_vertices, u32(outputs.size() + 1),
                                   plan.mesh_vertices, plan.topology, 1u << config.stream);
        if (result.payload.EmissionSizeBytes() / sizeof(u32) >
            u64{std::numeric_limits<u32>::max()} + 1) {
            throw std::runtime_error("Native geometry copy payload exceeds DWORD capacity");
        }
        program.info.hw_stage = Shader::HwStage::Compute;
        program.info.sw_stage = Shader::SwStage::Compute;
        runtime.hw_stage = Shader::HwStage::Compute;
        runtime.sw_stage = Shader::SwStage::Compute;
        runtime.hw.cs = {config.shared_memory_size, {64, 1, 1}, {false, false, false}};
        return true;
    };
    hooks.lower_resources = [this](IR::Program& program) {
        result.input_sources = InspectGeometryInputSources(program);
        result.address_program = std::make_shared<const GeometryBufferAddressProgram>(
            InspectGeometryBufferAddressProgram(program));
        result.address_origins = std::make_shared<const GeometryBufferAddressOrigins>(
            InspectGeometryBufferAddressOrigins(program));
        if (program.blocks.empty() ||
            program.info.buffers.size() + 3 + bool(vertex_runtime.user_clip_plane_mask) >
                Shader::NUM_BUFFERS) {
            throw std::runtime_error("Native geometry copy buffer capacity exceeded");
        }
        result.inputs = AddBuffer(program.info, Shader::BufferType::GeometryInputs, false);
        result.emissions = AddBuffer(program.info, Shader::BufferType::GeometryEmissions, false);
        result.output = AddBuffer(program.info, Shader::BufferType::GeometryPayload, true);
        clip_planes.Bind(program);
        auto* entry = program.blocks.front();
        IR::IREmitter initialize{*entry, entry->begin()};
        const auto work_item = initialize.IAdd(
            initialize.IMul(initialize.GetAttributeU32(IR::Attribute::WorkgroupId, 0),
                            initialize.Imm32(64u)),
            initialize.GetAttributeU32(IR::Attribute::LocalInvocationId, 0));
        input_base = IR::U32{initialize.IMul(
            work_item, initialize.Imm32(u32{sizeof(GeometryCopyInput) / sizeof(u32)}))};
        const auto load_input = [&](u32 word) {
            return IR::U32{
                initialize.LoadBufferU32(1, initialize.Imm32(result.inputs),
                                         initialize.IAdd(input_base, initialize.Imm32(word)), {})};
        };
        const auto invocation = load_input(offsetof(GeometryCopyInput, invocation) / sizeof(u32));
        const auto vertex = load_input(offsetof(GeometryCopyInput, vertex) / sizeof(u32));
        const auto& plan = config.emissions;
        const auto safe_invocation =
            initialize.UMin(invocation, initialize.Imm32(plan.invocations - 1));
        const auto safe_vertex = initialize.UMin(vertex, initialize.Imm32(plan.max_vertices - 1));
        const auto slot = initialize.IAdd(initialize.IMul(safe_invocation, initialize.Imm32(4u)),
                                          initialize.Imm32(config.stream));
        const auto metadata = initialize.IMul(slot, initialize.Imm32(plan.stream_stride_words));
        const auto count =
            IR::U32{initialize.LoadBufferU32(1, initialize.Imm32(result.emissions), metadata, {})};
        output_valid = initialize.LogicalAnd(
            initialize.ILessThan(invocation, initialize.Imm32(plan.invocations), false),
            initialize.LogicalAnd(
                initialize.ILessThan(vertex, count, false),
                initialize.ILessThanEqual(count, initialize.Imm32(plan.max_vertices), false)));
        const auto ordinal_address =
            initialize.IAdd(metadata, initialize.IAdd(initialize.Imm32(4u), safe_vertex));
        ordinal = IR::U32{initialize.LoadBufferU32(
            1, initialize.Imm32(result.emissions),
            initialize.GuardBufferAddress(IR::U32{ordinal_address}, output_valid), {})};
        output_row = IR::U32{initialize.IAdd(
            initialize.IMul(slot, initialize.Imm32(result.payload.stream_stride_words)),
            initialize.IAdd(
                initialize.Imm32(4u),
                initialize.IMul(safe_vertex, initialize.Imm32(result.payload.vertex_words))))};
        for (const auto& value : values) {
            initialize.SetVirtualReg(value, initialize.Imm32(0u));
        }
        for (auto* block : program.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                if (inst.GetOpcode() == IR::Opcode::GetAttributeU32) {
                    const auto attribute = inst.Arg(0).Attribute();
                    if (attribute == IR::Attribute::GeometryCopyActive) {
                        inst.ReplaceUsesWithAndRemove(
                            ir.Select(output_valid, ir.Imm32(1u), ir.Imm32(0u)));
                        continue;
                    }
                    if (attribute != IR::Attribute::GeometryCopyScalar &&
                        attribute != IR::Attribute::GeometryCopyVector) {
                        continue;
                    }
                    const u32 word = inst.Arg(1).U32() +
                                     (attribute == IR::Attribute::GeometryCopyVector ? 16u : 0u);
                    inst.ReplaceUsesWithAndRemove(ir.LoadBufferU32(
                        1, ir.Imm32(result.inputs), ir.IAdd(input_base, ir.Imm32(word)), {}));
                } else if (inst.GetOpcode() == IR::Opcode::SetAttribute) {
                    const auto attribute =
                        std::ranges::find_if(result.attributes, [&](const auto& out) {
                            return out.attribute == inst.Arg(0).Attribute() &&
                                   out.component == inst.Arg(2).U32();
                        });
                    if (attribute == result.attributes.end()) {
                        throw std::runtime_error("Unmapped native geometry copy export");
                    }
                    ir.SetVirtualReg(values[attribute - result.attributes.begin()],
                                     ir.BitCast<IR::U32>(IR::F32{inst.Arg(1)}));
                    inst.Invalidate();
                }
            }
        }
    };
    hooks.structured = [this](IR::Program& program) {
        LowerGeometryWaveOperations(program);
        for (auto* block : program.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                if (inst.GetOpcode() != IR::Opcode::Epilogue) {
                    continue;
                }
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                std::vector<IR::U32> exported;
                for (const auto& value : values) {
                    exported.push_back(IR::U32{ir.GetVirtualReg(value)});
                }
                const auto position = [&](u32 component) {
                    const auto it = std::ranges::find_if(result.attributes, [&](const auto& out) {
                        return out.attribute == IR::Attribute::Position0 &&
                               out.component == component;
                    });
                    return size_t(it - result.attributes.begin());
                };
                clip_planes.Evaluate(ir, result.attributes, exported);
                if (vertex_runtime.emulate_depth_negative_one_to_one) {
                    const auto z = position(2);
                    const auto w = position(3);
                    exported[z] = ir.BitCast<IR::U32>(
                        IR::F32{ir.FPMul(ir.FPAdd(ir.BitCast<IR::F32>(exported[z]),
                                                  ir.BitCast<IR::F32>(exported[w])),
                                         ir.Imm32(0.5f))});
                }
                if (vertex_runtime.clip_disable && !result.viewport_extent[0]) {
                    for (u32 component = 0; component < 2; ++component) {
                        const auto load_viewport = [&](u32 word) {
                            return ir.BitCast<IR::F32>(IR::U32{ir.LoadBufferU32(
                                1, ir.Imm32(result.inputs),
                                ir.IAdd(input_base,
                                        ir.Imm32(u32{offsetof(GeometryCopyInput, viewport) /
                                                     sizeof(u32)} +
                                                 word)),
                                {})});
                        };
                        const auto index = position(component);
                        const auto window = ir.FPAdd(ir.FPMul(ir.BitCast<IR::F32>(exported[index]),
                                                              load_viewport(component + 2)),
                                                     load_viewport(component));
                        const auto size = component == 0 ? viewport_width : viewport_height;
                        exported[index] = ir.BitCast<IR::U32>(IR::F32{
                            ir.FPSub(ir.FPDiv(window, ir.Imm32(float(size))), ir.Imm32(1.f))});
                    }
                }
                const auto store = [&](u32 word, const IR::U32& value) {
                    ir.StoreBufferU32(
                        1, ir.Imm32(result.output),
                        ir.GuardBufferAddress(IR::U32{ir.IAdd(output_row, ir.Imm32(word))},
                                              output_valid),
                        value, {});
                };
                store(0, ordinal);
                for (u32 index = 0; index < result.attributes.size(); ++index) {
                    store(result.attributes[index].word, exported[index]);
                }
            }
        }
    };
    return hooks;
}

void GeometryCopyTranslation::LowerWorkgroups(IR::Program& program) const {
    if (config.compact_workgroups) {
        LowerGeometryWorkgroupMap(program, result.inputs,
                                  u32(config.RegisterSizeBytes() / sizeof(u32)));
    }
}

} // namespace Metal
