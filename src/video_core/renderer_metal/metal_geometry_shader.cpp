// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <limits>
#include <stdexcept>
#include "common/scope_exit.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_metal/metal_geometry_output.h"
#include "video_core/renderer_metal/metal_geometry_shader.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

u32 AddBuffer(Shader::Info& info, Shader::BufferType type, bool written) {
    const auto index = u32(info.buffers.size());
    auto& buffer = info.buffers.emplace_back();
    buffer.buffer_type = type;
    buffer.used_types = IR::Type::U32;
    buffer.is_written = written;
    return index;
}

} // namespace

GeometryKernelTranslation::GeometryKernelTranslation(const GeometryKernelConfig& config_,
                                                     const Shader::RuntimeInfo& runtime,
                                                     const Shader::Profile& profile)
    : config{config_}, num_user_data{runtime.props.num_user_data} {
    const auto& plan = config.emissions;
    if (!profile.needs_memory_predication || !profile.needs_vector_register_predication ||
        !profile.needs_geometry_message_predication ||
        runtime.hw_stage != Shader::HwStage::Geometry ||
        runtime.sw_stage != Shader::SwStage::Geometry ||
        num_user_data > Shader::NUM_USER_DATA_REGS || !plan.invocations || plan.invocations % 64 ||
        plan.vertex_words != 1 || !plan.jobs ||
        plan.max_vertices != runtime.hw.gs.output_vertices || config.shared_memory_size % 4 ||
        config.shared_memory_size > profile.max_shared_memory_size ||
        config.InputSizeBytes() / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1 ||
        plan.EmissionSizeBytes() / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Invalid native geometry kernel context");
    }
    u32 active_streams{};
    for (u32 stream = 0; stream < 4; ++stream) {
        if (plan.mesh_counts[stream]) {
            active_streams |= 1u << stream;
            if (plan.topology[stream] != u32(runtime.hw.gs.out_primitive[stream])) {
                throw std::runtime_error("Native geometry kernel topology mismatch");
            }
        }
    }
    if (plan != GeometryAssemblyLayout(plan.invocations, plan.max_vertices, 1, plan.mesh_vertices,
                                       plan.topology, active_streams)) {
        throw std::runtime_error("Invalid native geometry kernel emission layout");
    }
}

Shader::TranslationHooks GeometryKernelTranslation::Hooks() {
    Shader::TranslationHooks hooks;
    hooks.prologue = [this](IR::Program& program, IR::IREmitter& ir) {
        if (program.info.hw_stage != Shader::HwStage::Geometry ||
            program.info.sw_stage != Shader::SwStage::Geometry) {
            throw std::runtime_error("Native geometry kernel requires a geometry shader");
        }
        ir.SetScalarReg(IR::ScalarReg(num_user_data),
                        ir.GetAttributeU32(IR::Attribute::GeometryRingOffset));
        ir.SetScalarReg(IR::ScalarReg(num_user_data + 1),
                        ir.GetAttributeU32(IR::Attribute::GeometryWaveId));
        constexpr std::array registers{IR::VectorReg::V0, IR::VectorReg::V1, IR::VectorReg::V3,
                                       IR::VectorReg::V4, IR::VectorReg::V5, IR::VectorReg::V6};
        for (u32 vertex = 0; vertex < registers.size(); ++vertex) {
            ir.SetVectorReg(registers[vertex],
                            ir.GetAttributeU32(IR::Attribute::GeometryVertexOffset, vertex));
        }
        ir.SetVectorReg(IR::VectorReg::V2, ir.GetAttributeU32(IR::Attribute::PrimitiveId));
        ir.SetVectorReg(IR::VectorReg::V7, ir.GetAttributeU32(IR::Attribute::InvocationId));
        InitializeGeometryExec(program, ir, IR::Attribute::GeometryInputActive);
    };
    hooks.lower_rings = [this](IR::Program& program, Shader::RuntimeInfo& runtime) {
        program.info.hw_stage = Shader::HwStage::Compute;
        program.info.sw_stage = Shader::SwStage::Compute;
        runtime.hw_stage = Shader::HwStage::Compute;
        runtime.sw_stage = Shader::SwStage::Compute;
        runtime.hw.cs = {config.shared_memory_size, {64, 1, 1}, {false, false, false}};
        return true;
    };
    hooks.lower_resources = [this](IR::Program& program) {
        bindings.input_sources = InspectGeometryInputSources(program);
        bindings.address_program = std::make_shared<const GeometryBufferAddressProgram>(
            InspectGeometryBufferAddressProgram(program));
        bindings.address_origins = std::make_shared<const GeometryBufferAddressOrigins>(
            InspectGeometryBufferAddressOrigins(program));
        if (program.info.buffers.size() + 2 > Shader::NUM_BUFFERS || program.blocks.empty()) {
            throw std::runtime_error("Native geometry kernel buffer capacity exceeded");
        }
        bindings.inputs = AddBuffer(program.info, Shader::BufferType::GeometryInputs, false);
        bindings.emissions = AddBuffer(program.info, Shader::BufferType::GeometryEmissions, true);
        auto* entry = program.blocks.front();
        IR::IREmitter initialize{*entry, entry->begin()};
        const auto invocation = IR::U32{initialize.IAdd(
            initialize.IMul(initialize.GetAttributeU32(IR::Attribute::WorkgroupId, 0),
                            initialize.Imm32(64u)),
            initialize.GetAttributeU32(IR::Attribute::LocalInvocationId, 0))};
        const auto input_base = IR::U32{initialize.IMul(
            invocation, initialize.Imm32(u32{sizeof(GeometryKernelInput) / sizeof(u32)}))};
        for (auto* block : program.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                if (inst.GetOpcode() != IR::Opcode::GetAttributeU32) {
                    continue;
                }
                u32 slot{};
                switch (inst.Arg(0).Attribute()) {
                case IR::Attribute::GeometryInputActive:
                    slot = offsetof(GeometryKernelInput, active) / sizeof(u32);
                    break;
                case IR::Attribute::GeometryRingOffset:
                    slot = 0;
                    break;
                case IR::Attribute::GeometryWaveId:
                    slot = 1;
                    break;
                case IR::Attribute::GeometryVertexOffset:
                    slot = inst.Arg(1).U32() + 2;
                    break;
                case IR::Attribute::PrimitiveId:
                    slot = 8;
                    break;
                case IR::Attribute::InvocationId:
                    slot = 9;
                    break;
                default:
                    continue;
                }
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                const auto address = ir.IAdd(input_base, ir.Imm32(slot));
                inst.ReplaceUsesWithAndRemove(
                    ir.LoadBufferU32(1, ir.Imm32(bindings.inputs), address, {}));
            }
        }
        const auto stage = program.info.hw_stage;
        SCOPE_EXIT {
            program.info.hw_stage = stage;
        };
        program.info.hw_stage = Shader::HwStage::Geometry;
        LowerGeometryOutputs(program, config.emissions, {}, bindings.emissions, invocation);
    };
    hooks.structured = LowerGeometryWaveOperations;
    return hooks;
}

void GeometryKernelTranslation::LowerWorkgroups(IR::Program& program) const {
    if (config.compact_workgroups) {
        LowerGeometryWorkgroupMap(program, bindings.inputs,
                                  u32(config.RegisterSizeBytes() / sizeof(u32)));
    }
}

} // namespace Metal
