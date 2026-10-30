// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <limits>
#include <stdexcept>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_metal/metal_geometry_export.h"
#include "video_core/renderer_metal/metal_geometry_output.h"

namespace Metal {
namespace IR = Shader::IR;

GeometryExportTranslation::GeometryExportTranslation(const GeometryExportConfig& config_,
                                                     const Shader::RuntimeInfo& runtime,
                                                     const Shader::Profile& profile)
    : config{config_}, num_user_data{runtime.props.num_user_data} {
    if (!profile.needs_geometry_export_lowering || !profile.needs_memory_predication ||
        !profile.needs_vector_register_predication ||
        runtime.hw_stage !=
            (config.raster_output ? Shader::HwStage::Vertex : Shader::HwStage::Export) ||
        runtime.sw_stage != Shader::SwStage::Vertex || num_user_data > Shader::NUM_USER_DATA_REGS ||
        config.shared_memory_size % 4 ||
        config.shared_memory_size > profile.max_shared_memory_size ||
        config.InputSizeBytes(config.dynamic_work_items ? 64 : 0) / sizeof(u32) >
            u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Invalid native geometry export context");
    }
    if (config.raster_output) {
        raster.emplace(runtime, profile);
    }
    if (config.indirect &&
        (config.indirect->vertex_sgpr_offset != runtime.sw.vs.vertex_sgpr_offset ||
         config.indirect->instance_sgpr_offset != runtime.sw.vs.instance_sgpr_offset)) {
        throw std::runtime_error("Native indirect raster SGPR layout differs from its runtime");
    }
    if (config.raster_output && !config.indirect &&
        (runtime.sw.vs.vertex_sgpr_offset || runtime.sw.vs.instance_sgpr_offset)) {
        throw std::runtime_error("Native raster SGPR copies require an indirect input layout");
    }
}

Shader::TranslationHooks GeometryExportTranslation::Hooks() {
    Shader::TranslationHooks hooks;
    hooks.prologue = [this](IR::Program& program, IR::IREmitter& ir) {
        if (program.info.hw_stage !=
                (config.raster_output ? Shader::HwStage::Vertex : Shader::HwStage::Export) ||
            program.info.sw_stage != Shader::SwStage::Vertex) {
            throw std::runtime_error("Native geometry export shader stage mismatch");
        }
        for (u32 scalar = 0; scalar < num_user_data; ++scalar) {
            const auto reg = IR::ScalarReg(scalar);
            ir.SetScalarReg(reg, ir.GetUserData(reg));
        }
        for (u32 scalar = 0; scalar < GeometryExportInput{}.system_sgprs.size(); ++scalar) {
            ir.SetScalarReg(IR::ScalarReg(num_user_data + scalar),
                            ir.GetAttributeU32(IR::Attribute::GeometryExportScalar, scalar));
        }
        if (config.indirect) {
            if (const auto reg =
                    RasterIndirectRegisters::UserDataIndex(config.indirect->vertex_sgpr_offset)) {
                ir.SetScalarReg(IR::ScalarReg(*reg), ir.GetAttributeU32(IR::Attribute::BaseVertex));
            }
            if (const auto reg =
                    RasterIndirectRegisters::UserDataIndex(config.indirect->instance_sgpr_offset)) {
                ir.SetScalarReg(IR::ScalarReg(*reg),
                                ir.GetAttributeU32(IR::Attribute::BaseInstance));
            }
        }
        for (u32 vector = 0; vector < GeometryExportInput{}.vgprs.size(); ++vector) {
            ir.SetVectorReg(IR::VectorReg(vector),
                            ir.GetAttributeU32(IR::Attribute::GeometryExportVector, vector));
        }
        InitializeGeometryExec(program, ir, IR::Attribute::GeometryInputActive);
    };
    hooks.lower_rings = [this](IR::Program& program, Shader::RuntimeInfo& runtime) {
        if (raster) {
            raster->Collect(program);
        } else {
            for (auto* block : program.post_order_blocks) {
                for (const IR::Inst& inst : *block) {
                    if (inst.GetOpcode() == IR::Opcode::SetAttribute) {
                        throw std::runtime_error("Native ES exports require guest ring stores");
                    }
                }
            }
        }
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
        if (program.blocks.empty() || program.info.buffers.size() == Shader::NUM_BUFFERS) {
            throw std::runtime_error("Native ES input exceeds shader buffer capacity");
        }
        bindings.inputs = u32(program.info.buffers.size());
        auto& input = program.info.buffers.emplace_back();
        input.buffer_type = Shader::BufferType::GeometryInputs;
        input.used_types = IR::Type::U32;
        auto* entry = program.blocks.front();
        IR::IREmitter initialize{*entry, entry->begin()};
        const auto work_item = initialize.IAdd(
            initialize.IMul(initialize.GetAttributeU32(IR::Attribute::WorkgroupId, 0),
                            initialize.Imm32(64u)),
            initialize.GetAttributeU32(IR::Attribute::LocalInvocationId, 0));
        auto input_base = initialize.IMul(
            work_item, initialize.Imm32(u32{sizeof(GeometryExportInput) / sizeof(u32)}));
        if (config.dynamic_work_items) {
            input_base = initialize.IAdd(
                input_base, initialize.Imm32(u32(config.HeaderSizeBytes() / sizeof(u32))));
        }
        if (raster) {
            raster->Lower(program, IR::U32{work_item}, IR::U32{input_base}, bindings.inputs,
                          config.dynamic_work_items
                              ? 0
                              : u32((config.InputSizeBytes() - 4 * sizeof(float)) / sizeof(u32)));
            bindings.raster = raster->Bindings();
        }
        for (auto* block : program.blocks) {
            for (IR::Inst& inst : *block) {
                if (inst.GetOpcode() != IR::Opcode::GetAttributeU32) {
                    continue;
                }
                const auto attribute = inst.Arg(0).Attribute();
                if (attribute == IR::Attribute::BaseVertex ||
                    attribute == IR::Attribute::BaseInstance) {
                    if (!config.indirect || !inst.Arg(1).IsImmediate() || inst.Arg(1).U32()) {
                        throw std::runtime_error(
                            "Native raster base input requires an indirect header");
                    }
                    const u32 word = u32((attribute == IR::Attribute::BaseVertex
                                              ? offsetof(GeometryIndirectHeader, base_vertex)
                                              : offsetof(GeometryIndirectHeader, first_instance)) /
                                         sizeof(u32));
                    IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                    inst.ReplaceUsesWithAndRemove(
                        ir.LoadBufferU32(1, ir.Imm32(bindings.inputs), ir.Imm32(word), {}));
                    continue;
                }
                if (attribute == IR::Attribute::GeometryInputActive) {
                    IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                    inst.ReplaceUsesWithAndRemove(ir.LoadBufferU32(
                        1, ir.Imm32(bindings.inputs),
                        ir.IAdd(input_base,
                                ir.Imm32(u32{offsetof(GeometryExportInput, active) / sizeof(u32)})),
                        {}));
                    continue;
                }
                if (attribute != IR::Attribute::GeometryExportScalar &&
                    attribute != IR::Attribute::GeometryExportVector) {
                    continue;
                }
                const u32 word = inst.Arg(1).U32() +
                                 (attribute == IR::Attribute::GeometryExportVector ? 16u : 0u);
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                inst.ReplaceUsesWithAndRemove(ir.LoadBufferU32(
                    1, ir.Imm32(bindings.inputs), ir.IAdd(input_base, ir.Imm32(word)), {}));
            }
        }
    };
    hooks.structured = [this](IR::Program& program) {
        if (raster) {
            raster->Finish(program);
        }
        LowerGeometryWaveOperations(program);
    };
    return hooks;
}

} // namespace Metal
