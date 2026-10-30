// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"
#include "video_core/renderer_metal/metal_geometry_output.h"
#include "video_core/renderer_metal/metal_tessellation_kernel.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

void LowerWaveRegisters(IR::Program& program) {
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : *block) {
            const auto opcode = inst.GetOpcode();
            if (opcode != IR::Opcode::InverseBallot && opcode != IR::Opcode::WriteLane &&
                opcode != IR::Opcode::ReadFirstLane) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            if (opcode == IR::Opcode::ReadFirstLane) {
                inst.ReplaceUsesWithAndRemove(
                    ir.ReadLane(IR::U32{inst.Arg(0)}, ir.BallotFindLsb(ir.Ballot(ir.GetExec()))));
                continue;
            }
            const auto lane = ir.BitwiseAnd(ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex),
                                            ir.Imm32(63u));
            if (opcode == IR::Opcode::WriteLane) {
                const auto target = ir.BitwiseAnd(IR::U32{inst.Arg(2)}, ir.Imm32(63u));
                inst.ReplaceUsesWithAndRemove(
                    ir.Select(ir.IEqual(lane, target), inst.Arg(1), inst.Arg(0)));
            } else {
                const auto bit = ir.BitwiseAnd(
                    IR::U64{ir.ShiftRightLogical(IR::U64{inst.Arg(0)}, lane)}, ir.Imm64(u64{1}));
                inst.ReplaceUsesWithAndRemove(ir.INotEqual(bit, ir.Imm64(u64{0})));
            }
        }
    }
}

} // namespace

TessellationKernelTranslation::TessellationKernelTranslation(
    const TessellationKernelConfig& config_, const Shader::RuntimeInfo& runtime_,
    const Shader::Profile& profile_)
    : config{config_}, runtime{runtime_}, profile{profile_} {
    const bool local =
        runtime.hw_stage == Shader::HwStage::Local && runtime.sw_stage == Shader::SwStage::Vertex;
    const bool hull = runtime.hw_stage == Shader::HwStage::Hull &&
                      runtime.sw_stage == Shader::SwStage::TessellationControl;
    const bool evaluation_export = runtime.hw_stage == Shader::HwStage::Export &&
                                   runtime.sw_stage == Shader::SwStage::TessellationEval;
    const bool evaluation_raster = runtime.hw_stage == Shader::HwStage::Vertex &&
                                   runtime.sw_stage == Shader::SwStage::TessellationEval;
    if ((!local && !hull && !evaluation_export && !evaluation_raster) ||
        config.raster_output != evaluation_raster ||
        (evaluation_raster && !config.compact_evaluation) ||
        (config.compact_evaluation && !evaluation_export && !evaluation_raster) ||
        runtime.props.num_user_data > Shader::NUM_USER_DATA_REGS ||
        !profile.needs_memory_predication || !profile.needs_vector_register_predication ||
        !profile.needs_storage_lds_barriers || !profile.lower_shared_memory_after_wave64 ||
        !config.workgroups || config.threads_per_group < 64 || config.threads_per_group > 256 ||
        config.threads_per_group % 64 ||
        ((evaluation_export || evaluation_raster) && config.threads_per_group != 64) ||
        !config.lds_bytes || config.lds_bytes % 8 || config.lds_bytes > 65536 ||
        config.InputSizeBytes() / sizeof(u32) > std::numeric_limits<u32>::max() ||
        config.LdsSizeBytes() > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Invalid Metal tessellation kernel configuration");
    }
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    if ((local && runtime.hw.ls.ls_stride != constants.ls_stride) ||
        (hull && (runtime.sw.tcs.ls_stride != constants.ls_stride ||
                  runtime.sw.tcs.hs_output_cp_stride != constants.hs_cp_stride ||
                  runtime.sw.tcs.hs_output_base != constants.hs_output_base)) ||
        ((evaluation_export || evaluation_raster) &&
         runtime.sw.tes.hs_output_cp_stride != constants.hs_cp_stride)) {
        throw std::runtime_error("Metal tessellation runtime does not match its constant snapshot");
    }
    if (evaluation_raster) {
        raster.emplace(runtime, profile);
    }
}

Shader::TranslationHooks TessellationKernelTranslation::Hooks() {
    Shader::TranslationHooks hooks;
    hooks.prologue = [this](IR::Program& program, IR::IREmitter& ir) {
        if (program.info.hw_stage != runtime.hw_stage ||
            program.info.sw_stage != runtime.sw_stage) {
            throw std::runtime_error("Metal tessellation kernel stage mismatch");
        }
        for (u32 scalar = 0; scalar < runtime.props.num_user_data; ++scalar) {
            const auto reg = IR::ScalarReg(scalar);
            ir.SetScalarReg(reg, ir.GetUserData(reg));
        }
        for (u32 scalar = 0; scalar < GeometryExportInput{}.system_sgprs.size(); ++scalar) {
            ir.SetScalarReg(IR::ScalarReg(runtime.props.num_user_data + scalar),
                            ir.GetAttributeU32(IR::Attribute::GeometryExportScalar, scalar));
        }
        for (u32 vector = 0; vector < GeometryExportInput{}.vgprs.size(); ++vector) {
            ir.SetVectorReg(IR::VectorReg(vector),
                            ir.GetAttributeU32(IR::Attribute::GeometryExportVector, vector));
        }
        const auto active =
            ir.IEqual(ir.GetAttributeU32(IR::Attribute::GeometryInputActive), ir.Imm32(1u));
        ir.SetExec(ir.Imm1(true));
        for (auto* block : program.blocks) {
            bool constrained = block != program.blocks.front();
            for (IR::Inst& inst : *block) {
                if (&inst == active.Inst()) {
                    constrained = true;
                }
                if (constrained && inst.GetOpcode() == IR::Opcode::SetExec) {
                    IR::IREmitter mask{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                    inst.SetArg(0, mask.LogicalAnd(active, IR::U1{inst.Arg(0)}));
                }
            }
        }
        program.info.sw_stage = Shader::SwStage::Compute;
    };
    hooks.lower_rings = [this](IR::Program& program, Shader::RuntimeInfo& translated_runtime) {
        if (raster) {
            raster->Collect(program);
        } else {
            for (auto* block : program.post_order_blocks) {
                for (const IR::Inst& inst : *block) {
                    if (inst.GetOpcode() == IR::Opcode::SetAttribute) {
                        throw std::runtime_error(
                            "Metal LS/HS/TES export kernels cannot export raster attributes");
                    }
                }
            }
        }
        program.info.hw_stage = Shader::HwStage::Compute;
        translated_runtime.hw_stage = Shader::HwStage::Compute;
        translated_runtime.sw_stage = Shader::SwStage::Compute;
        translated_runtime.hw.cs = {0, {config.threads_per_group, 1, 1}, {false, false, false}};
        return true;
    };
    hooks.lower_resources = [this](IR::Program& program) {
        auto& registers = bindings.registers;
        registers.input_sources = InspectGeometryInputSources(program);
        registers.address_program = std::make_shared<const GeometryBufferAddressProgram>(
            InspectGeometryBufferAddressProgram(program));
        registers.address_origins = std::make_shared<const GeometryBufferAddressOrigins>(
            InspectGeometryBufferAddressOrigins(program));
        scalar_constants.Lower(program);
        if (program.blocks.empty() || program.info.buffers.size() + 2 + bool(raster) +
                                              bool(raster && runtime.hw.vs.user_clip_plane_mask) >
                                          Shader::NUM_BUFFERS) {
            throw std::runtime_error("Metal tessellation kernel exceeds buffer capacity");
        }
        registers.inputs = u32(program.info.buffers.size());
        auto& input = program.info.buffers.emplace_back();
        input.buffer_type = Shader::BufferType::GeometryInputs;
        input.used_types = IR::Type::U32;
        auto* entry = program.blocks.front();
        IR::IREmitter initialize{*entry, entry->begin()};
        const auto work_item = initialize.IAdd(
            initialize.IMul(initialize.GetAttributeU32(IR::Attribute::WorkgroupId, 0),
                            initialize.Imm32(config.threads_per_group)),
            initialize.GetAttributeU32(IR::Attribute::LocalInvocationId, 0));
        const auto input_base = initialize.IMul(
            work_item, initialize.Imm32(u32{sizeof(GeometryExportInput) / sizeof(u32)}));
        if (raster) {
            raster->Lower(program, IR::U32{work_item}, IR::U32{input_base}, registers.inputs,
                          u32((config.InputSizeBytes() - 4 * sizeof(float)) / sizeof(u32)));
            bindings.raster = raster->Bindings();
        }
        for (auto* block : program.blocks) {
            for (IR::Inst& inst : *block) {
                if (inst.GetOpcode() != IR::Opcode::GetAttributeU32) {
                    continue;
                }
                const auto attribute = inst.Arg(0).Attribute();
                u32 word;
                if (attribute == IR::Attribute::GeometryInputActive) {
                    word = offsetof(GeometryExportInput, active) / sizeof(u32);
                } else if (attribute == IR::Attribute::GeometryExportScalar) {
                    word = inst.Arg(1).U32();
                } else if (attribute == IR::Attribute::GeometryExportVector) {
                    word = 16 + inst.Arg(1).U32();
                } else {
                    continue;
                }
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                inst.ReplaceUsesWithAndRemove(ir.LoadBufferU32(
                    1, ir.Imm32(registers.inputs), ir.IAdd(input_base, ir.Imm32(word)), {}));
            }
        }
    };
    hooks.structured = [this](IR::Program& program) {
        LowerWaveRegisters(program);
        if (raster) {
            raster->Finish(program);
        }
        Shader::Optimization::SsaRepairPass(program);
        Shader::Optimization::SsaRewritePass(program);
        Shader::Optimization::DeadCodeEliminationPass(program);
        bindings.scalar_reads = scalar_constants.LiveReads();
        auto storage_profile = profile;
        storage_profile.max_shared_memory_size = 0;
        auto storage_runtime = runtime;
        storage_runtime.hw_stage = Shader::HwStage::Compute;
        storage_runtime.sw_stage = Shader::SwStage::Compute;
        storage_runtime.hw.cs = {
            config.lds_bytes, {config.threads_per_group, 1, 1}, {false, false, false}};
        control_flow.CollectBarriers(program);
        const bool scheduled = SynchronizeTessellationLds(program);
        Shader::Optimization::SharedMemorySimplifyPass(program, storage_profile);
        const auto buffer_count = program.info.buffers.size();
        Shader::Optimization::SharedMemoryToStoragePass(program, storage_runtime, storage_profile);
        if (program.info.buffers.size() != buffer_count) {
            bindings.lds = u32(buffer_count);
        }
        if (scheduled) {
            control_flow.Lower(program, config.threads_per_group);
        }
        LowerTessellationWaves(program, config.threads_per_group);
    };
    return hooks;
}

void TessellationKernelTranslation::LowerEvaluationGroups(IR::Program& program) const {
    if (config.compact_evaluation) {
        LowerGeometryWorkgroupMap(program, bindings.registers.inputs,
                                  u32(config.RegisterSizeBytes() / sizeof(u32)));
    }
}

} // namespace Metal
