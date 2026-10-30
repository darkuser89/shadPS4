// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <stdexcept>
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "video_core/renderer_metal/metal_tessellation_kernel.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;
IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

void EmitControlFlowGraph(IR::Program& program, Shader::Pools& pools, Shader::Gcn::CFG& cfg,
                          Shader::RuntimeInfo& runtime_info, const Shader::Profile& profile,
                          const Shader::TranslationHooks& hooks) {
    Shader::Gcn::Translator translator{program.info, runtime_info, profile};
    for (auto& block : cfg) {
        const u32 start = block.begin_index;
        const u32 size = block.end_index - start + 1;
        auto* ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(start, size));
        program.blocks.push_back(ir_block);
    }
    auto* entry = program.blocks.front();
    const auto body = entry->begin();
    translator.EmitPrologue(entry);
    IR::IREmitter ir{*entry, body};
    hooks.prologue(program, ir);
    if (program.info.translation_failed) {
        throw std::runtime_error("Native Metal tessellation translation failed");
    }
    for (auto& block : cfg) {
        auto* ir_block = block.ir_block;
        if (block.branch_true) {
            auto* true_block = block.branch_true->ir_block;
            ir_block->AddBranch(true_block);
        }
        if (block.branch_false) {
            auto* false_block = block.branch_false->ir_block;
            ir_block->AddBranch(false_block);
        }
    }
    program.post_order_blocks = Shader::IR::PostOrder(program.blocks.front());
}

} // namespace

Shader::IR::Program TessellationKernelTranslation::Translate(
    std::span<const u32> code, Shader::Pools& pools, Shader::Info& info,
    Shader::RuntimeInfo& translated_runtime, const Shader::Profile& translation_profile,
    const Shader::TranslationHooks& hooks, void (*lower_resources)(Shader::IR::Program&),
    bool (*patch_buffer)(Shader::IR::Inst&, Shader::Info&),
    Shader::ImageReadConverter convert_image_read,
    Shader::ImageSampleConverter convert_image_sample) {
    if (code.empty() || !hooks.prologue || !hooks.lower_rings || !hooks.lower_resources ||
        !hooks.structured || !info.prepare_memory_read || translation_profile.subgroup_size != 64) {
        throw std::runtime_error("Incomplete native Metal tessellation translation contract");
    }
    Shader::Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Shader::Gcn::GcnDecodeContext decoder;

    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    pools.ReleaseContents();

    Common::ObjectPool<Shader::Gcn::Block> gcn_block_pool{64};
    Shader::Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    EmitControlFlowGraph(program, pools, cfg, translated_runtime, translation_profile, hooks);

    Shader::InjectClipDistanceAttributes(program, translated_runtime);

    if (!translation_profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::ReadLaneEliminationPass(program);
    if (info.sw_stage == Shader::SwStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, translated_runtime);
        Shader::Optimization::HullShaderTransform(program, translated_runtime);
    } else if (info.sw_stage == Shader::SwStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, translated_runtime);
        Shader::Optimization::DomainShaderTransform(program, translated_runtime);
    }
    if (!hooks.lower_rings(program, translated_runtime)) {
        Shader::Optimization::RingAccessElimination(program, translated_runtime);
    }
    auto resources = Shader::Optimization::ResourceDiscoverPass(program, translation_profile);
    scalar_constants.Prepare(program, resources);
    Shader::Optimization::FlattenExtendedUserdataPass(program, resources);
    Shader::Optimization::ResourcePatchingPass(program.info, resources, translation_profile,
                                               patch_buffer, convert_image_read,
                                               convert_image_sample);
    if (lower_resources) {
        lower_resources(program);
    }
    hooks.lower_resources(program);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, translation_profile);
    if (!translation_profile.lower_shared_memory_after_wave64) {
        Shader::Optimization::SharedMemoryToStoragePass(program, translated_runtime,
                                                        translation_profile);
    }
    Shader::Optimization::LowerUserClipPlanes(program, translated_runtime);
    Shader::Optimization::PhiSimplificationPass(program);
    Shader::Optimization::InverseBallotEliminationPass(program);

    for (auto* ir_block : program.blocks) {
        ir_block->imm_predecessors.clear();
        ir_block->imm_successors.clear();
        ir_block->ssa_state.Reset();
    }
    Shader::Optimization::LowerPhisToRegsPass(program);

    program.syntax_list = Shader::Gcn::BuildASL(pools, cfg, info);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front().data.block);
    hooks.structured(program);

    Shader::Optimization::SsaRepairPass(program);
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::SharedMemoryBarrierPass(program, translated_runtime, translation_profile);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::LowerWave64BallotPass(program, translated_runtime, translation_profile);
    if (translation_profile.lower_shared_memory_after_wave64) {
        Shader::Optimization::SharedMemoryToStoragePass(program, translated_runtime,
                                                        translation_profile);
    }
    Shader::Optimization::LowerHardwareIntrinsics(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, translation_profile);

    return program;
}

} // namespace Metal
