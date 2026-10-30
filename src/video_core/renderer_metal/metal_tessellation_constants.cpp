// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_metal/metal_tessellation_constants.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

std::optional<u32> UserRegister(IR::Value value) {
    const auto* inst = value.TryInst();
    if (!inst || inst->GetOpcode() != IR::Opcode::GetUserData) {
        return std::nullopt;
    }
    return u32(inst->Arg(0).ScalarReg());
}

TessellationConstantBinding InspectDescriptor(IR::Value value) {
    const auto* descriptor = value.TryInst();
    if (!descriptor || descriptor->GetOpcode() != IR::Opcode::CompositeConstructU32x4) {
        throw std::runtime_error("Metal tessellation constants require a complete descriptor");
    }
    std::optional<TessellationConstantBinding> binding;
    for (u32 word = 0; word < 4; ++word) {
        const auto source = descriptor->Arg(word);
        TessellationConstantBinding current{};
        if (const auto reg = UserRegister(source)) {
            current = {IR::NumScalarRegs, *reg};
        } else {
            const auto* read = source.TryInst();
            if (!read || read->GetOpcode() != IR::Opcode::ReadConst ||
                !read->Arg(1).IsImmediate() || read->Arg(1).Type() != IR::Type::U32) {
                throw std::runtime_error("Metal tessellation descriptor location is dynamic");
            }
            const auto* pointer = read->Arg(0).TryInst();
            if (!pointer || pointer->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
                throw std::runtime_error("Metal tessellation descriptor pointer is unsupported");
            }
            const auto low = UserRegister(pointer->Arg(0));
            const auto high = UserRegister(pointer->Arg(1));
            if (!low || !high || *low >= Shader::NUM_USER_DATA_REGS - 1 || *high != *low + 1) {
                throw std::runtime_error("Metal tessellation descriptor pointer is not contiguous");
            }
            current = {*low, read->Arg(1).U32()};
        }
        if (!binding) {
            binding = current;
        } else if (current.pointer_register != binding->pointer_register ||
                   u64{current.dword_offset} != u64{binding->dword_offset} + word) {
            throw std::runtime_error("Metal tessellation descriptor words are not contiguous");
        }
    }
    return *binding;
}

TessellationConstantBinding InspectAddresses(const IR::Program& program) {
    std::optional<TessellationConstantBinding> binding;
    std::unordered_set<const IR::Inst*> visited;
    std::vector<IR::Value> pending;
    for (const auto* block : program.blocks) {
        for (const auto& inst : *block) {
            if (Shader::Optimization::IsSharedMemoryInstruction(inst)) {
                pending.push_back(inst.Arg(0));
            }
        }
    }
    while (!pending.empty()) {
        const auto value = pending.back();
        pending.pop_back();
        const auto* inst = value.TryInst();
        if (!inst || !visited.insert(inst).second) {
            continue;
        }
        if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
            if (!inst->Arg(1).IsImmediate() || inst->Arg(1).Type() != IR::Type::U32 ||
                inst->Arg(1).U32() >=
                    sizeof(Shader::TessellationDataConstantBuffer) / sizeof(u32)) {
                throw std::runtime_error("Metal tessellation constant field is unsupported");
            }
            const auto candidate = InspectDescriptor(inst->Arg(0));
            if (binding && *binding != candidate) {
                throw std::runtime_error("Metal tessellation constant binding is ambiguous");
            }
            binding = candidate;
            continue;
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            pending.push_back(inst->Arg(arg));
        }
    }
    if (!binding) {
        throw std::runtime_error("Metal tessellation constant binding was not found");
    }
    return *binding;
}

} // namespace

TessellationConstantBinding InspectTessellationConstantBinding(std::span<const u32> code,
                                                               const Shader::RuntimeInfo& runtime,
                                                               const Shader::Profile& profile) {
    if (code.empty() || runtime.hw_stage != Shader::HwStage::Hull ||
        runtime.sw_stage != Shader::SwStage::TessellationControl ||
        runtime.props.num_user_data > Shader::NUM_USER_DATA_REGS) {
        throw std::runtime_error("Metal tessellation binding analysis requires a hull shader");
    }
    Shader::Pools pools;
    const std::array<u32, Shader::NUM_USER_DATA_REGS> user_data{};
    Shader::Info info{runtime.hw_stage, runtime.sw_stage, {user_data, code, 0}};
    IR::Program program{info};
    Shader::Gcn::GcnCodeSlice slice{code.data(), code.data() + code.size()};
    Shader::Gcn::GcnDecodeContext decoder;
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
        if (program.ins_list.back().opcode == Shader::Gcn::Opcode::S_SWAPPC_B64) {
            throw std::runtime_error("Metal hull constant analysis does not support shader calls");
        }
    }
    Common::ObjectPool<Shader::Gcn::Block> block_pool{64};
    Shader::Gcn::CFG cfg{block_pool, program.ins_list};
    Shader::Gcn::Translator translator{info, runtime, profile};
    for (auto& block : cfg) {
        auto* ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(
                                 block.begin_index, block.end_index - block.begin_index + 1));
        program.blocks.push_back(ir_block);
    }
    if (program.blocks.empty() || info.translation_failed) {
        throw std::runtime_error(
            "Metal tessellation binding analysis could not translate the shader");
    }
    translator.EmitPrologue(program.blocks.front());
    for (const auto& block : cfg) {
        if (block.branch_true) {
            block.ir_block->AddBranch(block.branch_true->ir_block);
        }
        if (block.branch_false) {
            block.ir_block->AddBranch(block.branch_false->ir_block);
        }
    }
    program.post_order_blocks = IR::PostOrder(program.blocks.front());
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::ReadLaneEliminationPass(program);
    return InspectAddresses(program);
}

Shader::TessellationDataConstantBuffer ReadTessellationConstants(
    const TessellationConstantBinding& binding, std::span<const u32> user_data,
    const std::function<void(VAddr, u64)>& prepare_memory_read) {
    constexpr u64 AddressLimit = u64{1} << 40;
    const auto read = [&](VAddr address, void* target, u64 size) {
        if (!prepare_memory_read || !address || address % sizeof(u32) || address >= AddressLimit ||
            size > AddressLimit - address) {
            throw std::runtime_error("Invalid Metal tessellation constant memory range");
        }
        prepare_memory_read(address, size);
        std::memcpy(target, reinterpret_cast<const void*>(address), size);
    };
    AmdGpu::Buffer descriptor{};
    if (binding.pointer_register == IR::NumScalarRegs) {
        if (binding.dword_offset > user_data.size() ||
            user_data.size() - binding.dword_offset < sizeof(descriptor) / sizeof(u32)) {
            throw std::runtime_error("Metal tessellation descriptor exceeds user data");
        }
        std::memcpy(&descriptor, user_data.data() + binding.dword_offset, sizeof(descriptor));
    } else {
        if (binding.pointer_register >= user_data.size() ||
            user_data.size() - binding.pointer_register < 2) {
            throw std::runtime_error("Metal tessellation pointer exceeds user data");
        }
        VAddr pointer;
        std::memcpy(&pointer, user_data.data() + binding.pointer_register, sizeof(pointer));
        pointer &= 0xFFFFFFFFFFFFULL;
        const u64 offset = u64{binding.dword_offset} * sizeof(u32);
        if (!pointer || pointer >= AddressLimit || offset >= AddressLimit - pointer) {
            throw std::runtime_error("Invalid Metal tessellation descriptor range");
        }
        read(pointer + offset, &descriptor, sizeof(descriptor));
    }
    Shader::TessellationDataConstantBuffer constants;
    static_assert(sizeof(constants) == 9 * sizeof(u32));
    const u64 descriptor_size = u64{descriptor.num_records} * std::max<u64>(descriptor.stride, 1);
    if (!descriptor.Valid() || descriptor.swizzle_enable || descriptor.add_tid_enable ||
        descriptor_size < sizeof(constants)) {
        throw std::runtime_error("Unsupported Metal tessellation constant descriptor");
    }
    read(descriptor.base_address, &constants, sizeof(constants));
    return constants;
}

} // namespace Metal
