// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include "shader_recompiler/ir/ir_emitter.h"
#include "video_core/renderer_metal/metal_tessellation_scalar.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

u32 Buffer(Shader::Info& info, Shader::BufferType type, IR::Type words) {
    for (u32 index = 0; index < info.buffers.size(); ++index) {
        auto& buffer = info.buffers[index];
        if (buffer.buffer_type == type) {
            buffer.used_types |= words;
            return index;
        }
    }
    if (info.buffers.size() == Shader::NUM_BUFFERS) {
        throw std::runtime_error("Metal scalar reads exceed shader buffer capacity");
    }
    const u32 index = u32(info.buffers.size());
    info.buffers.push_back({.used_types = words, .buffer_type = type});
    return index;
}

} // namespace

void TessellationScalarReads::Prepare(
    IR::Program& program, const Shader::Optimization::ResourceDiscoveryList& resources) {
    reads.clear();
    locations.clear();
    predicates.clear();
    addresses = {};
    std::unordered_set<const IR::Inst*> descriptors;
    std::vector<IR::Value> pending;
    for (const auto& resource : resources) {
        for (const auto& sharp : resource.sharps) {
            for (u32 word = 0; word < sharp.num_dwords; ++word) {
                pending.push_back(sharp.dwords[word]);
            }
        }
    }
    while (!pending.empty()) {
        const auto value = pending.back();
        pending.pop_back();
        const auto* inst = value.TryInst();
        if (!inst || !descriptors.insert(inst).second) {
            continue;
        }
        for (u32 arg = 0; arg < inst->NumArgs(); ++arg) {
            pending.push_back(inst->Arg(arg));
        }
    }
    std::vector<IR::Inst*> constants;
    for (auto* block : program.blocks) {
        for (auto& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::ReadConst && !descriptors.contains(&inst)) {
                constants.push_back(&inst);
            }
        }
    }
    if (constants.empty()) {
        return;
    }
    auto& info = program.info;
    const u32 flatbuf = Buffer(info, Shader::BufferType::Flatbuf, IR::Type::U32);
    info.has_flatbuf = true;
    struct Lowering {
        IR::Inst* inst;
        IR::U32 index;
        IR::Value address;
    };
    std::vector<Lowering> lowerings;
    for (auto* inst : constants) {
        IR::IREmitter ir{*inst->GetParent(), IR::Block::InstructionList::s_iterator_to(*inst)};
        const u32 word = inst->Flags<u32>();
        IR::Value address;
        auto index = ir.Imm32(word);
        if (word) {
            throw std::runtime_error("Metal scalar reads were prepared after descriptor analysis");
        }
        const auto base = ir.PackUint2x32(inst->Arg(0));
        const auto bytes =
            ir.ShiftLeftLogical(ir.UConvert(64, IR::U32{inst->Arg(1)}), ir.Imm32(2u));
        const IR::U64 location{ir.IAdd(base, bytes)};
        const auto valid = ir.LogicalAnd(
            ir.LogicalAnd(ir.ILessThanEqual(base, ir.Imm64((u64{1} << 40) - 4), false),
                          ir.IGreaterThanEqual(location, base, false)),
            ir.LogicalAnd(ir.ILessThanEqual(location, ir.Imm64((u64{1} << 40) - 4), false),
                          ir.IEqual(ir.BitwiseAnd(location, ir.Imm64(u64{3})), ir.Imm64(u64{0}))));
        index = ir.GuardBufferAddress(ir.Imm32(0u), valid);
        address = location;
        locations.push_back(location);
        predicates.push_back(valid);
        reads.push_back(inst);
        (void)Buffer(info, Shader::BufferType::GuestPageTable, IR::Type::U64);
        info.uses_guest_page_reads = true;
        lowerings.push_back({inst, index, address});
    }
    for (const auto& lowering : lowerings) {
        auto* inst = lowering.inst;
        IR::IREmitter ir{*inst->GetParent(), IR::Block::InstructionList::s_iterator_to(*inst)};
        inst->ClearArgs();
        inst->ReplaceOpcode(IR::Opcode::ReadConstBuffer);
        inst->SetArg(0, ir.Imm32(flatbuf));
        inst->SetArg(1, lowering.index);
        inst->SetArg(2, lowering.address);
        inst->SetFlags(u64{0});
    }
}

void TessellationScalarReads::Lower(IR::Program& program) {
    addresses = InspectScalarConstantAddressProgram(locations, predicates);
    for (auto* block : program.blocks) {
        for (auto& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::ReadConst) {
                continue;
            }
            const u32 word = inst.Flags<u32>();
            if (!word || word >= program.info.flattened_ud_buf.size()) {
                throw std::runtime_error("Metal scalar descriptor has no flattened snapshot");
            }
            const u32 flatbuf = Buffer(program.info, Shader::BufferType::Flatbuf, IR::Type::U32);
            program.info.has_flatbuf = true;
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            inst.ClearArgs();
            inst.ReplaceOpcode(IR::Opcode::ReadConstBuffer);
            inst.SetArg(0, ir.Imm32(flatbuf));
            inst.SetArg(1, ir.Imm32(word));
            inst.SetFlags(u64{0});
        }
    }
}

std::shared_ptr<const GeometryBufferAddressProgram> TessellationScalarReads::LiveReads() const {
    std::vector<u32> indices;
    for (u32 index = 0; index < reads.size(); ++index) {
        if (reads[index]->GetOpcode() == IR::Opcode::ReadConstBuffer) {
            indices.push_back(index);
        }
    }
    if (indices.empty()) {
        return {};
    }
    return std::make_shared<const GeometryBufferAddressProgram>(addresses.Select(indices));
}

std::vector<GeometryBufferByteInterval> ScalarReadRanges(
    const GeometryBufferAddressProgram& program, std::span<const GeometryAddressInput> inputs,
    std::span<const u32> user_data, u64 byte_limit) {
    std::vector<GeometryBufferByteInterval> ranges;
    for (const auto& access : program.Evaluate(inputs, user_data)) {
        if (!access.reachable) {
            continue;
        }
        if (!access.bytes || !access.read || access.write || !access.bytes->lower ||
            access.bytes->upper < access.bytes->lower || access.bytes->upper >= (u64{1} << 40) ||
            access.bytes->upper - access.bytes->lower + 1 > byte_limit) {
            throw std::runtime_error("Metal scalar read requires a bounded native guest range");
        }
        ranges.push_back(*access.bytes);
    }
    std::ranges::sort(ranges, {}, &GeometryBufferByteInterval::lower);
    std::vector<GeometryBufferByteInterval> merged;
    u64 total{};
    for (const auto range : ranges) {
        if (!merged.empty() && range.lower <= merged.back().upper + 1) {
            total -= merged.back().upper - merged.back().lower + 1;
            merged.back().upper = std::max(merged.back().upper, range.upper);
        } else {
            merged.push_back(range);
        }
        const u64 size = merged.back().upper - merged.back().lower + 1;
        if (size > byte_limit - total) {
            throw std::runtime_error("Metal scalar read ranges exceed the native storage budget");
        }
        total += size;
    }
    return merged;
}

} // namespace Metal
