// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "video_core/renderer_metal/metal_geometry_output.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

bool ValidAttribute(const GeometryOutputAttribute& output) {
    if (output.attribute == IR::Attribute::Position0 || IR::IsParam(output.attribute)) {
        return output.component < 4;
    }
    if (output.attribute == IR::Attribute::ClipDistance ||
        output.attribute == IR::Attribute::CullDistance) {
        return output.component < 8;
    }
    return output.component == 0 && (output.attribute == IR::Attribute::PointSize ||
                                     output.attribute == IR::Attribute::RenderTargetIndex ||
                                     output.attribute == IR::Attribute::ViewportIndex);
}
} // namespace

void GeometryClipPlanes::Collect(std::map<std::pair<IR::Attribute, u32>, u32>& outputs, u32 mask) {
    if (mask & ~0x3fu) {
        throw std::runtime_error("Invalid native geometry user clip plane mask");
    }
    if (!mask) {
        return;
    }
    for (u32 component = 0; component < 4; ++component) {
        if (!outputs.contains({IR::Attribute::Position0, component})) {
            throw std::runtime_error("Native user clip planes require complete position exports");
        }
    }
    u32 count{};
    for (const auto& [output, _] : outputs) {
        if (output.first == IR::Attribute::ClipDistance) {
            if (output.second >= 8) {
                throw std::runtime_error("Invalid native guest clip distance component");
            }
            count = std::max(count, output.second + 1);
        }
    }
    for (u32 plane = 0; plane < slots.size(); ++plane) {
        if (mask & (1u << plane)) {
            slots[plane] = count;
            outputs.try_emplace({IR::Attribute::ClipDistance, count++}, 0);
        }
    }
}

void GeometryClipPlanes::Bind(IR::Program& program) {
    if (std::ranges::none_of(slots, [](const auto& slot) { return slot.has_value(); })) {
        return;
    }
    if (program.info.buffers.size() >= Shader::NUM_BUFFERS) {
        throw std::runtime_error("Native geometry clip plane buffer capacity exceeded");
    }
    buffer = u32(program.info.buffers.size());
    auto& resource = program.info.buffers.emplace_back();
    resource.buffer_type = Shader::BufferType::ClipPlanes;
    resource.used_types = IR::Type::F32;
}

void GeometryClipPlanes::Evaluate(IR::IREmitter& ir,
                                  std::span<const GeometryOutputAttribute> attributes,
                                  std::span<IR::U32> values) const {
    const auto find = [&](IR::Attribute attribute, u32 component) {
        const auto output = std::ranges::find_if(attributes, [&](const auto& candidate) {
            return candidate.attribute == attribute && candidate.component == component;
        });
        if (output == attributes.end() || values.size() != attributes.size()) {
            throw std::runtime_error("Native geometry clip plane output mapping is inconsistent");
        }
        return size_t(output - attributes.begin());
    };
    for (u32 plane = 0; plane < slots.size(); ++plane) {
        if (!slots[plane]) {
            continue;
        }
        if (!buffer) {
            throw std::runtime_error("Native geometry clip plane uniform is missing");
        }
        const auto equation = ir.LoadBufferF32(4, ir.Imm32(*buffer), ir.Imm32(plane * 4), {});
        auto distance =
            IR::F32{ir.FPMul(ir.BitCast<IR::F32>(values[find(IR::Attribute::Position0, 0)]),
                             IR::F32{ir.CompositeExtract(equation, 0)})};
        for (u32 component = 1; component < 4; ++component) {
            distance = IR::F32{
                ir.FPFma(ir.BitCast<IR::F32>(values[find(IR::Attribute::Position0, component)]),
                         IR::F32{ir.CompositeExtract(equation, component)}, distance)};
        }
        values[find(IR::Attribute::ClipDistance, *slots[plane])] = ir.BitCast<IR::U32>(distance);
    }
}

void LowerGeometryWorkgroupMap(IR::Program& program, u32 input_buffer, u32 mapping_word) {
    std::vector<IR::Inst*> groups;
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::GetAttributeU32) {
                continue;
            }
            const auto attribute = inst.Arg(0).Attribute();
            if (attribute == IR::Attribute::WorkgroupIndex ||
                (attribute == IR::Attribute::WorkgroupId && inst.Arg(1).U32() == 0)) {
                groups.push_back(&inst);
            }
        }
    }
    if (groups.empty()) {
        return;
    }
    auto* entry = program.blocks.front();
    const auto prologue = std::ranges::find_if(
        *entry, [](const IR::Inst& inst) { return inst.GetOpcode() == IR::Opcode::Prologue; });
    if (prologue == entry->end()) {
        throw std::runtime_error("Metal geometry group mapping requires an entry prologue");
    }
    auto insertion = entry->begin();
    while (insertion != entry->end() && insertion->GetOpcode() == IR::Opcode::Phi) {
        ++insertion;
    }
    if (insertion != prologue) {
        entry->Instructions().splice(insertion, entry->Instructions(), prologue);
    }
    IR::IREmitter ir{*entry, std::next(prologue)};
    program.info.loads.Set(IR::Attribute::WorkgroupId, 0);
    const auto index =
        ir.IAdd(ir.GetAttributeU32(IR::Attribute::WorkgroupId, 0), ir.Imm32(mapping_word));
    const auto original = ir.LoadBufferU32(1, ir.Imm32(input_buffer), index, {});
    for (auto* inst : groups) {
        inst->ReplaceUsesWithAndRemove(original);
    }
    program.info.loads.flags[Shader::Info::AttributeFlags::Index(IR::Attribute::WorkgroupIndex)] =
        0;
}

u32 GeometryInitialActiveCount(u32 work_items, u32 requested) {
    if (!work_items || work_items % 64 ||
        (requested && (requested > work_items || work_items - requested >= 64))) {
        throw std::runtime_error("Invalid native geometry initial active count");
    }
    return requested ? requested : work_items;
}

void InitializeGeometryExec(IR::Program& program, IR::IREmitter& ir,
                            IR::Attribute active_attribute) {
    const IR::U1 active{ir.IEqual(ir.GetAttributeU32(active_attribute), ir.Imm32(1u))};
    const auto work_item =
        ir.IAdd(ir.IMul(ir.GetAttributeU32(IR::Attribute::WorkgroupId, 0), ir.Imm32(64u)),
                ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0));
    ir.SetExec(ir.ILessThan(work_item,
                            ir.GetAttributeU32(IR::Attribute::GeometryInitialActiveCount), false));
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
}

void LowerGeometryWaveOperations(IR::Program& program) {
    if (program.info.hw_stage != Shader::HwStage::Compute) {
        throw std::runtime_error("Native geometry wave operations require compute lowering");
    }
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const auto opcode = inst.GetOpcode();
            if (opcode != IR::Opcode::InverseBallot && opcode != IR::Opcode::WriteLane) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const auto lane = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
            if (opcode == IR::Opcode::WriteLane) {
                const auto target = ir.BitwiseAnd(IR::U32{inst.Arg(2)}, ir.Imm32(63u));
                inst.ReplaceUsesWithAndRemove(
                    ir.Select(ir.IEqual(lane, target), inst.Arg(1), inst.Arg(0)));
                continue;
            }
            const auto bit = ir.BitwiseAnd(
                IR::U64{ir.ShiftRightLogical(IR::U64{inst.Arg(0)}, lane)}, ir.Imm64(u64{1}));
            inst.ReplaceUsesWithAndRemove(ir.INotEqual(bit, ir.Imm64(u64{0})));
        }
    }
}

void LowerGeometryOutputs(IR::Program& program, const GeometryAssemblyParameters& parameters,
                          std::span<const GeometryOutputAttribute> attributes, u32 output_buffer,
                          const IR::U32& bounded_invocation) {
    if (program.info.hw_stage != Shader::HwStage::Geometry || program.blocks.empty() ||
        output_buffer >= program.info.buffers.size() || !parameters.invocations ||
        !parameters.jobs || !program.blocks.front()->ImmPredecessors().empty() ||
        parameters.EmissionSizeBytes() / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1) {
        throw std::runtime_error("Invalid native geometry output context");
    }
    u32 active_streams{};
    for (u32 stream = 0; stream < 4; ++stream) {
        if (parameters.mesh_counts[stream]) {
            active_streams |= 1u << stream;
        }
    }
    if (parameters != GeometryAssemblyLayout(parameters.invocations, parameters.max_vertices,
                                             parameters.vertex_words, parameters.mesh_vertices,
                                             parameters.topology, active_streams)) {
        throw std::runtime_error("Invalid native geometry output layout");
    }
    for (u32 index = 0; index < attributes.size(); ++index) {
        const auto& output = attributes[index];
        if (output.stream >= 4 || !(active_streams & (1u << output.stream)) || !output.word ||
            output.word >= parameters.vertex_words || !ValidAttribute(output)) {
            throw std::runtime_error("Invalid native geometry output attribute");
        }
        for (u32 previous = 0; previous < index; ++previous) {
            const auto& other = attributes[previous];
            if (output.stream == other.stream &&
                (output.word == other.word ||
                 (output.attribute == other.attribute && output.component == other.component))) {
                throw std::runtime_error("Aliased native geometry output attribute");
            }
        }
    }
    const auto find_attribute = [&](const IR::Inst& inst) {
        return std::ranges::find_if(attributes, [&](const auto& output) {
            return output.attribute == inst.Arg(0).Attribute() &&
                   output.component == inst.Arg(2).U32() && output.stream == inst.Flags<u32>();
        });
    };
    auto* entry = program.blocks.front();
    auto initialization = entry->begin();
    bool found_invocation = bounded_invocation.IsImmediate();
    if ((found_invocation && bounded_invocation.U32() >= parameters.invocations) ||
        (!found_invocation && bounded_invocation.Inst()->GetOpcode() == IR::Opcode::Phi)) {
        throw std::runtime_error("Native geometry invocation exceeds output capacity");
    }
    for (auto* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            const auto opcode = it->GetOpcode();
            if (!bounded_invocation.IsImmediate() && &*it == bounded_invocation.Inst() &&
                block == entry) {
                found_invocation = true;
                initialization = std::next(it);
            }
            if (opcode != IR::Opcode::SetAttribute && opcode != IR::Opcode::EmitVertex &&
                opcode != IR::Opcode::EmitPrimitive && opcode != IR::Opcode::EmitVertexMasked &&
                opcode != IR::Opcode::EmitPrimitiveMasked) {
                continue;
            }
            if (!found_invocation || it->Flags<u32>() >= 4 ||
                !(active_streams & (1u << it->Flags<u32>()))) {
                throw std::runtime_error("Invalid native geometry output stream or invocation");
            }
            if (opcode == IR::Opcode::SetAttribute &&
                (!it->Arg(0).IsImmediate() || !it->Arg(2).IsImmediate() ||
                 find_attribute(*it) == attributes.end())) {
                throw std::runtime_error("Unmapped native geometry output attribute");
            }
        }
    }
    if (!found_invocation ||
        attributes.size() + 8 > std::numeric_limits<u32>::max() - program.next_reg_index) {
        throw std::runtime_error("Native geometry output register capacity exceeded");
    }

    std::array<IR::VirtualReg, 4> counts;
    std::array<IR::VirtualReg, 4> ordinals;
    std::vector<IR::VirtualReg> values;
    const auto new_register = [&] {
        return IR::VirtualReg{program.next_reg_index++, IR::Type::U32};
    };
    for (u32 stream = 0; stream < 4; ++stream) {
        counts[stream] = new_register();
        ordinals[stream] = new_register();
    }
    for (u32 index = 0; index < attributes.size(); ++index) {
        values.push_back(new_register());
    }
    const auto stream_word_offset = [&](IR::IREmitter& ir, u32 stream) {
        const IR::U32 slot{ir.IAdd(ir.IMul(bounded_invocation, ir.Imm32(4u)), ir.Imm32(stream))};
        return IR::U32{ir.IMul(slot, ir.Imm32(parameters.stream_stride_words))};
    };
    const auto store = [&](IR::IREmitter& ir, const IR::U32& word_offset, u32 word,
                           const IR::U32& value, const IR::U1& predicate) {
        const auto address =
            ir.GuardBufferAddress(IR::U32{ir.IAdd(word_offset, ir.Imm32(word))}, predicate);
        ir.StoreBufferU32(1, ir.Imm32(output_buffer), address, value, {});
    };
    IR::IREmitter initialize{*entry, initialization};
    for (u32 stream = 0; stream < 4; ++stream) {
        initialize.SetVirtualReg(counts[stream], initialize.Imm32(0u));
        initialize.SetVirtualReg(ordinals[stream], initialize.Imm32(0u));
        const auto word_offset = stream_word_offset(initialize, stream);
        for (u32 word = 0; word < 4; ++word) {
            store(initialize, word_offset, word, initialize.Imm32(0u), initialize.Imm1(true));
        }
    }
    for (const auto& value : values) {
        initialize.SetVirtualReg(value, initialize.Imm32(0u));
    }
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const auto opcode = inst.GetOpcode();
            if (opcode != IR::Opcode::SetAttribute && opcode != IR::Opcode::EmitVertex &&
                opcode != IR::Opcode::EmitPrimitive && opcode != IR::Opcode::EmitVertexMasked &&
                opcode != IR::Opcode::EmitPrimitiveMasked) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const u32 stream = inst.Flags<u32>();
            const IR::U1 predicate =
                opcode == IR::Opcode::EmitVertexMasked || opcode == IR::Opcode::EmitPrimitiveMasked
                    ? IR::U1{inst.Arg(0)}
                    : ir.Imm1(true);
            if (opcode == IR::Opcode::SetAttribute) {
                const auto output = find_attribute(inst);
                ir.SetVirtualReg(values[output - attributes.begin()],
                                 ir.BitCast<IR::U32>(IR::F32{inst.Arg(1)}));
            } else if (opcode == IR::Opcode::EmitPrimitive ||
                       opcode == IR::Opcode::EmitPrimitiveMasked) {
                ir.SetVirtualReg(ordinals[stream],
                                 ir.Select(predicate, ir.Imm32(0u),
                                           IR::U32{ir.GetVirtualReg(ordinals[stream])}));
            } else {
                const auto word_offset = stream_word_offset(ir, stream);
                const IR::U32 count{ir.GetVirtualReg(counts[stream])};
                const IR::U32 ordinal{ir.GetVirtualReg(ordinals[stream])};
                if (parameters.max_vertices) {
                    const auto vertex = ir.UMin(count, ir.Imm32(parameters.max_vertices - 1));
                    const IR::U32 row{ir.IAdd(
                        word_offset,
                        ir.IAdd(ir.Imm32(4u), ir.IMul(vertex, ir.Imm32(parameters.vertex_words))))};
                    store(ir, row, 0, ordinal, predicate);
                    for (u32 index = 0; index < attributes.size(); ++index) {
                        if (attributes[index].stream == stream) {
                            store(ir, row, attributes[index].word,
                                  IR::U32{ir.GetVirtualReg(values[index])}, predicate);
                        }
                    }
                }
                const auto increment = [&](const IR::U32& value) {
                    return ir.UMin(IR::U32{ir.IAdd(value, ir.Imm32(1u))},
                                   ir.Imm32(parameters.max_vertices + 1));
                };
                const IR::U32 next_count{ir.Select(predicate, increment(count), count)};
                ir.SetVirtualReg(counts[stream], next_count);
                ir.SetVirtualReg(ordinals[stream],
                                 ir.Select(predicate, increment(ordinal), ordinal));
                store(ir, word_offset, 0, next_count, predicate);
            }
            inst.Invalidate();
        }
    }
    auto& buffer = program.info.buffers[output_buffer];
    buffer.used_types |= IR::Type::U32;
    buffer.is_written = true;
}

} // namespace Metal
