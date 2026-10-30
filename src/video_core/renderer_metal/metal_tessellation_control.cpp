// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/resource_pass.h"
#include "shader_recompiler/ir/post_order.h"
#include "video_core/renderer_metal/metal_tessellation_control.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;
constexpr u32 Done = std::numeric_limits<u32>::max();
constexpr u32 Parked = 1u << 31;

struct Shape {
    IR::Type scalar;
    u32 count;
};

Shape ValueShape(IR::Type type) {
    switch (type) {
    case IR::Type::U1:
    case IR::Type::U8:
    case IR::Type::U16:
    case IR::Type::U32:
    case IR::Type::U64:
    case IR::Type::F16:
    case IR::Type::F32:
    case IR::Type::F64:
        return {type, 1};
    case IR::Type::U32x2:
        return {IR::Type::U32, 2};
    case IR::Type::U32x3:
        return {IR::Type::U32, 3};
    case IR::Type::U32x4:
        return {IR::Type::U32, 4};
    case IR::Type::F32x2:
        return {IR::Type::F32, 2};
    case IR::Type::F32x3:
        return {IR::Type::F32, 3};
    case IR::Type::F32x4:
        return {IR::Type::F32, 4};
    default:
        throw std::runtime_error("Metal tessellation control value requires numeric lowering");
    }
}

std::vector<IR::Value> Encode(IR::IREmitter& ir, const IR::Value& value) {
    const auto shape = ValueShape(value.Type());
    std::vector<IR::Value> words;
    for (u32 component = 0; component < shape.count; ++component) {
        const auto scalar = shape.count == 1 ? value : ir.CompositeExtract(value, component);
        switch (shape.scalar) {
        case IR::Type::U1:
        case IR::Type::U32:
            words.push_back(scalar);
            break;
        case IR::Type::U8:
        case IR::Type::U16:
            words.push_back(ir.UConvert(32, IR::U8U16U32U64{scalar}));
            break;
        case IR::Type::F16:
            words.push_back(ir.UConvert(32, ir.BitCast<IR::U16>(IR::F16{scalar})));
            break;
        case IR::Type::F32:
            words.push_back(ir.BitCast<IR::U32>(IR::F32{scalar}));
            break;
        case IR::Type::U64:
        case IR::Type::F64: {
            const auto halves = shape.scalar == IR::Type::U64
                                    ? ir.UnpackUint2x32(IR::U64{scalar})
                                    : ir.UnpackDouble2x32(IR::F64{scalar});
            words.push_back(ir.CompositeExtract(halves, 0));
            words.push_back(ir.CompositeExtract(halves, 1));
            break;
        }
        default:
            throw std::runtime_error("Invalid Metal tessellation scalar state");
        }
    }
    return words;
}

IR::Value Decode(IR::IREmitter& ir, IR::Type type, std::span<const IR::Value> words) {
    const auto shape = ValueShape(type);
    std::array<IR::Value, 4> components;
    u32 word{};
    for (u32 component = 0; component < shape.count; ++component) {
        switch (shape.scalar) {
        case IR::Type::U1:
        case IR::Type::U32:
            components[component] = words[word++];
            break;
        case IR::Type::U8:
        case IR::Type::U16:
            components[component] =
                ir.UConvert(shape.scalar == IR::Type::U8 ? 8 : 16, IR::U32{words[word++]});
            break;
        case IR::Type::F16:
            components[component] =
                ir.BitCast<IR::F16>(IR::U16{ir.UConvert(16, IR::U32{words[word++]})});
            break;
        case IR::Type::F32:
            components[component] = ir.BitCast<IR::F32>(IR::U32{words[word++]});
            break;
        case IR::Type::U64:
        case IR::Type::F64: {
            const auto halves = ir.CompositeConstruct(words[word], words[word + 1]);
            word += 2;
            components[component] = shape.scalar == IR::Type::U64
                                        ? IR::Value{ir.PackUint2x32(halves)}
                                        : IR::Value{ir.PackDouble2x32(halves)};
            break;
        }
        default:
            throw std::runtime_error("Invalid Metal tessellation decoded state");
        }
    }
    switch (shape.count) {
    case 1:
        return components[0];
    case 2:
        return ir.CompositeConstruct(components[0], components[1]);
    case 3:
        return ir.CompositeConstruct(components[0], components[1], components[2]);
    case 4:
        return ir.CompositeConstruct(components[0], components[1], components[2], components[3]);
    default:
        throw std::runtime_error("Invalid Metal tessellation state component count");
    }
}

struct State {
    IR::Type type;
    std::vector<IR::VirtualReg> words;

    State(IR::Program& program, IR::Type type_) : type{type_} {
        const auto shape = ValueShape(type);
        const u32 count =
            shape.count * (shape.scalar == IR::Type::U64 || shape.scalar == IR::Type::F64 ? 2 : 1);
        for (u32 index = 0; index < count; ++index) {
            words.emplace_back(program.next_reg_index++,
                               type == IR::Type::U1 ? IR::Type::U1 : IR::Type::U32);
        }
    }

    IR::Value Load(IR::IREmitter& ir) const {
        std::vector<IR::Value> values;
        for (const auto& reg : words) {
            values.push_back(ir.GetVirtualReg(reg));
        }
        return Decode(ir, type, values);
    }

    void Store(IR::IREmitter& ir, const IR::Value& value) const {
        const auto values = Encode(ir, value);
        for (u32 index = 0; index < words.size(); ++index) {
            ir.SetVirtualReg(words[index], values[index]);
        }
    }
};

struct Edge {
    IR::U1 condition{IR::Value{true}};
    IR::Block* yes{};
    IR::Block* no{};
};

std::unordered_map<IR::Block*, Edge> ControlEdges(const IR::Program& program) {
    std::unordered_map<IR::Block*, Edge> edges;
    IR::Block* current{};
    using Type = IR::AbstractSyntaxNode::Type;
    for (const auto& node : program.syntax_list) {
        switch (node.type) {
        case Type::Block:
            if (current) {
                edges.emplace(current, Edge{IR::U1{IR::Value{true}}, node.data.block});
            }
            current = node.data.block;
            continue;
        case Type::If:
            edges.emplace(current, Edge{node.data.if_node.cond, node.data.if_node.body,
                                        node.data.if_node.merge});
            break;
        case Type::Loop:
            edges.emplace(current, Edge{IR::U1{IR::Value{true}}, node.data.loop.body});
            break;
        case Type::Repeat:
            edges.emplace(current, Edge{node.data.repeat.cond, node.data.repeat.loop_header,
                                        node.data.repeat.merge});
            break;
        case Type::Break:
            edges.emplace(current, Edge{node.data.break_node.cond, node.data.break_node.merge,
                                        node.data.break_node.skip});
            break;
        case Type::EndIf:
            if (current) {
                edges.emplace(current, Edge{IR::U1{IR::Value{true}}, node.data.end_if.merge});
            }
            break;
        case Type::Return:
            edges.emplace(current, Edge{});
            break;
        case Type::Unreachable:
            if (current) {
                throw std::runtime_error("Metal tessellation has a reachable undefined exit");
            }
            break;
        }
        current = nullptr;
    }
    if (edges.contains(nullptr) || current || edges.size() != program.blocks.size()) {
        throw std::runtime_error("Invalid Metal tessellation control-flow edges");
    }
    return edges;
}

void LowerPhis(IR::Program& program) {
    std::vector<IR::Inst*> phis;
    for (auto* block : program.blocks) {
        for (auto& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::Phi) {
                phis.push_back(&inst);
            }
        }
    }
    for (auto* phi : phis) {
        const State state{program, phi->Flags<IR::Type>()};
        for (u32 index = 0; index < phi->NumArgs(); ++index) {
            IR::IREmitter ir{*phi->PhiBlock(index)};
            state.Store(ir, phi->Arg(index));
        }
        IR::IREmitter ir{*phi->GetParent(), IR::Block::InstructionList::s_iterator_to(*phi)};
        phi->ReplaceUsesWithAndRemove(state.Load(ir));
    }
}

bool IsMemory(const IR::Inst& inst) {
    return Shader::Optimization::IsBufferInstruction(inst) ||
           Shader::Optimization::IsImageInstruction(inst) ||
           Shader::Optimization::IsSharedMemoryInstruction(inst) ||
           inst.GetOpcode() == IR::Opcode::DataAppend ||
           inst.GetOpcode() == IR::Opcode::DataConsume;
}

IR::U1 GroupAny(IR::IREmitter& ir, const IR::U1& predicate, u32 threads) {
    const auto halves = ir.UnpackUint2x32(ir.Ballot(predicate));
    const auto bits = ir.BitwiseOr(IR::U32{ir.CompositeExtract(halves, 0)},
                                   IR::U32{ir.CompositeExtract(halves, 1)});
    const auto local = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
    ir.WriteShared(32, bits, ir.ShiftLeftLogical(local, ir.Imm32(2u)));
    ir.Barrier();
    auto combined = ir.Imm32(0u);
    for (u32 wave = 0; wave < threads / 64; ++wave) {
        combined = ir.BitwiseOr(
            combined, IR::U32{ir.LoadShared(32, false, ir.Imm32(wave * 64u * u32{sizeof(u32)}))});
    }
    ir.Barrier();
    return ir.INotEqual(combined, ir.Imm32(0u));
}

} // namespace

void TessellationControlFlow::CollectBarriers(const IR::Program& program) {
    barriers.clear();
    for (auto* block : program.blocks) {
        for (const auto& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::Barrier) {
                barriers.insert(&inst);
            }
        }
    }
}

void TessellationControlFlow::Lower(IR::Program& program, u32 threads) {
    if (program.blocks.empty() || threads < 64 || threads > 256 || threads % 64 ||
        program.info.shared_memory_scratch_size) {
        throw std::runtime_error("Invalid Metal tessellation control-flow configuration");
    }
    auto edges = ControlEdges(program);
    for (auto& [block, edge] : edges) {
        IR::IREmitter ir{*block};
        edge.condition = ir.ConditionRef(edge.condition);
    }
    LowerPhis(program);
    auto& pool = *program.blocks.front()->inst_pool;
    const auto make_block = [&] {
        blocks.push_back(std::make_unique<IR::Block>(pool));
        return blocks.back().get();
    };
    struct Segment {
        IR::Block* block;
        Edge edge;
        bool parked{};
        IR::U1 active;
    };
    std::vector<Segment> segments;
    for (auto* block : program.blocks) {
        auto* current = block;
        for (;;) {
            auto found = std::ranges::find_if(
                *current, [&](const auto& inst) { return barriers.contains(&inst); });
            if (found == current->end()) {
                segments.push_back({current, edges.at(block)});
                break;
            }
            auto* tail = make_block();
            tail->Instructions().splice(tail->end(), current->Instructions(), std::next(found),
                                        current->end());
            for (auto& inst : *tail) {
                inst.SetParent(tail);
            }
            found->Invalidate();
            segments.push_back({current, {IR::U1{IR::Value{true}}, tail}, true});
            current = tail;
        }
    }
    if (segments.size() >= Parked) {
        throw std::runtime_error("Metal tessellation program counter exceeds capacity");
    }
    const IR::VirtualReg pc{program.next_reg_index++, IR::Type::U32};
    std::unordered_map<IR::Block*, u32> ids;
    for (u32 index = 0; index < segments.size(); ++index) {
        auto& segment = segments[index];
        ids.emplace(segment.block, index);
        IR::IREmitter ir{*segment.block, segment.block->begin()};
        segment.active = ir.IEqual(IR::U32{ir.GetVirtualReg(pc)}, ir.Imm32(index));
        IR::IREmitter end{*segment.block};
        segment.edge.condition = end.ConditionRef(segment.edge.condition);
    }
    std::vector<IR::Inst*> cross_values;
    for (const auto& segment : segments) {
        for (auto& inst : *segment.block) {
            if (inst.Type() != IR::Type::Void &&
                std::ranges::any_of(inst.Uses(), [&](const auto& use) {
                    return use.user->GetParent() != segment.block;
                })) {
                cross_values.push_back(&inst);
            }
        }
    }
    for (auto* inst : cross_values) {
        const State state{program, inst->Type()};
        std::vector<IR::Use> uses{inst->Uses().begin(), inst->Uses().end()};
        IR::IREmitter store{*inst->GetParent(),
                            std::next(IR::Block::InstructionList::s_iterator_to(*inst))};
        state.Store(store, IR::Value{inst});
        for (const auto& use : uses) {
            if (use.user->GetParent() == inst->GetParent()) {
                continue;
            }
            IR::IREmitter load{*use.user->GetParent(),
                               IR::Block::InstructionList::s_iterator_to(*use.user)};
            use.user->SetArg(use.operand, state.Load(load));
        }
    }
    for (auto& segment : segments) {
        std::vector<IR::Inst*> instructions;
        for (auto& inst : *segment.block) {
            instructions.push_back(&inst);
        }
        IR::U1 memory{IR::Value{true}};
        bool predicated{};
        for (auto* inst : instructions) {
            IR::IREmitter ir{*segment.block, IR::Block::InstructionList::s_iterator_to(*inst)};
            const auto opcode = inst->GetOpcode();
            if (opcode == IR::Opcode::ReadFirstLane) {
                throw std::runtime_error("Metal tessellation read-first requires captured EXEC");
            }
            if (opcode == IR::Opcode::ReadConst) {
                throw std::runtime_error(
                    "Metal tessellation scheduled constant read requires native predication");
            }
            if (opcode == IR::Opcode::BeginMemoryAccess) {
                if (predicated) {
                    throw std::runtime_error("Nested Metal tessellation memory predicate");
                }
                memory = IR::U1{inst->Arg(0)};
                predicated = true;
                inst->Invalidate();
            } else if (opcode == IR::Opcode::EndMemoryAccess) {
                if (!predicated) {
                    throw std::runtime_error("Unbalanced Metal tessellation memory predicate");
                }
                memory = ir.Imm1(true);
                predicated = false;
                inst->Invalidate();
            } else if (opcode == IR::Opcode::SetVirtualRegister) {
                const auto reg = inst->Arg(0).VirtualReg();
                inst->SetArg(1, ir.Select(segment.active, inst->Arg(1), ir.GetVirtualReg(reg)));
            } else if (opcode == IR::Opcode::Ballot || opcode == IR::Opcode::GroupAny) {
                inst->SetArg(0, ir.LogicalAnd(segment.active, IR::U1{inst->Arg(0)}));
            } else if (IsMemory(*inst)) {
                ir.BeginMemoryAccess(ir.LogicalAnd(segment.active, memory));
                IR::IREmitter end{*segment.block,
                                  std::next(IR::Block::InstructionList::s_iterator_to(*inst))};
                end.EndMemoryAccess();
            } else if (opcode == IR::Opcode::Prologue || opcode == IR::Opcode::Epilogue) {
                inst->Invalidate();
            } else if (inst->MayHaveSideEffects() && opcode != IR::Opcode::ConditionRef &&
                       opcode != IR::Opcode::Barrier &&
                       opcode != IR::Opcode::WorkgroupMemoryBarrier &&
                       opcode != IR::Opcode::DeviceMemoryBarrier &&
                       opcode != IR::Opcode::SubgroupBarrier) {
                throw std::runtime_error(
                    "Metal tessellation scheduled side effect requires lowering");
            }
        }
        if (predicated) {
            throw std::runtime_error("Metal tessellation memory predicate crosses a control block");
        }
        IR::IREmitter ir{*segment.block};
        const auto target = [&](IR::Block* block) { return block ? ids.at(block) : Done; };
        const auto yes = ir.Imm32(target(segment.edge.yes));
        const auto no = ir.Imm32(target(segment.edge.no));
        IR::U32 next{ir.Select(segment.edge.condition, yes, no)};
        if (segment.parked) {
            next = ir.BitwiseOr(next, ir.Imm32(Parked));
        }
        ir.SetVirtualReg(pc, ir.Select(segment.active, next, ir.GetVirtualReg(pc)));
    }
    auto* entry = make_block();
    auto* header = make_block();
    auto* repeat = make_block();
    auto* exit = make_block();
    IR::IREmitter initialize{*entry};
    initialize.Prologue();
    initialize.SetVirtualReg(pc, initialize.Imm32(0u));
    IR::IREmitter finish{*repeat};
    const IR::U32 current{finish.GetVirtualReg(pc)};
    const auto waiting =
        finish.INotEqual(finish.BitwiseAnd(current, finish.Imm32(Parked)), finish.Imm32(0u));
    const auto done = finish.IEqual(current, finish.Imm32(Done));
    const auto runnable = finish.LogicalNot(finish.LogicalOr(waiting, done));
    const auto release = finish.LogicalNot(GroupAny(finish, runnable, threads));
    finish.Barrier(IR::MemoryClass::All);
    const IR::U32 next{finish.Select(
        finish.LogicalAnd(release, finish.LogicalAnd(waiting, finish.LogicalNot(done))),
        finish.BitwiseAnd(current, finish.Imm32(~Parked)), current)};
    finish.SetVirtualReg(pc, next);
    const auto continuing =
        finish.ConditionRef(GroupAny(finish, finish.INotEqual(next, finish.Imm32(Done)), threads));
    IR::IREmitter{*exit}.Epilogue();
    program.blocks = {entry, header};
    for (auto& segment : segments) {
        program.blocks.push_back(segment.block);
    }
    program.blocks.push_back(repeat);
    program.blocks.push_back(exit);
    for (auto* block : program.blocks) {
        block->imm_predecessors.clear();
        block->imm_successors.clear();
        block->ssa_state.Reset();
    }
    for (u32 index = 1; index < program.blocks.size() - 1; ++index) {
        program.blocks[index - 1]->AddBranch(program.blocks[index]);
    }
    repeat->AddBranch(header);
    repeat->AddBranch(exit);
    using Node = IR::AbstractSyntaxNode;
    using Type = Node::Type;
    program.syntax_list = {
        {.data = {.block = entry}, .type = Type::Block},
        {.data = {.block = header}, .type = Type::Block},
        {.data = {.loop = {segments.front().block, repeat, exit}}, .type = Type::Loop}};
    for (auto& segment : segments) {
        program.syntax_list.push_back({.data = {.block = segment.block}, .type = Type::Block});
    }
    program.syntax_list.push_back({.data = {.block = repeat}, .type = Type::Block});
    program.syntax_list.push_back(
        {.data = {.repeat = {continuing, header, exit}}, .type = Type::Repeat});
    program.syntax_list.push_back({.data = {.block = exit}, .type = Type::Block});
    program.syntax_list.push_back({.type = Type::Return});
    program.post_order_blocks = IR::PostOrder(entry);
    program.info.shared_memory_scratch_size = threads * sizeof(u32);
}

} // namespace Metal
