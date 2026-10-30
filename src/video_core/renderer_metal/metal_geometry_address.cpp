// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/resource_pass.h"
#include "shader_recompiler/ir/program.h"
#include "video_core/renderer_metal/metal_geometry_address.h"
#include "video_core/renderer_metal/metal_guest_ranges.h"

namespace Metal {
namespace {

namespace IR = Shader::IR;
using Wide = unsigned __int128;

struct Interval {
    u64 lower{};
    u64 upper{std::numeric_limits<u64>::max()};
};

u64 DomainMask(u8 bits) {
    return bits == 64 ? std::numeric_limits<u64>::max() : (u64{1} << bits) - 1;
}

u8 ValueBits(IR::Value value) {
    switch (value.Type()) {
    case IR::Type::U1:
        return 1;
    case IR::Type::U64:
    case IR::Type::F64:
        return 64;
    default:
        return 32;
    }
}

Interval Wrap(Wide lower, Wide upper, u8 bits) {
    const auto mask = DomainMask(bits);
    return lower >> bits == upper >> bits ? Interval{u64(lower) & mask, u64(upper) & mask}
                                          : Interval{0, mask};
}

Interval Union(Interval a, Interval b) {
    return {std::min(a.lower, b.lower), std::max(a.upper, b.upper)};
}

u64 PossibleBits(u64 upper) {
    return upper ? std::numeric_limits<u64>::max() >> std::countl_zero(upper) : 0;
}

Interval Less(Interval a, Interval b, bool equal) {
    if (equal ? a.upper <= b.lower : a.upper < b.lower) {
        return {1, 1};
    }
    if (equal ? a.lower > b.upper : a.lower >= b.upper) {
        return {0, 0};
    }
    return {0, 1};
}

u32 OperationArguments(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::ConditionRef:
    case IR::Opcode::GuardBufferAddress:
    case IR::Opcode::ConvertU64U32:
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastF32U32:
    case IR::Opcode::INeg32:
    case IR::Opcode::INeg64:
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::BitwiseNot64:
    case IR::Opcode::LogicalNot:
        return 1;
    case IR::Opcode::IAdd32:
    case IR::Opcode::IAdd64:
    case IR::Opcode::ISub32:
    case IR::Opcode::ISub64:
    case IR::Opcode::IMul32:
    case IR::Opcode::IMul64:
    case IR::Opcode::UMulHi:
    case IR::Opcode::UDiv32:
    case IR::Opcode::UMod32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftLeftLogical64:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::ShiftRightLogical64:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseAnd64:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseOr64:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitwiseXor64:
    case IR::Opcode::UMin32:
    case IR::Opcode::UMax32:
    case IR::Opcode::IEqual32:
    case IR::Opcode::IEqual64:
    case IR::Opcode::INotEqual32:
    case IR::Opcode::INotEqual64:
    case IR::Opcode::ULessThan32:
    case IR::Opcode::ULessThan64:
    case IR::Opcode::ULessThanEqual32:
    case IR::Opcode::ULessThanEqual64:
    case IR::Opcode::UGreaterThan32:
    case IR::Opcode::UGreaterThan64:
    case IR::Opcode::UGreaterThanEqual32:
    case IR::Opcode::UGreaterThanEqual64:
    case IR::Opcode::LogicalAnd:
    case IR::Opcode::LogicalOr:
    case IR::Opcode::LogicalXor:
        return 2;
    case IR::Opcode::SelectU32:
    case IR::Opcode::SelectU64:
    case IR::Opcode::SelectF32:
    case IR::Opcode::SelectU1:
        return 3;
    default:
        return 0;
    }
}

Interval EvaluateOperation(IR::Opcode opcode, std::array<Interval, 3> args, u8 bits,
                           u32 component) {
    const auto [a, b, c] = args;
    const bool constants = a.lower == a.upper && b.lower == b.upper;
    const u64 mask = DomainMask(bits);
    const Interval unknown{0, mask};
    switch (opcode) {
    case IR::Opcode::ConditionRef:
    case IR::Opcode::GuardBufferAddress:
    case IR::Opcode::ConvertU64U32:
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastF32U32:
        return a;
    case IR::Opcode::CompositeExtractU32x2:
        return component == 2   ? a
               : component == 1 ? Interval{a.lower >> 32, a.upper >> 32}
                                : Wrap(a.lower, a.upper, 32);
    case IR::Opcode::PackUint2x32:
        return {a.lower | (b.lower << 32), a.upper | (b.upper << 32)};
    case IR::Opcode::IAdd32:
    case IR::Opcode::IAdd64:
        return Wrap(Wide{a.lower} + b.lower, Wide{a.upper} + b.upper, bits);
    case IR::Opcode::ISub32:
    case IR::Opcode::ISub64:
        return Wrap((Wide{1} << bits) + a.lower - b.upper, (Wide{1} << bits) + a.upper - b.lower,
                    bits);
    case IR::Opcode::INeg32:
    case IR::Opcode::INeg64:
        return Wrap((Wide{1} << bits) - a.upper, (Wide{1} << bits) - a.lower, bits);
    case IR::Opcode::IMul32:
    case IR::Opcode::IMul64:
        return Wrap(Wide{a.lower} * b.lower, Wide{a.upper} * b.upper, bits);
    case IR::Opcode::UMulHi:
        return {u64(Wide{a.lower} * b.lower >> 32), u64(Wide{a.upper} * b.upper >> 32)};
    case IR::Opcode::UDiv32:
        return b.lower ? Interval{a.lower / b.upper, a.upper / b.lower} : unknown;
    case IR::Opcode::UMod32:
        if (!b.lower) {
            return unknown;
        }
        if (a.upper < b.lower) {
            return a;
        }
        if (b.lower == b.upper && a.lower / b.lower == a.upper / b.lower) {
            return {a.lower % b.lower, a.upper % b.lower};
        }
        return {0, std::min(a.upper, b.upper - 1)};
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftLeftLogical64:
        return b.upper < bits ? Wrap(Wide{a.lower} << b.lower, Wide{a.upper} << b.upper, bits)
                              : unknown;
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::ShiftRightLogical64:
        return b.upper < bits ? Interval{a.lower >> b.upper, a.upper >> b.lower} : unknown;
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseAnd64:
        return constants ? Interval{a.lower & b.lower, a.upper & b.upper}
                         : Interval{0, std::min(a.upper, b.upper)};
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseOr64:
        return constants ? Interval{a.lower | b.lower, a.upper | b.upper}
                         : Interval{std::max(a.lower, b.lower),
                                    PossibleBits(a.upper) | PossibleBits(b.upper)};
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitwiseXor64:
        return constants ? Interval{a.lower ^ b.lower, a.upper ^ b.upper}
                         : Interval{0, PossibleBits(a.upper) | PossibleBits(b.upper)};
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::BitwiseNot64:
        return {mask ^ a.upper, mask ^ a.lower};
    case IR::Opcode::UMin32:
        return {std::min(a.lower, b.lower), std::min(a.upper, b.upper)};
    case IR::Opcode::UMax32:
        return {std::max(a.lower, b.lower), std::max(a.upper, b.upper)};
    case IR::Opcode::IEqual32:
    case IR::Opcode::IEqual64:
    case IR::Opcode::INotEqual32:
    case IR::Opcode::INotEqual64: {
        Interval equal{0, 1};
        if (a.upper < b.lower || b.upper < a.lower) {
            equal = {0, 0};
        } else if (constants) {
            equal = {1, 1};
        }
        return opcode == IR::Opcode::IEqual32 || opcode == IR::Opcode::IEqual64
                   ? equal
                   : Interval{1 - equal.upper, 1 - equal.lower};
    }
    case IR::Opcode::ULessThan32:
    case IR::Opcode::ULessThan64:
        return Less(a, b, false);
    case IR::Opcode::ULessThanEqual32:
    case IR::Opcode::ULessThanEqual64:
        return Less(a, b, true);
    case IR::Opcode::UGreaterThan32:
    case IR::Opcode::UGreaterThan64:
        return Less(b, a, false);
    case IR::Opcode::UGreaterThanEqual32:
    case IR::Opcode::UGreaterThanEqual64:
        return Less(b, a, true);
    case IR::Opcode::LogicalAnd:
        return {a.lower & b.lower, a.upper & b.upper};
    case IR::Opcode::LogicalOr:
        return {a.lower | b.lower, a.upper | b.upper};
    case IR::Opcode::LogicalXor:
        return constants ? Interval{a.lower ^ b.lower, a.upper ^ b.upper} : Interval{0, 1};
    case IR::Opcode::LogicalNot:
        return {1 - a.upper, 1 - a.lower};
    case IR::Opcode::SelectU32:
    case IR::Opcode::SelectU64:
    case IR::Opcode::SelectF32:
    case IR::Opcode::SelectU1:
        return a.lower ? b : !a.upper ? c : Union(b, c);
    default:
        return unknown;
    }
}

u32 AccessByteSize(const IR::Inst& inst, u32 shift, AmdGpu::DataFormat format) {
    using IR::Opcode;
    if (Shader::Optimization::IsBufferAtomic(inst)) {
        return 1u << shift;
    }
    switch (inst.GetOpcode()) {
    case Opcode::LoadBufferFormatF32:
    case Opcode::StoreBufferFormatF32:
        return format == AmdGpu::DataFormat::FormatInvalid ? 0
                                                           : AmdGpu::NumBitsPerBlock(format) / 8;
    case Opcode::LoadBufferU8:
    case Opcode::StoreBufferU8:
    case Opcode::LoadBufferU16:
    case Opcode::StoreBufferU16:
    case Opcode::LoadBufferU32:
    case Opcode::StoreBufferU32:
    case Opcode::LoadBufferU64:
    case Opcode::StoreBufferU64:
    case Opcode::LoadBufferF32:
    case Opcode::StoreBufferF32:
    case Opcode::ReadConstBuffer:
        return 1u << shift;
    case Opcode::LoadBufferU32x2:
    case Opcode::StoreBufferU32x2:
    case Opcode::LoadBufferF32x2:
    case Opcode::StoreBufferF32x2:
        return 8;
    case Opcode::LoadBufferU32x3:
    case Opcode::StoreBufferU32x3:
    case Opcode::LoadBufferF32x3:
    case Opcode::StoreBufferF32x3:
        return 12;
    case Opcode::LoadBufferU32x4:
    case Opcode::StoreBufferU32x4:
    case Opcode::LoadBufferF32x4:
    case Opcode::StoreBufferF32x4:
        return 16;
    default:
        return 0;
    }
}

struct BufferCapacityGuard {
    const IR::Inst* comparison;
    IR::Value bias;
};

std::optional<BufferCapacityGuard> FindBufferCapacityGuard(IR::Value predicate, u32 resource,
                                                           const Shader::Info& info) {
    const auto is_capacity = [&](IR::Value value) {
        const auto* convert = value.TryInst();
        if (!convert || convert->GetOpcode() != IR::Opcode::ConvertU64U32) {
            return false;
        }
        const auto* read = convert->Arg(0).TryInst();
        return read && read->GetOpcode() == IR::Opcode::ReadConstBuffer &&
               read->Arg(0).Type() == IR::Type::U32 && read->Arg(0).IsImmediate() &&
               read->Arg(0).U32() < info.buffers.size() &&
               info.buffers[read->Arg(0).U32()].buffer_type == Shader::BufferType::BufferBounds &&
               read->Arg(1).Type() == IR::Type::U32 && read->Arg(1).IsImmediate() &&
               read->Arg(1).U32() == resource;
    };
    std::optional<BufferCapacityGuard> result;
    std::vector<IR::Value> pending{predicate};
    std::unordered_set<const IR::Inst*> visited;
    while (!pending.empty()) {
        const auto* inst = pending.back().TryInst();
        pending.pop_back();
        if (!inst || !visited.insert(inst).second) {
            continue;
        }
        if (inst->GetOpcode() == IR::Opcode::LogicalAnd) {
            pending.push_back(inst->Arg(0));
            pending.push_back(inst->Arg(1));
            continue;
        }
        if (inst->GetOpcode() == IR::Opcode::ConditionRef) {
            pending.push_back(inst->Arg(0));
            continue;
        }
        if (inst->GetOpcode() != IR::Opcode::ULessThan64) {
            continue;
        }
        std::optional<IR::Value> bias;
        if (is_capacity(inst->Arg(1))) {
            bias = IR::Value{u64{0}};
        } else if (const auto* sum = inst->Arg(1).TryInst();
                   sum && sum->GetOpcode() == IR::Opcode::IAdd64) {
            if (is_capacity(sum->Arg(0))) {
                bias = sum->Arg(1);
            } else if (is_capacity(sum->Arg(1))) {
                bias = sum->Arg(0);
            }
        }
        if (bias) {
            if (result) {
                throw std::runtime_error("Ambiguous native guest buffer capacity guard");
            }
            result = BufferCapacityGuard{inst, *bias};
        }
    }
    return result;
}

} // namespace

class GeometryBufferAddressProgram::Builder {
public:
    static GeometryBufferAddressProgram Inspect(const IR::Program& program, bool binding_sizes);

    u32 Add(IR::Value root) {
        if (const auto* inst = root.TryInst()) {
            if (const auto found = completed.find(inst); found != completed.end()) {
                return found->second;
            }
        }
        struct Pending {
            IR::Value value;
            Node node;
            u32 next_argument{};
        };
        std::vector<Pending> pending;
        const auto push = [&](IR::Value value) {
            if (const auto* inst = value.TryInst(); inst && !active.insert(inst).second) {
                throw std::runtime_error("Cyclic native geometry address arithmetic");
            }
            auto node = MakeNode(value);
            if (node.num_args > std::numeric_limits<u32>::max() - result.arguments.size()) {
                throw std::runtime_error("Native geometry address operands exceed capacity");
            }
            node.first_argument = u32(result.arguments.size());
            result.arguments.resize(result.arguments.size() + node.num_args);
            pending.push_back({value, node});
        };
        push(root);
        while (!pending.empty()) {
            auto& frame = pending.back();
            const auto* inst = frame.value.TryInst();
            if (frame.next_argument < frame.node.num_args) {
                const auto child = Child(*inst, frame.node, frame.next_argument);
                const auto* child_inst = child.TryInst();
                const auto found = completed.find(child_inst);
                if (child_inst && found != completed.end()) {
                    result.arguments[frame.node.first_argument + frame.next_argument++] =
                        found->second;
                } else if (child_inst && active.contains(child_inst)) {
                    const auto cycle = std::ranges::find_if(pending, [&](const auto& item) {
                        return item.value.TryInst() == child_inst;
                    });
                    const bool phi_cycle = std::any_of(cycle, pending.end(), [](const auto& item) {
                        return item.node.kind == Kind::Merge;
                    });
                    if (!phi_cycle) {
                        throw std::runtime_error("Cyclic native geometry address arithmetic");
                    }
                    Node unknown;
                    unknown.bits = ValueBits(child);
                    result.arguments[frame.node.first_argument + frame.next_argument++] =
                        Append(unknown);
                } else {
                    push(child);
                }
                continue;
            }
            const u32 index = Append(frame.node);
            if (inst) {
                completed.insert_or_assign(inst, index);
                active.erase(inst);
            }
            pending.pop_back();
            if (pending.empty()) {
                return index;
            }
            auto& parent = pending.back();
            result.arguments[parent.node.first_argument + parent.next_argument++] = index;
        }
        throw std::runtime_error("Empty native geometry address expression");
    }

    GeometryBufferAddressProgram result;

private:
    u32 BindingEnd(const BufferCapacityGuard& guard) {
        const u32 address = Add(guard.comparison->Arg(0));
        const u32 bias = Add(guard.bias);
        Node node;
        node.kind = Kind::Operation;
        node.opcode = IR::Opcode::ISub64;
        node.bits = 64;
        node.num_args = 2;
        node.first_argument = u32(result.arguments.size());
        result.arguments.push_back(address);
        result.arguments.push_back(bias);
        return Append(node);
    }

    u32 Append(const Node& node) {
        if (result.nodes.size() >= std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Native geometry address program exceeds capacity");
        }
        const u32 index = u32(result.nodes.size());
        result.nodes.push_back(node);
        return index;
    }

    IR::Value Child(const IR::Inst& inst, const Node& node, u32 index) {
        if (node.opcode == IR::Opcode::CompositeExtractU32x2) {
            return inst.Arg(0).Inst()->Arg(node.component == 2 ? u32(node.key) : 0);
        }
        if (node.opcode == IR::Opcode::PackUint2x32) {
            return inst.Arg(0).Inst()->Arg(index);
        }
        return inst.Arg(index);
    }

    Node MakeNode(IR::Value value) {
        Node node;
        node.bits = ValueBits(value);
        const auto* inst = value.TryInst();
        if (!inst) {
            switch (value.Type()) {
            case IR::Type::U1:
                node.key = value.U1();
                break;
            case IR::Type::U32:
                node.key = value.U32();
                break;
            case IR::Type::U64:
                node.key = value.U64();
                break;
            case IR::Type::F32:
                node.key = std::bit_cast<u32>(value.F32());
                break;
            default:
                return node;
            }
            node.kind = Kind::Constant;
            return node;
        }
        if (ignored_conditions.contains(inst)) {
            node.kind = Kind::Constant;
            node.key = 1;
            return node;
        }
        node.opcode = inst->GetOpcode();
        if (node.opcode == IR::Opcode::GetAttributeU32 ||
            node.opcode == IR::Opcode::GetAttributeU1) {
            if (inst->Arg(0).Type() != IR::Type::Attribute || !inst->Arg(1).IsImmediate() ||
                inst->Arg(1).Type() != IR::Type::U32) {
                throw std::runtime_error("Unresolved native geometry address input");
            }
            node.kind = Kind::Attribute;
            node.key = u64(inst->Arg(0).Attribute());
            node.component = inst->Arg(1).U32();
        } else if (node.opcode == IR::Opcode::GetUserData) {
            if (inst->Arg(0).Type() != IR::Type::ScalarReg) {
                throw std::runtime_error("Unresolved native geometry address user data");
            }
            node.kind = Kind::UserData;
            node.key = u32(inst->Arg(0).ScalarReg());
        } else if (node.opcode == IR::Opcode::ReadConst && inst->Flags<u32>()) {
            node.kind = Kind::UserData;
            node.key = inst->Flags<u32>();
        } else if (node.opcode == IR::Opcode::Phi) {
            if (inst->NumArgs() > std::numeric_limits<u32>::max()) {
                throw std::runtime_error("Native geometry address merge exceeds capacity");
            }
            node.num_args = u32(inst->NumArgs());
            node.kind = node.num_args ? Kind::Merge : Kind::Unknown;
        } else if (node.opcode == IR::Opcode::CompositeExtractU32x2) {
            if (!inst->Arg(1).IsImmediate() || inst->Arg(1).Type() != IR::Type::U32 ||
                inst->Arg(1).U32() >= 2) {
                throw std::runtime_error("Invalid native geometry address vector component");
            }
            const auto* vector = inst->Arg(0).TryInst();
            if (vector && (vector->GetOpcode() == IR::Opcode::UnpackUint2x32 ||
                           vector->GetOpcode() == IR::Opcode::CompositeConstructU32x2)) {
                node.key = inst->Arg(1).U32();
                node.component =
                    vector->GetOpcode() == IR::Opcode::UnpackUint2x32 ? u32(node.key) : 2;
                node.kind = Kind::Operation;
                node.num_args = 1;
            }
        } else if (node.opcode == IR::Opcode::PackUint2x32) {
            const auto* vector = inst->Arg(0).TryInst();
            if (vector && vector->GetOpcode() == IR::Opcode::CompositeConstructU32x2) {
                node.kind = Kind::Operation;
                node.num_args = 2;
            }
        } else if (node.num_args = OperationArguments(node.opcode); node.num_args) {
            node.kind = Kind::Operation;
        }
        return node;
    }

    std::unordered_map<const IR::Inst*, u32> completed;
    std::unordered_set<const IR::Inst*> active;
    std::unordered_set<const IR::Inst*> ignored_conditions;
};

GeometryBufferAddressProgram GeometryBufferAddressProgram::Builder::Inspect(
    const IR::Program& program, bool binding_sizes) {
    using namespace Shader::Optimization;
    GeometryBufferAddressProgram::Builder builder;
    for (const auto* block : program.blocks) {
        std::optional<IR::Value> predicate;
        for (const IR::Inst& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::BeginMemoryAccess) {
                if (predicate || inst.Arg(0).Type() != IR::Type::U1) {
                    throw std::runtime_error("Invalid native geometry memory access scope");
                }
                predicate = inst.Arg(0);
                continue;
            }
            if (inst.GetOpcode() == IR::Opcode::EndMemoryAccess) {
                if (!predicate) {
                    throw std::runtime_error("Unmatched native geometry memory access scope");
                }
                predicate.reset();
                continue;
            }
            if (!IsBufferInstruction(inst)) {
                continue;
            }
            if (!inst.Arg(0).IsImmediate() || inst.Arg(0).Type() != IR::Type::U32 ||
                inst.Arg(0).U32() >= program.info.buffers.size() ||
                inst.Arg(0).U32() >= Shader::NUM_BUFFERS) {
                throw std::runtime_error("Native geometry address buffer handle was not resolved");
            }
            const u32 resource = inst.Arg(0).U32();
            if (program.info.buffers[resource].buffer_type != Shader::BufferType::Guest) {
                continue;
            }
            if (binding_sizes && inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
                continue;
            }
            auto format = AmdGpu::DataFormat::FormatInvalid;
            if (inst.GetOpcode() == IR::Opcode::LoadBufferFormatF32 ||
                inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32) {
                const auto flags = inst.Flags<IR::BufferInstInfo>();
                format = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid
                             ? AmdGpu::RemapDataFormat(flags.inst_data_fmt.Value())
                             : program.info.buffers[resource].GetSharp(program.info).GetDataFmt();
            }
            const u32 shift = inst.GetOpcode() == IR::Opcode::ReadConstBuffer
                                  ? 2
                                  : GetBufferAddressShift(inst, format);
            const bool atomic = IsBufferAtomic(inst);
            const bool store = IsBufferStore(inst);
            const u32 bytes =
                inst.Arg(1).Type() == IR::Type::U32 ? AccessByteSize(inst, shift, format) : 0;
            const auto no_condition = std::numeric_limits<u32>::max();
            const auto* address = inst.Arg(1).TryInst();
            u32 bounds_predicate = no_condition;
            if (address && address->GetOpcode() == IR::Opcode::GuardBufferAddress) {
                if (address->Arg(1).Type() != IR::Type::U1) {
                    throw std::runtime_error("Invalid native geometry buffer guard");
                }
                if (binding_sizes) {
                    const auto guard =
                        FindBufferCapacityGuard(address->Arg(1), resource, program.info);
                    if (guard) {
                        builder.ignored_conditions.insert(guard->comparison);
                        builder.result.accesses.push_back(
                            {resource, builder.BindingEnd(*guard), 0, 1,
                             predicate ? builder.Add(*predicate) : no_condition,
                             builder.Add(address->Arg(1)), !store || atomic, store || atomic, 64});
                    }
                    continue;
                }
                bounds_predicate = builder.Add(address->Arg(1));
            }
            if (binding_sizes) {
                continue;
            }
            builder.result.accesses.push_back({resource, builder.Add(inst.Arg(1)), shift, bytes,
                                               predicate ? builder.Add(*predicate) : no_condition,
                                               bounds_predicate, !store || atomic,
                                               store || atomic});
        }
        if (predicate) {
            throw std::runtime_error("Unclosed native geometry memory access scope");
        }
    }
    return std::move(builder.result);
}

GeometryBufferAddressProgram InspectGeometryBufferAddressProgram(const IR::Program& program) {
    return GeometryBufferAddressProgram::Builder::Inspect(program, false);
}

GeometryBufferAddressProgram InspectGuestBufferBindingProgram(const IR::Program& program) {
    return GeometryBufferAddressProgram::Builder::Inspect(program, true);
}

GeometryBufferAddressProgram InspectScalarConstantAddressProgram(
    std::span<const IR::Value> addresses, std::span<const IR::Value> predicates) {
    if (addresses.size() != predicates.size() ||
        addresses.size() > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Invalid Metal scalar address program inputs");
    }
    GeometryBufferAddressProgram::Builder builder;
    for (u32 index = 0; index < addresses.size(); ++index) {
        if (addresses[index].Type() != IR::Type::U64 || predicates[index].Type() != IR::Type::U1) {
            throw std::runtime_error("Invalid Metal scalar address program types");
        }
        const u32 address = builder.Add(addresses[index]);
        const u32 predicate = builder.Add(predicates[index]);
        builder.result.accesses.push_back({index, address, 0, sizeof(u32), predicate,
                                           std::numeric_limits<u32>::max(), true, false, 64});
    }
    return std::move(builder.result);
}

GeometryBufferAddressProgram GeometryBufferAddressProgram::Select(
    std::span<const u32> indices) const {
    auto result = *this;
    result.accesses.clear();
    for (const u32 index : indices) {
        result.accesses.push_back(accesses.at(index));
    }
    return result;
}

class GeometryBufferAddressProgram::Evaluator {
public:
    Evaluator(const GeometryBufferAddressProgram& program_,
              std::span<const GeometryAddressInput> inputs_, std::span<const u32> user_data_)
        : program{program_}, inputs{inputs_}, user_data{user_data_} {
        base.reserve(program.nodes.size());
        for (u32 index = 0; index < program.nodes.size(); ++index) {
            base.push_back(Value(index, base));
        }
    }

    std::vector<GeometryBufferAddressBounds> Run() const {
        std::vector<GeometryBufferAddressBounds> result;
        result.reserve(program.accesses.size());
        for (const auto& access : program.accesses) {
            const auto address = Address(access);
            std::optional<GeometryBufferByteInterval> bytes;
            if (address && access.byte_size) {
                const auto interval =
                    Wrap(Wide{address->lower} << access.shift,
                         (Wide{address->upper} << access.shift) + access.byte_size - 1,
                         access.address_bits + access.shift);
                bytes = {interval.lower, interval.upper};
            }
            result.push_back({access.resource_index, access.read, access.write, bytes,
                              address.has_value(), access.shift});
        }
        return result;
    }

private:
    u32 Argument(const Node& node, u32 index) const {
        return program.arguments.at(node.first_argument + index);
    }

    Interval Value(u32 index, std::span<const Interval> values) const {
        const auto& node = program.nodes[index];
        Interval value{0, DomainMask(node.bits)};
        switch (node.kind) {
        case Kind::Unknown:
            break;
        case Kind::Constant:
            value = {node.key, node.key};
            break;
        case Kind::Attribute:
            for (const auto& input : inputs) {
                if (u64(input.attribute) == node.key && input.component == node.component) {
                    value = {input.values.lower, input.values.upper};
                    break;
                }
            }
            break;
        case Kind::UserData:
            if (node.key < user_data.size()) {
                value = {user_data[node.key], user_data[node.key]};
            }
            break;
        case Kind::Operation: {
            std::array<Interval, 3> args;
            for (u32 argument = 0; argument < node.num_args; ++argument) {
                args[argument] = values[Argument(node, argument)];
            }
            value = EvaluateOperation(node.opcode, args, node.bits, node.component);
            break;
        }
        case Kind::Merge:
            value = values[Argument(node, 0)];
            for (u32 argument = 1; argument < node.num_args; ++argument) {
                value = Union(value, values[Argument(node, argument)]);
            }
            break;
        }
        if (value.lower > value.upper || value.upper > DomainMask(node.bits)) {
            throw std::runtime_error("Invalid native geometry address result interval");
        }
        return value;
    }

    bool Restrict(u32 index, Interval range, std::vector<Interval>& values) const {
        std::vector<std::pair<u32, Interval>> pending{{index, range}};
        while (!pending.empty()) {
            const auto [current, constraint] = pending.back();
            pending.pop_back();
            const auto& node = program.nodes[current];
            const auto restrict_value = [&](u32 target) {
                const auto lower = std::max(values[target].lower, constraint.lower);
                const auto upper = std::min(values[target].upper, constraint.upper);
                if (lower > upper) {
                    return false;
                }
                values[target] = {lower, upper};
                return true;
            };
            if (!restrict_value(current)) {
                return false;
            }
            const auto narrowed = values[current];
            if (node.kind == Kind::Operation && node.opcode == IR::Opcode::ConvertU64U32) {
                pending.emplace_back(Argument(node, 0), narrowed);
            } else if (node.kind == Kind::Operation &&
                       (node.opcode == IR::Opcode::IAdd32 || node.opcode == IR::Opcode::IAdd64)) {
                auto source = Argument(node, 0);
                auto offset = values[Argument(node, 1)];
                if (offset.lower != offset.upper) {
                    source = Argument(node, 1);
                    offset = values[Argument(node, 0)];
                }
                if (offset.lower == offset.upper &&
                    values[source].upper <= DomainMask(node.bits) - offset.upper) {
                    if (narrowed.upper < offset.upper) {
                        return false;
                    }
                    pending.emplace_back(
                        source,
                        Interval{narrowed.lower > offset.upper ? narrowed.lower - offset.upper : 0,
                                 narrowed.upper - offset.upper});
                }
            }
            if (node.kind == Kind::Attribute || node.kind == Kind::UserData) {
                for (u32 alias = 0; alias < program.nodes.size(); ++alias) {
                    const auto& candidate = program.nodes[alias];
                    if (candidate.kind == node.kind && candidate.bits == node.bits &&
                        candidate.key == node.key && candidate.component == node.component &&
                        !restrict_value(alias)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool Equal(u32 a, u32 b, bool equal, std::vector<Interval>& values) const {
        if (a == b) {
            return equal;
        }
        const auto av = values[a];
        const auto bv = values[b];
        if (equal) {
            const Interval common{std::max(av.lower, bv.lower), std::min(av.upper, bv.upper)};
            return Restrict(a, common, values) && Restrict(b, common, values);
        }
        const auto exclude_constant = [&](u32 variable, u64 constant) {
            const auto range = values[variable];
            if (range.lower == constant && range.upper == constant) {
                return false;
            }
            if (range.lower == constant) {
                return Restrict(variable, {constant + 1, range.upper}, values);
            }
            if (range.upper == constant) {
                return Restrict(variable, {range.lower, constant - 1}, values);
            }
            return true;
        };
        return (av.lower != av.upper || exclude_constant(b, av.lower)) &&
               (bv.lower != bv.upper || exclude_constant(a, bv.lower));
    }

    bool Compare(u32 a, u32 b, bool inclusive, std::vector<Interval>& values) const {
        if (a == b) {
            return inclusive;
        }
        const auto av = values[a];
        const auto bv = values[b];
        if (inclusive ? av.lower > bv.upper : av.lower >= bv.upper) {
            return false;
        }
        const u64 step = inclusive ? 0 : 1;
        return Restrict(a, {av.lower, std::min(av.upper, bv.upper - step)}, values) &&
               Restrict(b, {std::max(bv.lower, av.lower + step), bv.upper}, values);
    }

    bool Constrain(const Access& access, std::vector<Interval>& values) const {
        struct Truth {
            u32 node;
            bool value;
        };
        std::vector<Truth> pending;
        std::vector<u8> visited(program.nodes.size());
        const auto require = [&](u32 index, bool truth) {
            const u8 bit = truth ? 2 : 1;
            if (!Restrict(index, {u64(truth), u64(truth)}, values)) {
                return false;
            }
            if (!(visited[index] & bit)) {
                visited[index] |= bit;
                pending.push_back({index, truth});
            }
            return true;
        };
        const auto no_condition = std::numeric_limits<u32>::max();
        if ((access.predicate != no_condition && !require(access.predicate, true)) ||
            (access.bounds_predicate != no_condition && !require(access.bounds_predicate, true))) {
            return false;
        }
        while (!pending.empty()) {
            const auto [index, truth] = pending.back();
            pending.pop_back();
            const auto& node = program.nodes[index];
            if (node.kind != Kind::Operation) {
                continue;
            }
            const auto a = Argument(node, 0);
            const auto b = node.num_args > 1 ? Argument(node, 1) : a;
            const auto av = values[a];
            const auto bv = values[b];
            switch (node.opcode) {
            case IR::Opcode::ConditionRef:
                if (!require(a, truth)) {
                    return false;
                }
                break;
            case IR::Opcode::LogicalNot:
                if (!require(a, !truth)) {
                    return false;
                }
                break;
            case IR::Opcode::LogicalAnd:
            case IR::Opcode::LogicalOr: {
                const bool both = node.opcode == IR::Opcode::LogicalAnd ? truth : !truth;
                if (both) {
                    if (!require(a, truth) || !require(b, truth)) {
                        return false;
                    }
                } else {
                    if (av.lower == av.upper && bool(av.lower) != truth && !require(b, truth)) {
                        return false;
                    }
                    if (bv.lower == bv.upper && bool(bv.lower) != truth && !require(a, truth)) {
                        return false;
                    }
                }
                break;
            }
            case IR::Opcode::LogicalXor:
                if (av.lower == av.upper && !require(b, truth != bool(av.lower))) {
                    return false;
                }
                if (bv.lower == bv.upper && !require(a, truth != bool(bv.lower))) {
                    return false;
                }
                break;
            case IR::Opcode::IEqual32:
            case IR::Opcode::IEqual64:
            case IR::Opcode::INotEqual32:
            case IR::Opcode::INotEqual64: {
                const bool equal =
                    node.opcode == IR::Opcode::IEqual32 || node.opcode == IR::Opcode::IEqual64;
                if (!Equal(a, b, truth == equal, values)) {
                    return false;
                }
                break;
            }
            case IR::Opcode::ULessThan32:
            case IR::Opcode::ULessThan64:
                if (!(truth ? Compare(a, b, false, values) : Compare(b, a, true, values))) {
                    return false;
                }
                break;
            case IR::Opcode::ULessThanEqual32:
            case IR::Opcode::ULessThanEqual64:
                if (!(truth ? Compare(a, b, true, values) : Compare(b, a, false, values))) {
                    return false;
                }
                break;
            case IR::Opcode::UGreaterThan32:
            case IR::Opcode::UGreaterThan64:
                if (!(truth ? Compare(b, a, false, values) : Compare(a, b, true, values))) {
                    return false;
                }
                break;
            case IR::Opcode::UGreaterThanEqual32:
            case IR::Opcode::UGreaterThanEqual64:
                if (!(truth ? Compare(b, a, true, values) : Compare(a, b, false, values))) {
                    return false;
                }
                break;
            case IR::Opcode::SelectU1: {
                const auto c = Argument(node, 2);
                const auto cv = values[c];
                const auto can_match = [&](Interval range) {
                    return truth ? range.upper != 0 : range.lower == 0;
                };
                if (av.lower == av.upper) {
                    if (!require(av.lower ? b : c, truth)) {
                        return false;
                    }
                } else if (!can_match(bv)) {
                    if (!require(a, false) || !require(c, truth)) {
                        return false;
                    }
                } else if (!can_match(cv)) {
                    if (!require(a, true) || !require(b, truth)) {
                        return false;
                    }
                }
                break;
            }
            default:
                break;
            }
        }
        return true;
    }

    std::optional<Interval> Address(const Access& access) const {
        const auto no_condition = std::numeric_limits<u32>::max();
        if (access.predicate == no_condition && access.bounds_predicate == no_condition) {
            return base.at(access.node);
        }
        auto values = base;
        if (!Constrain(access, values)) {
            return std::nullopt;
        }
        for (u32 index = 0; index < program.nodes.size(); ++index) {
            const auto computed = Value(index, values);
            const auto lower = std::max(values[index].lower, computed.lower);
            const auto upper = std::min(values[index].upper, computed.upper);
            if (lower > upper) {
                return std::nullopt;
            }
            values[index] = {lower, upper};
        }
        return values.at(access.node);
    }

    const GeometryBufferAddressProgram& program;
    std::span<const GeometryAddressInput> inputs;
    std::span<const u32> user_data;
    std::vector<Interval> base;
};

std::vector<GeometryBufferAddressBounds> GeometryBufferAddressProgram::Evaluate(
    std::span<const GeometryAddressInput> inputs, std::span<const u32> user_data) const {
    for (size_t index = 0; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        if (input.values.lower > input.values.upper) {
            throw std::runtime_error("Invalid native geometry address input interval");
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (inputs[previous].attribute == input.attribute &&
                inputs[previous].component == input.component) {
                throw std::runtime_error("Duplicate native geometry address input interval");
            }
        }
    }
    return Evaluator{*this, inputs, user_data}.Run();
}

std::vector<GeometryBufferAddressMapping> MapGeometryBufferAddresses(
    std::span<const GeometryBufferAddressBounds> bounds,
    std::span<const GeometryBufferAddressBinding> bindings) {
    std::array<const GeometryBufferAddressBinding*, Shader::NUM_BUFFERS> resources{};
    for (const auto& binding : bindings) {
        if (binding.resource_index >= resources.size() || resources[binding.resource_index] ||
            (binding.size ? !binding.address || binding.address >= (u64{1} << 40) ||
                                binding.size > (u64{1} << 40) - binding.address ||
                                binding.offset > binding.address
                          : binding.address || binding.offset)) {
            throw std::runtime_error("Invalid native geometry address binding");
        }
        if (binding.paged) {
            if (binding.offset || binding.buffer_size ||
                binding.size > std::numeric_limits<u32>::max()) {
                throw std::runtime_error("Invalid paged native geometry address binding");
            }
        } else {
            (void)GuestBufferCapacity(binding.offset, binding.size, binding.buffer_size);
        }
        resources[binding.resource_index] = &binding;
    }
    if (bounds.size() > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Native geometry address mappings exceed capacity");
    }
    std::vector<GeometryBufferAddressMapping> result;
    result.reserve(bounds.size());
    for (u32 index = 0; index < bounds.size(); ++index) {
        const auto& bound = bounds[index];
        if (bound.resource_index >= resources.size() || (!bound.read && !bound.write) ||
            bound.address_shift > 3 ||
            (bound.bytes &&
             (bound.bytes->lower > bound.bytes->upper ||
              bound.bytes->upper > DomainMask(32 + bound.address_shift) || !bound.reachable))) {
            throw std::runtime_error("Invalid native geometry address bounds");
        }
        GeometryBufferAddressMapping mapping{
            index, bound.resource_index, bound.read, bound.write, GeometryAddressCoverage::Inactive,
            {}};
        const auto* binding = resources[bound.resource_index];
        if (!bound.reachable) {
            result.push_back(mapping);
            continue;
        }
        if (!binding) {
            mapping.coverage = GeometryAddressCoverage::Unbound;
        } else if (!binding->size) {
            mapping.coverage = GeometryAddressCoverage::Empty;
        } else if (!bound.bytes) {
            mapping.coverage = GeometryAddressCoverage::Unknown;
        } else {
            const u64 push_offset = binding->offset & 15;
            const u64 byte_offset = (push_offset >> bound.address_shift) << bound.address_shift;
            const auto relative =
                Wrap(Wide{bound.bytes->lower} + byte_offset, Wide{bound.bytes->upper} + byte_offset,
                     32 + bound.address_shift);
            const auto base = binding->paged
                                  ? binding->address & ~u64{(1u << bound.address_shift) - 1}
                                  : binding->address - push_offset;
            const auto lower = base + relative.lower;
            const auto upper = base + relative.upper;
            mapping.guest_bytes = {lower, upper};
            const auto end = binding->address + binding->size;
            mapping.coverage =
                lower >= binding->address && upper < end   ? GeometryAddressCoverage::Contained
                : upper < binding->address || lower >= end ? GeometryAddressCoverage::Disjoint
                                                           : GeometryAddressCoverage::Partial;
        }
        result.push_back(mapping);
    }
    return result;
}

} // namespace Metal
