// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/resource_pass.h"
#include "shader_recompiler/ir/program.h"
#include "video_core/renderer_metal/metal_geometry_resources.h"
#include "video_core/renderer_metal/metal_geometry_work.h"

namespace Metal {
namespace {

constexpr u64 SourceBit(GeometryAddressSource source) {
    return u64{1} << u32(source);
}

u64 GeometryAttributeSource(const Shader::IR::Inst& inst) {
    namespace IR = Shader::IR;
    const auto attribute = inst.Arg(0).Attribute();
    GeometryAddressSource source;
    u32 components = 1;
    switch (attribute) {
    case IR::Attribute::BaseVertex:
        source = GeometryAddressSource::BaseVertex;
        break;
    case IR::Attribute::BaseInstance:
        source = GeometryAddressSource::BaseInstance;
        break;
    case IR::Attribute::GeometryExportScalar:
        source = GeometryAddressSource::ExportScalar;
        components = 16;
        break;
    case IR::Attribute::GeometryExportVector:
        source = GeometryAddressSource::ExportVector;
        components = 4;
        break;
    case IR::Attribute::GeometryRingOffset:
        source = GeometryAddressSource::RingOffset;
        break;
    case IR::Attribute::GeometryWaveId:
        source = GeometryAddressSource::WaveId;
        break;
    case IR::Attribute::GeometryVertexOffset:
        source = GeometryAddressSource::VertexOffset;
        components = 6;
        break;
    case IR::Attribute::PrimitiveId:
        source = GeometryAddressSource::PrimitiveId;
        break;
    case IR::Attribute::InvocationId:
        source = GeometryAddressSource::InvocationId;
        break;
    case IR::Attribute::GeometryCopyScalar:
        source = GeometryAddressSource::CopyScalar;
        components = 16;
        break;
    case IR::Attribute::GeometryCopyVector:
        source = GeometryAddressSource::CopyVector;
        components = 4;
        break;
    case IR::Attribute::GeometryInputActive:
    case IR::Attribute::GeometryCopyActive:
        source = GeometryAddressSource::InputActive;
        break;
    default:
        return SourceBit(GeometryAddressSource::OtherInput);
    }
    if (!inst.Arg(1).IsImmediate() || inst.Arg(1).Type() != IR::Type::U32 ||
        inst.Arg(1).U32() >= components) {
        throw std::runtime_error("Invalid native geometry address input component");
    }
    return SourceBit(source) << inst.Arg(1).U32();
}

u64 AddressOrigins(std::vector<Shader::IR::Value> pending) {
    namespace IR = Shader::IR;
    using namespace Shader::Optimization;
    std::unordered_set<const IR::Inst*> visited;
    u64 result{};
    while (!pending.empty()) {
        const auto value = pending.back();
        pending.pop_back();
        const auto* inst = value.TryInst();
        if (!inst || !visited.insert(inst).second) {
            continue;
        }
        const auto opcode = inst->GetOpcode();
        if (opcode == IR::Opcode::GetAttribute || opcode == IR::Opcode::GetAttributeU32 ||
            opcode == IR::Opcode::GetAttributeU1) {
            result |= GeometryAttributeSource(*inst);
            continue;
        }
        switch (opcode) {
        case IR::Opcode::GetUserData:
        case IR::Opcode::GetScalarRegister:
        case IR::Opcode::GetVectorRegister:
        case IR::Opcode::GetVirtualRegister:
        case IR::Opcode::GetGotoVariable:
        case IR::Opcode::GetPatch:
        case IR::Opcode::GetTessGenericAttribute:
        case IR::Opcode::ReadTcsGenericOuputAttribute:
        case IR::Opcode::GetPcLo:
        case IR::Opcode::GetScc:
        case IR::Opcode::GetExec:
        case IR::Opcode::GetVcc:
        case IR::Opcode::GetVccLo:
        case IR::Opcode::GetVccHi:
        case IR::Opcode::GetM0:
            result |= SourceBit(GeometryAddressSource::OtherInput);
            break;
        default:
            if (!inst->NumArgs()) {
                result |= SourceBit(GeometryAddressSource::OtherInput);
            }
            break;
        }
        if (opcode == IR::Opcode::GuardBufferAddress) {
            pending.push_back(inst->Arg(0));
            continue;
        }
        if (IsBufferInstruction(*inst) || IsSharedMemoryInstruction(*inst) ||
            IsDataRingInstruction(*inst) || IsImageInstruction(*inst) ||
            opcode == IR::Opcode::ReadConst) {
            result |= SourceBit(GeometryAddressSource::MemoryData);
        }
        if (opcode == IR::Opcode::Phi) {
            result |= SourceBit(GeometryAddressSource::Phi);
        }
        for (size_t argument = 0; argument < inst->NumArgs(); ++argument) {
            pending.push_back(inst->Arg(argument));
        }
    }
    return result;
}

void ValidateGeometryBufferSnapshot(const GeometryBufferSnapshot& snapshot) {
    const auto& access = snapshot.accesses;
    const auto referenced = access.vector_reads | access.vector_writes | access.scalar_reads |
                            access.atomics | access.coherent_reads | access.coherent_writes;
    if ((referenced & ~snapshot.guest_resources).any() ||
        (access.atomics & ~access.vector_reads).any() ||
        (access.atomics & ~access.vector_writes).any() ||
        (access.coherent_reads & ~access.vector_reads).any() ||
        (access.coherent_writes & ~access.vector_writes).any()) {
        throw std::runtime_error("Invalid native geometry buffer access metadata");
    }
    std::bitset<Shader::NUM_BUFFERS> supplied;
    for (const auto& buffer : snapshot.buffers) {
        if (buffer.resource_index >= Shader::NUM_BUFFERS ||
            !snapshot.guest_resources.test(buffer.resource_index) ||
            supplied.test(buffer.resource_index)) {
            throw std::runtime_error("Invalid native geometry buffer snapshot resource");
        }
        supplied.set(buffer.resource_index);
        if ((!buffer.size && buffer.address) ||
            (buffer.size && (!buffer.address || buffer.address >= (u64{1} << 40) ||
                             buffer.size > (u64{1} << 40) - buffer.address ||
                             buffer.size > std::numeric_limits<u32>::max()))) {
            throw std::runtime_error("Invalid native geometry buffer snapshot range");
        }
    }
}

} // namespace

AmdGpu::Buffer ReadGuestBufferSharp(const Shader::BufferResource& resource,
                                    const Shader::Info& info) {
    if (resource.buffer_type != Shader::BufferType::Guest) {
        throw std::runtime_error("Metal guest buffer descriptor requires a guest resource");
    }
    const auto& fetch = resource.sharp_fetch;
    using Summary = Shader::SharpFetch<AmdGpu::Buffer>::Summary;
    const auto valid_range = [&](u32 offset, u32 words) {
        return offset != Shader::UNKNOWN_LOCATION && offset <= info.flattened_ud_buf.size() &&
               words <= info.flattened_ud_buf.size() - offset;
    };
    switch (fetch.summary) {
    case Summary::SingleLoad:
        if (!valid_range(fetch.offsets[0], fetch.N)) {
            throw std::runtime_error("Metal guest buffer descriptor exceeds its shader snapshot");
        }
        break;
    case Summary::MultiLoad:
        for (u32 word = 0; word < fetch.N; ++word) {
            if ((fetch.load_mask & (1u << word)) && !valid_range(fetch.offsets[word], 1)) {
                throw std::runtime_error(
                    "Metal guest buffer descriptor word exceeds its shader snapshot");
            }
        }
        break;
    case Summary::Invalid:
        break;
    default:
        throw std::runtime_error("Metal guest buffer descriptor has invalid fetch metadata");
    }
    return resource.GetSharp(info);
}

GeometryBufferAccesses InspectGeometryBufferAccesses(const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    using namespace Shader::Optimization;
    GeometryBufferAccesses result;
    for (const auto* block : program.blocks) {
        for (const IR::Inst& inst : *block) {
            if (!IsBufferInstruction(inst)) {
                continue;
            }
            if (!inst.Arg(0).IsImmediate() || inst.Arg(0).Type() != IR::Type::U32) {
                throw std::runtime_error("Native geometry buffer handle was not resolved");
            }
            const u32 index = inst.Arg(0).U32();
            if (index >= program.info.buffers.size() || index >= Shader::NUM_BUFFERS) {
                throw std::runtime_error("Native geometry buffer metadata exceeds capacity");
            }
            if (program.info.buffers[index].buffer_type != Shader::BufferType::Guest) {
                continue;
            }
            if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
                result.scalar_reads.set(index);
                continue;
            }
            const bool atomic = IsBufferAtomic(inst);
            const bool write = IsBufferStore(inst);
            const bool read = !write || atomic;
            const auto flags = inst.Flags<IR::BufferInstInfo>();
            const bool coherent = flags.globally_coherent && flags.system_coherent;
            if (read) {
                result.vector_reads.set(index);
                if (coherent) {
                    result.coherent_reads.set(index);
                }
            }
            if (write) {
                result.vector_writes.set(index);
                if (coherent) {
                    result.coherent_writes.set(index);
                }
            }
            if (atomic) {
                result.atomics.set(index);
            }
        }
    }
    return result;
}

u64 InspectGeometryInputSources(const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    u64 sources{};
    std::vector<const IR::Inst*> pending;
    std::unordered_set<const IR::Inst*> visited;
    for (const auto* block : program.blocks) {
        for (const IR::Inst& inst : *block) {
            if (inst.MayHaveSideEffects()) {
                pending.push_back(&inst);
            }
        }
    }
    while (!pending.empty()) {
        const auto* inst = pending.back();
        pending.pop_back();
        if (!visited.insert(inst).second) {
            continue;
        }
        const auto opcode = inst->GetOpcode();
        if (opcode == IR::Opcode::GetAttribute || opcode == IR::Opcode::GetAttributeU32 ||
            opcode == IR::Opcode::GetAttributeU1) {
            if (opcode == IR::Opcode::GetAttributeU32 && inst->Arg(1).IsImmediate()) {
                const auto attribute = inst->Arg(0).Attribute();
                const u32 component = inst->Arg(1).U32();
                if (((attribute == IR::Attribute::LocalInvocationIndex ||
                      attribute == IR::Attribute::GeometryInitialActiveCount) &&
                     !component) ||
                    ((attribute == IR::Attribute::WorkgroupId ||
                      attribute == IR::Attribute::LocalInvocationId) &&
                     component < 3)) {
                    continue;
                }
            }
            sources |= GeometryAttributeSource(*inst);
        }
        for (size_t argument = 0; argument < inst->NumArgs(); ++argument) {
            if (const auto* operand = inst->Arg(argument).TryInst()) {
                pending.push_back(operand);
            }
        }
    }
    return sources;
}

void ValidateGeometryRingBuffers(const GeometryRingLayout& rings, const Shader::Info& exports,
                                 const GeometryBufferAddressOrigins& export_origins,
                                 const Shader::Info& geometry,
                                 const GeometryBufferAddressOrigins& geometry_origins,
                                 const Shader::Info& copy,
                                 const GeometryBufferAddressOrigins& copy_origins) {
    constexpr u64 vertex_offsets = ((u64{1} << 6) - 1) << u32(GeometryAddressSource::VertexOffset);
    const auto sharp = [](const Shader::Info& info, u32 index) {
        if (index >= info.buffers.size()) {
            throw std::runtime_error("Native geometry ring resource exceeds shader metadata");
        }
        const auto buffer = ReadGuestBufferSharp(info.buffers[index], info);
        if (!buffer.Valid() || !buffer.base_address || buffer.base_address % 4 ||
            !buffer.num_records) {
            throw std::runtime_error("Invalid native geometry ring buffer descriptor");
        }
        return buffer;
    };
    const auto linear = [](const AmdGpu::Buffer& buffer) {
        return !buffer.swizzle_enable && !buffer.add_tid_enable && !buffer.stride;
    };
    const auto swizzled = [](const AmdGpu::Buffer& buffer, u32 indices) {
        return buffer.swizzle_enable && buffer.add_tid_enable &&
               buffer.GetIndexStride() == indices && buffer.GetElementSize() == 4;
    };
    std::vector<VAddr> export_rings;
    std::vector<VAddr> geometry_rings;
    for (u32 index = 0; index < Shader::NUM_BUFFERS; ++index) {
        if (export_origins.writes[index] & SourceBit(GeometryAddressSource::ExportScalar)) {
            const auto buffer = sharp(exports, index);
            if (!swizzled(buffer, 64) ||
                (buffer.stride && buffer.stride != rings.export_wave_bytes / 64)) {
                throw std::runtime_error(
                    "Native ESGS export ring requires additional layout lowering");
            }
            export_rings.push_back(buffer.base_address);
        }
        if (geometry_origins.writes[index] & SourceBit(GeometryAddressSource::RingOffset)) {
            const auto buffer = sharp(geometry, index);
            if (!swizzled(buffer, 16) ||
                buffer.stride != u64{rings.stream_components[0]} * rings.max_vertices * 4) {
                throw std::runtime_error(
                    "Native GSVS output ring requires additional layout lowering");
            }
            geometry_rings.push_back(buffer.base_address);
        }
    }
    for (u32 index = 0; index < Shader::NUM_BUFFERS; ++index) {
        if (geometry_origins.reads[index] & vertex_offsets) {
            const auto buffer = sharp(geometry, index);
            if (!linear(buffer) ||
                std::ranges::find(export_rings, buffer.base_address) == export_rings.end()) {
                throw std::runtime_error(
                    "Native ESGS consumer ring does not match its export buffer");
            }
        }
        if (copy_origins.reads[index] & SourceBit(GeometryAddressSource::CopyVector)) {
            const auto buffer = sharp(copy, index);
            if (!linear(buffer) ||
                std::ranges::find(geometry_rings, buffer.base_address) == geometry_rings.end()) {
                throw std::runtime_error(
                    "Native GSVS copy ring does not match its geometry buffer");
            }
        }
    }
}

GeometryBufferAddressOrigins InspectGeometryBufferAddressOrigins(
    const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    using namespace Shader::Optimization;
    std::array<std::vector<IR::Value>, Shader::NUM_BUFFERS> reads;
    std::array<std::vector<IR::Value>, Shader::NUM_BUFFERS> writes;
    for (const auto* block : program.blocks) {
        for (const IR::Inst& inst : *block) {
            if (!IsBufferInstruction(inst)) {
                continue;
            }
            if (!inst.Arg(0).IsImmediate() || inst.Arg(0).Type() != IR::Type::U32 ||
                inst.Arg(0).U32() >= program.info.buffers.size() ||
                inst.Arg(0).U32() >= Shader::NUM_BUFFERS) {
                throw std::runtime_error("Native geometry address buffer handle was not resolved");
            }
            const u32 index = inst.Arg(0).U32();
            if (program.info.buffers[index].buffer_type != Shader::BufferType::Guest) {
                continue;
            }
            const bool atomic = IsBufferAtomic(inst);
            const bool write = IsBufferStore(inst);
            if (!write || atomic) {
                reads[index].push_back(inst.Arg(1));
            }
            if (write || atomic) {
                writes[index].push_back(inst.Arg(1));
            }
        }
    }
    GeometryBufferAddressOrigins result;
    for (u32 index = 0; index < Shader::NUM_BUFFERS; ++index) {
        result.reads[index] = AddressOrigins(std::move(reads[index]));
        result.writes[index] = AddressOrigins(std::move(writes[index]));
    }
    return result;
}

GeometryBufferSnapshot CaptureGeometryBufferSnapshot(
    const Shader::Info& info, const GeometryBufferAccesses& accesses,
    std::span<const GeometryBufferRange> buffers,
    std::span<const GuestBufferExtent> guest_extents) {
    if (info.buffers.size() > Shader::NUM_BUFFERS) {
        throw std::runtime_error("Native geometry buffer snapshot exceeds resource capacity");
    }
    GeometryBufferSnapshot snapshot{accesses, {}, {buffers.begin(), buffers.end()}};
    for (u32 index = 0; index < info.buffers.size(); ++index) {
        if (info.buffers[index].buffer_type == Shader::BufferType::Guest) {
            snapshot.guest_resources.set(index);
        }
    }
    ValidateGeometryBufferSnapshot(snapshot);
    for (const auto& buffer : buffers) {
        const auto sharp = ReadGuestBufferSharp(info.buffers[buffer.resource_index], info);
        const u64 descriptor_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
        const u64 requested_size =
            GuestBufferSize(buffer.resource_index, descriptor_size, guest_extents);
        if (buffer.size ? buffer.address != sharp.base_address || buffer.size > requested_size
                        : requested_size != 0) {
            throw std::runtime_error("Native geometry buffer snapshot differs from its binding");
        }
    }
    return snapshot;
}

std::vector<GeometryBufferLink> LinkGeometryBufferSnapshots(
    const GeometryBufferSnapshot& producer, const GeometryBufferSnapshot& consumer) {
    ValidateGeometryBufferSnapshot(producer);
    ValidateGeometryBufferSnapshot(consumer);
    const auto producer_reads = producer.accesses.vector_reads | producer.accesses.scalar_reads;
    const auto consumer_reads = consumer.accesses.vector_reads | consumer.accesses.scalar_reads;
    std::vector<GeometryBufferLink> links;
    for (const auto& source : producer.buffers) {
        if (!source.size) {
            continue;
        }
        const u32 source_index = source.resource_index;
        for (const auto& target : consumer.buffers) {
            if (!target.size) {
                continue;
            }
            const u32 target_index = target.resource_index;
            const bool raw = producer.accesses.vector_writes.test(source_index) &&
                             consumer_reads.test(target_index);
            const bool war = producer_reads.test(source_index) &&
                             consumer.accesses.vector_writes.test(target_index);
            const bool waw = producer.accesses.vector_writes.test(source_index) &&
                             consumer.accesses.vector_writes.test(target_index);
            if (!raw && !war && !waw) {
                continue;
            }
            const VAddr begin = std::max(source.address, target.address);
            const VAddr end = std::min(source.address + source.size, target.address + target.size);
            if (begin < end) {
                links.push_back({source_index, target_index, begin, end - begin, raw, war, waw});
            }
        }
    }
    return links;
}

} // namespace Metal
