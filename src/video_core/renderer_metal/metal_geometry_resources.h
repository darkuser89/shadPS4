// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <span>
#include <vector>
#include "shader_recompiler/resource.h"
#include "video_core/renderer_metal/metal_guest_ranges.h"

namespace Shader::IR {
struct Program;
}

namespace Shader {
struct Info;
}

namespace Metal {

struct GeometryRingLayout;

struct GeometryBufferAccesses {
    std::bitset<Shader::NUM_BUFFERS> vector_reads;
    std::bitset<Shader::NUM_BUFFERS> vector_writes;
    std::bitset<Shader::NUM_BUFFERS> atomics;
    std::bitset<Shader::NUM_BUFFERS> scalar_reads;
    std::bitset<Shader::NUM_BUFFERS> coherent_reads;
    std::bitset<Shader::NUM_BUFFERS> coherent_writes;

    bool operator==(const GeometryBufferAccesses&) const = default;
};

enum class GeometryAddressSource : u32 {
    ExportScalar = 0,
    ExportVector = 16,
    RingOffset = 20,
    WaveId = 21,
    VertexOffset = 22,
    PrimitiveId = 28,
    InvocationId = 29,
    CopyScalar = 30,
    CopyVector = 46,
    InputActive = 50,
    OtherInput = 51,
    MemoryData = 52,
    Phi = 53,
    BaseVertex = 54,
    BaseInstance = 55,
};

struct GeometryBufferAddressOrigins {
    std::array<u64, Shader::NUM_BUFFERS> reads{};
    std::array<u64, Shader::NUM_BUFFERS> writes{};

    bool operator==(const GeometryBufferAddressOrigins&) const = default;
};

struct GeometryBufferRange {
    u32 resource_index;
    VAddr address;
    u64 size;

    bool operator==(const GeometryBufferRange&) const = default;
};

struct GeometryBufferSnapshot {
    GeometryBufferAccesses accesses;
    std::bitset<Shader::NUM_BUFFERS> guest_resources;
    std::vector<GeometryBufferRange> buffers;
};

struct GeometryBufferLink {
    u32 producer_resource;
    u32 consumer_resource;
    VAddr address;
    u64 size;
    bool read_after_write;
    bool write_after_read;
    bool write_after_write;

    bool operator==(const GeometryBufferLink&) const = default;
};

[[nodiscard]] GeometryBufferAccesses InspectGeometryBufferAccesses(
    const Shader::IR::Program& program);

[[nodiscard]] GeometryBufferAddressOrigins InspectGeometryBufferAddressOrigins(
    const Shader::IR::Program& program);

[[nodiscard]] u64 InspectGeometryInputSources(const Shader::IR::Program& program);

void ValidateGeometryRingBuffers(const GeometryRingLayout& rings, const Shader::Info& exports,
                                 const GeometryBufferAddressOrigins& export_origins,
                                 const Shader::Info& geometry,
                                 const GeometryBufferAddressOrigins& geometry_origins,
                                 const Shader::Info& copy,
                                 const GeometryBufferAddressOrigins& copy_origins);

[[nodiscard]] AmdGpu::Buffer ReadGuestBufferSharp(const Shader::BufferResource& resource,
                                                  const Shader::Info& info);

[[nodiscard]] GeometryBufferSnapshot CaptureGeometryBufferSnapshot(
    const Shader::Info& info, const GeometryBufferAccesses& accesses,
    std::span<const GeometryBufferRange> buffers,
    std::span<const GuestBufferExtent> guest_extents = {});

[[nodiscard]] std::vector<GeometryBufferLink> LinkGeometryBufferSnapshots(
    const GeometryBufferSnapshot& producer, const GeometryBufferSnapshot& consumer);

} // namespace Metal
