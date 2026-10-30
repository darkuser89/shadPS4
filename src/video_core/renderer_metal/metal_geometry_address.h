// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include "common/types.h"

namespace Shader::IR {
enum class Attribute : u64;
enum class Opcode;
struct Program;
class Value;
} // namespace Shader::IR

namespace Metal {

struct GeometryAddressInterval {
    u32 lower{};
    u32 upper{std::numeric_limits<u32>::max()};

    bool operator==(const GeometryAddressInterval&) const = default;
};

struct GeometryAddressInput {
    Shader::IR::Attribute attribute;
    u32 component;
    GeometryAddressInterval values;
};

struct GeometryBufferByteInterval {
    u64 lower;
    u64 upper;

    bool operator==(const GeometryBufferByteInterval&) const = default;
};

struct GeometryBufferAddressBounds {
    u32 resource_index;
    bool read;
    bool write;
    std::optional<GeometryBufferByteInterval> bytes;
    bool reachable{true};
    u32 address_shift{};
};

struct GeometryBufferAddressBinding {
    u32 resource_index;
    VAddr address;
    u64 size;
    u64 offset;
    u64 buffer_size;
    bool paged{};
};

enum class GeometryAddressCoverage : u32 {
    Inactive,
    Unbound,
    Empty,
    Unknown,
    Contained,
    Partial,
    Disjoint,
};

struct GeometryBufferAddressMapping {
    u32 access_index;
    u32 resource_index;
    bool read;
    bool write;
    GeometryAddressCoverage coverage;
    std::optional<GeometryBufferByteInterval> guest_bytes;
};

class GeometryBufferAddressProgram {
public:
    [[nodiscard]] std::vector<GeometryBufferAddressBounds> Evaluate(
        std::span<const GeometryAddressInput> inputs, std::span<const u32> user_data) const;
    [[nodiscard]] GeometryBufferAddressProgram Select(std::span<const u32> indices) const;

private:
    class Builder;
    class Evaluator;

    enum class Kind : u8 { Unknown, Constant, Attribute, UserData, Operation, Merge };

    struct Node {
        Shader::IR::Opcode opcode{};
        Kind kind{};
        u8 bits{32};
        u32 num_args{};
        u32 first_argument{};
        u64 key{};
        u32 component{};
    };

    struct Access {
        u32 resource_index;
        u32 node;
        u32 shift;
        u32 byte_size;
        u32 predicate;
        u32 bounds_predicate;
        bool read;
        bool write;
        u8 address_bits{32};
    };

    std::vector<Node> nodes;
    std::vector<u32> arguments;
    std::vector<Access> accesses;

    friend GeometryBufferAddressProgram InspectGeometryBufferAddressProgram(
        const Shader::IR::Program& program);
    friend GeometryBufferAddressProgram InspectGuestBufferBindingProgram(
        const Shader::IR::Program& program);
    friend GeometryBufferAddressProgram InspectScalarConstantAddressProgram(
        std::span<const Shader::IR::Value> addresses,
        std::span<const Shader::IR::Value> predicates);
};

[[nodiscard]] GeometryBufferAddressProgram InspectGeometryBufferAddressProgram(
    const Shader::IR::Program& program);

[[nodiscard]] GeometryBufferAddressProgram InspectGuestBufferBindingProgram(
    const Shader::IR::Program& program);

[[nodiscard]] GeometryBufferAddressProgram InspectScalarConstantAddressProgram(
    std::span<const Shader::IR::Value> addresses, std::span<const Shader::IR::Value> predicates);

[[nodiscard]] std::vector<GeometryBufferAddressMapping> MapGeometryBufferAddresses(
    std::span<const GeometryBufferAddressBounds> bounds,
    std::span<const GeometryBufferAddressBinding> bindings);

} // namespace Metal
