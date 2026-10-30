// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tessellation_factors.h"

namespace Metal {
namespace {

struct GroupData {
    u64 source_offset;
    u32 patch_count;
    u32 output_offset;
    u32 instance;
    u32 first_patch;
};
static_assert(sizeof(GroupData) == 24);

struct Parameters {
    u32 type;
    u32 groups;
};
static_assert(sizeof(Parameters) == 8);

constexpr std::string_view FactorSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct GroupData {
    ulong source_offset;
    uint patch_count;
    uint output_offset;
    uint instance;
    uint first_patch;
};
struct PatchFactors { uint outer[4]; uint inner[2]; uint instance; uint patch; };
struct Parameters { uint type; uint groups; };
static_assert(sizeof(GroupData) == 24);
static_assert(sizeof(PatchFactors) == 32);
static_assert(sizeof(Parameters) == 8);
uint load_word(device const uchar* bytes, ulong offset) {
    return uint(bytes[offset]) | (uint(bytes[offset + 1]) << 8) |
           (uint(bytes[offset + 2]) << 16) | (uint(bytes[offset + 3]) << 24);
}
kernel void decode_tessellation_factors(device const uchar* source [[buffer(0)]],
                                        device const GroupData* groups [[buffer(1)]],
                                        device PatchFactors* output [[buffer(2)]],
                                        device atomic_uint* errors [[buffer(3)]],
                                        constant Parameters& p [[buffer(4)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint2 grid [[threadgroups_per_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
    const ulong index = ulong(group.y) * grid.x + group.x;
    if (index >= p.groups)
        return;
    const GroupData g = groups[index];
    for (uint local = lane; local < g.patch_count; local += 64u) {
        PatchFactors result = {};
        result.instance = g.instance;
        result.patch = g.first_patch + local;
        if (load_word(source, g.source_offset) != 0x80000000u) {
            atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
        } else {
            const uint components = 2u + 2u * p.type;
            const ulong offset = g.source_offset + 4u + ulong(local) * components * 4u;
            if (p.type == 0) {
                result.outer[0] = load_word(source, offset + 4);
                result.outer[1] = load_word(source, offset);
            } else {
                const uint outer_count = p.type == 1 ? 3u : 4u;
                for (uint i = 0; i < outer_count; ++i)
                    result.outer[i] = load_word(source, offset + i * 4u);
                for (uint i = outer_count; i < components; ++i)
                    result.inner[i - outer_count] = load_word(source, offset + i * 4u);
            }
        }
        output[g.output_offset + local] = result;
    }
}
)metal";

} // namespace

TessellationFactors::TessellationFactors(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

u32 TessellationFactors::Components(AmdGpu::TessellationType type) {
    switch (type) {
    case AmdGpu::TessellationType::Isoline:
        return 2;
    case AmdGpu::TessellationType::Triangle:
        return 4;
    case AmdGpu::TessellationType::Quad:
        return 6;
    default:
        throw std::runtime_error("Invalid Metal tessellation factor domain");
    }
}

TessellationFactors::Result TessellationFactors::Decode(
    AmdGpu::TessellationType type, const std::shared_ptr<StagingBufferPool::Allocation>& source,
    std::span<const Group> groups) {
    const u32 components = Components(type);
    const u64 capacity = instance.GetDevice().maxBufferLength;
    if (!source || !source->GetBuffer() || source->GetBuffer().device != instance.GetDevice() ||
        groups.empty() || groups.size() > std::numeric_limits<u32>::max() ||
        groups.size() > capacity / sizeof(GroupData)) {
        throw std::runtime_error("Invalid Metal tessellation factor snapshot");
    }
    u64 patches{};
    std::vector<GroupData> data;
    data.reserve(groups.size());
    for (const auto& group : groups) {
        const u64 bytes = (u64{group.patch_count} * components + 1) * sizeof(u32);
        if (!group.patch_count || group.patch_count > 256 ||
            group.source_offset > source->SizeBytes() ||
            bytes > source->SizeBytes() - group.source_offset ||
            group.patch_count - 1 > std::numeric_limits<u32>::max() - group.first_patch ||
            patches + group.patch_count > std::numeric_limits<u32>::max() ||
            patches + group.patch_count > capacity / sizeof(TessellationPatchFactors)) {
            throw std::runtime_error("Metal tessellation factor group exceeds storage bounds");
        }
        data.push_back({group.source_offset, group.patch_count, u32(patches), group.instance,
                        group.first_patch});
        patches += group.patch_count;
    }
    if (!pipeline) {
        auto library = compiler.CompileLibrary(FactorSource, "metal4_tessellation_factors");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"decode_tessellation_factors";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    Result result{
        std::make_shared<StagingBufferPool::Allocation>(
            staging.AllocatePrivate(patches * sizeof(TessellationPatchFactors), alignof(u32))),
        std::make_shared<StagingBufferPool::Allocation>(staging.AllocateReadback(sizeof(u32))),
        u32(patches), type};
    auto group_data = staging.Allocate(data.size() * sizeof(GroupData), alignof(GroupData));
    std::memcpy(group_data.UploadData().data(), data.data(), data.size() * sizeof(GroupData));
    const Parameters params{u32(type), u32(groups.size())};
    auto parameters = staging.Allocate(sizeof(params), alignof(Parameters));
    std::memcpy(parameters.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 5, 0, 0};
    bindings.SetBuffer(0, source->GetBuffer(), source->GetOffset());
    bindings.SetBuffer(1, group_data.GetBuffer(), group_data.GetOffset());
    bindings.SetBuffer(2, result.storage->GetBuffer(), result.storage->GetOffset());
    bindings.SetBuffer(3, result.errors->GetBuffer(), result.errors->GetOffset());
    bindings.SetBuffer(4, parameters.GetBuffer(), parameters.GetOffset());
    bindings.RetainUploadAllocation(source);
    bindings.RetainUploadAllocation(result.storage);
    bindings.RetainUploadAllocation(result.errors);
    bindings.RetainUploadAllocation(std::move(group_data));
    bindings.RetainUploadAllocation(std::move(parameters));
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.FillBuffer(
        {result.errors->GetBuffer(), result.errors->GetOffset(), result.errors->SizeBytes()}, 0);
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageDispatch);
    const u64 width = std::min<u64>(groups.size(), 65535);
    commands.Dispatch(MTLSizeMake(width, (groups.size() + width - 1) / width, 1),
                      MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return result;
}

void TessellationFactors::Validate(const Result& result) const {
    if (!result.errors || !result.errors->IsReady()) {
        throw std::runtime_error(
            "Metal tessellation factor validation requires completed GPU work");
    }
    u32 errors;
    std::memcpy(&errors, result.errors->ReadbackData().data(), sizeof(errors));
    if (errors) {
        throw std::runtime_error("Metal tessellation control word requires additional decoding");
    }
}

} // namespace Metal
