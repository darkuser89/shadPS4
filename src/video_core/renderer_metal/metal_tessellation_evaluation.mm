// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tessellation_evaluation.h"

namespace Metal {
namespace {

struct GroupData {
    u32 domain_patch;
    u32 first_point;
    u32 relative_patch;
    u32 patch;
    u32 instance;
    std::array<u32, 16> system;
};
static_assert(sizeof(GroupData) == 84);

struct Parameters {
    u32 groups;
    u32 patches;
    u32 triangle;
    u32 raster;
    std::array<float, 4> viewport;
};
static_assert(sizeof(Parameters) == 32);

constexpr std::string_view EvaluationSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct DomainPoint { uint u; uint v; };
struct DomainPatch { uint point_count; uint index_count; uint instance; uint patch; };
struct GroupData {
    uint domain_patch; uint first_point; uint relative_patch;
    uint patch; uint instance; uint system[16];
};
struct Parameters { uint groups; uint patches; uint triangle; uint raster; float viewport[4]; };
struct RegisterInput { uint system[16]; uint vectors[4]; uint active; };
static_assert(sizeof(DomainPoint) == 8);
static_assert(sizeof(DomainPatch) == 16);
static_assert(sizeof(GroupData) == 84);
static_assert(sizeof(Parameters) == 32);
static_assert(sizeof(RegisterInput) == 84);
kernel void build_tessellation_evaluation_inputs(
    device const DomainPoint* points [[buffer(0)]],
    device const DomainPatch* patches [[buffer(1)]],
    device const GroupData* groups [[buffer(2)]],
    device RegisterInput* output [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]],
    device atomic_uint* source_errors [[buffer(5)]],
    constant Parameters& config [[buffer(6)]],
    device uint* group_map [[buffer(7)]],
    device atomic_uint* dispatch [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (group >= config.groups)
        return;
    threadgroup atomic_uint active;
    threadgroup atomic_uint group_errors;
    if (lane == 0) {
        atomic_store_explicit(&active, 0u, memory_order_relaxed);
        atomic_store_explicit(&group_errors, 0u, memory_order_relaxed);
        if (group == 0) {
            atomic_store_explicit(dispatch + 1, 1u, memory_order_relaxed);
            atomic_store_explicit(dispatch + 2, 1u, memory_order_relaxed);
            if (config.raster) {
                device float* viewport = reinterpret_cast<device float*>(
                    reinterpret_cast<device uint*>(output) + ulong(config.groups) * 64u * 21u + config.groups);
                for (uint word = 0; word < 4u; ++word)
                    viewport[word] = config.viewport[word];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const GroupData g = groups[group];
    RegisterInput result = {};
    for (uint word = 0; word < 16u; ++word) result.system[word] = g.system[word];
    uint error = 0;
    if (g.domain_patch != 0xffffffffu) {
        if (g.domain_patch >= config.patches) {
            error = 1u;
        } else if (atomic_load_explicit(source_errors, memory_order_relaxed)) {
            error = 2u;
        } else {
            const DomainPatch p = patches[g.domain_patch];
            if (p.point_count > 4225u || p.index_count > 24576u ||
                p.instance != g.instance || p.patch != g.patch) {
                error = 4u;
            } else {
                const ulong index = ulong(g.first_point) + lane;
                if (index < p.point_count) {
                    const DomainPoint point = points[ulong(g.domain_patch) * 4225u + index];
                    if (point.u > 65536u || point.v > 65536u ||
                        (config.triangle && point.u + point.v > 65536u)) {
                        error = 8u;
                    } else {
                        result.vectors[0] = as_type<uint>(float(point.u) * 0x1p-16f);
                        result.vectors[1] = as_type<uint>(float(point.v) * 0x1p-16f);
                        result.vectors[2] = g.relative_patch;
                        result.vectors[3] = p.patch;
                        result.active = 1u;
                    }
                }
            }
        }
    }
    output[ulong(group) * 64u + lane] = result;
    if (result.active)
        atomic_fetch_add_explicit(&active, 1u, memory_order_relaxed);
    if (error)
        atomic_fetch_or_explicit(&group_errors, error, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const uint failure = atomic_load_explicit(&group_errors, memory_order_relaxed);
        if (failure) {
            atomic_fetch_or_explicit(errors, failure, memory_order_relaxed);
        } else if (atomic_load_explicit(&active, memory_order_relaxed)) {
            const uint slot = atomic_fetch_add_explicit(dispatch, 1u, memory_order_relaxed);
            group_map[slot] = group;
        }
    }
}
)metal";

bool ValidAllocation(const std::shared_ptr<StagingBufferPool::Allocation>& allocation,
                     id<MTLDevice> device, u64 bytes) {
    return allocation && allocation->GetBuffer() && allocation->GetBuffer().device == device &&
           allocation->SizeBytes() >= bytes && allocation->GetOffset() % alignof(u32) == 0 &&
           allocation->GetOffset() <= allocation->GetBuffer().length &&
           allocation->SizeBytes() <= allocation->GetBuffer().length - allocation->GetOffset();
}

} // namespace

TessellationEvaluation::TessellationEvaluation(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

std::optional<TessellationEvaluation::Result> TessellationEvaluation::BuildInputs(
    const TessellationDomains::Result& domains, const TessellationWorkLayout& work,
    const TessellationExportRingLayout& ring, u32 relative_patch, u32 first_point,
    bool raster_output, const std::array<float, 4>& viewport,
    std::span<const std::array<u32, 16>> system_registers) {
    const auto config = work.KernelConfig().Evaluation();
    const auto device = instance.GetDevice();
    const u64 patches = work.PatchCount(0, config.workgroups);
    const u64 register_bytes = config.RegisterSizeBytes();
    const u64 input_bytes = register_bytes + u64{config.workgroups} * sizeof(u32) +
                            (raster_output ? sizeof(viewport) : 0);
    if (!config.workgroups || config.compact_evaluation || config.threads_per_group != 64 ||
        relative_patch >= 256 || first_point >= TessellationDomains::PointsPerPatch ||
        (!system_registers.empty() && system_registers.size() != config.workgroups) ||
        !domains.patch_count || domains.first_patch > patches ||
        domains.patch_count > patches - domains.first_patch ||
        input_bytes > device.maxBufferLength ||
        input_bytes / sizeof(u32) > std::numeric_limits<u32>::max() ||
        u64{config.workgroups} * sizeof(GroupData) > device.maxBufferLength ||
        config.raster_output ||
        (raster_output ? ring.wave_bytes || ring.size
                       : !ring.wave_bytes || ring.wave_bytes % 256 ||
                             ring.size > std::numeric_limits<u32>::max() ||
                             u64{config.workgroups} * ring.wave_bytes > ring.size) ||
        !ValidAllocation(domains.points, device,
                         u64{domains.patch_count} * TessellationDomains::PointsPerPatch *
                             sizeof(TessellationDomainPoint)) ||
        !ValidAllocation(domains.patches, device,
                         u64{domains.patch_count} * sizeof(TessellationDomainPatch)) ||
        !ValidAllocation(domains.errors, device, sizeof(u32))) {
        throw std::runtime_error("Metal tessellation GPU evaluation inputs exceed storage bounds");
    }
    std::vector<GroupData> groups;
    groups.reserve(config.workgroups);
    std::vector<GeometryExportInput> envelope(size_t(config.workgroups) * 64);
    bool enabled{};
    for (u32 index = 0; index < config.workgroups; ++index) {
        const auto group = work.Group(index);
        const u64 patch = work.PatchCount(0, index) + relative_patch;
        const bool selected = relative_patch < group.patch_count && patch >= domains.first_patch &&
                              patch - domains.first_patch < domains.patch_count;
        const u32 offset = u32(u64{index} * ring.wave_bytes);
        const u32 primitive = selected ? group.first_patch + relative_patch : 0;
        std::array<u32, 16> registers{};
        if (system_registers.empty()) {
            registers[0] = offset;
        } else {
            registers = system_registers[index];
        }
        groups.push_back(
            {selected ? u32(patch - domains.first_patch) : std::numeric_limits<u32>::max(),
             first_point, relative_patch, primitive, group.instance, registers});
        for (u32 lane = 0; lane < 64; ++lane) {
            auto& input = envelope[size_t(index) * 64 + lane];
            input.system_sgprs = registers;
            input.active = selected && lane != 0;
            if (input.active) {
                input.vgprs = {0x3f800000, 0x3f800000, relative_patch, primitive};
            }
        }
        enabled |= selected;
    }
    if (!enabled) {
        return std::nullopt;
    }
    if (!pipeline) {
        auto library = compiler.CompileLibrary(EvaluationSource, "metal4_tessellation_evaluation");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"build_tessellation_evaluation_inputs";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    Result result{
        std::make_shared<StagingBufferPool::Allocation>(
            staging.AllocatePrivate(input_bytes, alignof(u32))),
        std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
            sizeof(MTLDispatchThreadgroupsIndirectArguments), alignof(u32))),
        std::make_shared<StagingBufferPool::Allocation>(staging.AllocateReadback(sizeof(u32))),
        std::move(envelope)};
    auto group_data = staging.Allocate(groups.size() * sizeof(GroupData), alignof(GroupData));
    std::memcpy(group_data.UploadData().data(), groups.data(), groups.size() * sizeof(GroupData));
    const Parameters params{config.workgroups, domains.patch_count,
                            domains.config.tessellation.type == AmdGpu::TessellationType::Triangle,
                            raster_output, viewport};
    auto parameters = staging.Allocate(sizeof(params), alignof(Parameters));
    std::memcpy(parameters.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 9, 0, 0};
    bindings.SetBuffer(0, domains.points->GetBuffer(), domains.points->GetOffset());
    bindings.SetBuffer(1, domains.patches->GetBuffer(), domains.patches->GetOffset());
    bindings.SetBuffer(2, group_data.GetBuffer(), group_data.GetOffset());
    bindings.SetBuffer(3, result.registers->GetBuffer(), result.registers->GetOffset());
    bindings.SetBuffer(4, result.errors->GetBuffer(), result.errors->GetOffset());
    bindings.SetBuffer(5, domains.errors->GetBuffer(), domains.errors->GetOffset());
    bindings.SetBuffer(6, parameters.GetBuffer(), parameters.GetOffset());
    bindings.SetBuffer(7, result.registers->GetBuffer(),
                       result.registers->GetOffset() + register_bytes);
    bindings.SetBuffer(8, result.arguments->GetBuffer(), result.arguments->GetOffset());
    bindings.RetainUploadAllocation(domains.points);
    bindings.RetainUploadAllocation(domains.patches);
    bindings.RetainUploadAllocation(domains.errors);
    bindings.RetainUploadAllocation(result.registers);
    bindings.RetainUploadAllocation(result.arguments);
    bindings.RetainUploadAllocation(result.errors);
    bindings.RetainUploadAllocation(std::move(group_data));
    bindings.RetainUploadAllocation(std::move(parameters));
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.FillBuffer(
        {result.errors->GetBuffer(), result.errors->GetOffset(), result.errors->SizeBytes()}, 0);
    commands.FillBuffer({result.arguments->GetBuffer(), result.arguments->GetOffset(),
                         result.arguments->SizeBytes()},
                        0);
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake(config.workgroups, 1, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return result;
}

void TessellationEvaluation::Validate(const Result& result) const {
    if (!ValidAllocation(result.errors, instance.GetDevice(), sizeof(u32)) ||
        !result.errors->IsReady()) {
        throw std::runtime_error("Metal tessellation GPU input validation requires completed work");
    }
    u32 errors;
    std::memcpy(&errors, result.errors->ReadbackData().data(), sizeof(errors));
    if (errors) {
        throw std::runtime_error("Metal tessellation GPU evaluation input validation failed");
    }
}

} // namespace Metal
