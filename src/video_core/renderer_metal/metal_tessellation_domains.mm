// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tessellation_domains.h"
#include "video_core/renderer_metal/metal_tessellator_source.h"

namespace Metal {
namespace {

constexpr u64 PointBytes = TessellationDomains::PointsPerPatch * sizeof(TessellationDomainPoint);
constexpr u64 IndexBytes = TessellationDomains::IndicesPerPatch * sizeof(u32);
constexpr u64 BatchBytes = 32 * 1024 * 1024;

struct Parameters {
    u32 type;
    u32 partitioning;
    u32 topology;
    u32 patches;
    float minimum_factor;
    float maximum_factor;
};
static_assert(sizeof(Parameters) == 24);

void ValidateConfig(const TessellationDomainConfig& domain) {
    const auto config = domain.tessellation;
    if (!std::isfinite(domain.minimum_factor) || !std::isfinite(domain.maximum_factor) ||
        domain.minimum_factor < 0.0f || domain.minimum_factor > domain.maximum_factor ||
        domain.maximum_factor > 64.0f) {
        throw std::runtime_error("Invalid Metal tessellation factor clamp");
    }
    if (config.type != AmdGpu::TessellationType::Isoline &&
        config.type != AmdGpu::TessellationType::Triangle &&
        config.type != AmdGpu::TessellationType::Quad) {
        throw std::runtime_error("Invalid Metal tessellation domain");
    }
    if (config.partitioning != AmdGpu::TessellationPartitioning::Integer &&
        config.partitioning != AmdGpu::TessellationPartitioning::Pow2 &&
        config.partitioning != AmdGpu::TessellationPartitioning::FracOdd &&
        config.partitioning != AmdGpu::TessellationPartitioning::FracEven) {
        throw std::runtime_error("Invalid Metal tessellation partitioning");
    }
    if (config.topology == AmdGpu::TessellationTopology::Point) {
        return;
    }
    if (config.type == AmdGpu::TessellationType::Isoline &&
        config.topology == AmdGpu::TessellationTopology::Line) {
        return;
    }
    if (config.type != AmdGpu::TessellationType::Isoline &&
        (config.topology == AmdGpu::TessellationTopology::TriangleCw ||
         config.topology == AmdGpu::TessellationTopology::TriangleCcw)) {
        return;
    }
    throw std::runtime_error("Unsupported Metal tessellation domain/topology combination");
}

bool ValidAllocation(const std::shared_ptr<StagingBufferPool::Allocation>& allocation,
                     id<MTLDevice> device, u64 size) {
    return allocation && allocation->GetBuffer() && allocation->GetBuffer().device == device &&
           allocation->SizeBytes() >= size && !(allocation->GetOffset() % alignof(u32)) &&
           allocation->GetOffset() <= allocation->GetBuffer().length &&
           allocation->SizeBytes() <= allocation->GetBuffer().length - allocation->GetOffset();
}

} // namespace

TessellationDomainConfig BuildTessellationDomainConfig(const AmdGpu::Regs& regs) {
    constexpr u32 MaxFactorRegister = 0x28a18 / sizeof(u32);
    constexpr u32 MinFactorRegister = 0x28a1c / sizeof(u32);
    constexpr u32 ParameterRegister = 0x28b6c / sizeof(u32);
    constexpr u32 SupportedParameters = 0xff | (0xf << 10) | (0x3 << 15);
    const u32 parameter = regs.reg_array[ParameterRegister];
    if ((parameter & ~SupportedParameters) || ((parameter >> 15) & 3) == 3) {
        throw std::runtime_error("Metal tessellation walking/control flags require lowering");
    }
    TessellationDomainConfig result{
        .tessellation = {AmdGpu::TessellationType(parameter & 3),
                         AmdGpu::TessellationPartitioning((parameter >> 2) & 7),
                         AmdGpu::TessellationTopology((parameter >> 5) & 7)},
        .minimum_factor = std::bit_cast<float>(regs.reg_array[MinFactorRegister]),
        .maximum_factor = std::bit_cast<float>(regs.reg_array[MaxFactorRegister]),
    };
    ValidateConfig(result);
    return result;
}

TessellationDomains::TessellationDomains(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

u32 TessellationDomains::MaxBatchPatches() const {
    const u64 capacity = instance.GetDevice().maxBufferLength;
    return std::min(
        {capacity / PointBytes, capacity / IndexBytes, capacity / sizeof(TessellationDomainPatch),
         (BatchBytes - sizeof(u32)) / (PointBytes + IndexBytes + sizeof(TessellationDomainPatch))});
}

TessellationDomains::Result TessellationDomains::Generate(
    const TessellationDomainConfig& config, const TessellationFactors::Result& factors,
    u32 first_patch, u32 patch_count) {
    ValidateConfig(config);
    const auto device = instance.GetDevice();
    if (config.tessellation.type != factors.type || !patch_count ||
        patch_count > MaxBatchPatches() || first_patch > factors.patch_count ||
        patch_count > factors.patch_count - first_patch ||
        !ValidAllocation(factors.storage, device,
                         u64{factors.patch_count} * sizeof(TessellationPatchFactors)) ||
        !ValidAllocation(factors.errors, device, sizeof(u32))) {
        throw std::runtime_error("Metal tessellation domain batch exceeds storage bounds");
    }
    if (!pipeline) {
        auto library = compiler.CompileLibrary(TessellatorSource, "metal4_tessellator");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"generate_tessellation_domains";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(32, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 32;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    Result result{
        .points = std::make_shared<StagingBufferPool::Allocation>(
            staging.AllocatePrivate(u64{patch_count} * PointBytes, alignof(u32))),
        .indices = std::make_shared<StagingBufferPool::Allocation>(
            staging.AllocatePrivate(u64{patch_count} * IndexBytes, alignof(u32))),
        .patches = std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
            u64{patch_count} * sizeof(TessellationDomainPatch), alignof(u32))),
        .errors =
            std::make_shared<StagingBufferPool::Allocation>(staging.AllocateReadback(sizeof(u32))),
        .factors = factors.storage,
        .config = config,
        .first_patch = first_patch,
        .patch_count = patch_count,
    };
    const Parameters params{u32(config.tessellation.type),
                            u32(config.tessellation.partitioning),
                            u32(config.tessellation.topology),
                            patch_count,
                            config.minimum_factor,
                            config.maximum_factor};
    auto parameters = staging.Allocate(sizeof(params), alignof(Parameters));
    std::memcpy(parameters.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 7, 0, 0};
    bindings.SetBuffer(0, factors.storage->GetBuffer(),
                       factors.storage->GetOffset() +
                           u64{first_patch} * sizeof(TessellationPatchFactors));
    bindings.SetBuffer(1, result.points->GetBuffer(), result.points->GetOffset());
    bindings.SetBuffer(2, result.indices->GetBuffer(), result.indices->GetOffset());
    bindings.SetBuffer(3, result.patches->GetBuffer(), result.patches->GetOffset());
    bindings.SetBuffer(4, result.errors->GetBuffer(), result.errors->GetOffset());
    bindings.SetBuffer(5, parameters.GetBuffer(), parameters.GetOffset());
    bindings.SetBuffer(6, factors.errors->GetBuffer(), factors.errors->GetOffset());
    bindings.RetainUploadAllocation(factors.storage);
    bindings.RetainUploadAllocation(factors.errors);
    bindings.RetainUploadAllocation(result.points);
    bindings.RetainUploadAllocation(result.indices);
    bindings.RetainUploadAllocation(result.patches);
    bindings.RetainUploadAllocation(result.errors);
    bindings.RetainUploadAllocation(std::move(parameters));
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.FillBuffer(
        {result.errors->GetBuffer(), result.errors->GetOffset(), result.errors->SizeBytes()}, 0);
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake((patch_count + 31) / 32, 1, 1), MTLSizeMake(32, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    return result;
}

void TessellationDomains::Validate(const Result& result) const {
    if (!ValidAllocation(result.errors, instance.GetDevice(), sizeof(u32)) ||
        !result.errors->IsReady()) {
        throw std::runtime_error(
            "Metal tessellation domain validation requires completed GPU work");
    }
    u32 errors;
    std::memcpy(&errors, result.errors->ReadbackData().data(), sizeof(errors));
    if (errors) {
        throw std::runtime_error("Metal tessellation domain factor or output validation failed");
    }
}

} // namespace Metal
