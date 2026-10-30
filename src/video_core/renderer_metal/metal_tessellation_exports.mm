// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tessellation_exports.h"

namespace Metal {
namespace {

constexpr u64 BatchBytes = 32 * 1024 * 1024;

u32 PrimitiveWidth(AmdGpu::TessellationTopology topology) {
    switch (topology) {
    case AmdGpu::TessellationTopology::Point:
        return 1;
    case AmdGpu::TessellationTopology::Line:
        return 2;
    case AmdGpu::TessellationTopology::TriangleCw:
    case AmdGpu::TessellationTopology::TriangleCcw:
        return 3;
    default:
        throw std::runtime_error("Metal tessellation primitive topology requires lowering");
    }
}

u32 PrimitiveSlots(u32 width) {
    return width == 1 ? TessellationDomains::PointsPerPatch
                      : TessellationDomains::IndicesPerPatch / width;
}

bool ValidAllocation(const std::shared_ptr<StagingBufferPool::Allocation>& allocation,
                     id<MTLDevice> device, u64 bytes) {
    return allocation && allocation->GetBuffer() && allocation->GetBuffer().device == device &&
           allocation->SizeBytes() >= bytes && allocation->GetOffset() % alignof(u32) == 0 &&
           allocation->GetOffset() <= allocation->GetBuffer().length &&
           allocation->SizeBytes() <= allocation->GetBuffer().length - allocation->GetOffset();
}

struct Parameters {
    u32 patches;
    u32 primitive_width;
    u32 primitive_slots;
    u32 vertex_words;
    u32 raster;
};
static_assert(sizeof(Parameters) == 20);

constexpr std::string_view PrimitiveSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct DomainPatch { uint point_count; uint index_count; uint instance; uint patch; };
struct Primitive { uint vertex_offsets[3]; uint patch; uint instance; uint active; };
struct Parameters { uint patches; uint primitive_width; uint primitive_slots; uint vertex_words; uint raster; };
static_assert(sizeof(DomainPatch) == 16);
static_assert(sizeof(Primitive) == 24);
static_assert(sizeof(Parameters) == 20);
kernel void assemble_tessellation_exports(
    device const uint* indices [[buffer(0)]],
    device const DomainPatch* patches [[buffer(1)]],
    device Primitive* primitives [[buffer(2)]],
    device atomic_uint& errors [[buffer(3)]],
    device atomic_uint& source_errors [[buffer(4)]],
    constant Parameters& p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong item = (ulong(group.y) * grid.x + group.x) * 64u + lane;
    if (item >= ulong(p.patches) * p.primitive_slots)
        return;
    const uint patch_index = uint(item / p.primitive_slots);
    const uint primitive_index = uint(item % p.primitive_slots);
    const DomainPatch patch = patches[patch_index];
    Primitive result = {};
    uint error = 0;
    if (atomic_load_explicit(&source_errors, memory_order_relaxed)) {
        error = 1u;
    } else if (patch.point_count > 4225u || patch.index_count > 24576u ||
               patch.index_count % p.primitive_width ||
               patch.index_count / p.primitive_width > p.primitive_slots) {
        error = 2u;
    } else if (primitive_index < patch.index_count / p.primitive_width) {
        for (uint slot = 0; slot < p.primitive_width; ++slot) {
            const uint point = indices[ulong(patch_index) * 24576u +
                                       primitive_index * p.primitive_width + slot];
            if (point >= patch.point_count) {
                error = 4u;
            } else {
                result.vertex_offsets[slot] =
                    p.raster ? (patch_index * 4288u + point) * p.vertex_words
                             : (patch_index * 67u + point / 64u) * p.vertex_words * 64u + point % 64u;
            }
        }
        result.patch = patch.patch;
        result.instance = patch.instance;
        result.active = error ? 0u : 1u;
    }
    if (error) {
        result = {};
        atomic_fetch_or_explicit(&errors, error, memory_order_relaxed);
    }
    primitives[item] = result;
}
)metal";

} // namespace

TessellationExports::TessellationExports(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

u32 TessellationExports::MaxBatchPatches(u32 vertex_words,
                                         AmdGpu::TessellationTopology topology) const {
    const u32 width = PrimitiveWidth(topology);
    if (!vertex_words || vertex_words >= (1u << 15)) {
        throw std::runtime_error("Metal tessellation export vertex size exceeds its ring ABI");
    }
    const u64 export_bytes = u64{VerticesPerPatch} * vertex_words * sizeof(u32);
    const u64 primitive_bytes = u64{PrimitiveSlots(width)} * sizeof(TessellationPrimitive);
    const u64 capacity = instance.GetDevice().maxBufferLength;
    return u32(std::min({capacity / export_bytes, capacity / primitive_bytes,
                         (BatchBytes - sizeof(u32)) / (export_bytes + primitive_bytes),
                         u64{std::numeric_limits<u32>::max()} / export_bytes}));
}

TessellationExports::Batch TessellationExports::Allocate(const TessellationDomains::Result& domains,
                                                         const TessellationWorkLayout& work,
                                                         const TessellationExportRingLayout& ring) {
    const auto device = instance.GetDevice();
    const auto config = work.KernelConfig().Evaluation();
    const u64 patches = work.PatchCount(0, config.workgroups);
    if (!ring.wave_bytes || ring.wave_bytes % 256 || !config.workgroups ||
        config.threads_per_group != 64 || config.compact_evaluation ||
        u64{config.workgroups} * ring.wave_bytes > ring.size || !domains.patch_count ||
        domains.first_patch > patches || domains.patch_count > patches - domains.first_patch ||
        domains.patch_count >
            MaxBatchPatches(ring.wave_bytes / 256, domains.config.tessellation.topology) ||
        !ValidAllocation(domains.indices, device,
                         u64{domains.patch_count} * TessellationDomains::IndicesPerPatch *
                             sizeof(u32)) ||
        !ValidAllocation(domains.patches, device,
                         u64{domains.patch_count} * sizeof(TessellationDomainPatch)) ||
        !ValidAllocation(domains.errors, device, sizeof(u32))) {
        throw std::runtime_error("Metal tessellation retained exports exceed their batch bounds");
    }
    const u32 width = PrimitiveWidth(domains.config.tessellation.topology);
    Batch batch{*this, domains, work, ring.wave_bytes / 256, width, PrimitiveSlots(width)};
    batch.exports = std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
        u64{domains.patch_count} * WavesPerPatch * ring.wave_bytes, alignof(u32)));
    batch.primitives = std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(
        u64{domains.patch_count} * batch.primitive_slots * sizeof(TessellationPrimitive),
        alignof(u32)));
    batch.errors =
        std::make_shared<StagingBufferPool::Allocation>(staging.AllocateReadback(sizeof(u32)));
    return batch;
}

TessellationExports::Batch TessellationExports::AllocateRaster(
    const TessellationDomains::Result& domains, const TessellationWorkLayout& work,
    u32 vertex_words, std::span<const GeometryOutputAttribute> attributes) {
    if (vertex_words < 5 || vertex_words >= (1u << 15) || attributes.size() + 1 != vertex_words) {
        throw std::runtime_error("Invalid native TES raster record size");
    }
    for (size_t index = 0; index < attributes.size(); ++index) {
        if (attributes[index].stream || attributes[index].word != index + 1) {
            throw std::runtime_error(
                "Native TES raster attributes do not match their record words");
        }
    }
    const u32 bytes = vertex_words * 256;
    auto batch = Allocate(domains, work, {bytes, u64{work.KernelConfig().workgroups} * bytes});
    batch.raster = true;
    batch.raster_attributes.assign(attributes.begin(), attributes.end());
    return batch;
}

void TessellationExports::Capture(Batch& batch, u32 relative_patch, u32 first_point,
                                  const std::shared_ptr<StagingBufferPool::Allocation>& snapshot,
                                  std::span<const u64> group_offsets,
                                  const GeometryBufferRange& guest_ring) {
    CaptureImpl(batch, relative_patch, first_point, snapshot, group_offsets, guest_ring);
}

void TessellationExports::CaptureRaster(
    Batch& batch, u32 relative_patch, u32 first_point,
    const std::shared_ptr<StagingBufferPool::Allocation>& snapshot) {
    std::vector<u64> offsets(batch.work.KernelConfig().workgroups);
    for (u32 index = 0; index < offsets.size(); ++index) {
        offsets[index] = u64{index} * batch.vertex_words * 256;
    }
    CaptureImpl(batch, relative_patch, first_point, snapshot, offsets, std::nullopt);
}

void TessellationExports::CaptureImpl(
    Batch& batch, u32 relative_patch, u32 first_point,
    const std::shared_ptr<StagingBufferPool::Allocation>& snapshot,
    std::span<const u64> group_offsets, const std::optional<GeometryBufferRange>& guest_ring) {
    const auto& work = batch.work;
    const u64 wave_bytes = u64{batch.vertex_words} * 256;
    if (batch.owner != this || batch.assembled || relative_patch >= 256 || first_point % 64 ||
        batch.raster != !guest_ring ||
        (guest_ring &&
         (guest_ring->resource_index >= Shader::NUM_BUFFERS || !guest_ring->address ||
          guest_ring->address >= (u64{1} << 40) || guest_ring->address % sizeof(u32) ||
          guest_ring->size > (u64{1} << 40) - guest_ring->address ||
          u64{group_offsets.size()} * wave_bytes > guest_ring->size ||
          (batch.guest_ring && batch.guest_ring != guest_ring))) ||
        first_point >= TessellationDomains::PointsPerPatch ||
        batch.captured.size() != batch.domains.patch_count ||
        !ValidAllocation(batch.exports, instance.GetDevice(),
                         u64{batch.domains.patch_count} * WavesPerPatch * wave_bytes) ||
        group_offsets.size() != work.KernelConfig().workgroups ||
        !ValidAllocation(snapshot, instance.GetDevice(), u64{group_offsets.size()} * wave_bytes)) {
        throw std::runtime_error("Invalid Metal tessellation export capture");
    }
    for (u32 group = 0; group < group_offsets.size(); ++group) {
        if (group_offsets[group] != u64{group} * wave_bytes) {
            throw std::runtime_error("Metal tessellation snapshot group order is inconsistent");
        }
    }
    const u32 wave = first_point / 64;
    const u64 bit = u64{1} << (wave % 64);
    struct Copy {
        u64 source;
        u64 destination;
        u32 patch;
    };
    std::vector<Copy> copies;
    for (u32 group = 0; group < group_offsets.size(); ++group) {
        if (relative_patch >= work.Group(group).patch_count) {
            continue;
        }
        const u64 patch = work.PatchCount(0, group) + relative_patch;
        if (patch < batch.domains.first_patch ||
            patch - batch.domains.first_patch >= batch.domains.patch_count) {
            continue;
        }
        const u32 local = u32(patch - batch.domains.first_patch);
        if (batch.captured[local][wave / 64] & bit || group_offsets[group] % sizeof(u32) ||
            group_offsets[group] > snapshot->SizeBytes() ||
            wave_bytes > snapshot->SizeBytes() - group_offsets[group]) {
            throw std::runtime_error(
                "Metal tessellation export capture overlaps or leaves snapshot");
        }
        const u64 destination = (u64{local} * WavesPerPatch + wave) * wave_bytes;
        if (snapshot->GetBuffer() == batch.exports->GetBuffer() &&
            snapshot->GetOffset() + group_offsets[group] <
                batch.exports->GetOffset() + destination + wave_bytes &&
            batch.exports->GetOffset() + destination <
                snapshot->GetOffset() + group_offsets[group] + wave_bytes) {
            throw std::runtime_error("Metal tessellation export capture aliases its destination");
        }
        copies.push_back({group_offsets[group], destination, local});
    }
    if (copies.empty()) {
        throw std::runtime_error("Metal tessellation export capture has no matching patch");
    }
    scheduler.RetainLifetime(snapshot);
    scheduler.RetainLifetime(batch.exports);
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    for (const auto& copy : copies) {
        commands.CopyBuffer(
            {snapshot->GetBuffer(), snapshot->GetOffset() + copy.source, wave_bytes},
            {batch.exports->GetBuffer(), batch.exports->GetOffset() + copy.destination, wave_bytes},
            wave_bytes);
        batch.captured[copy.patch][wave / 64] |= bit;
    }
    batch.guest_ring = guest_ring;
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
}

void TessellationExports::Assemble(Batch& batch) {
    static_assert(WavesPerPatch == 67);
    if (batch.owner != this || batch.assembled || !batch.domains.patch_count ||
        (!batch.raster && !batch.guest_ring) ||
        batch.captured.size() != batch.domains.patch_count ||
        !ValidAllocation(batch.primitives, instance.GetDevice(),
                         u64{batch.domains.patch_count} * batch.primitive_slots *
                             sizeof(TessellationPrimitive)) ||
        !ValidAllocation(batch.errors, instance.GetDevice(), sizeof(u32)) ||
        std::ranges::any_of(batch.captured, [](const auto& mask) {
            return mask[0] != std::numeric_limits<u64>::max() || mask[1] != 7;
        })) {
        throw std::runtime_error("Metal tessellation primitives require every export chunk");
    }
    if (!pipeline) {
        auto library = compiler.CompileLibrary(PrimitiveSource, "metal4_tessellation_exports");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"assemble_tessellation_exports";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    const Parameters params{batch.domains.patch_count, batch.primitive_width, batch.primitive_slots,
                            batch.vertex_words, batch.raster};
    auto parameters = staging.Allocate(sizeof(params), alignof(Parameters));
    std::memcpy(parameters.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 6, 0, 0};
    const std::array allocations{batch.domains.indices, batch.domains.patches, batch.primitives,
                                 batch.errors, batch.domains.errors};
    for (u32 index = 0; index < allocations.size(); ++index) {
        bindings.SetBuffer(index, allocations[index]->GetBuffer(), allocations[index]->GetOffset());
        bindings.RetainUploadAllocation(allocations[index]);
    }
    bindings.SetBuffer(5, parameters.GetBuffer(), parameters.GetOffset());
    bindings.RetainUploadAllocation(std::move(parameters));
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    commands.FillBuffer(
        {batch.errors->GetBuffer(), batch.errors->GetOffset(), batch.errors->SizeBytes()}, 0);
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageDispatch);
    const u64 groups = (u64{params.patches} * params.primitive_slots + 63) / 64;
    const u32 width = u32(std::min<u64>(groups, 65535));
    commands.Dispatch(MTLSizeMake(width, (groups + width - 1) / width, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    batch.assembled = true;
}

void TessellationExports::Validate(Batch& batch) const {
    if (batch.owner != this || !batch.assembled ||
        !ValidAllocation(batch.errors, instance.GetDevice(), sizeof(u32)) ||
        !batch.errors->IsReady()) {
        throw std::runtime_error(
            "Metal tessellation export validation requires completed assembly");
    }
    u32 errors;
    std::memcpy(&errors, batch.errors->ReadbackData().data(), sizeof(errors));
    if (errors) {
        throw std::runtime_error("Metal tessellation primitive/export mapping failed validation");
    }
    batch.validated = true;
}

} // namespace Metal
