// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "core/memory.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_graphics_runtime.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tessellation_dispatch.h"

namespace Metal {
namespace {

void ValidateInputs(const TessellationWorkLayout& layout, Shader::HwStage stage,
                    std::span<const GeometryExportInput> inputs) {
    const auto& config = layout.KernelConfig();
    if (inputs.size() != u64{config.workgroups} * config.threads_per_group) {
        throw std::runtime_error("Metal tessellation register input count does not match dispatch");
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        const auto& first = inputs[index - index % config.threads_per_group];
        const auto group = layout.Group(u32(index / config.threads_per_group));
        const auto active =
            stage == Shader::HwStage::Local ? group.local_invocations : group.hull_invocations;
        if (input.active != (index % config.threads_per_group < active) ||
            input.system_sgprs != first.system_sgprs) {
            throw std::runtime_error(
                "Metal tessellation inputs do not match patch activity or wave scalar registers");
        }
    }
}

std::vector<GeometryAddressInput> InputDomains(std::span<const GeometryExportInput> inputs) {
    using Attribute = Shader::IR::Attribute;
    std::vector<GeometryAddressInput> domains;
    const auto add = [&](Attribute attribute, u32 component, auto value) {
        GeometryAddressInterval interval{std::numeric_limits<u32>::max(), 0};
        for (const auto& input : inputs) {
            interval.lower = std::min(interval.lower, value(input));
            interval.upper = std::max(interval.upper, value(input));
        }
        domains.push_back({attribute, component, interval});
    };
    for (u32 scalar = 0; scalar < GeometryExportInput{}.system_sgprs.size(); ++scalar) {
        add(Attribute::GeometryExportScalar, scalar,
            [scalar](const auto& input) { return input.system_sgprs[scalar]; });
    }
    for (u32 vector = 0; vector < GeometryExportInput{}.vgprs.size(); ++vector) {
        add(Attribute::GeometryExportVector, vector,
            [vector](const auto& input) { return input.vgprs[vector]; });
    }
    add(Attribute::GeometryInputActive, 0, [](const auto& input) { return input.active; });
    return domains;
}

} // namespace

TessellationDispatch::TessellationDispatch(const Instance& instance_, Scheduler& scheduler_,
                                           BufferCache& buffer_cache_, ShaderCache& shaders_,
                                           GuestResources& resources_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_}, shaders{shaders_},
      resources{resources_}, commands{instance, scheduler}, staging{instance, scheduler},
      factor_decoder{instance, scheduler}, domain_generator{instance, scheduler},
      evaluation_inputs{instance, scheduler}, evaluation_exports{instance, scheduler} {}

void TessellationDispatch::RunDomainBatches(
    const AmdGpu::Regs& regs, const Result& hull,
    const std::function<void(TessellationDomains::Result&&)>& consume, u32 maximum_patches) {
    if (!consume || !hull.factors || !hull.factors->decoded || !hull.config.workgroups) {
        throw std::runtime_error("Metal tessellation domain generation requires hull factors");
    }
    const auto constants = ReadConstants(regs, hull.config);
    const auto work =
        TessellationWorkLayout{regs, constants}.Slice(hull.first_group, hull.config.workgroups);
    const auto& factors = *hull.factors->decoded;
    if (work.KernelConfig() != hull.config ||
        work.PatchCount(0, hull.config.workgroups) != factors.patch_count) {
        throw std::runtime_error("Metal tessellation domains do not match their hull work");
    }
    const auto config = BuildTessellationDomainConfig(regs);
    const u32 capacity = std::min(domain_generator.MaxBatchPatches(), maximum_patches);
    if (!capacity) {
        throw std::runtime_error("Metal tessellation domain batch cannot fit on this device");
    }
    for (u32 first = 0; first < factors.patch_count;) {
        const u32 count = std::min(capacity, factors.patch_count - first);
        auto domains = domain_generator.Generate(config, factors, first, count);
        consume(std::move(domains));
        first += count;
    }
}

void TessellationDispatch::RunEvaluationBatches(
    const AmdGpu::Regs& regs, const Result& hull, bool neo_mode,
    const std::function<void(TessellationExports::Batch&&)>& consume,
    std::span<const ImageInfo> attachments) {
    if (!consume) {
        throw std::runtime_error("Metal tessellation evaluation requires a batch consumer");
    }
    const auto work = EvaluationWork(regs, hull);
    const auto ring = BuildTessellationExportRingLayout(regs, work);
    const auto config = BuildTessellationDomainConfig(regs);
    const u32 capacity =
        evaluation_exports.MaxBatchPatches(ring.wave_bytes / 256, config.tessellation.topology);
    RunDomainBatches(
        regs, hull,
        [&](TessellationDomains::Result&& domains) {
            auto batch = evaluation_exports.Allocate(domains, work, ring);
            std::array<bool, 256> relative_patches{};
            for (u32 index = 0; index < work.KernelConfig().workgroups; ++index) {
                const auto group = work.Group(index);
                const u64 first = work.PatchCount(0, index);
                for (u32 patch = 0; patch < group.patch_count; ++patch) {
                    const u64 flat = first + patch;
                    if (flat >= domains.first_patch &&
                        flat - domains.first_patch < domains.patch_count) {
                        relative_patches[patch] = true;
                    }
                }
            }
            for (u32 patch = 0; patch < relative_patches.size(); ++patch) {
                if (!relative_patches[patch]) {
                    continue;
                }
                for (u32 point = 0; point < TessellationDomains::PointsPerPatch; point += 64) {
                    auto exports = RunEvaluationExport(regs, hull, domains, patch, point, neo_mode,
                                                       attachments);
                    if (!exports ||
                        exports->group_ranges.size() != work.KernelConfig().workgroups ||
                        std::ranges::any_of(exports->group_ranges, [&](const auto& range) {
                            return range.lower > range.upper ||
                                   range.upper - range.lower + 1 != ring.wave_bytes;
                        })) {
                        throw std::runtime_error(
                            "Metal tessellation evaluation lost an export chunk");
                    }
                    evaluation_exports.Capture(
                        batch, patch, point, exports->snapshot, exports->group_snapshot_offsets,
                        {exports->resource_index, exports->address, exports->size});
                }
            }
            evaluation_exports.Assemble(batch);
            scheduler.Finish();
            evaluation_exports.Validate(batch);
            domain_generator.Validate(domains);
            consume(std::move(batch));
        },
        capacity);
}

void TessellationDispatch::ValidateDomains(const TessellationDomains::Result& domains) const {
    domain_generator.Validate(domains);
}

TessellationRasterBindings TessellationDispatch::DescribeRaster(const AmdGpu::Regs& regs) {
    ValidateTessellationEvaluationState(regs, true);
    const auto work = PlanLocalHull(regs);
    if (!work.KernelConfig().workgroups) {
        return {};
    }
    auto config = work.Slice(0, 1).KernelConfig().Evaluation();
    config.compact_evaluation = true;
    config.raster_output = true;
    const auto constants = std::bit_cast<Shader::TessellationDataConstantBuffer>(config.constants);
    const auto runtime = BuildTessellationRuntime(regs, Shader::HwStage::Vertex, constants);
    Shader::Backend::Bindings binding{};
    auto& shader = shaders.GetShader(
        Shader::HwStage::Vertex, Shader::SwStage::TessellationEval,
        shaders.GetParams(regs.vs_program.Address<VAddr>(), regs.vs_program.user_data), runtime,
        binding, false, false, nullptr, nullptr, nullptr, nullptr, &config);
    if (!shader.shader.tessellation || !shader.shader.tessellation->raster) {
        throw std::runtime_error("Native direct TES description has no raster interface");
    }
    return *shader.shader.tessellation->raster;
}

void TessellationDispatch::RunRasterBatches(
    const AmdGpu::Regs& regs, const Result& hull, bool neo_mode,
    const std::function<void(TessellationExports::Batch&&)>& consume,
    std::span<const ImageInfo> attachments) {
    if (!consume) {
        throw std::runtime_error("Native direct TES requires a raster batch consumer");
    }
    const auto work = EvaluationWork(regs, hull, true);
    const auto interface = DescribeRaster(regs);
    const u32 capacity = evaluation_exports.MaxBatchPatches(
        interface.vertex_words, BuildTessellationDomainConfig(regs).tessellation.topology);
    RunDomainBatches(
        regs, hull,
        [&](TessellationDomains::Result&& domains) {
            auto batch = evaluation_exports.AllocateRaster(domains, work, interface.vertex_words,
                                                           interface.attributes);
            std::array<bool, 256> relative_patches{};
            for (u32 group = 0; group < work.KernelConfig().workgroups; ++group) {
                const auto metadata = work.Group(group);
                const u64 first = work.PatchCount(0, group);
                for (u32 patch = 0; patch < metadata.patch_count; ++patch) {
                    if (first + patch >= domains.first_patch &&
                        first + patch - domains.first_patch < domains.patch_count) {
                        relative_patches[patch] = true;
                    }
                }
            }
            for (u32 patch = 0; patch < relative_patches.size(); ++patch) {
                if (!relative_patches[patch]) {
                    continue;
                }
                for (u32 point = 0; point < TessellationDomains::PointsPerPatch; point += 64) {
                    auto chunk = RunEvaluationRaster(regs, hull, domains, patch, point, neo_mode,
                                                     attachments);
                    if (!chunk || chunk->vertex_words != interface.vertex_words ||
                        !std::ranges::equal(chunk->attributes, interface.attributes,
                                            [](const auto& left, const auto& right) {
                                                return left.attribute == right.attribute &&
                                                       left.component == right.component &&
                                                       left.stream == right.stream &&
                                                       left.word == right.word;
                                            })) {
                        throw std::runtime_error(
                            "Native direct TES raster interface changed after planning");
                    }
                    evaluation_exports.CaptureRaster(batch, patch, point, chunk->vertices);
                }
            }
            evaluation_exports.Assemble(batch);
            scheduler.Finish();
            evaluation_exports.Validate(batch);
            domain_generator.Validate(domains);
            consume(std::move(batch));
        },
        capacity);
}

std::vector<u32> TessellationDispatch::ReadGuestIndices(const AmdGpu::Regs& regs, u32 first_index,
                                                        bool neo_mode) {
    const auto range = BuildTessellationIndexRange(regs, first_index,
                                                   instance.GetDevice().maxBufferLength, neo_mode);
    std::vector<u32> indices(regs.num_indices);
    if (!range.count) {
        return indices;
    }
    const VAddr address = range.address;
    const u64 bytes = range.SizeBytes();
    if (bytes > indices.size() * sizeof(u32)) {
        throw std::runtime_error("Metal tessellation index source exceeds decoded storage");
    }
    auto& memory = *Core::Memory::Instance();
    const auto backing = memory.GetBackingRanges(address, bytes);
    if (std::ranges::any_of(backing, [](const auto& range) { return !range.physical_address; })) {
        throw std::runtime_error("Metal tessellation indices must be fully mapped");
    }
    buffer_cache.ReadMemory(address, bytes);
    auto* data = reinterpret_cast<u8*>(indices.data());
    memory.ReadSparseMemory(address, {data, size_t(bytes)});
    DecodeTessellationIndices(indices, range, regs.index_buffer_type.swap_mode);
    return indices;
}

void TessellationDispatch::RunGuestLocalHullBatches(const AmdGpu::Regs& regs, bool indexed,
                                                    u32 first_index, bool neo_mode,
                                                    const std::function<void(Result&&)>& consume,
                                                    std::span<const ImageInfo> attachments,
                                                    u32 maximum_groups) {
    if (!consume || (!indexed && first_index) || !maximum_groups) {
        throw std::runtime_error("Invalid Metal tessellation batch consumer or index offset");
    }
    const auto work = PlanLocalHull(regs);
    const auto batches =
        SplitTessellationWork(regs, work, instance.GetDevice().maxBufferLength, neo_mode);
    if (batches.empty()) {
        return;
    }
    const auto indices =
        indexed ? ReadGuestIndices(regs, first_index, neo_mode) : std::vector<u32>{};
    const auto decoded = indexed ? std::optional<std::span<const u32>>{indices} : std::nullopt;
    for (const auto& batch : batches) {
        for (u32 first = 0; first < batch.KernelConfig().workgroups;) {
            const u32 count = std::min(maximum_groups, batch.KernelConfig().workgroups - first);
            const auto slice = batch.Slice(first, count);
            const auto offsets = BuildTessellationGroupOffsets(regs, slice, neo_mode);
            consume(RunLocalHull(regs, slice, decoded, offsets, neo_mode, attachments));
            first += count;
        }
    }
}

TessellationDispatch::Result TessellationDispatch::RunGuestLocalHull(
    const AmdGpu::Regs& regs, bool indexed, u32 first_index, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    const auto layout = PlanLocalHull(regs);
    const auto offsets = BuildTessellationGroupOffsets(regs, layout, neo_mode);
    return DispatchGuestLocalHull(regs, layout, indexed, first_index, offsets, neo_mode,
                                  attachments);
}

TessellationDispatch::Result TessellationDispatch::RunGuestLocalHull(
    const AmdGpu::Regs& regs, bool indexed, u32 first_index,
    std::span<const TessellationGroupOffsets> group_offsets, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    const auto layout = PlanLocalHull(regs);
    return DispatchGuestLocalHull(regs, layout, indexed, first_index, group_offsets, neo_mode,
                                  attachments);
}

TessellationDispatch::Result TessellationDispatch::DispatchGuestLocalHull(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout, bool indexed, u32 first_index,
    std::span<const TessellationGroupOffsets> group_offsets, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    if (!indexed && first_index) {
        throw std::runtime_error("Metal tessellation auto indices cannot have a buffer offset");
    }
    if (group_offsets.size() != layout.KernelConfig().workgroups) {
        throw std::runtime_error("Metal tessellation group offsets do not match the draw");
    }
    if (!layout.KernelConfig().workgroups) {
        return {layout.KernelConfig(), {}};
    }
    if (!indexed) {
        return RunLocalHull(regs, layout, std::nullopt, group_offsets, neo_mode, attachments);
    }
    const auto indices = ReadGuestIndices(regs, first_index, neo_mode);
    return RunLocalHull(regs, layout, std::span<const u32>{indices}, group_offsets, neo_mode,
                        attachments);
}

TessellationWorkLayout TessellationDispatch::PlanLocalHull(const AmdGpu::Regs& regs) {
    const auto runtime = BuildTessellationRuntime(regs, Shader::HwStage::Hull, {});
    const auto params =
        shaders.GetParams(regs.hs_program.Address<VAddr>(), regs.hs_program.user_data);
    return TessellationWorkLayout{regs, shaders.GetTessellationConstants(params, runtime)};
}

Shader::TessellationDataConstantBuffer TessellationDispatch::ReadConstants(
    const AmdGpu::Regs& regs, const TessellationKernelConfig& config) {
    const auto runtime = BuildTessellationRuntime(regs, Shader::HwStage::Hull, {});
    const auto params =
        shaders.GetParams(regs.hs_program.Address<VAddr>(), regs.hs_program.user_data);
    const auto constants = shaders.GetTessellationConstants(params, runtime);
    if (std::bit_cast<std::array<u32, 9>>(constants) != config.constants) {
        throw std::runtime_error("Metal tessellation constants changed after work planning");
    }
    return constants;
}

TessellationDispatch::Ring TessellationDispatch::ValidateRing(
    const AmdGpu::Regs& regs, const ShaderCache::Variant& shader,
    const TessellationWorkLayout& work, std::span<const GeometryExportInput> inputs, RingKind kind,
    bool neo_mode) {
    const auto config =
        kind == RingKind::Export ? work.KernelConfig().Evaluation() : work.KernelConfig();
    const bool offchip = kind == RingKind::Offchip;
    const bool exports = kind == RingKind::Export;
    const auto& bindings = shader.shader.tessellation->registers;
    if (!bindings.address_origins || !bindings.address_program) {
        throw std::runtime_error("Metal tessellation ring addresses require native analysis");
    }
    const u32 export_scalar = regs.hs_program.settings.oc_lds_en ? 2 : 0;
    const u64 export_sources =
        (u64{1} << (u32(GeometryAddressSource::ExportScalar) + export_scalar)) |
        (regs.hs_program.settings.oc_lds_en ? u64{1} << u32(GeometryAddressSource::ExportScalar)
                                            : 0) |
        (u64{0xf} << u32(GeometryAddressSource::ExportVector)) |
        (u64{1} << u32(GeometryAddressSource::InputActive));
    if (exports && (bindings.input_sources & ~export_sources)) {
        throw std::runtime_error("Metal TES export shader requires additional system inputs");
    }
    const u32 scalar = exports ? export_scalar
                       : kind == RingKind::Factor && regs.hs_program.settings.oc_lds_en ? 1
                                                                                        : 0;
    const u64 source = u64{1} << (u32(GeometryAddressSource::ExportScalar) + scalar);
    std::optional<u32> ring_resource;
    for (u32 index = 0; index < shader.info.buffers.size(); ++index) {
        if (!(bindings.address_origins->writes.at(index) & source)) {
            continue;
        }
        if (ring_resource) {
            throw std::runtime_error("Metal tessellation ring descriptor is ambiguous");
        }
        ring_resource = index;
    }
    if (!ring_resource) {
        throw std::runtime_error("Metal tessellation ring descriptor was not identified");
    }
    const auto sharp = ReadGuestBufferSharp(shader.info.buffers[*ring_resource], shader.info);
    constexpr u32 FactorRingSizeRegister = 0x30938 / sizeof(u32);
    const auto offchip_layout = BuildTessellationOffchipLayout(regs, neo_mode);
    const auto export_layout =
        exports ? BuildTessellationExportRingLayout(regs, work) : TessellationExportRingLayout{};
    const u64 block_bytes = exports ? export_layout.wave_bytes : offchip_layout.block_bytes;
    const u64 ring_size = exports ? export_layout.size
                          : offchip
                              ? offchip_layout.SizeBytes()
                              : u64{regs.reg_array[FactorRingSizeRegister] & 0xffff} * sizeof(u32);
    const u64 descriptor_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
    const u64 size = std::min(ring_size, descriptor_size);
    const VAddr address =
        offchip || exports ? u64{sharp.base_address} : regs.vgt_tf_memory_base.MemoryBase();
    const bool valid_layout =
        exports ? sharp.swizzle_enable && sharp.add_tid_enable && sharp.GetIndexStride() == 64 &&
                      sharp.GetElementSize() == 4 &&
                      (!sharp.stride || sharp.stride == block_bytes / 64) && address % 4 == 0
                : !sharp.swizzle_enable && !sharp.add_tid_enable &&
                      (!sharp.stride || sharp.stride == 4);
    if (!sharp.Valid() || !address || sharp.base_address != address || !size || !valid_layout ||
        address >= (u64{1} << 40) || size > (u64{1} << 40) - address) {
        throw std::runtime_error("Metal tessellation ring requires additional layout lowering");
    }
    const auto backing = Core::Memory::Instance()->GetBackingRanges(address, size);
    std::vector<GeometryBufferByteInterval> physical;
    for (const auto& range : backing) {
        if (!range.physical_address) {
            throw std::runtime_error("Metal tessellation ring must be fully mapped");
        }
        physical.push_back({*range.physical_address, *range.physical_address + range.size - 1});
    }
    std::ranges::sort(physical, {}, &GeometryBufferByteInterval::lower);
    for (size_t index = 1; index < physical.size(); ++index) {
        if (physical[index].lower <= physical[index - 1].upper) {
            throw std::runtime_error("Metal tessellation ring aliases its own backing");
        }
    }
    Ring result{*ring_resource, address, size, {}, {}, {}};
    result.group_ranges.reserve(config.workgroups);
    for (u32 group = 0; group < config.workgroups; ++group) {
        const u64 group_start =
            inputs[size_t(group) * config.threads_per_group].system_sgprs[scalar];
        if ((offchip || exports) &&
            (group_start % block_bytes || group_start > size || block_bytes > size - group_start)) {
            throw std::runtime_error("Metal tessellation group block exceeds its ring");
        }
        const auto domains = InputDomains(
            inputs.subspan(size_t(group) * config.threads_per_group, config.threads_per_group));
        const auto bounds =
            bindings.address_program->Evaluate(domains, shader.info.flattened_ud_buf);
        std::optional<GeometryBufferByteInterval> writes;
        for (const auto& bound : bounds) {
            if (bound.resource_index != *ring_resource || !bound.reachable) {
                continue;
            }
            if (!bound.bytes || bound.bytes->upper >= size) {
                throw std::runtime_error("Metal tessellation ring access exceeds its ring");
            }
            if ((offchip || exports) && (bound.bytes->lower < group_start ||
                                         bound.bytes->upper - group_start >= block_bytes)) {
                throw std::runtime_error("Metal tessellation access leaves its group block");
            }
            if (bound.read && !offchip) {
                throw std::runtime_error(
                    "Metal tessellation output reads require ordered ring access");
            }
            if (bound.write) {
                if (writes) {
                    writes->lower = std::min(writes->lower, bound.bytes->lower);
                    writes->upper = std::max(writes->upper, bound.bytes->upper);
                } else {
                    writes = bound.bytes;
                }
            }
        }
        const auto wave =
            inputs.subspan(size_t(group) * config.threads_per_group, config.threads_per_group);
        const bool inactive_export =
            exports && std::ranges::none_of(wave, [](const auto& input) { return input.active; });
        if (!writes && !inactive_export) {
            throw std::runtime_error("Metal tessellation patch group has no proven ring writes");
        }
        result.group_ranges.push_back(
            offchip || exports
                ? GeometryBufferByteInterval{group_start, group_start + block_bytes - 1}
                : *writes);
    }
    auto ordered = result.group_ranges;
    std::ranges::sort(ordered, {}, &GeometryBufferByteInterval::lower);
    for (size_t index = 1; index < ordered.size(); ++index) {
        if (ordered[index].lower <= ordered[index - 1].upper) {
            throw std::runtime_error("Metal tessellation patch groups overlap in the guest ring");
        }
    }
    return result;
}

std::vector<BufferRange> TessellationDispatch::PrepareRingSnapshot(Ring& ring) {
    u64 bytes{};
    std::vector<BufferRange> sources;
    ring.group_snapshot_offsets.reserve(ring.group_ranges.size());
    for (const auto& interval : ring.group_ranges) {
        const u64 size = interval.upper - interval.lower + 1;
        if (bytes > instance.GetDevice().maxBufferLength ||
            size > instance.GetDevice().maxBufferLength - bytes) {
            throw std::runtime_error("Metal tessellation ring snapshot exceeds device capacity");
        }
        ring.group_snapshot_offsets.push_back(bytes);
        const auto ranges = buffer_cache.GetPagedBufferRanges(ring.address + interval.lower, size);
        sources.insert(sources.end(), ranges.begin(), ranges.end());
        bytes += size;
    }
    if (!bytes) {
        throw std::runtime_error("Metal tessellation ring snapshot requires written ranges");
    }
    ring.snapshot = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(bytes, alignof(u32)));
    return sources;
}

TessellationDispatch::StageResult TessellationDispatch::RunStage(
    const AmdGpu::Regs& regs, Shader::HwStage stage, const TessellationWorkLayout& work,
    const Shader::TessellationDataConstantBuffer& constants,
    std::span<const GeometryExportInput> inputs,
    const std::shared_ptr<StagingBufferPool::Allocation>& lds, bool neo_mode,
    std::span<const ImageInfo> attachments,
    const std::shared_ptr<StagingBufferPool::Allocation>& gpu_registers,
    const std::shared_ptr<StagingBufferPool::Allocation>& gpu_arguments,
    const Ring* evaluation_offchip) {
    const auto config = stage == Shader::HwStage::Export || stage == Shader::HwStage::Vertex
                            ? work.KernelConfig().Evaluation()
                            : work.KernelConfig();
    auto kernel_config = config;
    kernel_config.raster_output = stage == Shader::HwStage::Vertex;
    kernel_config.compact_evaluation =
        stage == Shader::HwStage::Export || kernel_config.raster_output;
    if (inputs.size_bytes() != config.RegisterSizeBytes() ||
        kernel_config.InputSizeBytes() > instance.GetDevice().maxBufferLength ||
        kernel_config.InputSizeBytes() / sizeof(u32) > std::numeric_limits<u32>::max() ||
        bool(gpu_registers) != bool(gpu_arguments) ||
        (kernel_config.raster_output && !gpu_registers) ||
        (gpu_registers && ((stage != Shader::HwStage::Export && !kernel_config.raster_output) ||
                           !gpu_registers->GetBuffer() ||
                           gpu_registers->GetBuffer().device != instance.GetDevice() ||
                           gpu_registers->GetOffset() % alignof(u32) ||
                           gpu_registers->GetOffset() > gpu_registers->GetBuffer().length ||
                           gpu_registers->SizeBytes() < kernel_config.InputSizeBytes() ||
                           gpu_registers->SizeBytes() >
                               gpu_registers->GetBuffer().length - gpu_registers->GetOffset()))) {
        throw std::runtime_error("Metal tessellation register allocation exceeds dispatch bounds");
    }
    if (gpu_arguments &&
        (!gpu_arguments->GetBuffer() || gpu_arguments->GetBuffer().device != instance.GetDevice() ||
         gpu_arguments->GetOffset() % alignof(u32) ||
         gpu_arguments->GetOffset() > gpu_arguments->GetBuffer().length ||
         gpu_arguments->SizeBytes() < sizeof(MTLDispatchThreadgroupsIndirectArguments) ||
         gpu_arguments->SizeBytes() >
             gpu_arguments->GetBuffer().length - gpu_arguments->GetOffset())) {
        throw std::runtime_error("Metal tessellation indirect argument allocation is invalid");
    }
    const auto runtime = BuildTessellationRuntime(regs, stage, constants);
    const auto& program = *regs.ProgramForStage(u32(stage));
    const auto params = shaders.GetParams(program.Address<VAddr>(), program.user_data);
    Shader::Backend::Bindings bindings{};
    auto& shader = shaders.GetShader(stage, runtime.sw_stage, params, runtime, bindings, false,
                                     false, nullptr, nullptr, nullptr, nullptr, &kernel_config);
    if (!shader.shader.tessellation) {
        throw std::runtime_error("Metal tessellation dispatch requires a native kernel");
    }
    const bool evaluation = stage == Shader::HwStage::Export || stage == Shader::HwStage::Vertex;
    if (bool(evaluation_offchip) != (evaluation && bool(regs.hs_program.settings.oc_lds_en))) {
        throw std::runtime_error("Metal TES offchip input does not match its dispatch");
    }
    if (evaluation) {
        u64 allowed = (u64{0xf} << u32(GeometryAddressSource::ExportVector)) |
                      (u64{1} << u32(GeometryAddressSource::InputActive));
        if (evaluation_offchip) {
            allowed |= u64{1} << (u32(GeometryAddressSource::ExportScalar) +
                                  (kernel_config.raster_output ? 1 : 0));
        }
        if (stage == Shader::HwStage::Export) {
            allowed |=
                u64{1} << (u32(GeometryAddressSource::ExportScalar) + (evaluation_offchip ? 2 : 0));
        }
        if (shader.shader.tessellation->registers.input_sources & ~allowed) {
            throw std::runtime_error("Metal TES requires additional system-register inputs");
        }
    }
    std::optional<FactorRing> factors;
    std::optional<Ring> offchip;
    std::optional<Ring> exports;
    if (stage == Shader::HwStage::Hull) {
        factors =
            FactorRing{ValidateRing(regs, shader, work, inputs, RingKind::Factor, neo_mode), {}};
        if (regs.hs_program.settings.oc_lds_en) {
            offchip = ValidateRing(regs, shader, work, inputs, RingKind::Offchip, neo_mode);
        }
    } else if (stage == Shader::HwStage::Export) {
        exports = ValidateRing(regs, shader, work, inputs, RingKind::Export, neo_mode);
    }
    const auto pipeline = shaders.GetComputePipeline(shader);
    const auto& layout = *shader.shader.tessellation;
    auto registers = gpu_registers;
    auto arguments = gpu_arguments;
    if (!registers) {
        registers = std::make_shared<StagingBufferPool::Allocation>(
            staging.Allocate(kernel_config.InputSizeBytes(), alignof(u32)));
        auto data = registers->UploadData();
        std::memcpy(data.data(), inputs.data(), inputs.size_bytes());
        if (kernel_config.compact_evaluation) {
            u32 active_groups{};
            for (u32 group = 0; group < config.workgroups; ++group) {
                const auto wave = inputs.subspan(size_t(group) * config.threads_per_group,
                                                 config.threads_per_group);
                if (std::ranges::any_of(wave,
                                        [](const auto& input) { return input.active == 1; })) {
                    std::memcpy(data.data() + inputs.size_bytes() +
                                    u64{active_groups} * sizeof(u32),
                                &group, sizeof(group));
                    ++active_groups;
                }
            }
            static_assert(sizeof(MTLDispatchThreadgroupsIndirectArguments) == 3 * sizeof(u32));
            const std::array<u32, 3> counts{active_groups, 1, 1};
            arguments = std::make_shared<StagingBufferPool::Allocation>(
                staging.Allocate(sizeof(counts), alignof(u32)));
            std::memcpy(arguments->UploadData().data(), counts.data(), sizeof(counts));
        }
    }
    std::vector<GuestResources::GeometryBuffer> buffers{{layout.registers.inputs, registers}};
    std::optional<RasterChunk> raster;
    if (layout.raster) {
        const u64 bytes = u64{config.workgroups} * config.threads_per_group *
                          layout.raster->vertex_words * sizeof(u32);
        if (!kernel_config.raster_output || !bytes ||
            bytes > instance.GetDevice().maxBufferLength ||
            bytes / sizeof(u32) > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Metal direct TES payload exceeds native storage bounds");
        }
        raster = RasterChunk{std::make_shared<StagingBufferPool::Allocation>(
                                 staging.AllocatePrivate(bytes, alignof(u32))),
                             layout.raster->vertex_words, layout.raster->attributes};
        buffers.push_back({layout.raster->output, raster->vertices});
    } else if (kernel_config.raster_output) {
        throw std::runtime_error("Metal direct TES is missing its raster payload metadata");
    }
    if (layout.lds) {
        buffers.push_back({*layout.lds, lds});
    }
    const auto domains = InputDomains(inputs);
    std::optional<u32> offchip_reader;
    std::vector<GuestBufferExtent> extents;
    if (evaluation_offchip) {
        offchip_reader = ValidateEvaluationOffchip(shader, work, inputs, *evaluation_offchip,
                                                   kernel_config.raster_output);
        if (offchip_reader) {
            u64 bytes{};
            for (const auto& range : evaluation_offchip->group_ranges) {
                bytes = std::max(bytes, range.upper + 1);
            }
            extents.push_back({*offchip_reader, bytes});
        }
    }
    auto resources_stage = resources.Prepare(shader, neo_mode, buffers, 0, extents, domains);
    if (factors) {
        resources.ValidateTessellationRing(*resources_stage, factors->resource_index,
                                           factors->address, factors->size, attachments);
    }
    if (offchip) {
        resources.ValidateTessellationRing(*resources_stage, offchip->resource_index,
                                           offchip->address, offchip->size, attachments);
    }
    if (exports) {
        resources.ValidateTessellationRing(*resources_stage, exports->resource_index,
                                           exports->address, exports->size, attachments);
    }
    const std::array stages{resources_stage.get()};
    resources.ValidateAttachments(stages, attachments);
    if (evaluation_offchip) {
        resources.ReplayTessellationRing(*resources_stage,
                                         {offchip_reader.value_or(Shader::NUM_BUFFERS),
                                          evaluation_offchip->address, evaluation_offchip->size},
                                         evaluation_offchip->group_ranges,
                                         evaluation_offchip->snapshot, attachments);
    }
    resources.Acquire(stages);
    const auto factor_sources =
        factors ? PrepareRingSnapshot(*factors) : std::vector<BufferRange>{};
    const auto offchip_sources =
        offchip ? PrepareRingSnapshot(*offchip) : std::vector<BufferRange>{};
    const auto export_sources =
        exports ? PrepareRingSnapshot(*exports) : std::vector<BufferRange>{};
    std::vector<TessellationFactors::Group> factor_groups;
    if (factors) {
        const u32 components = TessellationFactors::Components(regs.tess_config.type);
        const u32 scalar = regs.hs_program.settings.oc_lds_en ? 1 : 0;
        factor_groups.reserve(config.workgroups);
        for (u32 index = 0; index < config.workgroups; ++index) {
            const auto group = work.Group(index);
            const u64 offset =
                inputs[size_t(index) * config.threads_per_group].system_sgprs[scalar];
            const auto& interval = factors->group_ranges[index];
            const u64 bytes = (u64{group.patch_count} * components + 1) * sizeof(u32);
            if (offset < interval.lower || offset > interval.upper ||
                bytes > interval.upper - offset + 1) {
                throw std::runtime_error(
                    "Metal tessellation factors are outside the captured group range");
            }
            factor_groups.push_back(
                {factors->group_snapshot_offsets[index] + offset - interval.lower,
                 group.patch_count, group.instance, group.first_patch});
        }
    }
    resources.MarkWrites(stages);
    const auto snapshot = resources.Snapshot(*resources_stage);
    if (raster) {
        scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
        scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
        commands.FillBuffer({raster->vertices->GetBuffer(), raster->vertices->GetOffset(),
                             raster->vertices->SizeBytes()},
                            0);
    }
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(pipeline, *snapshot);
    if (arguments) {
        commands.DispatchIndirect(
            {arguments->GetBuffer(), arguments->GetOffset(), arguments->SizeBytes()},
            MTLSizeMake(config.threads_per_group, 1, 1));
    } else {
        commands.Dispatch(MTLSizeMake(config.workgroups, 1, 1),
                          MTLSizeMake(config.threads_per_group, 1, 1));
    }
    if (factors || offchip || exports) {
        scheduler.EncoderBarrier(MTLStageDispatch, MTLStageBlit);
        const auto copy = [&](const Ring& ring, std::span<const BufferRange> sources) {
            const auto& storage = *ring.snapshot;
            u64 offset{};
            for (const auto& source : sources) {
                commands.CopyBuffer(
                    source, {storage.GetBuffer(), storage.GetOffset() + offset, source.size},
                    source.size);
                offset += source.size;
            }
        };
        if (offchip) {
            copy(*offchip, offchip_sources);
        }
        if (factors) {
            copy(*factors, factor_sources);
        }
        if (exports) {
            copy(*exports, export_sources);
        }
        scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    }
    if (factors) {
        factors->decoded =
            factor_decoder.Decode(regs.tess_config.type, factors->snapshot, factor_groups);
    }
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.Finish();
    resources.ValidateTessellation(*resources_stage);
    if (factors) {
        factor_decoder.Validate(*factors->decoded);
    }
    return {std::move(factors), std::move(offchip), std::move(exports), std::move(raster)};
}

std::vector<u32> TessellationDispatch::EvaluationOffchipOffsets(const AmdGpu::Regs& regs,
                                                                const Result& hull,
                                                                const TessellationWorkLayout& work,
                                                                bool neo_mode) const {
    if (!regs.hs_program.settings.oc_lds_en) {
        if (hull.offchip) {
            throw std::runtime_error("Metal onchip TES cannot consume an offchip snapshot");
        }
        return {};
    }
    if (!hull.offchip || work.KernelConfig() != hull.config) {
        throw std::runtime_error("Metal offchip TES requires its original Hull snapshot");
    }
    const auto& ring = *hull.offchip;
    const auto offchip_layout = BuildTessellationOffchipLayout(regs, neo_mode);
    const u64 block_bytes = offchip_layout.block_bytes;
    const u64 capacity = offchip_layout.SizeBytes();
    const u32 groups = work.KernelConfig().workgroups;
    const auto& storage = ring.snapshot;
    if (ring.resource_index >= Shader::NUM_BUFFERS || !ring.address || ring.address % 4 ||
        ring.address >= (u64{1} << 40) || !ring.size || ring.size > capacity ||
        ring.size > (u64{1} << 40) - ring.address || ring.group_ranges.size() != groups ||
        ring.group_snapshot_offsets.size() != groups || !storage || !storage->IsReady() ||
        !storage->GetBuffer() || storage->GetBuffer().device != instance.GetDevice() ||
        storage->GetOffset() % 4 || storage->GetOffset() > storage->GetBuffer().length ||
        storage->SizeBytes() > storage->GetBuffer().length - storage->GetOffset()) {
        throw std::runtime_error("Metal offchip TES snapshot extent or ownership is invalid");
    }
    std::vector<u32> offsets;
    offsets.reserve(groups);
    u64 cursor{};
    for (u32 group = 0; group < groups; ++group) {
        const auto& interval = ring.group_ranges[group];
        if (interval.lower % block_bytes || interval.lower >= ring.size ||
            block_bytes > ring.size - interval.lower ||
            interval.upper != interval.lower + block_bytes - 1 ||
            ring.group_snapshot_offsets[group] != cursor) {
            throw std::runtime_error("Metal offchip TES group does not match its captured block");
        }
        offsets.push_back(u32(interval.lower));
        cursor += block_bytes;
    }
    auto ordered = ring.group_ranges;
    std::ranges::sort(ordered, {}, &GeometryBufferByteInterval::lower);
    for (size_t index = 1; index < ordered.size(); ++index) {
        if (ordered[index].lower <= ordered[index - 1].upper) {
            throw std::runtime_error("Metal offchip TES groups overlap in their producer ring");
        }
    }
    if (cursor != storage->SizeBytes()) {
        throw std::runtime_error("Metal offchip TES packed snapshot size is inconsistent");
    }
    return offsets;
}

std::optional<u32> TessellationDispatch::ValidateEvaluationOffchip(
    const ShaderCache::Variant& shader, const TessellationWorkLayout& work,
    std::span<const GeometryExportInput> inputs, const Ring& ring, bool raster) const {
    const auto config = work.KernelConfig().Evaluation();
    const auto& bindings = shader.shader.tessellation->registers;
    if (!bindings.address_origins || !bindings.address_program ||
        inputs.size_bytes() != config.RegisterSizeBytes() ||
        ring.group_ranges.size() != config.workgroups ||
        shader.info.buffers.size() > Shader::NUM_BUFFERS) {
        throw std::runtime_error("Metal offchip TES requires native consumer address analysis");
    }
    const u32 scalar = raster ? 1 : 0;
    const u64 source = u64{1} << (u32(GeometryAddressSource::ExportScalar) + scalar);
    const auto all_bounds =
        bindings.address_program->Evaluate(InputDomains(inputs), shader.info.flattened_ud_buf);
    std::optional<u32> reader;
    for (u32 index = 0; index < shader.info.buffers.size(); ++index) {
        if (shader.info.buffers[index].buffer_type != Shader::BufferType::Guest) {
            continue;
        }
        const auto sharp = ReadGuestBufferSharp(shader.info.buffers[index], shader.info);
        const bool ring_read =
            sharp.base_address == ring.address &&
            std::ranges::any_of(all_bounds, [&](const auto& bound) {
                return bound.resource_index == index && bound.reachable && bound.read;
            });
        if (!(bindings.address_origins->reads[index] & source) && !ring_read) {
            continue;
        }
        const u64 descriptor_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
        if (reader || !sharp.Valid() || sharp.base_address != ring.address ||
            sharp.swizzle_enable || sharp.add_tid_enable || (sharp.stride && sharp.stride != 4) ||
            bindings.address_origins->writes[index] ||
            std::ranges::any_of(ring.group_ranges, [&](const auto& range) {
                return range.upper >= descriptor_size;
            })) {
            throw std::runtime_error("Metal offchip TES descriptor differs from its Hull producer");
        }
        reader = index;
    }
    if (!reader) {
        return std::nullopt;
    }
    for (u32 group = 0; group < config.workgroups; ++group) {
        const auto wave =
            inputs.subspan(size_t(group) * config.threads_per_group, config.threads_per_group);
        const auto& interval = ring.group_ranges[group];
        if (std::ranges::any_of(wave, [&](const auto& input) {
                return input.system_sgprs[scalar] != interval.lower;
            })) {
            throw std::runtime_error("Metal offchip TES scalar does not identify its Hull block");
        }
        for (const auto& bound :
             bindings.address_program->Evaluate(InputDomains(wave), shader.info.flattened_ud_buf)) {
            if (bound.resource_index != *reader || !bound.reachable) {
                continue;
            }
            if (!bound.read || bound.write || !bound.bytes ||
                bound.bytes->lower > bound.bytes->upper || bound.bytes->lower < interval.lower ||
                bound.bytes->upper > interval.upper) {
                throw std::runtime_error("Metal offchip TES read leaves its captured Hull block");
            }
        }
    }
    return reader;
}

TessellationDispatch::Ring TessellationDispatch::RunEvaluationExport(
    const AmdGpu::Regs& regs, const Result& hull,
    std::span<const std::span<const TessellationEvaluationPoint>> points, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    const auto work = EvaluationWork(regs, hull);
    const auto offsets = EvaluationOffchipOffsets(regs, hull, work, neo_mode);
    const auto waves = BuildTessellationExportWaves(regs, work, points, offsets, neo_mode);
    return RunEvaluationExport(regs, hull, waves, neo_mode, attachments);
}

TessellationWorkLayout TessellationDispatch::EvaluationWork(const AmdGpu::Regs& regs,
                                                            const Result& hull, bool raster) {
    const auto& config = hull.config;
    ValidateTessellationEvaluationState(regs, raster);
    if (regs.stage_enable.raw !=
            (raster ? AmdGpu::ShaderStageEnable::LsHs : AmdGpu::ShaderStageEnable::LsHsEsGs) ||
        bool(hull.offchip) != bool(regs.hs_program.settings.oc_lds_en) || !config.workgroups ||
        config.threads_per_group < 64 || config.threads_per_group > 256 ||
        config.threads_per_group % 64 || !config.lds_bytes || config.lds_bytes % 8 ||
        config.lds_bytes > 65536 ||
        config.InputSizeBytes() > instance.GetDevice().maxBufferLength ||
        config.LdsSizeBytes() > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Metal tessellation evaluation export state requires lowering");
    }
    if (!hull.lds || !hull.lds->GetBuffer() ||
        hull.lds->GetBuffer().device != instance.GetDevice() ||
        hull.lds->GetOffset() % alignof(u64) ||
        hull.lds->GetOffset() > hull.lds->GetBuffer().length ||
        hull.lds->SizeBytes() < config.LdsSizeBytes() ||
        hull.lds->SizeBytes() > hull.lds->GetBuffer().length - hull.lds->GetOffset()) {
        throw std::runtime_error("Metal tessellation evaluation requires retained hull LDS");
    }
    const auto constants = ReadConstants(regs, config);
    const auto work =
        TessellationWorkLayout{regs, constants}.Slice(hull.first_group, config.workgroups);
    if (work.KernelConfig() != config) {
        throw std::runtime_error("Metal tessellation evaluation does not match its hull work");
    }
    return work;
}

TessellationDispatch::Ring TessellationDispatch::RunEvaluationExport(
    const AmdGpu::Regs& regs, const Result& hull, std::span<const TessellationEvaluationWave> waves,
    bool neo_mode, std::span<const ImageInfo> attachments) {
    const auto work = EvaluationWork(regs, hull);
    const auto constants =
        std::bit_cast<Shader::TessellationDataConstantBuffer>(work.KernelConfig().constants);
    const auto offsets = EvaluationOffchipOffsets(regs, hull, work, neo_mode);
    const auto registers = BuildTessellationEvaluationRegisters(regs, work, offsets, neo_mode);
    if (waves.size() != registers.size()) {
        throw std::runtime_error("Metal TES waves differ from their original Hull groups");
    }
    for (u32 group = 0; group < waves.size(); ++group) {
        if (waves[group].system_sgprs != registers[group]) {
            throw std::runtime_error("Metal TES wave has inconsistent ring system registers");
        }
    }
    const auto inputs = BuildTessellationEvaluationInputs(work, waves);
    auto result = RunStage(regs, Shader::HwStage::Export, work, constants, inputs, hull.lds,
                           neo_mode, attachments, {}, {}, hull.offchip ? &*hull.offchip : nullptr);
    if (!result.exports) {
        throw std::runtime_error("Metal tessellation evaluation has no retained export ring");
    }
    return std::move(*result.exports);
}

void TessellationDispatch::ValidateEvaluationDomains(const AmdGpu::Regs& regs, const Result& hull,
                                                     const TessellationDomains::Result& domains,
                                                     const TessellationWorkLayout& work) const {
    const auto config = BuildTessellationDomainConfig(regs);
    if (!hull.factors || !hull.factors->decoded || !domains.factors ||
        domains.factors != hull.factors->decoded->storage ||
        config.tessellation.type != domains.config.tessellation.type ||
        config.tessellation.partitioning != domains.config.tessellation.partitioning ||
        config.tessellation.topology != domains.config.tessellation.topology ||
        std::bit_cast<u32>(config.minimum_factor) !=
            std::bit_cast<u32>(domains.config.minimum_factor) ||
        std::bit_cast<u32>(config.maximum_factor) !=
            std::bit_cast<u32>(domains.config.maximum_factor) ||
        work.PatchCount(0, work.KernelConfig().workgroups) != hull.factors->decoded->patch_count) {
        throw std::runtime_error("Metal tessellation GPU domains do not match their hull factors");
    }
}

std::optional<TessellationDispatch::Ring> TessellationDispatch::RunEvaluationExport(
    const AmdGpu::Regs& regs, const Result& hull, const TessellationDomains::Result& domains,
    u32 relative_patch, u32 first_point, bool neo_mode, std::span<const ImageInfo> attachments) {
    const auto work = EvaluationWork(regs, hull);
    ValidateEvaluationDomains(regs, hull, domains, work);
    const auto offsets = EvaluationOffchipOffsets(regs, hull, work, neo_mode);
    const auto registers = BuildTessellationEvaluationRegisters(regs, work, offsets, neo_mode);
    auto inputs =
        evaluation_inputs.BuildInputs(domains, work, BuildTessellationExportRingLayout(regs, work),
                                      relative_patch, first_point, false, {}, registers);
    if (!inputs) {
        return std::nullopt;
    }
    const auto constants =
        std::bit_cast<Shader::TessellationDataConstantBuffer>(work.KernelConfig().constants);
    auto result = RunStage(regs, Shader::HwStage::Export, work, constants, inputs->address_envelope,
                           hull.lds, neo_mode, attachments, inputs->registers, inputs->arguments,
                           hull.offchip ? &*hull.offchip : nullptr);
    evaluation_inputs.Validate(*inputs);
    domain_generator.Validate(domains);
    if (!result.exports) {
        throw std::runtime_error("Metal tessellation GPU evaluation has no retained export ring");
    }
    return std::move(*result.exports);
}

std::optional<TessellationDispatch::RasterChunk> TessellationDispatch::RunEvaluationRaster(
    const AmdGpu::Regs& regs, const Result& hull, const TessellationDomains::Result& domains,
    u32 relative_patch, u32 first_point, bool neo_mode, std::span<const ImageInfo> attachments) {
    const auto work = EvaluationWork(regs, hull, true);
    ValidateEvaluationDomains(regs, hull, domains, work);
    const auto offsets = EvaluationOffchipOffsets(regs, hull, work, neo_mode);
    const auto registers = BuildTessellationEvaluationRegisters(regs, work, offsets, neo_mode);
    const std::array viewport{
        regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f,
        regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f,
        regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f,
        regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f};
    auto inputs = evaluation_inputs.BuildInputs(domains, work, {}, relative_patch, first_point,
                                                true, viewport, registers);
    if (!inputs) {
        return std::nullopt;
    }
    const auto constants =
        std::bit_cast<Shader::TessellationDataConstantBuffer>(work.KernelConfig().constants);
    auto result = RunStage(regs, Shader::HwStage::Vertex, work, constants, inputs->address_envelope,
                           hull.lds, neo_mode, attachments, inputs->registers, inputs->arguments,
                           hull.offchip ? &*hull.offchip : nullptr);
    evaluation_inputs.Validate(*inputs);
    domain_generator.Validate(domains);
    if (!result.raster) {
        throw std::runtime_error("Metal direct TES produced no retained raster payload");
    }
    return std::move(result.raster);
}

TessellationDispatch::Result TessellationDispatch::RunLocalHull(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::optional<std::span<const u32>> decoded_indices,
    std::span<const TessellationGroupOffsets> group_offsets, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    const auto& config = layout.KernelConfig();
    if (config.InputSizeBytes() / sizeof(u32) > std::numeric_limits<u32>::max() ||
        config.LdsSizeBytes() > std::numeric_limits<u32>::max() ||
        config.InputSizeBytes() > instance.GetDevice().maxBufferLength ||
        config.LdsSizeBytes() > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Metal tessellation register storage exceeds device capacity");
    }
    const auto inputs =
        BuildTessellationRegisterInputs(regs, layout, decoded_indices, group_offsets, neo_mode);
    return DispatchLocalHull(regs, layout, inputs.local, inputs.hull, neo_mode, attachments);
}

TessellationDispatch::Result TessellationDispatch::RunLocalHull(
    const AmdGpu::Regs& regs, const TessellationKernelConfig& config,
    std::span<const GeometryExportInput> local_inputs,
    std::span<const GeometryExportInput> hull_inputs, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    const auto constants = ReadConstants(regs, config);
    const TessellationWorkLayout layout{regs, constants};
    if (layout.KernelConfig() != config) {
        throw std::runtime_error("Metal tessellation dispatch does not match guest work layout");
    }
    return DispatchLocalHull(regs, layout, local_inputs, hull_inputs, neo_mode, attachments);
}

TessellationDispatch::Result TessellationDispatch::DispatchLocalHull(
    const AmdGpu::Regs& regs, const TessellationWorkLayout& layout,
    std::span<const GeometryExportInput> local_inputs,
    std::span<const GeometryExportInput> hull_inputs, bool neo_mode,
    std::span<const ImageInfo> attachments) {
    const auto& config = layout.KernelConfig();
    if (!config.workgroups || config.threads_per_group < 64 || config.threads_per_group > 256 ||
        config.threads_per_group % 64 || !config.lds_bytes || config.lds_bytes % 8 ||
        config.lds_bytes > 65536 ||
        config.InputSizeBytes() / sizeof(u32) > std::numeric_limits<u32>::max() ||
        config.LdsSizeBytes() > std::numeric_limits<u32>::max() ||
        config.InputSizeBytes() > instance.GetDevice().maxBufferLength ||
        config.LdsSizeBytes() > instance.GetDevice().maxBufferLength) {
        throw std::runtime_error("Metal tessellation dispatch exceeds native storage limits");
    }
    auto constants = ReadConstants(regs, config);
    if (layout !=
        TessellationWorkLayout{regs, constants}.Slice(layout.FirstGroup(), config.workgroups)) {
        throw std::runtime_error("Metal tessellation dispatch does not match guest work layout");
    }
    ValidateInputs(layout, Shader::HwStage::Local, local_inputs);
    ValidateInputs(layout, Shader::HwStage::Hull, hull_inputs);
    auto lds = std::make_shared<StagingBufferPool::Allocation>(
        staging.AllocatePrivate(config.LdsSizeBytes(), alignof(u64)));
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    commands.FillBuffer({lds->GetBuffer(), lds->GetOffset(), lds->SizeBytes()}, 0);
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch);
    RunStage(regs, Shader::HwStage::Local, layout, constants, local_inputs, lds, neo_mode,
             attachments);
    constants = ReadConstants(regs, config);
    auto hull = RunStage(regs, Shader::HwStage::Hull, layout, constants, hull_inputs, lds, neo_mode,
                         attachments);
    return {config, std::move(lds), std::move(hull.factors), std::move(hull.offchip),
            layout.FirstGroup()};
}

} // namespace Metal
