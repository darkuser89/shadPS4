// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "common/alignment.h"
#include "core/libraries/kernel/memory.h"
#include "core/memory.h"
#include "shader_recompiler/ir/attribute.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_metal/metal_guest_resources.h"
#include "video_core/renderer_metal/metal_htile.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {

u64 MappedGuestBufferSize(Core::MemoryManager& memory, VAddr address, u64 requested_size) {
    constexpr u64 AddressSpaceSize = u64{1} << 40;
    if (!requested_size) {
        return 0;
    }
    if (!address || address >= AddressSpaceSize || requested_size > AddressSpaceSize - address) {
        throw std::runtime_error("Invalid Metal guest mapped buffer range");
    }
    u64 size{};
    while (size < requested_size) {
        const VAddr current = address + size;
        Libraries::Kernel::OrbisVirtualQueryInfo mapping{};
        if (memory.VirtualQuery(current, 0, &mapping) != 0 || !mapping.is_committed ||
            mapping.start > current || mapping.end <= current) {
            break;
        }
        size += std::min<u64>(requested_size - size, mapping.end - current);
    }
    return size;
}

bool ScalarReadOverlaps(Core::MemoryManager& memory, const GeometryBufferByteInterval& read,
                        VAddr address, u64 size) {
    if (!size) {
        return false;
    }
    if (address >= (u64{1} << 40) || size > (u64{1} << 40) - address) {
        throw std::runtime_error("Invalid Metal scalar alias extent");
    }
    if (read.lower < address + size && address <= read.upper) {
        return true;
    }
    const auto source = memory.GetBackingRanges(read.lower, read.upper - read.lower + 1);
    const auto target = memory.GetBackingRanges(address, size);
    for (const auto& left : source) {
        if (!left.physical_address) {
            throw std::runtime_error("Metal scalar read has no physical backing");
        }
        for (const auto& right : target) {
            if (right.physical_address &&
                *left.physical_address < *right.physical_address + right.size &&
                *right.physical_address < *left.physical_address + left.size) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

bool GuestResources::Stage::HasWrites() const noexcept {
    return std::ranges::any_of(buffers, [](const auto& buffer) { return buffer.written; }) ||
           std::ranges::any_of(images, [](const auto& image) { return image.written; });
}

GuestResources::GuestResources(const Instance& instance_, Scheduler& scheduler_,
                               BufferCache& buffer_cache_, TextureCache& texture_cache_,
                               AmdGpu::Liverpool& liverpool_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, liverpool{liverpool_}, memory{*Core::Memory::Instance()},
      staging{instance, scheduler}, commands{instance, scheduler}, samplers{instance} {}

std::unique_ptr<GuestResources::Stage> GuestResources::Prepare(
    ShaderCache::Variant& variant, bool neo_mode, std::span<const GeometryBuffer> geometry_buffers,
    u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
    std::span<const GeometryAddressInput> geometry_inputs) {
    if ((!guest_extents.empty() || !geometry_inputs.empty()) && !variant.geometry &&
        !variant.geometry_copy && !variant.geometry_export && !variant.tessellation) {
        throw std::runtime_error("Metal guest buffer extents require a geometry kernel");
    }
    auto it = bindings.find(&variant);
    if (it == bindings.end()) {
        it = bindings.emplace(&variant, std::make_unique<ShaderBindings>(instance, variant.shader))
                 .first;
    }
    auto stage = std::make_unique<Stage>();
    stage->owner = this;
    stage->variant = &variant;
    stage->bindings = it->second.get();
    for (const auto& extent : guest_extents) {
        if (extent.resource_index >= variant.info.buffers.size() ||
            variant.info.buffers[extent.resource_index].buffer_type != Shader::BufferType::Guest) {
            throw std::runtime_error("Metal guest buffer extent requires a guest resource");
        }
        const auto sharp =
            ReadGuestBufferSharp(variant.info.buffers[extent.resource_index], variant.info);
        const u64 descriptor_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
        const auto size = GuestBufferSize(extent.resource_index, descriptor_size, guest_extents);
        if (!sharp.base_address ||
            MappedGuestBufferSize(memory, sharp.base_address, size) != size) {
            throw std::runtime_error("Metal geometry guest buffer extent is not fully mapped");
        }
    }
    stage->flattened_user_data = variant.info.flattened_ud_buf;
    stage->geometry_inputs.assign(geometry_inputs.begin(), geometry_inputs.end());
    if (variant.info.sw_stage == Shader::SwStage::Compute && !variant.geometry &&
        !variant.geometry_copy && !variant.geometry_export && !variant.tessellation) {
        const auto& group_size = variant.specialization.runtime_info.hw.cs.workgroup_size;
        u64 num_threads = 1;
        for (u32 component = 0; component < group_size.size(); ++component) {
            const u32 size = group_size[component];
            if (!size || num_threads > std::numeric_limits<u32>::max() / size) {
                throw std::runtime_error("Invalid Metal compute workgroup dimensions");
            }
            stage->geometry_inputs.push_back(
                {Shader::IR::Attribute::LocalInvocationId, component, {0, size - 1}});
            num_threads *= size;
        }
        stage->geometry_inputs.push_back(
            {Shader::IR::Attribute::LocalInvocationIndex, 0, {0, u32(num_threads - 1)}});
    }
    std::vector<GuestBufferExtent> binding_extents{guest_extents.begin(), guest_extents.end()};
    if (variant.shader.guest_buffer_bindings) {
        const auto bounds = variant.shader.guest_buffer_bindings->Evaluate(
            stage->geometry_inputs, stage->flattened_user_data);
        for (const auto& bound : bounds) {
            if (!bound.reachable || !bound.bytes ||
                (!stage->bindings->UsesBinding(variant.specialization.start.unified +
                                               bound.resource_index) &&
                 !variant.shader.paged_buffers.test(bound.resource_index))) {
                continue;
            }
            const auto sharp =
                ReadGuestBufferSharp(variant.info.buffers[bound.resource_index], variant.info);
            if (!sharp.num_records) {
                continue;
            }
            constexpr u64 AddressSpaceSize = u64{1} << 40;
            if (!sharp.base_address || sharp.base_address >= AddressSpaceSize) {
                throw std::runtime_error("Invalid Metal guest binding address");
            }
            const auto limit = std::min<u64>(std::numeric_limits<u32>::max(),
                                             AddressSpaceSize - sharp.base_address);
            const auto size = bound.bytes->upper >= limit ? limit : bound.bytes->upper + 1;
            const u64 descriptor_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
            if (size <= GuestBufferSize(bound.resource_index, descriptor_size, binding_extents)) {
                continue;
            }
            const auto extent = std::ranges::find(binding_extents, bound.resource_index,
                                                  &GuestBufferExtent::resource_index);
            if (extent == binding_extents.end()) {
                binding_extents.push_back({bound.resource_index, size});
            } else {
                extent->size = size;
            }
        }
    }
    if (variant.geometry) {
        stage->workgroups = {variant.geometry->emissions.invocations / 64, 1, 1};
    } else if (variant.geometry_copy) {
        stage->workgroups = {variant.geometry_copy->work_items / 64, 1, 1};
    } else if (variant.geometry_export) {
        if (!variant.shader.geometry_export) {
            throw std::runtime_error("Native export dispatch requires its compiled bindings");
        }
        const auto input =
            std::ranges::find(geometry_buffers, variant.shader.geometry_export->inputs,
                              &GeometryBuffer::resource_index);
        if (input == geometry_buffers.end() || !input->storage) {
            throw std::runtime_error("Native export dispatch is missing its input allocation");
        }
        stage->workgroups = {
            variant.geometry_export->WorkItemsFromInputBytes(input->storage->SizeBytes()) / 64, 1,
            1};
    } else if (variant.tessellation) {
        stage->workgroups = {variant.tessellation->workgroups, 1, 1};
    } else {
        stage->workgroups = liverpool.GetCsRegs().WorkgroupExtent();
        const auto start = liverpool.GetCsRegs().WorkgroupStart();
        std::copy(start.begin(), start.end(), stage->dispatch_base.begin());
    }
    if (variant.geometry || variant.geometry_copy || variant.geometry_export) {
        stage->dispatch_base[3] =
            GeometryInitialActiveCount(stage->workgroups[0] * 64, initial_active_work_items);
    } else if (initial_active_work_items) {
        throw std::runtime_error("Metal initial active count requires a geometry kernel");
    }
    for (const auto& supplied : geometry_buffers) {
        if (supplied.resource_index >= variant.info.buffers.size() || !supplied.storage) {
            throw std::runtime_error("Invalid Metal geometry buffer resource");
        }
        const auto type = variant.info.buffers[supplied.resource_index].buffer_type;
        u64 required_size{};
        switch (type) {
        case Shader::BufferType::GeometryInputs:
            required_size = variant.geometry        ? variant.geometry->InputSizeBytes()
                            : variant.geometry_copy ? variant.geometry_copy->InputSizeBytes()
                            : variant.geometry_export
                                ? variant.geometry_export->InputSizeBytes(stage->workgroups[0] * 64)
                            : variant.tessellation ? variant.tessellation->InputSizeBytes()
                                                   : 0;
            break;
        case Shader::BufferType::GeometryEmissions:
            required_size = variant.geometry ? variant.geometry->emissions.EmissionSizeBytes()
                            : variant.geometry_copy
                                ? variant.geometry_copy->emissions.EmissionSizeBytes()
                                : 0;
            break;
        case Shader::BufferType::GeometryPayload:
            required_size =
                variant.geometry_export && variant.geometry_export->raster_output &&
                        variant.shader.geometry_export && variant.shader.geometry_export->raster &&
                        variant.shader.geometry_export->raster->output == supplied.resource_index
                    ? variant.shader.geometry_export->RasterSizeBytes(*variant.geometry_export,
                                                                      stage->workgroups[0] * 64)
                : variant.tessellation && variant.shader.tessellation &&
                        variant.shader.tessellation->raster &&
                        variant.shader.tessellation->raster->output == supplied.resource_index
                    ? u64{variant.tessellation->workgroups} *
                          variant.tessellation->threads_per_group *
                          variant.shader.tessellation->raster->vertex_words * sizeof(u32)
                : variant.shader.geometry_copy
                    ? variant.shader.geometry_copy->payload.EmissionSizeBytes()
                    : 0;
            break;
        case Shader::BufferType::SharedMemory:
            required_size = variant.tessellation && variant.shader.tessellation &&
                                    variant.shader.tessellation->lds == supplied.resource_index
                                ? variant.tessellation->LdsSizeBytes()
                                : 0;
            break;
        default:
            throw std::runtime_error("Metal geometry buffer cannot replace guest resources");
        }
        const BufferRange range{supplied.storage->GetBuffer(), supplied.storage->GetOffset(),
                                supplied.storage->SizeBytes()};
        if (!required_size || !range.buffer || range.buffer.device != instance.GetDevice() ||
            range.offset % sizeof(u32) || range.offset > range.buffer.length ||
            range.size < required_size || range.size > range.buffer.length - range.offset ||
            !stage->geometry_buffers.emplace(supplied.resource_index, range).second) {
            throw std::runtime_error("Invalid Metal geometry buffer capacity or duplicate");
        }
        stage->uploads.push_back(supplied.storage);
    }
    const auto& regs = liverpool.regs;
    stage->push.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    stage->push.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    stage->push.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    stage->push.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    const auto& info = variant.info;
    auto next = variant.specialization.start;
    for (u32 resource_index = 0; resource_index < info.buffers.size(); ++resource_index) {
        const auto& resource = info.buffers[resource_index];
        const u32 binding = next.unified++;
        const u32 offset_binding = next.buffer++;
        if (!stage->bindings->UsesBinding(binding) &&
            !variant.shader.paged_buffers.test(resource_index)) {
            continue;
        }
        Stage::Buffer buffer{binding, resource.buffer_type, resource_index};
        buffer.written = resource.is_written;
        const bool tessellation_errors =
            variant.shader.tessellation && variant.shader.tessellation->errors == resource_index;
        if (!tessellation_errors &&
            (resource.buffer_type == Shader::BufferType::GeometryInputs ||
             resource.buffer_type == Shader::BufferType::GeometryEmissions ||
             resource.buffer_type == Shader::BufferType::GeometryPayload ||
             (variant.shader.tessellation && variant.shader.tessellation->lds == resource_index)) &&
            !stage->geometry_buffers.contains(resource_index)) {
            throw std::runtime_error("Metal geometry shader requires its native buffers");
        }
        if (resource.buffer_type == Shader::BufferType::Guest) {
            if (stage->bindings->UsesBinding(binding) ||
                !variant.shader.paged_buffers.test(resource_index)) {
                throw std::runtime_error("Metal guest buffers require native paged bindings");
            }
            if (offset_binding >= Shader::NUM_BUFFERS) {
                throw std::runtime_error("Metal guest buffer offsets exceed the shader ABI");
            }
            const auto sharp = ReadGuestBufferSharp(resource, info);
            const u64 descriptor_size = u64{sharp.num_records} * std::max<u64>(sharp.stride, 1);
            const auto requested_size =
                GuestBufferSize(resource_index, descriptor_size, binding_extents);
            const bool empty_guarded_buffer =
                !requested_size && variant.shader.guarded_buffers.test(resource_index);
            if (empty_guarded_buffer) {
                stage->buffers.push_back(buffer);
                continue;
            }
            if (!sharp.base_address || !requested_size) {
                throw std::runtime_error("Metal null guest buffer requires robust-access lowering");
            }
            const auto size = MappedGuestBufferSize(memory, sharp.base_address, requested_size);
            if (!size || size > std::numeric_limits<u32>::max()) {
                throw std::runtime_error("Invalid Metal shader guest buffer size");
            }
            buffer.address = sharp.base_address;
            buffer.size = u32(size);
            buffer_cache.PreparePagedBuffer(buffer.address, buffer.size);
        } else if (resource.buffer_type == Shader::BufferType::BdaPagetable ||
                   resource.buffer_type == Shader::BufferType::FaultBuffer) {
            throw std::runtime_error(
                "Metal guest address translation requires native DMA lowering");
        }
        stage->buffers.push_back(buffer);
    }
    for (u32 image_index = 0; image_index < info.images.size(); ++image_index) {
        const auto& resource = info.images[image_index];
        const u32 binding = next.unified;
        const u32 count = resource.NumBindings(info);
        next.unified += count;
        if (!stage->bindings->UsesBinding(binding)) {
            continue;
        }
        const auto sharp = resource.GetSharp(info);
        if (!sharp.Address()) {
            throw std::runtime_error("Metal null guest image requires robust-access lowering");
        }
        Stage::Image image{binding,
                           count,
                           ImageInfo::FromShader(sharp, resource, neo_mode),
                           ImageViewInfo::FromShader(sharp, resource),
                           resource.mip_fallback_mode,
                           resource.is_written,
                           resource.is_atomic,
                           variant.shader.query_only_images.test(image_index)};
        if (variant.shader.lowered_minimum_lod_images.test(image_index)) {
            image.view.min_lod = 0.f;
        }
        if (resource.mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
            image.view.base_level = resource.constant_mip_index;
            image.view.levels = 1;
        } else if (resource.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
            image.view.levels = 1;
        }
        if (!image.query_only) {
            buffer_cache.PreparePagedBuffer(image.info.guest_address, image.info.layout.guest_size);
        }
        stage->images.push_back(image);
    }
    u32 sampler_index{};
    for (const auto& resource : info.samplers) {
        const u32 binding = next.unified++;
        if (stage->bindings->UsesBinding(binding)) {
            auto sharp = resource.GetSharp(info);
            if (variant.shader.post_clamp_samplers.test(sampler_index)) {
                sharp.min_lod.Assign(0);
                sharp.max_lod.Assign(0xfff);
            }
            stage->samplers.push_back({binding, sharp, resource.is_depth,
                                       variant.shader.sampler_lod[sampler_index].native_bias});
        }
        ++sampler_index;
    }
    if (variant.geometry || variant.geometry_copy || variant.geometry_export ||
        variant.tessellation) {
        std::vector<GeometryBufferRange> ranges;
        for (const auto& buffer : stage->buffers) {
            if (buffer.type == Shader::BufferType::Guest) {
                ranges.push_back({buffer.resource_index, buffer.address, buffer.size});
            }
        }
        stage->geometry_snapshot = CaptureGeometryBufferSnapshot(
            info, variant.shader.geometry_buffers, ranges, binding_extents);
    }
    if (variant.shader.tessellation && variant.shader.tessellation->scalar_reads &&
        std::ranges::any_of(stage->buffers, [](const auto& buffer) {
            return buffer.type == Shader::BufferType::GuestPageTable;
        })) {
        stage->scalar_reads =
            ScalarReadRanges(*variant.shader.tessellation->scalar_reads, stage->geometry_inputs,
                             stage->flattened_user_data, instance.GetDevice().maxBufferLength);
        for (const auto& range : stage->scalar_reads) {
            const u64 size = range.upper - range.lower + 1;
            if (MappedGuestBufferSize(memory, range.lower, size) != size) {
                throw std::runtime_error("Metal scalar read requires fully mapped guest memory");
            }
            for (const auto& backing : memory.GetBackingRanges(range.lower, size)) {
                if (!backing.physical_address) {
                    throw std::runtime_error("Metal scalar read requires physical guest backing");
                }
            }
            buffer_cache.PreparePagedBuffer(range.lower, size);
        }
    }
    return stage;
}

std::vector<GeometryBufferLink> GuestResources::GeometryDependencies(const Stage& producer,
                                                                     const Stage& consumer) const {
    if (producer.owner != this || consumer.owner != this || !producer.geometry_snapshot ||
        !consumer.geometry_snapshot) {
        throw std::runtime_error("Native geometry dependencies require prepared geometry stages");
    }
    return LinkGeometryBufferSnapshots(*producer.geometry_snapshot, *consumer.geometry_snapshot);
}

void GuestResources::ValidateTessellationRing(const Stage& stage, u32 resource_index, VAddr address,
                                              u64 size,
                                              std::span<const ImageInfo> attachments) const {
    if (stage.owner != this || stage.acquired || !stage.variant->shader.tessellation || !size) {
        throw std::runtime_error("Invalid Metal tessellation ring resources");
    }
    const auto ring =
        std::ranges::find(stage.buffers, resource_index, &Stage::Buffer::resource_index);
    if (ring == stage.buffers.end() || ring->type != Shader::BufferType::Guest || !ring->written ||
        ring->address != address || ring->size < size) {
        throw std::runtime_error("Metal tessellation ring binding differs from its ring");
    }
    ValidateRingBacking(stage, resource_index, address, size, attachments);
}

void GuestResources::ReplayTessellationRing(
    Stage& stage, const GeometryBufferRange& origin,
    std::span<const GeometryBufferByteInterval> ranges,
    const std::shared_ptr<StagingBufferPool::Allocation>& source,
    std::span<const ImageInfo> attachments) {
    if (stage.owner != this || stage.acquired || !stage.variant->tessellation ||
        !stage.variant->shader.tessellation || !stage.variant->tessellation->compact_evaluation ||
        stage.variant->specialization.runtime_info.sw_stage != Shader::SwStage::TessellationEval ||
        ranges.size() != stage.variant->tessellation->workgroups ||
        origin.resource_index > Shader::NUM_BUFFERS || !origin.address || origin.address % 4 ||
        origin.address >= (u64{1} << 40) || !origin.size ||
        origin.size > (u64{1} << 40) - origin.address || !source || !source->IsReady() ||
        !source->GetBuffer() || source->GetBuffer().device != instance.GetDevice() ||
        source->GetOffset() % 4 || source->GetOffset() > source->GetBuffer().length ||
        !source->SizeBytes() ||
        source->SizeBytes() > source->GetBuffer().length - source->GetOffset()) {
        throw std::runtime_error("Invalid native Metal retained offchip TES replay");
    }
    u64 bytes{};
    u64 required{};
    for (const auto& range : ranges) {
        if (range.lower % 4 || range.lower > range.upper || range.upper >= origin.size ||
            (range.upper - range.lower + 1) % 4 || bytes > source->SizeBytes() ||
            range.upper - range.lower + 1 > source->SizeBytes() - bytes) {
            throw std::runtime_error("Metal offchip TES replay leaves its retained source");
        }
        bytes += range.upper - range.lower + 1;
        required = std::max(required, range.upper + 1);
    }
    auto ordered = std::vector<GeometryBufferByteInterval>{ranges.begin(), ranges.end()};
    std::ranges::sort(ordered, {}, &GeometryBufferByteInterval::lower);
    for (size_t index = 1; index < ordered.size(); ++index) {
        if (ordered[index].lower <= ordered[index - 1].upper) {
            throw std::runtime_error("Metal offchip TES replay blocks overlap");
        }
    }
    if (bytes != source->SizeBytes()) {
        throw std::runtime_error("Metal offchip TES replay does not cover its packed source");
    }
    if (origin.resource_index != Shader::NUM_BUFFERS) {
        const auto target =
            std::ranges::find(stage.buffers, origin.resource_index, &Stage::Buffer::resource_index);
        if (target == stage.buffers.end() || target->type != Shader::BufferType::Guest ||
            target->written || target->address != origin.address || target->size < required) {
            throw std::runtime_error("Metal offchip TES binding differs from its producer ring");
        }
    }
    ValidateRingBacking(stage, origin.resource_index, origin.address, origin.size, attachments);
    if (origin.resource_index == Shader::NUM_BUFFERS) {
        return;
    }
    std::vector<BufferRange> destinations;
    for (const auto& range : ranges) {
        const u64 size = range.upper - range.lower + 1;
        buffer_cache.ObtainPagedBuffer(origin.address + range.lower, size, true);
        const auto pages = buffer_cache.GetPagedBufferRanges(origin.address + range.lower, size);
        u64 total{};
        for (const auto& page : pages) {
            if (!page.size || total > size || page.size > size - total ||
                (page.buffer == source->GetBuffer() &&
                 source->GetOffset() < page.offset + page.size &&
                 page.offset < source->GetOffset() + source->SizeBytes())) {
                throw std::runtime_error("Metal offchip TES replay destination aliases its source");
            }
            total += page.size;
            destinations.push_back(page);
        }
        if (total != size) {
            throw std::runtime_error("Metal offchip TES replay destination is incomplete");
        }
    }
    scheduler.RetainLifetime(source);
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    u64 offset{};
    for (const auto& destination : destinations) {
        commands.CopyBuffer({source->GetBuffer(), source->GetOffset() + offset, destination.size},
                            destination, destination.size);
        offset += destination.size;
    }
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
}

void GuestResources::ValidateRingBacking(const Stage& stage, u32 resource_index, VAddr address,
                                         u64 size, std::span<const ImageInfo> attachments) const {
    if (!size || !address || address >= (u64{1} << 40) || size > (u64{1} << 40) - address) {
        throw std::runtime_error("Invalid Metal native ring backing extent");
    }
    const auto backing = memory.GetBackingRanges(address, size);
    std::vector<GeometryBufferByteInterval> physical;
    for (const auto& range : backing) {
        if (!range.physical_address) {
            throw std::runtime_error("Metal tessellation ring must remain fully mapped");
        }
        physical.push_back({*range.physical_address, *range.physical_address + range.size - 1});
    }
    std::ranges::sort(physical, {}, &GeometryBufferByteInterval::lower);
    for (size_t index = 1; index < physical.size(); ++index) {
        if (physical[index].lower <= physical[index - 1].upper) {
            throw std::runtime_error("Metal native ring aliases its own physical backing");
        }
    }
    const auto validate = [&](VAddr other_address, u64 other_size) {
        if (!other_size) {
            return;
        }
        constexpr u64 AddressSpaceSize = u64{1} << 40;
        if (!other_address || other_address >= AddressSpaceSize ||
            other_size > AddressSpaceSize - other_address) {
            throw std::runtime_error("Invalid Metal tessellation resource range");
        }
        for (const auto& other : memory.GetBackingRanges(other_address, other_size)) {
            if (!other.physical_address) {
                continue;
            }
            for (const auto& range : backing) {
                if (*range.physical_address < *other.physical_address + other.size &&
                    *other.physical_address < *range.physical_address + range.size) {
                    throw std::runtime_error(
                        "Metal tessellation ring alias requires ordered resource lowering");
                }
            }
        }
    };
    const auto validate_image = [&](const ImageInfo& info) {
        validate(info.guest_address, info.layout.guest_size);
        if (info.cmask_address) {
            if (!info.cmask_slice_bytes ||
                info.layers > std::numeric_limits<u64>::max() / info.cmask_slice_bytes) {
                throw std::runtime_error("Invalid Metal tessellation CMASK alias extent");
            }
            validate(info.cmask_address, info.cmask_slice_bytes * info.layers);
        }
        if (info.fmask_address) {
            const auto fmask = ImageInfo::FromFmask(info);
            validate(fmask.guest_address, fmask.layout.guest_size);
        }
        if (info.htile_address) {
            validate(info.htile_address, Htile::MetadataSize(info));
        }
    };
    for (const auto& buffer : stage.buffers) {
        if (buffer.type == Shader::BufferType::Guest && buffer.resource_index != resource_index) {
            validate(buffer.address, buffer.size);
        }
    }
    for (const auto& read : stage.scalar_reads) {
        validate(read.lower, read.upper - read.lower + 1);
    }
    for (const auto& image : stage.images) {
        validate_image(image.info);
    }
    for (const auto& attachment : attachments) {
        validate_image(attachment);
    }
}

void GuestResources::ValidateGeometryOutputRing(const Stage& stage, u32 resource_index,
                                                VAddr address, u64 size,
                                                std::span<const ImageInfo> attachments) const {
    if (stage.owner != this || stage.acquired || !stage.variant->shader.geometry || !size) {
        throw std::runtime_error("Invalid Metal geometry output ring resources");
    }
    const auto ring =
        std::ranges::find(stage.buffers, resource_index, &Stage::Buffer::resource_index);
    if (ring == stage.buffers.end() || ring->type != Shader::BufferType::Guest || !ring->written ||
        ring->address != address || ring->size < size) {
        throw std::runtime_error("Metal geometry output binding differs from its ring");
    }
    const auto& program = stage.variant->shader.geometry->address_program;
    if (!program) {
        throw std::runtime_error("Metal geometry output ring requires native address analysis");
    }
    bool written{};
    for (const auto& access : program->Evaluate(stage.geometry_inputs, stage.flattened_user_data)) {
        if (!access.reachable || access.resource_index != resource_index) {
            continue;
        }
        if (!access.write || access.read || !access.bytes ||
            access.bytes->lower > access.bytes->upper || access.bytes->upper >= size) {
            throw std::runtime_error("Metal geometry output access leaves its bounded ring");
        }
        written = true;
    }
    if (!written) {
        throw std::runtime_error("Metal geometry output ring has no proven writes");
    }
    ValidateRingBacking(stage, resource_index, address, size, attachments);
}

void GuestResources::ValidateGeometryCopyRing(const Stage& geometry, const Stage& copy, u64 size,
                                              std::span<const ImageInfo> attachments) const {
    if (geometry.owner != this || copy.owner != this || geometry.acquired || copy.acquired ||
        !geometry.variant->shader.geometry || !copy.variant->shader.geometry_copy || !size) {
        throw std::runtime_error("Invalid native geometry/copy ring sequence");
    }
    const auto& source = *geometry.variant->shader.geometry;
    const auto& destination = *copy.variant->shader.geometry_copy;
    if (!source.address_origins || !destination.address_origins || !destination.address_program) {
        throw std::runtime_error("Native geometry/copy requires address analysis");
    }
    const Stage::Buffer* output{};
    for (const auto& buffer : geometry.buffers) {
        if (buffer.type != Shader::BufferType::Guest ||
            buffer.resource_index >= Shader::NUM_BUFFERS ||
            !(source.address_origins->writes[buffer.resource_index] &
              (u64{1} << u32(GeometryAddressSource::RingOffset)))) {
            continue;
        }
        if (output || !buffer.written || buffer.size < size) {
            throw std::runtime_error("Native GSVS producer ring is ambiguous or too small");
        }
        output = &buffer;
    }
    if (!output) {
        throw std::runtime_error("Native GSVS copy requires its producer ring");
    }
    const Stage::Buffer* input{};
    constexpr u64 copy_vector = u64{1} << u32(GeometryAddressSource::CopyVector);
    for (const auto& buffer : copy.buffers) {
        if (buffer.type != Shader::BufferType::Guest ||
            buffer.resource_index >= Shader::NUM_BUFFERS ||
            !(destination.address_origins->reads[buffer.resource_index] & copy_vector)) {
            continue;
        }
        const auto sharp = ReadGuestBufferSharp(copy.variant->info.buffers[buffer.resource_index],
                                                copy.variant->info);
        if (input || buffer.written || buffer.address != output->address || buffer.size < size ||
            !sharp.Valid() || sharp.base_address != output->address || sharp.stride ||
            sharp.swizzle_enable || sharp.add_tid_enable || sharp.num_records < size ||
            (destination.address_origins->reads[buffer.resource_index] & ~copy_vector) ||
            destination.address_origins->writes[buffer.resource_index]) {
            throw std::runtime_error("Native GSVS copy does not match its producer descriptor");
        }
        input = &buffer;
    }
    if (!input) {
        throw std::runtime_error("Native GSVS copy has no proven consumer ring");
    }
    bool read{};
    for (const auto& access :
         destination.address_program->Evaluate(copy.geometry_inputs, copy.flattened_user_data)) {
        if (!access.reachable || access.resource_index != input->resource_index) {
            continue;
        }
        if (!access.read || access.write || !access.bytes ||
            access.bytes->lower > access.bytes->upper || access.bytes->upper >= size) {
            throw std::runtime_error("Native GSVS copy read leaves its producer ring");
        }
        read = true;
    }
    if (!read) {
        throw std::runtime_error("Native GSVS copy has no bounded read access");
    }
    ValidateRingBacking(copy, input->resource_index, input->address, size, attachments);
}

void GuestResources::ReplayGeometryRing(
    Stage& geometry, const GeometryBufferRange& origin,
    const std::shared_ptr<StagingBufferPool::Allocation>& source,
    std::span<Stage* const> companions, std::span<const ImageInfo> attachments) {
    if (geometry.owner != this || geometry.acquired || !geometry.variant->shader.geometry ||
        !geometry.variant->geometry || !geometry.variant->geometry->compact_workgroups || !source ||
        !source->IsReady() || !source->GetBuffer() ||
        source->GetBuffer().device != instance.GetDevice() || !source->SizeBytes() ||
        source->GetOffset() % sizeof(u32) || source->SizeBytes() % sizeof(u32) ||
        source->GetOffset() > source->GetBuffer().length ||
        source->SizeBytes() > source->GetBuffer().length - source->GetOffset() ||
        origin.resource_index >= Shader::NUM_BUFFERS || !origin.address ||
        origin.address % sizeof(u32) || origin.address >= (u64{1} << 40) ||
        origin.size > (u64{1} << 40) - origin.address || source->SizeBytes() > origin.size) {
        throw std::runtime_error("Invalid Metal retained geometry ring replay");
    }
    const auto& shader = geometry.variant->shader;
    const auto& bindings = *shader.geometry;
    if (!bindings.address_origins || !bindings.address_program) {
        throw std::runtime_error("Metal geometry replay requires native address analysis");
    }
    constexpr u64 vertex_sources = u64{0x3f} << u32(GeometryAddressSource::VertexOffset);
    const Stage::Buffer* target{};
    for (const auto& buffer : geometry.buffers) {
        if (buffer.type != Shader::BufferType::Guest) {
            continue;
        }
        if (buffer.resource_index >= Shader::NUM_BUFFERS) {
            throw std::runtime_error("Metal geometry input ring index exceeds address metadata");
        }
        const auto reads = bindings.address_origins->reads[buffer.resource_index];
        if (!(reads & vertex_sources)) {
            continue;
        }
        const auto sharp = ReadGuestBufferSharp(
            geometry.variant->info.buffers[buffer.resource_index], geometry.variant->info);
        if (target || buffer.written || (reads & ~vertex_sources) ||
            bindings.address_origins->writes[buffer.resource_index] || !sharp.Valid() ||
            sharp.base_address != origin.address || buffer.address != origin.address ||
            sharp.stride || sharp.swizzle_enable || sharp.add_tid_enable ||
            source->SizeBytes() > buffer.size || source->SizeBytes() > sharp.num_records) {
            throw std::runtime_error(
                "Metal geometry input ring differs from its retained producer");
        }
        target = &buffer;
    }
    if (!target) {
        throw std::runtime_error("Metal geometry replay has no proven consumer ring");
    }
    bool accessed{};
    for (const auto& bound : bindings.address_program->Evaluate(geometry.geometry_inputs,
                                                                geometry.flattened_user_data)) {
        if (!bound.reachable || bound.resource_index != target->resource_index) {
            continue;
        }
        if (!bound.read || bound.write || !bound.bytes || bound.bytes->lower > bound.bytes->upper ||
            bound.bytes->upper >= source->SizeBytes()) {
            throw std::runtime_error("Metal geometry access leaves the replayed export extent");
        }
        accessed = true;
    }
    if (!accessed) {
        throw std::runtime_error("Metal geometry replay requires a bounded read access");
    }
    ValidateRingBacking(geometry, target->resource_index, origin.address, source->SizeBytes(),
                        attachments);
    for (size_t index = 0; index < companions.size(); ++index) {
        const auto* stage = companions[index];
        if (!stage || stage == &geometry || stage->owner != this || stage->acquired ||
            std::find(companions.begin(), companions.begin() + index, stage) !=
                companions.begin() + index) {
            throw std::runtime_error("Invalid Metal geometry replay companion stage");
        }
        ValidateRingBacking(*stage, Shader::NUM_BUFFERS, origin.address, source->SizeBytes(), {});
    }
    buffer_cache.ObtainPagedBuffer(origin.address, source->SizeBytes(), true);
    const auto destinations =
        buffer_cache.GetPagedBufferRanges(origin.address, source->SizeBytes());
    u64 total{};
    for (const auto& destination : destinations) {
        if (!destination.size || total > source->SizeBytes() ||
            destination.size > source->SizeBytes() - total ||
            (destination.buffer == source->GetBuffer() &&
             source->GetOffset() + total < destination.offset + destination.size &&
             destination.offset < source->GetOffset() + total + destination.size)) {
            throw std::runtime_error("Metal geometry replay overlaps or exceeds its source");
        }
        total += destination.size;
    }
    if (total != source->SizeBytes()) {
        throw std::runtime_error("Metal geometry replay destination is incomplete");
    }
    scheduler.RetainLifetime(source);
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageAll, MTLStageBlit);
    u64 offset{};
    for (const auto& destination : destinations) {
        commands.CopyBuffer({source->GetBuffer(), source->GetOffset() + offset, destination.size},
                            destination, destination.size);
        offset += destination.size;
    }
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
}

void GuestResources::ValidateAliases(std::span<Stage* const> stages) const {
    const auto overlaps = [](VAddr left, u64 left_size, VAddr right, u64 right_size) {
        return left_size && right_size && left < right + right_size && right < left + left_size;
    };
    for (size_t index = 0; index < stages.size(); ++index) {
        const Stage* stage = stages[index];
        if (!stage || stage->owner != this || stage->acquired) {
            throw std::runtime_error("Invalid prepared Metal shader resources");
        }
        if (std::find(stages.begin(), stages.begin() + index, stage) != stages.begin() + index) {
            throw std::runtime_error("Duplicate prepared Metal shader stage");
        }
        if (stage->geometry_snapshot) {
            for (size_t other_index = index + 1; other_index < stages.size(); ++other_index) {
                const Stage* other = stages[other_index];
                if (!other || other->owner != this) {
                    throw std::runtime_error("Metal geometry resources belong to another owner");
                }
                if (other->geometry_snapshot && !GeometryDependencies(*stage, *other).empty()) {
                    throw std::runtime_error(
                        "Simultaneous Metal geometry dependencies require ordered dispatch");
                }
            }
        }
        for (const auto& read : stage->scalar_reads) {
            for (const Stage* other : stages) {
                if (!other || other->owner != this) {
                    throw std::runtime_error("Metal scalar resources belong to another owner");
                }
                for (const auto& buffer : other->buffers) {
                    if (buffer.type == Shader::BufferType::Guest && buffer.written &&
                        ScalarReadOverlaps(memory, read, buffer.address, buffer.size)) {
                        throw std::runtime_error(
                            "Metal scalar read/write aliases require ordered native lowering");
                    }
                }
                for (const auto& image : other->images) {
                    if (image.written && ScalarReadOverlaps(memory, read, image.info.guest_address,
                                                            image.info.layout.guest_size)) {
                        throw std::runtime_error(
                            "Metal scalar/image aliases require ordered native lowering");
                    }
                }
            }
        }
        for (const auto& image : stage->images) {
            for (const Stage* other : stages) {
                if (!other || other->owner != this) {
                    throw std::runtime_error("Metal shader resources belong to a different owner");
                }
                for (const auto& buffer : other->buffers) {
                    if (buffer.type == Shader::BufferType::Guest &&
                        (buffer.written || image.written) &&
                        overlaps(buffer.address, buffer.size, image.info.guest_address,
                                 image.info.layout.guest_size)) {
                        throw std::runtime_error(
                            "Simultaneous Metal buffer/image writes require alias lowering");
                    }
                }
                for (const auto& alias : other->images) {
                    if ((image.written || alias.written) &&
                        (image.info != alias.info || (stage == other && &image != &alias)) &&
                        overlaps(image.info.guest_address, image.info.layout.guest_size,
                                 alias.info.guest_address, alias.info.layout.guest_size)) {
                        throw std::runtime_error(
                            "Simultaneous Metal image writes require alias lowering");
                    }
                }
            }
        }
        for (const auto& [resource_index, range] : stage->geometry_buffers) {
            for (const Stage* other : stages) {
                if (!other || other->owner != this) {
                    throw std::runtime_error("Metal geometry resources belong to another owner");
                }
                for (const auto& [other_index, alias] : other->geometry_buffers) {
                    if (stage == other && resource_index == other_index) {
                        continue;
                    }
                    if (range.buffer == alias.buffer &&
                        (stage->variant->info.buffers[resource_index].is_written ||
                         other->variant->info.buffers[other_index].is_written) &&
                        overlaps(range.offset, range.size, alias.offset, alias.size)) {
                        throw std::runtime_error(
                            "Overlapping Metal geometry buffers require lowering");
                    }
                }
            }
        }
    }
}

BufferRange GuestResources::Upload(Stage& stage, u32 binding, std::span<const u8> data) {
    if (data.empty()) {
        throw std::runtime_error("Metal shader upload requires data");
    }
    auto& cached = upload_cache[stage.bindings][binding];
    auto allocation = cached.lock();
    if (!allocation || !allocation->IsCurrentTick() || allocation->SizeBytes() != data.size() ||
        std::memcmp(allocation->UploadData().data(), data.data(), data.size()) != 0) {
        allocation =
            std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(data.size(), 16));
        std::memcpy(allocation->UploadData().data(), data.data(), data.size());
        cached = allocation;
    }
    const BufferRange range{allocation->GetBuffer(), allocation->GetOffset(), data.size()};
    stage.uploads.push_back(std::move(allocation));
    return range;
}

BufferRange GuestResources::SharedMemory(Stage& stage, u64 stride) {
    u64 size = stride;
    for (u32 count : stage.workgroups) {
        if (!count || size > std::numeric_limits<u64>::max() / count) {
            throw std::runtime_error("Invalid Metal guest shared-memory workgroup count");
        }
        size *= count;
    }
    auto allocation =
        std::make_shared<StagingBufferPool::Allocation>(staging.AllocatePrivate(size));
    const BufferRange range{allocation->GetBuffer(), allocation->GetOffset(), size};
    stage.uploads.push_back(std::move(allocation));
    (void)scheduler.BeginCompute();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
    commands.FillBuffer(range, 0);
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch | MTLStageBlit);
    return range;
}

void GuestResources::BindBuffer(Stage& stage, const Stage::Buffer& buffer) {
    BufferRange range;
    switch (buffer.type) {
    case Shader::BufferType::Guest: {
        stage.buffer_addresses.at(buffer.resource_index) = buffer.address;
        if (buffer.size) {
            buffer_cache.ObtainPagedBuffer(buffer.address, buffer.size, buffer.written);
            stage.buffer_capacities.at(buffer.resource_index) = buffer.size;
        }
        if (stage.geometry_snapshot) {
            stage.geometry_address_bindings.push_back(
                {buffer.resource_index, buffer.address, buffer.size, 0, 0, true});
        }
        return;
    }
    case Shader::BufferType::GdsBuffer:
        range = buffer_cache.GetGdsBuffer();
        break;
    case Shader::BufferType::BufferBounds: {
        std::array<u32, Shader::NUM_BUFFERS * 3> metadata{};
        std::copy(stage.buffer_capacities.begin(), stage.buffer_capacities.end(), metadata.begin());
        for (u32 index = 0; index < stage.buffer_addresses.size(); ++index) {
            metadata[ShaderCompiler::BufferAddressWordOffset + index * 2] =
                u32(stage.buffer_addresses[index]);
            metadata[ShaderCompiler::BufferAddressWordOffset + index * 2 + 1] =
                u32(stage.buffer_addresses[index] >> 32);
        }
        range = Upload(stage, buffer.binding,
                       {reinterpret_cast<const u8*>(metadata.data()), sizeof(metadata)});
        break;
    }
    case Shader::BufferType::GuestPageTable:
        if (!stage.page_snapshot) {
            throw std::runtime_error("Native scalar page read requires a page snapshot");
        }
        range = stage.page_snapshot->GetRoot();
        break;
    case Shader::BufferType::Flatbuf: {
        const auto& data = stage.flattened_user_data;
        range = Upload(stage, buffer.binding,
                       {reinterpret_cast<const u8*>(data.data()), data.size() * sizeof(u32)});
        break;
    }
    case Shader::BufferType::DispatchBase: {
        const auto& start = stage.dispatch_base;
        range = Upload(stage, buffer.binding,
                       {reinterpret_cast<const u8*>(start.data()), sizeof(start)});
        break;
    }
    case Shader::BufferType::ClipPlanes: {
        std::array<u32, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
        for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
            const auto& plane = liverpool.regs.clip_user_data[i];
            planes[i * 4 + 0] = plane.data_x;
            planes[i * 4 + 1] = plane.data_y;
            planes[i * 4 + 2] = plane.data_z;
            planes[i * 4 + 3] = plane.data_w;
        }
        range = Upload(stage, buffer.binding,
                       {reinterpret_cast<const u8*>(planes.data()), sizeof(planes)});
        break;
    }
    case Shader::BufferType::SharedMemory:
        if (stage.variant->shader.tessellation &&
            stage.variant->shader.tessellation->lds == buffer.resource_index) {
            range = stage.geometry_buffers.at(buffer.resource_index);
        } else {
            range = SharedMemory(stage, u64{stage.variant->shared_memory_size} +
                                            stage.variant->info.shared_memory_scratch_size);
        }
        break;
    case Shader::BufferType::GeometryInputs:
    case Shader::BufferType::GeometryEmissions:
    case Shader::BufferType::GeometryPayload:
        if (stage.variant->shader.tessellation &&
            stage.variant->shader.tessellation->errors == buffer.resource_index) {
            if (!stage.tessellation_errors) {
                stage.tessellation_errors = std::make_shared<StagingBufferPool::Allocation>(
                    staging.AllocateReadback(sizeof(u32), alignof(u32)));
                stage.uploads.push_back(stage.tessellation_errors);
                const auto& error = *stage.tessellation_errors;
                (void)scheduler.BeginCompute();
                scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
                scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageBlit);
                commands.FillBuffer({error.GetBuffer(), error.GetOffset(), sizeof(u32)}, 0);
                scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
                scheduler.EncoderBarrier(MTLStageBlit, MTLStageDispatch);
            }
            const auto& error = *stage.tessellation_errors;
            range = {error.GetBuffer(), error.GetOffset(), sizeof(u32)};
        } else {
            range = stage.geometry_buffers.at(buffer.resource_index);
        }
        break;
    default:
        throw std::runtime_error("Unsupported Metal guest special buffer");
    }
    if (stage.bindings->UsesBinding(buffer.binding)) {
        stage.bindings->SetBuffer(buffer.binding, range.buffer, range.offset);
    }
}

void GuestResources::BindImage(Stage& stage, Stage::Image& image, MTLTextureUsage usage) {
    if (!image.query_only) {
        image.cached = &texture_cache.ObtainImage(image.info, usage);
    }
    for (u32 element = 0; element < image.count; ++element) {
        auto view = image.view;
        if (image.mip_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
            view.base_level += element;
        }
        const auto texture = image.query_only ? texture_cache.ObtainQueryView(image.info, view)
                                              : texture_cache.ObtainView(*image.cached, view);
        stage.bindings->SetTexture(image.binding, texture, element);
    }
}

void GuestResources::ValidateAttachments(std::span<Stage* const> stages,
                                         std::span<const ImageInfo> attachments) const {
    ValidateAliases(stages);
    const auto overlaps = [](VAddr left, u64 size, const ImageInfo& right) {
        return size && right.layout.guest_size &&
               left < right.guest_address + right.layout.guest_size &&
               right.guest_address < left + size;
    };
    for (const auto& attachment : attachments) {
        for (const Stage* stage : stages) {
            for (const auto& read : stage->scalar_reads) {
                if (ScalarReadOverlaps(memory, read, attachment.guest_address,
                                       attachment.layout.guest_size)) {
                    throw std::runtime_error(
                        "Metal scalar/attachment feedback requires native alias lowering");
                }
            }
            for (const auto& buffer : stage->buffers) {
                if (buffer.type == Shader::BufferType::Guest &&
                    overlaps(buffer.address, buffer.size, attachment)) {
                    throw std::runtime_error(
                        "Metal attachment/buffer feedback requires native alias lowering");
                }
            }
            for (const auto& image : stage->images) {
                if (overlaps(image.info.guest_address, image.info.layout.guest_size, attachment)) {
                    throw std::runtime_error(
                        "Metal attachment/image feedback requires native alias lowering");
                }
            }
        }
    }
}

void GuestResources::Acquire(std::span<Stage* const> stages) {
    ValidateAliases(stages);
    for (Stage* stage : stages) {
        stage->geometry_address_bindings.clear();
        stage->geometry_address_mappings.clear();
        for (const auto& buffer : stage->buffers) {
            if (buffer.type != Shader::BufferType::BufferBounds &&
                buffer.type != Shader::BufferType::GuestPageTable) {
                BindBuffer(*stage, buffer);
            }
        }
        for (const auto& read : stage->scalar_reads) {
            buffer_cache.ObtainPagedBuffer(read.lower, read.upper - read.lower + 1, false);
        }
        for (const auto& buffer : stage->buffers) {
            if (buffer.type == Shader::BufferType::BufferBounds) {
                BindBuffer(*stage, buffer);
            }
        }
        if (stage->geometry_snapshot) {
            const auto& shader = stage->variant->shader;
            const auto& program = shader.geometry          ? shader.geometry->address_program
                                  : shader.geometry_copy   ? shader.geometry_copy->address_program
                                  : shader.geometry_export ? shader.geometry_export->address_program
                                  : shader.tessellation
                                      ? shader.tessellation->registers.address_program
                                      : nullptr;
            if (!program) {
                throw std::runtime_error("Native geometry requires its address program");
            }
            const auto bounds =
                program->Evaluate(stage->geometry_inputs, stage->flattened_user_data);
            stage->geometry_address_mappings =
                MapGeometryBufferAddresses(bounds, stage->geometry_address_bindings);
        }
        for (auto& image : stage->images) {
            MTLTextureUsage usage = MTLTextureUsageShaderRead;
            for (const Stage* other : stages) {
                for (const auto& alias : other->images) {
                    if (image.info == alias.info) {
                        if (alias.written) {
                            usage |= MTLTextureUsageShaderWrite;
                        }
                        if (alias.atomic) {
                            usage |= MTLTextureUsageShaderAtomic;
                        }
                    }
                }
            }
            BindImage(*stage, image, usage);
        }
        for (const auto& sampler : stage->samplers) {
            stage->bindings->SetSampler(
                sampler.binding,
                samplers.ObtainSampler(sampler.sharp, sampler.depth, sampler.native_bias));
        }
        stage->acquired = true;
    }
    RefreshPageTables(stages);
}

void GuestResources::ValidateTessellation(const Stage& stage) const {
    if (stage.owner != this || !stage.variant->tessellation || !stage.snapshotted) {
        throw std::runtime_error("Metal tessellation validation requires a native stage snapshot");
    }
    if (!stage.tessellation_errors) {
        return;
    }
    if (!stage.tessellation_errors->IsReady()) {
        throw std::runtime_error("Metal tessellation validation requires completed GPU work");
    }
    u32 errors;
    std::memcpy(&errors, stage.tessellation_errors->ReadbackData().data(), sizeof(errors));
    if (errors) {
        throw std::runtime_error("Unsupported Metal tessellation LDS address or alignment");
    }
}

void GuestResources::RefreshPageTables(std::span<Stage* const> stages) {
    for (const Stage* stage : stages) {
        if (!stage || stage->owner != this || !stage->acquired || stage->marked_written) {
            throw std::runtime_error("Metal page tables require acquired resources");
        }
    }
    std::shared_ptr<const GuestPageTableSnapshot> page_snapshot;
    for (Stage* stage : stages) {
        for (const auto& buffer : stage->buffers) {
            if (buffer.type != Shader::BufferType::GuestPageTable) {
                continue;
            }
            if (!page_snapshot) {
                page_snapshot = buffer_cache.CreatePageSnapshot();
            }
            stage->page_snapshot = page_snapshot;
            BindBuffer(*stage, buffer);
        }
    }
}

void GuestResources::MarkWrites(std::span<Stage* const> stages) {
    for (Stage* stage : stages) {
        if (!stage || stage->owner != this || !stage->acquired || stage->marked_written) {
            throw std::runtime_error("Metal shader writes require acquired resources");
        }
        for (auto& image : stage->images) {
            if (image.written) {
                texture_cache.MarkWritten(*image.cached);
            }
        }
        stage->marked_written = true;
    }
}

std::unique_ptr<ResourceBindings> GuestResources::Snapshot(
    Stage& stage, std::span<const BufferRange> vertex_buffers) {
    if (stage.owner != this || !stage.acquired || !stage.marked_written || stage.snapshotted) {
        throw std::runtime_error("Metal shader snapshot requires prepared write tracking");
    }
    auto snapshot = stage.bindings->CreateSnapshot(
        staging, {reinterpret_cast<const u8*>(&stage.push), sizeof(stage.push)}, vertex_buffers);
    if (stage.page_snapshot) {
        stage.page_snapshot->Retain(*snapshot);
    }
    for (auto& allocation : stage.uploads) {
        snapshot->RetainUploadAllocation(std::move(allocation));
    }
    stage.uploads.clear();
    stage.snapshotted = true;
    return snapshot;
}

} // namespace Metal
