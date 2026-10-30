// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_cmask.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_xmask.h"

namespace Metal {
namespace {

constexpr std::string_view CmaskSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    ulong offset;
    ulong clear_value;
    ulong fmask_offset;
    uint pitch;
    uint height;
    uint bytes;
    uint elements;
    uint samples;
    uint fmask_pitch;
    uint fmask_height;
    uint fmask_bytes;
};
static_assert(sizeof(Parameters) == 56);
kernel void materialize_cmask(device uchar* pixels [[buffer(0)]],
                              device const uchar* tile_modes [[buffer(1)]],
                              constant Parameters& p [[buffer(2)]],
                              device uchar* fmask [[buffer(3)]],
                              device atomic_uint* errors [[buffer(4)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint2 grid [[threadgroups_per_grid]],
                              uint local [[thread_index_in_threadgroup]]) {
    const ulong index = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (index >= p.elements)
        return;
    const ulong tile = (index / p.pitch / 8) * (p.pitch / 8) + (index % p.pitch) / 8;
    const uint mode = tile_modes[tile];
    if (!mode)
        return;
    const ulong pixel_offset = p.offset + index * p.samples * p.bytes;
    if (mode >= 4) {
        if (mode == 4 || mode == 5) {
            for (uint byte = 0; byte < p.bytes; ++byte)
                pixels[pixel_offset + byte] = uchar(p.clear_value >> (byte * 8));
        }
        if (mode == 5 || mode == 6) {
            const ulong row = index / p.pitch;
            const ulong fmask_index = (row / p.height * p.fmask_height + row % p.height) *
                                     p.fmask_pitch + index % p.pitch;
            for (uint byte = 0; byte < p.fmask_bytes; ++byte)
                fmask[p.fmask_offset + fmask_index * p.fmask_bytes + byte] = 0;
        }
        return;
    }
    ulong values[8];
    for (uint sample = 0; sample < p.samples; ++sample) {
        values[sample] = 0;
        for (uint byte = 0; byte < p.bytes; ++byte)
            values[sample] |= ulong(pixels[pixel_offset + sample * p.bytes + byte]) << (byte * 8);
    }
    ulong fmask_offset = 0;
    uint mapping = 0;
    const uint bits = p.samples == 2 ? 1 : p.samples == 4 ? 2 : 4;
    if (p.samples > 1) {
        const ulong row = index / p.pitch;
        const ulong fmask_index = (row / p.height * p.fmask_height + row % p.height) *
                                 p.fmask_pitch + index % p.pitch;
        fmask_offset = p.fmask_offset + fmask_index * p.fmask_bytes;
        if (mode == 3) {
            for (uint byte = 0; byte < p.fmask_bytes; ++byte)
                mapping |= uint(fmask[fmask_offset + byte]) << (byte * 8);
        }
        for (uint sample = 0; sample < p.samples; ++sample) {
            if (((mapping >> (sample * bits)) & ((1u << bits) - 1)) >= p.samples) {
                atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
                return;
            }
        }
    }
    uint identity = 0;
    for (uint sample = 0; sample < p.samples; ++sample) {
        const uint physical = (mapping >> (sample * bits)) & ((1u << bits) - 1);
        const ulong value = mode == 1 ? p.clear_value : values[physical];
        for (uint byte = 0; byte < p.bytes; ++byte)
            pixels[pixel_offset + sample * p.bytes + byte] = uchar(value >> (byte * 8));
        identity |= sample << (sample * bits);
    }
    if (p.samples > 1) {
        for (uint byte = 0; byte < p.fmask_bytes; ++byte)
            fmask[fmask_offset + byte] = uchar(identity >> (byte * 8));
    }
}
)metal";

} // namespace

Cmask::Cmask(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

void Cmask::Validate(const ImageInfo& info) {
    const u32 samples = info.layout.num_samples;
    if (!info.cmask_address || info.cmask_address_type || info.htile_address || info.is_depth ||
        info.is_stencil || info.is_volume || info.layout.num_mips != 1 ||
        (samples != 1 && samples != 2 && samples != 4 && samples != 8) ||
        (samples == 1 ? info.fmask_address != 0 : !info.fmask_address) ||
        info.layout.block_compressed || !info.layers || info.cmask_base_slice >= info.layers ||
        (info.layout.num_bits != 8 && info.layout.num_bits != 16 && info.layout.num_bits != 32 &&
         info.layout.num_bits != 64) ||
        AmdGpu::GetArrayMode(info.layout.tile_mode) != AmdGpu::ArrayMode::Array2DTiledThin1) {
        throw std::runtime_error("Metal CMASK image layout requires additional metadata lowering");
    }
    if (samples > 1) {
        (void)ImageInfo::FromFmask(info);
    }
    const auto layout =
        XmaskLayout::Create(info.layout, XmaskLayout::Kind::Cmask, info.cmask_linear);
    const auto& mip = info.layout.mips[0];
    const u64 size = layout.SliceBytes();
    const u64 pixels = u64{mip.pitch} * mip.height * info.layers;
    if (!mip.pitch || !mip.height || mip.pitch % 8 || mip.height % 8 || mip.offset ||
        pixels * samples * (info.layout.num_bits / 8) != info.layout.guest_size ||
        mip.size != info.layout.guest_size || size != info.cmask_slice_bytes ||
        size % (layout.pipes * 256) || info.cmask_address % (layout.pipes * 256) ||
        pixels > std::numeric_limits<u32>::max() ||
        size * info.layers > std::numeric_limits<u32>::max() ||
        info.cmask_address >= (u64{1} << 40) ||
        size * info.layers > (u64{1} << 40) - info.cmask_address) {
        throw std::runtime_error("Invalid Metal CMASK allocation or slice layout");
    }
}

Cmask::Plan Cmask::Decode(const ImageInfo& info, std::span<u8> metadata, Operation operation,
                          std::span<const u8> tiles) {
    Validate(info);
    if (metadata.size() != info.cmask_slice_bytes * info.layers) {
        throw std::runtime_error("Metal CMASK snapshot size differs from its layout");
    }
    const auto layout =
        XmaskLayout::Create(info.layout, XmaskLayout::Kind::Cmask, info.cmask_linear);
    const u32 width = info.layout.mips[0].pitch / 8;
    const u32 height = info.layout.mips[0].height / 8;
    const u64 tile_count = u64{width} * height * info.layers;
    if ((operation != Operation::Expand && operation != Operation::EliminateFastClear &&
         operation != Operation::DecompressFmask) ||
        (operation == Operation::DecompressFmask && info.layout.num_samples == 1) ||
        ((!tiles.empty() || operation != Operation::Expand) && tiles.size() != tile_count) ||
        std::ranges::any_of(tiles, [](u8 tile) { return tile > 1; })) {
        throw std::runtime_error("Invalid Metal color metadata operation or tile selection");
    }
    Plan plan{std::vector<u8>(tile_count)};
    const u32 samples = info.layout.num_samples;
    const u32 expanded = samples == 2 ? 13 : samples == 4 ? 14 : 15;
    for (u32 layer = info.cmask_base_slice; layer < info.layers; ++layer) {
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u64 tile = (u64{layer} * height + y) * width + x;
                if (!tiles.empty() && !tiles[tile]) {
                    continue;
                }
                const u64 nibble = layout.BitAddress(x, y, layer) / 4;
                if (nibble / 2 >= metadata.size()) {
                    throw std::runtime_error("Metal CMASK tile exceeds metadata allocation");
                }
                auto& byte = metadata[nibble / 2];
                const u32 shift = (nibble & 1) * 4;
                const u32 code = (byte >> shift) & 15;
                u32 mode = !code && info.cmask_fast_clear    ? 1
                           : samples > 1 && code == 12       ? 2
                           : samples > 1 && code == expanded ? 3
                           : samples == 1 && code == 15      ? 0
                                                             : 4;
                if (mode == 4) {
                    throw std::runtime_error("Metal CMASK compression code requires lowering");
                }
                u32 replacement = expanded;
                if (operation == Operation::EliminateFastClear) {
                    mode = mode == 1 ? 4 : 0;
                    replacement = samples > 1 ? 12 : 15;
                } else if (operation == Operation::DecompressFmask) {
                    mode = mode == 1 ? 5 : mode == 2 ? 6 : 0;
                }
                if (mode) {
                    plan.tile_modes[tile] = u8(mode);
                    ++plan.materialize_count;
                    plan.write_pixels |= mode != 6;
                    plan.write_fmask |= samples > 1 && mode != 4;
                    byte = u8((byte & ~(15u << shift)) | (replacement << shift));
                }
            }
        }
    }
    return plan;
}

void Cmask::Materialize(const ImageInfo& info, const Plan& plan, const BufferRange& linear,
                        const BufferRange& fmask) {
    Validate(info);
    const u32 bytes = info.layout.num_bits / 8;
    const u32 samples = info.layout.num_samples;
    if (!plan.materialize_count ||
        plan.tile_modes.size() != info.layout.guest_size / bytes / samples / 64 || !linear.buffer ||
        linear.buffer.device != instance.GetDevice() || linear.offset > linear.buffer.length ||
        linear.size < info.layout.guest_size ||
        linear.size > linear.buffer.length - linear.offset) {
        throw std::runtime_error("Invalid Metal CMASK materialization input");
    }
    if (plan.write_fmask) {
        const auto fmask_info = ImageInfo::FromFmask(info);
        if (!fmask.buffer || fmask.buffer.device != instance.GetDevice() ||
            fmask.offset > fmask.buffer.length || fmask.size < fmask_info.layout.guest_size ||
            fmask.size > fmask.buffer.length - fmask.offset) {
            throw std::runtime_error("Invalid Metal FMASK materialization input");
        }
    }
    if (!pipeline) {
        auto library = compiler.CompileLibrary(CmaskSource, "metal4_cmask_materialize");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"materialize_cmask";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    struct Parameters {
        u64 offset;
        u64 clear_value;
        u64 fmask_offset;
        u32 pitch;
        u32 height;
        u32 bytes;
        u32 elements;
        u32 samples;
        u32 fmask_pitch;
        u32 fmask_height;
        u32 fmask_bytes;
    };
    static_assert(sizeof(Parameters) == 56);
    const Parameters params{linear.offset,
                            info.cmask_clear_value,
                            fmask.offset,
                            info.layout.mips[0].pitch,
                            info.layout.mips[0].height,
                            bytes,
                            u32(info.layout.guest_size / bytes / samples),
                            samples,
                            info.fmask_pitch,
                            info.fmask_height,
                            samples == 8 ? 4u : 1u};
    auto data = staging.Allocate(sizeof(params));
    std::memcpy(data.UploadData().data(), &params, sizeof(params));
    auto tiles = staging.Allocate(plan.tile_modes.size());
    std::memcpy(tiles.UploadData().data(), plan.tile_modes.data(), plan.tile_modes.size());
    ResourceBindings bindings{instance, 5, 0, 0};
    bindings.SetBuffer(0, linear.buffer);
    bindings.SetBuffer(1, tiles.GetBuffer(), tiles.GetOffset());
    bindings.SetBuffer(2, data.GetBuffer(), data.GetOffset());
    bindings.SetBuffer(3, plan.write_fmask ? fmask.buffer : linear.buffer);
    if (errors.empty() || errors.back().tick != scheduler.CurrentTick()) {
        auto error = staging.AllocateReadback(sizeof(u32), alignof(u32));
        commands.FillBuffer({error.GetBuffer(), error.GetOffset(), error.SizeBytes()}, 0);
        errors.push_back({std::move(error), scheduler.CurrentTick()});
    }
    const auto& error = errors.back().allocation;
    bindings.SetBuffer(4, error.GetBuffer(), error.GetOffset());
    bindings.RetainUploadAllocation(std::move(tiles));
    bindings.RetainUploadAllocation(std::move(data));
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    const u64 groups = (u64{params.elements} + 63) / 64;
    const u64 width = std::min(groups, u64{65535});
    commands.Dispatch(MTLSizeMake(width, (groups + width - 1) / width, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageDispatch, MTLStageDispatch | MTLStageBlit);
}

void Cmask::CheckErrors() {
    while (!errors.empty() && errors.front().allocation.IsReady()) {
        u32 flags{};
        std::memcpy(&flags, errors.front().allocation.ReadbackData().data(), sizeof(flags));
        errors.pop_front();
        if (flags) {
            throw std::runtime_error("Metal FMASK contains an out-of-range physical sample");
        }
    }
}

} // namespace Metal
