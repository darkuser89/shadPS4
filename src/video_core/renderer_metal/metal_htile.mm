// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_htile.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_xmask.h"
namespace Metal {
namespace {
constexpr std::string_view HtileSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    ulong offset;
    uint clear_value;
    uint pitch;
    uint samples;
    uint bytes;
    uint elements;
    uint reserved;
};
static_assert(sizeof(Parameters) == 32);
kernel void materialize_htile(device uchar* pixels [[buffer(0)]],
                              device const uchar* clear_tiles [[buffer(1)]],
                              constant Parameters& p [[buffer(2)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint2 grid [[threadgroups_per_grid]],
                              uint local [[thread_index_in_threadgroup]]) {
    const ulong element = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (element >= p.elements)
        return;
    const ulong pixel = element / p.samples;
    const ulong tile = (pixel / p.pitch / 8) * (p.pitch / 8) + (pixel % p.pitch) / 8;
    if (!clear_tiles[tile])
        return;
    for (uint byte = 0; byte < p.bytes; ++byte)
        pixels[p.offset + element * p.bytes + byte] = uchar(p.clear_value >> (byte * 8));
}
)metal";
XmaskLayout Layout(const ImageInfo& info) {
    auto surface = info.layout;
    surface.tile_mode = info.htile_tile_mode;
    return XmaskLayout::Create(surface, XmaskLayout::Kind::Htile, info.htile_surface & 1);
}
u32 ClearValue(const ImageInfo& info) {
    if (info.is_stencil) {
        return info.htile_clear_value;
    }
    const float value = std::bit_cast<float>(info.htile_clear_value);
    if (!std::isfinite(value) || value < 0 || value > 1) {
        throw std::runtime_error("Metal HTILE depth clear requires a finite normalized value");
    }
    return info.layout.num_bits == 16 ? u32(value * 65535.f + 0.5f) : info.htile_clear_value;
}
} // namespace
Htile::Htile(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

u64 Htile::MetadataSize(const ImageInfo& info) {
    const u32 samples = info.layout.num_samples;
    if (!info.htile_address || info.cmask_address || info.fmask_address || info.htile_neo ||
        (info.htile_surface & ~0xffffu) || !(info.htile_surface & 2) ||
        info.is_depth == info.is_stencil || info.is_volume || info.layout.block_compressed ||
        info.layout.num_mips != 1 || info.htile_base_slice >= info.layers ||
        (samples != 1 && samples != 2 && samples != 4 && samples != 8) ||
        (info.is_stencil ? info.layout.num_bits != 8 || info.htile_stencil_disabled
                         : info.layout.num_bits != 16 && info.layout.num_bits != 32) ||
        AmdGpu::GetArrayMode(info.htile_tile_mode) != AmdGpu::ArrayMode::Array2DTiledThin1 ||
        AmdGpu::GetArrayMode(info.layout.tile_mode) != AmdGpu::ArrayMode::Array2DTiledThin1) {
        throw std::runtime_error("Metal HTILE layout requires additional metadata lowering");
    }
    const auto layout = Layout(info);
    const auto& mip = info.layout.mips[0];
    const u64 pixels = u64{mip.pitch} * mip.height * info.layers;
    const u64 size = layout.SliceBytes() * info.layers;
    if (!mip.pitch || !mip.height || mip.pitch % 8 || mip.height % 8 || mip.offset ||
        pixels * samples * (info.layout.num_bits / 8) != info.layout.guest_size ||
        mip.size != info.layout.guest_size || layout.SliceBytes() % (2048 * layout.pipes) ||
        info.htile_address % (256 * layout.pipes) || size > std::numeric_limits<u32>::max() ||
        info.htile_address >= (u64{1} << 40) || size > (u64{1} << 40) - info.htile_address) {
        throw std::runtime_error("Invalid Metal HTILE allocation or slice layout");
    }
    return size;
}

Htile::Plan Htile::Decode(const ImageInfo& info, std::span<u8> metadata) {
    if (metadata.size() != MetadataSize(info)) {
        throw std::runtime_error("Metal HTILE snapshot size differs from its layout");
    }
    const auto layout = Layout(info);
    const u32 width = info.layout.mips[0].pitch / 8;
    const u32 height = info.layout.mips[0].height / 8;
    Plan plan{std::vector<u8>(u64{width} * height * info.layers)};
    for (u32 layer = info.htile_base_slice; layer < info.layers; ++layer) {
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u64 address = layout.BitAddress(x, y, layer) / 8;
                if (address > metadata.size() || sizeof(u32) > metadata.size() - address) {
                    throw std::runtime_error("Metal HTILE tile exceeds metadata allocation");
                }
                u32 word{};
                std::memcpy(&word, metadata.data() + address, sizeof(word));
                const u32 state = info.is_stencil ? (word >> 8) & 3 : word & 15;
                if (state != 0 && state != (info.is_stencil ? 3 : 15)) {
                    throw std::runtime_error("Metal HTILE compression code requires lowering");
                }
                if (!state) {
                    if (!plan.clear_count) {
                        (void)ClearValue(info);
                    }
                    plan.clear_tiles[(u64{layer} * height + y) * width + x] = 1;
                    ++plan.clear_count;
                }
                const u32 expanded = info.is_stencil               ? word | 0x3f0u
                                     : info.htile_stencil_disabled ? 0xfffc000fu
                                                                   : (word & 0xff0u) | 0xfffff00fu;
                plan.metadata_changed |= expanded != word;
                std::memcpy(metadata.data() + address, &expanded, sizeof(expanded));
            }
        }
    }
    return plan;
}

void Htile::Materialize(const ImageInfo& info, const Plan& plan, const BufferRange& linear) {
    (void)MetadataSize(info);
    const u32 bytes = info.layout.num_bits / 8;
    if (!plan.clear_count ||
        plan.clear_tiles.size() != info.layout.guest_size / bytes / info.layout.num_samples / 64 ||
        !linear.buffer || linear.buffer.device != instance.GetDevice() ||
        linear.offset > linear.buffer.length || linear.size < info.layout.guest_size ||
        linear.size > linear.buffer.length - linear.offset) {
        throw std::runtime_error("Invalid Metal HTILE materialization input");
    }
    const u32 clear_value = ClearValue(info);
    if (!pipeline) {
        auto library = compiler.CompileLibrary(HtileSource, "metal4_htile_materialize");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"materialize_htile";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    struct Parameters {
        u64 offset;
        u32 clear_value;
        u32 pitch;
        u32 samples;
        u32 bytes;
        u32 elements;
        u32 reserved;
    };
    static_assert(sizeof(Parameters) == 32);
    const Parameters params{linear.offset,
                            clear_value,
                            info.layout.mips[0].pitch,
                            info.layout.num_samples,
                            bytes,
                            u32(info.layout.guest_size / bytes),
                            0};
    auto data = staging.Allocate(sizeof(params));
    std::memcpy(data.UploadData().data(), &params, sizeof(params));
    auto tiles = staging.Allocate(plan.clear_tiles.size());
    std::memcpy(tiles.UploadData().data(), plan.clear_tiles.data(), plan.clear_tiles.size());
    ResourceBindings bindings{instance, 3, 0, 0};
    bindings.SetBuffer(0, linear.buffer);
    bindings.SetBuffer(1, tiles.GetBuffer(), tiles.GetOffset());
    bindings.SetBuffer(2, data.GetBuffer(), data.GetOffset());
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
} // namespace Metal
