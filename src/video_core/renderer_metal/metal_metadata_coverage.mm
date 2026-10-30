// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_metadata_coverage.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
constexpr std::string_view CoverageSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Parameters { uint width; uint height; uint layers; uint count; };
static_assert(sizeof(Parameters) == 16);
kernel void classify_metadata_coverage(texture2d_array<float, access::read> coverage [[texture(0)]],
                                       device uchar* tiles [[buffer(0)]],
                                       constant Parameters& p [[buffer(1)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint2 grid [[threadgroups_per_grid]],
                                       uint local [[thread_index_in_threadgroup]]) {
    const ulong tile = (ulong(group.y) * grid.x + group.x) * 64 + local;
    if (tile >= p.count)
        return;
    const uint layer = uint(tile / (ulong(p.width) * p.height));
    const uint2 origin = uint2(uint(tile % p.width), uint(tile / p.width % p.height)) * 8;
    uint state = 0;
    for (uint y = 0; y < 8; ++y) {
        for (uint x = 0; x < 8; ++x) {
            const float value = coverage.read(origin + uint2(x, y), layer).r;
            state |= value == 0.0f ? 1u : value == 1.0f ? 2u : 4u;
        }
    }
    tiles[tile] = (state & 4) ? 3 : state == 3 ? 2 : state == 2 ? 1 : 0;
}
)metal";
} // namespace

MetadataCoverage::MetadataCoverage(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

void MetadataCoverage::Prepare(const ImageInfo& info, RenderState& render) {
    const auto& mip = info.layout.mips[0];
    if (!info.width || !info.height || info.width % 8 || info.height % 8 ||
        info.width > mip.pitch || info.height > mip.height || !render.num_layers ||
        info.cmask_base_slice >= info.layers ||
        render.num_layers > info.layers - info.cmask_base_slice ||
        u64{info.width / 8} * (info.height / 8) * render.num_layers >
            std::min<u64>(instance.GetDevice().maxBufferLength, std::numeric_limits<u32>::max())) {
        throw std::runtime_error("Metal metadata coverage requires complete guest tiles");
    }
    scheduler.EndEncoding();
    if (!target || target.width != info.width || target.height != info.height ||
        target.arrayLength != render.num_layers || target.sampleCount != render.num_samples) {
        auto descriptor = [MTLTextureDescriptor new];
        descriptor.textureType =
            render.num_samples > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
        descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
        descriptor.width = info.width;
        descriptor.height = info.height;
        descriptor.arrayLength = render.num_layers;
        descriptor.sampleCount = render.num_samples;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        target = instance.CreateTexture(descriptor);
        if (render.num_samples > 1) {
            descriptor.textureType = MTLTextureType2DArray;
            descriptor.sampleCount = 1;
            resolved = instance.CreateTexture(descriptor);
        } else {
            resolved = target;
        }
    }
    render.width = info.width;
    render.height = info.height;
    render.num_color_attachments = 1;
    auto& attachment = render.color_attachments[0];
    attachment.texture = target;
    attachment.load_action = MTLLoadActionClear;
    if (render.num_samples > 1) {
        attachment.resolve_texture = resolved;
        attachment.store_action = MTLStoreActionMultisampleResolve;
    }
}

std::vector<u8> MetadataCoverage::ReadTiles(const ImageInfo& info, const RenderState& render) {
    scheduler.EndEncoding();
    if (!pipeline) {
        auto library = compiler.CompileLibrary(CoverageSource, "metal4_metadata_coverage_reduce");
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = library;
        function.name = @"classify_metadata_coverage";
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        pipeline = compiler.CompileComputePipeline(descriptor);
    }
    struct Parameters {
        u32 width;
        u32 height;
        u32 layers;
        u32 count;
    };
    static_assert(sizeof(Parameters) == 16);
    const Parameters params{info.width / 8, info.height / 8, render.num_layers,
                            u32(u64{info.width / 8} * (info.height / 8) * render.num_layers)};
    auto readback = staging.AllocateReadback(params.count);
    auto data = staging.Allocate(sizeof(params));
    std::memcpy(data.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 2, 1, 0};
    bindings.SetBuffer(0, readback.GetBuffer(), readback.GetOffset());
    bindings.SetBuffer(1, data.GetBuffer(), data.GetOffset());
    bindings.SetTexture(0, resolved);
    bindings.RetainUploadAllocation(std::move(data));
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    const u64 groups = (u64{params.count} + 63) / 64;
    const u64 width = std::min(groups, u64{65535});
    commands.Dispatch(MTLSizeMake(width, (groups + width - 1) / width, 1), MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.Finish();
    const auto coverage = readback.ReadbackData();
    const auto& mip = info.layout.mips[0];
    const u32 pitch = mip.pitch / 8;
    const u32 height = mip.height / 8;
    std::vector<u8> tiles(u64{pitch} * height * info.layers);
    for (u32 layer = 0; layer < render.num_layers; ++layer) {
        for (u32 ty = 0; ty < info.height / 8; ++ty) {
            for (u32 tx = 0; tx < info.width / 8; ++tx) {
                const u8 state = coverage[(u64{layer} * params.height + ty) * params.width + tx];
                if (state == 3) {
                    throw std::runtime_error("Metal metadata draw has partial sample coverage");
                }
                if (state == 2) {
                    throw std::runtime_error("Metal metadata draw has partial tile coverage");
                }
                if (state > 3) {
                    throw std::runtime_error("Invalid Metal metadata coverage result");
                }
                tiles[(u64{layer + info.cmask_base_slice} * height + ty) * pitch + tx] = state;
            }
        }
    }
    return tiles;
}

} // namespace Metal
