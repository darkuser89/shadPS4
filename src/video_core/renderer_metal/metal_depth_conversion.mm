// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <stdexcept>
#include "video_core/renderer_metal/metal_depth_conversion.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {

constexpr std::string_view source_code = R"(
#include <metal_stdlib>
using namespace metal;

vertex float4 unorm32_depth_vertex(uint index [[vertex_id]]) {
    float2 uv = float2((index << 1) & 2, index & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

struct DepthValue {
    float value [[depth(any)]];
};

fragment DepthValue unorm32_depth_fragment(float4 position [[position]],
                                          texture2d<uint, access::read> source [[texture(0)]]) {
    return {float(source.read(uint2(position.xy)).x) / 4294967295.0f};
}

fragment DepthValue unorm32_depth_ms_fragment(
    float4 position [[position]], uint sample [[sample_id]],
    texture2d_ms<uint, access::read> source [[texture(0)]]) {
    return {float(source.read(uint2(position.xy), sample).x) / 4294967295.0f};
}
)";

void ValidateSource(const Instance& instance, id<MTLTexture> source) {
    if (!source || source.device != instance.GetDevice() ||
        source.pixelFormat != MTLPixelFormatR32Uint || source.framebufferOnly ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        (source.textureType != MTLTextureType2D && source.textureType != MTLTextureType2DArray &&
         source.textureType != MTLTextureType2DMultisample &&
         source.textureType != MTLTextureType2DMultisampleArray)) {
        throw std::runtime_error("Invalid Metal UNORM32 depth conversion source");
    }
}

} // namespace

DepthConversion::DepthConversion(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler},
      compiler{instance} {}

id<MTLTexture> DepthConversion::CreateTexture(id<MTLTexture> source) const {
    ValidateSource(instance, source);
    auto descriptor = [MTLTextureDescriptor new];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    descriptor.textureType = source.textureType;
    descriptor.width = source.width;
    descriptor.height = source.height;
    descriptor.depth = 1;
    descriptor.arrayLength = source.arrayLength;
    descriptor.mipmapLevelCount = source.mipmapLevelCount;
    descriptor.sampleCount = source.sampleCount;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    descriptor.usage =
        MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    return instance.CreateTexture(descriptor);
}

id<MTLRenderPipelineState> DepthConversion::Pipeline(u32 samples) {
    if (const auto it = pipelines.find(samples); it != pipelines.end()) {
        return it->second;
    }
    if (!library) {
        library = compiler.CompileLibrary(source_code, "UNORM32 depth conversion");
        auto depth = [MTLDepthStencilDescriptor new];
        depth.depthCompareFunction = MTLCompareFunctionAlways;
        depth.depthWriteEnabled = YES;
        depth_stencil = instance.CreateDepthStencilState(depth);
    }
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    auto vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = library;
    vertex.name = @"unorm32_depth_vertex";
    auto fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = library;
    fragment.name = samples > 1 ? @"unorm32_depth_ms_fragment" : @"unorm32_depth_fragment";
    descriptor.vertexFunctionDescriptor = vertex;
    descriptor.fragmentFunctionDescriptor = fragment;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    descriptor.rasterSampleCount = samples;
    auto pipeline = compiler.CompileRenderPipeline(descriptor);
    pipelines.emplace(samples, pipeline);
    return pipeline;
}

void DepthConversion::Convert(id<MTLTexture> source, id<MTLTexture> destination) {
    ValidateSource(instance, source);
    if (!destination || destination.device != instance.GetDevice() ||
        destination.pixelFormat != MTLPixelFormatDepth32Float || destination.framebufferOnly ||
        !(destination.usage & MTLTextureUsageRenderTarget) ||
        source.textureType != destination.textureType || source.width != destination.width ||
        source.height != destination.height || source.arrayLength != destination.arrayLength ||
        source.mipmapLevelCount != destination.mipmapLevelCount ||
        source.sampleCount != destination.sampleCount) {
        throw std::runtime_error("Invalid Metal UNORM32 depth conversion destination");
    }
    const auto pipeline = Pipeline(source.sampleCount);
    const auto view_type = source.sampleCount > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    for (u32 level = 0; level < source.mipmapLevelCount; ++level) {
        const u32 width = std::max(u32(source.width) >> level, 1u);
        const u32 height = std::max(u32(source.height) >> level, 1u);
        for (u32 layer = 0; layer < source.arrayLength; ++layer) {
            auto view = [source newTextureViewWithPixelFormat:MTLPixelFormatR32Uint
                                                  textureType:view_type
                                                       levels:NSMakeRange(level, 1)
                                                       slices:NSMakeRange(layer, 1)];
            if (!view) {
                throw std::runtime_error("Metal rejected the UNORM32 depth source view");
            }
            RenderState render{};
            render.width = width;
            render.height = height;
            render.num_samples = source.sampleCount;
            render.depth_attachment.texture = destination;
            render.depth_attachment.level = level;
            render.depth_attachment.slice = layer;
            render.depth_attachment.load_action = MTLLoadActionDontCare;
            GraphicsState graphics{};
            graphics.depth_stencil = depth_stencil;
            graphics.viewports[0] = {0, 0, double(width), double(height)};
            graphics.scissors[0] = {0, 0, width, height};
            ResourceBindings fragment{instance, 0, 1, 0};
            fragment.SetTexture(0, view);
            scheduler.ProducerBarrier(MTLStageAll, MTLStageVertex | MTLStageFragment);
            (void)commands.BindGraphics(render, graphics, pipeline, nullptr, &fragment);
            commands.Draw(MTLPrimitiveTypeTriangle, 3);
            scheduler.ProducerBarrier(MTLStageFragment, MTLStageAll);
        }
    }
}

} // namespace Metal
