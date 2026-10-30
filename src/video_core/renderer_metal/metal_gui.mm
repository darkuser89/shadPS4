// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "video_core/renderer_metal/metal_gui.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {
namespace {
static_assert(sizeof(ImDrawVert) == 20);
static_assert(sizeof(ImDrawIdx) == 4);
constexpr std::string_view HostSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct GuiVertex { packed_float2 position; packed_float2 uv; uint color; };
static_assert(sizeof(GuiVertex) == 20);
struct GuiUniforms { float2 scale; float2 translate; };
struct Output { float4 position [[position]]; float2 uv; float4 color; };
float3 decode_srgb(float3 c) {
    return select(c / 12.92f, pow((c + 0.055f) / 1.055f, float3(2.4f)), c > 0.04045f);
}
vertex Output gui_vertex(uint index [[vertex_id]], device const GuiVertex* vertices [[buffer(0)]],
                         constant GuiUniforms& uniforms [[buffer(1)]]) {
    GuiVertex v = vertices[index];
    float4 color = float4(v.color & 255u, (v.color >> 8u) & 255u,
                         (v.color >> 16u) & 255u, v.color >> 24u) / 255.0f;
    return {float4(v.position * uniforms.scale + uniforms.translate, 0, 1), v.uv,
            float4(decode_srgb(color.rgb), color.a)};
}
fragment float4 gui_fragment(Output in [[stage_in]], texture2d<float> image [[texture(0)]],
                             sampler filter [[sampler(0)]]) {
    return image.sample(filter, in.uv) * in.color;
}
vertex Output frame_vertex(uint index [[vertex_id]]) {
    float2 uv = float2((index << 1u) & 2u, index & 2u);
    return {float4(uv * float2(2, -2) + float2(-1, 1), 0, 1), uv, float4(1)};
}
float3 decode_pq(float3 encoded) {
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 32.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 128.0f;
    constexpr float c3 = 2392.0f / 128.0f;
    const float3 exponent = log(max(clamp(encoded, 0.0f, 1.0f), float3(0x1p-24f))) / m2;
    float3 polynomial = fma(exponent, float3(1.0f / 5040.0f), float3(1.0f / 720.0f));
    polynomial = fma(exponent, polynomial, float3(1.0f / 120.0f));
    polynomial = fma(exponent, polynomial, float3(1.0f / 24.0f));
    polynomial = fma(exponent, polynomial, float3(1.0f / 6.0f));
    polynomial = fma(exponent, polynomial, float3(0.5f));
    polynomial = fma(exponent, polynomial, float3(1.0f));
    const float3 power_minus_one = exponent * polynomial;
    const float3 numerator = max(power_minus_one + (1.0f - c1), float3(0));
    const float3 denominator = fma(-c3, power_minus_one, float3(c2 - c3));
    return 10000.0f * pow(numerator / denominator, float3(1.0f / m1));
}
float3 bt2020_to_srgb(float3 rgb) {
    return float3(dot(rgb, float3(1.660491002f, -0.587641139f, -0.072849863f)),
                  dot(rgb, float3(-0.124550475f, 1.132899897f, -0.008349422f)),
                  dot(rgb, float3(-0.018150763f, -0.100578898f, 1.118729661f)));
}
float3 srgb_to_bt2020(float3 rgb) {
    return float3(dot(rgb, float3(0.627403896f, 0.329283038f, 0.043313066f)),
                  dot(rgb, float3(0.069097289f, 0.919540395f, 0.011362316f)),
                  dot(rgb, float3(0.016391439f, 0.088013308f, 0.895595253f)));
}
float3 encode_pq(float3 nits) {
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 32.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 128.0f;
    constexpr float c3 = 2392.0f / 128.0f;
    const float3 power = pow(clamp(nits / 10000.0f, 0.0f, 1.0f), float3(m1));
    return pow((c1 + c2 * power) / (1.0f + c3 * power), float3(m2));
}
struct FrameUniforms { float gamma; uint transfer_function; float nits_per_unit; uint half_output; };
static_assert(sizeof(FrameUniforms) == 16);
float4 frame_color(float3 rgb, constant FrameUniforms& uniforms) {
    if (uniforms.half_output) rgb = float3(half3(rgb));
    return float4(rgb, 1);
}
float3 hdr_gamma(float3 rgb, float gamma) {
    if (gamma != 1.0f)
        rgb = select(sign(rgb) * min(pow(abs(rgb), float3(1.0f / gamma)), float3(65504.0f)),
                     rgb, rgb == 0.0f);
    return clamp(rgb, -65504.0f, 65504.0f);
}
fragment float4 frame_fragment(Output in [[stage_in]], texture2d<float> image [[texture(0)]],
                               sampler filter [[sampler(0)]],
                               constant FrameUniforms& uniforms [[buffer(0)]]) {
    float3 rgb = uniforms.transfer_function == 6 || uniforms.transfer_function == 9
                     ? image.read(uint2(in.position.xy)).rgb
                     : image.sample(filter, in.uv).rgb;
    if (uniforms.transfer_function == 1) rgb = decode_srgb(rgb);
    if (uniforms.transfer_function == 2 || uniforms.transfer_function == 3) {
        rgb = decode_pq(rgb) / uniforms.nits_per_unit;
        if (uniforms.transfer_function == 3) rgb = bt2020_to_srgb(rgb);
        return frame_color(rgb, uniforms);
    }
    if (uniforms.transfer_function == 4 || uniforms.transfer_function == 5) {
        if (uniforms.transfer_function == 5) rgb = srgb_to_bt2020(rgb);
        return frame_color(encode_pq(rgb * uniforms.nits_per_unit), uniforms);
    }
    if (uniforms.transfer_function == 6 || uniforms.transfer_function == 9) {
        if (uniforms.transfer_function == 6) rgb = bt2020_to_srgb(rgb);
        return frame_color(hdr_gamma(rgb, uniforms.gamma), uniforms);
    }
    return frame_color(pow(max(rgb, float3(0)), float3(1.0f / uniforms.gamma)), uniforms);
}
fragment float4 fsr_hdr_fragment(Output in [[stage_in]],
                                 texture2d<float, access::read> image [[texture(0)]],
                                 constant FrameUniforms& uniforms [[buffer(0)]]) {
    float3 rgb = image.read(uint2(in.position.xy)).rgb;
    if (uniforms.transfer_function == 7) {
        rgb = clamp(rgb, 0.0f, 65504.0f);
        return float4(rgb / (1.0f + max(max(rgb.r, rgb.g), rgb.b)), 1);
    }
    if (uniforms.transfer_function == 8) {
        rgb = clamp(rgb, 0.0f, 1.0f);
        return frame_color(rgb / max(1.0f / 32768.0f, 1.0f - max(max(rgb.r, rgb.g), rgb.b)),
                           uniforms);
    }
    return float4(0);
}
)metal";

RenderState TargetState(id<MTLTexture> target) {
    RenderState state;
    state.width = target.width;
    state.height = target.height;
    state.num_color_attachments = 1;
    state.color_attachments[0].texture = target;
    state.color_attachments[0].load_action = MTLLoadActionClear;
    state.color_attachments[0].clear_color = {0, 0, 0, 1};
    return state;
}

GraphicsState TargetGraphics(id<MTLTexture> target) {
    GraphicsState state;
    state.viewports[0].width = target.width;
    state.viewports[0].height = target.height;
    state.scissors[0].width = target.width;
    state.scissors[0].height = target.height;
    return state;
}
} // namespace

GuiRenderer::GuiRenderer(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler},
      transfers{instance, scheduler}, compiler{instance}, staging{instance, scheduler} {
    library = compiler.CompileLibrary(HostSource, "shadPS4:Metal4HostShaders");
    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.minFilter = MTLSamplerMinMagFilterLinear;
    descriptor.magFilter = MTLSamplerMinMagFilterLinear;
    descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler = instance.CreateSampler(descriptor);
}

GuiRenderer::~GuiRenderer() {
    ReleaseTextures();
}

void GuiRenderer::ReleaseTextures() {
    for (auto& [data, texture] : textures) {
        data->SetTexID(ImTextureID_Invalid);
        data->BackendUserData = nullptr;
        data->SetStatus(ImTextureStatus_Destroyed);
    }
    textures.clear();
}

ImTextureID GuiRenderer::TextureId(id<MTLTexture> texture) {
    return reinterpret_cast<ImTextureID>((__bridge void*)texture);
}

id<MTLRenderPipelineState> GuiRenderer::Pipeline(MTLPixelFormat format, PipelineType type) {
    const auto key = std::pair{format, type};
    const bool game = type != PipelineType::Gui;
    if (const auto found = pipelines.find(key); found != pipelines.end()) {
        return found->second;
    }
    MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
    MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = library;
    vertex.name = game ? @"frame_vertex" : @"gui_vertex";
    MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = library;
    fragment.name = type == PipelineType::FsrHdr ? @"fsr_hdr_fragment"
                    : game                       ? @"frame_fragment"
                                                 : @"gui_fragment";
    descriptor.vertexFunctionDescriptor = vertex;
    descriptor.fragmentFunctionDescriptor = fragment;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    descriptor.rasterSampleCount = 1;
    auto color = descriptor.colorAttachments[0];
    color.pixelFormat = format;
    color.blendingState = game ? MTL4BlendStateDisabled : MTL4BlendStateEnabled;
    color.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    color.sourceAlphaBlendFactor = MTLBlendFactorOne;
    color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    auto pipeline = compiler.CompileRenderPipeline(descriptor);
    pipelines.emplace(key, pipeline);
    return pipeline;
}

id<MTLTexture> GuiRenderer::UploadTexture(std::span<const u8> pixels, u32 width, u32 height,
                                          MTLPixelFormat format) {
    if (!width || !height || width > 16384 || height > 16384 ||
        pixels.size() != u64{width} * height * 4) {
        throw std::runtime_error("Invalid native Metal GUI texture dimensions");
    }
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                         width:width
                                                                        height:height
                                                                     mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    auto texture = instance.CreateTexture(descriptor);
    UploadTextureRegion(texture, pixels, MTLRegionMake2D(0, 0, width, height));
    return texture;
}

void GuiRenderer::UploadTextureRegion(id<MTLTexture> texture, std::span<const u8> pixels,
                                      const MTLRegion& region) {
    if (!texture || texture.device != instance.GetDevice() || texture.framebufferOnly ||
        texture.textureType != MTLTextureType2D || texture.sampleCount != 1 ||
        texture.width > 16384 || texture.height > 16384 ||
        (texture.pixelFormat != MTLPixelFormatRGBA8Unorm &&
         texture.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB) ||
        pixels.size() != u64{texture.width} * texture.height * 4 || region.origin.z ||
        region.size.depth != 1 || !region.size.width || !region.size.height ||
        region.origin.x >= texture.width || region.origin.y >= texture.height ||
        region.size.width > texture.width - region.origin.x ||
        region.size.height > texture.height - region.origin.y) {
        throw std::runtime_error("Invalid native Metal GUI texture update");
    }
    const u64 row = u64{region.size.width} * 4;
    auto allocation = staging.Allocate(row * region.size.height);
    auto destination = allocation.UploadData();
    if (region.size.width == texture.width) {
        std::memcpy(destination.data(), pixels.data() + region.origin.y * row, destination.size());
    } else {
        for (u64 y = 0; y < region.size.height; ++y) {
            const u64 source = ((region.origin.y + y) * texture.width + region.origin.x) * 4;
            std::memcpy(destination.data() + y * row, pixels.data() + source, row);
        }
    }
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
    transfers.Upload({allocation.GetBuffer(), allocation.GetOffset(), allocation.SizeBytes()},
                     {.texture = texture, .origin = region.origin, .size = region.size},
                     {.bytes_per_row = row, .bytes_per_image = 0});
    scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
}

ImGui::Core::TextureManager::NativeTexture GuiRenderer::Upload(std::span<const u8> pixels,
                                                               u32 width, u32 height) {
    auto holder = std::make_shared<Texture>();
    holder->resource = UploadTexture(pixels, width, height, MTLPixelFormatRGBA8Unorm_sRGB);
    return {TextureId(holder->resource), std::move(holder)};
}

void GuiRenderer::UpdateTextures(ImDrawData& data) {
    if (!data.Textures) {
        return;
    }
    for (ImTextureData* texture : *data.Textures) {
        if (texture->Status == ImTextureStatus_WantCreate ||
            texture->Status == ImTextureStatus_WantUpdates) {
            if (texture->Format != ImTextureFormat_RGBA32 || texture->BytesPerPixel != 4 ||
                !texture->Pixels || texture->Width <= 0 || texture->Height <= 0 ||
                texture->Width > 16384 || texture->Height > 16384) {
                throw std::runtime_error("Invalid native Metal GUI font texture");
            }
            const std::span<const u8> pixels{
                texture->Pixels, u64{static_cast<u32>(texture->Width)} * texture->Height * 4};
            if (texture->Status == ImTextureStatus_WantCreate) {
                auto resource = UploadTexture(pixels, texture->Width, texture->Height,
                                              MTLPixelFormatRGBA8Unorm);
                textures[texture].resource = resource;
                texture->SetTexID(TextureId(resource));
            } else {
                const auto found = textures.find(texture);
                if (found == textures.end() || found->second.resource.width != texture->Width ||
                    found->second.resource.height != texture->Height ||
                    texture->GetTexID() != TextureId(found->second.resource)) {
                    throw std::runtime_error("Invalid native Metal GUI font update target");
                }
                const auto& rectangle = texture->UpdateRect;
                if (rectangle.w && rectangle.h) {
                    UploadTextureRegion(
                        found->second.resource, pixels,
                        MTLRegionMake2D(rectangle.x, rectangle.y, rectangle.w, rectangle.h));
                }
            }
            texture->SetStatus(ImTextureStatus_OK);
        } else if (texture->Status == ImTextureStatus_WantDestroy) {
            textures.erase(texture);
            texture->SetTexID(ImTextureID_Invalid);
            texture->SetStatus(ImTextureStatus_Destroyed);
        }
    }
}

void GuiRenderer::ConvertFrame(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                               bool srgb_input, id<MTLFence> wait_fence,
                               id<MTLFence> signal_fence) {
    ConvertColorFrame(source, target, gamma, srgb_input ? 1u : 0u, 1.f, wait_fence, signal_fence);
}

void GuiRenderer::ConvertHdrFrame(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                                  id<MTLFence> wait_fence, id<MTLFence> signal_fence) {
    if (!source || !target || source == target || source.device != instance.GetDevice() ||
        target.device != instance.GetDevice() || source.textureType != MTLTextureType2D ||
        target.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        target.sampleCount != 1 || source.framebufferOnly || target.framebufferOnly ||
        source.pixelFormat != MTLPixelFormatRGBA16Float ||
        target.pixelFormat != MTLPixelFormatRGBA16Float ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        !(target.usage & MTLTextureUsageRenderTarget) || !source.width || !source.height ||
        source.width != target.width || source.height != target.height) {
        throw std::runtime_error("Invalid native Metal HDR conversion");
    }
    ConvertColorFrame(source, target, gamma, 9u, 1.f, wait_fence, signal_fence);
}

void GuiRenderer::ConvertPqFrame(id<MTLTexture> source, id<MTLTexture> target, float nits_per_unit,
                                 PqPrimaries primaries, id<MTLFence> wait_fence,
                                 id<MTLFence> signal_fence) {
    const float peak_multiplier = primaries == PqPrimaries::Srgb ? 1.6605f : 1.f;
    if (!source || !target || source == target || source.device != instance.GetDevice() ||
        target.device != instance.GetDevice() || source.textureType != MTLTextureType2D ||
        target.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        target.sampleCount != 1 || source.framebufferOnly || target.framebufferOnly ||
        (source.pixelFormat != MTLPixelFormatBGR10A2Unorm &&
         source.pixelFormat != MTLPixelFormatRGBA16Float) ||
        target.pixelFormat != MTLPixelFormatRGBA16Float ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        !(target.usage & MTLTextureUsageRenderTarget) || !source.width || !source.height ||
        source.width != target.width || source.height != target.height ||
        !std::isfinite(nits_per_unit) || nits_per_unit <= 0 ||
        (primaries != PqPrimaries::Bt2020 && primaries != PqPrimaries::Srgb) ||
        10000.f / nits_per_unit > 65504.f / peak_multiplier) {
        throw std::runtime_error("Invalid native Metal PQ linearization");
    }
    ConvertColorFrame(source, target, 1.f, primaries == PqPrimaries::Srgb ? 3u : 2u, nits_per_unit,
                      wait_fence, signal_fence);
}

void GuiRenderer::EncodePqFrame(id<MTLTexture> source, id<MTLTexture> target, float nits_per_unit,
                                PqPrimaries primaries, id<MTLFence> wait_fence,
                                id<MTLFence> signal_fence) {
    if (!source || !target || source.device != instance.GetDevice() ||
        target.device != instance.GetDevice() || source.textureType != MTLTextureType2D ||
        target.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        target.sampleCount != 1 || source.framebufferOnly ||
        source.pixelFormat != MTLPixelFormatRGBA16Float ||
        target.pixelFormat != MTLPixelFormatBGR10A2Unorm ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        !(target.usage & MTLTextureUsageRenderTarget) || !source.width || !source.height ||
        source.width != target.width || source.height != target.height ||
        !std::isfinite(nits_per_unit) || nits_per_unit <= 0 ||
        (primaries != PqPrimaries::Bt2020 && primaries != PqPrimaries::Srgb)) {
        throw std::runtime_error("Invalid native Metal PQ encoding");
    }
    ConvertColorFrame(source, target, 1.f, primaries == PqPrimaries::Srgb ? 5u : 4u, nits_per_unit,
                      wait_fence, signal_fence);
}

void GuiRenderer::ConvertLinearBt2020Frame(id<MTLTexture> source, id<MTLTexture> target,
                                           float gamma) {
    if (!source || !target || source == target || source.device != instance.GetDevice() ||
        target.device != instance.GetDevice() || source.textureType != MTLTextureType2D ||
        target.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        target.sampleCount != 1 || source.framebufferOnly || target.framebufferOnly ||
        source.pixelFormat != MTLPixelFormatRGBA16Float ||
        target.pixelFormat != MTLPixelFormatRGBA16Float ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        !(target.usage & MTLTextureUsageRenderTarget) || !source.width || !source.height ||
        source.width != target.width || source.height != target.height) {
        throw std::runtime_error("Invalid native Metal linear BT.2020 conversion");
    }
    ConvertColorFrame(source, target, gamma, 6u, 1.f, nil, nil);
}

void GuiRenderer::ConvertFsrHdrFrame(id<MTLTexture> source, id<MTLTexture> target,
                                     FsrHdrDirection direction) {
    if (!source || !target || source == target || source.device != instance.GetDevice() ||
        target.device != instance.GetDevice() || source.textureType != MTLTextureType2D ||
        target.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        target.sampleCount != 1 || source.framebufferOnly || target.framebufferOnly ||
        source.pixelFormat != (direction == FsrHdrDirection::Compress
                                   ? MTLPixelFormatRGBA16Float
                                   : MTLPixelFormatRGBA32Float) ||
        target.pixelFormat != (direction == FsrHdrDirection::Compress
                                   ? MTLPixelFormatRGBA32Float
                                   : MTLPixelFormatRGBA16Float) ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        !(target.usage & MTLTextureUsageRenderTarget) || !source.width || !source.height ||
        source.width != target.width || source.height != target.height ||
        (direction != FsrHdrDirection::Compress && direction != FsrHdrDirection::Expand)) {
        throw std::runtime_error("Invalid native Metal FSR HDR conversion");
    }
    ConvertColorFrame(source, target, 1.f, direction == FsrHdrDirection::Compress ? 7u : 8u, 1.f,
                      nil, nil);
}

void GuiRenderer::ConvertColorFrame(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                                    u32 transfer_function, float nits_per_unit,
                                    id<MTLFence> wait_fence, id<MTLFence> signal_fence) {
    if (!std::isfinite(gamma) || gamma <= 0) {
        throw std::runtime_error("Invalid native Metal presentation gamma");
    }
    struct Uniforms {
        float gamma;
        u32 transfer_function;
        float nits_per_unit;
        u32 half_output;
    };
    static_assert(sizeof(Uniforms) == 16);
    const Uniforms uniforms{gamma, transfer_function, nits_per_unit,
                            target.pixelFormat == MTLPixelFormatRGBA16Float};
    auto allocation = staging.Allocate(sizeof(uniforms));
    std::memcpy(allocation.UploadData().data(), &uniforms, sizeof(uniforms));
    ResourceBindings fragment{instance, 1, 1, 1};
    fragment.SetBuffer(0, allocation.GetBuffer(), allocation.GetOffset());
    fragment.RetainUploadAllocation(std::move(allocation));
    fragment.SetTexture(0, source);
    const bool fsr_hdr = transfer_function == 7u || transfer_function == 8u;
    if (!fsr_hdr) {
        fragment.SetSampler(0, sampler);
    }
    scheduler.ProducerBarrier(MTLStageAll, MTLStageFragment | MTLStageVertex);
    (void)commands.BindGraphics(
        TargetState(target), TargetGraphics(target),
        Pipeline(target.pixelFormat, fsr_hdr ? PipelineType::FsrHdr : PipelineType::Frame), nullptr,
        &fragment);
    auto encoder = scheduler.BeginRendering(TargetState(target));
    if (wait_fence) {
        scheduler.RetainObject(wait_fence);
        [encoder waitForFence:wait_fence beforeEncoderStages:MTLStageFragment];
    }
    commands.Draw(MTLPrimitiveTypeTriangle, 3);
    if (signal_fence) {
        scheduler.RetainObject(signal_fence);
        [encoder updateFence:signal_fence afterEncoderStages:MTLStageFragment];
    }
    scheduler.ProducerBarrier(MTLStageFragment, MTLStageAll);
}

void GuiRenderer::Render(ImDrawData& data, id<MTLTexture> target) {
    UpdateTextures(data);
    const auto render = TargetState(target);
    if (data.DisplaySize.x <= 0 || data.DisplaySize.y <= 0 || data.TotalVtxCount == 0) {
        (void)scheduler.BeginRendering(render);
        return;
    }
    struct Uniforms {
        float scale[2];
        float translate[2];
    };
    const Uniforms uniforms{{2.0f / data.DisplaySize.x, -2.0f / data.DisplaySize.y},
                            {-1.0f - data.DisplayPos.x * 2.0f / data.DisplaySize.x,
                             1.0f + data.DisplayPos.y * 2.0f / data.DisplaySize.y}};
    auto uniform_data = staging.Allocate(sizeof(uniforms));
    std::memcpy(uniform_data.UploadData().data(), &uniforms, sizeof(uniforms));
    ResourceBindings vertex{instance, 2, 0, 0};
    vertex.SetBuffer(1, uniform_data.GetBuffer(), uniform_data.GetOffset());
    vertex.RetainUploadAllocation(std::move(uniform_data));
    ResourceBindings fragment{instance, 0, 1, 1};
    fragment.SetSampler(0, sampler);
    auto graphics = TargetGraphics(target);
    auto pipeline = Pipeline(target.pixelFormat, PipelineType::Gui);
    scheduler.ProducerBarrier(MTLStageAll, MTLStageVertex | MTLStageFragment);
    for (const ImDrawList* list : data.CmdLists) {
        auto vertices = staging.Allocate(u64{std::max(1u, static_cast<u32>(list->VtxBuffer.Size))} *
                                         sizeof(ImDrawVert));
        auto indices = staging.Allocate(u64{std::max(1u, static_cast<u32>(list->IdxBuffer.Size))} *
                                        sizeof(ImDrawIdx));
        if (list->VtxBuffer.Size) {
            std::memcpy(vertices.UploadData().data(), list->VtxBuffer.Data,
                        u64{static_cast<u32>(list->VtxBuffer.Size)} * sizeof(ImDrawVert));
        }
        if (list->IdxBuffer.Size) {
            std::memcpy(indices.UploadData().data(), list->IdxBuffer.Data,
                        u64{static_cast<u32>(list->IdxBuffer.Size)} * sizeof(ImDrawIdx));
        }
        vertex.SetBuffer(0, vertices.GetBuffer(), vertices.GetOffset());
        vertex.RetainUploadAllocation(std::move(vertices));
        for (const ImDrawCmd& command : list->CmdBuffer) {
            if (command.UserCallback) {
                if (command.UserCallback != ImDrawCallback_ResetRenderState) {
                    command.UserCallback(list, &command);
                }
                scheduler.InvalidateEncoderState();
                continue;
            }
            if (!command.ElemCount) {
                continue;
            }
            const auto x0 =
                std::clamp((command.ClipRect.x - data.DisplayPos.x) * data.FramebufferScale.x, 0.0f,
                           static_cast<float>(target.width));
            const auto y0 =
                std::clamp((command.ClipRect.y - data.DisplayPos.y) * data.FramebufferScale.y, 0.0f,
                           static_cast<float>(target.height));
            const auto x1 =
                std::clamp((command.ClipRect.z - data.DisplayPos.x) * data.FramebufferScale.x, 0.0f,
                           static_cast<float>(target.width));
            const auto y1 =
                std::clamp((command.ClipRect.w - data.DisplayPos.y) * data.FramebufferScale.y, 0.0f,
                           static_cast<float>(target.height));
            if (x1 <= x0 || y1 <= y0) {
                continue;
            }
            graphics.scissors[0] = {static_cast<u32>(x0), static_cast<u32>(y0),
                                    static_cast<u32>(std::ceil(x1)) - static_cast<u32>(x0),
                                    static_cast<u32>(std::ceil(y1)) - static_cast<u32>(y0)};
            auto texture = (__bridge id<MTLTexture>)reinterpret_cast<void*>(command.GetTexID());
            if (!texture || texture.device != instance.GetDevice()) {
                throw std::runtime_error("Invalid native Metal GUI texture");
            }
            fragment.SetTexture(0, texture);
            (void)commands.BindGraphics(render, graphics, pipeline, &vertex, &fragment);
            commands.DrawIndexed(MTLPrimitiveTypeTriangle, MTLIndexTypeUInt32,
                                 {indices.GetBuffer(),
                                  indices.GetOffset() + u64{command.IdxOffset} * sizeof(ImDrawIdx),
                                  u64{command.ElemCount} * sizeof(ImDrawIdx)},
                                 command.ElemCount, 1, command.VtxOffset);
        }
    }
    scheduler.ProducerBarrier(MTLStageFragment, MTLStageAll);
}

} // namespace Metal
