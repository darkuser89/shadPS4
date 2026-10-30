// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "metal_fsr_easu.h"
#include "metal_fsr_easu_hdr.h"
#include "metal_fsr_rcas.h"
#include "metal_fsr_rcas_hdr.h"
#include "video_core/renderer_metal/metal_fsr.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

#define A_CPU
#include "video_core/host_shaders/fsr/ffx_a.h"
#include "video_core/host_shaders/fsr/ffx_fsr1.h"

namespace Metal {
namespace {

struct alignas(16) Constants {
    u32 con0[4];
    u32 con1[4];
    u32 con2[4];
    u32 con3[4];
    u32 control[4];
};
static_assert(sizeof(Constants) == 80 && offsetof(Constants, control) == 64);

} // namespace

FsrPass::FsrPass(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, commands{instance, scheduler}, compiler{instance},
      staging{instance, scheduler} {}

FsrPass::Kernel FsrPass::CreateKernel(std::span<const u32> spirv, std::string_view name) {
    CompiledShader shader;
    shader.translated = ShaderCompiler::TranslateSpirv(spirv);
    shader.library = compiler.CompileLibraryAsync(shader.translated.source, name);
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = shader.library.get();
    const auto& entry = shader.translated.entry_point;
    function.name = [[NSString alloc] initWithBytes:entry.data()
                                             length:entry.size()
                                           encoding:NSUTF8StringEncoding];
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
    auto pipeline = compiler.CompileComputePipeline(descriptor);
    if (pipeline.maxTotalThreadsPerThreadgroup < 64 || !pipeline.threadExecutionWidth ||
        64 % pipeline.threadExecutionWidth != 0) {
        throw std::runtime_error("Invalid native Metal FSR workgroup limits");
    }
    return {pipeline, std::make_unique<ShaderBindings>(instance, shader)};
}

id<MTLTexture> FsrPass::Output(id<MTLTexture> __strong& texture, u32 width, u32 height,
                               MTLPixelFormat format) {
    if (!texture || texture.pixelFormat != format || texture.width != width ||
        texture.height != height) {
        auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:width
                                                                            height:height
                                                                         mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        texture = instance.CreateTexture(descriptor);
    }
    return texture;
}

void FsrPass::Dispatch(Kernel& kernel, id<MTLTexture> source, id<MTLTexture> destination,
                       std::span<const u8> constants) {
    kernel.bindings->SetTexture(0, source);
    kernel.bindings->SetTexture(1, destination);
    if (kernel.bindings->UsesBinding(2)) {
        kernel.bindings->SetSampler(2, sampler);
    }
    auto snapshot = kernel.bindings->CreateSnapshot(staging, constants);
    scheduler.EndEncoding();
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    commands.BindCompute(kernel.pipeline, *snapshot);
    commands.Dispatch(MTLSizeMake((destination.width + 15) / 16, (destination.height + 15) / 16, 1),
                      MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
}

id<MTLTexture> FsrPass::Render(id<MTLTexture> source, u32 width, u32 height,
                               const VideoCore::FsrSettings& settings, bool perceptual_input) {
    if (!source || source.device != instance.GetDevice() ||
        source.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        source.framebufferOnly || !width || !height || width > 16384 || height > 16384 ||
        (settings.use_rcas &&
         (!std::isfinite(settings.rcas_attenuation) || settings.rcas_attenuation < 0))) {
        throw std::runtime_error("Invalid native Metal FSR input");
    }
    if (!settings.enable || (source.width >= width && source.height >= height)) {
        return source;
    }
    const bool full_precision = source.pixelFormat == MTLPixelFormatRGBA32Float;
    auto& easu_kernel = full_precision ? hdr_easu : easu;
    auto& rcas_kernel = full_precision ? hdr_rcas : rcas;
    if (!easu_kernel.pipeline) {
        easu_kernel = full_precision ? CreateKernel(METAL_FSR_EASU_HDR, "shadPS4:Metal4FsrHdrEasu")
                                     : CreateKernel(METAL_FSR_EASU, "shadPS4:Metal4FsrEasu");
    }
    if (!sampler) {
        auto descriptor = [MTLSamplerDescriptor new];
        descriptor.minFilter = MTLSamplerMinMagFilterLinear;
        descriptor.magFilter = MTLSamplerMinMagFilterLinear;
        descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler = instance.CreateSampler(descriptor);
    }
    if (settings.use_rcas && !rcas_kernel.pipeline) {
        rcas_kernel = full_precision ? CreateKernel(METAL_FSR_RCAS_HDR, "shadPS4:Metal4FsrHdrRcas")
                                     : CreateKernel(METAL_FSR_RCAS, "shadPS4:Metal4FsrRcas");
    }
    Constants constants{};
    FsrEasuCon(constants.con0, constants.con1, constants.con2, constants.con3, float(source.width),
               float(source.height), float(source.width), float(source.height), float(width),
               float(height));
    constants.control[0] = !settings.use_rcas;
    constants.control[1] = perceptual_input;
    const auto format = full_precision ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA16Float;
    auto result = Output(output, width, height, format);
    auto easu_output = settings.use_rcas ? Output(intermediate, width, height, format) : result;
    Dispatch(easu_kernel, source, easu_output,
             {reinterpret_cast<const u8*>(&constants), sizeof(constants)});
    if (settings.use_rcas) {
        constants = {};
        FsrRcasCon(constants.con0, settings.rcas_attenuation);
        constants.control[0] = 1;
        Dispatch(rcas_kernel, easu_output, result,
                 {reinterpret_cast<const u8*>(&constants), sizeof(constants)});
    }
    return result;
}

} // namespace Metal
