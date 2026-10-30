// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <stdexcept>
#include "video_core/renderer_metal/metal_fx.h"
#include "video_core/renderer_metal/metal_gui.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

MetalFxPass::MetalFxPass(const Instance& instance_, Scheduler& scheduler_, GuiRenderer& converter_)
    : instance{instance_}, scheduler{scheduler_}, converter{converter_} {}

bool MetalFxPass::IsSupported() const {
    return [MTLFXSpatialScalerDescriptor supportsMetal4FX:instance.GetDevice()];
}

void MetalFxPass::Render(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                         MTLFXSpatialScalerColorProcessingMode mode) {
    if (!source || !target || source.device != instance.GetDevice() ||
        target.device != instance.GetDevice() || source.textureType != MTLTextureType2D ||
        target.textureType != MTLTextureType2D || source.sampleCount != 1 ||
        target.sampleCount != 1 || source.framebufferOnly || target.framebufferOnly ||
        !(source.usage & MTLTextureUsageShaderRead) ||
        !(target.usage & MTLTextureUsageRenderTarget) || !source.width || !source.height ||
        target.width < source.width || target.height < source.height ||
        (target.width == source.width && target.height == source.height) || target.width > 16384 ||
        target.height > 16384 || !std::isfinite(gamma) || gamma <= 0 ||
        (mode == MTLFXSpatialScalerColorProcessingModeHDR &&
         (source.pixelFormat != MTLPixelFormatRGBA16Float ||
          target.pixelFormat != MTLPixelFormatRGBA16Float)) ||
        (mode != MTLFXSpatialScalerColorProcessingModePerceptual &&
         mode != MTLFXSpatialScalerColorProcessingModeLinear &&
         mode != MTLFXSpatialScalerColorProcessingModeHDR)) {
        throw std::runtime_error("Invalid native MetalFX spatial upscale");
    }
    if (!IsSupported()) {
        throw std::runtime_error("The selected Apple GPU does not support Metal 4 MetalFX");
    }
    if (!scaler || scaler.inputWidth != source.width || scaler.inputHeight != source.height ||
        scaler.outputWidth != target.width || scaler.outputHeight != target.height ||
        scaler.colorProcessingMode != mode) {
        auto descriptor = [MTLFXSpatialScalerDescriptor new];
        descriptor.inputWidth = source.width;
        descriptor.inputHeight = source.height;
        descriptor.outputWidth = target.width;
        descriptor.outputHeight = target.height;
        descriptor.colorTextureFormat = MTLPixelFormatRGBA16Float;
        descriptor.outputTextureFormat = MTLPixelFormatRGBA16Float;
        descriptor.colorProcessingMode = mode;
        auto next = [descriptor newSpatialScalerWithDevice:instance.GetDevice()
                                                  compiler:instance.GetCompiler()];
        if (!next) {
            throw std::runtime_error("Failed to create Metal 4 MetalFX spatial scaler");
        }
        auto texture =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                               width:source.width
                                                              height:source.height
                                                           mipmapped:NO];
        texture.storageMode = MTLStorageModePrivate;
        texture.usage =
            next.colorTextureUsage | MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        auto next_input = input;
        if (!next_input || next_input.width != texture.width ||
            next_input.height != texture.height ||
            (next_input.usage & texture.usage) != texture.usage) {
            next_input = instance.CreateTexture(texture);
        }
        texture.width = target.width;
        texture.height = target.height;
        texture.usage = next.outputTextureUsage | MTLTextureUsageShaderRead;
        auto next_output = output;
        if (!next_output || next_output.width != texture.width ||
            next_output.height != texture.height ||
            (next_output.usage & texture.usage) != texture.usage) {
            next_output = instance.CreateTexture(texture);
        }
        scaler = next;
        input = next_input;
        output = next_output;
    }
    if (!fence) {
        fence = [instance.GetDevice() newFence];
        if (!fence) {
            throw std::runtime_error("Failed to create MetalFX synchronization fence");
        }
        fence.label = @"shadPS4:Metal4MetalFX";
    }
    if (mode == MTLFXSpatialScalerColorProcessingModeHDR) {
        converter.ConvertHdrFrame(source, input, 1.f, has_encoded ? fence : nil, fence);
    } else {
        converter.ConvertFrame(source, input, 1.f, false, has_encoded ? fence : nil, fence);
    }
    scheduler.EndEncoding();
    scheduler.UseResource(input);
    scheduler.UseResource(output);
    scheduler.RetainObject(scaler);
    scheduler.RetainObject(fence);
    scaler.colorTexture = input;
    scaler.outputTexture = output;
    scaler.inputContentWidth = source.width;
    scaler.inputContentHeight = source.height;
    scaler.fence = fence;
    [scaler encodeToCommandBuffer:scheduler.CommandBuffer()];
    scheduler.InvalidateEncoderState();
    if (mode == MTLFXSpatialScalerColorProcessingModeHDR) {
        converter.ConvertHdrFrame(output, target, gamma, fence, fence);
    } else {
        converter.ConvertFrame(output, target, gamma,
                               mode == MTLFXSpatialScalerColorProcessingModePerceptual, fence,
                               fence);
    }
    has_encoded = true;
}

} // namespace Metal
