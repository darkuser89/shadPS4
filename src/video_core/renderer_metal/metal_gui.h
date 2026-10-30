// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <imgui.h>
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_texture_transfer.h"

namespace Metal {

class GuiRenderer {
public:
    enum class PqPrimaries : u32 {
        Bt2020,
        Srgb,
    };

    enum class FsrHdrDirection : u32 {
        Compress,
        Expand,
    };

    GuiRenderer(const Instance& instance, Scheduler& scheduler);
    ~GuiRenderer();
    void ReleaseTextures();

    void Render(ImDrawData& data, id<MTLTexture> target);
    void ConvertFrame(id<MTLTexture> source, id<MTLTexture> target, float gamma, bool srgb_input,
                      id<MTLFence> wait_fence = nil, id<MTLFence> signal_fence = nil);
    void ConvertHdrFrame(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                         id<MTLFence> wait_fence = nil, id<MTLFence> signal_fence = nil);
    void ConvertPqFrame(id<MTLTexture> source, id<MTLTexture> target, float nits_per_unit,
                        PqPrimaries primaries, id<MTLFence> wait_fence = nil,
                        id<MTLFence> signal_fence = nil);
    void EncodePqFrame(id<MTLTexture> source, id<MTLTexture> target, float nits_per_unit,
                       PqPrimaries primaries, id<MTLFence> wait_fence = nil,
                       id<MTLFence> signal_fence = nil);
    void ConvertLinearBt2020Frame(id<MTLTexture> source, id<MTLTexture> target, float gamma);
    void ConvertFsrHdrFrame(id<MTLTexture> source, id<MTLTexture> target,
                            FsrHdrDirection direction);
    [[nodiscard]] ImGui::Core::TextureManager::NativeTexture Upload(std::span<const u8> pixels,
                                                                    u32 width, u32 height);
    [[nodiscard]] static ImTextureID TextureId(id<MTLTexture> texture);

private:
    enum class PipelineType {
        Gui,
        Frame,
        FsrHdr,
    };

    struct Texture {
        id<MTLTexture> resource;
    };
    [[nodiscard]] id<MTLRenderPipelineState> Pipeline(MTLPixelFormat format, PipelineType type);
    [[nodiscard]] id<MTLTexture> UploadTexture(std::span<const u8> pixels, u32 width, u32 height,
                                               MTLPixelFormat format);
    void UploadTextureRegion(id<MTLTexture> texture, std::span<const u8> pixels,
                             const MTLRegion& region);
    void UpdateTextures(ImDrawData& data);
    void ConvertColorFrame(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                           u32 transfer_function, float nits_per_unit, id<MTLFence> wait_fence,
                           id<MTLFence> signal_fence);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    TextureTransfers transfers;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLLibrary> library;
    id<MTLSamplerState> sampler;
    std::map<std::pair<MTLPixelFormat, PipelineType>, id<MTLRenderPipelineState>> pipelines;
    std::map<ImTextureData*, Texture> textures;
};

} // namespace Metal
