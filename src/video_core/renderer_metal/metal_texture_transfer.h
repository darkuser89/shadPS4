// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace Metal {

struct TextureRegion {
    id<MTLTexture> texture;
    u32 level{};
    u32 slice{};
    MTLOrigin origin{};
    MTLSize size{};
};

struct TextureBufferLayout {
    u64 bytes_per_row{};
    u64 bytes_per_image{};
    MTLBlitOption options{MTLBlitOptionNone};
};

class TextureTransfers {
public:
    TextureTransfers(const Instance& instance, Scheduler& scheduler);
    ~TextureTransfers();

    void Upload(const BufferRange& source, const TextureRegion& destination,
                const TextureBufferLayout& layout);
    void Download(const TextureRegion& source, const BufferRange& destination,
                  const TextureBufferLayout& layout);
    void Copy(const TextureRegion& source, const TextureRegion& destination);
    void Copy(id<MTLTexture> source, id<MTLTexture> destination);

private:
    void Transfer(const BufferRange& buffer, const TextureRegion& region,
                  const TextureBufferLayout& layout, bool upload);
    void EncodeTransfer(const BufferRange& buffer, const TextureRegion& region,
                        const TextureBufferLayout& layout, bool upload);
    void ValidateRegion(const TextureRegion& region) const;
    void ValidateBufferLayout(const BufferRange& buffer, const TextureRegion& region,
                              const TextureBufferLayout& layout) const;
    void CopyRows(const BufferRange& source, const BufferRange& destination, u64 source_stride,
                  u64 destination_stride, u32 row_bytes, u32 rows);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> row_pipeline;
    id<MTLBuffer> scratch;
};

} // namespace Metal
