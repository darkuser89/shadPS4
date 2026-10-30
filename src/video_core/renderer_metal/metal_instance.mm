// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <mutex>
#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_pipeline_archive.h"
#include "video_core/renderer_metal/metal_platform.h"

namespace Metal {

struct Instance::ArgumentTables {
    struct Entry {
        id<MTL4ArgumentTable> table;
        std::array<u32, 3> layout{};
    };

    std::mutex mutex;
    std::array<Entry, 64> available;
    size_t replacement{};
};

Instance::Instance() : argument_tables{std::make_unique<ArgumentTables>()} {
    RequireAppleSilicon();
    @autoreleasepool {
        for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
            if (candidate.hasUnifiedMemory && [candidate supportsFamily:MTLGPUFamilyApple7] &&
                [candidate supportsFamily:MTLGPUFamilyMetal4]) {
                device = candidate;
                break;
            }
        }
        if (!device) {
            UNREACHABLE_MSG("No Apple Silicon GPU with Metal 4 support found");
        }
        device_name = device.name.UTF8String;
        supports_depth_bounds = [device supportsFamily:MTLGPUFamilyApple10];
        supports_sampler_lod_bias = [device supportsFamily:MTLGPUFamilyApple10];

        MTL4CompilerDescriptor* descriptor = [MTL4CompilerDescriptor new];
        descriptor.label = @"shadPS4:Metal4Compiler";
        pipeline_archive = std::make_unique<PipelineArchive>(device);
        descriptor.pipelineDataSetSerializer = pipeline_archive->GetSerializer();
        NSError* error = nil;
        compiler = [device newCompilerWithDescriptor:descriptor error:&error];
        if (!compiler) {
            UNREACHABLE_MSG("Failed to create Metal 4 compiler: {}", ErrorDescription(error));
        }
    }
}

Instance::~Instance() = default;

PipelineArchive& Instance::GetPipelineArchive() const noexcept {
    return *pipeline_archive;
}

id<MTL4ArgumentTable> Instance::AcquireArgumentTable(u32 buffer_count, u32 texture_count,
                                                     u32 sampler_count) const {
    const std::array layout{buffer_count, texture_count, sampler_count};
    {
        std::scoped_lock lock{argument_tables->mutex};
        for (auto& entry : argument_tables->available) {
            if (entry.table && entry.layout == layout) {
                id<MTL4ArgumentTable> table = entry.table;
                entry.table = nil;
                return table;
            }
        }
    }
    @autoreleasepool {
        MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = buffer_count;
        descriptor.maxTextureBindCount = texture_count;
        descriptor.maxSamplerStateBindCount = sampler_count;
        descriptor.initializeBindings = YES;
        NSError* error = nil;
        id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:descriptor
                                                                       error:&error];
        if (!table) {
            UNREACHABLE_MSG("Failed to create Metal 4 argument table: {}", ErrorDescription(error));
        }
        return table;
    }
}

void Instance::ReleaseArgumentTable(id<MTL4ArgumentTable> table, u32 buffer_count,
                                    u32 texture_count, u32 sampler_count) const {
    const std::array layout{buffer_count, texture_count, sampler_count};
    std::scoped_lock lock{argument_tables->mutex};
    ArgumentTables::Entry* available{};
    u32 matching{};
    for (auto& entry : argument_tables->available) {
        if (!entry.table) {
            if (!available) {
                available = &entry;
            }
        } else if (entry.layout == layout) {
            ++matching;
        }
    }
    if (matching < 4) {
        if (!available) {
            available = &argument_tables->available[argument_tables->replacement];
            argument_tables->replacement =
                (argument_tables->replacement + 1) % argument_tables->available.size();
        }
        *available = {table, layout};
    }
}

id<MTLBuffer> Instance::CreateBuffer(u64 size, MTLResourceOptions options) const {
    if (!size || size > device.maxBufferLength) {
        UNREACHABLE_MSG("Invalid Metal buffer size: {}", size);
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];
    if (!buffer) {
        UNREACHABLE_MSG("Failed to allocate Metal buffer: {} bytes", size);
    }
    return buffer;
}

id<MTLTexture> Instance::CreateTexture(MTLTextureDescriptor* descriptor) const {
    ASSERT(descriptor);
    @autoreleasepool {
        MTLTextureDescriptor* configuration = [descriptor copy];
        id<MTLTexture> texture = [device newTextureWithDescriptor:configuration];
        if (!texture) {
            UNREACHABLE_MSG("Failed to allocate Metal texture: {}x{}x{}", configuration.width,
                            configuration.height, configuration.depth);
        }
        return texture;
    }
}

id<MTLSamplerState> Instance::CreateSampler(MTLSamplerDescriptor* descriptor) const {
    ASSERT(descriptor);
    @autoreleasepool {
        MTLSamplerDescriptor* configuration = [descriptor copy];
        configuration.supportArgumentBuffers = YES;
        id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:configuration];
        if (!sampler) {
            UNREACHABLE_MSG("Failed to create Metal argument-table sampler");
        }
        return sampler;
    }
}

id<MTLDepthStencilState> Instance::CreateDepthStencilState(
    MTLDepthStencilDescriptor* descriptor) const {
    ASSERT(descriptor);
    @autoreleasepool {
        id<MTLDepthStencilState> state =
            [device newDepthStencilStateWithDescriptor:[descriptor copy]];
        if (!state) {
            UNREACHABLE_MSG("Failed to create Metal depth/stencil state");
        }
        return state;
    }
}

} // namespace Metal
