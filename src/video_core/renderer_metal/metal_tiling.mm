// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <fmt/format.h>
#include "common/hash.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_tiling.h"
#include "video_core/renderer_metal/metal_tiling_shader.h"

namespace Metal {

TileManager::TileManager(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, compiler{instance}, commands{instance, scheduler},
      staging{instance, scheduler},
      library{compiler.CompileLibraryAsync(TilingShaderSource, "metal4_gcn_tiling")} {}

size_t TileManager::KeyHash::operator()(const Key& key) const noexcept {
    u64 hash{};
    for (const u32 value : key) {
        hash = HashCombine(hash, u64{value});
    }
    return hash;
}

TileManager::Key TileManager::BuildKey(const TilingLayout& layout, bool is_tiler) const {
    static constexpr std::array supported_bits{8u, 16u, 32u, 64u, 96u, 128u};
    if (std::ranges::find(supported_bits, layout.num_bits) == supported_bits.end() ||
        !std::has_single_bit(layout.num_samples) || layout.num_samples > 8 ||
        u32(layout.tile_mode) >= AmdGpu::NUM_TILE_MODES) {
        throw std::runtime_error("Invalid Metal GCN tiling configuration");
    }
    const auto array_mode = AmdGpu::GetArrayMode(layout.tile_mode);
    if (array_mode == AmdGpu::ArrayMode::ArrayLinearGeneral ||
        array_mode == AmdGpu::ArrayMode::ArrayLinearAligned) {
        throw std::runtime_error("Metal linear surfaces do not require tiling conversion");
    }
    const auto micro_mode = AmdGpu::GetMicroTileMode(layout.tile_mode);
    if (micro_mode == AmdGpu::MicroTileMode::Rotated ||
        (layout.num_bits == 96 && micro_mode != AmdGpu::MicroTileMode::Thin &&
         micro_mode != AmdGpu::MicroTileMode::Depth)) {
        throw std::runtime_error("Unsupported Metal GCN microtile format");
    }
    const bool macro = AmdGpu::IsMacroTiled(array_mode);
    Key key{layout.num_samples,
            u32(micro_mode),
            AmdGpu::GetMicroTileThickness(array_mode),
            u32(is_tiler),
            u32(array_mode),
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            u32(macro),
            layout.num_bits};
    if (macro) {
        const auto mode =
            AmdGpu::CalculateMacrotileMode(layout.tile_mode, layout.num_bits, layout.num_samples);
        const auto pipe = layout.alt_tile ? AmdGpu::GetAltPipeConfig(layout.tile_mode)
                                          : AmdGpu::GetPipeConfig(layout.tile_mode);
        if (pipe != AmdGpu::PipeConfig::P2 && pipe != AmdGpu::PipeConfig::P8_32x32_8x16 &&
            pipe != AmdGpu::PipeConfig::P8_32x32_16x16 &&
            pipe != AmdGpu::PipeConfig::P16_32x32_8x16 &&
            pipe != AmdGpu::PipeConfig::P16_32x32_16x16) {
            throw std::runtime_error("Unsupported Metal GCN pipe configuration");
        }
        key[5] = u32(pipe);
        key[6] = AmdGpu::GetBankWidth(mode);
        key[7] = layout.alt_tile ? AmdGpu::GetAltBankHeight(mode) : AmdGpu::GetBankHeight(mode);
        key[8] = layout.alt_tile ? AmdGpu::GetAltNumBanks(mode) : AmdGpu::GetNumBanks(mode);
        key[9] = std::bit_width(key[8]) - 1;
        key[10] =
            AmdGpu::CalculateTileSplit(layout.tile_mode, array_mode, micro_mode, layout.num_bits);
        key[11] = layout.alt_tile ? AmdGpu::GetAltMacrotileAspect(mode)
                                  : AmdGpu::GetMacrotileAspect(mode);
    }
    return key;
}

TileManager::Parameters TileManager::BuildParameters(const TilingLayout& layout,
                                                     const Key& key) const {
    const u32 bytes = layout.num_bits / 8;
    if (!layout.guest_size || layout.guest_size > std::numeric_limits<u32>::max() ||
        layout.guest_size % bytes || !layout.num_mips || layout.num_mips > layout.mips.size() ||
        (layout.micro_tiled_mips >> layout.num_mips)) {
        throw std::runtime_error("Invalid Metal GCN tiling mip layout");
    }
    Parameters params{layout.bank_swizzle, layout.micro_tiled_mips, layout.num_mips,
                      u32(layout.guest_size / bytes), layout.mips};
    const u32 pipes = key[5] == 0 ? 2 : (key[5] >= 16 ? 16 : 8);
    const u32 macro_pitch = 8 * key[6] * pipes * key[11];
    const u32 macro_height = 8 * key[7] * key[8] / key[11];
    u64 offset{};
    for (u32 i = 0; i < params.num_mips; ++i) {
        auto& mip = params.mips[i];
        if (layout.block_compressed) {
            mip.pitch = std::max<u16>((mip.pitch + 3) / 4, 1);
            mip.height = std::max<u16>((mip.height + 3) / 4, 1);
        }
        const bool macro = key[12] && !(params.micro_tiled_mips & (1u << i));
        const u32 thickness = key[12] && !macro ? std::min(key[2], 4u) : key[2];
        const u64 slice_size = u64{mip.pitch} * mip.height * bytes * layout.num_samples;
        if (!mip.size || !slice_size || mip.offset != offset || mip.size % slice_size ||
            (mip.size / slice_size) % thickness || mip.pitch % (macro ? macro_pitch : 8) ||
            mip.height % (macro ? macro_height : 8)) {
            throw std::runtime_error("Invalid Metal GCN tiling slice or alignment");
        }
        offset += mip.size;
        if (offset > layout.guest_size) {
            throw std::runtime_error("Metal GCN tiling mip exceeds guest allocation");
        }
    }
    if (offset != layout.guest_size) {
        throw std::runtime_error("Metal GCN tiling mip sizes do not cover guest allocation");
    }
    return params;
}

id<MTLComputePipelineState> TileManager::GetPipeline(const Key& key) {
    if (const auto found = pipelines.find(key); found != pipelines.end()) {
        return found->second.get();
    }
    @autoreleasepool {
        MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
        function.library = library.get();
        function.name = @"transform_tiles";
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        for (u32 i = 0; i < key.size(); ++i) {
            [constants setConstantValue:&key[i] type:MTLDataTypeUInt atIndex:i];
        }
        MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
        specialized.functionDescriptor = function;
        specialized.constantValues = constants;
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = specialized;
        descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(64, 1, 1);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        auto pending = compiler.CompileComputePipelineAsync(descriptor);
        return pipelines.emplace(key, std::move(pending)).first->second.get();
    }
}

void TileManager::ValidateBuffer(const BufferRange& buffer, u64 size, u64 alignment) const {
    if (!buffer.buffer || buffer.buffer.device != instance.GetDevice() || buffer.size < size ||
        buffer.offset % alignment || buffer.offset > buffer.buffer.length ||
        buffer.size > buffer.buffer.length - buffer.offset) {
        throw std::runtime_error("Invalid Metal GCN tiling buffer range");
    }
}

void TileManager::Transform(const BufferRange& tiled, const BufferRange& linear,
                            const TilingLayout& layout, bool is_tiler) {
    const auto key = BuildKey(layout, is_tiler);
    const auto params = BuildParameters(layout, key);
    const u64 alignment = layout.num_bits == 96 ? 4 : layout.num_bits / 8;
    ValidateBuffer(tiled, layout.guest_size, alignment);
    ValidateBuffer(linear, layout.guest_size, alignment);
    if (tiled.buffer == linear.buffer && tiled.offset < linear.offset + layout.guest_size &&
        linear.offset < tiled.offset + layout.guest_size) {
        throw std::runtime_error("Metal GCN tiling conversion requires nonoverlapping buffers");
    }
    const auto pipeline = GetPipeline(key);
    auto allocation = staging.Allocate(sizeof(params), 16);
    std::memcpy(allocation.UploadData().data(), &params, sizeof(params));
    ResourceBindings bindings{instance, 4, 0, 0};
    bindings.SetBuffer(0, tiled.buffer, tiled.offset);
    bindings.SetBuffer(1, linear.buffer, linear.offset);
    bindings.SetBuffer(2, allocation.GetBuffer(), allocation.GetOffset());
    if (errors.empty() || errors.back().tick != scheduler.CurrentTick()) {
        auto error = staging.AllocateReadback(sizeof(u32), alignof(u32));
        commands.FillBuffer({error.GetBuffer(), error.GetOffset(), error.SizeBytes()}, 0);
        errors.push_back({std::move(error), scheduler.CurrentTick()});
    }
    const auto& error = errors.back().allocation;
    bindings.SetBuffer(3, error.GetBuffer(), error.GetOffset());
    bindings.RetainUploadAllocation(std::move(allocation));
    commands.BindCompute(pipeline, bindings);
    scheduler.QueueBarrier(MTLStageAll, MTLStageDispatch);
    scheduler.EncoderBarrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
    commands.Dispatch(MTLSizeMake((u64{params.element_count} + 63) / 64, 1, 1),
                      MTLSizeMake(64, 1, 1));
    scheduler.ProducerBarrier(MTLStageDispatch, MTLStageAll);
    scheduler.EncoderBarrier(MTLStageDispatch, MTLStageDispatch | MTLStageBlit);
}

void TileManager::CheckErrors() {
    while (!errors.empty() && errors.front().allocation.IsReady()) {
        u32 flags{};
        std::memcpy(&flags, errors.front().allocation.ReadbackData().data(), sizeof(flags));
        errors.pop_front();
        if (flags) {
            throw std::runtime_error(
                fmt::format("Metal GCN tiling address validation failed: {}", flags));
        }
    }
}

} // namespace Metal
