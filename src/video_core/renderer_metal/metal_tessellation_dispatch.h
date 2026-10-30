// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <limits>
#include "video_core/renderer_metal/metal_guest_resources.h"
#include "video_core/renderer_metal/metal_tessellation_domains.h"
#include "video_core/renderer_metal/metal_tessellation_evaluation.h"
#include "video_core/renderer_metal/metal_tessellation_exports.h"
#include "video_core/renderer_metal/metal_tessellation_factors.h"
#include "video_core/renderer_metal/metal_tessellation_work.h"

namespace AmdGpu {
union Regs;
}

namespace Metal {

class TessellationDispatch {
public:
    struct Ring {
        u32 resource_index;
        VAddr address;
        u64 size;
        std::vector<GeometryBufferByteInterval> group_ranges;
        std::vector<u64> group_snapshot_offsets;
        std::shared_ptr<StagingBufferPool::Allocation> snapshot;
    };

    struct FactorRing : Ring {
        std::optional<TessellationFactors::Result> decoded;
    };

    struct Result {
        TessellationKernelConfig config;
        std::shared_ptr<StagingBufferPool::Allocation> lds;
        std::optional<FactorRing> factors;
        std::optional<Ring> offchip;
        u32 first_group{};
    };

    struct RasterChunk {
        std::shared_ptr<StagingBufferPool::Allocation> vertices;
        u32 vertex_words;
        std::vector<GeometryOutputAttribute> attributes;
    };

    TessellationDispatch(const Instance& instance, Scheduler& scheduler, BufferCache& buffer_cache,
                         ShaderCache& shaders, GuestResources& resources);

    [[nodiscard]] TessellationWorkLayout PlanLocalHull(const AmdGpu::Regs& regs);

    void RunDomainBatches(const AmdGpu::Regs& regs, const Result& hull,
                          const std::function<void(TessellationDomains::Result&&)>& consume,
                          u32 maximum_patches = std::numeric_limits<u32>::max());

    void RunEvaluationBatches(const AmdGpu::Regs& regs, const Result& hull, bool neo_mode,
                              const std::function<void(TessellationExports::Batch&&)>& consume,
                              std::span<const ImageInfo> attachments = {});
    void RunRasterBatches(const AmdGpu::Regs& regs, const Result& hull, bool neo_mode,
                          const std::function<void(TessellationExports::Batch&&)>& consume,
                          std::span<const ImageInfo> attachments = {});
    [[nodiscard]] TessellationRasterBindings DescribeRaster(const AmdGpu::Regs& regs);

    void ValidateDomains(const TessellationDomains::Result& domains) const;

    [[nodiscard]] std::optional<RasterChunk> RunEvaluationRaster(
        const AmdGpu::Regs& regs, const Result& hull, const TessellationDomains::Result& domains,
        u32 relative_patch, u32 first_point, bool neo_mode,
        std::span<const ImageInfo> attachments = {});

    [[nodiscard]] std::optional<Ring> RunEvaluationExport(
        const AmdGpu::Regs& regs, const Result& hull, const TessellationDomains::Result& domains,
        u32 relative_patch, u32 first_point, bool neo_mode,
        std::span<const ImageInfo> attachments = {});

    [[nodiscard]] Ring RunEvaluationExport(
        const AmdGpu::Regs& regs, const Result& hull,
        std::span<const std::span<const TessellationEvaluationPoint>> points, bool neo_mode,
        std::span<const ImageInfo> attachments = {});

    [[nodiscard]] Ring RunEvaluationExport(const AmdGpu::Regs& regs, const Result& hull,
                                           std::span<const TessellationEvaluationWave> waves,
                                           bool neo_mode,
                                           std::span<const ImageInfo> attachments = {});

    void RunGuestLocalHullBatches(const AmdGpu::Regs& regs, bool indexed, u32 first_index,
                                  bool neo_mode, const std::function<void(Result&&)>& consume,
                                  std::span<const ImageInfo> attachments = {},
                                  u32 maximum_groups = std::numeric_limits<u32>::max());

    [[nodiscard]] Result RunGuestLocalHull(const AmdGpu::Regs& regs, bool indexed, u32 first_index,
                                           bool neo_mode,
                                           std::span<const ImageInfo> attachments = {});

    [[nodiscard]] Result RunGuestLocalHull(const AmdGpu::Regs& regs, bool indexed, u32 first_index,
                                           std::span<const TessellationGroupOffsets> group_offsets,
                                           bool neo_mode,
                                           std::span<const ImageInfo> attachments = {});

    [[nodiscard]] Result RunLocalHull(const AmdGpu::Regs& regs,
                                      const TessellationWorkLayout& layout,
                                      std::optional<std::span<const u32>> decoded_indices,
                                      std::span<const TessellationGroupOffsets> group_offsets,
                                      bool neo_mode, std::span<const ImageInfo> attachments = {});

    [[nodiscard]] Result RunLocalHull(const AmdGpu::Regs& regs,
                                      const TessellationKernelConfig& config,
                                      std::span<const GeometryExportInput> local_inputs,
                                      std::span<const GeometryExportInput> hull_inputs,
                                      bool neo_mode, std::span<const ImageInfo> attachments = {});

private:
    enum class RingKind { Factor, Offchip, Export };
    struct StageResult {
        std::optional<FactorRing> factors;
        std::optional<Ring> offchip;
        std::optional<Ring> exports;
        std::optional<RasterChunk> raster;
    };
    [[nodiscard]] Result DispatchLocalHull(const AmdGpu::Regs& regs,
                                           const TessellationWorkLayout& layout,
                                           std::span<const GeometryExportInput> local_inputs,
                                           std::span<const GeometryExportInput> hull_inputs,
                                           bool neo_mode, std::span<const ImageInfo> attachments);
    [[nodiscard]] Result DispatchGuestLocalHull(
        const AmdGpu::Regs& regs, const TessellationWorkLayout& layout, bool indexed,
        u32 first_index, std::span<const TessellationGroupOffsets> group_offsets, bool neo_mode,
        std::span<const ImageInfo> attachments);
    [[nodiscard]] std::vector<BufferRange> PrepareRingSnapshot(Ring& ring);
    [[nodiscard]] std::vector<u32> ReadGuestIndices(const AmdGpu::Regs& regs, u32 first_index,
                                                    bool neo_mode);
    [[nodiscard]] Shader::TessellationDataConstantBuffer ReadConstants(
        const AmdGpu::Regs& regs, const TessellationKernelConfig& config);
    [[nodiscard]] TessellationWorkLayout EvaluationWork(const AmdGpu::Regs& regs,
                                                        const Result& hull, bool raster = false);
    [[nodiscard]] std::vector<u32> EvaluationOffchipOffsets(const AmdGpu::Regs& regs,
                                                            const Result& hull,
                                                            const TessellationWorkLayout& work,
                                                            bool neo_mode) const;
    [[nodiscard]] std::optional<u32> ValidateEvaluationOffchip(
        const ShaderCache::Variant& shader, const TessellationWorkLayout& work,
        std::span<const GeometryExportInput> inputs, const Ring& ring, bool raster) const;
    void ValidateEvaluationDomains(const AmdGpu::Regs& regs, const Result& hull,
                                   const TessellationDomains::Result& domains,
                                   const TessellationWorkLayout& work) const;
    [[nodiscard]] Ring ValidateRing(const AmdGpu::Regs& regs, const ShaderCache::Variant& shader,
                                    const TessellationWorkLayout& work,
                                    std::span<const GeometryExportInput> inputs, RingKind kind,
                                    bool neo_mode);
    StageResult RunStage(const AmdGpu::Regs& regs, Shader::HwStage stage,
                         const TessellationWorkLayout& work,
                         const Shader::TessellationDataConstantBuffer& constants,
                         std::span<const GeometryExportInput> inputs,
                         const std::shared_ptr<StagingBufferPool::Allocation>& lds, bool neo_mode,
                         std::span<const ImageInfo> attachments,
                         const std::shared_ptr<StagingBufferPool::Allocation>& gpu_registers = {},
                         const std::shared_ptr<StagingBufferPool::Allocation>& gpu_arguments = {},
                         const Ring* evaluation_offchip = nullptr);

    const Instance& instance;
    Scheduler& scheduler;
    BufferCache& buffer_cache;
    ShaderCache& shaders;
    GuestResources& resources;
    Commands commands;
    StagingBufferPool staging;
    TessellationFactors factor_decoder;
    TessellationDomains domain_generator;
    TessellationEvaluation evaluation_inputs;
    TessellationExports evaluation_exports;
};

} // namespace Metal
