// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_metal/metal_tessellation_domains.h"

namespace Metal {

struct TessellationPrimitive {
    std::array<u32, 3> vertex_offsets;
    u32 patch;
    u32 instance;
    u32 active;
};
static_assert(sizeof(TessellationPrimitive) == 24);

class TessellationExports {
public:
    static constexpr u32 WavesPerPatch = (TessellationDomains::PointsPerPatch + 63) / 64;
    static constexpr u32 VerticesPerPatch = WavesPerPatch * 64;

    class Batch {
    public:
        Batch(Batch&&) = default;
        Batch& operator=(Batch&&) = default;

        [[nodiscard]] const auto& GetExports() const {
            return exports;
        }
        [[nodiscard]] const auto& GetPrimitives() const {
            return primitives;
        }
        [[nodiscard]] const auto& GetDomains() const {
            return domains;
        }
        [[nodiscard]] u32 GetVertexWords() const {
            return vertex_words;
        }
        [[nodiscard]] u32 GetPrimitiveWidth() const {
            return primitive_width;
        }
        [[nodiscard]] u32 GetPrimitiveSlotsPerPatch() const {
            return primitive_slots;
        }
        [[nodiscard]] bool IsValidated() const {
            return validated && errors && errors->IsReady();
        }
        [[nodiscard]] const auto& GetGuestRing() const {
            return guest_ring;
        }
        [[nodiscard]] bool IsRaster() const {
            return raster;
        }
        [[nodiscard]] const auto& GetRasterAttributes() const {
            return raster_attributes;
        }

    private:
        friend class TessellationExports;

        Batch(const TessellationExports& owner_, const TessellationDomains::Result& domains_,
              const TessellationWorkLayout& work_, u32 vertex_words_, u32 primitive_width_,
              u32 primitive_slots_)
            : owner{&owner_}, domains{domains_}, work{work_}, vertex_words{vertex_words_},
              primitive_width{primitive_width_}, primitive_slots{primitive_slots_},
              captured(domains_.patch_count) {}

        const TessellationExports* owner;
        TessellationDomains::Result domains;
        TessellationWorkLayout work;
        u32 vertex_words;
        u32 primitive_width;
        u32 primitive_slots;
        std::shared_ptr<StagingBufferPool::Allocation> exports;
        std::shared_ptr<StagingBufferPool::Allocation> primitives;
        std::shared_ptr<StagingBufferPool::Allocation> errors;
        std::vector<std::array<u64, 2>> captured;
        std::optional<GeometryBufferRange> guest_ring;
        std::vector<GeometryOutputAttribute> raster_attributes;
        bool raster{};
        bool assembled{};
        bool validated{};
    };

    TessellationExports(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] u32 MaxBatchPatches(u32 vertex_words,
                                      AmdGpu::TessellationTopology topology) const;
    [[nodiscard]] Batch Allocate(const TessellationDomains::Result& domains,
                                 const TessellationWorkLayout& work,
                                 const TessellationExportRingLayout& ring);
    [[nodiscard]] Batch AllocateRaster(const TessellationDomains::Result& domains,
                                       const TessellationWorkLayout& work, u32 vertex_words,
                                       std::span<const GeometryOutputAttribute> attributes);
    void CaptureRaster(Batch& batch, u32 relative_patch, u32 first_point,
                       const std::shared_ptr<StagingBufferPool::Allocation>& snapshot);
    void Capture(Batch& batch, u32 relative_patch, u32 first_point,
                 const std::shared_ptr<StagingBufferPool::Allocation>& snapshot,
                 std::span<const u64> group_offsets, const GeometryBufferRange& guest_ring);
    void Assemble(Batch& batch);
    void Validate(Batch& batch) const;

private:
    void CaptureImpl(Batch& batch, u32 relative_patch, u32 first_point,
                     const std::shared_ptr<StagingBufferPool::Allocation>& snapshot,
                     std::span<const u64> group_offsets,
                     const std::optional<GeometryBufferRange>& guest_ring);
    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLComputePipelineState> pipeline;
};

} // namespace Metal
