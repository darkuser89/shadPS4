// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <compare>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_geometry_layout.h"
#include "video_core/renderer_metal/metal_geometry_mesh.h"
#include "video_core/renderer_metal/metal_geometry_work.h"
#include "video_core/renderer_metal/metal_graphics_pipeline.h"
#include "video_core/renderer_metal/metal_guest_resources.h"
#include "video_core/renderer_metal/metal_indirect.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_tessellation_geometry.h"

namespace Metal {

class GeometryAssembly {
public:
    class Emissions {
    public:
        [[nodiscard]] BufferRange GetRange() const {
            return storage ? BufferRange{storage->GetBuffer(), storage->GetOffset(),
                                         storage->SizeBytes()}
                           : BufferRange{};
        }

        [[nodiscard]] const GeometryAssemblyParameters& GetParameters() const noexcept {
            return parameters;
        }

        void Retain(ResourceBindings& bindings) const {
            if (storage) {
                bindings.RetainUploadAllocation(storage);
            }
        }

    private:
        friend class GeometryAssembly;

        Emissions(std::shared_ptr<StagingBufferPool::Allocation> storage_,
                  GeometryAssemblyParameters parameters_)
            : storage{std::move(storage_)}, parameters{parameters_} {}

        std::shared_ptr<StagingBufferPool::Allocation> storage;
        GeometryAssemblyParameters parameters;
    };

    struct Result {
        Emissions emissions;
        std::shared_ptr<StagingBufferPool::Allocation> primitives;
        std::shared_ptr<StagingBufferPool::Allocation> indirect;
        u64 retained_bytes{};
        u32 primitive_slots{};
        bool primitive_ids{};
    };

    class RasterVertices {
    public:
        [[nodiscard]] BufferRange GetRange() const {
            return storage ? BufferRange{storage->GetBuffer(), storage->GetOffset(),
                                         storage->SizeBytes()}
                           : BufferRange{};
        }

        [[nodiscard]] u32 WorkItems() const noexcept {
            return work_items;
        }

        [[nodiscard]] const TessellationRasterBindings& GetLayout() const noexcept {
            return layout;
        }

        void Retain(ResourceBindings& bindings) const {
            bindings.RetainUploadAllocation(storage);
        }

    private:
        friend class GeometryAssembly;

        RasterVertices(const GeometryAssembly& owner_,
                       std::shared_ptr<StagingBufferPool::Allocation> storage_, u32 work_items_,
                       TessellationRasterBindings layout_)
            : owner{&owner_}, storage{std::move(storage_)}, work_items{work_items_},
              layout{std::move(layout_)} {}

        const GeometryAssembly* owner;
        std::shared_ptr<StagingBufferPool::Allocation> storage;
        u32 work_items;
        TessellationRasterBindings layout;
    };

    struct RasterDrawContinuation {
        RasterVertices vertices;
        RasterDrawParameters parameters;
    };

    struct DrawInputs {
        std::shared_ptr<StagingBufferPool::Allocation> exports;
        std::shared_ptr<StagingBufferPool::Allocation> work;
        GeometryDrawParameters parameters;
        std::shared_ptr<StagingBufferPool::Allocation> export_waves;
        std::shared_ptr<StagingBufferPool::Allocation> geometry_waves;
    };

    class RasterRestartWork {
    public:
        [[nodiscard]] const RasterRestartParameters& GetParameters() const noexcept {
            return parameters;
        }
        [[nodiscard]] BufferRange Records() const {
            return {records->GetBuffer(), records->GetOffset(), records->SizeBytes()};
        }
        [[nodiscard]] BufferRange Blocks() const {
            return {blocks->GetBuffer(), blocks->GetOffset(), blocks->SizeBytes()};
        }
        [[nodiscard]] BufferRange Superblocks() const {
            return {superblocks->GetBuffer(), superblocks->GetOffset(), superblocks->SizeBytes()};
        }
        void Retain(ResourceBindings& bindings) const {
            bindings.RetainUploadAllocation(records);
            bindings.RetainUploadAllocation(blocks);
            bindings.RetainUploadAllocation(superblocks);
        }

    private:
        friend class GeometryAssembly;
        RasterRestartWork(const GeometryAssembly& owner_, const RasterDrawParameters& draw_,
                          const RasterRestartParameters& parameters_,
                          std::shared_ptr<StagingBufferPool::Allocation> records_,
                          std::shared_ptr<StagingBufferPool::Allocation> blocks_,
                          std::shared_ptr<StagingBufferPool::Allocation> superblocks_)
            : owner{&owner_}, draw{draw_}, parameters{parameters_}, records{std::move(records_)},
              blocks{std::move(blocks_)}, superblocks{std::move(superblocks_)} {}

        const GeometryAssembly* owner;
        RasterDrawParameters draw;
        RasterRestartParameters parameters;
        std::shared_ptr<StagingBufferPool::Allocation> records;
        std::shared_ptr<StagingBufferPool::Allocation> blocks;
        std::shared_ptr<StagingBufferPool::Allocation> superblocks;
    };

    class IndirectRasterRestartWork {
    public:
        [[nodiscard]] const RasterRestartParameters& GetCapacity() const noexcept {
            return inputs.GetLayout().restart_capacity;
        }
        [[nodiscard]] BufferRange Parameters() const {
            return inputs.RestartParameters();
        }
        [[nodiscard]] BufferRange Records() const {
            return inputs.RestartRecords();
        }
        [[nodiscard]] BufferRange Blocks() const {
            return {blocks->GetBuffer(), blocks->GetOffset(), blocks->SizeBytes()};
        }
        [[nodiscard]] BufferRange Superblocks() const {
            return {superblocks->GetBuffer(), superblocks->GetOffset(), superblocks->SizeBytes()};
        }
        void Retain(ResourceBindings& bindings) const {
            inputs.Retain(bindings);
            bindings.RetainUploadAllocation(blocks);
            bindings.RetainUploadAllocation(superblocks);
        }

    private:
        friend class GeometryAssembly;
        IndirectRasterRestartWork(const GeometryAssembly& owner_,
                                  const IndirectDraws::RasterCaptureInputs& inputs_,
                                  std::shared_ptr<StagingBufferPool::Allocation> blocks_,
                                  std::shared_ptr<StagingBufferPool::Allocation> superblocks_)
            : owner{&owner_}, inputs{inputs_}, blocks{std::move(blocks_)},
              superblocks{std::move(superblocks_)} {}
        const GeometryAssembly* owner;
        IndirectDraws::RasterCaptureInputs inputs;
        std::shared_ptr<StagingBufferPool::Allocation> blocks;
        std::shared_ptr<StagingBufferPool::Allocation> superblocks;
    };

    class IndirectRasterState {
    public:
        [[nodiscard]] BufferRange GetRange() const {
            return state.GetRange();
        }
        void Retain(ResourceBindings& bindings) const {
            state.Retain(bindings);
            if (counter)
                bindings.RetainUploadAllocation(counter);
        }

    private:
        friend class GeometryAssembly;
        IndirectRasterState(const GeometryAssembly& owner_,
                            IndirectDraws::RasterCaptureState state_,
                            const TessellationRasterBindings& layout_,
                            const ShaderCache::Variant* shader_,
                            const std::array<u32, 16>& user_data_)
            : owner{&owner_}, state{std::move(state_)}, layout{layout_}, shader{shader_},
              user_data{user_data_} {}
        const GeometryAssembly* owner;
        IndirectDraws::RasterCaptureState state;
        TessellationRasterBindings layout;
        const ShaderCache::Variant* shader;
        std::array<u32, 16> user_data;
        std::shared_ptr<StagingBufferPool::Allocation> counter;
        std::optional<RasterPrimitiveCounterSettings> counter_settings;
    };

    class RasterRestartState {
    public:
        [[nodiscard]] BufferRange GetRange() const {
            return {storage->GetBuffer(), storage->GetOffset(), storage->SizeBytes()};
        }
        void Retain(ResourceBindings& bindings) const {
            bindings.RetainUploadAllocation(storage);
            if (counter)
                bindings.RetainUploadAllocation(counter);
        }

    private:
        friend class GeometryAssembly;
        RasterRestartState(const GeometryAssembly& owner_, const RasterDrawParameters& draw_,
                           const TessellationRasterBindings& layout_,
                           std::shared_ptr<StagingBufferPool::Allocation> storage_)
            : owner{&owner_}, draw{draw_}, layout{layout_}, storage{std::move(storage_)} {}

        const GeometryAssembly* owner;
        RasterDrawParameters draw;
        TessellationRasterBindings layout;
        std::shared_ptr<StagingBufferPool::Allocation> storage;
        std::shared_ptr<StagingBufferPool::Allocation> counter;
        std::optional<RasterPrimitiveCounterConfig> counter_config;
        AmdGpu::PrimitiveType counter_topology{};
        u32 counter_restart_index{};
    };

    struct RasterDrawInputs {
        std::shared_ptr<StagingBufferPool::Allocation> registers;
        RasterDrawParameters parameters;
        GeometryExportWave system_registers;
        bool dynamic_inputs{};
        std::optional<RasterRestartWork> restart;
    };

    struct MeshShader {
        std::shared_ptr<const GeometryMeshSource> mesh;
        u32 stream;
        u32 topology;
        u32 vertex_words;
        std::shared_ptr<const CompiledShader> shader;
        const ShaderCache::Variant* fragment;
        std::vector<GeometryOutputAttribute> attributes;
        std::array<u32, 2> viewport_extent{};
        bool last_provoking_vertex{};
    };

    struct IndirectRasterPlan {
        MeshShader mesh;
        u32 capacity;
    };

    class PreparedKernel {
    public:
        PreparedKernel(PreparedKernel&&) = default;
        PreparedKernel& operator=(PreparedKernel&&) = default;

        [[nodiscard]] GuestResources::Stage& GetResources() const;

    private:
        friend class GeometryAssembly;

        struct RingReplay {
            GeometryBufferRange origin;
            std::shared_ptr<StagingBufferPool::Allocation> source;
            std::vector<GuestResources::Stage*> companions;
        };

        struct CopyDispatch {
            GeometryCopyConfig config;
            Emissions source;
            std::shared_ptr<StagingBufferPool::Allocation> inputs;
            BufferRange errors;
        };

        PreparedKernel(const GeometryAssembly& owner_, GuestResources& resources_,
                       std::unique_ptr<GuestResources::Stage> stage_,
                       id<MTLComputePipelineState> pipeline_, u32 work_items_)
            : owner{&owner_}, resources{&resources_}, stage{std::move(stage_)}, pipeline{pipeline_},
              work_items{work_items_} {}

        const GeometryAssembly* owner;
        GuestResources* resources;
        std::unique_ptr<GuestResources::Stage> stage;
        id<MTLComputePipelineState> pipeline;
        u32 work_items;
        std::optional<std::pair<Emissions, Emissions>> header_copy;
        std::vector<std::shared_ptr<StagingBufferPool::Allocation>> retained_inputs;
        std::shared_ptr<StagingBufferPool::Allocation> arguments;
        u64 argument_offset{};
        std::optional<Emissions> clear_emissions;
        std::optional<BufferRange> clear_raster;
        std::optional<RingReplay> ring_replay;
        std::optional<CopyDispatch> copy_dispatch;
        u32 copy_stream{};
        bool submitted{};
    };

    struct PreparedEmissions {
        Emissions emissions;
        PreparedKernel kernel;
    };

    struct PreparedRasterVertices {
        RasterVertices vertices;
        PreparedKernel kernel;
        std::optional<IndirectRasterRestartWork> indirect_restart;
        std::optional<IndirectDraws::RasterCaptureInputs> indirect_inputs;
        const ShaderCache::Variant* indirect_shader{};
        std::array<u32, 16> indirect_user_data{};
    };

    struct PreparedRasterDraw {
        PreparedRasterVertices vertices;
        MeshShader mesh;
        std::optional<RasterRestartWork> restart;
    };

    struct PackedIndirectRaster {
        Result result;
        IndirectRasterState state;
    };

    struct PreparedDraw {
        PreparedKernel exports;
        PreparedEmissions geometry;
        PreparedEmissions copy;
        MeshShader mesh;
    };

    struct PreparedTessellationDraw {
        PreparedEmissions geometry;
        PreparedEmissions copy;
        MeshShader mesh;
    };

    GeometryAssembly(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] Emissions Allocate(u32 invocations, u32 max_vertices, u32 vertex_words,
                                     u32 mesh_vertices, const std::array<u32, 4>& topology,
                                     u32 active_streams);
    [[nodiscard]] Result Assemble(const Emissions& emissions, const BufferRange& errors);
    void PackViewportBatches(const Result& source, const MeshShader& mesh,
                             const BufferRange& errors, u32 viewport_count,
                             const std::function<void(Result&&)>& consume);
    void AssembleTessellationRaster(const TessellationExports::Batch& batch, const MeshShader& mesh,
                                    const BufferRange& errors,
                                    const std::function<void(Result&&)>& consume,
                                    u32 viewport_count = 0);
    std::optional<RasterRestartState> AssembleRasterQuads(
        const RasterVertices& vertices, const RasterDrawParameters& draw, const MeshShader& mesh,
        const BufferRange& errors, const std::function<void(Result&&)>& consume, u32 viewport_count,
        AmdGpu::PrimitiveType topology = AmdGpu::PrimitiveType::QuadList,
        const RasterDrawContinuation* previous = nullptr,
        const RasterDrawContinuation* origin = nullptr, const RasterRestartWork* restart = nullptr,
        const RasterRestartState* restart_previous = nullptr,
        const RasterPrimitiveIdParameters* primitive_ids = nullptr,
        const RasterPrimitiveCounterConfig* primitive_counter = nullptr,
        const RasterRestartState* counter_previous = nullptr);
    [[nodiscard]] DrawInputs PrepareDrawInputs(const GeometryDrawParameters& parameters,
                                               const BufferRange& indices,
                                               const BufferRange& export_waves,
                                               const BufferRange& geometry_waves);
    [[nodiscard]] DrawInputs PrepareDrawInputs(const GeometryDrawParameters& parameters,
                                               const BufferRange& indices,
                                               const GeometryRingLayout& rings,
                                               const GeometryExportWave& export_template,
                                               u32 first_geometry_wave_id);
    [[nodiscard]] RasterDrawInputs PrepareRasterInputs(
        const RasterDrawParameters& parameters, const BufferRange& indices,
        const std::array<float, 4>& viewport, const GeometryExportWave& system_registers = {},
        bool dynamic_inputs = false, std::optional<u32> restart_index = std::nullopt,
        u32 index_source_offset = 0);
    [[nodiscard]] std::shared_ptr<StagingBufferPool::Allocation> PrepareKernelInputs(
        const BufferRange& work, u32 invocations, u32 export_vertices, u32 export_vertex_words,
        const BufferRange& errors);
    [[nodiscard]] std::shared_ptr<StagingBufferPool::Allocation> PrepareCopyInputs(
        const GeometryCopyParameters& parameters, const BufferRange& geometry_waves,
        const BufferRange& copy_waves, const BufferRange& errors, bool compact_workgroups = false);
    [[nodiscard]] PreparedKernel PrepareExport(
        ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
        std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
        u32 initial_active_work_items = 0, std::span<const GuestBufferExtent> guest_extents = {},
        std::span<const GeometryAddressInput> geometry_inputs = {});
    [[nodiscard]] PreparedRasterVertices PrepareRasterVertices(
        ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
        std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
        u32 initial_active_work_items = 0, std::span<const GuestBufferExtent> guest_extents = {},
        std::span<const GeometryAddressInput> geometry_inputs = {});
    [[nodiscard]] PreparedRasterVertices PrepareRasterVertices(
        ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
        const RasterDrawInputs& inputs, bool neo_mode,
        std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] PreparedEmissions PrepareGenerate(
        ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
        std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
        u32 initial_active_work_items = 0, std::span<const GuestBufferExtent> guest_extents = {},
        std::span<const GeometryAddressInput> geometry_inputs = {},
        std::shared_ptr<StagingBufferPool::Allocation> arguments = {});
    [[nodiscard]] PreparedEmissions PrepareCopy(
        ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
        const Emissions& emissions, std::shared_ptr<StagingBufferPool::Allocation> inputs,
        bool neo_mode, u32 initial_active_work_items = 0,
        std::span<const GuestBufferExtent> guest_extents = {},
        std::span<const GeometryAddressInput> geometry_inputs = {}, const BufferRange& errors = {});
    [[nodiscard]] PreparedEmissions PrepareTessellationGenerate(
        ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
        const TessellationGeometry::Inputs& inputs, const GeometryAssemblyParameters& emissions,
        bool neo_mode, std::span<GuestResources::Stage* const> companions = {},
        std::span<const ImageInfo> attachments = {});
    [[nodiscard]] PreparedTessellationDraw PrepareTessellationDraw(
        ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
        const TessellationGeometry::Inputs& inputs, const BufferRange& errors, bool neo_mode,
        const ShaderCache::Variant* fragment, bool last_provoking_vertex, bool flip_vertex_y,
        std::span<GuestResources::Stage* const> companions = {},
        std::span<const ImageInfo> attachments = {});
    [[nodiscard]] u32 MaxTessellationPrimitives(ShaderCache& shader_cache, const AmdGpu::Regs& regs,
                                                const ShaderCache::Variant* fragment,
                                                bool last_provoking_vertex, bool flip_vertex_y);
    [[nodiscard]] PreparedDraw PrepareGuestDraw(ShaderCache& shader_cache,
                                                GuestResources& resources, const AmdGpu::Regs& regs,
                                                const GeometryDrawParameters& parameters,
                                                const BufferRange& indices,
                                                const BufferRange& errors, bool neo_mode,
                                                const ShaderCache::Variant* fragment,
                                                bool last_provoking_vertex, bool flip_vertex_y);
    [[nodiscard]] PreparedRasterDraw PrepareGuestRasterDraw(
        ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
        const RasterDrawParameters& parameters, const BufferRange& indices, bool neo_mode,
        const ShaderCache::Variant* fragment, bool last_provoking_vertex, bool flip_vertex_y,
        u32 index_source_offset = 0);
    [[nodiscard]] u32 MaxRasterBatchVertices(ShaderCache& shader_cache, const AmdGpu::Regs& regs,
                                             bool indexed, bool neo_mode,
                                             const RasterIndirectRegisters* indirect = nullptr);
    [[nodiscard]] PreparedRasterVertices PrepareIndirectRasterVertices(
        ShaderCache& shader_cache, GuestResources& resources, const AmdGpu::Regs& regs,
        const IndirectDraws::RasterCaptureInputs& inputs, bool neo_mode,
        std::span<const GuestBufferExtent> guest_extents = {}, bool primitive_ids = false);
    [[nodiscard]] IndirectRasterState SnapshotIndirectRasterVertices(
        const PreparedRasterVertices& prepared, const IndirectRasterState* previous = nullptr);
    [[nodiscard]] IndirectRasterPlan PrepareIndirectRasterPlan(
        ShaderCache& shader_cache, const AmdGpu::Regs& regs,
        const RasterIndirectRegisters& indirect, u32 index_width,
        const ShaderCache::Variant* fragment, bool last_provoking_vertex, bool flip_vertex_y,
        bool neo_mode, bool primitive_ids = false);
    [[nodiscard]] PackedIndirectRaster AssembleIndirectRasterQuads(
        const PreparedRasterVertices& prepared, const MeshShader& mesh,
        const IndirectRasterState* previous = nullptr, u32 viewport_count = 0,
        const RasterPrimitiveCounterSettings* primitive_counter = nullptr);
    void Execute(std::span<PreparedKernel* const> kernels,
                 std::span<const ImageInfo> attachments = {});
    void Export(ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
                std::shared_ptr<StagingBufferPool::Allocation> inputs, bool neo_mode,
                u32 initial_active_work_items = 0,
                std::span<const GuestBufferExtent> guest_extents = {});
    void Export(ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
                const DrawInputs& inputs, bool neo_mode,
                std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] Emissions Generate(ShaderCache& shader_cache, GuestResources& resources,
                                     ShaderCache::Variant& shader,
                                     std::shared_ptr<StagingBufferPool::Allocation> inputs,
                                     bool neo_mode, u32 initial_active_work_items = 0,
                                     std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] Emissions Generate(ShaderCache& shader_cache, GuestResources& resources,
                                     ShaderCache::Variant& shader, const BufferRange& work,
                                     u32 export_vertices, u32 export_vertex_words,
                                     const BufferRange& errors, bool neo_mode,
                                     u32 initial_active_work_items = 0,
                                     std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] Emissions Generate(ShaderCache& shader_cache, GuestResources& resources,
                                     ShaderCache::Variant& shader, const DrawInputs& inputs,
                                     u32 export_vertex_words, const BufferRange& errors,
                                     bool neo_mode,
                                     std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] Emissions Copy(ShaderCache& shader_cache, GuestResources& resources,
                                 ShaderCache::Variant& shader, const Emissions& emissions,
                                 std::shared_ptr<StagingBufferPool::Allocation> inputs,
                                 bool neo_mode, u32 initial_active_work_items = 0,
                                 std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] Emissions Copy(ShaderCache& shader_cache, GuestResources& resources,
                                 ShaderCache::Variant& shader, const Emissions& emissions,
                                 const GeometryCopyParameters& parameters,
                                 const BufferRange& geometry_waves, const BufferRange& copy_waves,
                                 const BufferRange& errors, bool neo_mode,
                                 std::span<const GuestBufferExtent> guest_extents = {});
    [[nodiscard]] MeshShader CompileMesh(u32 stream, u32 topology, u32 vertex_words,
                                         std::span<const GeometryOutputAttribute> attributes,
                                         const ShaderCache::Variant* fragment,
                                         bool last_provoking_vertex, bool flip_vertex_y,
                                         const std::array<u32, 2>& viewport_extent = {},
                                         bool primitive_ids = false);
    void Draw(const Result& result, const MeshShader& shader, const RenderState& state,
              const GraphicsState& graphics, GraphicsPipelineCache& pipelines,
              const ShaderCache::Variant* fragment, const GraphicsPipelineState& pipeline_state,
              ResourceBindings* fragment_bindings,
              std::span<const std::array<float, 4>> viewports = {});

private:
    [[nodiscard]] IndirectRasterRestartWork ScanIndirectRasterRestart(
        const IndirectDraws::RasterCaptureInputs& inputs);
    [[nodiscard]] ShaderCache::Variant& RasterVertexShader(
        ShaderCache& shader_cache, const AmdGpu::Regs& regs,
        const RasterIndirectRegisters* indirect = nullptr, bool primitive_ids = false);
    struct CopyMeshPlan {
        GeometryCopyConfig config;
        GeometryCopyParameters work;
        ShaderCache::Variant* shader;
        MeshShader mesh;
    };

    [[nodiscard]] CopyMeshPlan PrepareCopyMesh(ShaderCache& shader_cache, const AmdGpu::Regs& regs,
                                               const GeometryRingLayout& rings,
                                               const Shader::RuntimeInfo& runtime, u32 invocations,
                                               const ShaderCache::Variant* fragment,
                                               bool last_provoking_vertex, bool flip_vertex_y,
                                               bool compact_workgroups);
    struct MeshInterface {
        u32 stream;
        u32 topology;
        u32 vertex_words;
        bool last_provoking_vertex;
        bool flip_vertex_y;
        std::array<u32, 2> viewport_extent;
        bool primitive_ids;
        std::vector<std::array<u64, 4>> attributes;
        std::vector<std::array<u32, 3>> varyings;

        auto operator<=>(const MeshInterface&) const = default;
    };

    struct CachedMesh {
        std::shared_ptr<const GeometryMeshSource> mesh;
        std::shared_ptr<const CompiledShader> shader;
    };

    [[nodiscard]] id<MTLComputePipelineState> Pipeline();
    [[nodiscard]] id<MTLComputePipelineState> CopyHeadersPipeline();
    [[nodiscard]] id<MTLComputePipelineState> InputsPipeline();
    [[nodiscard]] id<MTLComputePipelineState> WorkPipeline();
    [[nodiscard]] id<MTLComputePipelineState> RasterInputsPipeline();
    [[nodiscard]] id<MTLComputePipelineState> RasterRestartPipeline(u32 stage);
    [[nodiscard]] id<MTLComputePipelineState> RasterQuadsPipeline(u32 stage = 0);
    [[nodiscard]] id<MTLComputePipelineState> RasterPrimitiveCounterPipeline(u32 stage);
    [[nodiscard]] std::shared_ptr<StagingBufferPool::Allocation> AssignRasterPrimitiveIds(
        Result& result, const RasterPrimitiveCounterSettings& config, const BufferRange& errors,
        const std::shared_ptr<StagingBufferPool::Allocation>& previous,
        const std::shared_ptr<StagingBufferPool::Allocation>& scan_work = {});
    [[nodiscard]] std::shared_ptr<StagingBufferPool::Allocation> PrepareIndirectPrimitiveScan(
        const GeometryAssemblyParameters& mesh, u32 capacity,
        const RasterPrimitiveCounterSettings& config,
        const IndirectDraws::RasterCapturePack& control,
        const IndirectDraws::RasterCapture& capture, const BufferRange& errors,
        const std::shared_ptr<StagingBufferPool::Allocation>& previous);
    [[nodiscard]] id<MTLComputePipelineState> CopyInputsPipeline();
    [[nodiscard]] id<MTLComputePipelineState> CopyDispatchPipeline();
    [[nodiscard]] id<MTLComputePipelineState> TessellationRasterPipeline();
    [[nodiscard]] id<MTLComputePipelineState> ViewportPackingPipeline();
    [[nodiscard]] id<MTLComputePipelineState> ViewportPackingDispatchPipeline();
    void BuildCopyDispatch(PreparedKernel& kernel);
    void CopyHeaders(const Emissions& source, const Emissions& destination, u32 stream);
    [[nodiscard]] PreparedKernel PrepareKernel(
        ShaderCache& shader_cache, GuestResources& resources, ShaderCache::Variant& shader,
        std::span<const GuestResources::GeometryBuffer> buffers, u32 work_items, bool neo_mode,
        u32 initial_active_work_items, std::span<const GuestBufferExtent> guest_extents,
        std::span<const GeometryAddressInput> geometry_inputs = {});

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    std::unordered_map<std::string, std::shared_ptr<const CompiledShader>> mesh_shaders;
    std::map<MeshInterface, CachedMesh> mesh_interfaces;
    id<MTLComputePipelineState> pipeline;
    id<MTLComputePipelineState> copy_headers_pipeline;
    id<MTLComputePipelineState> inputs_pipeline;
    id<MTLComputePipelineState> work_pipeline;
    id<MTLComputePipelineState> raster_inputs_pipeline;
    std::array<id<MTLComputePipelineState>, 4> raster_restart_pipelines{};
    std::array<id<MTLComputePipelineState>, 4> raster_quads_pipelines{};
    std::array<id<MTLComputePipelineState>, 4> raster_primitive_counter_pipelines{};
    id<MTLComputePipelineState> copy_inputs_pipeline;
    id<MTLComputePipelineState> copy_dispatch_pipeline;
    id<MTLComputePipelineState> tessellation_raster_pipeline;
    id<MTLComputePipelineState> viewport_packing_pipeline;
    id<MTLComputePipelineState> viewport_packing_dispatch_pipeline;
};

} // namespace Metal
