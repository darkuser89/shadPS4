// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>

#include <array>
#include <deque>
#include <map>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_indirect_work.h"
#include "video_core/renderer_metal/metal_pipeline.h"

namespace Metal {

class GuestPageTableSnapshot;

class IndirectDraws {
public:
    struct Batch {
        id<MTLIndirectCommandBuffer> commands;
        BufferRange range;
    };

    class RasterCount {
    private:
        friend class IndirectDraws;
        RasterCount(const IndirectDraws& owner_,
                    std::shared_ptr<StagingBufferPool::Allocation> storage_)
            : owner{&owner_}, storage{std::move(storage_)} {}
        const IndirectDraws* owner;
        std::shared_ptr<StagingBufferPool::Allocation> storage;
    };

    class RasterWindow {
    public:
        [[nodiscard]] const RasterIndirectParameters& GetParameters() const noexcept {
            return parameters;
        }
        [[nodiscard]] BufferRange Records() const {
            return records ? BufferRange{records->GetBuffer(), records->GetOffset(),
                                         records->SizeBytes()}
                           : BufferRange{};
        }
        void Retain(ResourceBindings& bindings) const {
            if (records)
                bindings.RetainUploadAllocation(records);
            if (source_complete)
                bindings.RetainUploadAllocation(source_complete);
        }

    private:
        friend class IndirectDraws;
        RasterWindow(const IndirectDraws& owner_, const RasterIndirectParameters& parameters_,
                     std::shared_ptr<StagingBufferPool::Allocation> records_,
                     std::shared_ptr<StagingBufferPool::Allocation> source_complete_)
            : owner{&owner_}, parameters{parameters_}, records{std::move(records_)},
              source_complete{std::move(source_complete_)} {}
        const IndirectDraws* owner;
        RasterIndirectParameters parameters;
        std::shared_ptr<StagingBufferPool::Allocation> records;
        std::shared_ptr<StagingBufferPool::Allocation> source_complete;
    };

    class RasterController {
    public:
        RasterController(RasterController&&) = default;
        RasterController& operator=(RasterController&&) = default;

    private:
        friend class IndirectDraws;
        RasterController(const IndirectDraws& owner_, const RasterWindow& window_,
                         const RasterCaptureParameters& parameters_,
                         std::shared_ptr<StagingBufferPool::Allocation> cursor_)
            : owner{&owner_}, window{window_}, parameters{parameters_}, cursor{std::move(cursor_)} {
        }
        const IndirectDraws* owner;
        RasterWindow window;
        RasterCaptureParameters parameters;
        std::shared_ptr<StagingBufferPool::Allocation> cursor;
    };

    class RasterCapture {
    public:
        [[nodiscard]] const RasterCaptureParameters& GetParameters() const noexcept {
            return parameters;
        }
        [[nodiscard]] BufferRange GetRange() const {
            return {batch->GetBuffer(), batch->GetOffset(), batch->SizeBytes()};
        }
        [[nodiscard]] BufferRange DispatchArguments() const {
            return {batch->GetBuffer(), batch->GetOffset() + offsetof(RasterCaptureBatch, dispatch),
                    3 * sizeof(u32)};
        }
        void Retain(ResourceBindings& bindings) const {
            bindings.RetainUploadAllocation(batch);
        }

    private:
        friend class IndirectDraws;
        RasterCapture(IndirectDraws& owner_, const RasterCaptureParameters& parameters_,
                      std::shared_ptr<StagingBufferPool::Allocation> batch_,
                      std::shared_ptr<StagingBufferPool::Allocation> sequence_)
            : owner{&owner_}, parameters{parameters_}, batch{std::move(batch_)},
              sequence{std::move(sequence_)} {}
        IndirectDraws* owner;
        RasterCaptureParameters parameters;
        std::shared_ptr<StagingBufferPool::Allocation> batch;
        std::shared_ptr<StagingBufferPool::Allocation> sequence;

        friend class GeometryAssembly;
    };

    class RasterCaptureInputs {
    public:
        [[nodiscard]] const RasterCapture& GetCapture() const noexcept {
            return capture;
        }
        [[nodiscard]] const RasterCaptureInputLayout& GetLayout() const noexcept {
            return layout;
        }
        [[nodiscard]] BufferRange Registers() const {
            return {registers->GetBuffer(), registers->GetOffset(), registers->SizeBytes()};
        }
        [[nodiscard]] BufferRange RestartRecords() const {
            return restart ? BufferRange{restart->GetBuffer(), restart->GetOffset(),
                                         restart->SizeBytes()}
                           : BufferRange{};
        }
        [[nodiscard]] BufferRange RestartParameters() const {
            return restart_work ? BufferRange{restart_work->GetBuffer(), restart_work->GetOffset(),
                                              sizeof(RasterRestartParameters)}
                                : BufferRange{};
        }
        [[nodiscard]] BufferRange RestartDispatchArguments(u32 stage) const {
            if (!restart_work || stage >= 4) {
                throw std::runtime_error("Invalid native raster restart dispatch stage");
            }
            return {restart_work->GetBuffer(),
                    restart_work->GetOffset() +
                        offsetof(RasterRestartDispatchParameters, dispatch) +
                        stage * 3 * sizeof(u32),
                    3 * sizeof(u32)};
        }
        void Retain(ResourceBindings& bindings) const {
            capture.Retain(bindings);
            bindings.RetainUploadAllocation(registers);
            if (restart)
                bindings.RetainUploadAllocation(restart);
            if (restart_work)
                bindings.RetainUploadAllocation(restart_work);
        }

    private:
        friend class IndirectDraws;
        RasterCaptureInputs(const RasterCapture& capture_, const RasterCaptureInputLayout& layout_,
                            std::shared_ptr<StagingBufferPool::Allocation> registers_,
                            std::shared_ptr<StagingBufferPool::Allocation> restart_,
                            std::shared_ptr<StagingBufferPool::Allocation> restart_work_)
            : capture{capture_}, layout{layout_}, registers{std::move(registers_)},
              restart{std::move(restart_)}, restart_work{std::move(restart_work_)} {}
        RasterCapture capture;
        RasterCaptureInputLayout layout;
        std::shared_ptr<StagingBufferPool::Allocation> registers;
        std::shared_ptr<StagingBufferPool::Allocation> restart;
        std::shared_ptr<StagingBufferPool::Allocation> restart_work;

        friend class GeometryAssembly;
    };

    class RasterCaptureState {
    public:
        [[nodiscard]] BufferRange GetRange() const {
            return {storage->GetBuffer(), storage->GetOffset(), storage->SizeBytes()};
        }
        void Retain(ResourceBindings& bindings) const {
            bindings.RetainUploadAllocation(storage);
        }

    private:
        friend class IndirectDraws;
        friend class GeometryAssembly;
        RasterCaptureState(const IndirectDraws& owner_, const RasterCaptureInputs& inputs,
                           u32 vertex_words_,
                           std::shared_ptr<StagingBufferPool::Allocation> storage_)
            : owner{&owner_}, context{inputs.GetLayout().parameters}, vertex_words{vertex_words_},
              sequence{inputs.capture.sequence}, storage{std::move(storage_)} {}
        const IndirectDraws* owner;
        RasterCaptureInputParameters context;
        u32 vertex_words;
        std::shared_ptr<StagingBufferPool::Allocation> sequence;
        std::shared_ptr<StagingBufferPool::Allocation> storage;
    };

    class RasterCapturePack {
    public:
        [[nodiscard]] BufferRange GetRange() const {
            return {storage->GetBuffer(), storage->GetOffset(), storage->SizeBytes()};
        }
        [[nodiscard]] BufferRange DispatchArguments() const {
            return {storage->GetBuffer(),
                    storage->GetOffset() + offsetof(RasterCapturePackWork, dispatch),
                    3 * sizeof(u32)};
        }
        void Retain(ResourceBindings& bindings) const {
            bindings.RetainUploadAllocation(storage);
        }

    private:
        friend class IndirectDraws;
        friend class GeometryAssembly;
        RasterCapturePack(std::shared_ptr<StagingBufferPool::Allocation> storage_,
                          bool uses_previous_)
            : storage{std::move(storage_)}, uses_previous{uses_previous_} {}
        std::shared_ptr<StagingBufferPool::Allocation> storage;
        bool uses_previous;
    };

    IndirectDraws(const Instance& instance, Scheduler& scheduler);
    [[nodiscard]] Batch Encode(MTLPrimitiveType primitive, const BufferRange& arguments,
                               const BufferRange& count, u32 max_count, u32 stride,
                               const BufferRange& indices, MTLIndexType index_type, bool indexed);
    [[nodiscard]] Batch EncodePaged(MTLPrimitiveType primitive, const GuestPageTableSnapshot& pages,
                                    VAddr argument_address, VAddr count_address, u32 max_count,
                                    u32 stride, const BufferRange& indices, MTLIndexType index_type,
                                    bool indexed);
    void CheckErrors();
    [[nodiscard]] BufferRange PrepareErrorBuffer();
    void TrackErrorUse(const BufferRange& range);
    [[nodiscard]] RasterCount SnapshotRasterCount(const BufferRange& count);
    [[nodiscard]] RasterWindow PrepareRasterWindow(const BufferRange& window_arguments,
                                                   const BufferRange& count, u32 max_count,
                                                   u32 first_draw, u32 record_count, u32 stride,
                                                   u32 index_width, u32 index_capacity,
                                                   u32 capture_capacity);
    [[nodiscard]] RasterWindow PrepareRasterWindowPaged(
        const GuestPageTableSnapshot& pages, VAddr window_argument_address, VAddr count_address,
        u32 max_count, u32 first_draw, u32 record_count, u32 stride, u32 index_width,
        u32 index_capacity, u32 capture_capacity, const RasterCount* count_snapshot = nullptr);
    [[nodiscard]] RasterController CreateRasterController(const RasterWindow& window,
                                                          u32 primitive_vertices, u32 step_rate_0,
                                                          u32 step_rate_1, u32 index_offset,
                                                          AmdGpu::IndexSwapMode swap_mode);
    [[nodiscard]] RasterCapture NextRasterCapture(const RasterController& controller);
    [[nodiscard]] bool RunRasterCaptures(const RasterController& controller,
                                         u32 captures_per_submission,
                                         const std::function<void(const RasterCapture&)>& consume);
    [[nodiscard]] RasterCaptureInputs PrepareRasterCaptureInputs(
        const RasterCapture& capture, const GuestPageTableSnapshot* pages, VAddr index_address,
        const std::array<float, 4>& viewport, const GeometryExportWave& system_registers = {},
        std::optional<u32> restart_index = std::nullopt,
        const RasterIndirectRegisters& indirect_registers = {});

private:
    friend class GraphicsRenderer;
    friend class GeometryAssembly;

    [[nodiscard]] u32 WaitForRasterCaptures(const RasterController& controller);
    struct Slot {
        id<MTLIndirectCommandBuffer> commands;
        u32 capacity{};
        bool leased{};
    };
    struct ErrorRecord {
        StagingBufferPool::Allocation allocation;
        u64 tick;
    };
    [[nodiscard]] Batch EncodeCommands(MTLPrimitiveType primitive, const BufferRange& arguments,
                                       const BufferRange& count, u32 max_count, u32 stride,
                                       const BufferRange& indices, MTLIndexType index_type,
                                       bool indexed, const GuestPageTableSnapshot* pages,
                                       VAddr argument_address, VAddr count_address);
    [[nodiscard]] id<MTLIndirectCommandBuffer> Obtain(u32 max_count, bool indexed);
    [[nodiscard]] id<MTLComputePipelineState> Pipeline(bool indexed, MTLIndexType index_type);
    [[nodiscard]] id<MTLComputePipelineState> RasterPipeline(bool capture = false);
    [[nodiscard]] id<MTLComputePipelineState> RasterInputPipeline(bool restart = false);
    [[nodiscard]] id<MTLComputePipelineState> RasterSnapshotPipeline();
    [[nodiscard]] id<MTLComputePipelineState> RasterPackPipeline();
    [[nodiscard]] RasterCapturePack PrepareRasterPack(const RasterCaptureInputs& inputs,
                                                      const RasterCaptureState& current,
                                                      const RasterCaptureState* previous,
                                                      RasterCapturePackConfig config,
                                                      bool same_layout);
    [[nodiscard]] RasterCaptureState SnapshotRasterCapture(
        const RasterCaptureInputs& inputs, std::shared_ptr<StagingBufferPool::Allocation> vertices,
        u32 vertex_words, const RasterCaptureState* previous, bool same_layout);
    [[nodiscard]] RasterWindow PrepareRasterWindowData(const BufferRange& window_arguments,
                                                       const BufferRange& count,
                                                       const RasterIndirectWindowLayout& layout,
                                                       const GuestPageTableSnapshot* pages,
                                                       VAddr window_argument_address,
                                                       VAddr count_address,
                                                       const RasterCount* count_snapshot = nullptr);
    void ReleaseAfterDraw(const Batch& batch);

    const Instance& instance;
    Scheduler& scheduler;
    Commands commands;
    PipelineCompiler compiler;
    StagingBufferPool staging;
    id<MTLLibrary> library;
    std::map<u32, id<MTLComputePipelineState>> pipelines;
    std::array<Slot, 2> slots{};
    std::deque<ErrorRecord> errors;
};

} // namespace Metal
