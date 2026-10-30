// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "video_core/renderer_metal/metal_common.h"
#include "video_core/renderer_metal/metal_render_state.h"

namespace Metal {

class Instance;
class Commands;
class StagingBufferPool;

class Scheduler {
public:
    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    [[nodiscard]] id<MTL4CommandBuffer> CommandBuffer() const noexcept;
    [[nodiscard]] id<MTL4CommandQueue> GetQueue() const noexcept;
    [[nodiscard]] u64 CurrentTick() const noexcept;
    [[nodiscard]] u64 EncodingGeneration() const noexcept;
    [[nodiscard]] bool IsFree(u64 tick) const;
    [[nodiscard]] bool IsResourceFree(id<MTLAllocation> allocation) const;

    void UseResource(id<MTLAllocation> allocation);
    void RetainObject(id object);
    void RetainLifetime(std::shared_ptr<void> lifetime);
    void TickFrame();
    void ReclaimCompletedResources();
    void WaitForDrawable(id<MTLDrawable> drawable);
    [[nodiscard]] u64 Present(id<MTLDrawable> drawable);
    [[nodiscard]] id<MTL4RenderCommandEncoder> BeginRendering(const RenderState& state);
    [[nodiscard]] id<MTL4ComputeCommandEncoder> BeginCompute();
    void EndEncoding();
    void InvalidateEncoderState();
    void QueueBarrier(MTLStages source_stages, MTLStages destination_stages,
                      MTL4VisibilityOptions visibility = MTL4VisibilityOptionDevice);
    void ProducerBarrier(MTLStages source_stages, MTLStages destination_stages,
                         MTL4VisibilityOptions visibility = MTL4VisibilityOptionDevice);
    void EncoderBarrier(MTLStages source_stages, MTLStages destination_stages,
                        MTL4VisibilityOptions visibility = MTL4VisibilityOptionDevice);
    void Flush();
    void Finish();
    void Wait(u64 tick);
    void WaitForSubmission(const Scheduler& producer, u64 tick);
    void DeferOperation(u64 tick, std::function<void()> operation);
    void WaitForDeferredOperations();
    void PushDebugGroup(std::string_view text);
    void PopDebugGroup();

private:
    friend class Commands;
    friend class StagingBufferPool;

    struct Session {
        id<MTL4CommandAllocator> allocator;
        id<MTL4CommandBuffer> command_buffer;
        id<MTLResidencySet> residency_set;
        NSMutableSet* retained_objects;
        std::vector<std::shared_ptr<void>> retained_lifetimes;
        u64 tick{};
        u64 last_submit_frame{};
    };
    struct CompletionState;

    void RegisterStagingPool(StagingBufferPool& pool);
    void UnregisterStagingPool(StagingBufferPool& pool);
    void AllocateSessions();
    void BeginSession();
    void Submit(id<MTLDrawable> drawable);
    void CheckDeviceLost() const;
    void ClaimEncoderState(const Commands& commands);
    void ValidateAttachment(const RenderAttachment& attachment, const RenderState& state) const;

    const Instance& instance;
    id<MTL4CommandQueue> queue;
    id<MTLSharedEvent> completion_event;
    id<MTL4RenderCommandEncoder> render_encoder;
    id<MTL4ComputeCommandEncoder> compute_encoder;
    RenderState render_state{};
    MTL4CommitOptions* commit_options;
    std::shared_ptr<CompletionState> completion_state;
    std::vector<Session> sessions;
    std::vector<StagingBufferPool*> staging_pools;
    std::vector<std::string> debug_groups;
    size_t current_session{};
    u64 current_tick{1};
    u64 current_frame{};
    u64 encoding_generation{};
    const Commands* commands_owner{};
};

} // namespace Metal
