// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer.h"

namespace Metal {

namespace {
constexpr size_t SessionBatchSize = 4;
constexpr u64 SessionIdleFrames = 300;
} // namespace

struct Scheduler::CompletionState {
    struct Operation {
        u64 tick;
        std::function<void()> callback;
    };
    std::atomic<bool> device_lost{};
    std::mutex mutex;
    std::string error;
    std::condition_variable condition;
    std::deque<Operation> pending;
    bool running{};
};

Scheduler::Scheduler(const Instance& instance_)
    : instance{instance_}, completion_state{std::make_shared<CompletionState>()} {
    @autoreleasepool {
        queue = [instance.GetDevice() newMTL4CommandQueue];
        completion_event = [instance.GetDevice() newSharedEvent];
        if (!queue || !completion_event) {
            UNREACHABLE_MSG("Failed to create Metal 4 command queue or completion event");
        }
        completion_event.label = @"shadPS4:Metal4Completion";
        commit_options = [MTL4CommitOptions new];
        auto state = completion_state;
        [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
          if (feedback.error) {
              std::scoped_lock lock{state->mutex};
              state->error = ErrorDescription(feedback.error);
              state->device_lost.store(true, std::memory_order_release);
          }
        }];
        AllocateSessions();
        BeginSession();
    }
}

Scheduler::~Scheduler() {
    while (!debug_groups.empty())
        PopDebugGroup();
    Finish();
    WaitForDeferredOperations();
}

void Scheduler::WaitForDeferredOperations() {
    std::unique_lock lock{completion_state->mutex};
    completion_state->condition.wait(
        lock, [&] { return completion_state->pending.empty() && !completion_state->running; });
}

void Scheduler::PushDebugGroup(std::string_view text) {
    NSString* label = [[NSString alloc] initWithBytes:text.data()
                                               length:text.size()
                                             encoding:NSUTF8StringEncoding];
    if (!label)
        throw std::runtime_error("Invalid UTF-8 Metal debug marker");
    debug_groups.emplace_back(text);
    [CommandBuffer() pushDebugGroup:label];
}

void Scheduler::PopDebugGroup() {
    if (debug_groups.empty())
        throw std::runtime_error("Unbalanced Metal debug marker group");
    [CommandBuffer() popDebugGroup];
    debug_groups.pop_back();
}

id<MTL4CommandBuffer> Scheduler::CommandBuffer() const noexcept {
    return sessions[current_session].command_buffer;
}

id<MTL4CommandQueue> Scheduler::GetQueue() const noexcept {
    return queue;
}

u64 Scheduler::CurrentTick() const noexcept {
    return current_tick;
}

u64 Scheduler::EncodingGeneration() const noexcept {
    return encoding_generation;
}

bool Scheduler::IsFree(u64 tick) const {
    CheckDeviceLost();
    return completion_event.signaledValue >= tick;
}

bool Scheduler::IsResourceFree(id<MTLAllocation> allocation) const {
    ASSERT(allocation);
    CheckDeviceLost();
    const u64 gpu_tick = completion_event.signaledValue;
    for (size_t i = 0; i < sessions.size(); ++i) {
        const auto& session = sessions[i];
        if ((i == current_session || session.tick > gpu_tick) &&
            [session.retained_objects containsObject:allocation]) {
            return false;
        }
    }
    return true;
}

void Scheduler::UseResource(id<MTLAllocation> allocation) {
    ASSERT(allocation);
    auto& session = sessions[current_session];
    if ([session.residency_set containsAllocation:allocation]) {
        return;
    }
    [session.residency_set addAllocation:allocation];
    RetainObject(allocation);
}

void Scheduler::RetainObject(id object) {
    ASSERT(object);
    [sessions[current_session].retained_objects addObject:object];
}

void Scheduler::RetainLifetime(std::shared_ptr<void> lifetime) {
    ASSERT(lifetime);
    sessions[current_session].retained_lifetimes.emplace_back(std::move(lifetime));
}

void Scheduler::RegisterStagingPool(StagingBufferPool& pool) {
    ASSERT(std::ranges::find(staging_pools, &pool) == staging_pools.end());
    staging_pools.push_back(&pool);
}

void Scheduler::UnregisterStagingPool(StagingBufferPool& pool) {
    std::erase(staging_pools, &pool);
}

void Scheduler::TickFrame() {
    ++current_frame;
    ReclaimCompletedResources();
}

void Scheduler::ReclaimCompletedResources() {
    CheckDeviceLost();
    const u64 gpu_tick = completion_event.signaledValue;
    for (size_t i = 0; i < sessions.size(); ++i) {
        auto& session = sessions[i];
        if (i == current_session || !session.tick || session.tick > gpu_tick) {
            continue;
        }
        [session.residency_set removeAllAllocations];
        [session.residency_set commit];
        [session.retained_objects removeAllObjects];
        session.retained_lifetimes.clear();
        session.tick = 0;
    }
    if (sessions.size() > SessionBatchSize && current_frame >= SessionIdleFrames) {
        size_t kept{};
        size_t remaining = sessions.size();
        size_t next_current_session{};
        for (size_t i = 0; i < sessions.size(); ++i) {
            auto& session = sessions[i];
            if (i != current_session && remaining > SessionBatchSize && !session.tick &&
                current_frame >= session.last_submit_frame &&
                current_frame - session.last_submit_frame >= SessionIdleFrames) {
                --remaining;
                continue;
            }
            if (i == current_session) {
                next_current_session = kept;
            }
            if (kept != i) {
                sessions[kept] = std::move(session);
            }
            ++kept;
        }
        sessions.resize(kept);
        current_session = next_current_session;
    }
    for (auto* pool : staging_pools) {
        pool->Trim(current_frame);
    }
}

void Scheduler::WaitForDrawable(id<MTLDrawable> drawable) {
    ASSERT(drawable);
    CheckDeviceLost();
    RetainObject(drawable);
    [queue waitForDrawable:drawable];
}

u64 Scheduler::Present(id<MTLDrawable> drawable) {
    ASSERT(drawable);
    RetainObject(drawable);
    const u64 tick = CurrentTick();
    Submit(drawable);
    return tick;
}

id<MTL4RenderCommandEncoder> Scheduler::BeginRendering(const RenderState& state) {
    if (render_encoder && render_state == state) {
        return render_encoder;
    }
    if (state.num_color_attachments > state.color_attachments.size() || !state.width ||
        !state.height || !state.num_layers ||
        ![instance.GetDevice() supportsTextureSampleCount:state.num_samples]) {
        UNREACHABLE_MSG("Invalid Metal render state");
    }
    if ((state.visibility_result_type != MTLVisibilityResultTypeReset &&
         state.visibility_result_type != MTLVisibilityResultTypeAccumulate) ||
        (state.visibility_buffer && (state.visibility_buffer.device != instance.GetDevice() ||
                                     state.visibility_buffer.length < sizeof(u64)))) {
        UNREACHABLE_MSG("Invalid Metal visibility buffer or result type");
    }
    if (state.custom_sample_positions &&
        (state.num_samples > state.sample_positions.size() ||
         !instance.GetDevice().areProgrammableSamplePositionsSupported ||
         std::ranges::any_of(
             state.sample_positions.begin(), state.sample_positions.begin() + state.num_samples,
             [](const auto& position) { return position[0] >= 16 || position[1] >= 16; }))) {
        UNREACHABLE_MSG("Invalid Metal programmable sample positions");
    }
    EndEncoding();
    @autoreleasepool {
        MTL4RenderPassDescriptor* descriptor = [MTL4RenderPassDescriptor new];
        descriptor.renderTargetWidth = state.width;
        descriptor.renderTargetHeight = state.height;
        descriptor.renderTargetArrayLength = state.num_layers;
        descriptor.defaultRasterSampleCount = state.num_samples;
        if (state.custom_sample_positions) {
            std::array<MTLSamplePosition, 8> positions;
            for (u32 sample = 0; sample < state.num_samples; ++sample) {
                const auto& position = state.sample_positions[sample];
                positions[sample] = MTLSamplePositionMake(position[0] / 16.f, position[1] / 16.f);
            }
            [descriptor setSamplePositions:positions.data() count:state.num_samples];
        }
        descriptor.visibilityResultBuffer = state.visibility_buffer;
        descriptor.visibilityResultType = state.visibility_result_type;
        if (state.visibility_buffer) {
            UseResource(state.visibility_buffer);
        }

        const auto configure_attachment = [&](MTLRenderPassAttachmentDescriptor* descriptor,
                                              const RenderAttachment& attachment) {
            ValidateAttachment(attachment, state);
            descriptor.texture = attachment.texture;
            descriptor.level = attachment.level;
            descriptor.slice = attachment.slice;
            descriptor.depthPlane = attachment.depth_plane;
            descriptor.resolveTexture = attachment.resolve_texture;
            descriptor.resolveLevel = attachment.resolve_level;
            descriptor.resolveSlice = attachment.resolve_slice;
            descriptor.resolveDepthPlane = attachment.resolve_depth_plane;
            descriptor.loadAction = attachment.load_action;
            descriptor.storeAction = attachment.store_action;
            if (attachment.texture) {
                UseResource(attachment.texture);
            }
            if (attachment.resolve_texture) {
                UseResource(attachment.resolve_texture);
            }
        };
        for (u32 i = 0; i < state.num_color_attachments; ++i) {
            const auto& attachment = state.color_attachments[i];
            auto color = descriptor.colorAttachments[i];
            configure_attachment(color, attachment);
            color.clearColor =
                MTLClearColorMake(attachment.clear_color[0], attachment.clear_color[1],
                                  attachment.clear_color[2], attachment.clear_color[3]);
        }
        configure_attachment(descriptor.depthAttachment, state.depth_attachment);
        descriptor.depthAttachment.clearDepth = state.depth_attachment.clear_depth;
        descriptor.depthAttachment.depthResolveFilter = state.depth_attachment.resolve_filter;
        configure_attachment(descriptor.stencilAttachment, state.stencil_attachment);
        descriptor.stencilAttachment.clearStencil = state.stencil_attachment.clear_stencil;
        descriptor.stencilAttachment.stencilResolveFilter = state.stencil_attachment.resolve_filter;

        render_encoder = [CommandBuffer() renderCommandEncoderWithDescriptor:descriptor];
        if (!render_encoder) {
            UNREACHABLE_MSG("Failed to create Metal 4 render encoder");
        }
        ++encoding_generation;
        render_state = state;
        return render_encoder;
    }
}

id<MTL4ComputeCommandEncoder> Scheduler::BeginCompute() {
    if (compute_encoder) {
        return compute_encoder;
    }
    EndEncoding();
    compute_encoder = [CommandBuffer() computeCommandEncoder];
    if (!compute_encoder) {
        UNREACHABLE_MSG("Failed to create Metal 4 compute encoder");
    }
    ++encoding_generation;
    return compute_encoder;
}

void Scheduler::EndEncoding() {
    if (render_encoder || compute_encoder) {
        InvalidateEncoderState();
    }
    [render_encoder endEncoding];
    [compute_encoder endEncoding];
    render_encoder = nil;
    compute_encoder = nil;
    render_state = {};
}

void Scheduler::InvalidateEncoderState() {
    ++encoding_generation;
    commands_owner = nullptr;
}

void Scheduler::ClaimEncoderState(const Commands& commands) {
    if (commands_owner != &commands) {
        ++encoding_generation;
        commands_owner = &commands;
    }
}

void Scheduler::ValidateAttachment(const RenderAttachment& attachment,
                                   const RenderState& state) const {
    if (!attachment.texture) {
        if (attachment.resolve_texture) {
            UNREACHABLE_MSG("Metal resolve attachment requires a source texture");
        }
        return;
    }
    const auto validate = [&](id<MTLTexture> texture, u32 level, u32 slice, u32 depth_plane,
                              u32 samples) {
        if (texture.device != instance.GetDevice() ||
            !(texture.usage & MTLTextureUsageRenderTarget) || level >= texture.mipmapLevelCount ||
            level >= std::numeric_limits<NSUInteger>::digits || texture.sampleCount != samples) {
            UNREACHABLE_MSG("Invalid Metal render attachment");
        }
        const auto width = std::max<NSUInteger>(1, texture.width >> level);
        const auto height = std::max<NSUInteger>(1, texture.height >> level);
        const auto depth = std::max<NSUInteger>(1, texture.depth >> level);
        const bool volume = texture.textureType == MTLTextureType3D;
        const bool cube = texture.textureType == MTLTextureTypeCube ||
                          texture.textureType == MTLTextureTypeCubeArray;
        const auto slices = texture.arrayLength * (cube ? 6 : 1);
        if (state.width > width || state.height > height || slice >= slices ||
            depth_plane >= depth || (volume && slice) || (!volume && depth_plane) ||
            state.num_layers > (volume ? depth - depth_plane : slices - slice)) {
            UNREACHABLE_MSG("Metal render attachment does not cover the render area or layers");
        }
    };
    validate(attachment.texture, attachment.level, attachment.slice, attachment.depth_plane,
             state.num_samples);
    if (attachment.resolve_texture) {
        if (state.num_samples <= 1 ||
            attachment.texture.pixelFormat != attachment.resolve_texture.pixelFormat) {
            UNREACHABLE_MSG("Metal resolve requires matching formats and a multisampled source");
        }
        validate(attachment.resolve_texture, attachment.resolve_level, attachment.resolve_slice,
                 attachment.resolve_depth_plane, 1);
    }
}

void Scheduler::QueueBarrier(MTLStages source_stages, MTLStages destination_stages,
                             MTL4VisibilityOptions visibility) {
    id<MTL4CommandEncoder> encoder = render_encoder ? (id<MTL4CommandEncoder>)render_encoder
                                                    : (id<MTL4CommandEncoder>)BeginCompute();
    [encoder barrierAfterQueueStages:source_stages
                        beforeStages:destination_stages
                   visibilityOptions:visibility];
}

void Scheduler::ProducerBarrier(MTLStages source_stages, MTLStages destination_stages,
                                MTL4VisibilityOptions visibility) {
    id<MTL4CommandEncoder> encoder = render_encoder ? (id<MTL4CommandEncoder>)render_encoder
                                                    : (id<MTL4CommandEncoder>)BeginCompute();
    [encoder barrierAfterStages:source_stages
              beforeQueueStages:destination_stages
              visibilityOptions:visibility];
}

void Scheduler::EncoderBarrier(MTLStages source_stages, MTLStages destination_stages,
                               MTL4VisibilityOptions visibility) {
    id<MTL4CommandEncoder> encoder = render_encoder ? (id<MTL4CommandEncoder>)render_encoder
                                                    : (id<MTL4CommandEncoder>)BeginCompute();
    [encoder barrierAfterEncoderStages:source_stages
                   beforeEncoderStages:destination_stages
                     visibilityOptions:visibility];
}

void Scheduler::Flush() {
    Submit(nil);
}

void Scheduler::WaitForSubmission(const Scheduler& producer, u64 tick) {
    if (producer.instance.GetDevice() != instance.GetDevice() || !tick) {
        UNREACHABLE_MSG("Invalid Metal queue dependency");
    }
    if (&producer != this) {
        [queue waitForEvent:producer.completion_event value:tick];
    }
}

void Scheduler::DeferOperation(u64 tick, std::function<void()> operation) {
    CheckDeviceLost();
    if (!tick || tick >= current_tick || !operation) {
        throw std::runtime_error("Metal completion operation requires submitted work");
    }
    auto state = completion_state;
    {
        std::scoped_lock lock{state->mutex};
        state->pending.push_back({tick, std::move(operation)});
    }
    [completion_event notifyListener:MTLSharedEventListener.sharedListener
                             atValue:tick
                               block:^(id<MTLSharedEvent> event, u64) {
                                 for (;;) {
                                     std::function<void()> callback;
                                     {
                                         std::scoped_lock lock{state->mutex};
                                         if (state->pending.empty() ||
                                             state->pending.front().tick > event.signaledValue) {
                                             return;
                                         }
                                         callback = std::move(state->pending.front().callback);
                                         state->pending.pop_front();
                                         state->running = true;
                                     }
                                     try {
                                         callback();
                                     } catch (const std::exception& error) {
                                         std::scoped_lock lock{state->mutex};
                                         state->error = error.what();
                                         state->device_lost.store(true, std::memory_order_release);
                                     } catch (...) {
                                         std::scoped_lock lock{state->mutex};
                                         state->error = "Metal completion operation failed";
                                         state->device_lost.store(true, std::memory_order_release);
                                     }
                                     {
                                         std::scoped_lock lock{state->mutex};
                                         state->running = false;
                                         state->condition.notify_all();
                                     }
                                 }
                               }];
}

void Scheduler::Submit(id<MTLDrawable> drawable) {
    CheckDeviceLost();
    @autoreleasepool {
        EndEncoding();
        auto& session = sessions[current_session];
        for (size_t index = 0; index < debug_groups.size(); ++index) {
            [session.command_buffer popDebugGroup];
        }
        [session.residency_set commit];
        [session.command_buffer endCommandBuffer];
        id<MTL4CommandBuffer> command_buffers[] = {session.command_buffer};
        [queue commit:command_buffers count:1 options:commit_options];
        session.tick = current_tick++;
        session.last_submit_frame = current_frame;
        if (drawable) {
            [queue signalDrawable:drawable];
        }
        [queue signalEvent:completion_event value:session.tick];
        if (drawable) {
            [drawable present];
        }
        BeginSession();
    }
}

void Scheduler::Finish() {
    const u64 tick = current_tick;
    Flush();
    Wait(tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick > current_tick) {
        UNREACHABLE_MSG("Cannot wait for unrecorded Metal work: {}", tick);
    }
    if (tick == current_tick) {
        Flush();
    }
    while (!IsFree(tick)) {
        [completion_event waitUntilSignaledValue:tick timeoutMS:100];
    }
}

void Scheduler::AllocateSessions() {
    @autoreleasepool {
        for (size_t i = 0; i < SessionBatchSize; ++i) {
            Session session;
            session.allocator = [instance.GetDevice() newCommandAllocator];
            session.command_buffer = [instance.GetDevice() newCommandBuffer];
            MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
            descriptor.initialCapacity = 128;
            NSError* error = nil;
            session.residency_set = [instance.GetDevice() newResidencySetWithDescriptor:descriptor
                                                                                  error:&error];
            session.retained_objects = [NSMutableSet new];
            if (!session.allocator || !session.command_buffer || !session.residency_set) {
                UNREACHABLE_MSG("Failed to allocate Metal 4 command session: {}",
                                ErrorDescription(error));
            }
            sessions.emplace_back(std::move(session));
        }
    }
}

void Scheduler::BeginSession() {
    const u64 gpu_tick = completion_event.signaledValue;
    size_t next_session = sessions.size();
    for (size_t i = 0; i < sessions.size(); ++i) {
        const size_t index = (current_session + i + 1) % sessions.size();
        if (sessions[index].tick <= gpu_tick) {
            next_session = index;
            break;
        }
    }
    if (next_session == sessions.size()) {
        AllocateSessions();
    }
    current_session = next_session;
    auto& session = sessions[current_session];
    [session.allocator reset];
    [session.residency_set removeAllAllocations];
    [session.residency_set commit];
    [session.retained_objects removeAllObjects];
    session.retained_lifetimes.clear();
    [session.command_buffer beginCommandBufferWithAllocator:session.allocator];
    [session.command_buffer useResidencySet:session.residency_set];
    for (const auto& text : debug_groups) {
        NSString* label = [[NSString alloc] initWithBytes:text.data()
                                                   length:text.size()
                                                 encoding:NSUTF8StringEncoding];
        [session.command_buffer pushDebugGroup:label];
    }
}

void Scheduler::CheckDeviceLost() const {
    if (completion_state->device_lost.load(std::memory_order_acquire)) {
        std::scoped_lock lock{completion_state->mutex};
        UNREACHABLE_MSG("Metal 4 command submission failed: {}", completion_state->error);
    }
}

} // namespace Metal
