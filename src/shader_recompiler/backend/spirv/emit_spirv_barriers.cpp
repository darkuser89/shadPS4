// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/microinstruction.h"

namespace Shader::Backend::SPIRV {
namespace {
spv::MemorySemanticsMask GuestMemorySemantics(const EmitContext& ctx, IR::MemoryClass memory) {
    auto semantics = spv::MemorySemanticsMask::MaskNone;
    for (const auto& buffer : ctx.info.buffers) {
        const bool selected =
            (buffer.buffer_type == BufferType::Guest && True(memory & IR::MemoryClass::Buffer)) ||
            (buffer.buffer_type == BufferType::GdsBuffer && True(memory & IR::MemoryClass::Gds)) ||
            (buffer.buffer_type == BufferType::SharedMemory &&
             True(memory & IR::MemoryClass::Shared));
        if (buffer.is_written && selected) {
            semantics = semantics | spv::MemorySemanticsMask::UniformMemory;
            break;
        }
    }
    if (True(memory & IR::MemoryClass::Shared) && ctx.info.shared_types != IR::Type::Void) {
        semantics = semantics | spv::MemorySemanticsMask::WorkgroupMemory;
    }
    if (True(memory & IR::MemoryClass::Image) &&
        std::ranges::any_of(ctx.info.images, [](const auto& image) { return image.is_written; })) {
        semantics = semantics | spv::MemorySemanticsMask::ImageMemory;
    }
    return semantics;
}

spv::MemorySemanticsMask SharedMemorySemantics(const EmitContext& ctx) {
    auto semantics =
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::WorkgroupMemory;
    if (ctx.profile.needs_storage_lds_barriers &&
        std::ranges::any_of(ctx.info.buffers, [](const auto& buffer) {
            return buffer.buffer_type == BufferType::SharedMemory;
        })) {
        semantics = semantics | spv::MemorySemanticsMask::UniformMemory;
    }
    return semantics;
}

void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(scope)),
                        ctx.ConstU32(static_cast<u32>(semantics)));
}
} // Anonymous namespace

void EmitBarrier(EmitContext& ctx, IR::Inst* inst) {
    const auto execution{spv::Scope::Workgroup};
    spv::Scope memory;
    spv::MemorySemanticsMask memory_semantics;
    if (ctx.sw_stage == Shader::SwStage::TessellationControl) {
        memory = spv::Scope::Invocation;
        memory_semantics = spv::MemorySemanticsMask::MaskNone;
    } else {
        memory = spv::Scope::Workgroup;
        memory_semantics = SharedMemorySemantics(ctx);
        if (ctx.profile.needs_guest_compute_barriers) {
            memory_semantics =
                memory_semantics | GuestMemorySemantics(ctx, inst->Flags<IR::MemoryClass>());
        }
    }
    ctx.OpControlBarrier(ctx.ConstU32(static_cast<u32>(execution)),
                         ctx.ConstU32(static_cast<u32>(memory)),
                         ctx.ConstU32(static_cast<u32>(memory_semantics)));
}

void EmitSubgroupBarrier(EmitContext& ctx) {
    const auto semantics{static_cast<u32>(SharedMemorySemantics(ctx))};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup)),
                        ctx.ConstU32(semantics));
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx, IR::Inst* inst) {
    if (!ctx.profile.needs_guest_compute_barriers) {
        return MemoryBarrier(ctx, spv::Scope::Device);
    }
    const auto semantics = GuestMemorySemantics(ctx, inst->Flags<IR::MemoryClass>());
    if (semantics == spv::MemorySemanticsMask::MaskNone) {
        return;
    }
    ctx.OpMemoryBarrier(
        ctx.ConstU32(static_cast<u32>(spv::Scope::Device)),
        ctx.ConstU32(static_cast<u32>(semantics | spv::MemorySemanticsMask::AcquireRelease)));
}

} // namespace Shader::Backend::SPIRV
