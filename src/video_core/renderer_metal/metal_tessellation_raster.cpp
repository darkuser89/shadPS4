// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "video_core/renderer_metal/metal_geometry_export.h"
#include "video_core/renderer_metal/metal_tessellation_raster.h"

namespace Metal {
namespace {
namespace IR = Shader::IR;

bool ValidOutput(IR::Attribute attribute, u32 component) {
    if (attribute == IR::Attribute::Position0 || IR::IsParam(attribute)) {
        return component < 4;
    }
    if (attribute == IR::Attribute::ClipDistance || attribute == IR::Attribute::CullDistance) {
        return component < 8;
    }
    return !component && (attribute == IR::Attribute::PointSize ||
                          attribute == IR::Attribute::RenderTargetIndex ||
                          attribute == IR::Attribute::ViewportIndex);
}
} // namespace

TessellationRasterTranslation::TessellationRasterTranslation(const Shader::RuntimeInfo& runtime,
                                                             const Shader::Profile& profile)
    : vertex_runtime{runtime.hw.vs},
      viewport_width{std::min(profile.max_viewport_width / 2, 8192u)},
      viewport_height{std::min(profile.max_viewport_height / 2, 8192u)} {
    if (runtime.hw_stage != Shader::HwStage::Vertex ||
        (runtime.sw_stage != Shader::SwStage::TessellationEval &&
         runtime.sw_stage != Shader::SwStage::Vertex) ||
        (vertex_runtime.user_clip_plane_mask & ~0x3fu) ||
        (vertex_runtime.clip_disable && (!viewport_width || !viewport_height))) {
        throw std::runtime_error("Invalid native direct TES raster context");
    }
}

void TessellationRasterTranslation::Collect(IR::Program& program) {
    std::map<std::pair<IR::Attribute, u32>, u32> outputs;
    for (auto* block : program.blocks) {
        for (const auto& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
                continue;
            }
            if (!inst.Arg(0).IsImmediate() || !inst.Arg(2).IsImmediate() || inst.Flags<u32>() ||
                !ValidOutput(inst.Arg(0).Attribute(), inst.Arg(2).U32())) {
                throw std::runtime_error("Unsupported native TES raster export");
            }
            outputs.try_emplace({inst.Arg(0).Attribute(), inst.Arg(2).U32()}, 0);
        }
    }
    for (u32 component = 0; component < 4; ++component) {
        if (!outputs.contains({IR::Attribute::Position0, component})) {
            throw std::runtime_error("Native TES raster requires complete position exports");
        }
    }
    dynamic_viewport =
        vertex_runtime.clip_disable && outputs.contains({IR::Attribute::ViewportIndex, 0});
    clip_planes.Collect(outputs, vertex_runtime.user_clip_plane_mask);
    if (program.blocks.empty() || outputs.empty() ||
        outputs.size() > std::numeric_limits<u32>::max() - program.next_reg_index) {
        throw std::runtime_error("Native TES raster output capacity exceeded");
    }
    for (const auto& [attribute, _] : outputs) {
        bindings.attributes.push_back(
            {attribute.first, attribute.second, 0, u32(bindings.attributes.size() + 1)});
        values.push_back(IR::VirtualReg{program.next_reg_index++, IR::Type::U32});
    }
    bindings.vertex_words = u32(values.size() + 1);
}

void TessellationRasterTranslation::Lower(IR::Program& program, const IR::U32& invocation,
                                          const IR::U32& input_base, u32 inputs,
                                          u32 viewport_word_) {
    if (!bindings.vertex_words ||
        program.info.buffers.size() + 1 + bool(vertex_runtime.user_clip_plane_mask) >
            Shader::NUM_BUFFERS) {
        throw std::runtime_error("Native TES raster payload buffer capacity exceeded");
    }
    input_buffer = inputs;
    viewport_word = viewport_word_;
    bindings.output = u32(program.info.buffers.size());
    auto& buffer = program.info.buffers.emplace_back();
    buffer.buffer_type = Shader::BufferType::GeometryPayload;
    buffer.used_types = IR::Type::U32;
    buffer.is_written = true;
    clip_planes.Bind(program);
    auto* entry = program.blocks.front();
    IR::IREmitter initialize{
        *entry, std::next(IR::Block::InstructionList::s_iterator_to(*input_base.Inst()))};
    const auto enabled = initialize.LoadBufferU32(
        1, initialize.Imm32(inputs),
        initialize.IAdd(input_base,
                        initialize.Imm32(u32{offsetof(GeometryExportInput, active) / sizeof(u32)})),
        {});
    active = initialize.IEqual(IR::U32{enabled}, initialize.Imm32(1u));
    row = IR::U32{initialize.IMul(invocation, initialize.Imm32(bindings.vertex_words))};
    for (const auto& value : values) {
        initialize.SetVirtualReg(value, initialize.Imm32(0u));
    }
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
                continue;
            }
            const auto found = std::ranges::find_if(bindings.attributes, [&](const auto& out) {
                return out.attribute == inst.Arg(0).Attribute() &&
                       out.component == inst.Arg(2).U32();
            });
            if (found == bindings.attributes.end()) {
                throw std::runtime_error("Unmapped native TES raster export");
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const auto& reg = values[found - bindings.attributes.begin()];
            const auto next = ir.Select(ir.GetExec(), ir.BitCast<IR::U32>(IR::F32{inst.Arg(1)}),
                                        IR::U32{ir.GetVirtualReg(reg)});
            ir.SetVirtualReg(reg, next);
            inst.Invalidate();
        }
    }
}

void TessellationRasterTranslation::Finish(IR::Program& program) {
    for (auto* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::Epilogue) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            std::vector<IR::U32> exported;
            for (const auto& value : values) {
                exported.push_back(IR::U32{ir.GetVirtualReg(value)});
            }
            const auto position = [&](u32 component) {
                return size_t(std::ranges::find_if(bindings.attributes,
                                                   [&](const auto& output) {
                                                       return output.attribute ==
                                                                  IR::Attribute::Position0 &&
                                                              output.component == component;
                                                   }) -
                              bindings.attributes.begin());
            };
            clip_planes.Evaluate(ir, bindings.attributes, exported);
            if (vertex_runtime.emulate_depth_negative_one_to_one) {
                const auto z = position(2);
                const auto w = position(3);
                exported[z] = ir.BitCast<IR::U32>(IR::F32{ir.FPMul(
                    ir.FPAdd(ir.BitCast<IR::F32>(exported[z]), ir.BitCast<IR::F32>(exported[w])),
                    ir.Imm32(0.5f))});
            }
            if (vertex_runtime.clip_disable && !dynamic_viewport) {
                const auto viewport = [&](u32 word) {
                    return ir.BitCast<IR::F32>(IR::U32{ir.LoadBufferU32(
                        1, ir.Imm32(input_buffer), ir.Imm32(viewport_word + word), {})});
                };
                for (u32 component = 0; component < 2; ++component) {
                    const auto index = position(component);
                    const auto window = ir.FPAdd(
                        ir.FPMul(ir.BitCast<IR::F32>(exported[index]), viewport(component + 2)),
                        viewport(component));
                    exported[index] = ir.BitCast<IR::U32>(IR::F32{ir.FPSub(
                        ir.FPDiv(window,
                                 ir.Imm32(float(component ? viewport_height : viewport_width))),
                        ir.Imm32(1.f))});
                }
            }
            const auto store = [&](u32 word, const IR::U32& value) {
                ir.StoreBufferU32(
                    1, ir.Imm32(bindings.output),
                    ir.GuardBufferAddress(IR::U32{ir.IAdd(row, ir.Imm32(word))}, active), value,
                    {});
            };
            store(0, ir.Imm32(1u));
            for (u32 index = 0; index < bindings.attributes.size(); ++index) {
                store(bindings.attributes[index].word, exported[index]);
            }
        }
    }
}

} // namespace Metal
