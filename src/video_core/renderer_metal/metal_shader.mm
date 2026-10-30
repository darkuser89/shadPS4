// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <fmt/format.h>
#include <spirv_msl.hpp>
#include "common/scope_exit.h"
#include "core/memory.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/operand_helper.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/passes/resource_pass.h"
#include "shader_recompiler/ir/reinterpret.h"
#include "video_core/renderer_metal/metal_format.h"
#include "video_core/renderer_metal/metal_guest_page_layout.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_sampler.h"
#include "video_core/renderer_metal/metal_shader.h"

namespace Metal {

namespace {

Shader::IR::Value SwizzleImageRead(Shader::IR::IREmitter& ir, const Shader::IR::Value& texel,
                                   const AmdGpu::Image& image, bool is_storage, bool integer) {
    namespace IR = Shader::IR;
    const auto mapping = image.DstSelect();
    std::array<IR::F32, 4> components;
    for (u32 index = 0; index < components.size(); ++index) {
        const auto selector = mapping.array[index];
        if (selector == AmdGpu::CompSwizzle::Zero) {
            components[index] = ir.Imm32(0.f);
        } else if (selector == AmdGpu::CompSwizzle::One) {
            components[index] = integer ? ir.BitCast<IR::F32>(ir.Imm32(1u)) : ir.Imm32(1.f);
        } else if (selector >= AmdGpu::CompSwizzle::Red && selector <= AmdGpu::CompSwizzle::Alpha) {
            components[index] =
                IR::F32{ir.CompositeExtract(texel, is_storage ? u32(selector) - 4 : index)};
        } else {
            throw std::runtime_error("Unsupported Metal image swizzle selector");
        }
    }
    return ir.CompositeConstruct(components[0], components[1], components[2], components[3]);
}

Shader::IR::Value ConvertImageRead(Shader::IR::IREmitter& ir, const Shader::IR::Value& texel,
                                   const AmdGpu::Image& image, bool is_storage) {
    const auto conversion = image.GetNumberConversion();
    if (!is_storage && conversion == AmdGpu::NumberConversion::None) {
        return texel;
    }
    const auto converted = Shader::IR::ApplyReadNumberConversionVec4(ir, texel, conversion);
    const bool integer =
        conversion == AmdGpu::NumberConversion::None && AmdGpu::IsInteger(image.GetNumberFmt());
    return SwizzleImageRead(ir, converted, image, is_storage, integer);
}

ImageAliasState LowerImageAliases(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_IMAGES> eligible;
    std::bitset<Shader::NUM_IMAGES> incompatible;
    for (const IR::Block* block : program.blocks) {
        for (const auto& inst : *block) {
            const auto name = IR::NameOf(inst.GetOpcode());
            if (!name.starts_with("Image")) {
                continue;
            }
            const u32 index = inst.Arg(0).U32() & 0xffff;
            if (index >= program.info.images.size() || index >= eligible.size()) {
                throw std::runtime_error("Invalid Metal image alias resource handle");
            }
            if (inst.GetOpcode() == IR::Opcode::ImageRead ||
                inst.GetOpcode() == IR::Opcode::ImageWrite ||
                inst.GetOpcode() == IR::Opcode::ImageQueryDimensions) {
                eligible.set(index);
            } else {
                incompatible.set(index);
            }
        }
    }
    const auto aliases = ImageAliasState::FromInfo(program.info, eligible & ~incompatible);
    std::bitset<Shader::NUM_IMAGES> written;
    for (u32 index = 0; index < program.info.images.size(); ++index) {
        if (program.info.images[index].is_written) {
            written.set(index);
            program.info.images[aliases.targets[index]].is_written = true;
        }
    }
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            const bool read = inst.GetOpcode() == IR::Opcode::ImageRead;
            if (!read && inst.GetOpcode() != IR::Opcode::ImageWrite &&
                inst.GetOpcode() != IR::Opcode::ImageQueryDimensions) {
                continue;
            }
            const u32 index = inst.Arg(0).U32();
            const u32 target = aliases.targets[index];
            if (read && !written.test(index) && program.info.images[target].is_written) {
                IR::IREmitter ir{*block, it};
                const auto lod = inst.Arg(2).IsEmpty() ? IR::U32{} : IR::U32{inst.Arg(2)};
                const auto sample = inst.Arg(3).IsEmpty() ? IR::U32{} : IR::U32{inst.Arg(3)};
                const auto raw = ir.ImageRead(ir.Imm32(target), inst.Arg(1), lod, sample,
                                              inst.Flags<IR::TextureInstInfo>());
                const auto sharp = program.info.images[index].GetSharp(program.info);
                const auto decoded =
                    aliases.srgb_images.test(index) ? IR::ApplyForceDegamma(ir, raw) : raw;
                inst.ReplaceUsesWith(SwizzleImageRead(ir, decoded, sharp, true,
                                                      AmdGpu::IsInteger(sharp.GetNumberFmt())));
                inst.Invalidate();
            } else {
                inst.SetArg(0, IR::Value{target});
            }
        }
    }
    return aliases;
}

void LowerMultisampleImageDimensions(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            if (inst.GetOpcode() != IR::Opcode::ImageQueryDimensions || !inst.Arg(2).U1()) {
                continue;
            }
            const auto& resource = program.info.images.at(inst.Arg(0).U32());
            const auto type = resource.GetSharp(program.info).GetViewType(resource.is_array);
            if (type != AmdGpu::ImageType::Color2DMsaa &&
                type != AmdGpu::ImageType::Color2DMsaaArray) {
                continue;
            }
            IR::IREmitter ir{*block, it};
            const auto size =
                ir.ImageQueryDimension(inst.Arg(0), IR::U32{inst.Arg(1)}, ir.Imm1(false),
                                       inst.Flags<IR::TextureInstInfo>());
            inst.ReplaceUsesWith(ir.CompositeConstruct(ir.CompositeExtract(size, 0),
                                                       ir.CompositeExtract(size, 1),
                                                       ir.CompositeExtract(size, 2), ir.Imm32(1u)));
            inst.Invalidate();
        }
    }
}

void LowerStorageImageDimensions(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            if (inst.GetOpcode() != IR::Opcode::ImageQueryDimensions) {
                continue;
            }
            const auto& resource = program.info.images[inst.Arg(0).U32()];
            if (!resource.is_written ||
                resource.mip_fallback_mode != Shader::MipStorageFallbackMode::None ||
                (inst.Arg(1).IsImmediate() && !inst.Arg(1).U32())) {
                continue;
            }
            const auto type = resource.GetSharp(program.info).GetViewType(resource.is_array);
            u32 spatial_components{};
            switch (type) {
            case AmdGpu::ImageType::Color1D:
            case AmdGpu::ImageType::Color1DArray:
                spatial_components = 1;
                break;
            case AmdGpu::ImageType::Color2D:
            case AmdGpu::ImageType::Color2DArray:
            case AmdGpu::ImageType::Cube:
                spatial_components = 2;
                break;
            case AmdGpu::ImageType::Color3D:
                spatial_components = 3;
                break;
            case AmdGpu::ImageType::Color2DMsaa:
            case AmdGpu::ImageType::Color2DMsaaArray:
                continue;
            default:
                throw std::runtime_error("Unsupported Metal storage image dimension query");
            }
            IR::IREmitter ir{*block, it};
            const auto base = ir.ImageQueryDimension(inst.Arg(0), ir.Imm32(0u), IR::U1{inst.Arg(2)},
                                                     inst.Flags<IR::TextureInstInfo>());
            const auto shift = ir.UMin(IR::U32{inst.Arg(1)}, ir.Imm32(31u));
            std::array<IR::U32, 4> dimensions;
            for (u32 component = 0; component < dimensions.size(); ++component) {
                const auto value = IR::U32{ir.CompositeExtract(base, component)};
                dimensions[component] =
                    component < spatial_components
                        ? ir.UMax(IR::U32{ir.ShiftRightLogical(value, shift)}, ir.Imm32(1u))
                        : value;
            }
            inst.ReplaceUsesWith(
                ir.CompositeConstruct(dimensions[0], dimensions[1], dimensions[2], dimensions[3]));
            inst.Invalidate();
        }
    }
}

Shader::IR::Value ConvertImageSample(Shader::IR::IREmitter& ir, const Shader::IR::Value& texel,
                                     const AmdGpu::Image& image,
                                     const Shader::IR::TextureInstInfo& flags) {
    namespace IR = Shader::IR;
    if (flags.is_depth) {
        return texel;
    }
    if (!flags.is_gather) {
        return ConvertImageRead(ir, texel, image, false);
    }
    const auto conversion = image.GetNumberConversion();
    if (conversion == AmdGpu::NumberConversion::None) {
        return texel;
    }
    const auto selector = image.DstSelect().array[flags.gather_comp.Value()];
    if (selector == AmdGpu::CompSwizzle::Zero || selector == AmdGpu::CompSwizzle::One) {
        const auto value = ir.Imm32(selector == AmdGpu::CompSwizzle::One ? 1.f : 0.f);
        return ir.CompositeConstruct(value, value, value, value);
    }
    if (selector < AmdGpu::CompSwizzle::Red || selector > AmdGpu::CompSwizzle::Alpha) {
        throw std::runtime_error("Unsupported Metal gather swizzle selector");
    }
    return IR::ApplyReadNumberConversionVec4(ir, texel, conversion);
}

template <typename T>
Shader::IR::U32 ShaderDescriptorWord(Shader::IR::IREmitter& ir, const Shader::SharpFetch<T>& fetch,
                                     u32 word) {
    namespace IR = Shader::IR;
    using Summary = typename Shader::SharpFetch<T>::Summary;
    if (fetch.summary == Summary::Invalid) {
        return ir.Imm32(0u);
    }
    if (fetch.summary == Summary::MultiLoad && !(fetch.load_mask & (1u << word))) {
        return ir.Imm32(fetch.immediates[word]);
    }
    const u32 location =
        fetch.summary == Summary::SingleLoad ? fetch.offsets[0] + word : fetch.offsets[word];
    if (location < Shader::NUM_USER_DATA_REGS) {
        return ir.GetUserData(static_cast<IR::ScalarReg>(location));
    }
    const auto base = ir.CompositeConstruct(ir.Imm32(0u), ir.Imm32(0u));
    auto value = ir.ReadConst(base, ir.Imm32(0u));
    value.Inst()->SetFlags(location);
    return value;
}

Shader::IR::U32 BufferDescriptorWord(Shader::IR::IREmitter& ir,
                                     const Shader::BufferResource& resource, u32 word) {
    return ShaderDescriptorWord(ir, resource.sharp_fetch, word);
}

std::bitset<Shader::NUM_IMAGES> LowerDescriptorImageDimensions(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_IMAGES> queried;
    std::bitset<Shader::NUM_IMAGES> accessed;
    for (const IR::Block* block : program.blocks) {
        for (const auto& inst : *block) {
            if (!IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                continue;
            }
            const u32 index = inst.Arg(0).U32() & 0xffff;
            if (index >= program.info.images.size() || index >= queried.size()) {
                throw std::runtime_error("Invalid Metal image dimensions resource handle");
            }
            if (inst.GetOpcode() == IR::Opcode::ImageQueryDimensions) {
                queried.set(index);
            } else {
                accessed.set(index);
            }
        }
    }
    auto lowered = queried & ~accessed;
    for (u32 index = 0; index < program.info.images.size(); ++index) {
        const auto& resource = program.info.images[index];
        if (!resource.GetSharp(program.info).Address() || resource.is_depth || resource.is_atomic ||
            resource.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
            lowered.reset(index);
        }
    }
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            if (inst.GetOpcode() != IR::Opcode::ImageQueryDimensions ||
                !lowered.test(inst.Arg(0).U32())) {
                continue;
            }
            const auto& resource = program.info.images[inst.Arg(0).U32()];
            const auto type = resource.GetSharp(program.info).GetViewType(resource.is_array);
            IR::IREmitter ir{*block, it};
            const auto word = [&](u32 index) {
                return resource.is_r128 && index >= 4
                           ? ir.Imm32(0u)
                           : ShaderDescriptorWord(ir, resource.sharp_fetch, index);
            };
            const auto field = [&](IR::U32 value, u32 offset, u32 bits) {
                return ir.BitFieldExtract(value, ir.Imm32(offset), ir.Imm32(bits), false);
            };
            const auto dimensions = word(2);
            const auto control = word(3);
            const auto base = field(control, 12, 4);
            const bool multisampled = type == AmdGpu::ImageType::Color2DMsaa ||
                                      type == AmdGpu::ImageType::Color2DMsaaArray;
            const auto shift =
                multisampled ? ir.Imm32(0u)
                             : ir.UMin(ir.IAdd(base, ir.UMin(IR::U32{inst.Arg(1)}, ir.Imm32(31u))),
                                       ir.Imm32(31u));
            const auto extent = [&](IR::U32 value) {
                return ir.UMax(IR::U32{ir.ShiftRightLogical(ir.IAdd(value, ir.Imm32(1u)), shift)},
                               ir.Imm32(1u));
            };
            const auto width = extent(field(dimensions, 0, 14));
            auto height = ir.Imm32(0u);
            auto depth = ir.Imm32(0u);
            if (type != AmdGpu::ImageType::Color1D && type != AmdGpu::ImageType::Color1DArray) {
                height = extent(field(dimensions, 14, 14));
            }
            if (type == AmdGpu::ImageType::Color3D) {
                depth = extent(field(word(4), 0, 13));
            } else if (type == AmdGpu::ImageType::Color1DArray ||
                       type == AmdGpu::ImageType::Color2DArray ||
                       type == AmdGpu::ImageType::Color2DMsaaArray ||
                       type == AmdGpu::ImageType::Cube) {
                auto layers = ir.IAdd(field(word(4), 0, 13), ir.Imm32(1u));
                if (type == AmdGpu::ImageType::Cube ||
                    resource.post_op == Shader::SharpFetchPostOp::ConvertCubeTo2DArray) {
                    layers = ir.IMul(layers, ir.Imm32(6u));
                    if (resource.post_op == Shader::SharpFetchPostOp::ConvertCubeTo2DArray) {
                        layers =
                            ir.IAdd(ir.BitwiseAnd(ir.ISub(layers, ir.Imm32(1u)), ir.Imm32(0x1fffu)),
                                    ir.Imm32(1u));
                    }
                }
                auto padded = ir.ISub(layers, ir.Imm32(1u));
                for (u32 bits : {1u, 2u, 4u, 8u, 16u}) {
                    padded = ir.BitwiseOr(padded, ir.ShiftRightLogical(padded, ir.Imm32(bits)));
                }
                layers = IR::U32{ir.Select(ir.INotEqual(field(control, 25, 1), ir.Imm32(0u)),
                                           ir.IAdd(padded, ir.Imm32(1u)), layers)};
                const auto slices = word(5);
                const auto first = field(slices, 0, 13);
                const auto end = ir.UMin(ir.IAdd(field(slices, 13, 13), ir.Imm32(1u)), layers);
                const auto count = IR::U32{ir.Select(ir.IGreaterThan(end, first, false),
                                                     ir.ISub(end, first), ir.Imm32(1u))};
                if (type == AmdGpu::ImageType::Color1DArray) {
                    height = count;
                } else {
                    depth = count;
                }
            }
            auto levels = ir.Imm32(0u);
            if (inst.Arg(2).U1()) {
                const auto last = ir.IAdd(field(control, 16, 4), ir.Imm32(1u));
                levels = multisampled ? ir.Imm32(1u)
                                      : IR::U32{ir.Select(ir.IGreaterThan(last, base, false),
                                                          ir.ISub(last, base), ir.Imm32(1u))};
            }
            inst.ReplaceUsesWith(ir.CompositeConstruct(width, height, depth, levels));
            inst.Invalidate();
        }
    }
    return lowered;
}

std::bitset<Shader::NUM_IMAGES> QueryOnlyImages(const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_IMAGES> queried;
    std::bitset<Shader::NUM_IMAGES> accessed;
    for (const IR::Block* block : program.blocks) {
        for (const auto& inst : *block) {
            if (!IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                continue;
            }
            const u32 index = inst.Arg(0).U32() & 0xffff;
            const auto& resource = program.info.images.at(index);
            if ((inst.GetOpcode() == IR::Opcode::ImageQueryDimensions ||
                 inst.GetOpcode() == IR::Opcode::ImageQueryLod) &&
                !resource.is_written && !resource.is_atomic) {
                queried.set(index);
            } else {
                accessed.set(index);
            }
        }
    }
    return queried & ~accessed;
}

std::bitset<Shader::NUM_IMAGES> LowerFmaskImageAccesses(
    Shader::IR::Program& program, std::array<FmaskImageState, Shader::NUM_IMAGES>& image_states) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_IMAGES> dependencies;
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            if (!IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                continue;
            }
            const u32 index = inst.Arg(0).U32() & 0xffff;
            const auto& resource = program.info.images.at(index);
            const auto image = resource.GetSharp(program.info);
            dependencies.set(index);
            image_states[index] = FmaskImageState::FromImage(image);
            if (!AmdGpu::IsFmask(image.GetDataFmt())) {
                continue;
            }
            if ((inst.GetOpcode() != IR::Opcode::ImageRead &&
                 inst.GetOpcode() != IR::Opcode::ImageQueryDimensions &&
                 inst.GetOpcode() != IR::Opcode::ImageQueryLod) ||
                resource.is_written || resource.is_depth || resource.is_atomic) {
                throw std::runtime_error(
                    fmt::format("Metal Fmask format {} requires native data/layout lowering for {}",
                                u32(image.GetDataFmt()), IR::NameOf(inst.GetOpcode())));
            }
            static_cast<void>(PhysicalFmaskImage(image));
            if (inst.GetOpcode() != IR::Opcode::ImageRead) {
                continue;
            }
            const auto format = GetFmaskFormat(image.GetDataFmt());
            IR::IREmitter ir{*block, it};
            const auto lod = inst.Arg(2).IsEmpty() ? IR::U32{} : IR::U32{inst.Arg(2)};
            const auto sample = inst.Arg(3).IsEmpty() ? IR::U32{} : IR::U32{inst.Arg(3)};
            const auto raw = ir.ImageRead(ir.Imm32(index), inst.Arg(1), lod, sample,
                                          inst.Flags<IR::TextureInstInfo>());
            std::array<IR::U32, 2> decoded{ir.Imm32(0u), ir.Imm32(0u)};
            if (format.index_bits == 4) {
                for (u32 word = 0; word < format.samples / 8; ++word) {
                    decoded[word] = ir.BitCast<IR::U32>(IR::F32{ir.CompositeExtract(raw, word)});
                }
            } else {
                for (u32 sample_index = 0; sample_index < format.samples; ++sample_index) {
                    const u32 bit = sample_index * format.index_bits;
                    const auto packed =
                        ir.BitCast<IR::U32>(IR::F32{ir.CompositeExtract(raw, bit / 32)});
                    const auto fragment = ir.BitFieldExtract(packed, ir.Imm32(bit % 32),
                                                             ir.Imm32(format.index_bits), false);
                    const auto shifted =
                        ir.ShiftLeftLogical(fragment, ir.Imm32((sample_index % 8) * 4));
                    decoded[sample_index / 8] =
                        IR::U32{ir.BitwiseOr(decoded[sample_index / 8], shifted)};
                }
            }
            const auto values = ir.CompositeConstruct(
                ir.BitCast<IR::F32>(decoded[0]), ir.BitCast<IR::F32>(decoded[1]),
                ir.CompositeExtract(raw, 2), ir.CompositeExtract(raw, 3));
            inst.ReplaceUsesWith(SwizzleImageRead(ir, values, image, true, true));
            inst.Invalidate();
        }
    }
    return dependencies;
}

u32 GuestPageTableHandle(Shader::Info& info) {
    for (u32 index = 0; index < info.buffers.size(); ++index) {
        if (info.buffers[index].buffer_type == Shader::BufferType::GuestPageTable) {
            return index;
        }
    }
    if (info.buffers.size() >= Shader::NUM_BUFFERS - 1) {
        throw std::runtime_error("Native page table exceeds shader buffer capacity");
    }
    const u32 handle = info.buffers.size();
    info.buffers.push_back(
        {.used_types = Shader::IR::Type::U64, .buffer_type = Shader::BufferType::GuestPageTable});
    return handle;
}

u32 BufferBoundsHandle(Shader::Info& info) {
    for (u32 index = 0; index < info.buffers.size(); ++index) {
        if (info.buffers[index].buffer_type == Shader::BufferType::BufferBounds) {
            return index;
        }
    }
    if (info.buffers.size() >= Shader::NUM_BUFFERS - 1) {
        throw std::runtime_error("Metal buffer bounds exceed shader buffer capacity");
    }
    const u32 handle = info.buffers.size();
    info.buffers.push_back(
        {.used_types = Shader::IR::Type::U32, .buffer_type = Shader::BufferType::BufferBounds});
    return handle;
}

std::optional<u32> GuestPageAddressArgument(Shader::IR::Opcode opcode) {
    using Shader::IR::Opcode;
    switch (opcode) {
    case Opcode::ReadConstBuffer:
    case Opcode::LoadBufferFormatF32:
    case Opcode::LoadBufferU8:
    case Opcode::LoadBufferU16:
    case Opcode::LoadBufferU32:
    case Opcode::LoadBufferU32x2:
    case Opcode::LoadBufferU32x3:
    case Opcode::LoadBufferU32x4:
    case Opcode::LoadBufferF32:
    case Opcode::LoadBufferF32x2:
    case Opcode::LoadBufferF32x3:
    case Opcode::LoadBufferF32x4:
        return 2;
    case Opcode::BufferAtomicIAdd32:
    case Opcode::BufferAtomicISub32:
    case Opcode::BufferAtomicSMin32:
    case Opcode::BufferAtomicUMin32:
    case Opcode::BufferAtomicSMax32:
    case Opcode::BufferAtomicUMax32:
    case Opcode::BufferAtomicAnd32:
    case Opcode::BufferAtomicOr32:
    case Opcode::BufferAtomicXor32:
    case Opcode::BufferAtomicSwap32:
    case Opcode::BufferAtomicInc32:
    case Opcode::BufferAtomicDec32:
    case Opcode::BufferAtomicFMin32:
    case Opcode::BufferAtomicFMax32:
    case Opcode::StoreBufferFormatF32:
    case Opcode::StoreBufferU8:
    case Opcode::StoreBufferU16:
    case Opcode::StoreBufferU32:
    case Opcode::StoreBufferU32x2:
    case Opcode::StoreBufferU32x3:
    case Opcode::StoreBufferU32x4:
    case Opcode::StoreBufferF32:
    case Opcode::StoreBufferF32x2:
    case Opcode::StoreBufferF32x3:
    case Opcode::StoreBufferF32x4:
        return 3;
    case Opcode::BufferAtomicFCmpSwapRaw32:
    case Opcode::BufferAtomicFCmpSwap32:
    case Opcode::BufferAtomicCmpSwap32:
        return 4;
    default:
        return std::nullopt;
    }
}

Shader::IR::U64 GuestPageBufferAddress(Shader::IR::IREmitter& ir, Shader::Info& info, u32 handle,
                                       const Shader::IR::U64& byte_offset, u32 shift = 2) {
    const u32 bounds = BufferBoundsHandle(info);
    const u32 address_word = ShaderCompiler::BufferAddressWordOffset + handle * 2;
    const auto low = ir.ReadConstBuffer(ir.Imm32(bounds), ir.Imm32(address_word), {});
    const auto high = ir.ReadConstBuffer(ir.Imm32(bounds), ir.Imm32(address_word + 1), {});
    const auto base = ir.PackUint2x32(ir.CompositeConstruct(low, high));
    (void)GuestPageTableHandle(info);
    info.uses_guest_page_reads = true;
    return ir.IAdd(ir.BitwiseAnd(base, ir.Imm64(~u64{(1u << shift) - 1})), byte_offset);
}

struct RawBufferAccess {
    u32 components{};
    u32 shift{};
    bool floating{};
};

RawBufferAccess GetRawBufferAccess(Shader::IR::Opcode opcode) {
    using Shader::IR::Opcode;
    switch (opcode) {
    case Opcode::LoadBufferU8:
    case Opcode::StoreBufferU8:
        return {1, 0, false};
    case Opcode::LoadBufferU16:
    case Opcode::StoreBufferU16:
        return {1, 1, false};
    case Opcode::LoadBufferU32:
    case Opcode::StoreBufferU32:
    case Opcode::BufferAtomicIAdd32:
    case Opcode::BufferAtomicISub32:
    case Opcode::BufferAtomicSMin32:
    case Opcode::BufferAtomicUMin32:
    case Opcode::BufferAtomicFMin32:
    case Opcode::BufferAtomicSMax32:
    case Opcode::BufferAtomicUMax32:
    case Opcode::BufferAtomicFMax32:
    case Opcode::BufferAtomicInc32:
    case Opcode::BufferAtomicDec32:
    case Opcode::BufferAtomicAnd32:
    case Opcode::BufferAtomicOr32:
    case Opcode::BufferAtomicXor32:
    case Opcode::BufferAtomicSwap32:
    case Opcode::BufferAtomicFCmpSwapRaw32:
    case Opcode::BufferAtomicCmpSwap32:
    case Opcode::BufferAtomicFCmpSwap32:
        return {1, 2, false};
    case Opcode::LoadBufferU32x2:
    case Opcode::StoreBufferU32x2:
        return {2, 2, false};
    case Opcode::LoadBufferU32x3:
    case Opcode::StoreBufferU32x3:
        return {3, 2, false};
    case Opcode::LoadBufferU32x4:
    case Opcode::StoreBufferU32x4:
        return {4, 2, false};
    case Opcode::LoadBufferF32:
    case Opcode::StoreBufferF32:
        return {1, 2, true};
    case Opcode::LoadBufferF32x2:
    case Opcode::StoreBufferF32x2:
        return {2, 2, true};
    case Opcode::LoadBufferF32x3:
    case Opcode::StoreBufferF32x3:
        return {3, 2, true};
    case Opcode::LoadBufferF32x4:
    case Opcode::StoreBufferF32x4:
        return {4, 2, true};
    default:
        return {};
    }
}

struct BufferByteAddress {
    Shader::IR::U64 offset;
    Shader::IR::U64 address;
};

BufferByteAddress GetBufferByteAddress(Shader::IR::IREmitter& ir, const Shader::IR::Inst& inst,
                                       const AmdGpu::Buffer& buffer, const Shader::IR::U32& index) {
    namespace IR = Shader::IR;
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    const auto wide = [&](const IR::U32& value) { return IR::U64{ir.UConvert(64, value)}; };
    IR::U64 offset = ir.Imm64(u64{flags.inst_offset.Value()});
    if (flags.voffset_enable) {
        offset = ir.IAdd(offset, wide(IR::GetBufferVOffsetArg(&inst)));
    }
    IR::U64 address;
    if (buffer.swizzle_enable) {
        const auto element_mask = ir.Imm64(u64{buffer.GetElementSize() - 1});
        const auto index_msb = ir.ShiftRightLogical(index, ir.Imm32(3 + buffer.index_stride));
        const auto high = ir.IMul(ir.IAdd(ir.IMul(wide(index_msb), ir.Imm64(u64{buffer.stride})),
                                          ir.BitwiseAnd(offset, ir.Imm64(~element_mask.U64()))),
                                  ir.Imm64(u64{buffer.GetIndexStride()}));
        const auto low = ir.IAdd(
            wide(ir.ShiftLeftLogical(ir.BitwiseAnd(index, ir.Imm32(buffer.GetIndexStride() - 1)),
                                     ir.Imm32(1 + buffer.element_size))),
            ir.BitwiseAnd(offset, element_mask));
        address = ir.IAdd(high, low);
    } else {
        address = ir.IAdd(ir.IMul(wide(index), ir.Imm64(u64{buffer.stride})), offset);
    }
    address = ir.IAdd(address, wide(IR::GetBufferSOffsetArg(&inst)));
    return {offset, address};
}

void PatchFormattedBufferAccess(Shader::IR::Inst& inst, Shader::Info& info,
                                const Shader::BufferResource& resource,
                                const AmdGpu::Buffer& buffer, const Shader::IR::U32& index,
                                AmdGpu::DataFormat format) {
    namespace IR = Shader::IR;
    switch (format) {
    case AmdGpu::DataFormat::Format8:
    case AmdGpu::DataFormat::Format8_8:
    case AmdGpu::DataFormat::Format16:
    case AmdGpu::DataFormat::Format8_8_8_8:
    case AmdGpu::DataFormat::Format16_16:
    case AmdGpu::DataFormat::Format10_11_11:
    case AmdGpu::DataFormat::Format2_10_10_10:
    case AmdGpu::DataFormat::Format16_16_16_16:
    case AmdGpu::DataFormat::Format32:
    case AmdGpu::DataFormat::Format32_32:
    case AmdGpu::DataFormat::Format32_32_32:
    case AmdGpu::DataFormat::Format32_32_32_32:
        break;
    default:
        throw std::runtime_error("Metal formatted buffer uses an unsupported data format");
    }
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    const auto wide = [&](const IR::U32& value) { return IR::U64{ir.UConvert(64, value)}; };
    const auto [offset, address] = GetBufferByteAddress(ir, inst, buffer, index);
    const auto count = wide(BufferDescriptorWord(ir, resource, 2));
    auto valid = buffer.stride ? ir.ILessThan(wide(index), count, false)
                               : ir.ILessThan(address, count, false);
    if (buffer.stride && (flags.index_enable || buffer.add_tid_enable)) {
        valid = ir.LogicalAnd(valid, ir.ILessThan(offset, ir.Imm64(u64{buffer.stride}), false));
    }
    const u32 shift = Shader::Optimization::GetBufferAddressShift(inst, format);
    const auto base_low =
        wide(ir.BitwiseAnd(BufferDescriptorWord(ir, resource, 0), ir.Imm32((1u << shift) - 1)));
    const auto aligned =
        ir.BitwiseAnd(ir.IAdd(address, base_low), ir.Imm64(~u64{(1u << shift) - 1}));
    const auto capacity = wide(
        ir.ReadConstBuffer(ir.Imm32(BufferBoundsHandle(info)), ir.Imm32(inst.Arg(0).U32()), {}));
    const u32 bytes = AmdGpu::NumBitsPerBlock(format) / 8;
    valid = ir.LogicalAnd(valid, ir.ILessThan(ir.IAdd(aligned, ir.Imm64(u64{bytes - 1})),
                                              ir.IAdd(capacity, base_low), false));
    const auto words = IR::U64{ir.ShiftRightLogical(aligned, ir.Imm32(shift))};
    const auto guarded =
        ir.GuardBufferAddress(IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(words), 0)}, valid);
    const auto guest_address = GuestPageBufferAddress(ir, info, inst.Arg(0).U32(), aligned, shift);
    if (inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32) {
        inst.SetArg(IR::StoreBufferArgs::Address, guarded);
        inst.SetArg(3, guest_address);
        return;
    }
    const auto value = ir.LoadBufferFormat(inst.Arg(0), guarded, flags, guest_address);
    const bool instruction_format = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid;
    const bool integer_format =
        AmdGpu::IsInteger(static_cast<AmdGpu::NumberFormat>(buffer.num_format));
    const auto swizzle = buffer.DstSelect();
    std::array<IR::Value, 4> components;
    for (u32 i = 0; i < components.size(); ++i) {
        auto component = ir.CompositeExtract(value, i);
        if (instruction_format && i >= AmdGpu::NumComponents(format)) {
            component = ir.Imm32(0.f);
        } else if (!instruction_format && integer_format &&
                   swizzle.array[i] == AmdGpu::CompSwizzle::One) {
            components[i] =
                ir.BitCast<IR::F32>(IR::U32{ir.Select(valid, ir.Imm32(1u), ir.Imm32(0u))});
            continue;
        }
        components[i] = ir.Select(valid, component, ir.Imm32(0.f));
    }
    inst.ReplaceUsesWith(ir.CompositeConstruct(components));
}

void PatchRawBufferAccess(Shader::IR::Inst& inst, Shader::Info& info,
                          const Shader::BufferResource& resource, const AmdGpu::Buffer& buffer,
                          const Shader::IR::U32& index, RawBufferAccess access) {
    namespace IR = Shader::IR;
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    const auto wide = [&](const IR::U32& value) { return IR::U64{ir.UConvert(64, value)}; };
    const auto [offset, address] = GetBufferByteAddress(ir, inst, buffer, index);
    const auto count = wide(BufferDescriptorWord(ir, resource, 2));
    const u32 handle = inst.Arg(0).U32();
    if (access.floating) {
        info.buffers[handle].used_types |= IR::Type::U32 | IR::Type::F32;
    }
    const auto capacity =
        wide(ir.ReadConstBuffer(ir.Imm32(BufferBoundsHandle(info)), ir.Imm32(handle), {}));
    const auto guest_valid_for = [&](const IR::U64& component_address, const IR::U64& delta) {
        if (!buffer.stride) {
            return ir.ILessThan(component_address, count, false);
        }
        auto valid = ir.ILessThan(wide(index), count, false);
        if (flags.index_enable || buffer.add_tid_enable) {
            valid = ir.LogicalAnd(
                valid, ir.ILessThan(ir.IAdd(offset, delta), ir.Imm64(u64{buffer.stride}), false));
        }
        return valid;
    };
    if (access.shift == 1) {
        info.buffers[handle].used_types |= IR::Type::U8 | IR::Type::U16;
        const auto base_low =
            wide(ir.BitwiseAnd(BufferDescriptorWord(ir, resource, 0), ir.Imm32(1u)));
        const auto adjusted = ir.IAdd(address, base_low);
        const auto aligned = ir.IEqual(ir.BitwiseAnd(adjusted, ir.Imm64(u64{1})), ir.Imm64(u64{0}));
        const auto valid =
            ir.LogicalAnd(guest_valid_for(address, ir.Imm64(u64{0})),
                          ir.ILessThan(ir.IAdd(address, ir.Imm64(u64{1})), capacity, false));
        const auto low32 = [&](const IR::U64& value) {
            return IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(value), 0)};
        };
        const auto word_address =
            ir.GuardBufferAddress(low32(IR::U64{ir.ShiftRightLogical(adjusted, ir.Imm32(1u))}),
                                  ir.LogicalAnd(valid, aligned));
        const auto bytes_valid = ir.LogicalAnd(valid, ir.LogicalNot(aligned));
        const auto low_address = ir.GuardBufferAddress(low32(address), bytes_valid);
        const auto high_address =
            ir.GuardBufferAddress(low32(ir.IAdd(address, ir.Imm64(u64{1}))), bytes_valid);
        const auto word_guest = GuestPageBufferAddress(ir, info, handle, adjusted, 1);
        const auto low_guest = GuestPageBufferAddress(ir, info, handle, address, 0);
        const auto high_guest =
            GuestPageBufferAddress(ir, info, handle, ir.IAdd(address, ir.Imm64(u64{1})), 0);
        if (inst.Type() == IR::Type::Void) {
            const IR::U16 data{inst.Arg(IR::StoreBufferArgs::Data)};
            const IR::U32 bits{ir.UConvert(32, data)};
            ir.StoreBufferU16(inst.Arg(0), word_address, data, flags, word_guest);
            ir.StoreBufferU8(inst.Arg(0), low_address, IR::U8{ir.UConvert(8, bits)}, flags,
                             low_guest);
            ir.StoreBufferU8(
                inst.Arg(0), high_address,
                IR::U8{ir.UConvert(8, IR::U32{ir.ShiftRightLogical(bits, ir.Imm32(8u))})}, flags,
                high_guest);
            inst.Invalidate();
        } else {
            const IR::U32 word{
                ir.UConvert(32, ir.LoadBufferU16(inst.Arg(0), word_address, flags, word_guest))};
            const IR::U32 low{
                ir.UConvert(32, ir.LoadBufferU8(inst.Arg(0), low_address, flags, low_guest))};
            const IR::U32 high{
                ir.UConvert(32, ir.LoadBufferU8(inst.Arg(0), high_address, flags, high_guest))};
            const auto packed = ir.BitwiseOr(low, ir.ShiftLeftLogical(high, ir.Imm32(8u)));
            inst.ReplaceUsesWith(ir.UConvert(16, ir.BitwiseOr(word, packed)));
        }
        return;
    }
    const IR::U64 base_low = access.shift
                                 ? wide(ir.BitwiseAnd(BufferDescriptorWord(ir, resource, 0),
                                                      ir.Imm32((1u << access.shift) - 1)))
                                 : ir.Imm64(u64{0});
    const auto aligned_capacity = ir.IAdd(capacity, base_low);
    std::array<IR::Value, 4> components;
    for (u32 component = 0; component < access.components; ++component) {
        const auto delta = ir.Imm64(u64{component} << access.shift);
        const auto component_address = ir.IAdd(address, delta);
        const auto guest_valid = guest_valid_for(component_address, delta);
        const auto aligned_address = ir.BitwiseAnd(ir.IAdd(component_address, base_low),
                                                   ir.Imm64(~u64{(1u << access.shift) - 1}));
        const auto host_valid =
            ir.ILessThan(ir.IAdd(aligned_address, ir.Imm64(u64{(1u << access.shift) - 1})),
                         aligned_capacity, false);
        const auto word_address = ir.ShiftRightLogical(aligned_address, ir.Imm32(access.shift));
        const auto guarded = ir.GuardBufferAddress(
            IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(IR::U64{word_address}), 0)},
            ir.LogicalAnd(guest_valid, host_valid));
        const auto page_argument = GuestPageAddressArgument(inst.GetOpcode());
        const IR::Value guest_address =
            page_argument ? IR::Value{GuestPageBufferAddress(
                                ir, info, handle, IR::U64{aligned_address}, access.shift)}
                          : IR::Value{};
        if (access.components == 1) {
            inst.SetArg(IR::LoadBufferArgs::Address, guarded);
            if (page_argument) {
                inst.SetArg(*page_argument, guest_address);
                if (Shader::Optimization::IsBufferAtomic(inst)) {
                    info.guest_page_atomics[inst.GetOpcode()] |= u8{1} << (buffer.mtype == 3);
                }
            }
            return;
        }
        if (inst.Type() == IR::Type::Void) {
            const auto data = ir.CompositeExtract(inst.Arg(IR::StoreBufferArgs::Data), component);
            if (access.floating) {
                ir.StoreBufferF32(1, inst.Arg(0), guarded, data, flags, guest_address);
            } else {
                ir.StoreBufferU32(1, inst.Arg(0), guarded, data, flags, guest_address);
            }
        } else {
            components[component] =
                access.floating ? ir.LoadBufferF32(1, inst.Arg(0), guarded, flags, guest_address)
                                : ir.LoadBufferU32(1, inst.Arg(0), guarded, flags, guest_address);
        }
    }
    if (inst.Type() != IR::Type::Void) {
        inst.ReplaceUsesWith(ir.CompositeConstruct(std::span{components}.first(access.components)));
    } else {
        inst.Invalidate();
    }
}

bool PatchGuestBufferAddress(Shader::IR::Inst& inst, Shader::Info& info) {
    namespace IR = Shader::IR;
    const u32 handle = inst.Arg(0).U32();
    const auto resource = info.buffers.at(handle);
    if (resource.buffer_type != Shader::BufferType::Guest) {
        return false;
    }
    const auto buffer = ReadGuestBufferSharp(resource, info);
    if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
        IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
        const IR::U32 index{inst.Arg(1)};
        const IR::U64 byte_offset =
            ir.ShiftLeftLogical(IR::U64{ir.UConvert(64, index)}, ir.Imm32(2u));
        const IR::U64 base_low{
            ir.UConvert(64, ir.BitwiseAnd(BufferDescriptorWord(ir, resource, 0), ir.Imm32(3u)))};
        const IR::U64 capacity{ir.UConvert(
            64, ir.ReadConstBuffer(ir.Imm32(BufferBoundsHandle(info)), ir.Imm32(handle), {}))};
        const auto valid = ir.ILessThan(ir.IAdd(byte_offset, ir.Imm64(u64{3})),
                                        ir.IAdd(capacity, base_low), false);
        inst.SetArg(1, ir.GuardBufferAddress(index, valid));
        inst.SetArg(2, GuestPageBufferAddress(ir, info, handle, byte_offset));
        return true;
    }
    const auto access = GetRawBufferAccess(inst.GetOpcode());
    const bool formatted = inst.GetOpcode() == IR::Opcode::LoadBufferFormatF32 ||
                           inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32;
    if (!buffer.swizzle_enable && !buffer.add_tid_enable && !access.components && !formatted) {
        return false;
    }
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    IR::U32 index = flags.index_enable ? IR::GetBufferIndexArg(&inst) : ir.Imm32(0u);
    if (buffer.add_tid_enable) {
        if (info.sw_stage != Shader::SwStage::Compute) {
            throw std::runtime_error("Metal thread ID buffer addressing requires compute");
        }
        const auto invocation = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
        index = ir.IAdd(index, ir.BitwiseAnd(invocation, ir.Imm32(63u)));
    }
    if (formatted) {
        const auto format = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid
                                ? AmdGpu::RemapDataFormat(flags.inst_data_fmt.Value())
                                : buffer.GetDataFmt();
        if (format != AmdGpu::DataFormat::FormatInvalid) {
            PatchFormattedBufferAccess(inst, info, resource, buffer, index, format);
            return true;
        }
    }
    if (access.components) {
        PatchRawBufferAccess(inst, info, resource, buffer, index, access);
        return true;
    }
    IR::U32 offset = ir.Imm32(static_cast<u32>(flags.inst_offset.Value()));
    if (flags.voffset_enable) {
        offset = ir.IAdd(offset, IR::GetBufferVOffsetArg(&inst));
    }
    IR::U32 address;
    if (buffer.swizzle_enable) {
        const auto index_stride = ir.Imm32(buffer.GetIndexStride());
        const auto element_size = ir.Imm32(buffer.GetElementSize());
        const IR::U32 index_msb{ir.IDiv(index, index_stride)};
        const IR::U32 index_lsb{ir.IMod(index, index_stride)};
        const IR::U32 offset_msb{ir.IDiv(offset, element_size)};
        const IR::U32 offset_lsb{ir.IMod(offset, element_size)};
        const auto high = ir.IMul(
            ir.IAdd(ir.IMul(index_msb, ir.Imm32(buffer.stride)), ir.IMul(offset_msb, element_size)),
            index_stride);
        const auto low = ir.IAdd(ir.IMul(index_lsb, element_size), offset_lsb);
        address = ir.IAdd(high, low);
    } else {
        address = ir.IAdd(ir.IMul(index, ir.Imm32(buffer.stride)), offset);
    }
    address = ir.IAdd(address, IR::GetBufferSOffsetArg(&inst));
    const auto data_format = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid
                                 ? AmdGpu::RemapDataFormat(flags.inst_data_fmt.Value())
                                 : buffer.GetDataFmt();
    const auto shift = Shader::Optimization::GetBufferAddressShift(inst, data_format);
    if (shift) {
        address = ir.ShiftRightLogical(address, ir.Imm32(shift));
    }
    inst.SetArg(IR::LoadBufferArgs::Address, address);
    return true;
}

std::bitset<Shader::NUM_BUFFERS> GuardedBuffers(const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_BUFFERS> seen;
    std::bitset<Shader::NUM_BUFFERS> unguarded;
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (!Shader::Optimization::IsBufferInstruction(inst)) {
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if (program.info.buffers.at(handle).buffer_type != Shader::BufferType::Guest) {
                continue;
            }
            seen.set(handle);
            const auto* address = inst.Arg(1).TryInst();
            if (!address || address->GetOpcode() != IR::Opcode::GuardBufferAddress) {
                unguarded.set(handle);
            }
        }
    }
    return seen & ~unguarded;
}

std::bitset<Shader::NUM_BUFFERS> PagedGuestBuffers(const Shader::IR::Program& program) {
    std::bitset<Shader::NUM_BUFFERS> buffers;
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (!Shader::Optimization::IsBufferInstruction(inst) ||
                program.info.buffers.at(inst.Arg(0).U32()).buffer_type !=
                    Shader::BufferType::Guest) {
                continue;
            }
            const auto argument = GuestPageAddressArgument(inst.GetOpcode());
            if (!argument || inst.Arg(*argument).IsEmpty()) {
                throw std::runtime_error("Metal guest buffer access requires page-table lowering");
            }
            buffers.set(inst.Arg(0).U32());
        }
    }
    return buffers;
}

void CollectBufferDescriptorFormatUsage(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    for (auto& resource : program.info.buffers) {
        if (resource.buffer_type == Shader::BufferType::Guest) {
            resource.is_formatted = false;
        }
    }
    for (const IR::Block* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if ((inst.GetOpcode() != IR::Opcode::LoadBufferFormatF32 &&
                 inst.GetOpcode() != IR::Opcode::StoreBufferFormatF32) ||
                inst.Flags<IR::BufferInstInfo>().inst_data_fmt !=
                    AmdGpu::DataFormat::FormatInvalid) {
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if (handle >= program.info.buffers.size()) {
                throw std::runtime_error("Invalid Metal formatted buffer handle");
            }
            auto& resource = program.info.buffers[handle];
            if (resource.buffer_type == Shader::BufferType::Guest) {
                resource.is_formatted = true;
            }
        }
    }
}

void LowerUnboundBufferAccesses(Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    CollectBufferDescriptorFormatUsage(program);
    const auto unbound = ShaderCompiler::UnboundBuffers(program.info);
    if (unbound.none()) {
        return;
    }
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            if (!Shader::Optimization::IsBufferInstruction(inst) ||
                inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if (handle >= program.info.buffers.size()) {
                throw std::runtime_error("Invalid Metal shader buffer handle");
            }
            if (!unbound[handle]) {
                continue;
            }
            IR::IREmitter ir{*block, it};
            IR::Value zero;
            switch (inst.Type()) {
            case IR::Type::Void:
                inst.Invalidate();
                continue;
            case IR::Type::U8:
                zero = ir.Imm8(0);
                break;
            case IR::Type::U16:
                zero = ir.Imm16(0);
                break;
            case IR::Type::U32:
                zero = ir.Imm32(0u);
                break;
            case IR::Type::U64:
                zero = ir.Imm64(u64{0});
                break;
            case IR::Type::F32:
                zero = ir.Imm32(0.f);
                break;
            case IR::Type::U32x2:
                zero = ir.CompositeConstruct(ir.Imm32(0u), ir.Imm32(0u));
                break;
            case IR::Type::U32x3:
                zero = ir.CompositeConstruct(ir.Imm32(0u), ir.Imm32(0u), ir.Imm32(0u));
                break;
            case IR::Type::U32x4:
                zero =
                    ir.CompositeConstruct(ir.Imm32(0u), ir.Imm32(0u), ir.Imm32(0u), ir.Imm32(0u));
                break;
            case IR::Type::F32x2:
                zero = ir.CompositeConstruct(ir.Imm32(0.f), ir.Imm32(0.f));
                break;
            case IR::Type::F32x3:
                zero = ir.CompositeConstruct(ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f));
                break;
            case IR::Type::F32x4:
                zero = ir.CompositeConstruct(ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f),
                                             ir.Imm32(0.f));
                break;
            default:
                throw std::runtime_error("Unsupported Metal unbound buffer result type");
            }
            inst.ReplaceUsesWithAndRemove(zero);
        }
    }
}

void LowerSharedMemoryAddresses(const Shader::IR::Program& program,
                                const TessellationKernelConfig* tessellation,
                                TessellationKernelBindings* tessellation_bindings) {
    namespace IR = Shader::IR;
    if (tessellation && tessellation_bindings && tessellation_bindings->lds) {
        if (program.info.buffers.size() == Shader::NUM_BUFFERS) {
            throw std::runtime_error("Metal tessellation LDS guard exceeds buffer capacity");
        }
        tessellation_bindings->errors = u32(program.info.buffers.size());
        program.info.buffers.push_back(Shader::BufferResource{
            .used_types = IR::Type::U32,
            .buffer_type = Shader::BufferType::GeometryEmissions,
            .is_written = true,
        });
    }
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            u32 shift;
            switch (inst.GetOpcode()) {
            case IR::Opcode::LoadBufferU16:
            case IR::Opcode::StoreBufferU16:
                shift = 1;
                break;
            case IR::Opcode::LoadBufferU32:
            case IR::Opcode::StoreBufferU32:
            case IR::Opcode::BufferAtomicIAdd32:
            case IR::Opcode::BufferAtomicISub32:
            case IR::Opcode::BufferAtomicRSub32:
            case IR::Opcode::BufferAtomicMaskOr32:
            case IR::Opcode::BufferAtomicWrap32:
            case IR::Opcode::BufferAtomicSMin32:
            case IR::Opcode::BufferAtomicUMin32:
            case IR::Opcode::BufferAtomicSMax32:
            case IR::Opcode::BufferAtomicUMax32:
            case IR::Opcode::BufferAtomicInc32:
            case IR::Opcode::BufferAtomicDec32:
            case IR::Opcode::BufferAtomicAnd32:
            case IR::Opcode::BufferAtomicOr32:
            case IR::Opcode::BufferAtomicXor32:
            case IR::Opcode::BufferAtomicSwap32:
            case IR::Opcode::BufferAtomicSrc2_32:
            case IR::Opcode::BufferAtomicFCmpSwapRaw32:
            case IR::Opcode::BufferAtomicCmpSwap32:
                shift = 2;
                break;
            case IR::Opcode::LoadBufferU64:
            case IR::Opcode::StoreBufferU64:
            case IR::Opcode::BufferAtomicISub64:
            case IR::Opcode::BufferAtomicAnd64:
            case IR::Opcode::BufferAtomicOr64:
            case IR::Opcode::BufferAtomicXor64:
            case IR::Opcode::BufferAtomicSwap64:
            case IR::Opcode::BufferAtomicSrc2_64:
            case IR::Opcode::BufferAtomicRSub64:
            case IR::Opcode::BufferAtomicMaskOr64:
            case IR::Opcode::BufferAtomicInc64:
            case IR::Opcode::BufferAtomicDec64:
            case IR::Opcode::BufferAtomicFCmpSwapRaw64:
            case IR::Opcode::BufferAtomicCmpSwap64:
            case IR::Opcode::BufferAtomicIAdd64:
            case IR::Opcode::BufferAtomicSMin64:
            case IR::Opcode::BufferAtomicUMin64:
            case IR::Opcode::BufferAtomicSMax64:
            case IR::Opcode::BufferAtomicUMax64:
                shift = 3;
                break;
            default:
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if (handle >= program.info.buffers.size() ||
                program.info.buffers[handle].buffer_type != Shader::BufferType::SharedMemory) {
                continue;
            }
            IR::IREmitter ir{*block, it};
            const bool source_memory = inst.GetOpcode() == IR::Opcode::BufferAtomicSrc2_32 ||
                                       inst.GetOpcode() == IR::Opcode::BufferAtomicSrc2_64;
            auto address = ir.ShiftRightLogical(IR::U32{inst.Arg(1)}, ir.Imm32(shift));
            if (tessellation && tessellation_bindings && tessellation_bindings->lds == handle) {
                const auto group = ir.GetAttributeU32(IR::Attribute::WorkgroupIndex);
                const auto group_base = ir.IMul(group, ir.Imm32(tessellation->lds_bytes));
                const u32 bytes = 1u << shift;
                const auto in_range = [&](const IR::U32& byte_address) {
                    const auto offset = ir.ISub(byte_address, group_base);
                    return ir.LogicalAnd(
                        ir.ILessThanEqual(offset, ir.Imm32(tessellation->lds_bytes - bytes), false),
                        ir.IEqual(ir.BitwiseAnd(offset, ir.Imm32(bytes - 1)), ir.Imm32(0u)));
                };
                auto valid =
                    ir.LogicalAnd(ir.ILessThan(group, ir.Imm32(tessellation->workgroups), false),
                                  in_range(IR::U32{inst.Arg(1)}));
                if (source_memory) {
                    valid = ir.LogicalAnd(valid, in_range(IR::U32{inst.Arg(2)}));
                }
                static_cast<void>(ir.BufferAtomicOr(
                    ir.Imm32(*tessellation_bindings->errors),
                    ir.GuardBufferAddress(ir.Imm32(0u), ir.LogicalNot(valid)), ir.Imm32(1u), {}));
                address = ir.GuardBufferAddress(IR::U32{address}, valid);
            }
            inst.SetArg(1, address);
            if (source_memory) {
                inst.SetArg(2, ir.ShiftRightLogical(IR::U32{inst.Arg(2)}, ir.Imm32(shift)));
            }
        }
    }
}

void LowerComputeDispatchArguments(const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    if (program.info.sw_stage != Shader::SwStage::Compute ||
        (!program.info.loads.GetAny(IR::Attribute::WorkgroupId) &&
         !program.info.loads.GetAny(IR::Attribute::GeometryInitialActiveCount))) {
        return;
    }
    auto& buffers = program.info.buffers;
    if (buffers.size() == buffers.capacity()) {
        throw std::runtime_error("Metal compute dispatch base exceeds shader resource capacity");
    }
    const u32 binding = buffers.size();
    buffers.push_back(Shader::BufferResource{
        .used_types = IR::Type::U32,
        .buffer_type = Shader::BufferType::DispatchBase,
    });
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            if (inst.GetOpcode() != IR::Opcode::GetAttributeU32) {
                continue;
            }
            const auto attribute = inst.Arg(0).Attribute();
            if (attribute != IR::Attribute::WorkgroupId &&
                attribute != IR::Attribute::GeometryInitialActiveCount) {
                continue;
            }
            const u32 component = inst.Arg(1).U32();
            IR::IREmitter ir{*block, it};
            if (attribute == IR::Attribute::GeometryInitialActiveCount) {
                if (component) {
                    throw std::runtime_error("Invalid Metal geometry initial count component");
                }
                inst.ReplaceUsesWithAndRemove(
                    ir.ReadConstBuffer(ir.Imm32(binding), ir.Imm32(3u), {}));
                continue;
            }
            if (component >= 3) {
                throw std::runtime_error("Invalid Metal compute workgroup ID component");
            }
            const auto local = ir.GetAttributeU32(IR::Attribute::WorkgroupId, component);
            const auto base = ir.ReadConstBuffer(ir.Imm32(binding), ir.Imm32(component), {});
            inst.ReplaceUsesWith(ir.IAdd(local, base));
        }
    }
}

void LowerFragmentSampleMask(const Shader::IR::Program& program) {
    namespace IR = Shader::IR;
    if (program.info.sw_stage != Shader::SwStage::Fragment) {
        throw std::runtime_error("Metal sample mask lowering requires a fragment shader");
    }
    const bool exports_mask = program.info.stores.Get(IR::Attribute::SampleMask);
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            const bool mask = inst.GetOpcode() == IR::Opcode::SetAttribute &&
                              inst.Arg(0).Attribute() == IR::Attribute::SampleMask;
            if (!mask && (exports_mask || inst.GetOpcode() != IR::Opcode::Epilogue)) {
                continue;
            }
            IR::IREmitter ir{*block, it};
            IR::U32 coverage{
                ir.ShiftLeftLogical(ir.Imm32(1u), ir.GetAttributeU32(IR::Attribute::SampleIndex))};
            if (mask) {
                coverage =
                    IR::U32{ir.BitwiseAnd(coverage, ir.BitCast<IR::U32>(IR::F32{inst.Arg(1)}))};
                inst.SetArg(1, ir.BitCast<IR::F32>(coverage));
            } else {
                ir.SetAttribute(IR::Attribute::SampleMask, ir.BitCast<IR::F32>(coverage));
            }
        }
    }
    program.info.loads.Set(IR::Attribute::SampleIndex);
    program.info.stores.Set(IR::Attribute::SampleMask);
}

std::bitset<Shader::NUM_IMAGES> LowerGatherLod(
    Shader::IR::Program& program, std::bitset<Shader::NUM_IMAGES>& base_mip,
    std::array<SamplerLodUsage, Shader::NUM_SAMPLERS>& sampler_lod) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_IMAGES> single_mip;
    std::map<u32, u32> base_views;
    for (IR::Block* block : program.blocks) {
        for (auto& inst : *block) {
            const auto opcode = inst.GetOpcode();
            if (opcode != IR::Opcode::ImageGather && opcode != IR::Opcode::ImageGatherDref) {
                continue;
            }
            const u32 lod_argument = opcode == IR::Opcode::ImageGather ? 3 : 4;
            if (inst.Arg(lod_argument).IsEmpty()) {
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            const u32 image_index = handle & 0xffff;
            if (image_index >= program.info.images.size() || image_index >= single_mip.size()) {
                throw std::runtime_error("Invalid Metal gather image handle");
            }
            const auto image = program.info.images[image_index].GetSharp(program.info);
            if (image.base_level != image.last_level) {
                const u32 sampler_index = handle >> 16;
                if (sampler_index >= program.info.samplers.size() ||
                    sampler_index >= sampler_lod.size()) {
                    throw std::runtime_error("Invalid Metal gather sampler handle");
                }
                const auto sampler = program.info.samplers[sampler_index].GetSharp(program.info);
                const auto& resource = program.info.images[image_index];
                if (!inst.Flags<IR::TextureInstInfo>().force_level0 || image.base_level ||
                    image.min_lod || sampler.LodBias() != 0.f || sampler.MinLod() != 0.f ||
                    !sampler.Valid() || resource.is_written ||
                    resource.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                    throw std::runtime_error("Metal gather LOD requires native mip-view lowering");
                }
                auto found = base_views.find(image_index);
                if (found == base_views.end()) {
                    if (program.info.images.size() >= Shader::NUM_IMAGES) {
                        throw std::runtime_error("Metal gather mip views exceed image capacity");
                    }
                    auto view = resource;
                    view.mip_fallback_mode = Shader::MipStorageFallbackMode::ConstantIndex;
                    view.constant_mip_index = 0;
                    const u32 index = program.info.images.size();
                    program.info.images.push_back(view);
                    found = base_views.emplace(image_index, index).first;
                }
                const u32 view_index = found->second;
                base_mip.set(view_index);
                sampler_lod[sampler_index].bias = true;
                sampler_lod[sampler_index].clamp = true;
                inst.SetArg(0, IR::Value{(handle & 0xffff0000) | view_index});
                inst.SetArg(lod_argument, IR::Value{});
                continue;
            }
            single_mip.set(image_index);
            inst.SetArg(lod_argument, IR::Value{});
        }
    }
    if (single_mip.any() || base_mip.any()) {
        Shader::Optimization::DeadCodeEliminationPass(program);
    }
    return single_mip;
}

void LowerSamplerLod(const Shader::IR::Program& program,
                     std::array<SamplerLodUsage, Shader::NUM_SAMPLERS>& sampler_lod,
                     bool native_sampler_lod_bias = false,
                     const std::bitset<Shader::NUM_IMAGES>& single_mip_gathers = {}) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_SAMPLERS> shader_bias_samplers;
    if (native_sampler_lod_bias) {
        for (const IR::Block* block : program.blocks) {
            for (const auto& inst : *block) {
                switch (inst.GetOpcode()) {
                case IR::Opcode::ImageSampleImplicitLod:
                case IR::Opcode::ImageSampleDrefImplicitLod:
                case IR::Opcode::ImageSampleExplicitLod:
                case IR::Opcode::ImageSampleDrefExplicitLod:
                case IR::Opcode::ImageGather:
                case IR::Opcode::ImageGatherDref: {
                    const u32 index = inst.Arg(0).U32() >> 16;
                    if (index >= shader_bias_samplers.size()) {
                        throw std::runtime_error("Invalid Metal shader sampler handle");
                    }
                    shader_bias_samplers.set(index);
                    break;
                }
                default:
                    break;
                }
            }
        }
    }
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            const auto opcode = inst.GetOpcode();
            switch (opcode) {
            case IR::Opcode::ImageSampleImplicitLod:
            case IR::Opcode::ImageSampleDrefImplicitLod:
            case IR::Opcode::ImageSampleExplicitLod:
            case IR::Opcode::ImageSampleDrefExplicitLod:
            case IR::Opcode::ImageSampleDrefGradient:
            case IR::Opcode::ImageGradient:
            case IR::Opcode::ImageGather:
            case IR::Opcode::ImageGatherDref:
            case IR::Opcode::ImageQueryLod:
                break;
            default:
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if ((opcode == IR::Opcode::ImageGather || opcode == IR::Opcode::ImageGatherDref) &&
                single_mip_gathers.test(handle & 0xffff)) {
                continue;
            }
            const u32 sampler_index = handle >> 16;
            if (sampler_index >= program.info.samplers.size() ||
                sampler_index >= sampler_lod.size()) {
                throw std::runtime_error("Invalid Metal shader sampler handle");
            }
            const auto sampler = program.info.samplers[sampler_index].GetSharp(program.info);
            const float bias = sampler.LodBias();
            auto& usage = sampler_lod[sampler_index];
            const bool gradient = opcode == IR::Opcode::ImageGradient ||
                                  opcode == IR::Opcode::ImageSampleDrefGradient;
            const bool native_bias =
                native_sampler_lod_bias && (gradient || opcode == IR::Opcode::ImageQueryLod) &&
                !shader_bias_samplers.test(sampler_index) && NativeSamplerLodBiasCompatible(bias);
            usage.native_bias |= native_bias;
            usage.bias |= !native_bias;
            usage.clamp |= opcode == IR::Opcode::ImageSampleExplicitLod ||
                           opcode == IR::Opcode::ImageSampleDrefExplicitLod;
            if (native_bias && gradient) {
                continue;
            }
            usage.anisotropic |= gradient && bias != 0.f;
            IR::IREmitter ir{*block, it};
            if (opcode == IR::Opcode::ImageQueryLod &&
                (sampler.force_unnormalized || inst.Flags<IR::TextureInstInfo>().is_unnormalized)) {
                const u32 image_index = handle & 0xffff;
                if (image_index >= program.info.images.size()) {
                    throw std::runtime_error("Invalid Metal LOD query image handle");
                }
                const auto& resource = program.info.images[image_index];
                const auto image = resource.GetSharp(program.info);
                const auto type = image.GetViewType(resource.is_array);
                const u32 dimensions =
                    type == AmdGpu::ImageType::Color1D || type == AmdGpu::ImageType::Color1DArray
                        ? 1
                    : type == AmdGpu::ImageType::Color3D ? 3
                                                         : 2;
                const u32 components =
                    type == AmdGpu::ImageType::Color1DArray ? 2
                    : type == AmdGpu::ImageType::Color2DArray || type == AmdGpu::ImageType::Cube
                        ? 3
                        : dimensions;
                const auto sizes =
                    ir.ImageQueryDimension(ir.Imm32(image_index), ir.Imm32(0u), ir.Imm1(false),
                                           inst.Flags<IR::TextureInstInfo>());
                const auto original = inst.Arg(1);
                std::array<IR::Value, 3> normalized;
                for (u32 component = 0; component < components; ++component) {
                    const auto coordinate =
                        components == 1 ? original : ir.CompositeExtract(original, component);
                    normalized[component] =
                        component < dimensions
                            ? IR::Value{ir.FPDiv(
                                  IR::F32{coordinate},
                                  ir.ConvertUToF(32, 32,
                                                 IR::U32{ir.CompositeExtract(sizes, component)}))}
                            : coordinate;
                }
                inst.SetArg(
                    1, components == 1 ? normalized[0]
                       : components == 2
                           ? ir.CompositeConstruct(normalized[0], normalized[1])
                           : ir.CompositeConstruct(normalized[0], normalized[1], normalized[2]));
            }
            if (native_bias) {
                continue;
            }
            const bool depth = opcode == IR::Opcode::ImageSampleDrefImplicitLod ||
                               opcode == IR::Opcode::ImageSampleDrefExplicitLod;
            const u32 lod_argument = depth ? 3 : 2;
            if (opcode == IR::Opcode::ImageSampleExplicitLod ||
                opcode == IR::Opcode::ImageSampleDrefExplicitLod) {
                IR::F32 lod{inst.Arg(lod_argument)};
                if (bias != 0.f) {
                    lod = IR::F32{ir.FPAdd(lod, ir.Imm32(bias))};
                }
                inst.SetArg(lod_argument, ir.FPClamp(lod, ir.Imm32(sampler.MinLod()),
                                                     ir.Imm32(sampler.MaxLod())));
            } else if (opcode == IR::Opcode::ImageSampleImplicitLod ||
                       opcode == IR::Opcode::ImageSampleDrefImplicitLod) {
                if (bias != 0.f) {
                    const auto original = inst.Arg(lod_argument);
                    inst.SetArg(lod_argument,
                                original.IsEmpty()
                                    ? IR::Value{ir.Imm32(bias)}
                                    : IR::Value{ir.FPAdd(IR::F32{original}, ir.Imm32(bias))});
                }
            } else if (bias != 0.f && gradient) {
                if (HasSamplerAnisotropy(sampler)) {
                    throw std::runtime_error(
                        "Metal biased anisotropic gradients require footprint lowering");
                }
                const u32 image_index = handle & 0xffff;
                if (image_index >= program.info.images.size()) {
                    throw std::runtime_error("Invalid Metal gradient image handle");
                }
                const auto& image_resource = program.info.images[image_index];
                const auto image = image_resource.GetSharp(program.info);
                const auto type = image.GetViewType(image_resource.is_array);
                const u32 dimensions =
                    type == AmdGpu::ImageType::Color1D || type == AmdGpu::ImageType::Color1DArray
                        ? 1
                    : type == AmdGpu::ImageType::Color3D ? 3
                                                         : 2;
                const auto scale = ir.Imm32(std::exp2(bias));
                for (u32 argument : {2u, 3u}) {
                    const auto original = inst.Arg(argument);
                    if (dimensions == 1) {
                        inst.SetArg(argument, ir.FPMul(IR::F32{original}, scale));
                    } else {
                        std::array<IR::Value, 3> scaled;
                        for (u32 component = 0; component < dimensions; ++component) {
                            scaled[component] =
                                ir.FPMul(IR::F32{ir.CompositeExtract(original, component)}, scale);
                        }
                        inst.SetArg(argument,
                                    dimensions == 2
                                        ? ir.CompositeConstruct(scaled[0], scaled[1])
                                        : ir.CompositeConstruct(scaled[0], scaled[1], scaled[2]));
                    }
                }
            } else if (bias != 0.f && opcode == IR::Opcode::ImageQueryLod) {
                std::vector<IR::Inst*> raw_lod_users;
                for (const auto& use : inst.Uses()) {
                    const auto user_opcode = use.user->GetOpcode();
                    if (use.operand != 0 ||
                        (user_opcode != IR::Opcode::CompositeExtractF32x2 &&
                         user_opcode != IR::Opcode::CompositeExtractF32x4) ||
                        !use.user->Arg(1).IsImmediate() ||
                        use.user->Arg(1).Type() != IR::Type::U32 || use.user->Arg(1).U32() != 1) {
                        throw std::runtime_error(
                            "Metal biased clamped LOD query requires mip-selection lowering");
                    }
                    raw_lod_users.push_back(use.user);
                }
                IR::IREmitter after_query{*block, std::next(it)};
                const auto raw = IR::F32{after_query.CompositeExtract(IR::Value{&inst}, 1)};
                const auto biased = after_query.FPAdd(raw, after_query.Imm32(bias));
                for (auto* user : raw_lod_users) {
                    user->ReplaceUsesWithAndRemove(biased);
                }
            } else if (bias != 0.f) {
                throw std::runtime_error(
                    "Metal biased gather/LOD query requires native shader lowering");
            }
        }
    }
}

void ValidateImageLodQueries(const Shader::IR::Program& program, bool native_query_texture_lod) {
    for (const auto* block : program.blocks) {
        for (const auto& inst : *block) {
            if (inst.GetOpcode() != Shader::IR::Opcode::ImageQueryLod) {
                continue;
            }
            if (program.info.sw_stage != Shader::SwStage::Fragment) {
                throw std::runtime_error(
                    "Metal non-fragment LOD query requires derivative lowering");
            }
            if (!native_query_texture_lod) {
                throw std::runtime_error("Metal LOD query requires native capability lowering");
            }
        }
    }
}

std::bitset<Shader::NUM_IMAGES> LowerImageMinimumLod(
    Shader::IR::Program& program, std::array<ImageMinimumLodState, Shader::NUM_IMAGES>& image_lod,
    std::bitset<Shader::NUM_SAMPLERS>& post_clamp_samplers,
    std::bitset<Shader::NUM_IMAGES>& dependencies) {
    namespace IR = Shader::IR;
    std::bitset<Shader::NUM_IMAGES> lowered_images;
    std::bitset<Shader::NUM_IMAGES> read_images;
    std::bitset<Shader::NUM_IMAGES> queried_images;
    std::bitset<Shader::NUM_IMAGES> accessed_images;
    std::bitset<Shader::NUM_IMAGES> other_image_uses;
    std::bitset<Shader::NUM_SAMPLERS> other_sampler_uses;
    std::map<u32, u32> image_views;
    std::map<u32, u32> sampler_views;
    for (const IR::Block* block : program.blocks) {
        for (const auto& inst : *block) {
            if (IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                const u32 index = inst.Arg(0).U32() & 0xffff;
                if (index >= program.info.images.size() || index >= queried_images.size()) {
                    throw std::runtime_error("Invalid Metal minimum image LOD resource handle");
                }
                if (inst.GetOpcode() == IR::Opcode::ImageQueryDimensions) {
                    queried_images.set(index);
                } else {
                    accessed_images.set(index);
                }
            }
            bool sampled{};
            switch (inst.GetOpcode()) {
            case IR::Opcode::ImageSampleImplicitLod:
            case IR::Opcode::ImageSampleDrefImplicitLod:
            case IR::Opcode::ImageGradient:
            case IR::Opcode::ImageSampleDrefGradient:
            case IR::Opcode::ImageGather:
            case IR::Opcode::ImageGatherDref:
            case IR::Opcode::ImageQueryLod:
                sampled = true;
                break;
            case IR::Opcode::ImageRead: {
                const auto& resource = program.info.images.at(inst.Arg(0).U32());
                if (resource.mip_fallback_mode == Shader::MipStorageFallbackMode::None) {
                    continue;
                }
                break;
            }
            case IR::Opcode::ImageWrite:
                continue;
            default:
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if ((handle & 0xffff) >= other_image_uses.size() ||
                (sampled && (handle >> 16) >= other_sampler_uses.size())) {
                throw std::runtime_error("Invalid Metal minimum image LOD resource handle");
            }
            other_image_uses.set(handle & 0xffff);
            if (sampled) {
                other_sampler_uses.set(handle >> 16);
            }
        }
    }
    for (IR::Block* block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto& inst = *it;
            const auto opcode = inst.GetOpcode();
            const bool read = opcode == IR::Opcode::ImageRead;
            if (!read && opcode != IR::Opcode::ImageSampleExplicitLod &&
                opcode != IR::Opcode::ImageSampleDrefExplicitLod) {
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            const u32 image_index = read ? handle : handle & 0xffff;
            const u32 sampler_index = handle >> 16;
            if (image_index >= program.info.images.size() ||
                (!read && sampler_index >= program.info.samplers.size())) {
                throw std::runtime_error("Invalid Metal minimum image LOD handle");
            }
            const auto resource = program.info.images[image_index];
            if (read && resource.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                continue;
            }
            const auto image = resource.GetSharp(program.info);
            if (read) {
                read_images.set(image_index);
            }
            const auto state = ImageMinimumLodState::FromImage(image, resource);
            dependencies.set(image_index);
            image_lod[image_index] = state;
            if (!image.min_lod) {
                continue;
            }
            if (!state.levels || (!read && resource.is_written) ||
                resource.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                throw std::runtime_error("Invalid Metal minimum image LOD state");
            }
            if (!read) {
                const auto sampler = program.info.samplers[sampler_index].GetSharp(program.info);
                if (sampler.MinLod() > sampler.MaxLod()) {
                    throw std::runtime_error("Invalid Metal minimum image sampler LOD state");
                }
            }
            auto view = image_views.find(image_index);
            if (view == image_views.end()) {
                u32 index = image_index;
                if (other_image_uses.test(image_index)) {
                    if (resource.is_written) {
                        throw std::runtime_error(
                            "Metal storage image sampling requires unified access lowering");
                    }
                    if (program.info.images.size() >= Shader::NUM_IMAGES) {
                        throw std::runtime_error("Metal minimum LOD views exceed image capacity");
                    }
                    index = program.info.images.size();
                    program.info.images.push_back(resource);
                }
                image_lod[index] = state;
                lowered_images.set(index);
                view = image_views.emplace(image_index, index).first;
            }
            if (read) {
                const u32 minimum =
                    std::min(state.minimum >> 8, state.base_level + state.levels - 1);
                if (minimum > state.base_level) {
                    IR::IREmitter ir{*block, it};
                    const IR::U32 lod = inst.Arg(2).IsEmpty() ? ir.Imm32(0u) : IR::U32{inst.Arg(2)};
                    const auto valid =
                        ir.IGreaterThanEqual(lod, ir.Imm32(minimum - state.base_level), false);
                    inst.SetArg(2,
                                ir.Select(valid, lod, ir.Imm32(std::numeric_limits<u32>::max())));
                }
                inst.SetArg(0, IR::Value{view->second});
                continue;
            }
            auto sampler_view = sampler_views.find(sampler_index);
            if (sampler_view == sampler_views.end()) {
                u32 index = sampler_index;
                if (other_sampler_uses.test(sampler_index)) {
                    if (program.info.samplers.size() >= Shader::NUM_SAMPLERS) {
                        throw std::runtime_error("Metal minimum LOD views exceed sampler capacity");
                    }
                    const auto sampler_resource = program.info.samplers[sampler_index];
                    index = program.info.samplers.size();
                    program.info.samplers.push_back(sampler_resource);
                }
                post_clamp_samplers.set(index);
                sampler_view = sampler_views.emplace(sampler_index, index).first;
            }
            const float minimum =
                std::min(float(state.minimum) / 256.f, float(state.base_level + state.levels - 1));
            const float relative_minimum = std::max(minimum - float(state.base_level), 0.f);
            const u32 lod_argument = opcode == IR::Opcode::ImageSampleDrefExplicitLod ? 3 : 2;
            IR::IREmitter ir{*block, it};
            inst.SetArg(lod_argument,
                        ir.FPMax(IR::F32{inst.Arg(lod_argument)}, ir.Imm32(relative_minimum)));
            inst.SetArg(0, IR::Value{(sampler_view->second << 16) | view->second});
        }
    }
    for (u32 index = 0; index < program.info.images.size(); ++index) {
        const auto& resource = program.info.images[index];
        if ((resource.is_written && !read_images.test(index) && !other_image_uses.test(index)) ||
            (queried_images.test(index) && !accessed_images.test(index))) {
            lowered_images.set(index);
        }
    }
    return lowered_images;
}

} // namespace

ShaderCompiler::ShaderCompiler(const Instance& instance, u32 sparse_page_shift)
    : profile{
          .max_viewport_width = 16384,
          .max_viewport_height = 16384,
          .max_shared_memory_size =
              static_cast<u32>(instance.GetDevice().maxThreadgroupMemoryLength),
          .supported_spirv = 0x00010600,
          .subgroup_size = 32,
          .sparse_page_shift = sparse_page_shift,
          .guest_page_shift = GuestPageTableLayout::PageBits,
          .guest_page_leaf_bits = GuestPageTableLayout::LeafBits,
          .support_int8 = true,
          .support_int16 = true,
          .support_int64 = true,
          .support_float16 = true,
          .supports_image_load_store_lod = true,
          .needs_lds_barriers = true,
          .needs_storage_lds_barriers = true,
          .needs_guest_compute_barriers = true,
          .lower_shared_memory_after_wave64 = true,
          .needs_geometry_output_lowering = true,
          .needs_geometry_message_predication = true,
          .needs_vertex_fetch_lowering = true,
          .needs_bounded_atomic_counters = true,
          .needs_extended_buffer_atomics = true,
          .needs_extended_ds_atomics = true,
          .needs_software_shared_int64_atomics = true,
          .needs_typed_image_atomics = true,
          .needs_multisample_array_images = true,
          .needs_software_float_atomics = true,
          .needs_buffer_offsets = true,
          .needs_wide_buffer_byte_offsets = true,
          .needs_unorm_fixup = true,
          .needs_unorm32_depth_conversion = true,
          .needs_depth_compare_gradients = true,
          .needs_depth_compare_explicit_lod = true,
          .needs_image_gradients = true,
          .needs_image_sample_clamp = true,
          .needs_image_gather_lod = true,
          .needs_image_query_lod_sampler = true,
          .supports_shader_stencil_export = true,
          .needs_fmask_lowering = true,
      },
      native_sampler_lod_bias{instance.SupportsSamplerLodBias()},
      native_query_texture_lod{bool(instance.GetDevice().supportsQueryTextureLOD)},
      compiler{instance} {
    ASSERT(sparse_page_shift >= 12 && sparse_page_shift < 32);
}

std::bitset<Shader::NUM_BUFFERS> ShaderCompiler::UnboundBuffers(const Shader::Info& info) {
    if (info.buffers.size() > Shader::NUM_BUFFERS) {
        throw std::runtime_error("Metal shader buffer metadata exceeds native capacity");
    }
    std::bitset<Shader::NUM_BUFFERS> unbound;
    for (size_t index = 0; index < info.buffers.size(); ++index) {
        const auto& resource = info.buffers[index];
        if (resource.buffer_type == Shader::BufferType::Guest &&
            ReadGuestBufferSharp(resource, info).data_format ==
                u32(AmdGpu::DataFormat::FormatInvalid)) {
            unbound[index] = true;
        }
    }
    return unbound;
}

TranslatedShader ShaderCompiler::TranslateSpirv(std::span<const u32> spirv, bool flip_vertex_y,
                                                bool sample_rate_shading) {
    spirv_cross::CompilerMSL msl{spirv.data(), spirv.size()};
    const auto entries = msl.get_entry_points_and_stages();
    if (entries.size() != 1) {
        throw std::runtime_error("Metal guest shader requires exactly one entry point");
    }
    const auto& entry = entries.front();
    if (entry.execution_model != spv::ExecutionModelVertex &&
        entry.execution_model != spv::ExecutionModelFragment &&
        entry.execution_model != spv::ExecutionModelGLCompute &&
        entry.execution_model != spv::ExecutionModelMeshEXT) {
        throw std::runtime_error("Metal guest shader requires stage lowering before translation");
    }
    msl.set_entry_point(entry.name, entry.execution_model);
    if (flip_vertex_y && entry.execution_model != spv::ExecutionModelVertex &&
        entry.execution_model != spv::ExecutionModelMeshEXT) {
        throw std::runtime_error("Metal viewport orientation requires a vertex shader");
    }
    if (sample_rate_shading && entry.execution_model != spv::ExecutionModelFragment) {
        throw std::runtime_error("Metal sample rate shading requires a fragment shader");
    }
    auto common_options = msl.get_common_options();
    common_options.vertex.flip_vert_y = flip_vertex_y;
    msl.set_common_options(common_options);

    auto options = msl.get_msl_options();
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    options.set_msl_version(4, 0);
    options.argument_buffers = true;
    options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    options.force_native_arrays = true;
    options.fixed_subgroup_size = 32;
    options.force_sample_rate_shading = sample_rate_shading;
    msl.set_msl_options(options);

    TranslatedShader shader;
    switch (entry.execution_model) {
    case spv::ExecutionModelVertex:
        shader.stage = Shader::SwStage::Vertex;
        break;
    case spv::ExecutionModelFragment:
        shader.stage = Shader::SwStage::Fragment;
        break;
    case spv::ExecutionModelGLCompute:
        shader.stage = Shader::SwStage::Compute;
        break;
    case spv::ExecutionModelMeshEXT: {
        shader.stage = Shader::SwStage::Geometry;
        spirv_cross::SpecializationConstant x{}, y{}, z{};
        const auto workgroup_constant = msl.get_work_group_size_specialization_constants(x, y, z);
        if (x.id || y.id || z.id) {
            throw std::runtime_error("Metal mesh workgroup specialization requires lowering");
        }
        for (u32 component = 0; component < shader.mesh_workgroup_size.size(); ++component) {
            shader.mesh_workgroup_size[component] =
                workgroup_constant
                    ? msl.get_constant(workgroup_constant).scalar(0, component)
                    : msl.get_execution_mode_argument(spv::ExecutionModeLocalSize, component);
            if (!shader.mesh_workgroup_size[component]) {
                throw std::runtime_error("Invalid Metal mesh workgroup size");
            }
        }
        msl.unset_execution_mode(spv::ExecutionModeLocalSizeId);
        msl.set_execution_mode(spv::ExecutionModeLocalSize, shader.mesh_workgroup_size[0],
                               shader.mesh_workgroup_size[1], shader.mesh_workgroup_size[2]);
        break;
    }
    default:
        UNREACHABLE();
    }
    const auto reflected = msl.get_shader_resources();
    std::map<std::pair<u32, u32>, ShaderResourceBinding> bindings;
    u32 next_argument_index{};

    const auto add_resources = [&](const auto& resources, ShaderResourceType type,
                                   u32& next_index) {
        for (const auto& resource : resources) {
            const u32 set = msl.get_decoration(resource.id, spv::DecorationDescriptorSet);
            const u32 binding = msl.get_decoration(resource.id, spv::DecorationBinding);
            if (set != 0) {
                throw std::runtime_error("Metal guest shader requires descriptor set zero");
            }
            const auto& resource_type = msl.get_type(resource.type_id);
            u32 count = 1;
            for (size_t i = 0; i < resource_type.array.size(); ++i) {
                if (!resource_type.array_size_literal[i] || resource_type.array[i] == 0 ||
                    resource_type.array[i] > UINT32_MAX / count) {
                    throw std::runtime_error("Metal guest descriptor array requires a fixed size");
                }
                count *= resource_type.array[i];
            }
            const auto [it, inserted] = bindings.try_emplace(
                std::pair{set, binding},
                ShaderResourceBinding{type, set, binding, next_argument_index, count});
            if (it->second.type != type || it->second.count != count) {
                throw std::runtime_error("Metal guest shader has incompatible aliased bindings");
            }
            if (inserted) {
                if (count > UINT32_MAX - next_index || count > UINT32_MAX - next_argument_index) {
                    throw std::runtime_error("Metal guest shader resource count overflow");
                }
                next_index += count;
                next_argument_index += count;
            }
        }
    };
    add_resources(reflected.uniform_buffers, ShaderResourceType::Buffer, shader.buffer_count);
    add_resources(reflected.storage_buffers, ShaderResourceType::Buffer, shader.buffer_count);
    add_resources(reflected.separate_images, ShaderResourceType::Texture, shader.texture_count);
    add_resources(reflected.storage_images, ShaderResourceType::Texture, shader.texture_count);
    add_resources(reflected.separate_samplers, ShaderResourceType::Sampler, shader.sampler_count);

    if (!reflected.sampled_images.empty() || !reflected.subpass_inputs.empty() ||
        !reflected.acceleration_structures.empty() || !reflected.atomic_counters.empty()) {
        throw std::runtime_error("Unsupported Metal guest shader resource class");
    }
    for (const auto& [key, binding] : bindings) {
        spirv_cross::MSLResourceBinding remap;
        remap.stage = entry.execution_model;
        remap.desc_set = binding.descriptor_set;
        remap.binding = binding.binding;
        remap.count = binding.count;
        remap.msl_buffer = binding.index;
        remap.msl_texture = binding.index;
        remap.msl_sampler = binding.index;
        msl.add_msl_resource_binding(remap);
        shader.resources.push_back(binding);
    }

    spirv_cross::MSLResourceBinding push_data;
    push_data.stage = entry.execution_model;
    push_data.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
    push_data.binding = spirv_cross::ResourceBindingPushConstantBinding;
    push_data.msl_buffer = PushDataBufferIndex;
    msl.add_msl_resource_binding(push_data);

    spirv_cross::MSLResourceBinding descriptors;
    descriptors.stage = entry.execution_model;
    descriptors.desc_set = 0;
    descriptors.binding = spirv_cross::kArgumentBufferBinding;
    descriptors.msl_buffer = DescriptorBufferIndex;
    msl.add_msl_resource_binding(descriptors);

    shader.source = msl.compile();
    shader.entry_point = msl.get_cleansed_entry_point_name(entry.name, entry.execution_model);
    if (shader.stage == Shader::SwStage::Fragment) {
        const auto active = msl.get_active_interface_variables();
        for (const auto& input : reflected.stage_inputs) {
            if (!active.contains(input.id)) {
                continue;
            }
            if (!msl.has_decoration(input.id, spv::DecorationLocation)) {
                shader.geometry_inputs_supported = false;
                continue;
            }
            const auto& type = msl.get_type(input.type_id);
            const u32 location = msl.get_decoration(input.id, spv::DecorationLocation);
            if (!msl.is_msl_shader_input_used(location)) {
                continue;
            }
            if (location >= Shader::IR::NumParams ||
                msl.get_decoration(input.id, spv::DecorationComponent) != 0 ||
                type.basetype != spirv_cross::SPIRType::Float || type.width != 32 ||
                type.columns != 1 || !type.array.empty() || !type.vecsize || type.vecsize > 4 ||
                msl.has_decoration(input.id, spv::DecorationPerVertexKHR) ||
                msl.has_decoration(input.id, spv::DecorationExplicitInterpAMD)) {
                shader.geometry_inputs_supported = false;
                continue;
            }
            shader.geometry_inputs.push_back(
                {location, type.vecsize, msl.has_decoration(input.id, spv::DecorationFlat)});
        }
    }
    for (auto& binding : shader.resources) {
        binding.used = msl.is_msl_resource_binding_used(entry.execution_model,
                                                        binding.descriptor_set, binding.binding);
    }
    shader.uses_push_data = msl.is_msl_resource_binding_used(entry.execution_model,
                                                             push_data.desc_set, push_data.binding);
    if (msl.needs_swizzle_buffer() || msl.needs_buffer_size_buffer() ||
        msl.needs_view_mask_buffer() || msl.needs_depth_clip_state_buffer() ||
        msl.needs_dispatch_base_buffer() || msl.needs_output_buffer() ||
        msl.needs_patch_output_buffer()) {
        throw std::runtime_error("Metal guest shader requires an unbound auxiliary buffer");
    }
    return shader;
}

void ShaderCompiler::ConfigureVertexFetch(Shader::Info& info, const VertexFetchLimits* limits) {
    info.supports_vertex_attribute = [limits = limits ? std::make_optional(*limits) : std::nullopt](
                                         const Shader::Gcn::VertexAttribute& attribute,
                                         const Shader::Gcn::FetchShaderData& fetch,
                                         const Shader::Info& info) {
        if (attribute.data_format) {
            const auto descriptor =
                info.ReadUdReg<AmdGpu::Buffer>(attribute.sgpr_base, attribute.dword_offset);
            if (descriptor.data_format == u32(AmdGpu::DataFormat::FormatInvalid)) {
                return false;
            }
        }
        const auto buffer = attribute.GetSharp(info);
        if (VertexFormat(buffer) == MTLVertexFormatInvalid || buffer.swizzle_enable ||
            buffer.add_tid_enable) {
            return false;
        }
        using Data = AmdGpu::DataFormat;
        const auto data = buffer.GetDataFmt();
        const u32 alignment =
            data == Data::Format8 || data == Data::Format8_8 || data == Data::Format8_8_8_8
                ? 1
                : (data == Data::Format16 || data == Data::Format16_16 ||
                           data == Data::Format16_16_16_16
                       ? 2
                       : 4);
        if (buffer.base_address % alignment || buffer.GetStride() % alignment) {
            return false;
        }
        if (!limits) {
            return true;
        }
        const u64 bytes = AmdGpu::NumBitsPerBlock(data) / 8;
        if (!buffer.GetStride()) {
            return u64{attribute.inst_offset} + bytes <= buffer.num_records &&
                   Core::Memory::Instance()->ClampRangeSize(buffer.base_address, bytes) == bytes;
        }
        const auto offset = [&](s8 sgpr) -> u32 {
            if (sgpr < 0) {
                return 0;
            }
            if (u32(sgpr) >= info.user_data.size()) {
                throw std::runtime_error("Invalid Metal guest vertex offset register");
            }
            return info.user_data[sgpr];
        };
        using Step = Shader::Gcn::VertexAttribute::InstanceIdType;
        const auto step = attribute.GetStepRate();
        u64 last_index{};
        if (step == Step::None) {
            if (limits->indexed || limits->indirect || !limits->vertex_count) {
                return false;
            }
            last_index = u64{limits->vertex_offset ? limits->vertex_offset
                                                   : offset(fetch.vertex_offset_sgpr)} +
                         limits->vertex_count - 1;
        } else {
            const u32 divisor = step == Step::OverStepRate0   ? limits->step_rate_0
                                : step == Step::OverStepRate1 ? limits->step_rate_1
                                                              : 1;
            if (divisor) {
                if (limits->indirect || !limits->instance_count) {
                    return false;
                }
                last_index =
                    (u64{offset(fetch.instance_offset_sgpr)} + limits->instance_count - 1) /
                    divisor;
            }
        }
        if (last_index >= buffer.num_records) {
            return false;
        }
        const u64 size = last_index * buffer.GetStride() + bytes;
        return Core::Memory::Instance()->ClampRangeSize(buffer.base_address, size) == size;
    };
}

PreparedShader ShaderCompiler::PrepareGuest(
    Shader::Pools& pools, const Shader::Profile& profile, bool native_sampler_lod_bias,
    bool native_query_texture_lod, Shader::Info& info, Shader::RuntimeInfo runtime,
    std::span<const u32> code, Shader::Backend::Bindings& binding, bool flip_vertex_y,
    bool sample_rate_shading, const GeometryKernelConfig* geometry,
    const GeometryCopyConfig* geometry_copy, const GeometryExportConfig* geometry_export,
    const VertexFetchLimits* vertex_fetch_limits, const TessellationKernelConfig* tessellation) {
    const bool tessellation_stage = info.hw_stage == Shader::HwStage::Local ||
                                    info.hw_stage == Shader::HwStage::Hull ||
                                    info.sw_stage == Shader::SwStage::TessellationEval;
    if (tessellation_stage != (tessellation != nullptr) ||
        (tessellation &&
         (geometry || geometry_copy || geometry_export || flip_vertex_y || sample_rate_shading))) {
        throw std::runtime_error("Metal tessellation shader requires native kernel configuration");
    }
    if ((info.hw_stage == Shader::HwStage::Geometry) != (geometry != nullptr) ||
        (geometry &&
         (flip_vertex_y || sample_rate_shading || !profile.needs_geometry_output_lowering ||
          geometry->shared_memory_size > profile.max_shared_memory_size))) {
        throw std::runtime_error("Metal geometry shader requires native kernel configuration");
    }
    if (geometry_copy &&
        (geometry || info.hw_stage != Shader::HwStage::Vertex ||
         info.sw_stage != Shader::SwStage::Vertex || flip_vertex_y || sample_rate_shading)) {
        throw std::runtime_error("Metal geometry copy requires a native vertex configuration");
    }
    if ((info.hw_stage == Shader::HwStage::Export && !tessellation && !geometry_export) ||
        (geometry_export &&
         (geometry || geometry_copy ||
          info.hw_stage != (geometry_export->raster_output ? Shader::HwStage::Vertex
                                                           : Shader::HwStage::Export) ||
          info.sw_stage != Shader::SwStage::Vertex || flip_vertex_y || sample_rate_shading))) {
        throw std::runtime_error("Metal geometry export requires a native ES configuration");
    }
    ConfigureVertexFetch(info, vertex_fetch_limits);
    std::vector<u32> spirv;
    std::array<SamplerLodUsage, Shader::NUM_SAMPLERS> sampler_lod{};
    std::bitset<Shader::NUM_BUFFERS> guarded_buffers;
    std::bitset<Shader::NUM_IMAGES> single_mip_gathers;
    std::bitset<Shader::NUM_IMAGES> base_mip_gathers;
    std::bitset<Shader::NUM_IMAGES> lowered_minimum_lod_images;
    std::bitset<Shader::NUM_IMAGES> minimum_lod_dependencies;
    std::bitset<Shader::NUM_IMAGES> descriptor_query_images;
    std::bitset<Shader::NUM_IMAGES> query_only_images;
    std::bitset<Shader::NUM_IMAGES> fmask_dependencies;
    std::array<FmaskImageState, Shader::NUM_IMAGES> fmask_images{};
    std::array<ImageMinimumLodState, Shader::NUM_IMAGES> image_minimum_lod{};
    std::bitset<Shader::NUM_SAMPLERS> post_clamp_samplers;
    std::optional<GeometryKernelBindings> geometry_bindings;
    std::optional<GeometryCopyResult> copy_result;
    std::optional<GeometryExportBindings> export_bindings;
    GeometryBufferAccesses geometry_buffers;
    std::shared_ptr<const GeometryBufferAddressProgram> guest_buffer_bindings;
    std::bitset<Shader::NUM_BUFFERS> paged_buffers;
    ImageAliasState image_aliases;
    std::optional<TessellationKernelBindings> tessellation_bindings;
    {
        SCOPE_EXIT {
            pools.ReleaseContents();
        };
        const auto hw_stage = info.hw_stage;
        const auto sw_stage = info.sw_stage;
        SCOPE_EXIT {
            info.hw_stage = hw_stage;
            info.sw_stage = sw_stage;
        };
        std::optional<GeometryKernelTranslation> geometry_translation;
        std::optional<GeometryCopyTranslation> copy_translation;
        std::optional<GeometryExportTranslation> export_translation;
        std::optional<TessellationKernelTranslation> tessellation_translation;
        auto translation_profile = profile;
        translation_profile.needs_vector_register_predication =
            geometry || geometry_copy || geometry_export || tessellation;
        translation_profile.needs_memory_predication =
            translation_profile.needs_vector_register_predication;
        std::optional<Shader::TranslationHooks> hooks;
        if (geometry) {
            geometry_translation.emplace(*geometry, runtime, translation_profile);
            hooks = geometry_translation->Hooks();
        } else if (geometry_copy) {
            translation_profile.needs_geometry_copy_lowering = true;
            copy_translation.emplace(*geometry_copy, runtime, translation_profile);
            hooks = copy_translation->Hooks();
        } else if (geometry_export) {
            translation_profile.needs_geometry_export_lowering = true;
            export_translation.emplace(*geometry_export, runtime, translation_profile);
            hooks = export_translation->Hooks();
        } else if (tessellation) {
            translation_profile.needs_geometry_export_lowering = true;
            tessellation_translation.emplace(*tessellation, runtime, translation_profile);
            hooks = tessellation_translation->Hooks();
            // The Metal hook lowers logical wave64 operations before physical SIMD32 emission.
            translation_profile.subgroup_size = 64;
        }
        if (!hooks) {
            hooks.emplace();
        }
        const auto lower_geometry_resources = hooks->lower_resources;
        hooks->lower_resources = [&](Shader::IR::Program& program) {
            guest_buffer_bindings = std::make_shared<const GeometryBufferAddressProgram>(
                InspectGuestBufferBindingProgram(program));
            if (lower_geometry_resources) {
                lower_geometry_resources(program);
            }
        };
        info.uses_guest_page_reads = false;
        info.guest_page_atomics.clear();
        auto program =
            tessellation_translation
                ? tessellation_translation->Translate(
                      code, pools, info, runtime, translation_profile, *hooks,
                      LowerUnboundBufferAccesses, PatchGuestBufferAddress, ConvertImageRead,
                      ConvertImageSample)
                : Shader::TranslateProgram(code, pools, info, runtime, translation_profile,
                                           LowerUnboundBufferAccesses, PatchGuestBufferAddress,
                                           ConvertImageRead, ConvertImageSample, &*hooks);
        if (geometry_translation) {
            geometry_bindings = geometry_translation->Bindings();
        }
        if (copy_translation) {
            copy_result = copy_translation->Result();
        }
        if (export_translation) {
            export_bindings = export_translation->Bindings();
        }
        if (tessellation_translation) {
            tessellation_bindings = tessellation_translation->Bindings();
            tessellation_bindings->registers.first_binding = binding.unified;
        }
        if (info.translation_failed) {
            throw std::runtime_error("Guest shader translation failed");
        }
        if (info.uses_buffer_int64_atomics && !profile.supports_buffer_int64_atomics) {
            throw std::runtime_error("Metal guest shader requires global 64-bit buffer atomics");
        }
        descriptor_query_images = LowerDescriptorImageDimensions(program);
        fmask_dependencies = LowerFmaskImageAccesses(program, fmask_images);
        image_aliases = LowerImageAliases(program);
        LowerMultisampleImageDimensions(program);
        LowerStorageImageDimensions(program);
        ValidateImageLodQueries(program, native_query_texture_lod);
        single_mip_gathers = LowerGatherLod(program, base_mip_gathers, sampler_lod);
        LowerSamplerLod(program, sampler_lod, native_sampler_lod_bias,
                        single_mip_gathers | base_mip_gathers);
        lowered_minimum_lod_images = LowerImageMinimumLod(
            program, image_minimum_lod, post_clamp_samplers, minimum_lod_dependencies);
        LowerSharedMemoryAddresses(program, tessellation,
                                   tessellation_bindings ? &*tessellation_bindings : nullptr);
        LowerComputeDispatchArguments(program);
        if (tessellation_translation) {
            tessellation_translation->LowerEvaluationGroups(program);
        }
        if (geometry_translation) {
            geometry_translation->LowerWorkgroups(program);
        }
        if (copy_translation) {
            copy_translation->LowerWorkgroups(program);
        }
        if (sample_rate_shading) {
            LowerFragmentSampleMask(program);
        }
        guarded_buffers = GuardedBuffers(program);
        paged_buffers = PagedGuestBuffers(program);
        query_only_images = QueryOnlyImages(program);
        if (geometry || geometry_copy || geometry_export || tessellation) {
            geometry_buffers = InspectGeometryBufferAccesses(program);
        }
        if (geometry_bindings) {
            geometry_bindings->first_binding = binding.unified;
        }
        if (copy_result) {
            copy_result->first_binding = binding.unified;
        }
        if (export_bindings) {
            export_bindings->first_binding = binding.unified;
        }
        auto emission_profile = translation_profile;
        emission_profile.subgroup_size = profile.subgroup_size;
        spirv = Shader::Backend::SPIRV::EmitSPIRV(emission_profile, runtime, program, binding);
    }
    return {ShaderMetadata{sampler_lod,
                           UnboundBuffers(info),
                           guarded_buffers,
                           single_mip_gathers,
                           base_mip_gathers,
                           lowered_minimum_lod_images,
                           minimum_lod_dependencies,
                           descriptor_query_images,
                           query_only_images,
                           fmask_dependencies,
                           fmask_images,
                           image_minimum_lod,
                           post_clamp_samplers,
                           geometry_bindings,
                           std::move(copy_result),
                           export_bindings,
                           geometry_buffers,
                           std::move(guest_buffer_bindings),
                           paged_buffers,
                           image_aliases,
                           tessellation_bindings},
            std::move(spirv)};
}

CompiledShader ShaderCompiler::CompileGuest(
    Shader::Info& info, Shader::RuntimeInfo runtime, std::span<const u32> code,
    Shader::Backend::Bindings& binding, std::string_view name, bool flip_vertex_y,
    bool sample_rate_shading, const GeometryKernelConfig* geometry,
    const GeometryCopyConfig* geometry_copy, const GeometryExportConfig* geometry_export,
    const VertexFetchLimits* vertex_fetch_limits, const TessellationKernelConfig* tessellation) {
    auto prepared = [&] {
        std::scoped_lock lock{translation_mutex};
        return PrepareGuest(pools, profile, native_sampler_lod_bias, native_query_texture_lod, info,
                            runtime, code, binding, flip_vertex_y, sample_rate_shading, geometry,
                            geometry_copy, geometry_export, vertex_fetch_limits, tessellation);
    }();
    auto translated = TranslateSpirv(prepared.spirv, flip_vertex_y, sample_rate_shading);
    auto library = compiler.CompileLibraryAsync(translated.source, name);
    return {std::move(static_cast<ShaderMetadata&>(prepared)), std::move(translated),
            std::move(library)};
}

} // namespace Metal
