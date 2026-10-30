// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <boost/container/static_vector.hpp>
#include "common/hash.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_fetch_shader.h"
#include "video_core/renderer_metal/metal_guest_shader.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_sampler.h"
#include "video_core/renderer_metal/metal_shader_cache.h"

namespace Metal {

namespace {

bool ReuseSrtWalker(Shader::Info& info, const Shader::Info& cached) {
    if (!info.walker_with_memory_read || !cached.walker_with_memory_read ||
        !info.walker_code_owner || !cached.walker_code_owner || info.walker_code.empty() ||
        info.srt_info.flattened_bufsize_dw != cached.srt_info.flattened_bufsize_dw ||
        !std::ranges::equal(info.walker_code, cached.walker_code)) {
        return false;
    }
    info.walker_with_memory_read = cached.walker_with_memory_read;
    info.walker_code = cached.walker_code;
    info.walker_code_owner = cached.walker_code_owner;
    return true;
}

bool ReuseSrtSnapshot(Shader::Info& info, const Shader::Info& cached) {
    if (!info.walker_with_memory_read || !info.walker_code_owner ||
        info.walker_code_owner != cached.walker_code_owner ||
        info.walker_with_memory_read != cached.walker_with_memory_read ||
        info.user_data.size() != Shader::NUM_USER_DATA_REGS ||
        cached.flattened_ud_buf.size() != info.srt_info.flattened_bufsize_dw ||
        info.user_data.size() > cached.flattened_ud_buf.size() ||
        !std::ranges::equal(info.user_data,
                            std::span{cached.flattened_ud_buf}.first(info.user_data.size()))) {
        return false;
    }
    info.flattened_ud_buf = cached.flattened_ud_buf;
    return true;
}

} // namespace

Shader::StageSpecialization ShaderCache::SpecializeShader(Shader::Info& info,
                                                          const Shader::RuntimeInfo& runtime,
                                                          const Shader::Profile& profile,
                                                          Shader::Backend::Bindings bindings,
                                                          bool native_tessellation) {
    const auto stage = info.sw_stage;
    SCOPE_EXIT {
        info.sw_stage = stage;
    };
    if (native_tessellation) {
        info.sw_stage = Shader::SwStage::Compute;
    }
    for (const auto& resource : info.buffers) {
        if (resource.buffer_type == Shader::BufferType::Guest) {
            static_cast<void>(ReadGuestBufferSharp(resource, info));
        }
    }
    const auto specialize_buffer = [](auto& spec, const auto& resource, AmdGpu::Buffer sharp) {
        spec.stride = sharp.GetStride();
        spec.is_formatted =
            resource.is_formatted && (resource.buffer_type != Shader::BufferType::Guest ||
                                      sharp.data_format != u32(AmdGpu::DataFormat::FormatInvalid));
        spec.swizzle_enable = sharp.swizzle_enable;
        if (spec.is_formatted) {
            spec.data_format = static_cast<u32>(sharp.GetDataFmt());
            spec.num_format = sharp.num_format;
            spec.dst_select = sharp.DstSelect();
            spec.num_conversion = sharp.GetNumberConversion();
        }
        if (spec.swizzle_enable) {
            spec.index_stride = sharp.index_stride;
            spec.element_size = sharp.element_size;
        }
    };
    Shader::StageSpecialization result{info, runtime, profile, bindings, specialize_buffer};
    for (u32 index = 0; index < info.images.size(); ++index) {
        const auto& resource = info.images[index];
        const auto sharp = resource.GetSharp(info);
        auto& spec = result.images[index];
        spec.is_signed = sharp.GetNumberFmt() == AmdGpu::NumberFormat::Sint;
        if (!resource.is_written && !resource.is_depth &&
            spec.num_conversion != AmdGpu::NumberConversion::None) {
            spec.dst_select = sharp.DstSelect();
            for (auto& selector : spec.dst_select.array) {
                if (selector >= AmdGpu::CompSwizzle::Red &&
                    selector <= AmdGpu::CompSwizzle::Alpha) {
                    selector = AmdGpu::CompSwizzle::Red;
                }
            }
        }
    }
    for (u32 index = 0; index < info.buffers.size(); ++index) {
        const auto& resource = info.buffers[index];
        if (resource.buffer_type != Shader::BufferType::Guest) {
            continue;
        }
        const auto sharp = resource.GetSharp(info);
        if (sharp.num_records) {
            continue;
        }
        result.bitset[index] = true;
        specialize_buffer(result.buffers[index], resource, sharp);
    }
    return result;
}

namespace {

bool HasCompatibleImageViews(const Shader::Info& info) {
    for (const auto& resource : info.images) {
        if (!resource.is_atomic) {
            continue;
        }
        auto original = resource;
        original.is_atomic_view = false;
        const auto image = original.GetSharp(info);
        if (AmdGpu::IsFmask(image.GetDataFmt()) ||
            AmdGpu::NumBitsPerBlock(image.GetDataFmt()) != 32) {
            return false;
        }
        const bool atomic_view = image.GetDataFmt() != AmdGpu::DataFormat::Format32 ||
                                 !AmdGpu::IsInteger(image.GetNumberFmt());
        if (resource.is_atomic_view != atomic_view) {
            return false;
        }
    }
    return true;
}

} // namespace

ShaderCache::ShaderCache(const Instance& instance_, BufferCache& buffer_cache_,
                         u32 sparse_page_shift)
    : instance{instance_}, buffer_cache{buffer_cache_}, shaders{instance, sparse_page_shift},
      pipelines{instance} {}

size_t ShaderCache::KeyHash::operator()(const Key& key) const noexcept {
    return HashCombine(
        HashCombine(HashCombine(key.hash, key.code_hash), static_cast<u64>(key.hw_stage)),
        static_cast<u64>(key.sw_stage));
}

Shader::ShaderParams ShaderCache::GetParams(
    VAddr address, std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data) {
    return ReadShaderParams(address, user_data, [this](VAddr source, u64 size) {
        buffer_cache.ReadMemory(source, size);
    });
}

ShaderCache::Program& ShaderCache::GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                                              const Shader::ShaderParams& params) {
    const std::string_view bytes{reinterpret_cast<const char*>(params.code.data()),
                                 params.code.size_bytes()};
    const u64 code_hash = std::hash<std::string_view>{}(bytes);
    auto& bucket = programs[Key{params.hash, code_hash, hw_stage, sw_stage}];
    for (auto& program : bucket) {
        if (std::ranges::equal(program.code, params.code)) {
            return program;
        }
    }
    Program program;
    program.code_hash = code_hash;
    program.code.assign(params.code.begin(), params.code.end());
    return bucket.emplace_back(std::move(program));
}

Shader::TessellationDataConstantBuffer ShaderCache::GetTessellationConstants(
    const Shader::ShaderParams& hull, const Shader::RuntimeInfo& runtime) {
    if (runtime.hw_stage != Shader::HwStage::Hull ||
        runtime.sw_stage != Shader::SwStage::TessellationControl) {
        throw std::runtime_error("Metal tessellation constants require a hull shader runtime");
    }
    auto& program = GetProgram(runtime.hw_stage, runtime.sw_stage, hull);
    auto analysis = std::ranges::find_if(
        program.tessellation, [&](const auto& cached) { return cached.runtime == runtime; });
    if (analysis == program.tessellation.end()) {
        const auto binding =
            InspectTessellationConstantBinding(program.code, runtime, GetProfile());
        program.tessellation.push_back({runtime, binding});
        analysis = std::prev(program.tessellation.end());
    }
    return ReadTessellationConstants(
        analysis->binding, hull.user_data, [this](VAddr address, u64 size) {
            const auto backing = Core::Memory::Instance()->GetBackingRanges(address, size);
            if (std::ranges::any_of(backing,
                                    [](const auto& range) { return !range.physical_address; })) {
                throw std::runtime_error("Metal tessellation constants must be fully mapped");
            }
            buffer_cache.ReadMemory(address, size);
        });
}

ShaderCache::Variant& ShaderCache::GetShader(
    Shader::HwStage hw_stage, Shader::SwStage sw_stage, const Shader::ShaderParams& params,
    const Shader::RuntimeInfo& runtime, Shader::Backend::Bindings& bindings, bool flip_vertex_y,
    bool sample_rate_shading, const GeometryKernelConfig* geometry,
    const GeometryCopyConfig* geometry_copy, const GeometryExportConfig* geometry_export,
    const VertexFetchLimits* vertex_fetch_limits, const TessellationKernelConfig* tessellation) {
    if (runtime.hw_stage != hw_stage || runtime.sw_stage != sw_stage || params.code.empty()) {
        throw std::runtime_error("Invalid Metal guest shader stage parameters");
    }
    if ((u32(geometry != nullptr) + u32(geometry_copy != nullptr) +
             u32(geometry_export != nullptr) + u32(tessellation != nullptr) >
         1) ||
        (geometry && hw_stage != Shader::HwStage::Geometry) ||
        (geometry_copy &&
         (hw_stage != Shader::HwStage::Vertex || sw_stage != Shader::SwStage::Vertex)) ||
        (geometry_export &&
         (hw_stage != (geometry_export->raster_output ? Shader::HwStage::Vertex
                                                      : Shader::HwStage::Export) ||
          sw_stage != Shader::SwStage::Vertex)) ||
        (tessellation &&
         !((hw_stage == Shader::HwStage::Local && sw_stage == Shader::SwStage::Vertex) ||
           (hw_stage == Shader::HwStage::Hull &&
            sw_stage == Shader::SwStage::TessellationControl) ||
           ((hw_stage == Shader::HwStage::Export || hw_stage == Shader::HwStage::Vertex) &&
            sw_stage == Shader::SwStage::TessellationEval))) ||
        (tessellation &&
         tessellation->raster_output != (hw_stage == Shader::HwStage::Vertex &&
                                         sw_stage == Shader::SwStage::TessellationEval)) ||
        ((geometry || geometry_copy || geometry_export || tessellation) &&
         (flip_vertex_y || sample_rate_shading))) {
        throw std::runtime_error("Invalid Metal geometry shader cache context");
    }
    const auto requested_geometry = geometry ? std::make_optional(*geometry) : std::nullopt;
    const auto requested_copy = geometry_copy ? std::make_optional(*geometry_copy) : std::nullopt;
    const auto requested_export =
        geometry_export ? std::make_optional(*geometry_export) : std::nullopt;
    const auto requested_tessellation =
        tessellation ? std::make_optional(*tessellation) : std::nullopt;
    auto& program = GetProgram(hw_stage, sw_stage, params);
    auto& variants = program.variants;
    boost::container::static_vector<Shader::Info*, 8> refreshed_infos;
    const auto refresh_info = [&](Shader::Info& info) {
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        ShaderCompiler::ConfigureVertexFetch(info, vertex_fetch_limits);
        for (const auto* cached : refreshed_infos) {
            if (ReuseSrtSnapshot(info, *cached)) {
                return;
            }
        }
        info.RefreshFlatBuf();
        if (info.walker_code_owner && refreshed_infos.size() < refreshed_infos.capacity()) {
            refreshed_infos.push_back(&info);
        }
    };
    const u32 shared_memory_size = geometry          ? geometry->shared_memory_size
                                   : geometry_copy   ? geometry_copy->shared_memory_size
                                   : geometry_export ? geometry_export->shared_memory_size
                                   : hw_stage == Shader::HwStage::Compute
                                       ? runtime.hw.cs.shared_memory_size
                                       : 0;
    const auto thread_id_buffers = [](const Shader::Info& info) {
        if (info.buffers.size() > Shader::NUM_BUFFERS) {
            throw std::runtime_error("Metal shader buffer metadata exceeds native capacity");
        }
        std::bitset<Shader::NUM_BUFFERS> values;
        for (size_t index = 0; index < info.buffers.size(); ++index) {
            const auto& resource = info.buffers[index];
            values[index] = resource.buffer_type == Shader::BufferType::Guest &&
                            ReadGuestBufferSharp(resource, info).add_tid_enable;
        }
        return values;
    };
    const auto vertex_fetch_formats = [](const Shader::Info& info) {
        SmallVector<Variant::VertexFetchFormat, 32> values;
        if (info.has_fetch_shader) {
            Shader::Gcn::FetchShaderData fetch;
            if (!Shader::Gcn::ParseFetchShader(info, fetch)) {
                throw std::runtime_error("Metal vertex fetch format requires a valid fetch shader");
            }
            for (const auto& attribute : fetch.attributes) {
                const auto buffer = attribute.GetSharp(info);
                values.push_back({static_cast<AmdGpu::DataFormat>(buffer.data_format),
                                  static_cast<AmdGpu::NumberFormat>(buffer.num_format)});
            }
        }
        return values;
    };
    const auto sampler_lod = [](const Shader::Info& info) {
        std::array<SamplerLodState, Shader::NUM_SAMPLERS> values{};
        if (info.samplers.size() > values.size()) {
            throw std::runtime_error("Metal shader sampler metadata exceeds native capacity");
        }
        u32 index{};
        for (const auto& resource : info.samplers) {
            const auto sampler = resource.GetSharp(info);
            values[index++] = {sampler.LodBias(), sampler.MinLod(), sampler.MaxLod(),
                               HasSamplerAnisotropy(sampler)};
        }
        return values;
    };
    if (!variants.empty()) {
        const auto matches = [&](Variant& variant) {
            if (variant.flip_vertex_y != flip_vertex_y ||
                variant.sample_rate_shading != sample_rate_shading ||
                variant.shared_memory_size != shared_memory_size ||
                variant.geometry != requested_geometry || variant.geometry_copy != requested_copy ||
                variant.geometry_export != requested_export ||
                variant.tessellation != requested_tessellation) {
                return false;
            }
            for (const auto& fetch : variant.fetch_programs) {
                if (!FetchShaderMatches(params.user_data, fetch.sgpr_base, fetch.code,
                                        [this](VAddr address, u64 size) {
                                            buffer_cache.ReadMemory(address, size);
                                        })) {
                    return false;
                }
            }
            auto& info = variant.info;
            refresh_info(info);
            if (vertex_fetch_limits && !variant.fetch_programs.empty()) {
                bool lower_fetch = EmulatorSettings.IsInlineFetchShader();
                Shader::Info fetch_info;
                fetch_info.user_data = info.user_data;
                fetch_info.prepare_memory_read = info.prepare_memory_read;
                fetch_info.read_fetch_shader = info.read_fetch_shader;
                fetch_info.supports_vertex_attribute = info.supports_vertex_attribute;
                fetch_info.has_fetch_shader = true;
                for (const auto& program : variant.fetch_programs) {
                    fetch_info.fetch_shader_sgpr_base = program.sgpr_base;
                    Shader::Gcn::FetchShaderData fetch;
                    if (!Shader::Gcn::ParseFetchShader(fetch_info, fetch)) {
                        throw std::runtime_error("Metal vertex fetch requires valid metadata");
                    }
                    lower_fetch |= fetch.requires_native_lowering;
                }
                if (lower_fetch == info.has_fetch_shader) {
                    return false;
                }
            }
            if (variant.vertex_fetch_formats != vertex_fetch_formats(info)) {
                return false;
            }
            if (!HasCompatibleImageViews(info)) {
                return false;
            }
            if (variant.shader.image_aliases !=
                ImageAliasState::FromInfo(info, variant.shader.image_aliases.eligible)) {
                return false;
            }
            for (u32 index = 0; index < info.images.size(); ++index) {
                if (variant.shader.fmask_dependencies.test(index) &&
                    variant.shader.fmask_images[index] !=
                        FmaskImageState::FromImage(info.images[index].GetSharp(info))) {
                    return false;
                }
                if (variant.shader.descriptor_query_images.test(index) &&
                    !info.images[index].GetSharp(info).Address()) {
                    return false;
                }
                if (variant.shader.minimum_lod_dependencies.test(index)) {
                    const auto& resource = info.images[index];
                    const auto image = resource.GetSharp(info);
                    const auto requested = ImageMinimumLodState::FromImage(image, resource);
                    if (variant.shader.image_minimum_lod[index] != requested) {
                        return false;
                    }
                }
                if (variant.shader.single_mip_gathers.test(index)) {
                    const auto image = info.images[index].GetSharp(info);
                    if (image.base_level != image.last_level) {
                        return false;
                    }
                }
                if (variant.shader.base_mip_gathers.test(index)) {
                    const auto image = info.images[index].GetSharp(info);
                    if (image.base_level || !image.last_level || image.min_lod) {
                        return false;
                    }
                }
            }
            const auto requested_lod = sampler_lod(info);
            for (u32 index = 0; index < info.samplers.size(); ++index) {
                const auto& usage = variant.shader.sampler_lod[index];
                const auto& cached = variant.sampler_lod[index];
                const auto& requested = requested_lod[index];
                if ((usage.native_bias && !NativeSamplerLodBiasCompatible(requested.bias)) ||
                    (usage.bias && cached.bias != requested.bias) ||
                    (usage.clamp && (cached.minimum != requested.minimum ||
                                     cached.maximum != requested.maximum)) ||
                    (usage.anisotropic && cached.anisotropic != requested.anisotropic)) {
                    return false;
                }
            }
            const auto requested =
                SpecializeShader(info, runtime, GetProfile(), bindings, tessellation != nullptr);
            const auto& cached = variant.specialization;
            return variant.shader.unbound_buffers == ShaderCompiler::UnboundBuffers(info) &&
                   variant.thread_id_buffers == thread_id_buffers(info) &&
                   cached.buffers.size() == requested.buffers.size() &&
                   cached.images.size() == requested.images.size() &&
                   cached.samplers.size() == requested.samplers.size() && cached == requested;
        };
        Variant* selected = program.last_used;
        if (!selected || !matches(*selected)) {
            selected = nullptr;
            for (auto& variant : variants) {
                if (variant.get() != program.last_used && matches(*variant)) {
                    selected = variant.get();
                    break;
                }
            }
        }
        if (selected) {
            program.last_used = selected;
            bindings.unified += selected->binding_count.unified;
            bindings.buffer += selected->binding_count.buffer;
            return *selected;
        }
    }
    auto variant = std::make_unique<Variant>();
    variant->info = Shader::Info{hw_stage, sw_stage, params};
    variant->info.prepare_memory_read = [this](VAddr address, u64 size) {
        buffer_cache.ReadMemory(address, size);
    };
    variant->info.read_fetch_shader =
        [this, target = variant.get()](u32 sgpr_base) -> std::span<const u32> {
        for (const auto& fetch : target->fetch_programs) {
            if (fetch.sgpr_base == sgpr_base) {
                return fetch.code;
            }
        }
        auto code =
            ReadFetchShader(target->info.user_data, sgpr_base, [this](VAddr address, u64 size) {
                buffer_cache.ReadMemory(address, size);
            });
        return target->fetch_programs.emplace_back(sgpr_base, std::move(code)).code;
    };
    variant->shared_memory_size = shared_memory_size;
    variant->geometry = requested_geometry;
    variant->geometry_copy = requested_copy;
    variant->geometry_export = requested_export;
    variant->tessellation = requested_tessellation;
    variant->flip_vertex_y = flip_vertex_y;
    variant->sample_rate_shading = sample_rate_shading;
    auto next_bindings = bindings;
    const auto name = fmt::format("metal4_{}_{:016x}_{:016x}_{}", static_cast<u32>(hw_stage),
                                  params.hash, program.code_hash, variants.size());
    variant->shader =
        shaders.CompileGuest(variant->info, runtime, program.code, next_bindings, name,
                             flip_vertex_y, sample_rate_shading, geometry, geometry_copy,
                             geometry_export, vertex_fetch_limits, tessellation);
    for (const auto& cached : variants) {
        if (ReuseSrtWalker(variant->info, cached->info)) {
            break;
        }
    }
    variant->binding_count = {next_bindings.unified - bindings.unified,
                              next_bindings.buffer - bindings.buffer};
    variant->sampler_lod = sampler_lod(variant->info);
    variant->thread_id_buffers = thread_id_buffers(variant->info);
    variant->vertex_fetch_formats = vertex_fetch_formats(variant->info);
    variant->specialization =
        SpecializeShader(variant->info, runtime, GetProfile(), bindings, tessellation != nullptr);
    auto& result = *variants.emplace_back(std::move(variant));
    program.last_used = &result;
    bindings = next_bindings;
    return result;
}

id<MTLComputePipelineState> ShaderCache::GetComputePipeline(Variant& variant) {
    if (variant.shader.translated.stage != Shader::SwStage::Compute) {
        throw std::runtime_error("Metal compute pipeline requires a compute shader");
    }
    if (!variant.compute_pipeline.valid()) {
        @autoreleasepool {
            const auto size =
                variant.geometry || variant.geometry_copy || variant.geometry_export
                    ? std::array<u32, 3>{64, 1, 1}
                : variant.tessellation
                    ? std::array<u32, 3>{variant.tessellation->threads_per_group, 1, 1}
                    : variant.specialization.runtime_info.hw.cs.workgroup_size;
            const MTLSize limit = instance.GetDevice().maxThreadsPerThreadgroup;
            u64 total_threads{1};
            for (const u32 dimension : size) {
                if (!dimension || total_threads > std::numeric_limits<u64>::max() / dimension) {
                    throw std::runtime_error("Invalid Metal guest workgroup dimensions");
                }
                total_threads *= dimension;
            }
            if (size[0] > limit.width || size[1] > limit.height || size[2] > limit.depth ||
                total_threads > limit.width) {
                throw std::runtime_error("Metal guest workgroup exceeds device limits");
            }
            const auto& entry = variant.shader.translated.entry_point;
            MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
            function.library = variant.shader.library.get();
            function.name = [[NSString alloc] initWithBytes:entry.data()
                                                     length:entry.size()
                                                   encoding:NSUTF8StringEncoding];
            MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
            descriptor.computeFunctionDescriptor = function;
            descriptor.requiredThreadsPerThreadgroup = MTLSizeMake(size[0], size[1], size[2]);
            descriptor.maxTotalThreadsPerThreadgroup = total_threads;
            variant.compute_pipeline = pipelines.CompileComputePipelineAsync(descriptor);
        }
    }
    id<MTLComputePipelineState> pipeline = variant.compute_pipeline.get();
    if (!pipeline || pipeline.threadExecutionWidth != GetProfile().subgroup_size) {
        throw std::runtime_error("Metal guest pipeline has an incompatible SIMD-group width");
    }
    return pipeline;
}

} // namespace Metal
