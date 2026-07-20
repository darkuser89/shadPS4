// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/scope_exit.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include <xxhash.h>

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 3u;
// SRT metadata embeds native walker machine code. Keep ARM64 and x86-64
// metadata incompatible so switching between FEX-native and Rosetta builds can
// never execute cached code for the other host ISA.
static constexpr u32 ShaderMetaVersion = 2u
#ifdef ARCH_ARM64
                                         | 0x80000000u
#endif
    ;
static constexpr u32 PipelineKeyVersion = 2u;
} // namespace Serialization

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, u64 expected_perm_hash, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    if (perm_hash_ar != expected_perm_hash) {
        LOG_WARNING(Render,
                    "Ignoring shader metadata with mismatched permutation hash {:#x} "
                    "(expected {:#x})",
                    perm_hash_ar, expected_perm_hash);
        return false;
    }
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    SCOPE_EXIT {
        infos.fill(nullptr);
        modules.fill(nullptr);
        fetch_shader.reset();
    };

    compute_key.Deserialize(ar);

    // Archives may contain multiple entries for a key. Avoid repeatedly restoring the same shader
    // modules, which is especially expensive on translation drivers.
    if (compute_pipelines.contains(compute_key)) {
        return true;
    }

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    if (!LoadPipelineStage(0, compute_key.value)) {
        return false;
    }

    if (DeferPipelineCreationDuringWarmup()) {
        // Preserve the key as a null placeholder. First use creates the VkPipeline without
        // serializing the already known key a second time.
        compute_pipelines.try_emplace(compute_key);
        return true;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    std::scoped_lock cache_lock{driver_pipeline_cache_mutex};
    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);
    driver_pipeline_cache_dirty = true;

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    SCOPE_EXIT {
        infos.fill(nullptr);
        modules.fill(nullptr);
        fetch_shader.reset();
    };

    graphics_key.Deserialize(ar);

    if (graphics_pipelines.contains(graphics_key)) {
        return true;
    }

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        if (!LoadPipelineStage(stage_idx, hash)) {
            return false;
        }
    }

    if (DeferPipelineCreationDuringWarmup()) {
        graphics_pipelines.try_emplace(graphics_key);
        return true;
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    std::scoped_lock cache_lock{driver_pipeline_cache_mutex};
    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);
    driver_pipeline_cache_dirty = true;

    return true;
}

bool PipelineCache::LoadPipelineStage(size_t stage, u64 permutation_hash) {
    if (const auto restored = restored_shader_stages.find(permutation_hash);
        restored != restored_shader_stages.end()) {
        auto& restored_program = *restored->second.program;
        const auto& permutation = restored_program.modules[restored->second.permutation_index];
        infos[stage] = &restored_program.info;
        modules[stage] = permutation.module;
        if (permutation.spec.fetch_shader_data) {
            fetch_shader = permutation.spec.fetch_shader_data;
        }
        return true;
    }

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", permutation_hash), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }
    Serialization::Archive ar{std::move(meta_blob)};

    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    std::optional<Shader::Gcn::FetchShaderData> stage_fetch_shader;
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, permutation_hash, program->info, stage_fetch_shader, spec, perm_idx)) {
        return false;
    }
    if (stage_fetch_shader) {
        fetch_shader = std::move(stage_fetch_shader);
    }

    Program* cached_program{};
    const auto cached_program_it = program_cache.find(program->info.pgm_hash);
    if (cached_program_it != program_cache.end()) {
        cached_program = cached_program_it->second.get();
        const auto& it = std::ranges::find(cached_program->modules, spec, &Program::Module::spec);
        if (it != cached_program->modules.end()) {
            // The same specialization can be referenced by stale cache metadata with a different
            // permutation index. The already loaded module is equivalent, so reuse it rather than
            // aborting cache warmup or inserting the same specialization twice.
            const auto idx = std::distance(cached_program->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render,
                            "Ignoring stale permutation index {} for {} shader {:#x}; already "
                            "loaded at {}",
                            perm_idx, program->info.stage, program->info.pgm_hash, idx);
            }
            infos[stage] = &cached_program->info;
            modules[stage] = it->module;
            restored_shader_stages.try_emplace(
                permutation_hash, RestoredShaderStage{cached_program, static_cast<size_t>(idx)});
            return true;
        }
    }

    // KosmicKrisp translates SPIR-V further when a pipeline is created. Keep the serialized
    // specialization available, but defer even the VkShaderModule creation until a draw actually
    // uses it. This avoids restoring every historical shader at startup together with pipelines
    // that are already created on demand on this driver.
    vk::ShaderModule module{};
    if (!DeferShaderModuleCreationDuringWarmup()) {
        // Only touch the SPIR-V file when this shader permutation has not already been restored by
        // a different pipeline. Titles commonly share stages across hundreds of pipeline keys.
        std::vector<u32> spv;
        Storage::DataBase::Instance().Load(
            Storage::BlobType::ShaderBinary,
            fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx), spv);
        if (spv.empty()) {
            return false;
        }
        module = CompileSPV(spv, instance.GetDevice());
    }
    if (cached_program == nullptr) {
        const auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
        ASSERT(new_program);
        it_pgm.value() = std::move(program);
        cached_program = it_pgm.value().get();
    }

    // Permutation hash depends on shader variation index. To prevent collisions, insert it at the
    // exact position rather than append.
    spec.info = &cached_program->info;
    cached_program->InsertPermut(module, std::move(spec), perm_idx);
    restored_shader_stages.try_emplace(permutation_hash,
                                       RestoredShaderStage{cached_program, perm_idx});

    infos[stage] = &cached_program->info;
    modules[stage] = module;

    return true;
}

void PipelineCache::RestoreDriverPipelineCache() {
    // Opaque driver blobs are already compressed and are replaced once per clean shutdown. Keep
    // them out of the append-only archive mode to avoid duplicate ZIP entries and cache growth.
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        return;
    }
    std::vector<u8> cache_data;
    Storage::DataBase::Instance().Load(Storage::BlobType::VulkanPipelineCache, "vulkan",
                                       cache_data);
    if (cache_data.empty()) {
        return;
    }

    // Vulkan pipeline-cache blobs are driver specific. Validate the standard header ourselves so
    // an old cache from another GPU or driver is never handed to the active implementation.
    if (cache_data.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
        LOG_WARNING(Render, "Ignoring truncated Vulkan pipeline cache");
        return;
    }
    VkPipelineCacheHeaderVersionOne header{};
    std::memcpy(&header, cache_data.data(), sizeof(header));
    const auto uuid = instance.GetPipelineCacheUUID();
    if (header.headerSize < sizeof(header) || header.headerSize > cache_data.size() ||
        header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
        header.vendorID != instance.GetVendorID() || header.deviceID != instance.GetDeviceID() ||
        std::memcmp(header.pipelineCacheUUID, uuid.data(), VK_UUID_SIZE) != 0) {
        LOG_INFO(Render, "Ignoring Vulkan pipeline cache created by a different device or driver");
        return;
    }

    const vk::PipelineCacheCreateInfo cache_ci{
        .initialDataSize = cache_data.size(),
        .pInitialData = cache_data.data(),
    };
    auto [result, restored_cache] = instance.GetDevice().createPipelineCacheUnique(cache_ci);
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render, "Failed to restore Vulkan pipeline cache: {}", vk::to_string(result));
        return;
    }
    pipeline_cache = std::move(restored_cache);
    driver_pipeline_cache_hash = XXH3_64bits(cache_data.data(), cache_data.size());
    LOG_INFO(Render, "Restored {} KiB Vulkan pipeline cache", cache_data.size() / 1024);
}

void PipelineCache::SaveDriverPipelineCache() {
    if (!pipeline_cache || !Storage::DataBase::Instance().IsOpened() ||
        EmulatorSettings.IsPipelineCacheArchived()) {
        return;
    }
    std::scoped_lock cache_lock{driver_pipeline_cache_mutex};
    if (!driver_pipeline_cache_dirty) {
        return;
    }
    auto [result, cache_data] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render, "Failed to serialize Vulkan pipeline cache: {}", vk::to_string(result));
        return;
    }
    const u64 cache_hash = XXH3_64bits(cache_data.data(), cache_data.size());
    if (cache_hash == driver_pipeline_cache_hash) {
        driver_pipeline_cache_dirty = false;
        return;
    }
    if (Storage::DataBase::Instance().Save(Storage::BlobType::VulkanPipelineCache, "vulkan",
                                           std::move(cache_data))) {
        driver_pipeline_cache_dirty = false;
        driver_pipeline_cache_hash = cache_hash;
    }
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Starting a new "
                    "cache generation",
                    profile_data.size(), sizeof(Shader::Profile));
        if (!Storage::DataBase::Instance().Reset()) {
            return;
        }
        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Starting a new cache "
                    "generation");
        if (!Storage::DataBase::Instance().Reset()) {
            return;
        }
        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }

    RestoreDriverPipelineCache();

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        });

    if (DeferPipelineCreationDuringWarmup()) {
        LOG_INFO(Render,
                 "Prepared {} cached pipelines from {} shader permutations for on-demand "
                 "creation",
                 num_pipelines, restored_shader_stages.size());
    } else {
        LOG_INFO(Render, "Preloaded {} pipelines from {} shader permutations", num_pipelines,
                 restored_shader_stages.size());
    }
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    SaveDriverPipelineCache();
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (fetch_shader_data) {
        spec.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
