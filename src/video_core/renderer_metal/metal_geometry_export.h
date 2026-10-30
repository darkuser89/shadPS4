// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_metal/metal_geometry_address.h"
#include "video_core/renderer_metal/metal_geometry_resources.h"
#include "video_core/renderer_metal/metal_tessellation_raster.h"

namespace Metal {

struct GeometryExportInput {
    std::array<u32, 16> system_sgprs;
    std::array<u32, 4> vgprs;
    u32 active{1};
};
static_assert(sizeof(GeometryExportInput) == 84);

struct RasterIndirectRegisters {
    u16 vertex_sgpr_offset{};
    u16 instance_sgpr_offset{};

    bool operator==(const RasterIndirectRegisters&) const = default;

    [[nodiscard]] static std::optional<u32> UserDataIndex(u16 offset) {
        if (!offset)
            return std::nullopt;
        if (offset < 0x4c || offset >= 0x5c)
            throw std::runtime_error("Invalid native raster indirect SGPR offset");
        return offset - 0x4c;
    }

    void Validate() const {
        static_cast<void>(UserDataIndex(vertex_sgpr_offset));
        static_cast<void>(UserDataIndex(instance_sgpr_offset));
    }
};

struct GeometryIndirectHeader {
    std::array<float, 4> viewport;
    u32 base_vertex;
    u32 first_instance;
};
static_assert(sizeof(GeometryIndirectHeader) == 24);

struct GeometryExportConfig {
    u32 work_items;
    u32 shared_memory_size{};
    bool raster_output{};
    bool dynamic_work_items{};
    std::optional<RasterIndirectRegisters> indirect;

    bool operator==(const GeometryExportConfig&) const = default;

    [[nodiscard]] u32 ResolveWorkItems(u32 runtime_items = 0) const {
        if ((indirect && (!dynamic_work_items || !raster_output)) ||
            (dynamic_work_items && (!raster_output || work_items)) ||
            (!dynamic_work_items && runtime_items && runtime_items != work_items)) {
            throw std::runtime_error("Invalid native export dispatch context");
        }
        if (indirect)
            indirect->Validate();
        const u32 count = dynamic_work_items ? runtime_items : work_items;
        if (!count || count % 64) {
            throw std::runtime_error("Invalid native export dispatch size");
        }
        return count;
    }

    [[nodiscard]] u64 HeaderSizeBytes() const noexcept {
        return !raster_output ? 0 : indirect ? sizeof(GeometryIndirectHeader) : 4 * sizeof(float);
    }

    [[nodiscard]] u64 InputSizeBytes(u32 runtime_items = 0) const {
        return u64{ResolveWorkItems(runtime_items)} * sizeof(GeometryExportInput) +
               HeaderSizeBytes();
    }

    [[nodiscard]] u32 WorkItemsFromInputBytes(u64 bytes) const {
        if (!dynamic_work_items) {
            return ResolveWorkItems();
        }
        const u64 HeaderBytes = HeaderSizeBytes();
        if (bytes <= HeaderBytes || (bytes - HeaderBytes) % sizeof(GeometryExportInput) ||
            (bytes - HeaderBytes) / sizeof(GeometryExportInput) > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Invalid native dynamic raster input extent");
        }
        const u32 count =
            ResolveWorkItems(u32((bytes - HeaderBytes) / sizeof(GeometryExportInput)));
        if (InputSizeBytes(count) / sizeof(u32) > u64{std::numeric_limits<u32>::max()} + 1) {
            throw std::runtime_error("Native dynamic raster inputs exceed shader addressing");
        }
        return count;
    }
};

struct GeometryExportBindings {
    u32 inputs;
    u32 first_binding{};
    std::shared_ptr<const GeometryBufferAddressOrigins> address_origins;
    std::shared_ptr<const GeometryBufferAddressProgram> address_program;
    u64 input_sources{};
    std::optional<TessellationRasterBindings> raster;

    [[nodiscard]] u32 InputBinding() const {
        return first_binding + inputs;
    }

    [[nodiscard]] u64 RasterSizeBytes(const GeometryExportConfig& config,
                                      u32 runtime_items = 0) const {
        if (!config.raster_output || !raster || raster->vertex_words < 5 ||
            raster->attributes.size() + 1 != raster->vertex_words || raster->output == inputs) {
            throw std::runtime_error("Invalid native vertex raster payload layout");
        }
        const u64 words = u64{config.ResolveWorkItems(runtime_items)} * raster->vertex_words;
        if (words > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Native vertex raster payload exceeds shader addressing");
        }
        return words * sizeof(u32);
    }
};

class GeometryExportTranslation {
public:
    GeometryExportTranslation(const GeometryExportConfig& config,
                              const Shader::RuntimeInfo& runtime, const Shader::Profile& profile);

    [[nodiscard]] Shader::TranslationHooks Hooks();
    [[nodiscard]] GeometryExportBindings Bindings() const {
        return bindings;
    }

private:
    GeometryExportConfig config;
    u32 num_user_data;
    GeometryExportBindings bindings{};
    std::optional<TessellationRasterTranslation> raster;
};

} // namespace Metal
