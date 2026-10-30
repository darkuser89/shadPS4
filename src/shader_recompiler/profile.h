// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader {

struct Profile {
    u32 max_viewport_width{};
    u32 max_viewport_height{};
    u32 max_shared_memory_size{};
    u32 supported_spirv{0x00010000};
    u32 subgroup_size{};
    u32 sparse_page_shift{};
    u32 guest_page_shift{};
    u32 guest_page_leaf_bits{};
    bool support_int8{};
    bool support_int16{};
    bool support_int64{};
    bool support_float16{};
    bool support_float64{};
    bool supports_denorm_behavior_independence{};
    bool supports_rounding_mode_independence{};
    bool support_fp16_denorm_preserve{};
    bool support_fp16_denorm_flush{};
    bool support_fp16_round_to_zero{};
    bool support_fp32_denorm_preserve{};
    bool support_fp32_denorm_flush{};
    bool support_fp32_round_to_zero{};
    bool support_fp64_denorm_preserve{};
    bool support_fp64_denorm_flush{};
    bool support_fp64_round_to_zero{};
    bool support_fp16_signed_zero_inf_nan_preserve{};
    bool support_fp32_signed_zero_inf_nan_preserve{};
    bool support_fp64_signed_zero_inf_nan_preserve{};
    bool supports_image_load_store_lod{};
    bool supports_native_cube_calc{};
    bool supports_trinary_minmax{};
    bool supports_buffer_fp32_atomic_min_max{};
    bool supports_image_fp32_atomic_min_max{};
    bool supports_buffer_int64_atomics{};
    bool supports_shared_int64_atomics{};
    bool supports_workgroup_explicit_memory_layout{};
    bool supports_amd_shader_explicit_vertex_parameter{};
    bool supports_fragment_shader_barycentric{};
    bool supports_shader_subgroup_clock{};
    bool has_broken_spirv_clamp{};
    bool lower_left_origin_mode{};
    bool needs_manual_interpolation{};
    bool needs_lds_barriers{};
    bool needs_storage_lds_barriers{};
    bool needs_guest_compute_barriers{};
    bool lower_shared_memory_after_wave64{};
    bool needs_geometry_output_lowering{};
    bool needs_geometry_message_predication{};
    bool needs_vector_register_predication{};
    bool needs_memory_predication{};
    bool needs_geometry_copy_lowering{};
    bool needs_geometry_export_lowering{};
    bool needs_vertex_fetch_lowering{};
    bool needs_bounded_atomic_counters{};
    bool needs_extended_buffer_atomics{};
    bool needs_extended_ds_atomics{};
    bool needs_software_shared_int64_atomics{};
    bool needs_typed_image_atomics{};
    bool needs_multisample_array_images{};
    bool needs_software_float_atomics{};
    bool needs_buffer_offsets{};
    bool needs_wide_buffer_byte_offsets{};
    bool needs_unorm_fixup{};
    bool needs_unorm32_depth_conversion{};
    bool needs_depth_compare_gradients{};
    bool needs_depth_compare_explicit_lod{};
    bool needs_image_gradients{};
    bool needs_image_sample_clamp{};
    bool needs_image_gather_lod{};
    bool needs_image_query_lod_sampler{};
    bool needs_clip_distance_emulation{};
    bool supports_shader_stencil_export{};
    bool needs_fmask_lowering{};

    bool operator==(const Profile&) const = default;
};

} // namespace Shader
