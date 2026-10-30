# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "26.0")
    message(FATAL_ERROR "The native Metal 4 renderer requires macOS 26.0 or later")
endif()

set(SPIRV_CROSS_CLI OFF)
set(SPIRV_CROSS_ENABLE_TESTS OFF)
set(SPIRV_CROSS_ENABLE_HLSL OFF)
set(SPIRV_CROSS_ENABLE_CPP OFF)
set(SPIRV_CROSS_ENABLE_REFLECT OFF)
set(SPIRV_CROSS_ENABLE_C_API OFF)
set(SPIRV_CROSS_ENABLE_UTIL OFF)
set(SPIRV_CROSS_SKIP_INSTALL ON)
FetchContent_Declare(spirv_cross
    URL https://codeload.github.com/KhronosGroup/SPIRV-Cross/tar.gz/aa217aeb6c9f0ace7a0ab233b28807edf45eb165
    URL_HASH SHA256=b99db6ee6182dbeb102257e755bf6d4978c23dcfd3329033304779279bc27212
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(spirv_cross)

if(TARGET glslang-standalone)
    set(METAL_GLSLANG glslang-standalone)
elseif(TARGET glslang::glslang-standalone)
    set(METAL_GLSLANG glslang::glslang-standalone)
else()
    message(FATAL_ERROR "The native Metal FSR shaders require the glslang compiler")
endif()

set(METAL_FSR_HEADERS "")
foreach(PASS IN ITEMS easu rcas easu_hdr rcas_hdr)
    set(HEADER "${CMAKE_CURRENT_BINARY_DIR}/metal_fsr_${PASS}.h")
    string(TOUPPER "metal_fsr_${PASS}" SYMBOL)
    if(PASS MATCHES "^easu")
        set(EASU 1)
    else()
        set(EASU 0)
    endif()
    if(PASS MATCHES "_hdr$")
        set(FULL_PRECISION 1)
    else()
        set(FULL_PRECISION 0)
    endif()
    add_custom_command(
        OUTPUT "${HEADER}"
        COMMAND ${METAL_GLSLANG} --target-env spirv1.6 -V
                -I${PROJECT_SOURCE_DIR}/src -DFSR_EASU=${EASU}
                -DFSR_FULL_PRECISION=${FULL_PRECISION}
                --vn ${SYMBOL} --depfile "${HEADER}.d" -o "${HEADER}"
                "${CMAKE_CURRENT_SOURCE_DIR}/metal_fsr.comp"
        COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${HEADER}"
                -P "${PROJECT_SOURCE_DIR}/src/video_core/host_shaders/AddShaderHeaderInclude.cmake"
        DEPENDS ${METAL_GLSLANG} metal_fsr.comp
                "${PROJECT_SOURCE_DIR}/src/video_core/host_shaders/AddShaderHeaderInclude.cmake"
        DEPFILE "${HEADER}.d"
        BYPRODUCTS "${HEADER}.d"
        VERBATIM
    )
    list(APPEND METAL_FSR_HEADERS "${HEADER}")
endforeach()

add_library(shadps4_metal4 STATIC
    ${METAL_FSR_HEADERS}
    metal_color.h
    metal_cmask.h
    metal_cmask.mm
    metal_xmask.h
    metal_xmask.mm
    metal_htile.h
    metal_htile.mm
    metal_metadata_coverage.h
    metal_metadata_coverage.mm
    metal_big_picture.h
    metal_big_picture.mm
    metal_binding.h
    metal_binding.mm
    metal_buffer_cache.h
    metal_buffer_cache.mm
    metal_guest_backing.h
    metal_guest_write_alias.h
    metal_guest_physical_mappings.h
    metal_guest_read_watches.h
    metal_guest_protection.h
    metal_commands.h
    metal_commands.mm
    metal_depth_conversion.h
    metal_depth_conversion.mm
    metal_compute.h
    metal_compute.mm
    metal_common.h
    metal_format.h
    metal_format.mm
    metal_fsr.h
    metal_fsr.mm
    metal_fsr.comp
    metal_fx.h
    metal_fx.mm
    metal_geometry_address.h
    metal_geometry_address.cpp
    metal_geometry.h
    metal_geometry.mm
    metal_geometry_draw.mm
    metal_geometry_raster.mm
    metal_tessellation_mesh.mm
    metal_geometry_layout.h
    metal_geometry_work.h
    metal_geometry_work.cpp
    metal_geometry_work.mm
    metal_geometry_mesh.cpp
    metal_geometry_mesh.h
    metal_geometry_viewport.mm
    metal_geometry_shader.cpp
    metal_geometry_shader.h
    metal_geometry_copy.cpp
    metal_geometry_copy.mm
    metal_geometry_copy.h
    metal_geometry_export.cpp
    metal_geometry_export.h
    metal_geometry_output.cpp
    metal_geometry_output.h
    metal_geometry_resources.cpp
    metal_geometry_resources.h
    metal_graphics_pipeline.h
    metal_graphics_pipeline.mm
    metal_graphics_state.h
    metal_graphics.h
    metal_graphics.mm
    metal_graphics_runtime.cpp
    metal_graphics_runtime.h
    metal_tessellation_constants.cpp
    metal_tessellation_constants.h
    metal_tessellation_translate.cpp
    metal_tessellation_kernel.cpp
    metal_tessellation_wave.cpp
    metal_tessellation_control.cpp
    metal_tessellation_control.h
    metal_tessellation_scalar.cpp
    metal_tessellation_scalar.h
    metal_tessellation_kernel.h
    metal_tessellation_raster.h
    metal_tessellation_raster.cpp
    metal_tessellation_work.cpp
    metal_tessellation_work.h
    metal_tessellation_dispatch.mm
    metal_tessellation_dispatch.h
    metal_tessellation_factors.mm
    metal_tessellation_factors.h
    metal_tessellation_domains.mm
    metal_tessellation_domains.h
    metal_tessellation_evaluation.mm
    metal_tessellation_evaluation.h
    metal_tessellation_exports.mm
    metal_tessellation_exports.h
    metal_tessellation_geometry.mm
    metal_tessellation_geometry.h
    metal_tessellator_source.h
    metal_gui.h
    metal_gui.mm
    metal_presenter.h
    metal_presenter.mm
    metal_guest_resources.h
    metal_guest_resources.mm
    metal_guest_page_layout.h
    metal_guest_page_table.h
    metal_guest_page_table.mm
    metal_image_info.h
    metal_image_info.mm
    metal_indirect.h
    metal_indirect.mm
    metal_indirect_work.h
    metal_indirect_work.cpp
    metal_instance.h
    metal_instance.mm
    metal_pipeline_archive.h
    metal_pipeline_archive.mm
    metal_pipeline.h
    metal_pipeline.mm
    metal_primitive.h
    metal_primitive.mm
    metal_platform.cpp
    metal_platform.h
    metal_rasterizer.h
    metal_rasterizer.mm
    metal_render_state.h
    metal_sampler.h
    metal_sampler.mm
    metal_scheduler.h
    metal_scheduler.mm
    metal_shader.h
    metal_guest_shader.h
    metal_fetch_shader.h
    metal_shader.mm
    metal_shader_binding.h
    metal_shader_binding.mm
    metal_shader_cache.h
    metal_shader_cache.mm
    metal_staging_buffer.h
    metal_staging_buffer.mm
    metal_swapchain.h
    metal_swapchain.mm
    metal_texture_transfer.h
    metal_texture_transfer.mm
    metal_texture_cache.h
    metal_texture_cache.mm
    metal_tiling.h
    metal_tiling.mm
    metal_tiling_shader.h
    metal_visibility.h
    metal_visibility.mm
)

set_target_properties(shadps4_metal4 PROPERTIES
    OBJCXX_STANDARD 23
    OBJCXX_STANDARD_REQUIRED ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_compile_options(shadps4_metal4 PRIVATE -fobjc-arc)
target_compile_definitions(shadps4_metal4 PRIVATE IMGUI_USER_CONFIG="imgui/imgui_config.h")
target_include_directories(shadps4_metal4 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(shadps4_metal4 PRIVATE fmt::fmt Boost::headers magic_enum::magic_enum
    nlohmann_json::nlohmann_json spirv-cross-msl Dear_ImGui SDL3::SDL3 stb::headers
    "-framework CoreGraphics" "-framework AppKit"
    "-framework Metal" "-framework MetalFX" "-framework Foundation" "-framework QuartzCore")
