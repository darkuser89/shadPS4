// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/pixel_format.h"
#include "video_core/renderer_metal/metal_common.h"

namespace AmdGpu {
struct Buffer;
}

namespace Metal {

[[nodiscard]] MTLVertexFormat VertexFormat(const AmdGpu::Buffer& buffer);

[[nodiscard]] MTLPixelFormat SurfaceFormat(AmdGpu::DataFormat data, AmdGpu::NumberFormat number);
[[nodiscard]] MTLPixelFormat DepthFormat(MTLPixelFormat format);
[[nodiscard]] MTLPixelFormat LinearFormat(MTLPixelFormat format);
[[nodiscard]] MTLPixelFormat StorageFormat(MTLPixelFormat format);
[[nodiscard]] MTLTextureSwizzleChannels ComponentMapping(AmdGpu::CompMapping mapping);

} // namespace Metal
