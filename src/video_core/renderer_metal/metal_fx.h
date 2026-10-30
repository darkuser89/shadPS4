// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#import <MetalFX/MetalFX.h>
#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

class Instance;
class Scheduler;
class GuiRenderer;

class MetalFxPass {
public:
    MetalFxPass(const Instance& instance, Scheduler& scheduler, GuiRenderer& converter);
    [[nodiscard]] bool IsSupported() const;
    void Render(id<MTLTexture> source, id<MTLTexture> target, float gamma,
                MTLFXSpatialScalerColorProcessingMode mode);

private:
    const Instance& instance;
    Scheduler& scheduler;
    GuiRenderer& converter;
    id<MTL4FXSpatialScaler> scaler;
    id<MTLTexture> input;
    id<MTLTexture> output;
    id<MTLFence> fence;
    bool has_encoded{};
};

} // namespace Metal
