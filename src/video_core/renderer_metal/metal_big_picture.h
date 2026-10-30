// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "imgui/renderer/texture_manager.h"

struct SDL_Window;

namespace Metal {

class BigPictureRenderer {
public:
    explicit BigPictureRenderer(SDL_Window* window);
    ~BigPictureRenderer();

    BigPictureRenderer(const BigPictureRenderer&) = delete;
    BigPictureRenderer& operator=(const BigPictureRenderer&) = delete;

    [[nodiscard]] ImGui::Core::TextureManager::NativeTexture Upload(std::span<const u8> pixels,
                                                                    u32 width, u32 height);
    void Render(ImDrawData& data);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
