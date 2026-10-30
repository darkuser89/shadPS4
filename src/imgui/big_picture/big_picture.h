// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <vector>
#include <SDL3/SDL.h>
#include <imgui.h>

#include "common/types.h"
#include "imgui/renderer/texture_manager.h"

namespace BigPictureMode {

struct IconInfo {
    ImTextureID textureId{};
    std::shared_ptr<void> textureLifetime;
    std::filesystem::path ebootPath;
    std::string title;
    std::string serial;
    bool focusState;
};

void Launch(char* executableName, bool sameProcess);
void GetGameIconInfo(std::vector<IconInfo>& icons);
ImGui::Core::TextureManager::NativeTexture LoadTextureData(std::vector<u8> data);
ImGui::Core::TextureManager::NativeTexture LoadTextureDataFromFile(std::filesystem::path filePath);

} // namespace BigPictureMode
