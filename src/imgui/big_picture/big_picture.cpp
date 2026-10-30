//  SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <fstream>
#include <stb_image.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/file_format/psf.h"
#include "core/file_sys/fs.h"
#include "core/file_sys/ifile.h"
#include "emulator.h"
#include "imgui/big_picture/big_picture.h"
#include "imgui/big_picture/imgui_impl_sdl3_big_picture.h"
#include "imgui/big_picture/imgui_impl_sdlrenderer3.h"
#include "imgui/big_picture/settings_dialog_imgui.h"
#include "imgui/imgui_std.h"
#include "imgui/renderer/font_stack.h"
#include "sdl_window.h"
#ifdef __APPLE__
#include "video_core/renderer_metal/metal_big_picture.h"
#include "video_core/renderer_metal/metal_platform.h"
#endif

namespace BigPictureMode {

constexpr float gameImageSize = 200.f;

bool done = false;
bool showSettings = false;

std::filesystem::path runEbootPath = "";
std::vector<IconInfo> gameIcons = {};

float uiScale = 1.0f;
SDL_Renderer* renderer;
#ifdef __APPLE__
std::unique_ptr<Metal::BigPictureRenderer> metal_renderer;
#endif

namespace {

std::filesystem::path UpdateChecker(const std::string sceItem, std::filesystem::path game_folder) {
    std::filesystem::path updatedPath = "";
    std::filesystem::path basePath = game_folder.parent_path();
    std::string fileName;
    std::string item = "sce_sys/" + sceItem;

    if (Core::FileSys::IsZArchiveFile(game_folder)) {
        fileName = Core::FileSys::StripZArchiveExtension(game_folder).filename().string();
    } else {
        fileName = game_folder.filename().string();
    }

    if (std::filesystem::exists(basePath / (fileName + "-UPDATE") / item)) {
        updatedPath = basePath / (fileName + "-UPDATE") / item;
    } else if (Core::FileSys::ResolveGameFilePath(basePath / (fileName + "-UPDATE.zar"), item)
                   .has_value()) {
        updatedPath =
            Core::FileSys::ResolveGameFilePath(basePath / (fileName + "-UPDATE.zar"), item).value();
    } else if (std::filesystem::exists(basePath / (fileName + "-patch") / item)) {
        updatedPath = basePath / (fileName + "-patch") / item;
    } else if (Core::FileSys::ResolveGameFilePath(basePath / (fileName + "-patch.zar"), item)
                   .has_value()) {
        updatedPath =
            Core::FileSys::ResolveGameFilePath(basePath / (fileName + "-patch.zar"), item).value();
    } else if (Core::FileSys::ResolveGameFilePath(game_folder, item).has_value()) {
        updatedPath = Core::FileSys::ResolveGameFilePath(game_folder, item).value();
    }

    return updatedPath;
}

void SetGameIcons(std::vector<IconInfo>& gameIcons) {
    ImGuiStyle& style = ImGui::GetStyle();
    const float maxAvailableWidth = ImGui::GetContentRegionAvail().x;
    const float itemSpacing = style.ItemSpacing.x; // already scaled
    const float padding = 10.0f * uiScale;
    float rowContentWidth = gameImageSize * uiScale + itemSpacing;

    for (int i = 0; i < gameIcons.size(); i++) {
        ImGui::BeginGroup();
        std::string ButtonName = "Button" + std::to_string(i);
        const char* ButtonNameChar = ButtonName.c_str();

        bool buttonFocused = (ImGui::GetID(ButtonNameChar) == ImGui::GetFocusID());
        if (buttonFocused) {
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered]);
        }

        ImTextureID id = gameIcons[i].textureId;
        if (id != nullptr) {
            if (ImGui::ImageButton(ButtonNameChar, id,
                                   ImVec2(gameImageSize * uiScale, gameImageSize * uiScale))) {
                done = true;
                Core::FileSys::MntPoints::ignore_game_patches =
                    ImGui::IsKeyDown(ImGuiKey::ImGuiKey_LeftCtrl);
                runEbootPath = gameIcons[i].ebootPath;
            }
        }

        if (buttonFocused) {
            ImGui::PopStyleColor();
        }

        // Scroll to item only when newly-focused
        if (ImGui::IsItemFocused() && !gameIcons[i].focusState) {
            ImGui::SetScrollHereY(0.5f);
        }

        if (ImGui::IsWindowFocused()) {
            gameIcons[i].focusState = ImGui::IsItemFocused();
        }

        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + gameImageSize * uiScale);
        ImGui::TextWrapped("%s", gameIcons[i].title.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();

        // Use same line if content fits horizontally, move to next line if not
        rowContentWidth += (gameImageSize * uiScale + itemSpacing * 2 + padding);
        if (rowContentWidth < maxAvailableWidth) {
            ImGui::SameLine(0.0f, padding);
        } else {
            ImGui::Dummy(ImVec2(0.0f, padding));
            rowContentWidth = gameImageSize * uiScale + itemSpacing;
        }
    }
}

} // namespace

ImGui::Core::TextureManager::NativeTexture LoadTextureData(std::vector<u8> data) {
    int width{}, height{};
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{
        stbi_load_from_memory(data.data(), static_cast<int>(data.size()), &width, &height, nullptr,
                              4),
        stbi_image_free};
    if (!pixels) {
        LOG_ERROR(ImGui, "Failed to load image: {}", stbi_failure_reason());
        return {};
    }
#ifdef __APPLE__
    if (!metal_renderer) {
        LOG_ERROR(ImGui, "Metal 4 Big Picture renderer is not initialized");
        return {};
    }
    return metal_renderer->Upload({pixels.get(), size_t{static_cast<u32>(width)} * height * 4},
                                  width, height);
#else
    auto* surface =
        SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32, pixels.get(), width * 4);
    if (!surface) {
        LOG_ERROR(ImGui, "Unable to create SDL surface: {}", SDL_GetError());
        return {};
    }
    auto* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_DestroySurface(surface);
    if (!texture) {
        LOG_ERROR(ImGui, "Unable to create SDL texture: {}", SDL_GetError());
        return {};
    }
    return {ImTextureID(texture), std::shared_ptr<SDL_Texture>{texture, SDL_DestroyTexture}};
#endif
}

ImGui::Core::TextureManager::NativeTexture LoadTextureDataFromFile(std::filesystem::path filePath) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<u8> data =
        std::vector<u8>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return LoadTextureData(std::move(data));
}

void scanFolder(std::vector<IconInfo>& icons, std::filesystem::path folderPath, int depth) {
    if (depth > EmulatorSettings.GetBigPictureFolderDepth()) {
        return;
    }

    std::error_code ec;
    std::filesystem::directory_iterator it(folderPath, std::filesystem::directory_options::none,
                                           ec);
    std::filesystem::directory_iterator end;

    if (ec) {
        LOG_ERROR(ImGui, "Cannot open folder: {} due to filesystem error: {}", folderPath.string(),
                  ec.message());
        return;
    }

    while (it != end) {
        // For access errors getting it.path may not be safe, so log parent folder
        if (ec) {
            LOG_ERROR(ImGui, "Cannot access item inside folder {} due to error: {}",
                      folderPath.string(), ec.message());
            ec.clear();
            it.increment(ec);
            continue;
        }

        const auto& entry = *it;
        bool isEntryDir = std::filesystem::is_directory(entry, ec);

        if (ec) {
            LOG_ERROR(ImGui, "Skipping file/subfolder {} due to filesystem error: {}",
                      entry.path().string(), ec.message());
            ec.clear();
            it.increment(ec);
            continue;
        }

        if (isEntryDir && !std::filesystem::exists(entry.path() / "eboot.bin", ec)) {
            scanFolder(icons, entry.path(), depth + 1);
        }

        std::string pathstring = entry.path().filename().string();
        if (pathstring.ends_with("-UPDATE") || pathstring.ends_with("-patch") ||
            (!isEntryDir && !Core::FileSys::IsZArchiveFile(entry))) {
            it.increment(ec);
            continue;
        }

        if (Core::FileSys::IsZArchiveFile(entry)) {
            size_t start = pathstring.length() - 3;
            for (size_t i = start; i < pathstring.length(); ++i) {
                pathstring[i] =
                    static_cast<char>(std::tolower(static_cast<unsigned char>(pathstring[i])));
            }

            if (pathstring.ends_with("-UPDATE.zar") || pathstring.ends_with("-patch.zar")) {
                it.increment(ec);
                continue;
            }
        }

        IconInfo icon;
        PSF psf;
        const std::string sfoFileName = "param.sfo";
        std::filesystem::path sfoPath = UpdateChecker(sfoFileName, entry.path());

        if (std::filesystem::exists(sfoPath, ec) && psf.Open(sfoPath)) {
            if (const auto title = psf.GetString("TITLE"); title.has_value()) {
                icon.title = *title;
            }

            if (const auto title_id = psf.GetString("TITLE_ID"); title_id.has_value()) {
                icon.serial = *title_id;
            }
        } else {
            it.increment(ec);
            continue;
        }

        const std::string iconFileName = "icon0.png";
        std::filesystem::path iconPath = UpdateChecker(iconFileName, entry.path());

        auto texture = LoadTextureDataFromFile(iconPath);
        icon.textureId = texture.id;
        icon.textureLifetime = std::move(texture.lifetime);

        icon.ebootPath = entry.path() / "eboot.bin";
        if (Core::FileSys::IsZArchiveFile(entry.path())) {
            icon.ebootPath = entry.path();
        }

        icon.focusState = false;
        icons.push_back(icon);
        it.increment(ec);
    }
}

void GetGameIconInfo(std::vector<IconInfo>& icons) {
    icons.clear();

    for (const auto& installLoc : EmulatorSettings.GetAllGameInstallDirs()) {
        if (installLoc.enabled && std::filesystem::exists(installLoc.path)) {
            scanFolder(icons, installLoc.path, 0);
        }
    }

    std::sort(icons.begin(), icons.end(), [](const IconInfo& a, const IconInfo& b) {
        return a.title < b.title; // Alphabetical order
    });
}

void Launch(char* executableName, bool sameProcess) {
    done = false;
    showSettings = false;
    runEbootPath.clear();
#ifdef __APPLE__
    constexpr bool native_metal = true;
    Metal::RequireAppleSilicon();
#else
    constexpr bool native_metal = false;
#endif
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        LOG_ERROR(ImGui, "SDL_INIT_VIDEO Error: {}", SDL_GetError());
        SDL_Quit();
        return;
    }

    if (!SDL_Init(SDL_INIT_GAMEPAD)) {
        LOG_ERROR(ImGui, "SDL_INIT_GAMEPAD Error: {}", SDL_GetError());
    }

    SDL_Window* window = SDL_CreateWindow(
        native_metal ? "shadPS4 Big Picture Mode - Metal 4" : "shadPS4 Big Picture Mode", 1280, 720,
        (EmulatorSettings.IsFullScreen() ? SDL_WINDOW_FULLSCREEN : 0) |
            (native_metal ? SDL_WINDOW_METAL : 0));
    if (window == nullptr) {
        LOG_ERROR(ImGui, "SDL Window Creation Error: {}", SDL_GetError());
        SDL_Quit();
        return;
    }

    Frontend::SetDefaultWindowIcon(window);
    if (!native_metal) {
        renderer = SDL_CreateRenderer(window, nullptr);
        if (!renderer) {
            LOG_ERROR(ImGui, "SDL Renderer Creation Error: {}", SDL_GetError());
            SDL_DestroyWindow(window);
            SDL_Quit();
            return;
        }
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigNavCursorVisibleAlways = true;

    ImFontConfig cfgBase;
    cfgBase.OversampleH = 2;
    cfgBase.OversampleV = 1;

    io.FontDefault = ImGui::FontStack::AddPrimaryUiFont(
        io.Fonts, 64.0f, EmulatorSettings.GetConsoleLanguage(), cfgBase, true);
    io.FontGlobalScale = 0.5f;
    // size the big picture font atlas cap from the renderer limit
    const auto max_dim =
        native_metal ? 16384
                     : SDL_GetNumberProperty(SDL_GetRendererProperties(renderer),
                                             SDL_PROP_RENDERER_MAX_TEXTURE_SIZE_NUMBER, 8192);
    const int atlas_max = static_cast<int>(std::bit_floor(std::max<u64>(max_dim, 512)));
    io.Fonts->TexMaxWidth = atlas_max;
    io.Fonts->TexMaxHeight = atlas_max;

#ifdef __APPLE__
    if (native_metal) {
        ImGui_ImplSDL3_InitForMetal(window);
        metal_renderer = std::make_unique<Metal::BigPictureRenderer>(window);
    } else
#endif
    {
        ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
        ImGui_ImplSDLRenderer3_Init(renderer);
    }

    auto settingsWindow = std::make_unique<ImGuiEmuSettings::SettingsWindow>(false);

    float sliderScale = 1.0f;
    auto applySettings = [&] {
        uiScale = EmulatorSettings.GetBigPictureScale() / 1000.f;
        sliderScale = uiScale;
        GetGameIconInfo(gameIcons);
        SDL_SetWindowFullscreen(window,
                                EmulatorSettings.IsFullScreen() ? SDL_WINDOW_FULLSCREEN : 0);
    };
    applySettings();

    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                done = true;
            }
        }

        if (!native_metal) {
            ImGui_ImplSDLRenderer3_NewFrame();
        }
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.06f, 1.00f)); // black
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.20f, 0.40f, 0.70f, 1.00f));   // blue
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                              ImVec4(0.25f, 0.50f, 0.85f, 1.00f)); // lighter blue
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive,
                              ImVec4(0.26f, 0.59f, 0.98f, 0.80f)); // another light blue
        ImGui::PushStyleColor(ImGuiCol_SliderGrab,
                              ImVec4(0.26f, 0.59f, 0.98f, 0.80f)); // another light blue

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f * uiScale);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f * uiScale, 10.0f * uiScale));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f * uiScale, 10.0f * uiScale));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.5f * uiScale);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f * uiScale, 20.0f * uiScale));
        ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 20.0f * uiScale);

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        ImGuiWindowFlags window_flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::Begin("Game Window", &done, window_flags);
        ImGui::DrawPrettyBackground();
        ImGui::SetWindowFontScale(uiScale);

        ImGuiChildFlags child_flags = ImGuiChildFlags_Borders | ImGuiChildFlags_NavFlattened;

        ImGuiWindowFlags child_window_flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;

        if (ImGui::IsWindowAppearing()) {
            ImGui::SetNextWindowFocus();
        }

        ImGui::BeginChild("ContentRegion", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()),
                          child_flags, child_window_flags);

        Overlay::TextCentered("Select Game");
        ImGui::Dummy(ImVec2(0.0f, 10.f * uiScale));

        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }

        SetGameIcons(gameIcons);
        ImGui::EndChild();
        ImGui::Separator();

        ImGui::SetNextItemWidth(300.0f * uiScale);

        if (ImGui::IsWindowAppearing()) {
            sliderScale = uiScale;
        }
        ImGui::SliderFloat("UI Scale", &sliderScale, 0.25f, 3.0f);

        // Only update when user is not interacting with slider
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            uiScale = sliderScale;
        }

        ImGui::SameLine();

        // Align buttons right
        float buttonsWidth = ImGui::CalcTextSize("Settings").x + ImGui::CalcTextSize("Exit").x +
                             ImGui::GetStyle().FramePadding.x * 4.0f +
                             ImGui::GetStyle().ItemSpacing.x;
        ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - buttonsWidth);

        if (ImGui::Button("Settings")) {
            EmulatorSettings.SetBigPictureScale(static_cast<int>(uiScale * 1000));
            EmulatorSettings.Save();
            settingsWindow->Prepare();
            showSettings = true;
        }

        ImGui::SameLine();

        if (ImGui::Button("Exit")) {
            ImGui::OpenPopup("Confirm Exit");
        }

        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("Confirm Exit", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("This will exit shadPS4!\nAre you sure?");
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120 * uiScale, 0))) {
                ImGui::CloseCurrentPopup();
                done = true;
            }
            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120 * uiScale, 0))) {
                ImGui::CloseCurrentPopup();
            }

            if (ImGui::IsWindowAppearing()) {
                ImGui::SetItemDefaultFocus();
            }

            ImGui::EndPopup();
        }

        if (showSettings) {
            settingsWindow->DrawSettings(&showSettings, applySettings);
        }

        ImGui::PopStyleVar(8);
        ImGui::PopStyleColor(5);
        ImGui::End();
        ImGui::Render();
#ifdef __APPLE__
        if (native_metal) {
            metal_renderer->Render(*ImGui::GetDrawData());
        } else
#endif
        {
            SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
            SDL_RenderClear(renderer);
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
            SDL_RenderPresent(renderer);
        }
    }
    settingsWindow.reset();
    gameIcons.clear();
#ifdef __APPLE__
    metal_renderer.reset();
#endif
    if (!native_metal) {
        ImGui_ImplSDLRenderer3_Shutdown();
    }
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    SDL_DestroyWindow(window);
    SDL_Quit();

    EmulatorSettings.SetBigPictureScale(static_cast<int>(uiScale * 1000));
    EmulatorSettings.Save();

    if (runEbootPath != "") {
        auto* emulator = Common::Singleton<Core::Emulator>::Instance();
        emulator->executableName = executableName;
        if (sameProcess) {
            emulator->Run(runEbootPath);
        } else {
            emulator->Relaunch(
                {"--log-append", "--game", Common::FS::PathToUTF8String(runEbootPath)});
        }
    }
}

} // namespace BigPictureMode
