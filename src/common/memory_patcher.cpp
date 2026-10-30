// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <charconv>
#include <codecvt>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/emulator_state.h"
#include "core/file_format/psf.h"
#include "core/memory.h"
#include "memory_patcher.h"

namespace MemoryPatcher {

EXPORT uintptr_t g_eboot_address;
uint64_t g_eboot_image_size;
std::string g_eboot_name;
std::string g_game_serial;
std::string patch_file;
bool patches_applied = false;
std::vector<patchInfo> pending_patches;

std::string toHex(u64 value, size_t byteSize) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(byteSize * 2) << value;
    return ss.str();
}

static bool isHexSym(const std::string& s) {
    return (s.size() >= 2 && (s[0] == '$' || s[0] == '#'));
}

static bool isHex0x(const std::string& s) {
    return (s.size() >= 3 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'));
}

// possible prefixes == (s == `0x` || s == `#` || s == `$`) will be interpet as hex value, else is
// decimal
static int convertNumBase(const std::string& s) {
    return (isHex0x(s) || isHexSym(s)) ? 16 : 10;
}

std::string convertValueToHex(const std::string type, const std::string valueStr) {
    std::string result;

    if (type == "byte") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 1);
    } else if (type == "bytes16") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 2);
    } else if (type == "bytes32") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 4);
    } else if (type == "bytes64") {
        const u64 value = std::stoull(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 8);
    } else if (type == "float32") {
        union {
            float f;
            uint32_t i;
        } floatUnion;
        floatUnion.f = std::stof(valueStr);
        result = toHex(std::byteswap(floatUnion.i), sizeof(floatUnion.i));
    } else if (type == "float64") {
        union {
            double d;
            uint64_t i;
        } doubleUnion;
        doubleUnion.d = std::stod(valueStr);
        result = toHex(std::byteswap(doubleUnion.i), sizeof(doubleUnion.i));
    } else if (type == "utf8") {
        std::vector<unsigned char> byteArray =
            std::vector<unsigned char>(valueStr.begin(), valueStr.end());
        byteArray.push_back('\0');
        std::stringstream ss;
        for (unsigned char c : byteArray) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
        }
        result = ss.str();
    } else if (type == "utf16") {
        std::wstring wide_str(valueStr.size(), L'\0');
        std::mbstowcs(&wide_str[0], valueStr.c_str(), valueStr.size());
        wide_str.resize(std::wcslen(wide_str.c_str()));

        std::u16string valueStringU16;

        for (wchar_t wc : wide_str) {
            if (wc <= 0xFFFF) {
                valueStringU16.push_back(static_cast<char16_t>(wc));
            } else {
                wc -= 0x10000;
                valueStringU16.push_back(static_cast<char16_t>(0xD800 | (wc >> 10)));
                valueStringU16.push_back(static_cast<char16_t>(0xDC00 | (wc & 0x3FF)));
            }
        }

        std::vector<unsigned char> byteArray;
        // convert to little endian
        for (char16_t ch : valueStringU16) {
            unsigned char low_byte = static_cast<unsigned char>(ch & 0x00FF);
            unsigned char high_byte = static_cast<unsigned char>((ch >> 8) & 0x00FF);

            byteArray.push_back(low_byte);
            byteArray.push_back(high_byte);
        }
        byteArray.push_back('\0');
        byteArray.push_back('\0');
        std::stringstream ss;

        for (unsigned char ch : byteArray) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
        result = ss.str();
    } else if (type == "bytes") {
        result = valueStr;
    } else if (type == "mask" || type == "mask_jump32") {
        result = valueStr;
    } else {
        LOG_INFO(Loader, "Error applying Patch, unknown type: {}", type);
    }
    return result;
}

void ApplyPendingPatches();

void ApplyPatchesFromXML(std::filesystem::path path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");

    if (result) {
        auto patchXML = doc.child("Patch");
        for (pugi::xml_node_iterator it = patchXML.children().begin();
             it != patchXML.children().end(); ++it) {

            if (std::string(it->name()) == "Metadata") {
                if (std::string(it->attribute("isEnabled").value()) == "true") {
                    std::string currentPatchName = it->attribute("Name").value();
                    std::string metadataAppVer = it->attribute("AppVer").value();
                    std::string metadataAppElf = it->attribute("AppElf").value();
                    bool versionMatches = metadataAppVer == app_version;

                    if (metadataAppElf != g_eboot_name)
                        continue;

                    auto patchList = it->first_child();
                    for (pugi::xml_node_iterator patchLineIt = patchList.children().begin();
                         patchLineIt != patchList.children().end(); ++patchLineIt) {

                        std::string type = patchLineIt->attribute("Type").value();
                        if (!versionMatches && type != "mask" && type != "mask_jump32")
                            continue;

                        std::string address = patchLineIt->attribute("Address").value();
                        std::string patchValue = patchLineIt->attribute("Value").value();
                        std::string maskOffsetStr = patchLineIt->attribute("Offset").value();
                        std::string targetStr = "";
                        std::string sizeStr = "";
                        if (type == "mask_jump32") {
                            targetStr = patchLineIt->attribute("Target").value();
                            sizeStr = patchLineIt->attribute("Size").value();
                        } else {
                            try {
                                patchValue = convertValueToHex(type, patchValue);
                            } catch (std::exception& e) {
                                ASSERT_MSG(false,
                                           "Failed to parse patch value \"{}\" for \"{}\" in "
                                           "patch \"{}\", error: \"{}\"\n"
                                           "If the patch was working on earlier versions, then it "
                                           "was using a format that shadPS4 handled incorrectly, "
                                           "and the patch should instead be fixed.",
                                           patchValue, address, currentPatchName, e.what());
                            }
                        }

                        bool littleEndian = false;
                        if (type == "bytes16" || type == "bytes32" || type == "bytes64") {
                            littleEndian = true;
                        }

                        MemoryPatcher::PatchMask patchMask = MemoryPatcher::PatchMask::None;
                        int maskOffsetValue = 0;

                        if (type == "mask")
                            patchMask = MemoryPatcher::PatchMask::Mask;

                        if (type == "mask_jump32")
                            patchMask = MemoryPatcher::PatchMask::Mask_Jump32;

                        if ((type == "mask" || type == "mask_jump32") && !maskOffsetStr.empty()) {
                            maskOffsetValue = std::stoi(maskOffsetStr, 0, 10);
                        }

                        const patchInfo patch = {
                            .gameSerial = "*",
                            .modNameStr = currentPatchName,
                            .offsetStr = address,
                            .valueStr = patchValue,
                            .targetStr = targetStr,
                            .sizeStr = sizeStr,
                            .littleEndian = littleEndian,
                            .patchMask = patchMask,
                            .maskOffset = maskOffsetValue,
                        };
                        MemoryPatcher::PatchMemory(patch);
                    }
                }
            }
        }
    } else {
        LOG_ERROR(Loader, "Could not parse patch XML: {}", result.description());
    }
}

void OnGameLoaded() {
    std::filesystem::path patch_dir = Common::FS::GetUserPath(Common::FS::PathType::PatchesDir);
    if (!patch_file.empty()) {

        auto file_path = (patch_dir / patch_file).native();
        if (std::filesystem::exists(patch_file)) {
            ApplyPatchesFromXML(patch_file);
        } else {
            ApplyPatchesFromXML(file_path);
        }
    } else if (EmulatorState::GetInstance()->IsAutoPatchesLoadEnabled()) {
        for (auto const& repo : std::filesystem::directory_iterator(patch_dir)) {
            if (!repo.is_directory()) {
                continue;
            }

            const auto files_json_path = repo.path() / "files.json";

            if (!std::filesystem::is_regular_file(files_json_path)) {
                LOG_ERROR(Loader, "The folder '{}' does not contain the files.json file.",
                          repo.path().filename().string());
                continue;
            }

            std::ifstream json_file{files_json_path};
            nlohmann::json available_patches;

            if (!json_file.is_open()) {
                LOG_ERROR(Loader, "Could not open patch index: {}", files_json_path.string());
                continue;
            }

            try {
                available_patches = nlohmann::json::parse(json_file);
            } catch (const nlohmann::json::parse_error& e) {
                LOG_ERROR(Loader, "Could not parse patch index {}: {}", files_json_path.string(),
                          e.what());
                continue;
            }

            std::filesystem::path game_patch_file;
            for (auto const& [filename, serials] : available_patches.items()) {
                if (std::find(serials.begin(), serials.end(), g_game_serial) != serials.end()) {
                    game_patch_file = repo.path() / filename;
                    break;
                }
            }
            if (std::filesystem::exists(game_patch_file)) {
                ApplyPatchesFromXML(game_patch_file);
            }
        }
    }
    ApplyPendingPatches();
}

void AddPatchToQueue(const patchInfo& patchToAdd) {
    if (patches_applied) {
        PatchMemory(patchToAdd);
        return;
    }
    pending_patches.push_back(patchToAdd);
}

void ApplyPendingPatches() {
    patches_applied = true;
    for (size_t i = 0; i < pending_patches.size(); ++i) {
        const patchInfo& currentPatch = pending_patches[i];

        if (currentPatch.gameSerial != "*" && currentPatch.gameSerial != g_game_serial)
            continue;

        PatchMemory(currentPatch);
    }

    pending_patches.clear();
}

static bool UsesNativeMemory() {
#ifdef __APPLE__
    return EmulatorSettings.GetRenderer() == "Metal4";
#else
    return false;
#endif
}

static void WritePatchMemory(uintptr_t address, std::span<const u8> bytes) {
    if (UsesNativeMemory()) {
        Core::Memory::Instance()->WriteModuleMemory(address, bytes);
    } else {
        std::memcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size());
    }
}

static int HexDigit(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

static bool ParsePatchBytes(const std::string& value, std::vector<u8>& bytes) {
    if (value.empty() || value.size() % 2 != 0) {
        return false;
    }
    bytes.reserve(value.size() / 2);
    for (size_t i = 0; i < value.size(); i += 2) {
        const int high = HexDigit(value[i]);
        const int low = HexDigit(value[i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        bytes.push_back((high << 4) | low);
    }
    return true;
}

static bool IsPatchRange(uintptr_t address, u64 size) {
    return g_eboot_address != 0 &&
           g_eboot_image_size <= std::numeric_limits<uintptr_t>::max() - g_eboot_address &&
           address >= g_eboot_address && address - g_eboot_address <= g_eboot_image_size &&
           size <= g_eboot_image_size - (address - g_eboot_address);
}

static uintptr_t OffsetPattern(uintptr_t address, int offset) {
    if (!address) {
        return 0;
    }
    if (offset < 0) {
        const u64 magnitude = -static_cast<s64>(offset);
        return magnitude <= address ? address - magnitude : 0;
    }
    return static_cast<u64>(offset) <= std::numeric_limits<uintptr_t>::max() - address
               ? address + offset
               : 0;
}

static bool SetRelativeJump(std::span<u8> bytes, uintptr_t next, uintptr_t target) {
    const bool forward = target >= next;
    const u64 distance = forward ? target - next : next - target;
    if (distance > (forward ? u64{0x7fffffff} : u64{0x80000000})) {
        return false;
    }
    const s32 offset =
        static_cast<s32>(forward ? static_cast<s64>(distance) : -static_cast<s64>(distance));
    bytes[0] = 0xE9;
    std::memcpy(bytes.data() + 1, &offset, sizeof(offset));
    return true;
}

void PatchMemory(const patchInfo& patch) {
    std::vector<u8> payload;
    if (!ParsePatchBytes(patch.valueStr, payload)) {
        LOG_ERROR(Loader, "Invalid patch bytes for {}: {}", patch.modNameStr, patch.valueStr);
        return;
    }
    uintptr_t address = 0;
    if (patch.patchMask == PatchMask::None) {
        u64 offset = 0;
        std::string_view value = patch.offsetStr;
        if (value.starts_with("0x") || value.starts_with("0X")) {
            value.remove_prefix(2);
        }
        const auto [end, error] =
            std::from_chars(value.data(), value.data() + value.size(), offset, 16);
        if (error != std::errc{} || end != value.data() + value.size() ||
            (!patch.isOffset && offset < 0x400000)) {
            LOG_ERROR(Loader, "Invalid patch address: {}", patch.offsetStr);
            return;
        }
        if (!patch.isOffset) {
            offset -= 0x400000;
        }
        if (offset <= std::numeric_limits<uintptr_t>::max() - g_eboot_address) {
            address = g_eboot_address + offset;
        }
    } else if (patch.patchMask == PatchMask::Mask || patch.patchMask == PatchMask::Mask_Jump32) {
        address = OffsetPattern(PatternScan(patch.offsetStr), patch.maskOffset);
    }

    if (patch.patchMask == PatchMask::Mask_Jump32) {
        u32 jump_size = 0;
        const auto [end, error] = std::from_chars(
            patch.sizeStr.data(), patch.sizeStr.data() + patch.sizeStr.size(), jump_size);
        if (error != std::errc{} || end != patch.sizeStr.data() + patch.sizeStr.size() ||
            jump_size < 5 || jump_size > 256) {
            LOG_ERROR(Loader, "Jump size must be between 5 and 256 bytes");
            return;
        }
        const uintptr_t target = PatternScan(patch.targetStr);
        const u64 cave_size = payload.size() + 5;
        if (!IsPatchRange(address, jump_size) || !IsPatchRange(target, cave_size) ||
            (address < target + cave_size && target < address + jump_size)) {
            LOG_ERROR(Loader, "Invalid or overlapping jump patch ranges for {}", patch.modNameStr);
            return;
        }
        const uintptr_t cave_end = target + payload.size();
        payload.resize(cave_size);
        std::vector<u8> entry(jump_size, 0x90);
        if (!SetRelativeJump(entry, address + 5, target) ||
            !SetRelativeJump(std::span{payload}.last(5), cave_end + 5, address + jump_size)) {
            LOG_ERROR(Loader, "Jump patch is outside the relative jump range for {}",
                      patch.modNameStr);
            return;
        }
        if (UsesNativeMemory()) {
            std::vector<u8> original(std::max<u64>(entry.size(), payload.size()));
            auto* memory = Core::Memory::Instance();
            if (!memory->ReadModuleMemory(address, std::span{original}.first(entry.size())) ||
                !memory->ReadModuleMemory(target, std::span{original}.first(payload.size()))) {
                LOG_ERROR(Loader, "Jump patch contains unmapped memory for {}", patch.modNameStr);
                return;
            }
        }
        WritePatchMemory(target, payload);
        WritePatchMemory(address, entry);
        LOG_INFO(Loader,
                 "Applied Patch mask_jump32: {}, PatchAddress: {:#x}, JumpTarget: {:#x}, "
                 "CodeCaveEnd: {:#x}, JumpSize: {}",
                 patch.modNameStr, address, target, cave_end, jump_size);
        return;
    }

    if (!IsPatchRange(address, payload.size())) {
        LOG_ERROR(Loader, "Invalid address for patch {}", patch.modNameStr);
        return;
    }
    if (patch.littleEndian) {
        std::reverse(payload.begin(), payload.end());
    }
    WritePatchMemory(address, payload);
    LOG_INFO(Loader, "Applied patch: {}, Offset: {:#x}, Value: {}", patch.modNameStr, address,
             patch.valueStr);
}

static std::vector<int32_t> PatternToByte(const std::string& pattern) {
    std::vector<int32_t> bytes;
    for (size_t i = 0; i < pattern.size();) {
        if (pattern[i] == ' ' || pattern[i] == '\t' || pattern[i] == '\r' || pattern[i] == '\n') {
            ++i;
        } else if (pattern[i] == '?') {
            ++i;
            if (i < pattern.size() && pattern[i] == '?') {
                ++i;
            }
            bytes.push_back(-1);
        } else {
            if (i + 1 >= pattern.size()) {
                return {};
            }
            const int high = HexDigit(pattern[i]);
            const int low = HexDigit(pattern[i + 1]);
            if (high < 0 || low < 0) {
                return {};
            }
            bytes.push_back((high << 4) | low);
            i += 2;
        }
    }
    return bytes;
}

uintptr_t PatternScan(const std::string& signature) {
    const auto pattern = PatternToByte(signature);
    if (pattern.empty() || pattern.size() > g_eboot_image_size ||
        !IsPatchRange(g_eboot_address, g_eboot_image_size)) {
        return 0;
    }
    const auto matches = [&](std::span<const u8> bytes) -> size_t {
        for (size_t i = 0; i + pattern.size() <= bytes.size(); ++i) {
            bool found = true;
            for (size_t j = 0; j < pattern.size(); ++j) {
                if (pattern[j] != -1 && bytes[i + j] != pattern[j]) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return i;
            }
        }
        return bytes.size();
    };
    if (!UsesNativeMemory()) {
        const std::span bytes{reinterpret_cast<const u8*>(g_eboot_address), g_eboot_image_size};
        const size_t offset = matches(bytes);
        return offset < bytes.size() ? g_eboot_address + offset : 0;
    }

    std::vector<u8> bytes;
    uintptr_t address = g_eboot_address;
    const uintptr_t end = address + g_eboot_image_size;
    while (address < end) {
        constexpr u64 PageSize = 16_KB;
        const u64 length = std::min<u64>(PageSize - address % PageSize, end - address);
        const size_t tail = bytes.size();
        bytes.resize(tail + length);
        if (!Core::Memory::Instance()->ReadModuleMemory(address, std::span{bytes}.subspan(tail))) {
            bytes.clear();
        } else {
            const size_t offset = matches(bytes);
            if (offset < bytes.size()) {
                return address - tail + offset;
            }
            const size_t retain = std::min(bytes.size(), pattern.size() - 1);
            bytes.erase(bytes.begin(), bytes.end() - retain);
        }
        address += length;
    }
    return 0;
}

} // namespace MemoryPatcher
