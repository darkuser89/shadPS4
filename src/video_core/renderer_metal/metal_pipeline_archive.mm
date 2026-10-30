// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <vector>
#include <CommonCrypto/CommonDigest.h>
#include <fmt/format.h>
#include "common/hash.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_metal/metal_pipeline_archive.h"

namespace Metal {
namespace {
constexpr size_t MaxArchives = 8;
constexpr u64 MaxArchiveBytes = 256 * 1024 * 1024;

struct ArchiveFile {
    std::filesystem::path path;
    u64 size;
};

std::vector<ArchiveFile> ArchiveFiles(const std::filesystem::path& directory) {
    std::vector<ArchiveFile> files;
    std::error_code error;
    std::filesystem::directory_iterator iterator{directory, error};
    const std::filesystem::directory_iterator end;
    while (!error && iterator != end) {
        const auto& entry = *iterator;
        std::error_code entry_error;
        if (entry.path().extension() == ".metalarc" && entry.is_regular_file(entry_error)) {
            const auto size = entry.file_size(entry_error);
            if (!entry_error)
                files.push_back({entry.path(), size});
        }
        if (entry_error)
            LOG_WARNING(Render, "Cannot inspect native Metal pipeline archive {}: {}",
                        Common::FS::PathToUTF8String(entry.path()), entry_error.message());
        iterator.increment(error);
    }
    if (error)
        LOG_WARNING(Render, "Cannot enumerate native Metal pipeline archives: {}", error.message());
    std::ranges::sort(files, [](const auto& left, const auto& right) {
        return left.path.filename() > right.path.filename();
    });
    return files;
}

NSURL* FileUrl(const std::filesystem::path& path) {
    return [NSURL fileURLWithFileSystemRepresentation:path.c_str()
                                          isDirectory:NO
                                        relativeToURL:nil];
}

std::optional<std::string> ArchiveDigest(const ArchiveFile& file) {
    if (!file.size || file.size > MaxArchiveBytes)
        return std::nullopt;
    NSError* error = nil;
    NSData* data = [NSData dataWithContentsOfURL:FileUrl(file.path)
                                         options:NSDataReadingMappedIfSafe
                                           error:&error];
    if (!data || data.length != file.size)
        return std::nullopt;
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> bytes;
    CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), bytes.data());
    std::string digest;
    digest.reserve(bytes.size() * 2);
    for (const auto byte : bytes)
        digest += fmt::format("{:02x}", byte);
    return digest;
}

bool HasArchiveDigest(const ArchiveFile& file) {
    const auto name = file.path.stem().string();
    const auto separator = name.rfind('-');
    if (separator == std::string::npos ||
        name.size() - separator - 1 != CC_SHA256_DIGEST_LENGTH * 2)
        return false;
    const auto digest = ArchiveDigest(file);
    return digest && name.substr(separator + 1) == *digest;
}
} // namespace

struct PipelineArchive::State {
    std::mutex mutex;
    std::condition_variable condition;
    u64 pending{};
    u64 pipelines{};
};

PipelineArchive::PipelineArchive(id<MTLDevice> device) : state{std::make_shared<State>()} {
    if (!EmulatorSettings.IsPipelineCacheEnabled())
        return;
    @autoreleasepool {
        u64 identity{1};
        const std::string device_name{device.name.UTF8String};
        const std::string system{NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String};
        for (unsigned char byte : device_name)
            identity = HashCombine(identity, u64{byte});
        identity = HashCombine(identity, u64{0});
        for (unsigned char byte : system)
            identity = HashCombine(identity, u64{byte});
        directory = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "metal4" / "v1" /
                    fmt::format("{:016x}", identity);
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) {
            LOG_WARNING(Render, "Cannot create native Metal pipeline cache: {}", error.message());
            directory.clear();
            return;
        }
        auto loaded = [NSMutableArray<id<MTL4Archive>> new];
        u64 bytes{};
        for (const auto& file : ArchiveFiles(directory)) {
            if (loaded.count == MaxArchives)
                break;
            if (!file.size || file.size > MaxArchiveBytes - bytes)
                continue;
            if (!HasArchiveDigest(file)) {
                LOG_WARNING(Render, "Ignoring unverified native Metal pipeline archive {}",
                            Common::FS::PathToUTF8String(file.path));
                continue;
            }
            NSError* archive_error = nil;
            auto archive = [device newArchiveWithURL:FileUrl(file.path) error:&archive_error];
            if (!archive) {
                LOG_WARNING(Render, "Cannot load native Metal pipeline archive {}: {}",
                            Common::FS::PathToUTF8String(file.path),
                            ErrorDescription(archive_error));
                continue;
            }
            [loaded addObject:archive];
            bytes += file.size;
        }
        archives = [loaded copy];
        auto descriptor = [MTL4PipelineDataSetSerializerDescriptor new];
        descriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
        serializer = [device newPipelineDataSetSerializerWithDescriptor:descriptor];
        if (!serializer)
            LOG_WARNING(Render, "Cannot create native Metal pipeline archive serializer");
    }
}

PipelineArchive::~PipelineArchive() {
    Save();
}

id<MTL4PipelineDataSetSerializer> PipelineArchive::GetSerializer() const noexcept {
    return serializer;
}

MTL4CompilerTaskOptions* PipelineArchive::TaskOptions() const {
    if (!archives.count)
        return nil;
    auto options = [MTL4CompilerTaskOptions new];
    options.lookupArchives = archives;
    return options;
}

std::shared_ptr<void> PipelineArchive::TrackCompilation(bool pipeline) {
    struct Lease {
        explicit Lease(std::shared_ptr<State> state_, bool pipeline) : state{std::move(state_)} {
            std::scoped_lock lock{state->mutex};
            ++state->pending;
            state->pipelines += pipeline;
        }
        ~Lease() {
            std::scoped_lock lock{state->mutex};
            --state->pending;
            state->condition.notify_all();
        }
        std::shared_ptr<State> state;
    };
    return std::make_shared<Lease>(state, pipeline);
}

void PipelineArchive::Save() {
    {
        std::unique_lock lock{state->mutex};
        state->condition.wait(lock, [&] { return state->pending == 0; });
        if (!serializer || !state->pipelines || directory.empty())
            return;
    }
    @autoreleasepool {
        const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
        const std::string unique{NSUUID.UUID.UUIDString.UTF8String};
        const auto name = fmt::format("{:020}-{}", timestamp, unique);
        const auto temporary = directory / (name + ".tmp");
        NSError* archive_error = nil;
        std::error_code error;
        if (![serializer serializeAsArchiveAndFlushToURL:FileUrl(temporary) error:&archive_error]) {
            LOG_WARNING(Render, "Cannot save native Metal pipeline archive: {}",
                        ErrorDescription(archive_error));
            std::filesystem::remove(temporary, error);
            return;
        }
        const auto size = std::filesystem::file_size(temporary, error);
        if (error || !size || size > MaxArchiveBytes) {
            LOG_WARNING(Render,
                        "Native Metal pipeline archive exceeds cache budget or is unreadable");
            std::filesystem::remove(temporary, error);
            return;
        }
        const auto digest = ArchiveDigest({temporary, size});
        if (!digest) {
            LOG_WARNING(Render, "Cannot verify native Metal pipeline archive");
            std::filesystem::remove(temporary, error);
            return;
        }
        const auto destination = directory / fmt::format("{}-{}.metalarc", name, *digest);
        std::filesystem::rename(temporary, destination, error);
        if (error) {
            LOG_WARNING(Render, "Cannot publish native Metal pipeline archive: {}",
                        error.message());
            std::filesystem::remove(temporary, error);
            return;
        }
        size_t retained{};
        u64 bytes{};
        for (const auto& file : ArchiveFiles(directory)) {
            if (retained < MaxArchives && file.size && file.size <= MaxArchiveBytes - bytes &&
                HasArchiveDigest(file)) {
                ++retained;
                bytes += file.size;
            } else {
                std::filesystem::remove(file.path, error);
                if (error)
                    LOG_WARNING(Render, "Cannot trim native Metal pipeline archive: {}",
                                error.message());
            }
        }
    }
}

} // namespace Metal
