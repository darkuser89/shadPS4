// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <deque>
#include <utility>

#include <imgui.h>
#include "common/assert.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/stb.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "imgui_impl_vulkan.h"
#include "texture_manager.h"

namespace ImGui {

namespace Core::TextureManager {
struct Inner {
    std::atomic_int count = 0;
    ImTextureID texture_id = nullptr;
    u32 width = 0;
    u32 height = 0;

    Vulkan::UploadTextureData upload_data{};
    NativeTexture native_texture;
    bool native_backend{};

    ~Inner();
};
} // namespace Core::TextureManager

using namespace Core::TextureManager;

RefCountedTexture::RefCountedTexture(Inner* inner) : inner(inner) {
    ++inner->count;
}

RefCountedTexture RefCountedTexture::DecodePngTexture(std::vector<u8> data) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngTexture(std::move(data), core);
    return RefCountedTexture(core);
}

RefCountedTexture RefCountedTexture::DecodePngFile(std::filesystem::path path) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngFile(std::move(path), core);
    return RefCountedTexture(core);
}

RefCountedTexture::RefCountedTexture() : inner(nullptr) {}

RefCountedTexture::RefCountedTexture(const RefCountedTexture& other) : inner(other.inner) {
    if (inner != nullptr) {
        ++inner->count;
    }
}

RefCountedTexture::RefCountedTexture(RefCountedTexture&& other) noexcept : inner(other.inner) {
    other.inner = nullptr;
}

RefCountedTexture& RefCountedTexture::operator=(const RefCountedTexture& other) {
    if (this == &other)
        return *this;
    inner = other.inner;
    if (inner != nullptr) {
        ++inner->count;
    }
    return *this;
}

RefCountedTexture& RefCountedTexture::operator=(RefCountedTexture&& other) noexcept {
    if (this == &other)
        return *this;
    std::swap(inner, other.inner);
    return *this;
}

RefCountedTexture::~RefCountedTexture() {
    if (inner != nullptr) {
        if (inner->count.fetch_sub(1) == 1) {
            delete inner;
        }
    }
}

RefCountedTexture::Image RefCountedTexture::GetTexture() const {
    if (inner == nullptr) {
        return {};
    }
    return Image{
        .im_id = inner->texture_id,
        .width = inner->width,
        .height = inner->height,
    };
}

RefCountedTexture::operator bool() const {
    return inner != nullptr && inner->texture_id != nullptr;
}

struct Job {
    Inner* core;
    std::vector<u8> data;
    std::filesystem::path path;
};

struct UploadJob {
    Inner* core = nullptr;
    Vulkan::UploadTextureData data{};
    std::vector<u8> pixels;
    int tick = 0; // Used to skip the first frame when destroying to await the current frame to draw
};

static std::atomic_bool g_is_worker_running{false};
static bool g_native_backend{};
static NativeUpload g_native_upload;
static std::jthread g_worker_thread;
static std::condition_variable g_worker_cv;

static std::mutex g_job_list_mtx;
static std::deque<Job> g_job_list;

static std::mutex g_upload_mtx;
static std::deque<UploadJob> g_upload_list;

namespace Core::TextureManager {

Inner::~Inner() {
    if (!native_backend && upload_data.im_texture != nullptr) {
        std::unique_lock lk{g_upload_mtx};
        g_upload_list.emplace_back(UploadJob{
            .data = this->upload_data,
            .tick = 2,
        });
    }
}

void SetNativeUpload(NativeUpload upload) {
    if (g_is_worker_running) {
        UNREACHABLE_MSG("Cannot change texture upload backend while its worker is running");
    }
    g_native_upload = std::move(upload);
}

void WorkerLoop() {
    Common::SetCurrentThreadName("shadPS4:ImGuiTextureManager");
    while (g_is_worker_running) {
        std::unique_lock lk{g_job_list_mtx};
        g_worker_cv.wait(lk, [] { return !g_is_worker_running || !g_job_list.empty(); });
        if (!g_is_worker_running) {
            break;
        }
        lk.unlock();
        while (true) {
            g_job_list_mtx.lock();
            if (g_job_list.empty()) {
                g_job_list_mtx.unlock();
                break;
            }
            auto [core, png_raw, path] = std::move(g_job_list.front());
            g_job_list.pop_front();
            g_job_list_mtx.unlock();

            if (core->native_backend != g_native_backend) {
                UNREACHABLE_MSG("Texture decode job does not match the active renderer backend");
            }
            if (!g_native_backend && EmulatorSettings.IsVkCrashDiagnosticEnabled()) {
                if (core->count.fetch_sub(1) == 1) {
                    delete core;
                }
                continue;
            }

            if (!path.empty()) { // Decode PNG from file
                Common::FS::IOFile file(path, Common::FS::FileAccessMode::Read);
                if (!file.IsOpen()) {
                    LOG_ERROR(ImGui, "Failed to open PNG file: {}", path.string());
                    if (core->count.fetch_sub(1) == 1) {
                        delete core;
                    }
                    continue;
                }
                png_raw.resize(file.GetSize());
                file.Seek(0);
                file.ReadRaw<u8>(png_raw.data(), png_raw.size());
                file.Close();
            }

            int width, height;
            const stbi_uc* pixels =
                stbi_load_from_memory(png_raw.data(), png_raw.size(), &width, &height, nullptr, 4);

            if (!pixels || width <= 0 || height <= 0) {
                stbi_image_free((void*)pixels);
                LOG_ERROR(ImGui, "Failed to decode PNG texture");
                if (core->count.fetch_sub(1) == 1) {
                    delete core;
                }
                continue;
            }
            UploadJob upload{.core = core};
            if (g_native_backend) {
                upload.pixels.assign(pixels, pixels + size_t{static_cast<u32>(width)} * height * 4);
            } else {
                core->upload_data =
                    Vulkan::UploadTexture(pixels, vk::Format::eR8G8B8A8Unorm, width, height,
                                          size_t{static_cast<u32>(width)} * height * 4);
            }
            stbi_image_free((void*)pixels);
            core->width = width;
            core->height = height;

            std::unique_lock upload_lk{g_upload_mtx};
            g_upload_list.push_back(std::move(upload));
        }
    }
}

void StartWorker(bool native_backend) {
    if (g_is_worker_running) {
        UNREACHABLE_MSG("Texture upload worker is already running");
    }
    if (static_cast<bool>(g_native_upload) != native_backend) {
        UNREACHABLE_MSG("Texture upload callback does not match the selected renderer backend");
    }
    g_native_backend = native_backend;
    g_is_worker_running = true;
    g_worker_thread = std::jthread(WorkerLoop);
}

void StopWorker() {
    ASSERT(g_is_worker_running);
    {
        std::scoped_lock lock{g_job_list_mtx};
        g_is_worker_running = false;
    }
    g_worker_cv.notify_one();
    g_worker_thread.join();
    std::deque<Job> pending_jobs;
    std::deque<UploadJob> pending_uploads;
    {
        std::scoped_lock lock{g_job_list_mtx, g_upload_mtx};
        pending_jobs.swap(g_job_list);
        pending_uploads.swap(g_upload_list);
    }
    for (auto& job : pending_jobs) {
        if (job.core->count.fetch_sub(1) == 1) {
            delete job.core;
        }
    }
    for (auto& upload : pending_uploads) {
        if (upload.core) {
            if (upload.core->count.fetch_sub(1) == 1) {
                delete upload.core;
            }
        } else if (!g_native_backend) {
            upload.data.Destroy();
        }
    }
    if (!g_native_backend) {
        std::deque<UploadJob> deferred_deletions;
        {
            std::scoped_lock lock{g_upload_mtx};
            deferred_deletions.swap(g_upload_list);
        }
        for (auto& upload : deferred_deletions) {
            upload.data.Destroy();
        }
    }
}

void DecodePngTexture(std::vector<u8> data, Inner* core) {
    core->native_backend = g_native_backend;
    ++core->count;
    Job job{
        .core = core,
        .data = std::move(data),
    };
    std::unique_lock lk{g_job_list_mtx};
    g_job_list.push_back(std::move(job));
    g_worker_cv.notify_one();
}

void DecodePngFile(std::filesystem::path path, Inner* core) {
    core->native_backend = g_native_backend;
    ++core->count;
    Job job{
        .core = core,
        .path = std::move(path),
    };
    std::unique_lock lk{g_job_list_mtx};
    g_job_list.push_back(std::move(job));
    g_worker_cv.notify_one();
}

void Submit() {
    UploadJob upload;
    {
        std::unique_lock lk{g_upload_mtx};
        if (g_upload_list.empty()) {
            return;
        }
        // Upload one texture at a time to avoid slow down
        upload = std::move(g_upload_list.front());
        g_upload_list.pop_front();
        if (upload.tick > 0) {
            --upload.tick;
            g_upload_list.emplace_back(upload);
            return;
        }
    }
    if (upload.core != nullptr) {
        if (upload.core->native_backend != g_native_backend) {
            UNREACHABLE_MSG("Texture upload job does not match the active renderer backend");
        }
        if (g_native_backend) {
            upload.core->native_texture =
                g_native_upload(upload.pixels, upload.core->width, upload.core->height);
            upload.core->texture_id = upload.core->native_texture.id;
        } else {
            upload.core->upload_data.Upload();
            upload.core->texture_id = upload.core->upload_data.im_texture;
        }
        if (upload.core->count.fetch_sub(1) == 1) {
            delete upload.core;
        }
    } else {
        if (g_native_backend) {
            UNREACHABLE_MSG("Native texture upload queue contains a foreign deletion job");
        }
        upload.data.Destroy();
    }
}
} // namespace Core::TextureManager

} // namespace ImGui
