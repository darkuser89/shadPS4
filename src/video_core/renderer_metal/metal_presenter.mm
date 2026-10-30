// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <imgui.h>
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/number_utils.h"
#include "common/path_util.h"
#include "common/stb.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/system/systemservice.h"
#include "core/libraries/videoout/video_out.h"
#include "imgui/imgui_texture.h"
#include "imgui/notifications_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/presenter.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_metal/metal_color.h"
#include "video_core/renderer_metal/metal_format.h"
#include "video_core/renderer_metal/metal_fsr.h"
#include "video_core/renderer_metal/metal_fx.h"
#include "video_core/renderer_metal/metal_gui.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_presenter.h"
#include "video_core/renderer_metal/metal_rasterizer.h"
#include "video_core/renderer_metal/metal_swapchain.h"

namespace Metal {
namespace {

struct Frame final : VideoCore::Frame {
    id<MTLTexture> texture;
    id<MTLTexture> screenshot;
    u32 screenshot_count{};
    u64 ready_tick{};
    u64 present_tick{};
    bool hdr{};
    ScreenshotTransfer screenshot_transfer{ScreenshotTransfer::Encoded};
    bool screenshot_bgra16{};
};

id<MTLTexture> ScreenshotTexture(const Instance& instance, id<MTLTexture> source,
                                 MTLTextureUsage usage) {
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                                         width:source.width
                                                                        height:source.height
                                                                     mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = usage;
    return instance.CreateTexture(descriptor);
}

class Screenshots {
public:
    Screenshots(const Instance& instance_, Scheduler& scheduler_)
        : instance{instance_}, scheduler{scheduler_}, staging{instance, scheduler},
          transfers{instance, scheduler} {}

    void Capture(id<MTLTexture> texture, u32 count, bool overlays, ScreenshotTransfer transfer,
                 bool bgra16 = false) {
        if (!texture || texture.device != instance.GetDevice() || texture.framebufferOnly ||
            texture.textureType != MTLTextureType2D || texture.sampleCount != 1) {
            throw std::runtime_error("Invalid native Metal screenshot texture");
        }
        Poll();
        while (pending.size() + writers.size() >= 4) {
            if (!pending.empty()) {
                scheduler.Wait(pending.front().tick);
            }
            if (!writers.empty()) {
                writers.front().wait();
            }
            Poll();
        }
        const u32 bytes_per_pixel = texture.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : 4;
        const u64 row = (u64{texture.width} * bytes_per_pixel + 255) & ~u64{255};
        const u64 size = row * texture.height;
        if (!count || !texture.width || !texture.height ||
            texture.width > std::numeric_limits<int>::max() / 4 ||
            texture.height > std::numeric_limits<int>::max() / (texture.width * 4) ||
            size > instance.GetDevice().maxBufferLength) {
            throw std::runtime_error("Invalid native Metal screenshot dimensions");
        }
        switch (LinearFormat(texture.pixelFormat)) {
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGR10A2Unorm:
        case MTLPixelFormatRGBA16Float:
            break;
        default:
            throw std::runtime_error("Unsupported native Metal screenshot format");
        }
        auto allocation = staging.AllocateReadback(size, 256);
        scheduler.EndEncoding();
        scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
        transfers.Download({texture, 0, 0, {}, MTLSizeMake(texture.width, texture.height, 1)},
                           {allocation.GetBuffer(), allocation.GetOffset(), size}, {row, 0});
        scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
        static std::atomic<u64> sequence{};
        std::string serial{Common::ElfInfo::Instance().GameSerial()};
        for (char& character : serial) {
            if (!std::isalnum(static_cast<unsigned char>(character)) && character != '_' &&
                character != '-') {
                character = '_';
            }
        }
        const auto now = std::chrono::system_clock::now();
        const auto time = std::chrono::system_clock::to_time_t(now);
        std::tm local{};
        localtime_r(&time, &local);
        char stamp[32]{};
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);
        const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
            1000;
        pending.push_back({std::move(allocation),
                           scheduler.CurrentTick(),
                           {u32(texture.width),
                            u32(texture.height),
                            row,
                            texture.pixelFormat,
                            Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir),
                            fmt::format("{}_{}", serial.empty() ? "UNKNOWN" : serial, stamp),
                            milliseconds,
                            sequence.fetch_add(count, std::memory_order_relaxed),
                            count,
                            overlays,
                            transfer,
                            bgra16,
                            {}}});
    }

    void Poll() {
        while (!writers.empty() &&
               writers.front().wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
            try {
                writers.front().get();
            } catch (const std::exception& error) {
                LOG_ERROR(Render, "Cannot save native Metal screenshot: {}", error.what());
            }
            writers.pop_front();
        }
        while (!pending.empty() && pending.front().allocation.IsReady()) {
            auto& capture = pending.front();
            const auto bytes = capture.allocation.ReadbackData();
            capture.pixels.bytes.assign(bytes.begin(), bytes.end());
            writers.push_back(std::async(std::launch::async,
                                         [pixels = std::move(capture.pixels)] { Save(pixels); }));
            pending.pop_front();
        }
    }

    void Drain() {
        if (!pending.empty()) {
            scheduler.Wait(pending.back().tick);
        }
        Poll();
        for (auto& writer : writers) {
            writer.wait();
        }
        Poll();
    }

private:
    struct Pixels {
        u32 width;
        u32 height;
        u64 row;
        MTLPixelFormat format;
        std::filesystem::path directory;
        std::string prefix;
        s64 milliseconds;
        u64 sequence;
        u32 count;
        bool overlays;
        ScreenshotTransfer transfer;
        bool bgra16;
        std::vector<u8> bytes;
    };
    struct Pending {
        StagingBufferPool::Allocation allocation;
        u64 tick;
        Pixels pixels;
    };

    static void Save(const Pixels& pixels) {
        std::vector<u8> rgba(u64{pixels.width} * pixels.height * 4);
        for (u32 y = 0; y < pixels.height; ++y) {
            const u8* source = pixels.bytes.data() + y * pixels.row;
            for (u32 x = 0; x < pixels.width; ++x) {
                u8* destination = rgba.data() + (u64{y} * pixels.width + x) * 4;
                std::array<float, 3> rgb;
                if (pixels.format == MTLPixelFormatRGBA16Float) {
                    u16 channels[4];
                    std::memcpy(channels, source + x * 8, sizeof(channels));
                    for (u32 component = 0; component < 3; ++component) {
                        rgb[component] = NumberUtils::Uf16ToF32(
                            channels[pixels.bgra16 ? 2 - component : component]);
                    }
                } else if (pixels.format == MTLPixelFormatBGR10A2Unorm) {
                    u32 packed;
                    std::memcpy(&packed, source + x * 4, sizeof(packed));
                    for (u32 component = 0; component < 3; ++component) {
                        rgb[component] = ((packed >> ((2 - component) * 10)) & 1023) / 1023.f;
                    }
                } else {
                    const bool bgra = LinearFormat(pixels.format) == MTLPixelFormatBGRA8Unorm;
                    for (u32 component = 0; component < 3; ++component) {
                        rgb[component] = source[x * 4 + (bgra ? 2 - component : component)] / 255.f;
                    }
                }
                if (pixels.transfer == ScreenshotTransfer::Pq) {
                    for (auto& value : rgb) {
                        value = PqComponentNits(value) / PresentationNitsPerUnit;
                    }
                    rgb = LinearBt2020ToSrgb(rgb);
                }
                for (u32 component = 0; component < 3; ++component) {
                    destination[component] = ScreenshotComponent(
                        rgb[component], pixels.transfer != ScreenshotTransfer::Encoded);
                }
                destination[3] = 255;
            }
        }
        std::vector<u8> png;
        const auto append = [](void* context, void* bytes, int size) {
            auto& output = *static_cast<std::vector<u8>*>(context);
            const auto* source = static_cast<const u8*>(bytes);
            output.insert(output.end(), source, source + size);
        };
        if (!stbi_write_png_to_func(append, &png, pixels.width, pixels.height, 4, rgba.data(), 0)) {
            throw std::runtime_error("Native Metal screenshot PNG encoding failed");
        }
        const auto& directory = pixels.directory;
        std::filesystem::create_directories(directory);
        for (u32 index = 0; index < pixels.count; ++index) {
            const auto path =
                directory / fmt::format("{}_{:03}_{}_{:06}.png", pixels.prefix, pixels.milliseconds,
                                        pixels.overlays ? "hud" : "game", pixels.sequence + index);
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(png.data()), png.size());
            file.close();
            if (!file) {
                LOG_ERROR(Render, "Cannot write native Metal screenshot: {}", path.string());
                continue;
            }
            LOG_INFO(Render, "Saved screenshot: {}", path.string());
            shadNotifications::QueueNotification("Saved screenshot:\n" + path.string(), 3.f,
                                                 shadNotifications::position::BottomRight, png);
        }
    }

    const Instance& instance;
    Scheduler& scheduler;
    StagingBufferPool staging;
    TextureTransfers transfers;
    std::deque<Pending> pending;
    std::deque<std::future<void>> writers;
};

CAMetalLayer* PresentationLayer(Frontend::WindowSDL& window) {
    const auto info = window.GetWindowInfo();
    if (info.type != Frontend::WindowSystemType::Metal || !info.render_surface) {
        throw std::runtime_error("Native Metal presentation requires an SDL Metal surface");
    }
    auto layer = (__bridge CAMetalLayer*)info.render_surface;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    auto colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    layer.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);
    return layer;
}

class Presenter final : public VideoCore::Presenter {
public:
    Presenter(Frontend::WindowSDL& window, AmdGpu::Liverpool& liverpool)
        : VideoCore::Presenter{window}, draw_scheduler{instance}, present_scheduler{instance},
          rasterizer{instance,
                     draw_scheduler,
                     liverpool,
                     EmulatorSettings.IsNeo(),
                     EmulatorSettings.IsHostMarkersEnabled(),
                     EmulatorSettings.IsGuestMarkersEnabled()},
          converter{instance, draw_scheduler}, frame_transfers{instance, draw_scheduler},
          fsr{instance, draw_scheduler}, metal_fx{instance, draw_scheduler, converter},
          gui{instance, present_scheduler}, screenshots{instance, present_scheduler},
          swapchain{instance, present_scheduler, PresentationLayer(window), window.GetSDLWindow()} {
        SetFsrEnabled(EmulatorSettings.IsFsrEnabled());
        SetMetalFxEnabled(EmulatorSettings.IsMetalFxEnabled());
        SetRcasEnabled(EmulatorSettings.IsRcasEnabled());
        SetRcasAttenuation(EmulatorSettings.GetRcasAttenuation() / 1000.0f);
        display_extent.store((u64{static_cast<u32>(std::max(window.GetWidth(), 0))} << 32) |
                             static_cast<u32>(std::max(window.GetHeight(), 0)));
        ImGui::Core::TextureManager::SetNativeUpload(
            [this](std::span<const u8> pixels, u32 width, u32 height) {
                return gui.Upload(pixels, width, height);
            });
        ImGui::Core::InitializeNative(window, 16384);
        ImGui::GetIO().BackendRendererName = "shadPS4_Metal4";
        ImGui::GetIO().BackendFlags |=
            ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
    }

    ~Presenter() override {
        BeginShutdown();
        Shutdown();
        gui.ReleaseTextures();
        ImGui::Core::ShutdownNative();
        ImGui::Core::TextureManager::SetNativeUpload({});
    }

    void BeginShutdown() {
        std::scoped_lock lock{frame_mutex};
        shutting_down = true;
        frame_cv.notify_all();
    }

    void Shutdown() {
        if (shutdown_complete) {
            return;
        }
        rasterizer.PrepareShutdown();
        for (auto& frame : frames) {
            if (frame->screenshot_count) {
                screenshots.Capture(frame->screenshot, frame->screenshot_count, false,
                                    frame->screenshot_transfer, frame->screenshot_bgra16);
                frame->screenshot = nil;
                frame->screenshot_count = 0;
            }
        }
        present_scheduler.Finish();
        screenshots.Drain();
        shutdown_complete = true;
    }

    VideoCore::Rasterizer& GetRasterizer() const override {
        return const_cast<Rasterizer&>(rasterizer);
    }
    bool IsHDRSupported() const override {
        return swapchain.SupportsHDR();
    }
    bool SupportsMetalFx() const override {
        return metal_fx.IsSupported();
    }
    void SetHDR(bool enable) override {
        if (enable && !requested_hdr.load(std::memory_order_acquire) && !IsHDRSupported()) {
            throw std::runtime_error("Native Metal HDR requires an EDR-capable screen");
        }
        requested_hdr.store(enable, std::memory_order_release);
    }

    void RegisterVideoOutSurface(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                                 VAddr address) override {
        const auto info = ImageInfo::FromVideoOut(attribute, address);
        rasterizer.RegisterMemory(info.guest_address, info.layout.guest_size);
    }

    VideoCore::Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                                   VAddr address) override {
        const auto pixel_format = attribute.attrib.pixel_format;
        const bool pq_input = pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq;
        if (pq_input && !requested_hdr.load(std::memory_order_acquire) && !IsHDRSupported()) {
            throw std::runtime_error("Native Metal PQ video-out requires an EDR-capable screen");
        }
        Frame* frame = ObtainFrame();
        if (!frame) {
            return nullptr;
        }
        frame->hdr = pq_input;
        frame->screenshot_transfer =
            pq_input ? ScreenshotTransfer::Pq
            : (pixel_format == Libraries::VideoOut::PixelFormat::A16R16G16B16Float ||
               pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10)
                ? ScreenshotTransfer::Linear
                : ScreenshotTransfer::Encoded;
        frame->screenshot_bgra16 =
            pixel_format == Libraries::VideoOut::PixelFormat::A16R16G16B16Float;
        frame->screenshot = nil;
        frame->screenshot_count = 0;
        try {
            rasterizer.RunOnGpu([&] {
                draw_scheduler.TickFrame();
                if (frame->present_tick) {
                    draw_scheduler.WaitForSubmission(present_scheduler, frame->present_tick);
                }
                const auto info = ImageInfo::FromVideoOut(attribute, address);
                const auto settings = GetDisplaySettings();
                auto& cache = rasterizer.GetTextureCache();
                auto& image = cache.ObtainImage(info, MTLTextureUsageShaderRead);
                u32 width = info.width;
                u32 height = info.height;
                const u64 extent = display_extent.load(std::memory_order_relaxed);
                const u32 available_width = extent >> 32;
                const u32 available_height = extent;
                if ((settings.fsr.enable || settings.metal_fx) && available_width &&
                    available_height) {
                    if (u64{available_width} * info.height <= u64{available_height} * info.width) {
                        width = available_width;
                        height = std::max<u32>(1, u64{width} * info.height / info.width);
                    } else {
                        height = available_height;
                        width = std::max<u32>(1, u64{height} * info.width / info.height);
                    }
                }
                const bool upscale = width > info.width || height > info.height;
                if (!upscale) {
                    width = info.width;
                    height = info.height;
                }
                ImageViewInfo view;
                view.format = info.format;
                view.mapping = info.view_mapping;
                view.mapping.a = AmdGpu::CompSwizzle::One;
                const bool srgb_input = attribute.attrib.pixel_format ==
                                        Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb;
                const bool perceptual_input =
                    srgb_input || LinearFormat(info.format) != info.format;
                if (upscale) {
                    view.format = LinearFormat(view.format);
                }
                auto source = cache.ObtainView(image, view);
                const auto frame_format =
                    pq_input || pixel_format == Libraries::VideoOut::PixelFormat::A16R16G16B16Float
                        ? MTLPixelFormatRGBA16Float
                        : MTLPixelFormatBGRA8Unorm_sRGB;
                if (!frame->texture || frame->texture.pixelFormat != frame_format ||
                    frame->texture.width != width || frame->texture.height != height) {
                    auto descriptor =
                        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:frame_format
                                                                           width:width
                                                                          height:height
                                                                       mipmapped:NO];
                    descriptor.storageMode = MTLStorageModePrivate;
                    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                    frame->texture = instance.CreateTexture(descriptor);
                }
                if (const u32 count = VideoCore::ConsumeGameOnlyScreenshotRequests()) {
                    auto original_view = view;
                    original_view.format = LinearFormat(info.format);
                    original_view.mapping = AmdGpu::IdentityMapping;
                    auto original = cache.ObtainView(image, original_view);
                    frame->screenshot =
                        ScreenshotTexture(instance, original, MTLTextureUsageUnknown);
                    const auto size = MTLSizeMake(info.width, info.height, 1);
                    draw_scheduler.EndEncoding();
                    draw_scheduler.QueueBarrier(MTLStageAll, MTLStageBlit);
                    frame_transfers.Copy({original, 0, 0, {}, size},
                                         {frame->screenshot, 0, 0, {}, size});
                    draw_scheduler.ProducerBarrier(MTLStageBlit, MTLStageAll);
                    frame->screenshot_count = count;
                }
                if (pq_input) {
                    auto normalized = ProcessingTexture(pq_normalized, info.width, info.height);
                    converter.ConvertPqFrame(source, normalized, PresentationNitsPerUnit,
                                             GuiRenderer::PqPrimaries::Bt2020);
                    id<MTLTexture> linear = normalized;
                    if (upscale && settings.metal_fx) {
                        linear = ProcessingTexture(pq_processed, width, height);
                        metal_fx.Render(normalized, linear, 1.f,
                                        MTLFXSpatialScalerColorProcessingModeHDR);
                    } else if (upscale) {
                        linear = HdrFsrFrame(normalized, width, height, settings.fsr);
                    }
                    converter.ConvertLinearBt2020Frame(linear, frame->texture,
                                                       settings.post_processing.gamma);
                } else if (upscale && settings.metal_fx) {
                    const auto mode = attribute.attrib.pixel_format ==
                                              Libraries::VideoOut::PixelFormat::A16R16G16B16Float
                                          ? MTLFXSpatialScalerColorProcessingModeHDR
                                      : perceptual_input
                                          ? MTLFXSpatialScalerColorProcessingModePerceptual
                                          : MTLFXSpatialScalerColorProcessingModeLinear;
                    metal_fx.Render(source, frame->texture, settings.post_processing.gamma, mode);
                } else {
                    if (upscale) {
                        source =
                            pixel_format == Libraries::VideoOut::PixelFormat::A16R16G16B16Float
                                ? HdrFsrFrame(source, width, height, settings.fsr)
                                : fsr.Render(source, width, height, settings.fsr, perceptual_input);
                    }
                    if (pixel_format == Libraries::VideoOut::PixelFormat::A16R16G16B16Float) {
                        converter.ConvertHdrFrame(source, frame->texture,
                                                  settings.post_processing.gamma);
                    } else {
                        converter.ConvertFrame(source, frame->texture,
                                               settings.post_processing.gamma,
                                               !upscale && srgb_input);
                    }
                }
                VideoCore::UpdateFsrState(upscale && !settings.metal_fx);
                VideoCore::UpdateGameResolution(info.width, info.height);
                frame->ready_tick = draw_scheduler.CurrentTick();
                draw_scheduler.Flush();
            });
        } catch (...) {
            ReleaseFrame(frame);
            throw;
        }
        return frame;
    }

    VideoCore::Frame* PrepareBlankFrame(bool present_thread) override {
        Frame* frame;
        if (present_thread) {
            std::scoped_lock lock{frame_mutex};
            if (shutting_down) {
                return nullptr;
            }
            frame = &present_blank_frame;
        } else {
            frame = ObtainFrame();
        }
        if (!frame) {
            return nullptr;
        }
        frame->texture = nil;
        frame->screenshot = nil;
        frame->screenshot_count = 0;
        frame->ready_tick = 0;
        frame->hdr = false;
        return frame;
    }
    VideoCore::Frame* PrepareLastFrame() override {
        return last_frame;
    }

    void Present(VideoCore::Frame* base, bool reuse, bool game) override {
        screenshots.Poll();
        auto* frame = dynamic_cast<Frame*>(base);
        if (!frame) {
            throw std::runtime_error("Invalid native Metal presentation frame");
        }
        present_scheduler.TickFrame();
        const auto remember = [&] {
            if (!reuse) {
                if (last_frame) {
                    ReleaseFrame(last_frame);
                }
                last_frame = frame;
            }
        };
        const bool hdr_output = requested_hdr.load(std::memory_order_acquire) || frame->hdr;
        swapchain.SetHDR(hdr_output);
        swapchain.Resize(std::max(window.GetWidth(), 0), std::max(window.GetHeight(), 0));
        auto target = swapchain.Acquire();
        if (!target) {
            display_extent.store(0, std::memory_order_relaxed);
            if (frame->screenshot_count) {
                present_scheduler.WaitForSubmission(draw_scheduler, frame->ready_tick);
                screenshots.Capture(frame->screenshot, frame->screenshot_count, false,
                                    frame->screenshot_transfer, frame->screenshot_bgra16);
                frame->screenshot = nil;
                frame->screenshot_count = 0;
                present_scheduler.Flush();
            }
            remember();
            return;
        }
        if (frame->ready_tick) {
            present_scheduler.WaitForSubmission(draw_scheduler, frame->ready_tick);
        }
        if (frame->screenshot_count) {
            screenshots.Capture(frame->screenshot, frame->screenshot_count, false,
                                frame->screenshot_transfer, frame->screenshot_bgra16);
            frame->screenshot = nil;
            frame->screenshot_count = 0;
        }
        ImGui::Core::TextureManager::Submit();
        const ImGuiID dock_id = ImGui::Core::NewFrame(reuse);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 1));
        ImGui::SetNextWindowDockID(dock_id, ImGuiCond_Once);
        u64 next_extent{};
        if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
            ImTextureID texture = GuiRenderer::TextureId(frame->texture);
            float width = frame->texture ? frame->texture.width : 0;
            float height = frame->texture ? frame->texture.height : 0;
            if (Libraries::SystemService::IsSplashVisible()) {
                if (!splash) {
                    splash.emplace();
                    const auto& pixels = Common::ElfInfo::Instance().GetSplashData();
                    if (!pixels.empty()) {
                        splash = ImGui::RefCountedTexture::DecodePngTexture(pixels);
                    }
                }
                if (*splash) {
                    const auto image = splash->GetTexture();
                    texture = image.im_id;
                    width = image.width;
                    height = image.height;
                }
            }
            const auto area = ImGui::GetContentRegionAvail();
            const auto pixels = area * ImGui::GetIO().DisplayFramebufferScale;
            if (std::isfinite(pixels.x) && std::isfinite(pixels.y) && pixels.x > 0 &&
                pixels.y > 0) {
                const auto width = u32(std::clamp(pixels.x, 1.f, float(target.width)));
                const auto height = u32(std::clamp(pixels.y, 1.f, float(target.height)));
                next_extent = (u64{width} << 32) | height;
            }
            if (texture && width > 0 && height > 0 && area.x > 0 && area.y > 0) {
                const float scale = std::min(area.x / width, area.y / height);
                const ImVec2 size{width * scale, height * scale};
                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + (area - size) * 0.5f);
                ImGui::Image(texture, size);
            }
            if (game && EmulatorSettings.IsNullGPU()) {
                Core::Devtools::Layer::DrawNullGpuNotice();
            }
        }
        ImGui::End();
        display_extent.store(next_extent, std::memory_order_relaxed);
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(3);
        ImGui::Render();
        const u32 overlays = VideoCore::ConsumeWithOverlaysScreenshotRequests();
        if (hdr_output) {
            auto composite = ProcessingTexture(hdr_composite, target.width, target.height);
            gui.Render(*ImGui::GetDrawData(), composite);
            if (overlays) {
                screenshots.Capture(composite, overlays, true, ScreenshotTransfer::Linear);
            }
            gui.EncodePqFrame(composite, target, PresentationNitsPerUnit,
                              GuiRenderer::PqPrimaries::Srgb);
        } else if (overlays) {
            auto capture = ScreenshotTexture(
                instance, target, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
            gui.Render(*ImGui::GetDrawData(), capture);
            screenshots.Capture(capture, overlays, true, ScreenshotTransfer::Encoded);
            gui.ConvertFrame(capture, target, 1.f, false);
        } else {
            gui.Render(*ImGui::GetDrawData(), target);
        }
        VideoCore::UpdateOutputResolution(target.width, target.height);
        frame->present_tick = present_scheduler.CurrentTick();
        swapchain.Present();
        remember();
        if (!reuse && game) {
            VideoCore::CompleteGameFlip();
        }
    }

private:
    id<MTLTexture> HdrFsrFrame(id<MTLTexture> source, u32 width, u32 height,
                               const VideoCore::FsrSettings& settings) {
        auto compressed = ProcessingTexture(fsr_hdr_input, source.width, source.height,
                                            MTLPixelFormatRGBA32Float);
        converter.ConvertFsrHdrFrame(source, compressed, GuiRenderer::FsrHdrDirection::Compress);
        auto filtered = fsr.Render(compressed, width, height, settings, false);
        auto expanded = ProcessingTexture(fsr_hdr_output, width, height);
        converter.ConvertFsrHdrFrame(filtered, expanded, GuiRenderer::FsrHdrDirection::Expand);
        return expanded;
    }

    id<MTLTexture> ProcessingTexture(id<MTLTexture> __strong& texture, u32 width, u32 height,
                                     MTLPixelFormat format = MTLPixelFormatRGBA16Float) {
        if (!texture || texture.pixelFormat != format || texture.width != width ||
            texture.height != height) {
            auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                 width:width
                                                                                height:height
                                                                             mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
            texture = instance.CreateTexture(descriptor);
        }
        return texture;
    }
    Frame* ObtainFrame() {
        std::unique_lock lock{frame_mutex};
        if (shutting_down) {
            return nullptr;
        }
        if (free_frames.empty() && frames.size() < 18) {
            auto frame = std::make_unique<Frame>();
            Frame* result = frame.get();
            frames.push_back(std::move(frame));
            return result;
        }
        frame_cv.wait(lock, [&] { return shutting_down || !free_frames.empty(); });
        if (shutting_down) {
            return nullptr;
        }
        Frame* frame = free_frames.front();
        free_frames.pop_front();
        return frame;
    }
    void ReleaseFrame(Frame* frame) {
        if (frame == &present_blank_frame) {
            return;
        }
        std::scoped_lock lock{frame_mutex};
        free_frames.push_back(frame);
        frame_cv.notify_one();
    }

    Instance instance;
    Scheduler draw_scheduler;
    Scheduler present_scheduler;
    Rasterizer rasterizer;
    GuiRenderer converter;
    TextureTransfers frame_transfers;
    FsrPass fsr;
    MetalFxPass metal_fx;
    GuiRenderer gui;
    Screenshots screenshots;
    Swapchain swapchain;
    Frame present_blank_frame;
    std::vector<std::unique_ptr<Frame>> frames;
    std::deque<Frame*> free_frames;
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    bool shutting_down{};
    bool shutdown_complete{};
    Frame* last_frame{};
    std::optional<ImGui::RefCountedTexture> splash;
    std::atomic<u64> display_extent{};
    std::atomic<bool> requested_hdr{};
    id<MTLTexture> pq_normalized;
    id<MTLTexture> pq_processed;
    id<MTLTexture> fsr_hdr_input;
    id<MTLTexture> fsr_hdr_output;
    id<MTLTexture> hdr_composite;
};

} // namespace

std::unique_ptr<VideoCore::Presenter> CreatePresenter(Frontend::WindowSDL& window,
                                                      AmdGpu::Liverpool& liverpool) {
    return std::make_unique<Presenter>(window, liverpool);
}

void ShutdownPresenter(VideoCore::Presenter& presenter, AmdGpu::Liverpool& liverpool) {
    if (auto* native = dynamic_cast<Presenter*>(&presenter)) {
        static std::mutex shutdown_mutex;
        std::scoped_lock lock{shutdown_mutex};
        native->BeginShutdown();
        liverpool.StopSubmissions();
        Libraries::VideoOut::Shutdown();
        native->Shutdown();
    }
}

} // namespace Metal
