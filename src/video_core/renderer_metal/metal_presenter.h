// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

namespace AmdGpu {
class Liverpool;
}
namespace Frontend {
class WindowSDL;
}
namespace VideoCore {
class Presenter;
}

namespace Metal {

[[nodiscard]] std::unique_ptr<VideoCore::Presenter> CreatePresenter(Frontend::WindowSDL& window,
                                                                    AmdGpu::Liverpool& liverpool);
void ShutdownPresenter(VideoCore::Presenter& presenter, AmdGpu::Liverpool& liverpool);

} // namespace Metal
