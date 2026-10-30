// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <memory>
#include "video_core/renderer_metal/metal_common.h"

namespace Metal {

class PipelineArchive {
public:
    explicit PipelineArchive(id<MTLDevice> device);
    ~PipelineArchive();

    PipelineArchive(const PipelineArchive&) = delete;
    PipelineArchive& operator=(const PipelineArchive&) = delete;

    [[nodiscard]] id<MTL4PipelineDataSetSerializer> GetSerializer() const noexcept;
    [[nodiscard]] MTL4CompilerTaskOptions* TaskOptions() const;
    [[nodiscard]] std::shared_ptr<void> TrackCompilation(bool pipeline);

private:
    struct State;
    void Save();

    std::shared_ptr<State> state;
    std::filesystem::path directory;
    id<MTL4PipelineDataSetSerializer> serializer;
    NSArray<id<MTL4Archive>>* archives;
};

} // namespace Metal
