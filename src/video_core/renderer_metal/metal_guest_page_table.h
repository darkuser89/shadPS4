// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>
#include "video_core/renderer_metal/metal_commands.h"
#include "video_core/renderer_metal/metal_guest_page_layout.h"

namespace Metal {

struct GuestPageTableRange {
    VAddr address;
    BufferRange storage;
};

class GuestPageTableSnapshot {
public:
    GuestPageTableSnapshot(const Instance& instance, StagingBufferPool& staging,
                           std::span<const GuestPageTableRange> ranges);

    [[nodiscard]] BufferRange GetRoot() const;
    void Retain(ResourceBindings& bindings) const;

private:
    std::vector<std::shared_ptr<StagingBufferPool::Allocation>> tables;
    std::vector<id<MTLResource>> resources;
};

} // namespace Metal
