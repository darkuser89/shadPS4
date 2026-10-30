// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "video_core/renderer_metal/metal_guest_page_table.h"
#include "video_core/renderer_metal/metal_instance.h"

namespace Metal {

namespace {
u64 TableAddress(const StagingBufferPool::Allocation& allocation) {
    const auto buffer = allocation.GetBuffer();
    const u64 offset = allocation.GetOffset();
    const u64 size = allocation.SizeBytes();
    if (!buffer || !buffer.gpuAddress || offset > buffer.length || size > buffer.length - offset ||
        !size || offset > std::numeric_limits<u64>::max() - buffer.gpuAddress) {
        throw std::runtime_error("Invalid native page-table storage");
    }
    const u64 address = buffer.gpuAddress + offset;
    if (address % sizeof(u64) || size - 1 > std::numeric_limits<u64>::max() - address) {
        throw std::runtime_error("Invalid native page-table GPU address");
    }
    return address;
}
} // namespace

GuestPageTableSnapshot::GuestPageTableSnapshot(const Instance& instance, StagingBufferPool& staging,
                                               std::span<const GuestPageTableRange> ranges) {
    GuestPageTableLayout layout;
    for (const auto& range : ranges) {
        const auto& storage = range.storage;
        if (!storage.buffer || storage.buffer.device != instance.GetDevice() ||
            storage.offset > storage.buffer.length ||
            storage.size > storage.buffer.length - storage.offset ||
            storage.offset > std::numeric_limits<u64>::max() - storage.buffer.gpuAddress) {
            throw std::runtime_error("Invalid native page-table backing buffer");
        }
        layout.Map(range.address, storage.size, storage.buffer.gpuAddress + storage.offset);
        id<MTLResource> resource = storage.buffer;
        if (std::ranges::find(resources, resource) == resources.end()) {
            resources.push_back(resource);
        }
    }
    auto root = std::make_shared<StagingBufferPool::Allocation>(
        staging.Allocate(GuestPageTableLayout::RootEntries * sizeof(u64)));
    (void)TableAddress(*root);
    auto root_data = root->UploadData();
    std::memset(root_data.data(), 0, root_data.size());
    resources.push_back(root->GetBuffer());
    tables.push_back(std::move(root));
    for (const auto& [index, entries] : layout.Leaves()) {
        auto leaf =
            std::make_shared<StagingBufferPool::Allocation>(staging.Allocate(sizeof(entries)));
        std::memcpy(leaf->UploadData().data(), entries.data(), sizeof(entries));
        const u64 address = TableAddress(*leaf);
        std::memcpy(root_data.data() + index * sizeof(address), &address, sizeof(address));
        id<MTLResource> resource = leaf->GetBuffer();
        if (std::ranges::find(resources, resource) == resources.end()) {
            resources.push_back(resource);
        }
        tables.push_back(std::move(leaf));
    }
}

BufferRange GuestPageTableSnapshot::GetRoot() const {
    const auto& root = tables.front();
    return {root->GetBuffer(), root->GetOffset(), root->SizeBytes()};
}

void GuestPageTableSnapshot::Retain(ResourceBindings& bindings) const {
    bindings.AddIndirectResources(resources);
    for (const auto& table : tables) {
        bindings.RetainUploadAllocation(table);
    }
}

} // namespace Metal
