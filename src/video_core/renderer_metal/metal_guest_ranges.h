// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>
#include "common/types.h"

namespace Metal {

struct GuestBufferExtent {
    u32 resource_index;
    u64 size;
};

[[nodiscard]] inline u64 GuestBufferSize(u32 resource_index, u64 descriptor_size,
                                         std::span<const GuestBufferExtent> extents) {
    u64 size = descriptor_size;
    bool found{};
    for (const auto& extent : extents) {
        if (extent.resource_index != resource_index) {
            continue;
        }
        if (found || !extent.size || extent.size < descriptor_size ||
            extent.size > std::numeric_limits<u32>::max()) {
            throw std::runtime_error("Invalid native geometry guest buffer extent");
        }
        found = true;
        size = extent.size;
    }
    return size;
}

[[nodiscard]] inline u32 GuestBufferCapacity(u64 offset, u64 size, u64 buffer_length) {
    if (offset > buffer_length || size > buffer_length - offset ||
        size > std::numeric_limits<u32>::max()) {
        throw std::runtime_error("Invalid native guest buffer binding range");
    }
    return u32(size);
}

template <typename T>
class GuestRangeMap {
public:
    void StageRange(VAddr address, u64 size, GuestRangeMap& prepared,
                    std::vector<VAddr>& replaced) const {
        if (!size) {
            return;
        }
        const VAddr end = End(address, size);
        for (auto it = First(address); it != ranges.end() && it->first < end; ++it) {
            if (!prepared.ranges.contains(it->first)) {
                prepared.ranges.emplace(*it);
                replaced.push_back(it->first);
            }
        }
    }

    void CommitPrepared(GuestRangeMap& prepared, std::span<const VAddr> replaced) {
        for (const VAddr address : replaced) {
            ranges.erase(address);
        }
        while (!prepared.ranges.empty()) {
            auto result = ranges.insert(prepared.ranges.extract(prepared.ranges.begin()));
            Coalesce(result.position);
        }
    }

    void Add(VAddr address, u64 size, const T& value) {
        if (!size) {
            return;
        }
        const VAddr end = End(address, size);
        std::map<VAddr, Entry> insertion;
        insertion.emplace(address, Entry{end, value});
        Subtract(address, size);
        ranges.merge(insertion);
        Coalesce(ranges.find(address));
    }

    void Subtract(VAddr address, u64 size) {
        if (!size) {
            return;
        }
        const VAddr end = End(address, size);
        Split(address);
        Split(end);
        ranges.erase(ranges.lower_bound(address), ranges.lower_bound(end));
    }

    void Clear() {
        ranges.clear();
    }

    [[nodiscard]] bool Contains(VAddr address, u64 size) const {
        const VAddr end = End(address, size);
        for (auto it = First(address); address < end && it != ranges.end(); ++it) {
            if (it->first > address) {
                return false;
            }
            address = std::max(address, it->second.end);
        }
        return address >= end;
    }

    [[nodiscard]] bool Intersects(VAddr address, u64 size) const {
        const VAddr end = End(address, size);
        const auto it = First(address);
        return size && it != ranges.end() && it->first < end;
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        for (const auto& [address, entry] : ranges) {
            func(address, entry.end, entry.value);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr address, u64 size, Func&& func) const {
        if (!size) {
            return;
        }
        const VAddr end = End(address, size);
        for (auto it = First(address); it != ranges.end() && it->first < end; ++it) {
            func(std::max(address, it->first), std::min(end, it->second.end), it->second.value);
        }
    }

    template <typename Func>
    void ForEachIntersectingRange(VAddr address, u64 size, Func&& func) const {
        if (!size) {
            return;
        }
        const VAddr end = End(address, size);
        for (auto it = First(address); it != ranges.end() && it->first < end; ++it) {
            func(it->first, it->second.end, it->second.value);
        }
    }

private:
    struct Entry {
        VAddr end;
        T value;
    };

    void Coalesce(typename std::map<VAddr, Entry>::iterator it) {
        if (it != ranges.begin()) {
            const auto previous = std::prev(it);
            if (previous->second.end == it->first && previous->second.value == it->second.value) {
                previous->second.end = it->second.end;
                ranges.erase(it);
                it = previous;
            }
        }
        const auto next = std::next(it);
        if (next != ranges.end() && it->second.end == next->first &&
            it->second.value == next->second.value) {
            it->second.end = next->second.end;
            ranges.erase(next);
        }
    }

    static VAddr End(VAddr address, u64 size) {
        if (size > std::numeric_limits<VAddr>::max() - address) {
            throw std::runtime_error("Metal guest interval exceeds the address range");
        }
        return address + size;
    }

    [[nodiscard]] auto First(VAddr address) const {
        auto it = ranges.upper_bound(address);
        if (it != ranges.begin()) {
            const auto previous = std::prev(it);
            if (previous->second.end > address) {
                it = previous;
            }
        }
        return it;
    }

    void Split(VAddr address) {
        auto it = ranges.upper_bound(address);
        if (it == ranges.begin()) {
            return;
        }
        --it;
        if (it->first < address && address < it->second.end) {
            ranges.emplace(address, it->second);
            it->second.end = address;
        }
    }

    std::map<VAddr, Entry> ranges;
};

class GuestRangeSet {
public:
    void Add(VAddr address, u64 size) {
        ranges.Add(address, size, true);
    }

    void Subtract(VAddr address, u64 size) {
        ranges.Subtract(address, size);
    }

    void Clear() {
        ranges.Clear();
    }

    [[nodiscard]] bool Contains(VAddr address, u64 size) const {
        return ranges.Contains(address, size);
    }

    [[nodiscard]] bool Intersects(VAddr address, u64 size) const {
        return ranges.Intersects(address, size);
    }

    template <typename Func>
    void ForEachInRange(VAddr address, u64 size, Func&& func) const {
        ranges.ForEachInRange(address, size,
                              [&](VAddr begin, VAddr end, bool) { func(begin, end); });
    }

    template <typename Func>
    void ForEachIntersectingRange(VAddr address, u64 size, Func&& func) const {
        ranges.ForEachIntersectingRange(address, size,
                                        [&](VAddr begin, VAddr end, bool) { func(begin, end); });
    }

private:
    GuestRangeMap<bool> ranges;
};

} // namespace Metal
