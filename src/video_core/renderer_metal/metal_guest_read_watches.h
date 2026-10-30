// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>
#include "common/alignment.h"
#include "common/types.h"

namespace Metal {

class GuestReadWatches {
public:
    struct Range {
        VAddr address;
        u64 size;
    };

    template <typename Func>
    void Update(VAddr address, u64 size, bool track, Func&& func) {
        constexpr u64 AddressSpaceSize = u64{1} << 40;
        constexpr u64 CpuPageSize = 4096;
        if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address) {
            throw std::runtime_error("Invalid native guest read-watch range");
        }
        const VAddr begin = Common::AlignDown(address, CpuPageSize);
        const VAddr end = Common::AlignUp(address + size, CpuPageSize);
        std::vector<Range> transitions;
        VAddr cursor = begin;
        auto first = ranges.lower_bound(begin);
        if (first != ranges.begin()) {
            const auto previous = std::prev(first);
            if (previous->second > begin || (track && previous->second == begin)) {
                first = previous;
            }
        }
        const auto last = track ? ranges.upper_bound(end) : ranges.lower_bound(end);
        for (auto it = first; it != last; ++it) {
            const VAddr range_begin = std::max(begin, it->first);
            const VAddr range_end = std::min(end, it->second);
            if (track && cursor < range_begin) {
                transitions.push_back({cursor, range_begin - cursor});
            } else if (!track && range_begin < range_end) {
                transitions.push_back({range_begin, range_end - range_begin});
            }
            cursor = range_end;
        }
        if (track && cursor < end) {
            transitions.push_back({cursor, end - cursor});
        }
        if (transitions.empty()) {
            return;
        }
        std::map<VAddr, VAddr> replacement;
        if (track) {
            const VAddr merged_begin = first == last ? begin : std::min(begin, first->first);
            const VAddr merged_end = first == last ? end : std::max(end, std::prev(last)->second);
            replacement.emplace(merged_begin, merged_end);
        } else {
            if (first->first < begin) {
                replacement.emplace(first->first, begin);
            }
            const auto previous = std::prev(last);
            if (previous->second > end) {
                replacement.emplace(end, previous->second);
            }
        }
        func(std::span<const Range>{transitions}, track);
        ranges.erase(first, last);
        ranges.merge(replacement);
    }

private:
    std::map<VAddr, VAddr> ranges;
};

} // namespace Metal
