// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <set>
#include <vector>
#include "video_core/renderer_metal/metal_guest_ranges.h"

namespace Metal {

class GuestPhysicalMappings {
public:
    struct BackingRange {
        VAddr address;
        u64 size;
        std::optional<PAddr> physical_address;
    };

    struct Range : BackingRange {
        u64 mapping_epoch;
    };

    struct Alias {
        VAddr address;
        VAddr alias_address;
        u64 size;
        PAddr physical_address;
        u64 mapping_epoch;
    };

    struct PhysicalMapping {
        PAddr address;
        u64 size;
        u64 mapping_epoch;
    };

    void Map(std::span<const BackingRange> ranges) {
        VAddr previous_end{};
        for (const auto& range : ranges) {
            Validate(range.address, range.size);
            if (range.address < previous_end) {
                throw std::runtime_error("Metal backing ranges overlap or are unordered");
            }
            if (range.physical_address) {
                Validate(*range.physical_address, range.size);
            }
            previous_end = range.address + range.size;
        }
        if (std::all_of(ranges.begin(), ranges.end(),
                        [this](const BackingRange& range) { return Matches(range); })) {
            return;
        }
        GuestPhysicalMappings prepared;
        prepared.current_epoch = current_epoch;
        std::vector<VAddr> replaced_virtual;
        std::vector<PAddr> replaced_physical;
        for (const auto& range : ranges) {
            virtual_ranges.StageRange(range.address, range.size, prepared.virtual_ranges,
                                      replaced_virtual);
            for (const auto& old : VirtualRanges(range.address, range.size)) {
                StagePhysical(old.address + old.translation, old.size, prepared, replaced_physical);
            }
            if (range.physical_address) {
                StagePhysical(*range.physical_address, range.size, prepared, replaced_physical);
            }
        }
        prepared.ApplyMap(ranges);
        virtual_ranges.CommitPrepared(prepared.virtual_ranges, replaced_virtual);
        for (const PAddr address : replaced_physical) {
            physical.erase(address);
        }
        while (!prepared.physical.empty()) {
            const PAddr address = prepared.physical.begin()->first;
            const PAddr end = prepared.physical.begin()->second.end;
            physical.insert(prepared.physical.extract(prepared.physical.begin()));
            Coalesce(address, end);
        }
        current_epoch = prepared.current_epoch;
    }

    void Unmap(VAddr address, u64 size) {
        const BackingRange range{address, size, std::nullopt};
        Map(std::span{&range, 1});
    }

    void Clear() {
        virtual_ranges.Clear();
        physical.clear();
    }

    [[nodiscard]] std::vector<Range> GetRanges(VAddr address, u64 size) const {
        Validate(address, size);
        std::vector<Range> result;
        VAddr cursor = address;
        const auto append = [&](VAddr start, u64 length, std::optional<PAddr> backing, u64 epoch) {
            if (!result.empty() && result.back().address + result.back().size == start &&
                result.back().mapping_epoch == epoch &&
                ((!backing && !result.back().physical_address) ||
                 (backing && result.back().physical_address &&
                  *result.back().physical_address + result.back().size == *backing))) {
                result.back().size += length;
            } else {
                result.push_back({{start, length, backing}, epoch});
            }
        };
        for (const auto& range : VirtualRanges(address, size)) {
            if (cursor < range.address) {
                append(cursor, range.address - cursor, std::nullopt, 0);
            }
            ForEachPhysical(range.address + range.translation, range.size,
                            [&](PAddr first, PAddr last, const PhysicalRange& owner) {
                                append(first - range.translation, last - first, first,
                                       owner.mapping_epoch);
                            });
            cursor = range.address + range.size;
        }
        if (cursor < address + size) {
            append(cursor, address + size - cursor, std::nullopt, 0);
        }
        return result;
    }

    [[nodiscard]] std::vector<Alias> GetAliases(VAddr address, u64 size) const {
        Validate(address, size);
        std::vector<Alias> result;
        for (const auto& range : VirtualRanges(address, size)) {
            ForEachPhysical(range.address + range.translation, range.size,
                            [&](PAddr first, PAddr last, const PhysicalRange& owner) {
                                for (const u64 translation : owner.translations) {
                                    result.push_back({first - range.translation,
                                                      first - translation, last - first, first,
                                                      owner.mapping_epoch});
                                }
                            });
        }
        return result;
    }

    [[nodiscard]] std::vector<PhysicalMapping> GetPhysicalRanges(PAddr address, u64 size) const {
        Validate(address, size);
        std::vector<PhysicalMapping> result;
        const PAddr end = address + size;
        auto it = physical.upper_bound(address);
        if (it != physical.begin()) {
            --it;
        }
        for (; it != physical.end() && it->first < end; ++it) {
            const PAddr first = std::max(address, it->first);
            const PAddr last = std::min(end, it->second.end);
            if (first < last) {
                result.push_back({first, last - first, it->second.mapping_epoch});
            }
        }
        return result;
    }

private:
    struct VirtualRange {
        VAddr address;
        u64 size;
        u64 translation;
    };

    struct PhysicalRange {
        PAddr end;
        u64 mapping_epoch;
        std::set<u64> translations;
    };

    [[nodiscard]] bool Matches(const BackingRange& range) const {
        if (!range.physical_address) {
            return !virtual_ranges.Intersects(range.address, range.size);
        }
        const u64 translation = *range.physical_address - range.address;
        VAddr cursor = range.address;
        bool matches = true;
        virtual_ranges.ForEachInRange(
            range.address, range.size, [&](VAddr first, VAddr last, u64 previous_translation) {
                matches &= first == cursor && previous_translation == translation;
                cursor = last;
            });
        return matches && cursor == range.address + range.size;
    }

    void StagePhysical(PAddr address, u64 size, GuestPhysicalMappings& prepared,
                       std::vector<PAddr>& replaced) const {
        const PAddr end = address + size;
        auto it = physical.upper_bound(address);
        if (it != physical.begin()) {
            --it;
        }
        for (; it != physical.end() && it->first < end; ++it) {
            if (it->second.end > address && !prepared.physical.contains(it->first)) {
                prepared.physical.emplace(*it);
                replaced.push_back(it->first);
            }
        }
    }

    void ApplyMap(std::span<const BackingRange> ranges) {
        for (const auto& range : ranges) {
            if (!range.physical_address) {
                ApplyUnmap(range.address, range.size);
                continue;
            }
            const u64 translation = *range.physical_address - range.address;
            const auto previous = VirtualRanges(range.address, range.size);
            VAddr cursor = range.address;
            for (const auto& old : previous) {
                if (cursor < old.address) {
                    Add(cursor, old.address - cursor, translation);
                }
                if (old.translation != translation) {
                    ApplyUnmap(old.address, old.size);
                    Add(old.address, old.size, translation);
                }
                cursor = old.address + old.size;
            }
            if (cursor < range.address + range.size) {
                Add(cursor, range.address + range.size - cursor, translation);
            }
        }
    }

    void ApplyUnmap(VAddr address, u64 size) {
        Validate(address, size);
        for (const auto& range : VirtualRanges(address, size)) {
            const PAddr begin = range.address + range.translation;
            const PAddr end = begin + range.size;
            Split(begin);
            Split(end);
            for (auto it = physical.lower_bound(begin); it != physical.end() && it->first < end;) {
                if (!it->second.translations.erase(range.translation)) {
                    throw std::runtime_error("Metal physical alias is missing during unmap");
                }
                if (it->second.translations.empty()) {
                    it = physical.erase(it);
                } else {
                    ++it;
                }
            }
            virtual_ranges.Subtract(range.address, range.size);
            Coalesce(begin, end);
        }
    }

    static void Validate(VAddr address, u64 size) {
        constexpr u64 AddressSpaceSize = u64{1} << 40;
        if (!size || address >= AddressSpaceSize || size > AddressSpaceSize - address) {
            throw std::runtime_error("Invalid Metal physical mapping range");
        }
    }

    [[nodiscard]] std::vector<VirtualRange> VirtualRanges(VAddr address, u64 size) const {
        std::vector<VirtualRange> result;
        virtual_ranges.ForEachInRange(address, size, [&](VAddr first, VAddr last, u64 translation) {
            result.push_back({first, last - first, translation});
        });
        return result;
    }

    template <typename Func>
    void ForEachPhysical(PAddr address, u64 size, Func&& func) const {
        const PAddr end = address + size;
        auto it = physical.upper_bound(address);
        if (it != physical.begin()) {
            --it;
        }
        while (address < end) {
            if (it == physical.end() || it->first > address || it->second.end <= address) {
                throw std::runtime_error("Metal virtual mapping has no physical owner");
            }
            const PAddr next = std::min(end, it->second.end);
            func(address, next, it->second);
            address = next;
            ++it;
        }
    }

    void Split(PAddr address) {
        auto it = physical.upper_bound(address);
        if (it == physical.begin()) {
            return;
        }
        --it;
        if (it->first < address && address < it->second.end) {
            physical.emplace(address, it->second);
            it->second.end = address;
        }
    }

    void Coalesce(PAddr begin, PAddr end) {
        auto it = physical.lower_bound(begin);
        if (it != physical.begin()) {
            --it;
        }
        while (it != physical.end() && it->first <= end) {
            const auto next = std::next(it);
            if (next == physical.end()) {
                break;
            }
            if (it->second.end == next->first &&
                it->second.mapping_epoch == next->second.mapping_epoch &&
                it->second.translations == next->second.translations) {
                it->second.end = next->second.end;
                physical.erase(next);
            } else {
                it = next;
            }
        }
    }

    void Add(VAddr address, u64 size, u64 translation) {
        const PAddr begin = address + translation;
        const PAddr end = begin + size;
        Split(begin);
        Split(end);
        PAddr cursor = begin;
        while (cursor < end) {
            auto it = physical.lower_bound(cursor);
            if (it == physical.end() || it->first != cursor) {
                if (current_epoch == std::numeric_limits<u64>::max()) {
                    throw std::runtime_error("Metal physical mapping epoch exhausted");
                }
                const PAddr next = it == physical.end() ? end : std::min(end, it->first);
                physical.emplace(cursor, PhysicalRange{next, ++current_epoch, {translation}});
                cursor = next;
            } else {
                it->second.translations.insert(translation);
                cursor = it->second.end;
            }
        }
        virtual_ranges.Add(address, size, translation);
        Coalesce(begin, end);
    }

    GuestRangeMap<u64> virtual_ranges;
    std::map<PAddr, PhysicalRange> physical;
    u64 current_epoch{};
};

} // namespace Metal
