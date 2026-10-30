// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <exception>
#include <functional>
#include <span>
#include <string_view>
#include <thread>
#include <fmt/format.h>
#include "common/scope_exit.h"
#include "common/types.h"

namespace AmdGpu {
struct PM4CmdCopyData;
struct PM4CmdEventWrite;
struct PM4CmdEventWriteEop;
struct PM4CmdReleaseMem;
struct PM4CmdWaitRegMem;
struct PM4CmdWriteData;
} // namespace AmdGpu

namespace VideoCore {

template <typename Function>
[[nodiscard]] auto GuardMemoryOperation(Function&& operation_failed) {
    const int exception_count = std::uncaught_exceptions();
    return ::detail::MakeScopeGuard(
        [exception_count, operation_failed = std::forward<Function>(operation_failed)] {
            if (std::uncaught_exceptions() > exception_count) {
                operation_failed();
            }
        });
}

class Rasterizer {
public:
    virtual ~Rasterizer() = default;

    virtual void Draw(bool is_indexed, u32 index_offset = 0) = 0;
    virtual void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) = 0;
    virtual void DispatchDirect() = 0;
    virtual void DispatchIndirect(VAddr address, u32 offset, u32 size) = 0;

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    virtual void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false) = 0;
    virtual void ScopeMarkerEnd(bool from_guest = false) = 0;
    virtual void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false) = 0;
    virtual void ScopedMarkerInsertColor(const std::string_view& str, u32 color,
                                         bool from_guest = false) = 0;

    virtual void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) = 0;
    virtual void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) = 0;
    virtual void SynchronizeDma() {}
    virtual bool CopyData(const AmdGpu::PM4CmdCopyData&) {
        return false;
    }
    virtual bool WriteData(const AmdGpu::PM4CmdWriteData&) {
        return false;
    }
    virtual bool PixelStatistics(const AmdGpu::PM4CmdEventWrite&, u32) {
        return false;
    }
    virtual bool EventWriteEop(const AmdGpu::PM4CmdEventWriteEop&) {
        return false;
    }
    virtual bool ReleaseMem(const AmdGpu::PM4CmdReleaseMem&, u32) {
        return false;
    }
    virtual void PrepareWaitRegMem(const AmdGpu::PM4CmdWaitRegMem&) {}
    virtual void PrepareMemoryAccess(VAddr, u64, bool = false) {}
    virtual u32 ReadDataFromGds(u32 gds_offset) = 0;
    virtual bool StoreGds(VAddr, u32, u32) {
        return false;
    }
    virtual bool WriteMemory(VAddr, std::span<const u8>) {
        return false;
    }
    virtual bool WithMemoryWriteWatch(VAddr, u64, const std::function<void()>&) {
        return false;
    }
    virtual bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false) = 0;
    virtual bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false) = 0;
    virtual bool IsMapped(VAddr addr, u64 size) = 0;
    virtual bool RequiresMemoryMappingDispatch() const {
        return false;
    }
    virtual void DispatchMemoryMapping(const std::function<void()>& function) {
        function();
    }
    virtual void OnMemoryOperationException() noexcept {}
    virtual void MapMemory(VAddr addr, u64 size) = 0;
    virtual void RegisterMemory(VAddr addr, u64 size) = 0;
    virtual void UnmapMemory(VAddr addr, u64 size) = 0;
    virtual bool ProtectGuestMemory(VAddr, u64, u32) {
        return false;
    }
    virtual bool HandleGuestMemoryFault(VAddr addr, u64 size, bool is_write, bool is_execute,
                                        bool assume_locks) {
        return is_write ? InvalidateMemory(addr, size, assume_locks)
                        : ReadMemory(addr, size, assume_locks);
    }

    virtual u64 Flush() = 0;
    virtual void Finish() = 0;
    virtual void OnSubmit() = 0;
    virtual void OnFence() = 0;

    virtual std::thread::id GetGpuCommandProcessorThread() = 0;
#ifdef __linux__
    virtual u32 GetGpuCommandProcessorThreadId() = 0;
#endif

protected:
    explicit Rasterizer(bool host_markers_enabled_) : host_markers_enabled{host_markers_enabled_} {}

private:
    const bool host_markers_enabled;
};

} // namespace VideoCore
