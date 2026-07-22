// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <limits>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/sleepq.h"
#include "core/libraries/libs.h"

namespace Libraries::Kernel {

enum class UmtxOp : s32 {
    SemWait = 19,
    SemWake = 20,
};

struct UmtxSemaphore {
    u32 count;
    u32 has_waiters;
    u32 flags;
};

static s32 SetUmtxError(s32 error) {
    *__Error() = error;
    return -1;
}

static bool HasWaiters(UmtxSemaphore* sem) {
    return SleepqLookup(sem) != nullptr;
}

static s32 WaitUmtxSemaphore(UmtxSemaphore* sem, const OrbisKernelTimespec* timeout) {
    if (timeout != nullptr &&
        (timeout->tv_sec < 0 || timeout->tv_nsec < 0 || timeout->tv_nsec >= 1000000000)) {
        return SetUmtxError(POSIX_EINVAL);
    }

    auto count = std::atomic_ref{sem->count};
    auto has_waiters = std::atomic_ref{sem->has_waiters};
    Pthread* curthread = g_curthread;

    ASSERT_MSG(curthread->wchan == nullptr, "Thread was already on a sleep queue.");
    SleepqLock(sem);

    if (count.load(std::memory_order_acquire) != 0) {
        SleepqUnlock(sem);
        return 0;
    }

    curthread->ClearWake();
    has_waiters.store(1, std::memory_order_release);

    // Check the count again after publishing has_waiters. A concurrent post may have
    // incremented the count before it could observe the waiter flag.
    if (count.load(std::memory_order_acquire) != 0) {
        if (!HasWaiters(sem)) {
            has_waiters.store(0, std::memory_order_release);
        }
        SleepqUnlock(sem);
        return 0;
    }

    curthread->will_sleep = true;
    SleepqAdd(sem, curthread);
    SleepqUnlock(sem);

    bool awakened;
    if (timeout == nullptr) {
        awakened = curthread->Sleep(nullptr, 0);
    } else {
        constexpr u64 MaxMicroseconds = std::numeric_limits<s64>::max();
        const u64 seconds = static_cast<u64>(timeout->tv_sec);
        const u64 nanoseconds = static_cast<u64>(timeout->tv_nsec);
        const u64 microseconds =
            seconds > MaxMicroseconds / 1000000
                ? MaxMicroseconds
                : std::min(MaxMicroseconds,
                           seconds * 1000000 + (nanoseconds + 999) / 1000);
        awakened = curthread->Sleep(THR_RELTIME, microseconds);
    }

    SleepqLock(sem);
    if (curthread->wchan == nullptr) {
        // Wake removes the thread from the queue before releasing its semaphore.
        // Consume a release which raced with the timeout path, if there is one.
        curthread->ClearWake();
        SleepqUnlock(sem);
        return 0;
    }

    ASSERT(!awakened);
    SleepQueue* sq = SleepqLookup(sem);
    ASSERT(sq != nullptr);
    const bool remaining = SleepqRemove(sq, curthread);
    has_waiters.store(remaining ? 1 : 0, std::memory_order_release);
    SleepqUnlock(sem);
    return SetUmtxError(POSIX_ETIMEDOUT);
}

static s32 WakeUmtxSemaphore(UmtxSemaphore* sem) {
    auto has_waiters = std::atomic_ref{sem->has_waiters};

    SleepqLock(sem);
    SleepQueue* sq = SleepqLookup(sem);
    if (sq == nullptr) {
        has_waiters.store(0, std::memory_order_release);
        SleepqUnlock(sem);
        return 0;
    }

    Pthread* thread = sq->sq_blocked.front();
    const bool remaining = SleepqRemove(sq, thread);
    has_waiters.store(remaining ? 1 : 0, std::memory_order_release);
    thread->wake_sema.release();
    SleepqUnlock(sem);
    return 0;
}

s32 PS4_SYSV_ABI posix_umtx_op(void* obj, UmtxOp op, u64 val, void* uaddr, void* uaddr2) {
    if (obj == nullptr) {
        return SetUmtxError(POSIX_EFAULT);
    }

    auto* sem = static_cast<UmtxSemaphore*>(obj);
    switch (op) {
    case UmtxOp::SemWait:
        return WaitUmtxSemaphore(sem, static_cast<const OrbisKernelTimespec*>(uaddr2));
    case UmtxOp::SemWake:
        return WakeUmtxSemaphore(sem);
    default:
        LOG_ERROR(Kernel_Pthread,
                  "Unsupported _umtx_op: obj = {}, op = {}, val = {:#x}, uaddr = {}, "
                  "uaddr2 = {}",
                  fmt::ptr(obj), static_cast<s32>(op), val, fmt::ptr(uaddr), fmt::ptr(uaddr2));
        return SetUmtxError(POSIX_EINVAL);
    }
}

void RegisterUmtx(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("04AjkP0jO9U", "libkernel", 1, "libkernel", posix_umtx_op);
}

} // namespace Libraries::Kernel
