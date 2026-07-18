// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Native ARM64/FEX port: real implementation of the FreeBSD syscall
// `_umtx_op` (libkernel NID `04AjkP0jO9U`). Previously the function was only an
// aerolib STUB that returned 0 without ever blocking. As a result, futex-like
// wait paths (pthread condvars, libc-internal locks, POSIX semaphores) returned
// immediately, and the caller entered a pure busy-wait loop. Visible in the log
// as endless `_umtx_op` repetitions on "Background Job.Worker".
//
// This file maps the relevant `_umtx_op` operations onto a host-side,
// address-based futex (std::mutex + std::condition_variable). Under FEX the
// guest memory is mapped flat into the host address space, so the passed
// pointers are directly dereferenceable.
//
// The blocking operations follow the FreeBSD 9 (= PS4) kernel semantics from
// sys/kern/kern_umtx.c, not Linux futex semantics. Two aspects are essential
// for correctness (they caused permanent lost wakeups when modelled as plain
// futexes):
//  * MUTEX_WAIT must *set UMUTEX_CONTESTED* before sleeping and return as soon
//    as the owner word is free. libthr's unlock fast path only issues
//    MUTEX_WAKE/MUTEX_WAKE2 when it observed the contested bit.
//  * CV_WAIT must *set c_has_waiters* before releasing the mutex. libthr's
//    _thr_ucond_signal/_broadcast check this word in userland and skip the
//    CV_SIGNAL syscall entirely while it is 0.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <mutex>
#include <pthread.h>

#include "common/logging/log.h"
#include "common/types.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"

namespace Libraries::Kernel {

namespace {

// FreeBSD `sys/umtx.h` - operations (stable across FreeBSD 9..13, matching the
// PS4 libkernel ABI).
enum UmtxOp : s32 {
    UMTX_OP_WAIT = 2,
    UMTX_OP_WAKE = 3,
    UMTX_OP_MUTEX_TRYLOCK = 4,
    UMTX_OP_MUTEX_LOCK = 5,
    UMTX_OP_MUTEX_UNLOCK = 6,
    UMTX_OP_SET_CEILING = 7,
    UMTX_OP_CV_WAIT = 8,
    UMTX_OP_CV_SIGNAL = 9,
    UMTX_OP_CV_BROADCAST = 10,
    UMTX_OP_WAIT_UINT = 11,
    UMTX_OP_RW_RDLOCK = 12,
    UMTX_OP_RW_WRLOCK = 13,
    UMTX_OP_RW_UNLOCK = 14,
    UMTX_OP_WAIT_UINT_PRIVATE = 15,
    UMTX_OP_WAKE_PRIVATE = 16,
    UMTX_OP_MUTEX_WAIT = 17,
    UMTX_OP_MUTEX_WAKE = 18,
    UMTX_OP_SEM_WAIT = 19,
    UMTX_OP_SEM_WAKE = 20,
    UMTX_OP_NWAKE_PRIVATE = 21,
    UMTX_OP_MUTEX_WAKE2 = 22,
    UMTX_OP_SEM2_WAIT = 23,
    UMTX_OP_SEM2_WAKE = 24,
};

// Flags in `struct _umtx_time._flags`.
constexpr u32 UMTX_ABSTIME = 0x01;

// Flags passed in `val` for UMTX_OP_CV_WAIT.
[[maybe_unused]] constexpr u32 CVWAIT_CHECK_UNPARKING = 0x01;
constexpr u32 CVWAIT_ABSTIME = 0x02;
constexpr u32 CVWAIT_CLOCKID = 0x04;

// umutex word bits (m_owner).
constexpr u32 UMUTEX_UNOWNED = 0x0;
constexpr u32 UMUTEX_CONTESTED = 0x80000000u;

// _usem2 word bits (count).
constexpr u32 USEM_HAS_WAITERS = 0x80000000u;
constexpr u32 USEM_COUNT_MASK = 0x7fffffffu;

// FreeBSD clock ids that tick CLOCK_REALTIME; everything else is treated as
// monotonic (CLOCK_MONOTONIC/UPTIME variants), matching the mapping used by the
// posix_clock_gettime HLE in time.cpp.
constexpr bool IsRealtimeClockId(u32 clockid) {
    return clockid == 0 /*REALTIME*/ || clockid == 9 /*REALTIME_PRECISE*/ ||
           clockid == 10 /*REALTIME_FAST*/ || clockid == 13 /*SECOND*/;
}

struct GuestTimespec {
    s64 tv_sec;
    s64 tv_nsec;
};

// struct _umtx_time.
struct GuestUmtxTime {
    GuestTimespec timeout;
    u32 flags;
    u32 clockid;
};

// struct ucond { volatile u32 c_has_waiters; u32 c_flags; u32 c_clockid; u32 c_spare[1]; }
constexpr uintptr_t UcondClockidOffset = 8;

// A futex bucket groups several guest addresses. Waiters wait on a sequence
// number; every wake increments it and wakes all waiters of the bucket. After
// waking up the caller re-checks its own guest word, so collective wakes across
// bucket collisions are semantically harmless (at most one extra pass that
// immediately blocks again).
struct FutexBucket {
    std::mutex m;
    std::condition_variable cv;
    std::atomic<u64> seq{0};
};

constexpr size_t NumBuckets = 256;
std::array<FutexBucket, NumBuckets> g_buckets{};

size_t BucketIndexFor(const void* addr) {
    const auto key = reinterpret_cast<uintptr_t>(addr) >> 3;
    return key % NumBuckets;
}

FutexBucket& BucketFor(const void* addr) {
    return g_buckets[BucketIndexFor(addr)];
}

u32 LoadU32(const void* addr) {
    return reinterpret_cast<const volatile std::atomic<u32>*>(addr)->load(std::memory_order_acquire);
}

u64 LoadU64(const void* addr) {
    return reinterpret_cast<const volatile std::atomic<u64>*>(addr)->load(std::memory_order_acquire);
}

void StoreU32(void* addr, u32 value) {
    reinterpret_cast<volatile std::atomic<u32>*>(addr)->store(value, std::memory_order_release);
}

u32 FetchOrU32(void* addr, u32 value) {
    return reinterpret_cast<volatile std::atomic<u32>*>(addr)->fetch_or(
        value, std::memory_order_acq_rel);
}

bool CasU32(void* addr, u32& expected, u32 desired) {
    return reinterpret_cast<volatile std::atomic<u32>*>(addr)->compare_exchange_strong(
        expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Diagnostics: lock-free ring buffer of the most recent wait/wake events. Never
// alters guest-visible behavior; used only to report waiters that stay blocked
// (see WaitForeverWithDiagnostics) so lost-wakeup bugs show up in the log with
// the full wait/wake history of the affected bucket.

enum class UmtxEvent : u8 {
    WaitEnter,
    WaitExit,
    Wake,
};

struct TraceEntry {
    u64 order;
    u64 bucket_seq;
    const void* obj;
    u64 val;
    u64 word;
    s32 op;
    s32 ret;
    u16 bucket;
    UmtxEvent event;
    char thread_name[16];
};

constexpr size_t TraceSize = 1024;
std::array<TraceEntry, TraceSize> g_trace{};
std::atomic<u64> g_trace_order{0};

const char* OpName(s32 op) {
    switch (op) {
    case UMTX_OP_WAIT:
        return "WAIT";
    case UMTX_OP_WAKE:
        return "WAKE";
    case UMTX_OP_CV_WAIT:
        return "CV_WAIT";
    case UMTX_OP_CV_SIGNAL:
        return "CV_SIGNAL";
    case UMTX_OP_CV_BROADCAST:
        return "CV_BROADCAST";
    case UMTX_OP_WAIT_UINT:
        return "WAIT_UINT";
    case UMTX_OP_WAIT_UINT_PRIVATE:
        return "WAIT_UINT_PRIVATE";
    case UMTX_OP_WAKE_PRIVATE:
        return "WAKE_PRIVATE";
    case UMTX_OP_MUTEX_WAIT:
        return "MUTEX_WAIT";
    case UMTX_OP_MUTEX_WAKE:
        return "MUTEX_WAKE";
    case UMTX_OP_MUTEX_WAKE2:
        return "MUTEX_WAKE2";
    case UMTX_OP_SEM_WAIT:
        return "SEM_WAIT";
    case UMTX_OP_SEM_WAKE:
        return "SEM_WAKE";
    case UMTX_OP_SEM2_WAIT:
        return "SEM2_WAIT";
    case UMTX_OP_SEM2_WAKE:
        return "SEM2_WAKE";
    case UMTX_OP_NWAKE_PRIVATE:
        return "NWAKE_PRIVATE";
    default:
        return "?";
    }
}

const char* EventName(UmtxEvent ev) {
    switch (ev) {
    case UmtxEvent::WaitEnter:
        return "wait-enter";
    case UmtxEvent::WaitExit:
        return "wait-exit";
    case UmtxEvent::Wake:
        return "wake";
    }
    return "?";
}

void Trace(UmtxEvent event, s32 op, const void* obj, u64 val, u64 word, s32 ret) {
    const u64 order = g_trace_order.fetch_add(1, std::memory_order_relaxed);
    TraceEntry& e = g_trace[order % TraceSize];
    e.order = order;
    e.obj = obj;
    e.val = val;
    e.word = word;
    e.op = op;
    e.ret = ret;
    e.bucket = static_cast<u16>(BucketIndexFor(obj));
    e.bucket_seq = BucketFor(obj).seq.load(std::memory_order_relaxed);
    e.event = event;
    e.thread_name[0] = '\0';
    pthread_getname_np(pthread_self(), e.thread_name, sizeof(e.thread_name));
}

u64 SampleWord(s32 op, const void* obj) {
    if (op == UMTX_OP_NWAKE_PRIVATE) {
        return 0; // obj is a pointer array, not a futex word.
    }
    if (op == UMTX_OP_WAIT) {
        return LoadU64(obj);
    }
    return LoadU32(obj);
}

// Logs one stuck waiter together with the recent wait/wake history of its
// bucket. Called while holding the bucket lock; pure logging, the caller
// continues to wait with unchanged semantics afterwards.
void LogStuckWaiter(const FutexBucket& b, const void* obj, s32 op, u64 val, u64 start_seq) {
    char name[16]{};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    LOG_WARNING(Lib_Kernel,
                "_umtx_op: waiter blocked >30s: thread='{}' op={} obj={} val={:#x} word={:#x} "
                "bucket={} seq_start={} seq_now={}",
                name, OpName(op), obj, val, SampleWord(op, obj), BucketIndexFor(obj), start_seq,
                b.seq.load(std::memory_order_relaxed));

    // Dump the ring entries of this bucket in order (oldest first).
    const u64 next = g_trace_order.load(std::memory_order_relaxed);
    const u64 first = next > TraceSize ? next - TraceSize : 0;
    const auto bucket = static_cast<u16>(BucketIndexFor(obj));
    for (u64 i = first; i < next; ++i) {
        const TraceEntry& e = g_trace[i % TraceSize];
        if (e.order != i || e.bucket != bucket) {
            continue; // overwritten concurrently or different bucket
        }
        LOG_WARNING(Lib_Kernel,
                    "_umtx_op:   [{}] {} {} thread='{}' obj={} val={:#x} word={:#x} seq={} ret={}",
                    e.order, EventName(e.event), OpName(e.op), e.thread_name, e.obj, e.val, e.word,
                    e.bucket_seq, e.ret);
    }
    // The process is usually killed while stuck; flush so the diagnosis
    // survives in the log file instead of dying in the sink buffer.
    Common::Log::Flush();
}

// Untimed wait: block until the bucket sequence moves past `start`. Purely as a
// diagnostic (never guest-visible) the first 30s are waited in a separate step;
// if no wake arrived by then, the stuck waiter and the bucket history are
// logged once and the wait continues indefinitely.
void WaitForeverWithDiagnostics(FutexBucket& b, std::unique_lock<std::mutex>& lk, u64 start,
                                const void* obj, s32 op, u64 val) {
    const auto moved = [&] { return b.seq.load(std::memory_order_relaxed) != start; };
    if (b.cv.wait_for(lk, std::chrono::seconds{30}, moved)) {
        return;
    }
    LogStuckWaiter(b, obj, op, val, start);
    b.cv.wait(lk, moved);
}

// ---------------------------------------------------------------------------

void FutexWakeLocked(FutexBucket& bucket) {
    bucket.seq.fetch_add(1, std::memory_order_relaxed);
    bucket.cv.notify_all();
}

// Wake waiters on `addr`. The guest has already updated the associated memory
// word before reaching here (futex contract).
void FutexWake(const void* addr) {
    auto& b = BucketFor(addr);
    std::scoped_lock lk{b.m};
    FutexWakeLocked(b);
}

// Current time of the guest-visible clock `clockid`, matching the host mapping
// used by posix_clock_gettime (realtime ids -> CLOCK_REALTIME, everything else
// -> CLOCK_MONOTONIC).
std::chrono::nanoseconds GuestClockNow(u32 clockid) {
    struct timespec t{};
    clock_gettime(IsRealtimeClockId(clockid) ? CLOCK_REALTIME : CLOCK_MONOTONIC, &t);
    return std::chrono::seconds{t.tv_sec} + std::chrono::nanoseconds{t.tv_nsec};
}

// Converts a guest timeout into a steady_clock deadline. `abstime` timeouts are
// absolute on the guest clock `clockid` and get rebased onto steady_clock via
// the remaining duration.
std::chrono::steady_clock::time_point DeadlineFor(const GuestTimespec& ts, bool abstime,
                                                  u32 clockid) {
    const auto dur = std::chrono::seconds{ts.tv_sec} + std::chrono::nanoseconds{ts.tv_nsec};
    if (!abstime) {
        return std::chrono::steady_clock::now() +
               std::chrono::duration_cast<std::chrono::steady_clock::duration>(dur);
    }
    return std::chrono::steady_clock::now() +
           std::chrono::duration_cast<std::chrono::steady_clock::duration>(dur -
                                                                           GuestClockNow(clockid));
}

// Computes the deadline time point (steady_clock) from the `_umtx_op` timeout
// arguments of the WAIT/MUTEX_WAIT/SEM waits. `size_arg` is the 4th argument
// (size of the structure), `ts_ptr` the 5th (pointer to timespec or
// _umtx_time); both may be absent. Matches the kernel's umtx_copyin_umtx_time:
// a plain timespec is always relative, a _umtx_time carries flags + clockid.
// Return value: false => no timeout (wait forever).
bool ParseTimeout(uintptr_t size_arg, const void* ts_ptr,
                  std::chrono::steady_clock::time_point& deadline) {
    if (ts_ptr == nullptr) {
        return false;
    }
    if (size_arg >= sizeof(GuestUmtxTime)) {
        const auto* ut = reinterpret_cast<const GuestUmtxTime*>(ts_ptr);
        deadline = DeadlineFor(ut->timeout, (ut->flags & UMTX_ABSTIME) != 0, ut->clockid);
    } else {
        deadline = DeadlineFor(*reinterpret_cast<const GuestTimespec*>(ts_ptr), false, 0);
    }
    return true;
}

// Core futex wait (UMTX_OP_WAIT / WAIT_UINT[_PRIVATE]): block while the guest
// word is still `val`.
s32 FutexWait(const void* addr, u64 val, bool is64, uintptr_t size_arg, const void* ts_ptr) {
    auto& b = BucketFor(addr);
    std::unique_lock lk{b.m};

    // Re-checking under the bucket lock prevents lost wakeups: a concurrent wake
    // must take the same bucket lock and therefore cannot be lost between the
    // check and going to sleep.
    const auto matches = [&] { return (is64 ? LoadU64(addr) : LoadU32(addr)) == val; };
    if (!matches()) {
        return 0;
    }

    const u64 start = b.seq.load(std::memory_order_relaxed);
    std::chrono::steady_clock::time_point deadline{};
    const bool has_timeout = ParseTimeout(size_arg, ts_ptr, deadline);

    if (!has_timeout) {
        WaitForeverWithDiagnostics(b, lk, start, addr, is64 ? UMTX_OP_WAIT : UMTX_OP_WAIT_UINT,
                                   val);
        return 0;
    }
    if (!b.cv.wait_until(lk, deadline,
                         [&] { return b.seq.load(std::memory_order_relaxed) != start; })) {
        SetPosixErrno(POSIX_ETIMEDOUT);
        return -1;
    }
    return 0;
}

// MUTEX_WAIT (do_lock_umutex with _UMUTEX_WAIT in FreeBSD): block while the
// umutex owner word is owned, return 0 as soon as it is free (UMUTEX_UNOWNED or
// bare UMUTEX_CONTESTED) - the caller then CAS-acquires it in userland. Before
// each sleep the UMUTEX_CONTESTED bit must be set: libthr's unlock fast path
// (CAS owner -> UNOWNED) only issues MUTEX_WAKE/MUTEX_WAKE2 when the owner word
// it replaced carried the contested bit. Without publishing contention here,
// the unlock never wakes and the waiter sleeps forever.
//
// NOTE: `val` is unused by this operation (libthr passes 0). It must NOT be
// treated as an expected futex value - doing so inverts the blocking condition
// (sleeping exactly when the mutex is free) and loses wakeups permanently.
s32 UmtxMutexWait(void* mtx_ptr, uintptr_t size_arg, const void* ts_ptr) {
    auto& b = BucketFor(mtx_ptr);
    std::unique_lock lk{b.m};

    std::chrono::steady_clock::time_point deadline{};
    const bool has_timeout = ParseTimeout(size_arg, ts_ptr, deadline);

    for (;;) {
        u32 owner = LoadU32(mtx_ptr);
        while ((owner & ~UMUTEX_CONTESTED) != 0 && (owner & UMUTEX_CONTESTED) == 0) {
            // Owned and not yet marked contested: publish contention. On CAS
            // failure `owner` is reloaded and the condition re-evaluated.
            if (CasU32(mtx_ptr, owner, owner | UMUTEX_CONTESTED)) {
                owner |= UMUTEX_CONTESTED;
            }
        }
        if ((owner & ~UMUTEX_CONTESTED) == 0) {
            return 0;
        }

        const u64 start = b.seq.load(std::memory_order_relaxed);
        if (!has_timeout) {
            WaitForeverWithDiagnostics(b, lk, start, mtx_ptr, UMTX_OP_MUTEX_WAIT, 0);
        } else if (!b.cv.wait_until(lk, deadline, [&] {
                       return b.seq.load(std::memory_order_relaxed) != start;
                   })) {
            SetPosixErrno(POSIX_ETIMEDOUT);
            return -1;
        }
        // Woken: re-evaluate the owner word (another thread may have acquired
        // the mutex in the meantime).
    }
}

// Unlocks a normal umutex word (for CV_WAIT handoff) and wakes waiters.
void UmtxMutexUnlock(void* mutex_ptr, FutexBucket* locked_bucket = nullptr) {
    if (mutex_ptr == nullptr) {
        return;
    }
    const u32 old = LoadU32(mutex_ptr);
    StoreU32(mutex_ptr, UMUTEX_UNOWNED);
    if (old & UMUTEX_CONTESTED) {
        auto& mutex_bucket = BucketFor(mutex_ptr);
        if (&mutex_bucket == locked_bucket) {
            // CV_WAIT already owns this bucket. Taking the same non-recursive mutex again would
            // deadlock when the condvar and its mutex hash to the same bucket.
            FutexWakeLocked(mutex_bucket);
        } else {
            FutexWake(mutex_ptr);
        }
    }
}

// CV_WAIT: the guest passes the condvar (obj), flags (val), the associated
// mutex (uaddr) and a plain timespec (uaddr2; absolute iff CVWAIT_ABSTIME in
// val, on the clock from c_clockid iff CVWAIT_CLOCKID, else CLOCK_REALTIME).
// FreeBSD semantics: mark c_has_waiters, enqueue the condvar, release the
// mutex, sleep. After returning, libthr re-locks the mutex itself.
s32 UmtxCvWait(void* cvp, u32 wflags, void* mutex_ptr, const void* ts_ptr) {
    auto& b = BucketFor(cvp);
    std::unique_lock lk{b.m};

    const u64 start = b.seq.load(std::memory_order_relaxed);

    // Publish the waiter BEFORE releasing the mutex, exactly like the kernel:
    // libthr's _thr_ucond_signal/_broadcast read c_has_waiters in userland and
    // skip the CV_SIGNAL/CV_BROADCAST syscall entirely while it is 0. Without
    // this store every signal is lost and the waiter sleeps forever.
    StoreU32(cvp, 1);

    // Release the mutex inside the condvar bucket lock: a concurrent CV_SIGNAL
    // takes the same bucket lock, so a signal cannot be lost while we are not
    // yet asleep.
    UmtxMutexUnlock(mutex_ptr, &b);

    std::chrono::steady_clock::time_point deadline{};
    bool has_timeout = false;
    if (ts_ptr != nullptr) {
        const auto ts = *reinterpret_cast<const GuestTimespec*>(ts_ptr);
        const bool abstime = (wflags & CVWAIT_ABSTIME) != 0;
        const u32 clockid = (wflags & CVWAIT_CLOCKID) != 0
                                ? LoadU32(reinterpret_cast<u8*>(cvp) + UcondClockidOffset)
                                : 0 /*CLOCK_REALTIME*/;
        deadline = DeadlineFor(ts, abstime, clockid);
        has_timeout = true;
    }

    if (!has_timeout) {
        WaitForeverWithDiagnostics(b, lk, start, cvp, UMTX_OP_CV_WAIT, wflags);
        return 0;
    }
    if (!b.cv.wait_until(lk, deadline,
                         [&] { return b.seq.load(std::memory_order_relaxed) != start; })) {
        SetPosixErrno(POSIX_ETIMEDOUT);
        return -1;
    }
    return 0;
}

// SEM_WAIT (FreeBSD-9 `_usem`, actually used by the PS4): wait for a token.
// Layout: { u32 _has_waiters; u32 _count; u32 _flags; }.
s32 UmtxSemWait(void* sem_ptr, uintptr_t size_arg, const void* ts_ptr) {
    auto* base = reinterpret_cast<u8*>(sem_ptr);
    void* has_waiters = base + 0;
    void* count = base + 4;

    auto& b = BucketFor(sem_ptr);
    std::unique_lock lk{b.m};

    // Set the waiter bit, then check the token - both under the bucket lock, so
    // a concurrent SEM_WAKE (takes the same lock) cannot be lost.
    StoreU32(has_waiters, 1);
    if (LoadU32(count) != 0) {
        // Token available - the guest decrements it itself in userland.
        return 0;
    }

    const u64 start = b.seq.load(std::memory_order_relaxed);
    std::chrono::steady_clock::time_point deadline{};
    const bool has_timeout = ParseTimeout(size_arg, ts_ptr, deadline);

    if (!has_timeout) {
        WaitForeverWithDiagnostics(b, lk, start, sem_ptr, UMTX_OP_SEM_WAIT, 0);
        return 0;
    }
    if (!b.cv.wait_until(lk, deadline,
                         [&] { return b.seq.load(std::memory_order_relaxed) != start; })) {
        SetPosixErrno(POSIX_ETIMEDOUT);
        return -1;
    }
    return 0;
}

// SEM2_WAIT: wait for a token in the _usem2 count word.
s32 UmtxSem2Wait(void* sem_ptr, uintptr_t size_arg, const void* ts_ptr) {
    auto& b = BucketFor(sem_ptr);
    std::unique_lock lk{b.m};

    const u32 count = LoadU32(sem_ptr);
    if ((count & USEM_COUNT_MASK) != 0) {
        // Token available - the guest decrements it itself in userland.
        return 0;
    }
    // Set the waiter bit atomically. A plain store can overwrite a token posted by another
    // guest thread between the load above and this update, leaving the waiter asleep forever.
    const u32 previous = FetchOrU32(sem_ptr, USEM_HAS_WAITERS);
    if ((previous & USEM_COUNT_MASK) != 0) {
        return 0;
    }

    const u64 start = b.seq.load(std::memory_order_relaxed);
    std::chrono::steady_clock::time_point deadline{};
    const bool has_timeout = ParseTimeout(size_arg, ts_ptr, deadline);

    if (!has_timeout) {
        WaitForeverWithDiagnostics(b, lk, start, sem_ptr, UMTX_OP_SEM2_WAIT, 0);
        return 0;
    }
    if (!b.cv.wait_until(lk, deadline,
                         [&] { return b.seq.load(std::memory_order_relaxed) != start; })) {
        SetPosixErrno(POSIX_ETIMEDOUT);
        return -1;
    }
    return 0;
}

bool IsWaitOp(s32 op) {
    switch (op) {
    case UMTX_OP_WAIT:
    case UMTX_OP_WAIT_UINT:
    case UMTX_OP_WAIT_UINT_PRIVATE:
    case UMTX_OP_MUTEX_WAIT:
    case UMTX_OP_SEM_WAIT:
    case UMTX_OP_SEM2_WAIT:
    case UMTX_OP_CV_WAIT:
        return true;
    default:
        return false;
    }
}

std::atomic<u64> g_unhandled_log_throttle{0};

} // namespace

s32 PS4_SYSV_ABI _umtx_op(void* obj, s32 op, u64 val, void* uaddr, void* uaddr2) {
    const bool is_wait = IsWaitOp(op);
    if (is_wait) {
        Trace(UmtxEvent::WaitEnter, op, obj, val, SampleWord(op, obj), 0);
    }

    s32 result;
    switch (op) {
    case UMTX_OP_WAIT:
        result = FutexWait(obj, val, /*is64=*/true, reinterpret_cast<uintptr_t>(uaddr), uaddr2);
        break;
    case UMTX_OP_WAIT_UINT:
    case UMTX_OP_WAIT_UINT_PRIVATE:
        result = FutexWait(obj, static_cast<u32>(val), /*is64=*/false,
                           reinterpret_cast<uintptr_t>(uaddr), uaddr2);
        break;
    case UMTX_OP_MUTEX_WAIT:
        result = UmtxMutexWait(obj, reinterpret_cast<uintptr_t>(uaddr), uaddr2);
        break;
    case UMTX_OP_SEM_WAIT:
        result = UmtxSemWait(obj, reinterpret_cast<uintptr_t>(uaddr), uaddr2);
        break;
    case UMTX_OP_SEM2_WAIT:
        result = UmtxSem2Wait(obj, reinterpret_cast<uintptr_t>(uaddr), uaddr2);
        break;
    case UMTX_OP_CV_WAIT:
        result = UmtxCvWait(obj, static_cast<u32>(val), uaddr, uaddr2);
        break;

    case UMTX_OP_WAKE:
    case UMTX_OP_WAKE_PRIVATE:
    case UMTX_OP_MUTEX_WAKE:
    case UMTX_OP_MUTEX_WAKE2:
    case UMTX_OP_SEM_WAKE:
    case UMTX_OP_SEM2_WAKE:
    case UMTX_OP_CV_SIGNAL:
    case UMTX_OP_CV_BROADCAST:
        FutexWake(obj);
        Trace(UmtxEvent::Wake, op, obj, val, SampleWord(op, obj), 0);
        return 0;

    case UMTX_OP_NWAKE_PRIVATE: {
        // obj points to an array of `val` pointers; wake each of them.
        auto** addrs = reinterpret_cast<void**>(obj);
        for (u64 i = 0; i < val; ++i) {
            FutexWake(addrs[i]);
            Trace(UmtxEvent::Wake, op, addrs[i], val, LoadU32(addrs[i]), 0);
        }
        return 0;
    }

    default:
        // The remaining operations (priority-mutex lock/unlock, RW locks,
        // set-ceiling, obsolete SEM/MUTEX variants) are not on the hot path for
        // the currently tested titles. Return success (previous STUB behavior),
        // but log in a throttled manner.
        if (g_unhandled_log_throttle.fetch_add(1) % 4096 == 0) {
            LOG_WARNING(Lib_Kernel, "_umtx_op: unhandled operation {} (val={:#x})", op, val);
        }
        return 0;
    }

    Trace(UmtxEvent::WaitExit, op, obj, val, SampleWord(op, obj), result);
    return result;
}

void RegisterUmtx(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("04AjkP0jO9U", "libkernel", 1, "libkernel", _umtx_op);
}

} // namespace Libraries::Kernel
