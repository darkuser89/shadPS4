// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <semaphore>
#include <unordered_map>

#include "core/libraries/kernel/sync/semaphore.h"

#include "common/logging/log.h"
#include "common/slot_vector.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/libs.h"

namespace Libraries::Kernel {

constexpr s32 ORBIS_KERNEL_SEM_VALUE_MAX = 0x7FFFFFFF;

struct PthreadSem {
    explicit PthreadSem(s32 value_) : semaphore{value_}, value{value_} {}

    CountingSemaphore semaphore;
    std::atomic<s32> value;
};

// Guest sem_t storage normally contains a pointer created by posix_sem_init. Some Unity
// runtimes also use libc-internal/static semaphore representations which aren't host pointers.
// Keep ownership and a per-guest-address shadow so those representations never get dereferenced
// by native ARM64 HLE code and concurrent wait/post calls still share the same state.
static std::mutex pthread_sems_mutex;
static std::once_flag shadow_semaphore_warning;
static std::unordered_map<PthreadSem**, std::shared_ptr<PthreadSem>> pthread_sems_by_handle;
static std::unordered_map<PthreadSem*, std::weak_ptr<PthreadSem>> pthread_sems_by_impl;

enum class SemaphoreActivity {
    Create,
    Cancel,
    Delete,
};

struct SemaphoreActivityStats {
    std::atomic<u64> created{};
    std::atomic<u64> canceled{};
    std::atomic<u64> deleted{};
    std::atomic<u64> total{};
};

static SemaphoreActivityStats semaphore_activity;

static void RecordSemaphoreActivity(SemaphoreActivity activity) {
    switch (activity) {
    case SemaphoreActivity::Create:
        semaphore_activity.created.fetch_add(1, std::memory_order_relaxed);
        break;
    case SemaphoreActivity::Cancel:
        semaphore_activity.canceled.fetch_add(1, std::memory_order_relaxed);
        break;
    case SemaphoreActivity::Delete:
        semaphore_activity.deleted.fetch_add(1, std::memory_order_relaxed);
        break;
    }

    // Successful semaphore operations are extremely hot in some Unity titles. Keep detailed
    // per-object diagnostics at trace level and periodically expose an aggregate at info level.
    constexpr u64 ReportInterval = 4096;
    const u64 total = semaphore_activity.total.fetch_add(1, std::memory_order_relaxed) + 1;
    if (total % ReportInterval != 0) {
        return;
    }

    const u64 created = semaphore_activity.created.exchange(0, std::memory_order_relaxed);
    const u64 canceled = semaphore_activity.canceled.exchange(0, std::memory_order_relaxed);
    const u64 deleted = semaphore_activity.deleted.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Lib_Kernel,
             "Semaphore activity (last {} operations): created={}, canceled={}, deleted={}",
             created + canceled + deleted, created, canceled, deleted);
}

static std::shared_ptr<PthreadSem> GetPthreadSem(PthreadSem** sem, bool create_shadow) {
    if (sem == nullptr || *sem == nullptr) {
        return {};
    }

    std::scoped_lock lock{pthread_sems_mutex};
    if (const auto it = pthread_sems_by_handle.find(sem); it != pthread_sems_by_handle.end()) {
        return it->second;
    }

    if (const auto it = pthread_sems_by_impl.find(*sem); it != pthread_sems_by_impl.end()) {
        if (auto implementation = it->second.lock()) {
            pthread_sems_by_handle.emplace(sem, implementation);
            return implementation;
        }
        pthread_sems_by_impl.erase(it);
    }

    if (!create_shadow) {
        return {};
    }

    auto shadow = std::make_shared<PthreadSem>(0);
    pthread_sems_by_handle.emplace(sem, shadow);
    std::call_once(shadow_semaphore_warning, [sem] {
        LOG_WARNING(Lib_Kernel,
                    "Using shadow POSIX semaphores for libc-internal guest representations "
                    "(first handle {}, value {})",
                    fmt::ptr(sem), fmt::ptr(*sem));
    });
    return shadow;
}

class OrbisSem {
public:
    OrbisSem(s32 init_count, s32 max_count, std::string_view name, bool is_fifo)
        : name{name}, token_count{init_count}, max_count{max_count}, init_count{init_count},
          is_fifo{is_fifo} {}
    ~OrbisSem() = default;

    s32 Wait(bool can_block, s32 need_count, u32* timeout) {
        std::unique_lock lk{mutex};
        if (is_deleted) {
            return ORBIS_KERNEL_ERROR_EACCES;
        }
        if (token_count >= need_count) {
            token_count -= need_count;
            return ORBIS_OK;
        }
        if (!can_block) {
            return ORBIS_KERNEL_ERROR_EBUSY;
        }

        if (timeout && *timeout == 0) {
            return ORBIS_KERNEL_ERROR_ETIMEDOUT;
        }

        // Create waiting thread object and add it into the list of waiters.
        WaitingThread waiter{need_count, is_fifo};
        const auto it = AddWaiter(&waiter);

        // Perform the wait.
        const s32 result = waiter.Wait(lk, timeout);
        if (result == ORBIS_KERNEL_ERROR_ETIMEDOUT) {
            wait_list.erase(it);
        }
        if (result != ORBIS_OK) {
            // Diagnostic: a blocking wait that ends in an error is what job/mixer
            // worker loops typically treat as a shutdown signal - make it visible.
            LOG_WARNING(Lib_Kernel,
                        "sceKernelWaitSema: thread '{}' wait on sem '{}' failed: result={:#x} "
                        "(need={}, tokens={}, timed={})",
                        g_curthread->name, name, static_cast<u32>(result), need_count,
                        token_count.load(), timeout != nullptr);
        }
        return result;
    }

    s32 Signal(s32 signal_count) {
        std::scoped_lock lk{mutex};
        if (is_deleted) {
            return ORBIS_KERNEL_ERROR_EACCES;
        }
        if (token_count + signal_count > max_count) {
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
        token_count += signal_count;

        // Wake up threads in order of priority.
        for (auto it = wait_list.begin(); it != wait_list.end();) {
            auto* waiter = *it;
            if (waiter->need_count > token_count) {
                ++it;
                continue;
            }
            it = wait_list.erase(it);
            token_count -= waiter->need_count;
            waiter->was_signaled = true;
            waiter->sem.release();
        }

        return ORBIS_OK;
    }

    s32 Cancel(s32 set_count, s32* num_waiters) {
        std::scoped_lock lk{mutex};
        if (is_deleted) {
            return ORBIS_KERNEL_ERROR_EACCES;
        }
        if (num_waiters) {
            *num_waiters = static_cast<s32>(wait_list.size());
        }
        for (auto* waiter : wait_list) {
            waiter->was_canceled = true;
            waiter->sem.release();
        }
        wait_list.clear();
        token_count = set_count < 0 ? init_count : set_count;
        return ORBIS_OK;
    }

    void Delete() {
        std::scoped_lock lk{mutex};
        is_deleted = true;
        for (auto* waiter : wait_list) {
            waiter->was_deleted = true;
            waiter->sem.release();
        }
        wait_list.clear();
    }

public:
    struct WaitingThread {
        BinarySemaphore sem;
        u32 priority;
        s32 need_count;
        std::string thr_name;
        bool was_signaled{};
        bool was_deleted{};
        bool was_canceled{};

        explicit WaitingThread(s32 need_count, bool is_fifo)
            : sem{0}, priority{0}, need_count{need_count} {
            // Retrieve calling thread priority for sorting into waiting threads list.
            if (!is_fifo) {
                priority = g_curthread->attr.prio;
            }

            thr_name = g_curthread->name;
        }

        [[nodiscard]] s32 GetResult() const {
            if (was_signaled) {
                return ORBIS_OK;
            }
            if (was_deleted) {
                return ORBIS_KERNEL_ERROR_EACCES;
            }
            if (was_canceled) {
                return ORBIS_KERNEL_ERROR_ECANCELED;
            }
            return ORBIS_KERNEL_ERROR_ETIMEDOUT;
        }

        s32 Wait(std::unique_lock<std::mutex>& lk, u32* timeout) {
            lk.unlock();
            if (!timeout) {
                // Wait indefinitely until we are woken up.
                sem.acquire();
                lk.lock();
            } else {
                // Wait until timeout runs out, recording how much remaining time there was.
                const auto start = std::chrono::high_resolution_clock::now();
                sem.try_acquire_for(std::chrono::microseconds(*timeout));
                const auto end = std::chrono::high_resolution_clock::now();
                const auto time =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                lk.lock();
                if (was_signaled) {
                    *timeout -= time;
                } else {
                    *timeout = 0;
                }
            }
            return GetResult();
        }
    };

    using WaitList = std::list<WaitingThread*>;

    WaitList::iterator AddWaiter(WaitingThread* waiter) {
        // Insert at the end of the list for FIFO order.
        if (is_fifo) {
            wait_list.push_back(waiter);
            return --wait_list.end();
        }
        // Find the first with lower priority (greater number) than us and insert right before it.
        auto it = wait_list.begin();
        while (it != wait_list.end() && (*it)->priority <= waiter->priority) {
            ++it;
        }
        return wait_list.insert(it, waiter);
    }

    WaitList wait_list;
    std::string name;
    std::atomic<s32> token_count;
    std::mutex mutex;
    s32 max_count;
    s32 init_count;
    bool is_fifo;
    bool is_deleted{};
};

// SceKernelSema is a 32-bit guest ABI handle.  Keep the SlotId wrapper on the
// implementation side only: passing the C++ struct by value needlessly turns
// these exports into an aggregate ABI case for translated guests.
using OrbisKernelSema = u32;

static Common::SlotVector<std::shared_ptr<OrbisSem>> orbis_sems;
static std::mutex orbis_sems_mutex;

static std::shared_ptr<OrbisSem> GetOrbisSem(Common::SlotId id) {
    std::scoped_lock lock{orbis_sems_mutex};
    if (!orbis_sems.is_allocated(id)) {
        return {};
    }
    return orbis_sems[id];
}

s32 PS4_SYSV_ABI sceKernelCreateSema(OrbisKernelSema* sem, const char* pName, u32 attr,
                                     s32 initCount, s32 maxCount, const void* pOptParam) {
    if (!pName || attr > 2 || initCount < 0 || maxCount <= 0 || initCount > maxCount) {
        LOG_ERROR(Lib_Kernel, "Semaphore creation parameters are invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    {
        std::scoped_lock lock{orbis_sems_mutex};
        *sem = orbis_sems
                   .insert(std::make_shared<OrbisSem>(initCount, maxCount, pName, attr == 1))
                   .index;
    }
    LOG_TRACE(Lib_Kernel, "sceKernelCreateSema: thread '{}' created sem[{}] '{}' init={} max={}",
              g_curthread->name, *sem, pName, initCount, maxCount);
    RecordSemaphoreActivity(SemaphoreActivity::Create);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelWaitSema(OrbisKernelSema sem, s32 needCount, u32* pTimeout) {
    const Common::SlotId id{sem};
    const auto implementation = GetOrbisSem(id);
    if (!implementation) {
        LOG_WARNING(Lib_Kernel, "sceKernelWaitSema: thread '{}' waited on invalid sem[{}] -> ESRCH",
                    g_curthread->name, sem);
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    return implementation->Wait(true, needCount, pTimeout);
}

s32 PS4_SYSV_ABI sceKernelSignalSema(OrbisKernelSema sem, s32 signalCount) {
    const Common::SlotId id{sem};
    const auto implementation = GetOrbisSem(id);
    if (!implementation) {
        LOG_WARNING(Lib_Kernel,
                    "sceKernelSignalSema: thread '{}' signaled invalid sem[{}] -> ESRCH",
                    g_curthread->name, sem);
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    const s32 result = implementation->Signal(signalCount);
    if (result == ORBIS_KERNEL_ERROR_EINVAL) {
        LOG_WARNING(Lib_Kernel,
                    "sceKernelSignalSema: thread '{}' overflowed sem[{}] (count={}) -> EINVAL",
                    g_curthread->name, sem, signalCount);
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelPollSema(OrbisKernelSema sem, s32 needCount) {
    const Common::SlotId id{sem};
    const auto implementation = GetOrbisSem(id);
    if (!implementation) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    return implementation->Wait(false, needCount, nullptr);
}

s32 PS4_SYSV_ABI sceKernelCancelSema(OrbisKernelSema sem, s32 setCount, s32* pNumWaitThreads) {
    const Common::SlotId id{sem};
    const auto implementation = GetOrbisSem(id);
    if (!implementation) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    LOG_TRACE(Lib_Kernel, "sceKernelCancelSema: thread '{}' canceled sem[{}] set_count={}",
              g_curthread->name, sem, setCount);
    RecordSemaphoreActivity(SemaphoreActivity::Cancel);
    return implementation->Cancel(setCount, pNumWaitThreads);
}

s32 PS4_SYSV_ABI sceKernelDeleteSema(OrbisKernelSema sem) {
    const Common::SlotId id{sem};
    std::shared_ptr<OrbisSem> implementation;
    {
        std::scoped_lock lock{orbis_sems_mutex};
        if (!orbis_sems.is_allocated(id)) {
            return ORBIS_KERNEL_ERROR_ESRCH;
        }
        implementation = std::move(orbis_sems[id]);
        orbis_sems.erase(id);
    }
    LOG_TRACE(Lib_Kernel, "sceKernelDeleteSema: thread '{}' deleted sem[{}]", g_curthread->name,
              sem);
    RecordSemaphoreActivity(SemaphoreActivity::Delete);
    implementation->Delete();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_sem_init(PthreadSem** sem, s32 pshared, u32 value) {
    if (value > ORBIS_KERNEL_SEM_VALUE_MAX) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    if (sem != nullptr) {
        auto implementation = std::make_shared<PthreadSem>(static_cast<s32>(value));
        std::scoped_lock lock{pthread_sems_mutex};
        if (const auto old = pthread_sems_by_handle.find(sem);
            old != pthread_sems_by_handle.end()) {
            pthread_sems_by_impl.erase(old->second.get());
            pthread_sems_by_handle.erase(old);
        }
        *sem = implementation.get();
        pthread_sems_by_impl.emplace(implementation.get(), implementation);
        pthread_sems_by_handle.emplace(sem, std::move(implementation));
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_destroy(PthreadSem** sem) {
    auto implementation = GetPthreadSem(sem, false);
    if (!implementation) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    {
        std::scoped_lock lock{pthread_sems_mutex};
        pthread_sems_by_handle.erase(sem);
        pthread_sems_by_impl.erase(implementation.get());
    }
    *sem = nullptr;
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_wait(PthreadSem** sem) {
    auto implementation = GetPthreadSem(sem, true);
    if (!implementation) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    implementation->semaphore.acquire();
    --implementation->value;
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_trywait(PthreadSem** sem) {
    auto implementation = GetPthreadSem(sem, true);
    if (!implementation) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    if (!implementation->semaphore.try_acquire()) {
        *__Error() = POSIX_EAGAIN;
        return -1;
    }
    --implementation->value;
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_timedwait(PthreadSem** sem, const OrbisKernelTimespec* t) {
    auto implementation = GetPthreadSem(sem, true);
    if (!implementation || t == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    if (!implementation->semaphore.try_acquire_until(t->TimePoint())) {
        *__Error() = POSIX_ETIMEDOUT;
        return -1;
    }
    --implementation->value;
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_post(PthreadSem** sem) {
    auto implementation = GetPthreadSem(sem, true);
    if (!implementation) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    // Atomically check for overflow and increment in one step.
    s32 current = implementation->value.load();
    do {
        if (current == ORBIS_KERNEL_SEM_VALUE_MAX) {
            *__Error() = POSIX_EOVERFLOW;
            return -1;
        }
    } while (!implementation->value.compare_exchange_weak(current, current + 1));
    implementation->semaphore.release();
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_getvalue(PthreadSem** sem, s32* sval) {
    auto implementation = GetPthreadSem(sem, true);
    if (!implementation) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    if (sval) {
        *sval = implementation->value;
    }
    return 0;
}

s32 PS4_SYSV_ABI scePthreadSemInit(PthreadSem** sem, s32 flag, u32 value, const char* name) {
    if (flag != 0) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    s32 ret = posix_sem_init(sem, 0, value);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemDestroy(PthreadSem** sem) {
    s32 ret = posix_sem_destroy(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemWait(PthreadSem** sem) {
    s32 ret = posix_sem_wait(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemTrywait(PthreadSem** sem) {
    s32 ret = posix_sem_trywait(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemTimedwait(PthreadSem** sem, u32 usec) {
    OrbisKernelTimespec now{};
    posix_clock_gettime(ORBIS_CLOCK_REALTIME, &now);
    const u64 total_nsec = now.tv_nsec + (usec % 1000000) * 1000ULL;

    OrbisKernelTimespec time{};
    time.tv_sec = now.tv_sec + usec / 1000000 + total_nsec / 1000000000;
    time.tv_nsec = total_nsec % 1000000000;

    s32 ret = posix_sem_timedwait(sem, &time);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemPost(PthreadSem** sem) {
    s32 ret = posix_sem_post(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemGetvalue(PthreadSem** sem, s32* sval) {
    s32 ret = posix_sem_getvalue(sem, sval);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

void RegisterSemaphore(Core::Loader::SymbolsResolver* sym) {
    // Orbis
    LIB_FUNCTION("188x57JYp0g", "libkernel", 1, "libkernel", sceKernelCreateSema);
    LIB_FUNCTION("Zxa0VhQVTsk", "libkernel", 1, "libkernel", sceKernelWaitSema);
    LIB_FUNCTION("4czppHBiriw", "libkernel", 1, "libkernel", sceKernelSignalSema);
    LIB_FUNCTION("12wOHk8ywb0", "libkernel", 1, "libkernel", sceKernelPollSema);
    LIB_FUNCTION("4DM06U2BNEY", "libkernel", 1, "libkernel", sceKernelCancelSema);
    LIB_FUNCTION("R1Jvn8bSCW8", "libkernel", 1, "libkernel", sceKernelDeleteSema);

    // Posix
    LIB_FUNCTION("pDuPEf3m4fI", "libScePosix", 1, "libkernel", posix_sem_init);
    LIB_FUNCTION("cDW233RAwWo", "libScePosix", 1, "libkernel", posix_sem_destroy);
    LIB_FUNCTION("YCV5dGGBcCo", "libScePosix", 1, "libkernel", posix_sem_wait);
    LIB_FUNCTION("WBWzsRifCEA", "libScePosix", 1, "libkernel", posix_sem_trywait);
    LIB_FUNCTION("w5IHyvahg-o", "libScePosix", 1, "libkernel", posix_sem_timedwait);
    LIB_FUNCTION("IKP8typ0QUk", "libScePosix", 1, "libkernel", posix_sem_post);
    LIB_FUNCTION("Bq+LRV-N6Hk", "libScePosix", 1, "libkernel", posix_sem_getvalue);

    // Some SDK/Unity builds import the POSIX semaphore NIDs from the libkernel library
    // namespace instead of libScePosix. Both names refer to the same ABI and implementation.
    LIB_FUNCTION("pDuPEf3m4fI", "libkernel", 1, "libkernel", posix_sem_init);
    LIB_FUNCTION("cDW233RAwWo", "libkernel", 1, "libkernel", posix_sem_destroy);
    LIB_FUNCTION("YCV5dGGBcCo", "libkernel", 1, "libkernel", posix_sem_wait);
    LIB_FUNCTION("WBWzsRifCEA", "libkernel", 1, "libkernel", posix_sem_trywait);
    LIB_FUNCTION("w5IHyvahg-o", "libkernel", 1, "libkernel", posix_sem_timedwait);
    LIB_FUNCTION("IKP8typ0QUk", "libkernel", 1, "libkernel", posix_sem_post);
    LIB_FUNCTION("Bq+LRV-N6Hk", "libkernel", 1, "libkernel", posix_sem_getvalue);

    LIB_FUNCTION("GEnUkDZoUwY", "libkernel", 1, "libkernel", scePthreadSemInit);
    LIB_FUNCTION("Vwc+L05e6oE", "libkernel", 1, "libkernel", scePthreadSemDestroy);
    LIB_FUNCTION("C36iRE0F5sE", "libkernel", 1, "libkernel", scePthreadSemWait);
    LIB_FUNCTION("H2a+IN9TP0E", "libkernel", 1, "libkernel", scePthreadSemTrywait);
    LIB_FUNCTION("fjN6NQHhK8k", "libkernel", 1, "libkernel", scePthreadSemTimedwait);
    LIB_FUNCTION("aishVAiFaYM", "libkernel", 1, "libkernel", scePthreadSemPost);
    LIB_FUNCTION("DjpBvGlaWbQ", "libkernel", 1, "libkernel", scePthreadSemGetvalue);
}

} // namespace Libraries::Kernel
