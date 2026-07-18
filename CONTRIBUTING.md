<!--
SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Style guidelines

## General Rules

* Line width is typically 100 characters. Please do not use 80-characters.
* Don't ever introduce new external dependencies into Core
* Don't use any platform specific code in Core
* Use namespaces often
* Avoid the use of C-style casts and instead prefer C++-style static_cast and reinterpret_cast. Try to avoid using dynamic_cast. Never use const_cast except for when dealing with external const-incorrect APIs.

## Naming Rules

* Functions: `PascalCase`
* Variables: `lower_case_underscored. Prefix with g_ if global.`
* Classes: `PascalCase`
* Files and Directories: `lower_case_underscored`
* Namespaces: `PascalCase`, `_` may also be used for clarity (e.g. `ARM_InitCore`)

# Indentation/Whitespace Style

Follow the indentation/whitespace style shown below. Do not use tabs, use 4-spaces instead.

# Comments

* For regular comments, use C++ style (//) comments, even for multi-line ones.
* For doc-comments (Doxygen comments), use /// if it's a single line, else use the /** */ style featured in the example. Start the text on the second line, not the first containing /**.
* For items that are both defined and declared in two separate files, put the doc-comment only next to the associated declaration. (In a header file, usually.) Otherwise, put it next to the implementation. Never duplicate doc-comments in both places.

```c++
// Includes should be sorted lexicographically
// STD includes first
#include <array>
#include <map>
#include <memory>

// then, library includes
#include <nihstro/shared_binary.h>

// finally, shadps4 includes
#include "common/math_util.h"
#include "common/vector_math.h"

// each major module is separated
#include "video_core/pica.h"
#include "video_core/video_core.h"

namespace Example {

// Namespace contents are not indented

// Declare globals at the top (better yet, don't use globals at all!)
int g_foo{}; // {} can be used to initialize types as 0, false, or nullptr
char* g_some_pointer{}; // Pointer * and reference & stick to the type name, and make sure to initialize as nullptr!

/// A colorful enum.
enum class SomeEnum {
    Red,   ///< The color of fire.
    Green, ///< The color of grass.
    Blue,  ///< Not actually the color of water.
};

/**
 * Very important struct that does a lot of stuff.
 * Note that the asterisks are indented by one space to align to the first line.
 */
struct Position {
    // Always initialize member variables!
    int x{};
    int y{};
};

// Use "typename" rather than "class" here
template <typename T>
void FooBar() {
    const std::string some_string{"prefer uniform initialization"};

    const std::array<int, 4> some_array{
        5,
        25,
        7,
        42,
    };

    if (note == the_space_after_the_if) {
        CallAFunction();
    } else {
        // Use a space after the // when commenting
    }

    // Place a single space after the for loop semicolons, prefer pre-increment
    for (int i = 0; i != 25; ++i) {
        // This is how we write loops
    }

    DoStuff(this, function, call, takes, up, multiple,
            lines, like, this);

    if (this || condition_takes_up_multiple &&
        lines && like && this || everything ||
        alright || then) {

        // Leave a blank space before the if block body if the condition was continued across
        // several lines.
    }

    // No indentation for case labels
    switch (var) {
    case 1: {
        const int case_var{var + 3};
        DoSomething(case_var);
        break;
    }
    case 3:
        DoSomething(var);
        return;
    default:
        // Yes, even break for the last case
        break;
    }
}

} // namespace Example
```

# Apple Silicon / FEXCore Rules

The native Apple Silicon backend deliberately mixes two architectures in one process. Keep this
boundary intact:

* shadPS4, FEXCore, Vulkan/audio libraries, native plugins, and every loaded host `.dylib` must be
  ARM64 Mach-O.
* PS4 `eboot.bin`, guest PRX modules, and dumped PS4 system modules remain x86-64 guest code. Do not
  rebuild or load these as ARM64 host plugins.
* Verify suspicious artifacts with `file` or `lipo -archs`. A native ARM64 library must never be
  exposed as guest code, and an x86-64 guest module must never be passed to `dlopen`.

Darwin reserves ARM register `x18` as a platform register. FEX JIT register-allocation tables must
not allocate it, use it as scratch storage, or assume that a native call will preserve a JIT value
in it. Changes to ARM64 register lists must keep the `__APPLE__` exclusion in
`FEXCore/Source/Interface/Core/ArchHelpers/Arm64Emitter.cpp`.

Translated x86 registers are live in FEX's fixed ARM registers only while ordinary JIT code is
running. HLE thunks spill them to `CPUState` before calling native code and refill them before
returning to guest code. An asynchronous signal can arrive after the native call has returned but
before `FillStaticRegs()` is complete. During this interval:

* `CPUState` is authoritative; do not spill caller-saved host scratch registers over it.
* Keep the thunk's `InSyscallInfo` marker set until the complete static-register refill finishes.
* Treat that marker as a boolean "state already spilled" value in the shadPS4 Darwin signal path;
  it is not a general full-register spill mask.
* Temporarily clear and then restore an outer marker when a native HLE call re-enters real guest
  code through `HandleCallback()`.
* Do not defer Boehm/Unity suspension signals across a blocking native wait. POSIX signal handlers
  must only publish fixed TLS state and redirect execution; allocation, logging, locks, and direct
  guest callbacks belong outside the handler.

Every function pointer supplied by PS4 guest code must be treated as guest code even when a native
host subsystem invokes it later. Route allocator, deallocator, file-I/O, event, and similar
callbacks through `Core::CPU::InvokeGuestOrHost`; do not call such pointers directly from ARM64.
`AvPlayer` is an important regression case because FFmpeg calls native wrapper functions which must
dispatch the game's original memory and file replacement callbacks back through FEX.

For changes to the FEX backend, at minimum rebuild FEXCore and shadPS4, then run a real-game smoke
test that exercises HLE calls, worker threads, and GC signals from both native waits and translated
JIT code. On Apple Silicon also test 16-KB page protection transitions. A clean startup alone is not
enough; the game must reach active video/shader work without oversized memory reservations or guest
register faults. For titles with preroll movies, verify decoded video frames and a visible menu, not
only `sceVideoOutOpen` or shader compilation in the log.

Do not advance the embedded FEX baseline based on a successful library build alone. The attempted
upgrade from local baseline `73a32ff22` to upstream tag `FEX-2607` (`1cc4b93e`) built and linked on
macOS, but regressed the pointer-in-hash HLE-thunk return: `Cult of the Lamb` reached the first
unresolved `vprintf`, logged a valid guest return address (`0x70027291b3`), and FEX then attempted to
compile guest RIP zero. Keep upgrades in a separate worktree, preserve the last game-tested build,
and promote only after the HLE return, AvPlayer preroll, title/menu, and input-to-gameplay checks all
pass. The investigation worktree is `../FEX_MacOs_FEX-2607`; it is not the active baseline.

On Apple Silicon, `HostFeatures::SupportsAVX` means that FEX may expose and emulate guest AVX; it
does not claim that the ARM host has native 256-bit vectors. FEX derives the guest AVX2 CPUID bit
from this flag. Keep `SupportsAVX` and `SupportsSSE4a` enabled for the PS4 Jaguar guest, but never
force `SupportsSVE128` or `SupportsSVE256` on Apple M-series CPUs. Without SVE, FEX must use its
AVX-128 split path backed by NEON. Changes to that path need explicit guest CPUID tests plus real
AVX, AVX2/SSE4a instruction tests; a successful host compile alone is insufficient. The active
baseline includes the narrow upstream `ContextClear`/`VZEROUPPER`/`VZEROALL` backport from FEX 2607,
which clears the saved upper YMM halves with paired 128-bit stores on Apple instead of individual
register stores. Do not enable `dc zva` for this path unless the host feature probe explicitly marks
cache-line zeroing safe.

Do not identify M1, M2, M3, M4, or later Apple CPUs by product/model strings. macOS publishes the
usable EL0 instruction set through `hw.optional.arm.FEAT_*`; query those sysctls and treat a missing
key as unsupported. This keeps one binary conservative on older machines while allowing newer
machines to use LRCPC2, AFP, RPRES, CSSC, MOPS, and later features when the OS explicitly exposes
them. Keep guest ISA flags such as AVX/SSE4a separate from these host ARM flags. Do not enable SVE,
ECV, WFXT, cache-line zeroing, or FEX's work-in-progress code cache merely from a CPU generation
name; each requires an independently verified OS capability and an end-to-end game test.

KosmicKrisp currently compiles complete Metal render pipelines and does not implement Vulkan
Graphics Pipeline Library or shader objects. Do not call a draw-skipping mode "async shaders": it
can turn required UI and first-use effects black. Keep the driver's implicit memory pipeline cache
and physical-device disk cache alive until device destruction, and create shadPS4's explicit Vulkan
pipeline cache before `PipelineCache::WarmUp()`. This preserves translated MSL between launches
without changing rendering order; the final Metal pipeline creation remains synchronous.

`sceAvPlayerJumpToTime` is a required preroll regression path. Do not return success without moving
the media timeline: stop and join the active demux/decoder threads, clear their packet/frame queues,
seek the shared FFmpeg format context in a selected stream time base, recreate the decoders and guest
buffers, and preserve whether playback was paused. `Cult of the Lamb` v1.00 pauses its second player
and calls `sceAvPlayerJumpToTime(0)`; a no-op stub leaves the presentation black even though shaders
continue compiling. The regression passes only when the restart is visible in the log and the title
menu is actually rendered. For release-level validation, continue through `Play` into a controllable
gameplay scene; the 2026-07-13 Cult v1.00 test confirmed that complete path.

Native host workers and PS4 guest pthreads share the libkernel pthread implementation on ARM64, but
they must not share exit semantics. Classify the start routine when the pthread is created. Only a
guest thread may enter FEX or run the process-wide PS4 thread destructor and guest TLS destructors;
an FFmpeg, AvPlayer, audio, or other native worker must execute and terminate entirely as host code.
Workers must also never join themselves. For a cooperating worker group, publish every stop token and
wake every condition variable before joining the first thread. Validate lifecycle changes with several
cold launches: one successful preroll does not expose an intermittent stop/join deadlock.

Darwin `__ulock` has no equivalent of Linux futex bitset wakeups. Never replace a targeted bitset wake
with a single unfiltered `__ulock_wake`: in FEXCore's write-priority mutex it can wake a reader while
the required writer remains asleep, permanently stranding the shared lookup-cache lock. When mixed
reader/writer waiters share one Darwin ulock word, the targeted-writer fallback must wake all waiters;
each waiter then rechecks the atomic state. Exercise this with several concurrent cold game starts,
not only a standalone or single-thread FEX test.

GPU page watchers use 4-KB PS4 pages, but Apple Silicon `mprotect` operates on 16-KB host pages. Fold
the permissions of all four tracker pages into the most restrictive host-page permission. Watcher
requests are allowed to span holes in the GPU mapping; never pass those holes to `mprotect`, because
Darwin returns `ENOMEM` when any host page in the requested range is unmapped. Intersect the request
with the rasterizer's current mapped intervals, align only those intersections to host pages, and
deduplicate host pages selected by adjacent intervals.

Do not assume that the value stored in a guest `sem_t` is a host pointer. Unity and PS4 libc may use
internal/static semaphore representations such as `0xffff736d` without calling the HLE
`posix_sem_init`. Native ARM64 semaphore HLE must validate pointers against objects it owns and use a
shadow semaphore keyed by the guest `sem_t` address for foreign representations. Do not overwrite
the guest value: wait, post, timed-wait, try-wait, and get-value must share the shadow state while a
`shared_ptr` keeps it alive across concurrent destroy/wait activity. Regression-test this with Unity
main, asset-GC, job-worker, and preload threads rather than testing only an HLE-created semaphore.

When merging upstream shadPS4 into the ARM64/FEX branch, audit changed HLE signatures even if Git
reports a conflict-free merge. The ARM64 bridge intentionally passes some small aggregate guest
arguments as raw integers so FEX does not apply the native AArch64 aggregate ABI. Upstream commit
`16b708d` began reading the final `sceAudio3dAudioOutOpen` format argument; keep the exported raw
`u32` signature and reconstruct `OrbisAudioOutParamExtendedInformation` locally before using it.
The 2026-07-14 upstream integration point is `6d37f61`; the complete pre-merge recovery tag is
`pre-shadps4-update-20260714`.

NP state, toolkit-state, and reachability handlers are PS4 guest callbacks. Dispatch them through
`Core::CPU::InvokeGuestOrHost`, just like AvPlayer and IME callbacks; never invoke the stored
function pointer directly from native ARM64. Tomb Raider registers guest address `0x7000617d20` as
an NP state callback. A direct C++ call makes Apple Silicon execute the x86-64 bytes as AArch64 and
can be misdiagnosed as a GPU page-watcher storm because the generic access-violation chain sees the
resulting fault first.

On Apple Silicon, do not reserve the generic 48-bit shadPS4 user range through
`0x5fffffffffff`. It can push native Metal/libmalloc allocations above Darwin xzone's segment-table
limit and produce a misleading crash inside `kk_CmdBeginRendering`. The tested ARM64 macOS ceiling
is `0xfffffffffff` (44-bit / 16 TiB), which is still much larger than PS4 title mappings. A VM-layout
change must be tested with real Metal command-buffer creation, not only address-space initialization.

Do not assume `FEXCore::Context::IsAddressInCodeBuffer(thread, host_pc)` recognizes every valid
shared JIT generation. Before classifying an ARM64 access violation as native code, call FEX's
`RestoreRIPFromHostPC`; it validates the current inline-block range and can recover the guest RIP
when the per-thread generation check is stale. Tomb Raider's apparent IOAccelerator crash thereby
resolved to guest `mov cl,[r8+0x500]` at `0x70007b6f59` with `R8=0`. Log the live fixed-register
mapping, guest bytes, and reconstructed RIP before changing Vulkan or Metal code.

Do not call `SetHardwareTSOSupport(true)` on native macOS unless hardware TSO was actually enabled
for the process. That API is a promise to FEX, not an activation request, and it disables FEX's
atomic software-TSO lowering. The embedded backend uses scalar software TSO; vector and REP
MOVS/STOS atomics stay disabled. An earlier scalar-TSO test caused GPU page-watch/readback fault
storms, so this mode must be retested after the invalid-VMA hardening and must not be expanded to
more atomic modes without another game regression run. Compare bounded
`FEX_MAXINST`/`FEX_MULTIBLOCK` runs separately. Also validate content before renderer workarounds: the
CUSA43774 black/pink UI case on 2026-07-15 disappeared with a correct v1.04 extraction and required
no shader, texture or blend fix.

Do not treat a value below the GPU's 40-bit ceiling as proof that it is a usable guest address.
T#/V# binding must also validate the address against the guest VMA map before creating cache
resources. Keep `CopySparseMemory` defensive under the VMA lock because an otherwise valid resource
can be unmapped between descriptor binding and upload. Invalid ranges are zero-filled and logged;
do not guess a missing descriptor shift from the address alone. Sea Islands T# decoding already
applies its 256-byte `<< 8` unit conversion, whereas V# base addresses are byte addresses.

## Regression lessons from `poc1.41` through `poc1.46`

AVPlayer output buffers are shared with the renderer after decode. Do not immediately clear or
reuse a buffer after publishing a frame: the GPU may still be reading its Y and UV planes. Keep the
frame storage alive until the consumer has released it. The `poc1.41` lifetime correction removed
the reproduced alternating green/valid preroll frames. Validate both video and subtitles because a
correct subtitle overlay does not prove that the underlying YUV frame is valid.

Do not fix a green Bink presentation by changing the bundled KosmicKrisp revision without first
reproducing it on another Vulkan host. New Super Lucky's Tale shows the same all-green Bink video in
the official Windows compatibility report. Both the speculative KosmicKrisp update and its follow-up
video workaround were reverted before `poc1.46`; neither belongs to the active fix set.

PS4 libc mspaces cannot be replaced with a host `dlmalloc` build merely because the basic
`create/malloc/free` surface matches. A prototype registered the mspace entry points and placed
allocator metadata inside the guest-provided range. It regressed SWORD ART ONLINE: FATAL BULLET
v1.14 during `libSceNpToolkit2` initialization: guest code executed `call [rax+0x10]` with a null
`RAX` and faulted at address `0x10`. Removing the prototype restored the title. Any future mspace
implementation must first verify the PS4 flags, structure ownership, locking, failure behavior,
statistics calls, and NpToolkit2 allocator callbacks in an isolated build.

Do not turn off page-manager read synchronization as a generic black-screen workaround. Two
experiments for SWORD ART ONLINE Re: Hollow Fragment v1.05 either skipped `ReadMemory` on protected
writes or converted protected reads into invalidation when readbacks were disabled. Neither produced
a first flip, and both were reverted. Page-fault frequency alone is not proof that the readback is
unnecessary.

Trophy release keys are user configuration. Store them only in the application `keys.json`, never
in source, logs, commits, test fixtures, or documentation. Successful extraction is a separate test
result: CUSA02607 decrypted 60 files for its trophy set, but still remained black before the first
present. Do not report a trophy extraction as proof that title startup was fixed.
