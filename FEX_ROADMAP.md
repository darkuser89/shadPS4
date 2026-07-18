# Work Plan: shadPS4 native on Apple Silicon (FEXCore instead of Rosetta 2)

Status 2026-07-13: The proof of concept and real-game bring-up have been achieved.
`Cat From Hell` and `Cult of the Lamb` run through module start, Unity/il2cpp
initialization, HLE, VideoOut and active shader compilation in the native ARM64
process. After the AvPlayer callback fix, `Cult of the Lamb` also runs through its
MP4 prerolls up to the visible title screen. The repeat runs ended without the
previous guest-register/memory fault (details in
[INTEGRATION.md](INTEGRATION.md)):

- ✅ FEXCore compiles natively on macOS (arm64), incl. W^X/MAP_JIT
- ✅ x86→ARM64 translation **and execution** (24/24 harness tests: ALU, memory,
  all branches, Call/Ret, SSE2, AVX, HLE thunk, multi-arg marshalling, SMC)
- ✅ HLE boundary guest→host→guest works (thunk + SysV marshalling)
- ✅ shadPS4 integration code compile+link-verified against a real `libFEXCore.a`
- ✅ Two real PS4 games reach the video/shader workload with GC signals in both
  native waits and directly in translated JIT code
- ✅ Darwin ABI fixed: the reserved ARM64 platform register `x18` is removed from
  FEX's dynamic register allocation
- ✅ HLE thunk signal race secured: an already-spilled `CPUState` is not
  overwritten with host scratch registers in the native return/refill window

What follows is stabilization and compatibility expansion: interactive gameplay
over longer sessions, more games, remaining HLE stubs and performance. The older
effort estimates below are kept as historical planning.

---

## Phase A — FEXCore macOS port production-ready (≈ 2–4 W)

The core paths currently run; some stubs are deliberately provisional.

1. ✅ **`SYS_getcpu`** — done: `ProcessorID` emission returns cpu=0/node=0 directly
   on macOS; verified via RDTSCP test.
2. ✅ **`MAP_FIXED_NOREPLACE`** — done: `0` on macOS (no-op hint), FEX's address
   check takes over the "do-not-overwrite" semantics.
3. ✅ **Signal delegation for the game path** — the Darwin handler publishes a
   fixed TLS snapshot and jumps into FEX's pause trampoline; the actual guest
   callback execution happens outside the POSIX handler. Native waits use the
   already-spilled `CPUState`, while signals in normal JIT code spill the fixed
   ARM guest registers. Boehm/Unity GC has exercised both paths repeatedly in real
   runs. Nested/overflowing signal queues and broader PS4 exception coverage remain
   open.
4. ✅ **futex shim (`__ulock`)** — validated: under concurrent JIT compilation the
   4-thread W^X test uses the FEX mutexes (which build on `__ulock`);
   deterministically green over many runs.
5. ✅ **Guard page** — done: skipped on macOS. The cause was the **16 KB pages vs.
   `FEX_PAGE_SIZE=4096`** discrepancy (mprotect EINVAL). Keep an eye on this
   page-size question. — T
6. ✅ **Multi-thread W^X** — validated: 4 host threads compile+execute
   simultaneously, all correct, deterministic (thread_local nesting counter).
7. ✅ **16 KB pages in shadPS4 memory tracking** — four 4 KB guest pages are
   coalesced into one 16 KB host page for `mprotect`; write-protection faults
   report reads and invalidation for the complete host page. Further new FEX
   allocator/`mprotect` paths must still be checked for 16 KB alignment.

### Completed correctness fixes from the first game bring-up

- ✅ **Darwin register ABI:** `x18` is reserved on Apple platforms and removed from
  FEX's RA, PreserveAll and NotPreserved dynamic GPR sets. The former allocation
  could destabilize native runtime/signal paths and corrupt guest memory sizes
  derived from them.
- ✅ **Signal race at the HLE thunk:** `DEF_OP(Thunk)` sets an `InSyscallInfo`
  marker after `SpillStaticRegs()` and only clears it after the complete
  `FillStaticRegs()`. The shadPS4 Darwin signal path interprets the marker as
  "guest state already spilled" and avoids a second spill of host scratch
  registers in this window.
- ✅ **Nested guest callbacks:** `HandleCallback()` temporarily clears an outer
  thunk marker during the real guest JIT and restores it afterward. Signals in the
  callback thereby spill the correct live state.
- ✅ **AvPlayer black screen:** The native FFmpeg wrappers call the game-supplied
  memory and file callbacks via `InvokeGuestOrHost`. Eight previously direct
  x86-64 function-pointer calls are thereby correctly executed through FEX; Cult
  plays back the prerolls and reaches the title screen.
- ✅ **Host/guest pthread lifecycle:** Native AvPlayer workers are distinguished
  when creating PS4 guest threads. Only guest threads run through FEX and execute
  PS4 thread/TLS destructors. AvPlayer also publishes all stop requests before the
  first join; no worker joins itself.
- ✅ **ABI bring-up:** Guest stack bounds for Boehm GC, scalar semaphore handles,
  Darwin `sigaction` flags as well as AudioOut/Audio3D by-value unions are adapted
  for the FEX HLE path.

## Phase B — Build consolidation ✅ ACHIEVED

**shadPS4 is built as a native arm64-macOS binary with FEXCore and runs a real
game all the way into the active video/shader workload:**
- `ninja -C build-arm64 shadps4` → 385/385 objects, 0 errors, final link ok.
- Binary: 68 MB, **arm64 Mach-O** (no Rosetta), **3599 FEXCore symbols** linked in,
  `FexBackend::{Initialize,RunMainThread,SetupThread}` present.
- **Starts natively** and loads `Cat From Hell` including Unity/il2cpp, VideoOut and
  KosmicKrisp pipelines. The former `Module::Start` blocker is fixed.
- Host binary, FEXCore and native plugins remain ARM64; `eboot.bin`, PRX and
  PS4 system modules remain as x86-64 guest code.

### Details (Phase B)

7. ✅ **Build shadPS4 natively as arm64** — all required deps are available for
   arm64-macOS:
   - macOS Vulkan is **fully vendored**: shadPS4 builds its own `vulkan-loader`
     + `mesa-kosmickrisp` (Mesa-based Vulkan driver) — no system MoltenVK needed.
   - `ext-ffmpeg-core` builds ffmpeg **from source** (no x86-only binary blocker).
   - **Blocker 1 (solved):** `git submodule --init` without `--recursive` leaves
     nested submodules empty (`mesa-kosmickrisp/externals/mesa`) → configure
     aborts. Fix: `git submodule update --init --recursive --depth 1`.
   - **arm64 configure SUCCESSFUL** (`-DCMAKE_OSX_ARCHITECTURES=arm64
     -DENABLE_QT_GUI=OFF -DENABLE_UPDATER=OFF -DENABLE_FEX_CPU=ON -DFEXCORE_DIR=...`;
     Abseil loaded, fonts embedded, `build.ninja` generated).
   - **Complete externals stack builds for arm64** (verified, 1930 objects):
     mesa-kosmickrisp Vulkan driver, OpenAL, protobuf, LibreSSL, host shaders. — W
8. ✅ **Integrate FEXCore into shadPS4** via `-DENABLE_FEX_CPU=ON -DFEXCORE_DIR=...`
   (include paths now active). **VERIFIED:** `fex_backend.cpp.o` (27 KB) AND
   `linker.cpp.o` (with `RunMainEntry`→FexBackend) compile cleanly in the real
   shadPS4 arm64 build (arm64, shadPS4 + FEXCore headers). By now all shadPS4
   sources and the final binary link against the FEXCore archives.
9. Disconnect `cpu_patches.cpp` on arm64 (made redundant by FEX). — T

## Phase C — Roll out HLE boundary generically (game path reached, coverage growing)

10. ✅ **Guest memory and threads in the shadPS4 address space:** FEX sees guest
    code, PS4 stacks and shadPS4's memory manager; call/return shadow stacks are
    managed separately per guest thread.
11. ✅ **Signature-driven HLE thunks:** `HleThunk<Fn>` marshals integer, XMM and
    relevant by-value arguments. Further rare signature classes are added during
    title bring-up.
12. ✅ **aerolib `Relocate` hook:** ARM64 import slots point to `0F 3F` stubs
    (`EmitThunkStub`) instead of native ARM function addresses.
13. Fill `Ps4SyscallHandler` with real PS4 kernel HLE (raw `syscall`s from
    libkernel → shadPS4 syscall dispatcher). — W
14. ✅ **Guest entries and callbacks:** `RunMainEntry`, `Module::Start`, guest
    pthreads and HLE callbacks run through FEX; stack layout, TLS/`fs` base and
    HLT return sentinel are validated in the Unity/il2cpp game path.

## Phase D — Bring-up with real games (≈ months, iterative)

15. ✅ Two commercial games through the full startup pipeline up to VideoOut and
    shader workload; longer interactive sessions remain a manual regression test.
16. ✅ Base paths for XMM/float marshalling, indirect calls and x87; rare HLE
    signatures remain title-driven extension work.
17. Performance: scalar software TSO is enabled on native macOS; evaluate a
    supported hardware-TSO activation path separately, plus block linking/code-cache
    persistence and JIT warmup.
18. Graphics remain untouched (Vulkan/GCN→SPIR-V) — only the CPU side changes.
19. Systematically cover more titles, nested signals/callbacks and not-yet-
    implemented HLE stubs.

### Current title smoke tests

| Title | Test runs | Reached state | FEX result |
|---|---:|---|---|
| `Cat From Hell` (`CUSA54365`, Update 1.01) | 4+ | Update PRX, Unity workers, shader/compute workload | No PLT/semaphore fault |
| `Cult of the Lamb` (`CUSA32184`) | multiple | AvPlayer prerolls, title menu and manually confirmed gameplay, async streaming | No guest fault/size overflow |

For `Cult of the Lamb`, `sceAvPlayerSetLogCallback`, pad device information,
network/HTTP callbacks as well as some libc/kernel helpers, among others, remain
stubbed. `sceAvPlayerJumpToTime` has no longer been a no-op since the test on
2026-07-13: the player stops the running decoders in a controlled manner, flushes
packet/frame queues, performs an FFmpeg seek in the stream time base, restarts
demuxer and decoders and restores the previous pause state. With v1.00 the concrete
call `JumpToTime(0)` was confirmed in the log; afterward the main menu was visible
instead of a permanently black window. The subsequent manual test up to a playable
in-game scene was also confirmed.

`Cat From Hell` Update 1.01 contains in `Il2CppUserAssemblies.prx` six pre-bound
16-byte PLT slots (`E9 rel32; NOP`) for the `scePthreadSem*` family. Their stored
jump targets belong to the original PS4 process layout and lay outside all mappings
in the FEX address space. The ARM64 linker recognizes the PLT table via its ordered
`JUMP_SLOT` relocations and restores only the pre-bound slots as `FF 25 rel32` over
the already correctly resolved GOT entries. This made the jump to `0x6f970f1230`
disappear.

Afterward an independent POSIX semaphore bug became visible: for numerous internal
semaphores Unity uses the guest value `0xffff736d` without creating it as a host
object via shadPS4's `posix_sem_init`. `sem_wait` and `sem_post` had until then
dereferenced this value as an ARM64 pointer. The implementation now maintains
ownership registries and creates a shared shadow state for foreign representations,
indexed by the guest address of the `sem_t`. A Cat 1.01 run subsequently ran
through 67 such semaphores and 55 graphics/compute pipelines without a Critical.
The following Cult 1.21 regression run ran through FMOD, AvPlayer seek and 22
pipelines likewise without a Critical.

A later observed, intermittent black screen was not another decoder or FEXCore
version error. During the seek the game main thread waited in
`sceAvPlayerStart -> AvPlayerSource::Stop -> pthread_join`, while native AvPlayer
workers erroneously executed the global PS4 thread destructor through FEX on exit.
Additionally the workers contained self-joins. Since 2026-07-14 host and guest
threads are explicitly classified, guest destructors are executed only for guest
threads, and demuxer/decoders are stopped and woken together and only then joined.
The renderer test subsequently captured, in a cold run, the intro, the Devolver
logo and the complete "X To Start" title screen.

A following normal repeat run uncovered a second, independent race: all FEX threads
waited in `LookupCache::{FindBlock,AddBlockMapping}`. The macOS port had replaced
Linux's `FUTEX_WAKE_BITSET` for the preferred writer with an untargeted
`__ulock_wake(..., 1)`. If a reader was woken by it, the reader went back to sleep
because of the waiting writer, while the writer was never woken. The Darwin path
now wakes all threads for this mixed waiter set; their atomic state check lets
exactly the entitled writer proceed. Three consecutive cold starts of Cult v1.00
subsequently ran through seek, all worker exits, the new `StatePlay` and the later
FMOD asset streaming.

A longer gameplay run then showed a separate Darwin GPU watcher error: a texture
watcher spanned `0x709cc000 - 0x70c00000`, although the range was not fully
GPU-mapped. The 16 KB adjustment had until then also rounded fully unmapped gaps up
to host pages; `mprotect` therefore aborted with `ENOMEM`. The page manager now
clips protection requests against the current rasterizer mappings, aligns only
their intersections to 16 KB host pages and processes each host page at most once.
The ARM64 build and another start up to past AvPlayer and FMOD streaming
subsequently ran without an `mprotect` assertion; the concrete later gameplay path
remains a manual long-duration test.

---

## Shortest regression path

1. Build FEXCore for ARM64:
   `cmake --build ../FEX_MacOs/build --target FEXCore FEXCore_Base JemallocLibs`.
2. Build shadPS4: `cmake --build build-arm64 -j6` and confirm with `file` that
   `build-arm64/shadps4` is an ARM64 Mach-O.
3. Start a real title repeatedly and wait at least for module start, HLE,
   VideoOut, shader compilation and one GC cycle.
4. Check the log for negative/overflowed reservation sizes, guest memory faults and
   lost signal states. Merely reaching the CLI is not a sufficient end-to-end test.

---

## What is NOT needed

- No custom x86 decoder/JIT — FEXCore provides it completely.
- No re-development of the graphics, audio or HLE implementations; at the
  guest/host boundary, however, ARM64 ABI adapters and signature fixes are needed.
- No waiting for an official FEX macOS port — it is already done in this tree
  (see `FEX_MacOs/`, ~20 changed files).
- No ARM64 rebuilds of PS4 guest modules: `eboot.bin`, PRX and guest sysmodules
  remain x86-64. Only the macOS host, FEXCore and host-side plugins/`.dylib`s are
  ARM64.

---

## FEX update audit 2026-07-13

An update to the official stable tag `FEX-2607` was fully tested through to the
arm64 link in a separate worktree:

- Upstream tag/commit: `FEX-2607`, `1cc4b93e` (2026-07-02)
- Port worktree: `../FEX_MacOs_FEX-2607`
- previous game-tested state: `../FEX_MacOs`, `73a32ff22`
- FEXCore archives and `build-arm64/shadps4` build and link successfully
- necessary API adaptations: `LookupExecutableFileSection` receives a thread
  pointer in 2607; `GuestSignal_SIGILL` lies directly in `CpuStateFrame::Pointers`;
  additionally `External/unordered_dense/include` is required

The state is **not yet activated**. `Cult.of.the.Lamb_CUSA32184_v1.00` reproduces,
directly after the first aerolib HLE fallback, a regression: `CommonFexStub` sees
the correct return `0x70027291b3`, but afterward FEX compiles `GuestRIP=0` and the
frontend decoder receives a null instruction stream. LLDB confirms the path
`CompileBlock -> GenerateIR -> DecodeInstructionsAtEntry`; multiblock disabling and
a null-target guard do not resolve the actual RA64/thunk return. The experimental
port is kept for further root-cause work, and the active shadPS4 build is again
built against the proven FEX tree.

Next update step: isolate the HLE thunk as its own FEX regression test, bisect the
changes between `73a32ff22` and `FEX-2607` in register allocation, `FillStaticRegs`
and `ExitFunction(Return)` down to the first faulty commit, and only then repeat
the real game test.

---

## AVX/AVX2/SSE4a audit and narrow 2607 backport (2026-07-14)

The Apple M3 host features were already set correctly: `SupportsAVX` activates
FEX's guest AVX frontend and simultaneously implies AVX2 in the emulated CPUID;
`SupportsSSE4a` publishes the Jaguar SSE4a extension. Apple Silicon, however, has
no SVE. Therefore `SupportsSVE128/256` stays off and FEX correctly decomposes
256-bit AVX/AVX2 operations into two 128-bit NEON halves. Forcing SVE would not
optimize but rather generate unsupported ARM instructions.

From FEX 2607 only the related upstream changes `12fcf96e9` (`ContextClear`) and
`d6d4f84c3` (AVX128 `vzeroupper/vzeroall`) were backported onto the active,
game-tested FEX tree. The dispatcher discards the cached upper YMM halves and then
clears the contiguous `CPUState::avx_high` region. On the M3 `SupportsCLZERO` stays
off; the JIT therefore uses paired Q-register stores with 32 bytes per instruction
instead of 8 or 16 individual 128-bit stores. The problematic full switch to
FEX 2607 remains deactivated.

Verification:

- `FEXCore`, `FEXCore_Base`, `JemallocLibs` and `build-arm64/shadps4` build and
  link successfully.
- The macOS FEX harness passes 30/30 tests. Newly covered are guest CPUID for
  AVX, AVX2 and SSE4a, a real SSE4a `EXTRQ` execution as well as the semantics of
  `VZEROUPPER` and `VZEROALL` over the new context-clear path.
- `Cat From Hell` 1.01 reached the update PRX, Unity workers, GC signals and
  running asset streaming without a `Critical` or guest memory fault.
- `Cult of the Lamb` 1.21 ran through AvPlayer `StatePause`, normal worker exits,
  `JumpToTime(0)`, renewed `StatePlay` and several graphics pipelines without a
  `Critical` or access violation.

---

## shadPS4 upstream update 2026-07-14

The local ARM64/FEX state was updated to shadPS4 upstream `6d37f61`. Before the
merge the complete local state was secured as commit `42fa55f` and tag
`pre-shadps4-update-20260714`; the merge commit is `3ff0324`. Included are, among
others, the new shader cache invalidation, OpenAL fixes, more precise GCN thread
masks, Vulkan Robustness2 as a mandatory feature and sampler garbage collection.

The conflict-free git merge contained a semantic ABI collision: the new OpenAL
changes read the last parameter of `sceAudio3dAudioOutOpen`, while the FEX port
deliberately passes it as a raw `u32` instead of a C++ union. Both Audio3D backends
now reconstruct the four bytes locally. This keeps the guest/FEX ABI stable and the
new upstream format logic works.

Verification:

- complete ARM64 build successful; binary is Mach-O ARM64;
- macOS FEX harness still passes 30/30 tests;
- Cat From Hell 1.01 loads FEX, normalizes six PLT slots, starts the update PRX
  and compiles graphics pipelines without a `Critical` or guest fault;
- Cult of the Lamb 1.32 loads the Unity/FMOD modules and reaches AvPlayer
  `StateReady` without a hard FEX fault. Repeated `_is_signal_return` stub calls
  still produce a very large amount of log output and remain to be clarified
  separately.

---

## Tomb Raider: NP guest callback on ARM64 (2026-07-14)

Tomb Raider CUSA00109 stalled directly after initialization with high CPU load.
LLDB showed the PC `0x7000617d20` and an apparent ARM instruction `str z21`; the
bytes actually belong to the x86-64 guest code. The cause was four direct calls to
registered NP state, toolkit and reachability callbacks in
`DispatchPendingNpStateCallbacks`. These calls bypass FEX on ARM64.

All four paths now use `Core::CPU::InvokeGuestOrHost`. Afterward the title ran
through NP/party/IME polling, asset streaming and 581 first-time-compiled graphics
pipelines without a `Critical`, FEX guest fault or access violation. The previously
blurry start frame switched to the regular TR loading screen; the manual test
subsequently reached the main menu.

---

## Tomb Raider CUSA00109: Darwin VM ceiling and real guest fault (2026-07-15)

Two independent failures were separated using the complete 21-GB v1.00 dump:

1. Reserving the full PS4 user range through `0x5fffffffffff` on Apple Silicon
   forced later Metal/libmalloc allocations above Darwin xzone's segment-table
   limit. The resulting `BUG IN LIBMALLOC` occurred in
   `kk_CmdBeginRendering`, before it could describe a guest error. ARM64 macOS now
   caps `USER_MAX` at `0xfffffffffff` (44-bit / 16 TiB). Tomb Raider passes the old
   immediate crash and performs sustained shader and asset work.
2. A later intermittent access violation was previously labelled as native
   IOAccelerator code because `IsAddressInCodeBuffer` returned false and only the
   stale saved RIP `0x700060b140` was available. FEX can execute a shared code-buffer
   generation not reported as the thread's current generation. The backend now also
   asks `RestoreRIPFromHostPC`, whose inline-block range check reconstructs the real
   RIP safely. The actual fault is guest RIP `0x70007b6f59`, instruction
   `mov cl,[r8+0x500]`, with live `R8=0`. It is Tomb Raider's resource-command
   parser reading a null entry from the table at `[r12+0x150]`, not Metal.

Rejected workarounds:

- An earlier global scalar software-TSO test avoided the null-table crash but caused
  a severe GPU page-watch/readback fault storm and left the title black during module
  startup. Scalar software TSO is now intentionally retested after invalid-VMA
  hardening; vector and REP MOVS/STOS atomics remain disabled.
- `FEX_MAXINST=1 FEX_MULTIBLOCK=0` does not repair the resource state and later
  reaches `CopySparseMemory: invalid address 0x704f0000`.

The dump itself is complete: PlayGo files are under `sce_sys`, all base, language
and PATCH tiger archives exist and are opened successfully. The remaining blocker
is therefore a guest resource-publication/memory-mapping issue. Keep the normal FEX
block settings and scalar software-TSO configuration while investigating that producer;
do not hide the null dereference or reclassify it as a native driver fault.

---

## Dynamic Apple M1-M5 host features (2026-07-15)

The embedded FEX setup previously used one hard-coded "M1+" feature set with a
64-byte cache line. That was safe enough for the first port, but it left useful
instructions disabled on newer CPUs and could not adapt to future M-series chips.
FEX cannot use its normal privileged ARM identification-register path from macOS
EL0, so shadPS4 now builds `HostFeatures` from `hw.optional.arm.FEAT_*` sysctls and
reads `hw.cachelinesize` at runtime. Missing sysctl keys are false; no M1-M5 model
whitelist is needed.

On the M3 Pro test host macOS reports a 128-byte cache line plus LSE, LRCPC,
LRCPC2, AFP, and RPRES. Those paths are now available to FEX code generation;
CSSC and MOPS remain disabled because this machine/OS does not report them. An M1
therefore keeps only its reported baseline, while M4/M5 machines automatically
gain any newer instructions that their installed macOS exposes. SVE, ECV, WFXT,
and cache-line zeroing remain conservative rather than being inferred from the
marketing generation.

This can reduce the instruction count of translated addressing, floating-point,
reciprocal, integer, and memory operations. It does not eliminate cold shader and
Vulkan pipeline compilation, which is usually the larger first-start cost. FEX's
disk code cache remains work in progress and was deliberately not enabled. The
normal multiblock settings are unchanged. The former hardware-TSO declaration
was later found to be invalid on native macOS and has been replaced by scalar
software TSO; see the finding below.

Verification: the complete ARM64 build succeeds, and CUSA43774 reached asset,
audio, and shader work on the M3 Pro without SIGILL, a FEX critical, or a guest
fault during the smoke-test interval.

---

## Correct FEX software-TSO setup on macOS (2026-07-15)

`Context::SetHardwareTSOSupport(true)` does not enable Apple's optional TSO mode.
It tells FEX that the host process already has hardware TSO and consequently turns
off FEX's atomic software-TSO lowering. shadPS4 never enabled a hardware memory
model for its native macOS process, so the old call silently selected the weaker
ARM memory model and could expose guest publication races.

The embedded backend now sets `CONFIG_TSOENABLED=1`, leaves the FEX-default vector
and REP MOVS/STOS atomic modes disabled, and explicitly reports no hardware TSO.
This gives scalar x86 guest loads/stores FEX's software ordering. It intentionally
retests the earlier page-watch/readback failure after invalid sparse guest ranges
were hardened; startup logs identify the selected memory model for regression reports.

---

## KosmicKrisp shader-cache startup optimization (2026-07-15)

True draw-time asynchronous graphics compilation is not safe with the current
KosmicKrisp backend. Its Mesa source explicitly builds whole Metal render
pipelines and currently supports neither Graphics Pipeline Library nor shader
objects. Returning to the game before that pipeline exists would require waiting
on first use or skipping the draw, which merely trades stutter for missing/black
graphics. Compute pipelines cannot be skipped at all without changing results.

The safe repeat-load path is now enabled instead. KosmicKrisp creates Mesa's
implicit in-memory pipeline cache and a physical-device disk cache keyed by the
driver build. The serialized cache keeps the expensive SPIR-V -> NIR -> MSL work
between launches; Metal still creates the device-specific final pipeline.
shadPS4 also creates its explicit Vulkan pipeline cache before warm-up, so cached
pipelines no longer preload with a null cache handle.

The ARM64/KosmicKrisp build succeeds and CUSA43774 reached normal graphics,
asset, and audio work. The cache test produced 2.8 MiB of Mesa entries. A small
26-pipeline warm-up measured 1.17 s with an empty Mesa cache and 1.05 s on the
immediate cached run; titles with hundreds of repeated pipelines should benefit
more. No draw skipping or experimental async mode was enabled.

---

## Invalid sparse GPU address hardening (2026-07-15)

Tomb Raider later reached `CopySparseMemory` with guest address `0x70500000`.
The value fits inside the GPU's 40-bit address width but lies below the ARM64 PS4
user VMA range, so the old BindTextures/BindBuffers ceiling check accepted it and
the strict sparse-copy assertion terminated the GPU command processor.

The apparent relation `0x70500000 << 8 == 0x7050000000` is diagnostic evidence,
not a safe repair: Sea Islands image descriptors already return
`base_address << 8`, while buffer descriptors use a byte address. Blindly shifting
the decoded value would corrupt valid resources. Normal T#/V# binding now requires
both a valid 40-bit GPU range and a real guest VMA range. `CopySparseMemory` also
validates the complete range while holding the memory lock and supplies zeroes for
an invalid or concurrently unmapped range instead of asserting. The rejected raw
and encoded texture addresses are logged for tracing the descriptor producer.

The ARM64/KosmicKrisp build completes successfully. Runtime validation must check
whether the title now continues with an empty resource and whether the logged
encoded base identifies a stale descriptor or its producer.

---

## Regression update and preserved checkpoints (2026-07-16)

The currently preserved source checkpoint is `poc1.46` (`b58b35f`). The recent
checkpoints and their confirmed purpose are:

| Checkpoint | Confirmed result |
|---|---|
| `poc1.4` | Working scalar software TSO plus invalid-GPU-address hardening baseline |
| `poc1.41` | AVPlayer buffer lifetime stabilized; reproduced green preroll flicker disappeared |
| `poc1.42` | ARM64 fixed guest mappings restored after the upstream merge |
| `poc1.43` | Present/WaitRegMem loading-stall corrections retained |
| `poc1.46` | AJM flag marshalling, FEX `snprintf` varargs and stale shader-permutation tolerance retained |
| `poc1.48` | CUSA05033 in-game: umtx FreeBSD semantics, vector/memcpy TSO + strict split locks, fiber-owned sceFiber contexts, silent-guest-death diagnostics |
| `poc1.49` | Canonical PS4 address translation + verified carveout load/store emulation; CUSA32809 boots through pool phase |

### Confirmed negative results

- **KosmicKrisp/Bink:** updating the driver and adding a local green-video
  workaround did not solve New Super Lucky's Tale reliably. Both commits were
  reverted. The same green Bink output is reported on Windows, so it is not an
  Apple driver-only regression.
- **Dark Souls Remastered:** the oversized/corrupt character geometry also occurs
  in official Windows reports. Renderer experiments did not correct it and were
  reverted; it remains a cross-platform shadPS4 graphics issue.
- **CUSA02607, Re: Hollow Fragment v1.05:** a valid trophy key successfully
  extracted 60 trophy files, but the title still registered VideoOut without a
  first flip. Two PageManager/readback bypass experiments left the screen black
  and were reverted.
- **CUSA05033, Hollow Realization:** experimental ARM64 Fiber and libc-mspace work
  compiled, but the title did not establish a validated boot path at the time; the
  mspace implementation was removed from the build. (Superseded 2026-07-18: after
  the `_umtx_op` semantics fix, full vector/memcpy TSO ordering and the
  fiber-owned sceFiber context fix, the title reaches gameplay on native
  ARM64/FEX — see the update below and INTEGRATION.md.)
- **CUSA10168, Fatal Bullet v1.14:** the experimental `dlmalloc`-backed mspace
  implementation caused a repeatable NpToolkit2 null virtual call at guest RIP
  `0x7012ca4984`, fault address `0x10`. Removing only the mspace implementation
  restored startup. This is the current regression proof that it must not be
  promoted.

### Confirmed data/configuration results

- CUSA43774's black/magenta menu assets were caused by a bad extraction. A correct
  v1.04 dump rendered normally without a shader workaround.
- Trophy keys remain external application configuration in `keys.json`; no key
  value is embedded in the executable or repository.
- Missing PS4 system fonts are still reported at startup and can cause text or
  stability problems. They are independent of the CUSA02607 trophy extraction.

Before the next checkpoint, regression-test at least Cult of the Lamb gameplay,
Fatal Bullet through NpToolkit2 initialization, one AVPlayer preroll, and a title
that uses AJM. A successful compile or module load is not sufficient.

---

## CUSA05033 in-game milestone and sync-correctness layer (2026-07-18)

SWORD ART ONLINE: Hollow Realization reaches gameplay on native ARM64/FEX
(user-confirmed in-game). Three general correctness fixes landed on top of
`poc1.47`, each independently valuable beyond this title:

1. **`_umtx_op` FreeBSD-9 semantics** (`umtx.cpp`): `MUTEX_WAIT` now blocks
   while the umutex is owned and publishes `UMUTEX_CONTESTED` before sleeping;
   `CV_WAIT` sets `c_has_waiters` before releasing the mutex and parses its
   timeout as a plain `timespec` with `CVWAIT_ABSTIME`/`CVWAIT_CLOCKID`;
   absolute `_umtx_time` deadlines honour `_clockid`. Previously the op was
   modelled as a Linux futex, which lost wakeups permanently.
2. **Full TSO ordering for vector and rep-string stores plus strict
   in-process split locks** (`fex_backend.cpp`): `VECTORTSOENABLED`,
   `MEMCPYSETTSOENABLED` and `STRICTINPROCESSSPLITLOCKS` are now enabled;
   `FEX_VECTORTSO`/`FEX_MEMCPYSETTSO` env overrides exist for performance
   experiments. This may also be relevant to other stale-read corruption
   (candidate: the parked Dark Souls Remastered geometry issue).
3. **Fiber-owned sceFiber contexts** (`fiber_fex.cpp`): suspended contexts
   moved from a thread_local map to a global per-fiber store (consuming
   resume, XMM0-15 + MXCSR saved). This fixed the deterministic silent death
   of one RenderMixing worker per run and is the ARM64 fiber implementation's
   first real runtime validation.

New permanent diagnostics: `RunGuestThread` refuses to treat a non-sentinel
dispatcher exit as a clean thread return (dumps GPRs/stack/code/call-ret
chain), fiber resumes validate their target is executable, guest thread exits
and semaphore lifecycle/errors are logged, umtx waiters stuck >30s dump their
bucket history. A silent guest-thread death can no longer masquerade as a
normal exit.

Regression-test note for the next checkpoint: include one fiber-using title
(CUSA05033 voice-bank loading exercises cross-thread fiber resume) alongside
the existing Cult of the Lamb / Fatal Bullet / AVPlayer / AJM set, and verify
the vector-TSO cost is acceptable on GPU-light titles before considering any
per-title toggle.
