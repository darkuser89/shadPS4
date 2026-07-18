# FEXCore integration in shadPS4 (native ARM64 instead of Rosetta 2)

> **STATUS 2026-07-14:** shadPS4 runs as a native ARM64 macOS process with
> FEXCore linked in. `Cat From Hell` passes through `Module::Start`, Unity/il2cpp,
> HLE, VideoOut (1920×1080), splash-hide and active KosmicKrisp shader
> compilation. Three repeat runs ended without the previously observed
> negative memory reservation or the guest fault at `0x70009e9ed6`; at least
> one run processed asynchronous GC signals directly out of translated JIT code.
> `Cult of the Lamb` (`CUSA32184`) likewise reaches VideoOut, splash-hide,
> shader compilation and asynchronous streaming in several native runs
> without a FEX guest fault or overflowed reservation size. The subsequently
> reported black screen is fixed: the AvPlayer prerolls are decoded and
> a frame dump confirms the visible title screen with "X To Start".

The goal remains to run PS4 x86-64 guest code on Apple Silicon via FEXCore, without
starting the entire emulator under Rosetta 2. The sections from chapter 1 onward
also contain the historical derivation and earlier intermediate states; in case of
contradictions, the current state in chapter 0 is authoritative.

---

## 0. Authoritative current integration state

### 0.1 Architecture boundary

| Component | Architecture | Role |
|---|---|---|
| shadPS4 process | `arm64` | Native macOS host |
| FEXCore and static FEX libraries | `arm64` | x86-64→ARM64 JIT in the host |
| Host-side plugins and `.dylib` libraries | `arm64` | Loaded directly by macOS |
| `eboot.bin`, PRX and PS4 sysmodules | `x86-64` | Guest code executed by FEX |

Rosetta translates a process built entirely as x86-64. The native
FEX integration separates host and guest: everything that macOS loads and executes
directly is ARM64; PS4 code stays x86-64. A guest module must therefore not be
built as an ARM64 library because of the ARM64 host. Conversely, a native ARM64
`.dylib` must never be treated as a guest PRX or guest code.

This was additionally verified on the second test title against the SELF content: `Cult of the
Lamb` contains an x86-64 `eboot.bin` and 21 x86-64 PRX files. This also includes
the files under `Media/Plugins`; "plugin" there denotes a PS4 guest module and
not a native macOS `.dylib`. None of these guest files is ARM64.

### 0.2 Reserved Darwin register `x18`

Darwin reserves AArch64 register `x18` as a platform register. FEX had nevertheless
carried it as a dynamic RA register in the x86-64 JIT. Generated code can,
however, be interrupted by macOS signals and calls native HLE/runtime
functions directly; a JIT guest value in `x18` therefore violates the Darwin ABI.

`FEXCore/Source/Interface/Core/ArchHelpers/Arm64Emitter.cpp` now uses, under
`__APPLE__`, RA, `PreserveAll_Dynamic` and `NotPreserved_Dynamic` sets without
`r18`. The register stays under host control throughout the entire JIT entry.
New ARM64 register lists must preserve this exclusion rule.

The earlier allocation is the most likely explanation for the
nondeterministic data corruption in the Unity allocation path. The observed
end fault was only a consequence:

```text
corrupted/underflowed allocation size
→ sceKernelReserveVirtualRange receives e.g. 0xffffffff9d540000
→ guest allocator returns null
→ local output pointer/R9 becomes null
→ mov word [R9+0xc], 0
→ write access to 0xc at guest RIP 0x70009e9ed6
```

The fault instruction itself did not spontaneously lose `R9`: `R9` was previously loaded from the
null result of the allocator and was not modified again up to the fault.

### 0.3 Signal race between HLE thunk and register refill

`DEF_OP(Thunk)` spills the fixed x86 guest registers into
`CpuStateFrame::State` before the native HLE call. Immediately after the native `blr`,
the host PC is back in the JIT code buffer even though `PopDynamicRegs()` and `FillStaticRegs()`
have not yet fully restored the fixed ARM
guest registers. A signal that blindly spills the SRA state at this
PC would overwrite the correct `CPUState` with host scratch
registers.

The current contract is therefore:

1. After `SpillStaticRegs()`, `DEF_OP(Thunk)` sets `InSyscallInfo` to `0xffff`.
2. The marker stays set across the native call, `PopDynamicRegs()` and the entire
   `FillStaticRegs()` process.
3. `FexBackend::QueueGuestSignal()` interprets any set marker as
   "guest state already spilled" and selects the pause trampoline without a renewed
   SRA spill.
4. Only after the complete refill does the JIT clear the marker.
5. If a native HLE call reenters real guest JIT code via `HandleCallback()`,
   the outer marker is cleared for the duration of the callback and restored afterward.

`0xffff` here is, in the shadPS4 Darwin path, a boolean state marker and
not a general complete SRA mask. In particular, the value must not be used as
proof that arbitrary ARM register masks are fully encoded.
Boehm/Unity suspend signals must also not be deferred across a blocking
native wait.

### 0.4 Further necessary bring-up corrections

- The POSIX handler writes only a fixed TLS snapshot and redirects to the FEX
  pause trampoline; guest callbacks run outside the signal handler in an
  isolated temporary FEX thread/stack.
- Guest stack bounds are published for the Boehm GC.
- Darwin `sigaction` flags are translated into PS4 semantics.
- Semaphore HLE handles are marshalled as scalar values.
- AudioOut/Audio3D by-value unions and GPU extended userdata follow the
  actual guest ABI instead of the native ARM64 C++ ABI.
- Four 4-KB guest pages are combined into one 16-KB Apple host page for page
  protection and write-fault processing. A write fault reports read and
  invalidation for the entire affected host page.
- MAP_JIT W^X toggling remains thread-local and covers emission, relocation and
  live code patches.

### 0.5 Resolved AvPlayer black screen in `Cult of the Lamb`

Cult starts in a Unity splash scene and uses the three local MP4
prerolls under `Media/StreamingAssets/MMVideoPlayer/Videos`. Before the correction
the visible path ended after `sceAvPlayerSetLogCallback`; neither the MP4 file
nor a decoder was opened and the window stayed black.

`AvPlayer::StubInitData()` deliberately replaces the memory and file
callbacks supplied by the guest with native wrappers so that FFmpeg can
call them on host threads. These wrappers, however, called the original function pointers
directly. That is possible under Rosetta because host and guest are x86-64; in the
native ARM64 process they are x86-64 guest addresses and not callable ARM64
functions.

`src/core/libraries/avplayer/avplayer_impl.cpp` therefore now routes all eight
original callbacks through `Core::CPU::InvokeGuestOrHost`: four memory operations
(`allocate`, `deallocate` and their texture variants) as well as `open`, `close`,
`read_offset` and `size`. Native helpers remain direct host calls, while
guest addresses are executed via FEX with the PS4 x86-64 ABI.

The validation opened `Devolver_Animated_Splash.mp4`, detected video and
audio streams, started the demuxer and both decoders and continued the GNM EOP flips.
A temporary present dump first showed the expected black
start frame, then the Devolver video and finally the fully visible
`Cult of the Lamb` title screen. The dump hook was removed after the test.
The call `sceAvPlayerJumpToTime(0)`, initially still implemented as a no-op, and
a host/guest pthread exit race found later are described in chapters 7 and 8;
both by now belong to the corrected regression path.

### 0.6 Reproduction and regression test

From the shadPS4 checkout:

```sh
cmake --build ../FEX_MacOs/build \
  --target FEXCore FEXCore_Base JemallocLibs -j6
cmake --build build-arm64 -j6
file build-arm64/shadps4
./build-arm64/shadps4 -g /path/to/game/eboot.bin
```

| Title | Runs | State reached | Result of the FEX path |
|---|---:|---|---|
| `Cat From Hell` (`CUSA54365`) | 3 | VideoOut, splash-hide, shader workload | No negative reserve value, no guest fault |
| `Cult of the Lamb` (`CUSA32184`) | multiple | MP4 prerolls and visible title screen | No negative reserve value, no guest fault |

The longest Cat run lasted over a minute and processed hundreds of Unity GC
signals from native waits as well as several signals directly in the JIT. The Cult runs
repeatedly processed Unity suspend signals and reached the title path.
Remaining visible in Cult are, among others, stubs for AvPlayer logging, pad
device information, network/HTTP callbacks, `_is_signal_return`,
`__cxa_guard_acquire` and `sceLibcMspaceCreate`. These findings belong to HLE
compatibility; they are not an observed FEX CPU crash.

This is a regression validation of this error class, not a general
compatibility guarantee. The raw SDL game window is not visible as an app bundle
to the automated computer-use interface; interactive gameplay,
input and longer sessions therefore remain manual tests.

## 1. Why Rosetta is needed at all today

shadPS4 has **no CPU emulator/JIT**. It is compiled as an **x86_64 binary**
and executes the PS4's x86-64 game code *directly* — emulator and
guest code lie as x86-64 in the same address space. On Apple Silicon,
Rosetta 2 translates the entire process. That is by design, not an oversight.

Rosetta forces three concrete dependencies that shadPS4 has to work around today:

| x86 feature                    | Rosetta 2            | Consequence in shadPS4 today                   | FEXCore  |
|--------------------------------|----------------------|------------------------------------------------|----------|
| **AVX / AVX2**                 | only from macOS 15   | hard macOS minimum version; otherwise crash    | ✅ native |
| **SSE4a** (`EXTRQ`/`INSERTQ`)  | ❌ not supported     | runtime patcher rewrites instructions          | ✅ native |
| **`fs` segment / TCB**         | special handling     | platform-specific TLS paths                    | ✅ native |

→ With FEXCore **all three** fall away: FEX decodes AVX2, SSE4a and
segment accesses itself. This also removes the macOS-15 requirement.

Relevant locations:
- SSE4a patches: `src/core/cpu_patches.cpp:549` (`EXTRQ`/`INSERTQ`/`MOVNTSS`/`MOVNTSD`)
- FS segment patches (non-Apple): `src/core/cpu_patches.cpp:555`
- TLS/TCB base per platform: `src/core/tls.cpp` (`SetTcbBase`/`GetTcbBase`)

On ARM64, `cpu_patches.cpp` becomes **completely superfluous** — FEX takes over the ISA.

---

## 2. The favorable starting situation in shadPS4

The codebase already knows ARM64 as a build target — the seams are present:

- **Arch detection:** `CMakeLists.txt:57` → `set(ARCHITECTURE "arm64")`
- **Rosetta case detected:** `CMakeLists.txt:72`
  (`APPLE AND ARCHITECTURE x86_64 AND HOST arm64`)
- **Guest entry already stubbed:** `src/core/linker.cpp:58`
  → `#else UNREACHABLE_MSG("RunMainEntry unimplemented for current architecture.")`

So the ARM branch already exists as an empty shell — exactly where FEXCore docks in.

---

## 3. Historical bring-up planning

This section documents the initial analysis and the order in which
the current state was reached. It is not a current open list; for that,
chapter 0 and [FEX_ROADMAP.md](FEX_ROADMAP.md) apply.

### Construction site 0 — porting FEXCore to macOS (the actual first mountain face)

FEXCore is used **only as a CPU core**. **Not** needed: Linux RootFS,
FEX syscall emulation, thunk libs — shadPS4's own HLE does that.

> ⚠️ **Historical starting point:** The `FEX_MacOs` fork was initially not a
> real port — without `__APPLE__`/`MAP_JIT` code. The following subsections
> capture the greenfield porting that was necessary back then.

To do:
- Replace `MAP_FIXED_NOREPLACE` (Linux-only) — among others in
  `FEXCore/Source/Utils/Allocator/64BitAllocator.cpp`
- **W^X JIT:** `MAP_JIT` + `pthread_jit_write_protect_np()` around every codegen block
- Signal glue: Linux `ucontext` → macOS `__darwin_ucontext`
- **Software TSO on native macOS** — `SetHardwareTSOSupport(true)` only declares
  that TSO was already enabled externally; it does not activate Apple's optional
  mode. The embedded backend therefore keeps FEX scalar software TSO enabled.

#### Empirical build test (already performed)

FEXCore was configured and compiled on Apple Silicon (Apple clang 21,
cmake 4.3, ninja). Result:

**Configure blockers (worked around):**
- `/proc/cpuinfo` scripts (`aarch64_fit_native.py`, `NeedDisabledSVE.py`)
  → skip with `-DTUNE_CPU=none`
- Catch2 `catch_discover_tests` → with `-DBUILD_TESTING=FALSE`
- Git submodules were missing → `git submodule update --init`
- Configure then runs **cleanly through** (arm64 → `_M_ARM_64=1`, no x86 FATAL)

**Compile error classes (48 files affected, but concentrated):**

| Error | × | Origin (header) | Kind |
|--------|---|-------------------|-----|
| `SYS_rt_sigprocmask` undeclared | 81 | **`SignalScopeGuards.h`** (1 header!) | Linux signal syscalls |
| `'futex.h'` missing | 7 | `Threads.cpp` among others | macOS has no futex → `__ulock`/pthread |
| `'syscall.h'` missing | 4 | `FEXHeaderUtils/Syscalls.h` | Linux syscall path |
| `'linux/limits.h'`, `'sys/prctl.h'`, `'linux/magic.h'` | 4 | `PrctlUtils.h`, `Filesystem.h` | Linux headers |
| `'malloc.h'` missing | — | `AllocatorHooks.h` | **✅ already fixed** (`<malloc/malloc.h>`) |
| `sincos`→`__sincos`, `pthread_setname_np` signature | 2 | individually | trivial |

**Core statement:** The errors sit in a **small set of OS abstraction headers**
(above all `SignalScopeGuards.h`, `FEXHeaderUtils/Syscalls.h`, `Threads.cpp`).
The actual translation logic — `Interface/Core/JIT/*`, `X86Tables/*`,
`OpcodeDispatcher/*` (AVX/Crypto/X87/Vector) — has **no macOS problem of its own**;
it fails only **transitively** through these headers. If one fixes the ~5 glue headers,
the majority of the 48 errors cascade away.

→ Construction site 0 is thus **porting the OS glue layer**, not the JIT core.
The hard remaining chunks: futex→`__ulock`, signal delegation
(`rt_sigprocmask`), and after that `MAP_JIT`/W^X (which only comes at runtime, not at
compile time).

**Reproduce:**
```sh
cd FEX_MacOs && git submodule update --init --depth 1
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=FALSE -DTUNE_CPU=none \
  -DENABLE_JEMALLOC=FALSE -DENABLE_JEMALLOC_GLIBC_ALLOC=FALSE -DENABLE_LTO=FALSE \
  -DBUILD_FEXCONFIG=FALSE -DBUILD_THUNKS=FALSE \
  -DENABLE_OFFLINE_TELEMETRY=FALSE -DENABLE_CCACHE=FALSE
ninja -C build -k 0 FEXCore
```

#### ✅ Result: FEXCore compiles natively on Apple Silicon

After **15 changed files**, FEXCore builds as an arm64 static lib:
`build/FEXCore/Source/libFEXCore.a` — **57 object files, all arm64**.
The 48 → 15 → 4 → 1 → 0 errors fell cascading over a small set of
glue fixes. Changes performed (by class):

- **Header replacements:** `<malloc.h>`→`<malloc/malloc.h>`,
  `<syscall.h>`→`<sys/syscall.h>`, `<linux/limits.h>`→`<limits.h>`,
  `<linux/prctl.h>`/`<sys/prctl.h>`/`<linux/magic.h>`/`<sys/vfs.h>` guarded
- **Signal mask:** `SYS_rt_sigprocmask` → `pthread_sigmask` (Darwin `sigset_t`
  is uint32_t, same bit layout) — in `SignalScopeGuards.h` + `Threads.cpp`
- **futex → `__ulock`:** new shim `FEXCore/include/FEXCore/Utils/AppleFutex.h`,
  wired into `InterruptableConditionVariable.h` + `WritePriorityMutex.h`
  (bitset variants collapse to plain wait/wake — safe, only more spurious wakes)
- **Thread syscalls:** `gettid`→`pthread_threadid_np`, `renameat2`→`renameat`,
  `getcpu`/`tgkill`/`statx`/`pidfd_open` → fallbacks (`Syscalls.h`)
- **Miscellaneous:** `sincos`→`__sincos`, `sendfile`→`fcopyfile`,
  `pthread_setname_np`(2-arg→1-arg), `prctl`/`MADV_HUGEPAGE`/MDWE guarded

#### W^X / MAP_JIT — framework implemented

FEX maps JIT memory permanently RWX and writes into it freely; macOS forbids
that and enforces per-thread W^X. Implemented:

- **`MAP_JIT`** is set for executable allocations (`AllocatorHooks.h`,
  `VirtualAlloc(..., Execute=true)` → `MAP_JIT`)
- **Per-thread toggle** `pthread_jit_write_protect_np()` as RAII guard
  `FEXCore::Allocator::JITWriteRegion` in new header
  `FEXCore/include/FEXCore/Utils/AppleJIT.h`
- **All JIT write paths bracketed:**
  - `Arm64JITCore::CompileCode` (block emission, `JIT.cpp`)
  - `Dispatcher::EmitDispatcher` (dispatch-loop emission)
  - 4 backpatch locations (block link/delink, live code patches in `JIT.cpp`)
- ICache flush already ran portably via `__builtin___clear_cache`
- Verified: `_pthread_jit_write_protect_np` is a symbol reference in
  `libFEXCore.a` (compiled, not optimized away)

The only two executable buffers (CodeBuffer via `VirtualAlloc(...,true)`
in `CPUBackend.cpp`, dispatcher buffer) are thus covered.

#### ✅✅✅ Runtime milestone: x86→ARM64 code is EXECUTED on macOS

A minimal embedding harness (`FEX_MacOs/Harness/`, built with
`-DBUILD_FEX_HARNESS=ON`) creates a FEXCore context, translates **and executes**
an x86-64 block. Guest `mov eax,20; add eax,22; hlt` → `ExecuteThread`
runs, `hlt` terminates cleanly (`EnableExitOnHLT`), result **RAX == 42**,
deterministic across multiple runs. This validates the **entire** porting
end-to-end: decoder → IR → ARM64 JIT emission into MAP_JIT buffer → W^X toggle →
execution of the generated ARM64 code → correct result. `MAP_JIT` +
`pthread_jit_write_protect_np` work at runtime.

Additional execution prerequisites (beyond translation):
- `CTX->EnableExitOnHLT()` + a `hlt` (0xF4) in the guest code as a clean exit
- `SignalDelegation->RegisterTLSState(Thread)` before `ExecuteThread`
  (for which `RegisterTLSState`/`UninstallTLSState` in `DummyHandlers.h` were made public)
- mapped guest stack (RSP), result from
  `Thread->CurrentFrame->State.gregs[X86State::REG_RAX]`
- **CS.L = 1 (long mode) in the CS segment descriptor** — critical! FEX derives the
  64-bit mode from `CS.L`, NOT from `CONFIG_IS64BIT_MODE`. Without L=1, FEX decodes
  as 32-bit and **truncates the guest RIP to 32 bit** (`InlineEntrypointOffset`
  with `GetGPROpSize()==i32Bit`, ALUOps.cpp:49 / JIT.cpp:735). That breaks *every*
  inter-block branch as soon as guest code lies >4GB — and macOS always places anonymous mmaps
  >4GB (the lower 4GB are reserved; MAP_FIXED below fails).
  Setup: `GetSegmentFromIndex(State, cs_idx)` → `SetGDTBase/Limit`, `CS->L=1`,
  `CS->D=0`, `State.cs_cached = CalculateGDTBase(*CS)`.

#### ISA coverage validated (24/24 tests green, deterministic)

The test battery (`build_and_test.sh` → `fex_harness`) executes and checks RAX:

| Group | Tests |
|--------|-------|
| **ALU/Integer** | mov+add, shl, imul, **div**, movzx, cmov(nz), setcc |
| **Memory/Stack** | `[rsp-4]` store+load, push/pop, **rep movsb (memcpy)** |
| **Control flow** | jmp, jnz, backward loop, **indirect call (`call rax`)**, **call/ret** |
| **Vector/FP** | SSE2 paddd, **AVX VEX-128 vpaddd**, **SSE4.1 pmulld**, **x87 fld1/faddp/fistp** |
| **System** | **RDTSCP** (getcpu fix → 0) |
| **Emulator core** | **SMC** (overwrite + `InvalidateGuestCodeRange` 42→99), **multi-thread W^X** (4 threads) |
| **HLE boundary** | **Thunk** (guest→native ARM), **SysV multi-arg marshalling** |

This confirms ALU, memory load/store, unconditional/conditional jumps, loops
(multi-block + block linking), **`call`/`ret` (indirect control flow)**,
SSE2 vector + GPR↔XMM and **VEX-encoded AVX** (a PS4 requirement) on macOS.

**HLE thunk boundary validated (guest x86 → native ARM64 C++ code → back):**
The path for the 217 `PS4_SYSV_ABI` functions. For this FEX has a thunk opcode
**`0F 3F`** (ex-VIA ALTINST, block-ending) followed by 32 bytes of SHA256; RDI points
to an args/return blob. At JIT compile time FEX calls
`ThunkHandler::LookupThunk(sha256)` → native `ThunkedFunction*` and bakes in the
pointer; the guest reaches the host function (args pointer in x0), it
writes back into the blob, the guest reads the result. Proven in the harness:
guest sets a=20/b=22, calls the thunk, native `HleAddThunk` writes ret=42,
guest reads `eax=42`.

**Realistic multi-arg HLE call (also validated):** Real PS4 functions
take multiple SysV args (RDI/RSI/RDX/…) and return in RAX, but FEX
passes only x0=RDI. A **marshalling thunk** reads the guest arg registers from
the paused CPU state and calls the native function with its real signature.
In the harness: guest sets rdi=10/rsi=14/rdx=18, calls the thunk, native
`hle_add3(10,14,18)` → RAX=42.

**Implementation for shadPS4** ([src/core/cpu/fex_thunks.h](src/core/cpu/fex_thunks.h)):
- **No hash map:** the guest stub carries the host thunk pointer directly in the
  32 bytes; `FexThunkHandler::LookupThunk` reads it back.
- **Per-function marshalling via template:** `HleThunk<Fn>` reads the SysV args
  from the state and calls `Fn`; a generator produces one per import.
- **aerolib hook:** on ARM64, `Linker::Relocate` writes, instead of the native
  HLE address, the address of a `0F 3F` stub (`EmitThunkStub`) that points to
  `HleThunk<Fn>`. `SetThunkHandler(&FexThunkHandler)` before `InitCore`.

**Call-return shadow stack (prerequisite for `call`/`ret`):** For
return prediction, FEX uses a shadow stack of `<GuestReturnRIP, HostReturnPC>` pairs,
addressed via `REG_CALLRET_SP` (x25), loaded from `State.callret_sp`. **FEXCore
does NOT allocate it** — the embedder must (otherwise only the Linux frontend does, in
`ThreadManager`). Without it, `call` pushes to x25=0 → fault. Setup per thread:
map 4 MB (`InternalThreadState::CALLRET_STACK_SIZE`) **directly RW** (on macOS
NOT PROT_NONE+mprotect — max-protection cap → SIGBUS), then
`Thread->CallRetStackBase = base+PAGE`, `State.callret_sp = CallRetStackBase +
CALLRET_STACK_SIZE/4`.

Three runtime insights (important for the shadPS4 embedding):
1. **Do NOT call `FEXCore::Allocator::SetupHooks()`.** It installs FEX's
   Linux 64-bit OSAllocator (guest address-space reservation), which crashes
   on macOS — and which shadPS4 replaces itself anyway. The default `mmap{::mmap}`
   suffices.
2. **`aligned_alloc` bug fixed** (`AllocatorHooks.cpp`): macOS's `aligned_alloc`
   requires alignment ≥ `sizeof(void*)`; FEX calls it with `alignof(uint32_t)=4` →
   NULL. Affected **every** fextl allocation. Now normalized via `posix_memalign`.
3. **Set `CONFIG_IS64BIT_MODE=1`** before context creation; otherwise
   X86HelperGen takes the 32-bit sigret path with low-memory `MAP_FIXED_NOREPLACE`,
   which fails on macOS. Execution needs the `allow-jit` entitlement
   (ad-hoc codesign in the harness CMake).

Reproduce:
```sh
cmake -B build -DBUILD_FEX_HARNESS=ON <flags wie oben>
ninja -C build fex_harness && ./build/Bin/fex_harness
```

#### shadPS4 side: integration seam written

The ARM64 branch of `RunMainEntry` ([src/core/linker.cpp:58](src/core/linker.cpp))
now calls a new FEX CPU backend instead of `UNREACHABLE`:

- **`src/core/cpu/fex_backend.{h,cpp}`** — encapsulates the FEXCore context with all
  the recipes proven in the harness: hand-built Apple Silicon `HostFeatures`
  (no `FetchHostFeatures` → MRS SIGILL), `CONFIG_IS64BIT_MODE=1`, **no**
  `SetupHooks()`, scalar software TSO, `InitCore`. `RunMainThread`
  lays out guest stack + entry layout (RDI=params, RSI=exit, RSP), sets per thread
  **CS.L=1** and the **call-return shadow stack**, and calls `ExecuteThread`.
- **CMake:** option `ENABLE_FEX_CPU` (arm64, default OFF) adds the sources;
  x86 builds remain untouched. `target_link_libraries(... FEXCore)` is prepared as
  a comment.

**Build verification (performed):** `fex_backend.cpp` + `fex_thunks.h`
**compile and link** against the real `libFEXCore.a` on Apple Silicon
(with minimal shadPS4 header stubs) → 3.2 MB arm64 executable, zero errors.
The integration therefore demonstrably uses the FEXCore API correctly. Link recipe
(also as a comment in the CMake `ENABLE_FEX_CPU` block):
```
-I FEXCore/include -I build/include -I build/FEXCore/Source
-I FEXCore/Source/IncludePrivate -I FEXHeaderUtils -I CodeEmitter
-I External/robin-map/include   -D_M_ARM_64=1 -DFEX_DISABLE_TELEMETRY=1
Link: libFEXCore.a libFEXCore_Base.a libJemallocLibs.a
      libcephes_128bit.a libsoftfloat_3e.a  fmt  xxhash  -ldl
```

At that intermediate state, the linking-in of the real
FEXCore archives, generic HLE thunks and shared observed guest memory were still missing.
All three prerequisites are by now implemented in the real game path; see
chapter 0 and the roadmap.

#### Historical remaining list of this intermediate state

The following list stems from the harness phase. Several points are by now
done and are authoritatively tracked in chapter 0 or the roadmap.

The harness battery later grew to 24/24 tests and covers ALU, memory,
control flow, x87/SSE/AVX, SMC, HLE marshalling and multi-thread W^X. The
then-open `SYS_getcpu`, `MAP_FIXED_NOREPLACE`, futex/`__ulock` and
Darwin `ucontext` paths are implemented for the current game path. Currently
open are, in particular, broader PS4 exception policy, raw guest syscalls,
rare HLE signatures, long-term tests and further titles. A separate FEXCore
`.dylib` target is still not required for the static embedding.

### Construction site 1 — guest entry: `RunMainEntry` → FEXCore

File: `src/core/linker.cpp:37`. Today (`ARCH_X86_64`, lines 38–57) an
inline `jmp *entry_addr`. The ARM branch (line 58) becomes the FEXCore call.

One-time context initialization at startup:

```cpp
g_fex_ctx = FEXCore::Context::Context::CreateNewContext(hostFeatures); // Context.h:62
g_fex_ctx->SetSignalDelegator(&shad_signals);
g_fex_ctx->SetSyscallHandler(&shad_syscalls);   // Baustelle 2b, Context.h:134
g_fex_ctx->SetThunkHandler(&shad_thunks);        // Baustelle 2a
g_fex_ctx->SetHardwareTSOSupport(false);         // FEX software TSO on native macOS
g_fex_ctx->InitCore();                            // Context.h:72
```

Guest entry:

```cpp
static PS4_SYSV_ABI void* RunMainEntry(EntryParams* params) {
#ifdef ARCH_X86_64
    asm volatile( /* ... bestehender x86-Sprung, linker.cpp:39 ... */ );
    UNREACHABLE();
#elif defined(ARCH_ARM64)
    auto* thread = g_fex_ctx->CreateThread(
        params->entry_addr,          // GuestRIP  (linker.cpp:214)
        prepared_guest_stack_ptr);   // StackPointer, SysV layout like the PS4 kernel
    g_fex_ctx->ExecuteThread(thread); // x86→ARM64 JIT, runs   (Context.h:77)
    UNREACHABLE();
#endif
}
```

The caller remains `src/core/linker.cpp:215`.

### Construction site 2 — the HLE boundary (guest x86 → native ARM host)

The actual grind. Two return paths:

**2a — resolved library imports (217 files with `PS4_SYSV_ABI`)**
Today `Linker::Relocate` (`src/core/linker.cpp:280`, via
`ForEachRelocation`, symbol resolution around line 320) writes the **native**
HLE function pointer directly into each import location. Under FEX this address is **ARM code** —
FEX would wrongly decode it as x86.

→ Solution: resolve imports to a **guest thunk** that FEX knows via
`AddThunkTrampolineIRHandler(entrypoint, guestThunk)` as "leave the JIT, call the native
host function". Because all imports have the same `PS4_SYSV_ABI` signature,
this works **generically** in the aerolib resolution
(`src/core/aerolib/`) — not 217 times by hand.

**2b — raw `syscall` instructions**
libkernel internally makes real `syscall`s. FEXCore's `SyscallHandler`
(`Context.h:134`) intercepts them and routes them into shadPS4's existing syscall dispatcher.

---

## 4. Order

| # | Step | Why first |
|---|---------|--------------|
| 1 | Build FEXCore standalone on M-chip, translate a trivial x86 test program (construction site 0) | Hardest unknown; without it nothing runs |
| 2 | `RunMainEntry` ARM branch + context init (construction site 1) | Small, clearly bounded seam |
| 3 | Thunk for **one** HLE function end-to-end (e.g. `sceKernelAllocateDirectMemory`) | Proves the boundary, then roll out generically |
| 4 | Raw syscalls + signal delegator (construction site 2b) | |
| 5 | Disconnect `cpu_patches.cpp` on ARM | Made superfluous by FEX |

The lion's share is **step 1** (macOS port of FEXCore, so far done by nobody).
The shadPS4 side (2–5) is manageable because the seams are cleanly in place.

---

## 5. References

- FEXCore embedding API: `FEXCore/include/FEXCore/Core/Context.h`
  (`CreateNewContext` :62, `InitCore` :72, `ExecuteThread` :77,
  `CreateThread`, `SetSyscallHandler` :134, `SetThunkHandler`,
  `SetHardwareTSOSupport`, `AddThunkTrampolineIRHandler`)
- shadPS4 guest entry: `src/core/linker.cpp:37`
- shadPS4 import relocation: `src/core/linker.cpp:280`
- shadPS4 CPU patches (Rosetta workarounds): `src/core/cpu_patches.cpp:549`
- shadPS4 TLS/TCB: `src/core/tls.cpp`
- FEX (upstream): https://github.com/FEX-Emu/FEX
- FEX_MacOs (historical starting fork; the local tree contains the macOS port):
  https://github.com/Jpkovas/FEX_MacOs

---

## 6. FEX-2607 update attempt (2026-07-13)

The official stable tag `FEX-2607` (`1cc4b93e`) is ported and buildable in
`../FEX_MacOs_FEX-2607`. The existing tree `../FEX_MacOs` (`73a32ff22`) was
not overwritten in the process. For 2607, the following mechanical
adjustments are required on the shadPS4 side:

```text
Include: External/unordered_dense/include
SyscallHandler::LookupExecutableFileSection(InternalThreadState*, uint64_t)
CpuStateFrame::Pointers.GuestSignal_SIGILL
```

The archives can be built selectively without compiling the still
Linux-specific FEX frontend tools:

```sh
cmake --build ../FEX_MacOs_FEX-2607/build --target FEXCore -j8
cmake -S . -B build-arm64 -DENABLE_FEX_CPU=ON \
  -DFEXCORE_DIR="$PWD/../FEX_MacOs_FEX-2607"
cmake --build build-arm64 -j8
```

The link success is not yet a release: in the test with
`Cult.of.the.Lamb_CUSA32184_v1.00`, 2607 loses the guest return after the first dynamic
`0F 3F` HLE thunk. The native stub correctly reads `0x70027291b3` from the guest stack;
immediately after, the dispatcher calls `CompileBlock`
with RIP null. A decoder guard accordingly shows `entry=0, block=0,
source=0`; without the guard a null access arises in `DecodeInstructionImpl`.
Therefore the active build stays on `../FEX_MacOs` until this is fixed.

The architecture separation does not change with the update: PS4 extensions
and `.sprx` are x86-64 guest code and are translated by FEX. Only host plugins and
`.dylib`s actually loaded by the macOS process must be arm64. A PS4 guest module rebuilt as
arm64 would be wrong and could not be executed by FEX as the
expected x86-64 guest code.

---

## 7. Cult of the Lamb: AvPlayer seek against black screen (2026-07-13)

The FEX fallback to `73a32ff22` eliminated the 2607 return regression, but
not the black picture by itself. The game flow pauses the second AvPlayer
and then calls `sceAvPlayerJumpToTime(handle, 0)`. The previous function
only returned `ORBIS_OK` and left the format context, decoder queues and media clock
unchanged.

The call now runs through `AvPlayer`, `AvPlayerState` and `AvPlayerSource`.
`AvPlayerSource::JumpToTime` remembers pause/play, stops and joins the demuxer as well as
the decoders, flushes old guest buffers, packets and frames, seeks with
`avformat_seek_file` in the time base of the video or audio stream, sets the
internal clock to the target and restarts the pipeline. A paused player
stays paused after the restart.

Verification with
`/Users/jbw/Desktop/ps4/Cult.of.the.Lamb_CUSA32184_v1.00`:

```text
StatePause
Video Decoder Thread exited normally
Audio Decoder Thread exited normally
Demuxer Thread exited normally
Demuxer Thread started
JumpToTime: Jumped playback to 0 ms
Video Decoder Thread started
Audio Decoder Thread started
```

The arm64 build links successfully, produces no guest fault in the seek path and
subsequently renders the Cult of the Lamb main menu (`Play`, `Settings`,
`Credits`) instead of the reproduced black window. After selecting
`Play`, the transition into a playable in-game scene was also manually confirmed at the
same state; the fix is thus not just a title-menu smoke test.

---

## 8. Intermittent black screen: pthread exit race (2026-07-14)

After the working seek, the black screen reappeared intermittently on
further cold starts. The log broke off when terminating the AvPlayer workers.
A sample of the hanging process showed the game main thread in this path:

```text
sceAvPlayerStart
-> AvPlayerState::Start
-> AvPlayerSource::Stop
-> Kernel::Thread::Stop
-> posix_pthread_join
```

Demuxer, video and audio workers used the same libkernel pthread
implementation as PS4 guest threads. On exit of a native ARM64 worker, therefore,
the per-process registered PS4 thread destructor and the guest TLS
cleanup were run. The native worker thereby reentered FEX while the
main thread waited on its join. Self-joins at the end of the three workers and a
sequential stop-and-join additionally increased the deadlock risk.

The correction separates the lifecycles explicitly:

- `Pthread::is_guest_thread` is set at creation based on the start address.
- Only guest threads start via FEX and run PS4 thread and guest TLS
  destructors; native host workers stay entirely in the ARM64 host path.
- `AvPlayerSource::Stop` first publishes the stop request for all three
  workers and wakes all packet/buffer condition variables. Only then are
  demuxer, video and audio workers joined.
- Workers no longer join themselves. An external stop also triggers no
  spurious `OnEOF`.

A native renderer dump from the subsequent cold start confirmed three
distinct emitted frames: early intro animation, Devolver Digital
logo and the full `Cult of the Lamb` title screen with "X To Start".
The log passed through `StatePause`, all three complete worker exits,
`JumpToTime(0)`, `StateStop`, a new `StatePlay` and further asset
streaming without a join hang. The automatic dump hook used only for this check
was subsequently removed again.

---

## 9. FEX LookupCache: Darwin ulock wake race (2026-07-14)

A normal run without the dump hook then stalled once more at the same
visible spot, but had a different stack. This time the game main thread, the
AvPlayer demuxer and numerous Unity guest threads waited in
`FEXCore::LookupCache::FindBlock`; another thread waited in
`LookupCache::AddBlockMapping`. Thereby the shared write-priority lock of the
code cache was permanently blocked.

FEX uses futex bitsets on Linux to selectively wake a writer when readers and writers
are waiting. The local macOS port mapped this call to
`__ulock_wake(..., 1)` without a bitset. Darwin could thereby wake an arbitrary
reader. This reader still saw a waiting writer and went back to
sleep; the actual writer, however, remained asleep and nobody could
change the lock state anymore.

`FEXCore/Source/Utils/WritePriorityMutex.h` now uses wake-all for this Darwin
writer handoff. All woken threads re-check the atomic futex word,
so that only one eligible writer takes over the lock. The comment in the
Apple futex shim was corrected accordingly: a single untargeted wake
is not correct for a mixed bitset waiter set.

After rebuilding `FEXCore`, `FEXCore_Base`, `JemallocLibs` and the ARM64 shadPS4
binary, three consecutive cold starts of
`Cult.of.the.Lamb_CUSA32184_v1.00` were checked. All three passed through
`StatePause`, complete demuxer/decoder exits, `JumpToTime(0)`, the following
`StateStop`, a new `StatePlay`, the next preroll completion and the
loading of `Media/StreamingAssets/Music.streams.bank`. The previously observed
LookupCache standstill did not reappear.

---

## 10. GPU page watcher over mapping gaps (2026-07-14)

In a longer Cult gameplay run, the texture cache requested a watcher
for `0x709cc000 - 0x70c00000`. The rasterizer itself reported that this
range was not fully GPU-mapped. Immediately after,
`AddressSpace::Protect` failed:

```text
UpdatePageWatchers: Tracking memory region 0x709cc000 - 0x70c00000 which is not fully GPU mapped.
AddressSpace::Protect: mprotect failed: Cannot allocate memory
```

The page manager manages PS4 pages in 4-KB units. The existing Darwin
adaptation combined four of them each into one 16-KB host page, but iterated
over the entire requested range. If it contained GPU mapping
gaps, `mprotect` was also applied to fully unmapped host pages;
Darwin returns `ENOMEM` for that.

`PageManager::Protect` now clips the protection request under Apple Silicon
against `Rasterizer::ForEachMappedRangeInRange`. Only host pages that contain a current
GPU-mapping intersection are protected. Adjacent intervals
that hit the same host page after the 16-KB alignment are deduplicated via
`protected_until`. The effective permission remains the most restrictive
combination of the four 4-KB tracker pages.

The corrected ARM64 build linked successfully. A renewed Cult start
passed through AvPlayer seek, title path and FMOD streaming without an `mprotect` assertion.
Since the original range only arose in a longer interactive play session,
exactly this late gameplay path must additionally be repeated manually.
The preceding message about unexpected texture metadata is a
separate renderer compatibility hint and was not the direct cause of the
Darwin assertion.

---

## 11. Cat From Hell 1.01: prebound PLT and Unity semaphores (2026-07-14)

The update PRX `Media/Modules/Il2cppUserAssemblies.prx` mixes normal indirect
PLT entries (`FF 25 rel32`) with six entries prebound on the PS4
(`E9 rel32; NOP`). Affected are `scePthreadSemInit`, `Destroy`, `Post`, `Wait`,
`Trywait` and `Timedwait`. The first direct jump under FEX landed at the
unmapped address `0x6f970f1230`, even though the associated `JUMP_SLOT` relocation
had correctly set the GOT entry to a FEX HLE stub.

`NormalizeFexPlt` reconstructs, on ARM64/FEX, the table base from a
valid indirect slot and the index of its ordered `JUMP_SLOT` relocation.
The full table is validated; afterward only
the `E9` slots are rewritten back into indirect `FF 25` jumps to their respective GOT
entries. Cat 1.01 reports six normalized entries and starts
`Il2CppUserAssemblies.prx` successfully afterward.

In the continuing Unity startup path, a second error occurred. Internal Unity
semaphores contain the PS4 libc value `0xffff736d`; they were not created via the
HLE path `posix_sem_init`. The previous `PthreadSem**` model treated
the value as a native pointer anyway. `sem_wait` thereby read from `0xffff736d`,
`sem_post` from the field offset `0xffff7375`.

The POSIX semaphore implementation now registers each of its own
`PthreadSem` with a `shared_ptr` by guest handle and implementation pointer. If
the value stored in the guest is not registered, it is not dereferenced;
instead, all operations on the same guest address receive a
shared, null-initialized shadow semaphore state. The guest value
stays unchanged. This lets, for example, `Game:Main` wait while
`AssetGarbageCollectorHelper` posts the same state.

Verification:

```text
Cat From Hell CUSA54365 App Version 01.01
FEX normalized 6 prebound PLT entries in Il2CppUserAssemblies.prx
67 foreign Unity semaphores intercepted
55 graphics/compute pipelines compiled
0 Critical / 0 Access Violations

Cult of the Lamb CUSA32184 App Version 01.21
FMOD Music.streams.bank loaded
AvPlayer JumpToTime(0) completed
22 graphics/compute pipelines compiled
0 Critical / 0 Access Violations
```

---

## 12. AVX-128 context-clear backport (2026-07-14)

A comparison of the active FEX state `73a32ff22` with the separate
FEX-2607 worktree showed that AVX, AVX2 and SSE4a cannot be improved by additional
host feature bits. `SupportsAVX` is FEX's switch for
the emulated guest functionality and also sets AVX2 in the guest CPUID;
`SupportsSSE4a` sets the extended CPUID. Apple M3 provides only 128-bit NEON and
no SVE. The correct implementation therefore remains FEX's AVX-128 path,
which executes YMM operations in two halves.

The full FEX-2607 tree remains inactive because of the known HLE return regression.
Instead, two mutually dependent upstream commits were
backported in isolation:

- `12fcf96e9`: new IR operation `ContextClear` along with the ARM64 JIT implementation.
- `d6d4f84c3`: `VZEROUPPER`/`VZEROALL` invalidate the cached upper AVX
  registers and clear `CPUState::avx_high` as a contiguous region.

Since `SupportsCLZERO` is not set on the Apple host,
`ContextClear` there emits one `STP` each for two 128-bit zero vectors. For the 16
upper YMM halves in 64-bit guest code, the clear path thereby drops from 16 individual
stores to 8 paired stores. Neither SVE nor `dc zva` is falsely
activated.

The local FEX harness now contains targeted regression tests for:

```text
CPUID AVX advertised       PASS
CPUID AVX2 advertised      PASS
CPUID SSE4a advertised     PASS
SSE4a extrq immediate      PASS
AVX vzeroupper high half   PASS
AVX vzeroall low half      PASS
Gesamt                     30/30 PASS
```

Subsequently, both known Unity titles were started with the newly linked ARM64
binary. `Cat From Hell` 1.01 reached update PRX, worker/GC and
asset-streaming paths. `Cult of the Lamb` 1.21 reached AvPlayer pause, normal
demuxer/decoder exits, `JumpToTime(0)`, the next `StatePlay` cycle and
graphics-pipeline compilation. In no run did a new FEX guest fault,
`Critical` or an access violation occur.

---

## 13. shadPS4 upstream integration `6d37f61` (2026-07-14)

Before the update, the complete ARM64/FEX state including all
local integration files was secured as commit `42fa55f` and tagged with
`pre-shadps4-update-20260714`. Afterward, upstream `6d37f61` was
integrated in merge commit `3ff0324`. This preserves both the exact
restore point and the upstream history.

The merge was textually conflict-free but initially broke when compiling
the two Audio3D backends. Upstream `16b708d` now uses in
`sceAudio3dAudioOutOpen` `param.data_format`, whereas the ARM64/FEX port had
deliberately replaced the small union at this ABI boundary with a raw `u32`.
A return to the native C++ union signature would make FEX and AArch64 diverge again
in the parameter classification. Instead,
the four-byte object is reconstructed after the HLE entry via `memcpy` from `param_raw`.
A `static_assert` fixes the size assumption.

After the adjustment, `cmake --build build-arm64 -j6` built all 167 targets and
linked an ARM64 Mach-O. The separate FEX harness again passed 30/30 tests.

End-to-end smoke tests:

```text
Cat From Hell CUSA54365 App Version 01.01
FEXCore initialized and guest main thread entered
6 pre-bound PLT entries normalized
Il2CppUserAssemblies.prx started, Vulkan pipeline compiled
0 Critical / 0 Access Violations / 0 FEX guest faults

Cult of the Lamb CUSA32184 App Version 01.32
FEXCore initialized and guest main thread entered
Unity, Burst, NP, Save and FMOD modules started
AvPlayer StateReady reached
0 Critical / 0 Access Violations / 0 FEX guest faults within the smoke-test window
known: very many _is_signal_return stub calls and a correspondingly large log
```

The Cult finding is therefore only a start/AvPlayer smoke test and no statement
that the title screen or gameplay are already regression-free. The
`_is_signal_return` implementation, or rather the Unity unwind path, must
be investigated separately before such a release.

---

## Fix 2026-07-14 — `_umtx_op` and `_is_signal_return` really implemented

### Symptom

`Cult of the Lamb` (CUSA32184) hung after startup. The log showed two
endless floods: `Background Job.Worker` called the STUB `_umtx_op`
(NID `04AjkP0jO9U`) millions of times, `Game:Main` flooded the STUB `_is_signal_return`
(NID `crb5j7mkk1c`) along with an accompanying `sceKernelGetModuleInfoForUnwind`.

### Cause

Both libkernel exports were pure aerolib STUBs that returned 0 without ever
blocking. `_umtx_op` is the FreeBSD syscall for thread synchronization; if
its wait path immediately returns 0, the caller enters a pure busy-wait
loop. `_is_signal_return` is queried by the C++ unwinder per frame; as a STUB
each call logged an error line.

### Change

- New file `src/core/libraries/kernel/threads/umtx.cpp`: a real `_umtx_op`
  implementation as a host-side, address-based futex (std::mutex +
  std::condition_variable, 256 buckets). Handles the wait ops WAIT,
  WAIT_UINT[_PRIVATE], MUTEX_WAIT, CV_WAIT, **SEM_WAIT (Op 19, old FreeBSD-9
  `_usem`)** and SEM2_WAIT as well as the associated wake ops. Registered via
  `RegisterUmtx` (in `RegisterThreads`). The guest memory is flatly
  mapped under FEX, the passed pointers are directly dereferenceable.
- `_is_signal_return` in `threads/exception.cpp` as a real function (returns 0 =
  "no signal-trampoline frame", correct for all HLE unwinds), registered in
  `RegisterException`.
- Registered `LIB_FUNCTION`s take precedence over aerolib STUBs; `aerolib.inl`
  stays untouched.

Important insight: the PS4 is based on FreeBSD 9 and uses the **old**
`_usem` semaphore (`UMTX_OP_SEM_WAIT` = 19, `_SEM_WAKE` = 20), not the newer
SEM2 variant. The first test run with only SEM2 still showed ~6.9 million
"unhandled operation 19"; only with op 19/20 did the worker block correctly.

### Verification (native run, 60-s window)

- "unhandled operation" in the log: **0** (previously ~6.9 million).
- `_is_signal_return` calls in the log: **0** (flood fully gone).
- Process alive after 60 s, no hard crash; log only ~48k lines instead of 118k.
- Progress up to FMOD bank loading and active Vulkan tiling-pipeline compilation;
  the worker really blocks on the semaphore instead of spinning.

### Separately open (not caused by this fix): black screen

After the fix, the game runs far beyond the earlier hang but shows
a black screen and eventually crashes. Cause chain:

1. **Incomplete data dump.** The test folder (`CUSA32184`) is only ~1.0 GB
   in size; `Media/StreamingAssets/aa/PS4/` contains only **47 bundles** with
   a descriptive prefix (`base_assets_all_<hash>.bundle`). `catalog.json` loads
   but references hundreds of further bundles under pure hash names
   (`aa/PS4/<md5>.bundle`). In an 80-s run: **782 successful vs. 758
   missing** bundle opens. The requested hashes appear nowhere on disk
   – so it is not a name-mapping problem, but a completeness problem.
2. **Follow-up crash in the graphics worker.** Because content is missing, an object
   stays uninitialized; `UnityGfxDeviceWorker` dereferences a broken pointer
   and triggers a FEX access violation:
   `guest RIP 0x7001446a5e`, read access to `0x10000004c` (instr.
   `cmp dword ptr [r15+0x48], 0` with `r15=0x100000004`). A frame is never
   flipped (0 SubmitFlip) → black screen.

The black screen is thus primarily a data/content problem (a complete
game dump is needed), not a regression effect of the `_umtx_op`/`_is_signal_return`
implementation. Whether the `UnityGfxDeviceWorker` access error also occurs with
complete assets (a real emulation bug) can only be decided with a
complete dump and is then to be investigated separately.

---

## Third test title 2026-07-14 — Tomb Raider (CUSA00109, EU, Build 200)

Native FEX run, complete 21-GB dump (Crystal Dynamics Foundation engine,
`.tiger` archives). The `_umtx_op` fix also takes effect here (0 unhandled ops, no
semaphore spin, no thread in its own futex code).

### Finding 1 — PlayGo file location (fixed, data fix)

The first run hung immediately after `VideoOut RegisterBuffers`: Game:Main spun with
100% CPU in its own engine code (guest PC `0x700733b6b0`, called from eboot),
the guest workers were parked. Cause: `scePlayGoInitialize` could not open
`/app0/sce_sys/playgo-chunk.dat` because the dump had `playgo-chunk.dat`,
`playgo-manifest.xml` and `playgo-chunk.sha` in the game root instead of in `sce_sys/`.
After copying these three files to `sce_sys/`, PlayGo reports
`Num Chunks = 82` and the engine runs much further.

### Finding 2 — progress after the PlayGo fix

The game then passes through: shader/graphics-pipeline compilation
(KosmicKrisp/Metal, only harmless denorm warnings), active streaming of the
`.tiger` archives (PRIORITY0..3/PATCH0), `sceNpMatching2Initialize`, P2P sockets,
`sceCamera`, `sceUserService`, `sceNpParty`. Real draw shaders (vs/fs) are
compiled, but still **0 SubmitFlip** – a frame is never presented.

### Finding 3 — remaining blocker (open)

After the online/network init, Game:Main spins again; the process lives stably
~90–100 s and then terminates cleanly (`cache_storage Close: Cache dumped`,
no `Critical`/FEX fault) – characteristic of an **init watchdog in
the game**. The network HLE is only partially stubbed (`sys_sendto` returns `35`
EAGAIN, P2P `SendPacket`/`Close` STUBBED, DNS to `tras.os.eidos.com` fails).
Additionally, the GPU CP logs `ProcessCompute: SetQueueReg
reg_offset=0xb` as unhandled (it is ignored, after which `DispatchDirect` runs).

### Finding 4 — precise diagnosis of the stall (fault storm on guest code)

Sample + throttled fault instrumentation in the `GuestFaultSignalHandler` show:

- Game:Main pumps `sceNpCheckCallback` in its loop (normal frame pumping,
  no blocker – `DispatchPendingNpStateCallbacks` returns immediately on an empty
  queue). So network/online is **not** the cause.
- The dominant cost point is the GPU fault path
  `GuestFaultSignalHandler → Rasterizer::ReadMemory → BufferCache::ReadMemory`.
- The hammered fault address is **`0x7000617d20` (in eboot), `write=false`**,
  value constant `0x56415741e5894855` = x86 prologue `push rbp; mov rbp,rsp; push
  r15; push r14`. So it faults on a **guest-code region**, not on a
  data word. Under FEX the host reads the x86 code as data to JIT-translate;
  the read fault arises because the same **16-KiB Apple host page** is GPU-tracked
  (comment in `page_manager.cpp`: the host page takes on the most restrictive
  permission of all the contained 4-KiB tracker pages).

The first interpretation was 16-KiB GPU-page false sharing. **Disproven through targeted
instrumentation of `AddressSpace::Protect`:** for the hot page
`0x7000617d20` there are **zero** protection changes by the GPU page manager. So the
page is not tracked by the buffer cache (consistent with `readbacksMode=0`,
which does not create GPU read watches/PROT_NONE in the first place). The fault storm is thus
**FEX/guest-side**, not caused by the shadPS4 GPU tracking. Further
exclusions: `FEX_MULTIBLOCK=0` does not change the behavior; `copyGPUBuffers`
affects only command buffers. The exact protection/fault origin sits in the
FEX code-translation/memory interaction and can only be narrowed further with
FEX-internal instrumentation. The engine meanwhile sits in a
wait loop for a condition that does not occur.

Result: from "immediate freeze/black" (before the PlayGo fix) to deep
init/shader-compilation/asset-streaming/online-init. Up to the visible picture,
however, a solution to this 16-KiB-page/GPU-tracking interaction, or rather the
GPU wait condition, is missing – an architectural topic of the Apple Silicon GPU memory tracking,
not a single fix and **not** caused by the `_umtx_op`/`_is_signal_return` work.
`compilerrt_abort_impl` remains unresolved as an import (so far without an
observed call/fault).

Data fix that is retained: `playgo-chunk.dat`, `playgo-manifest.xml` and
`playgo-chunk.sha` were copied from the dump root to `Tomb_Raider/sce_sys/`.

### Finding 5 — TR reaches in-game, two GPU-side crash spots

Tomb Raider now starts up to **in-game** (rendering, compute dispatches,
FMOD, IME dialog active). Via lldb, two error spots were located exactly:

1. **Benign but expensive write-watch faults** in
   `AmdGpu::Liverpool::ProcessCompute` when writing back the ring read pointer:
   `*queue.read_addr %= queue.ring_size_dw`. `read_addr` = the `read_ptr_addr`
   (`0x7049c0e900`) passed by the game via `sceGnmMapComputeQueue`. The page
   is mapped but **write-protected** by the buffer cache (`memory region` →
   `r--`); every store faults, is handled by shadPS4 (InvalidateMemory) and
   continued. Functionally correct, only performance (fault storm on the
   GpuCommandProcessor thread).
2. **Fatal crash (crash cause):** fault with **host PC `0x1306615ac`**, which
   according to `vmmap` lies in the **`IOAccelerator (graphics)` region** (Metal/GPU driver
   memory), read access to the broken low address **`0x4bf480`**, from
   guest RIP `0x700060b140`. FEX reports "access violation outside current code
   buffer", i.e. the fault happens in native (driver) code that no handler
   catches. Interpretation: an invalid pointer (`0x4bf480`) gets into a
   Metal/KosmicKrisp GPU call and the driver dereferences it.

Crash 2 is a GPU-driver/pointer topic at the Metal level and cannot be cleanly fixed
with the means available in this session (host PC not symbolizable,
since it is in the driver data area; reproduction behind millions of benign Mach
exceptions). The next step would be to instrument the HLE GPU path around guest RIP `0x700060b140`
to find which Gnm/Video call passes the pointer `0x4bf480`
onward.

### Finding 5 — cause corrected: NP callback bypassed FEX

LLDB stopped before the general signal handler directly on PC `0x7000617d20`.
The bytes disassembled as `str z21, [x2, #0x4a, mul vl]` were not ARM code
generated by FEX, but the x86-64 entry of the NP state callback registered by
the game, executed as ARM64. `DispatchPendingNpStateCallbacks` called
legacy, A, toolkit and reachability callbacks as normal C++ function pointers.
The subsequent GPU fault path was therefore a symptom of the misdirected
execution, not the cause. The previous conclusion of a still-open
FEX/GPU-page interaction is thereby disproven.

All four NP callback kinds now run via
`Core::CPU::InvokeGuestOrHost`. The intermediate page-manager test change
was fully reverted. After a rebuild, the same cold start showed:

```text
Tomb Raider CUSA00109 App Version 01.00
NP guest callback returns correctly through FEX
581 new graphics pipelines compiled
regular TR loading screen and subsequently the main menu reached
0 Critical / 0 Access Violations / 0 FEX guest faults
```

### Finding 6 — full rebuild regresses `__tls_get_addr` (2026-07-14, continued)

The build was blocked by FEXCore header drift (not by our own changes):
`JITPointers` was switched from a nested `Common` substruct to flat,
and `SyscallHandler::LookupExecutableFileSection` took the `InternalThreadState`
from reference to pointer. Two mechanical 1-line adjustments in
`fex_backend.cpp` (line 832 `Pointers.Common.GuestSignal_SIGILL` -> flat; line
138 `&` -> `*`) resolved the compile. Build+link successful.

**BUT:** the freshly built binary (18:50) regresses. 3/3 runs crash
deterministically at the start of `libc.prx` in `__tls_get_addr`:
`index == null` (guest passes `RDI = 0`), `Read from address 0x0`,
`saved guest RIP 0x0`. The binary never reaches in-game.

The old binary (17:22, `shadps4.working`, from the same HEAD 17:10) gets past
`__tls_get_addr` all the way in-game (libSceRtc, ImeDialog, NpParty, compute
rendering) and then crashes at a fixed `guest RIP 0x700075dd60` with a
per-run varying garbage pointer (`0x3e1d5f47...`, `0x3f3b118b...`) — a separate
in-game crash, NOT the NP-callback path fixed in Finding 5.

**Analysis:** register indices (`REG_RDI=7`) and the gregs layout match old=new.
All 64 FEXCore `.o` are newer than the header change -> lib consistent. The source
is identical (HEAD 17:10, only our own uncommitted edits). The only difference
old<->new is the **full rebuild of 378 TUs** (triggered by the CMakeLists change
+ failed builds). Most likely cause: the rebuild with new
-O3 codegen unmasks a **latent UB/uninitialized bug** in the TLS relocation
or FEX thread-setup path — the same error class as the in-game garbage pointer.

**Consequence:** the working old `.o` are overwritten; every rebuild
now produces the regressed binary. To ship any code fix, the
TLS regression must be solved first (critical path). The working binary was secured as
`build-arm64/shadps4.working` and restored as the active `shadps4`;
the regressed one was filed as `build-arm64/shadps4.hardened_tls_regression`.

### Finding 7 — in-game crash is intermittent (race/uninitialized memory), NOT deterministic (2026-07-14)

Measured extensively on the working binary (`shadps4.working`) (5 short
runs + 1 long run):

- **~60–80% of starts do NOT crash** in the 1–2-minute window. They render actively
  (13–22 graphics/compute pipelines compiled, up to 173 BindTextures/Draws,
  playtime keeps running). TR is thus largely runnable but hangs in TR's
  long pipeline-compilation/loading phase (no visible flip in the window yet;
  Finding 5 reached the main menu earlier, only later).
- **~20–40% crash intermittently** at fixed guest RIPs that **cluster**:
  `0x700060b140` and `0x700075dd60` (plus 1 outlier `0x123fa1010`). Both RIPs
  lie in the **R_X segment of eboot.bin** (base `0x7000400000`, size `0x1270000`)
  -> **TR's own game code**, not a Sony lib.
- Fault addresses vary per run but consistently follow the pattern
  **`garbage base pointer + small field offset`**: `0x439c2187_000001cf`,
  `0x10001000_10181`, `0x3f3b118b_bf453df1`, `null+0x220`. That is, the game loads at
  a fixed location an object pointer that is sometimes valid, sometimes garbage, and
  dereferences `base->field`.

**Excluded as cause:** the preceding HLE polling loop
(`sceImeDialogGetStatus` with `client_state=-1` = only log fallback on null client
-> return `None`; `sceNpPartyCheckCallback` STUBBED -> `ORBIS_OK`). Both benign.

**Most likely cause:** intermittently **non-null-initialized or
reused guest memory** (the PS4 guarantees zeroed allocations) OR a
**race on shared state** that a worker thread fills. Same error class
as the TLS regression (Finding 6: `index==null`).

**Blockade:** confirmation (instrumentation of the allocation/crash site) and fix
need a runnable rebuild -> first the TLS rebuild regression
(Finding 6) must be solved. Thereby Finding 6 is the critical path for ANY further
code fix on TR.

### Finding 8 — Cat From Hell hangs: FEX cannot deliver GC "stop the world" signals to HLE-blocked threads (2026-07-14)

Cat From Hell (CUSA54365) now reliably hangs during asset loading. A `sample`
of the frozen process shows a **total deadlock**: every guest thread
(Game:Main, Job.Worker 0-4, Background Job.Worker 0-15,
AssetGarbageCollectorHelper, BatchDeleteObjects, the SceFios* threads, ...) is
parked in `Libraries::Kernel::posix_sem_wait` /
`Libraries::Kernel::PthreadCond::Wait` -> `Pthread::Sleep`, all bottoming out in
macOS `dispatch_semaphore_wait` -> `semaphore_wait_trap`. Nothing runs; the log
is completely stable.

Just before the freeze the log shows the Unity/il2cpp garbage collector doing a
"stop the world": `posix_pthread_kill: Raising signal 30 on thread '...'` for
every thread (signal 30 = SIGUSR1). The GC suspends all mutators via SIGUSR1,
waits for each to acknowledge (park), collects, then resumes them via a second
signal. The acknowledge/resume handshake never completes.

**Root cause (NOT the `_umtx_op` change - the deadlock sample contains zero umtx
frames):** under FEX, an async guest signal (SIGUSR1) that arrives on a thread
blocked inside an HLE host wait (`dispatch_semaphore_wait`) is only *queued* as
`g_pending_guest_signal`; it is never delivered to the guest handler, because
(1) the macOS semaphore trap is not interrupted by the signal (it auto-resumes),
and (2) `FexBackend::DispatchPendingGuestSignal()` only runs at FEX safe points
(e.g. `SyscallHandler::SleepThread`) that a thread parked in `sem_wait` never
reaches. So the guest's suspend handler never runs, the thread never
acknowledges suspension, and the GC deadlocks the whole process.

This is an architectural FEX signal-delivery gap: shadPS4's blocking HLE
primitives (`posix_sem_wait`, `PthreadCond::Wait`, event flags) built on
`dispatch_semaphore` are not interruptible by pending guest signals. On the
native-x86/Rosetta build this works because the guest handler runs directly on
the interrupted thread without the JIT indirection.

**Fix direction (needs a working rebuild):** make the HLE blocking waits
cooperate with FEX signal delivery - e.g. have `posix_sem_wait` /
`PthreadCond::Wait` wake on SIGUSR1 (bounded `wait_for` poll or an interruptible
semaphore) and call `DispatchPendingGuestSignal()` before re-waiting, so the
guest suspend handler runs and completes the GC handshake.

**Blocker:** applying this needs a rebuild, which currently regresses
`__tls_get_addr` (Finding 6). The TLS rebuild regression remains the master
blocker for every TR/CFH code fix.

### Finding 9 — Fresh-build re-entrant callback crash: missing callback-return trampoline (2026-07-14)

A clean rebuild of the tree crashed deterministically the moment any HLE
function re-entered guest execution via `CPU::InvokeGuestOrHost` /
`FexBackend::CallGuestCallback` (guest heap_malloc during TLS setup,
`posix_pthread_once` init routines, `sceNpCheckCallback` state callbacks, ...).
The fault was always the same shape: FEX decoding guest code at RIP 0
(`saved guest RIP 0x0`, `Read from address 0x0`), reported as
`DecodeInstructionsAtEntry`.

**Root cause:** the updated FEXCore callback path
(`ContextImpl::HandleCallback` -> `Dispatcher::ExecuteJITCallback`) pushes
`CpuStateFrame::Pointers.ThunkCallbackRet` as the callback's return address.
That field is populated in `ContextImpl::ExecuteThread` from
`SignalDelegator::GetThunkCallbackRET()`. The base implementation returns 0, and
`Ps4SignalDelegator` did not override it (the upstream Linux frontend overrides
it to return a VDSO `0F 3E` / `CALLBACKRET` instruction). So every re-entrant
guest callback returned to RIP 0 and faulted in the decoder.

Why the old (17:22) binary did not hit it: its `fex_backend.o` predated the
FEXCore "upstream update" that `42fa55f` snapshotted before, i.e. it was built
against a FEXCore whose callback return did not depend on
`GetThunkCallbackRET()`. Linking a freshly compiled `fex_backend.o` against the
current `libFEXCore.a` exposed the missing override. This - not a data null - is
the real "fresh-build __tls_get_addr / pthread_create regression".

**Fix (commit 15c3ee1):** emit a `0F 3E` (FEX CALLBACKRET) instruction into the
executable sentinel page and override
`Ps4SignalDelegator::GetThunkCallbackRET()` to return its address. All
re-entrant guest callbacks now unwind cleanly. Verified: Tomb Raider runs 100s+
with 0 crashes (previously crashed within ~20s at `__tls_get_addr`).

The earlier host-malloc TLS change (commit fffc1fa) is retained - it avoids an
unnecessary guest re-entry during early init - but is no longer load-bearing for
correctness now that callbacks return properly.

**Still open after this fix (separate issues, no longer crashes):**
- Tomb Raider: runs but spins its main loop on `sceNpPartyCheckCallback` polling
  without reaching rendering.
- Cat From Hell: GC "stop the world" deadlock (Finding 8).
- Cult of the Lamb: after FMOD/audio init, Game:Main sits in a `sceKernelUsleep`
  retry loop (Mono/il2cpp exception unwinding as a symptom) waiting for a
  resource/second video that never arrives; no `sceAvPlayer` calls reached.

### Finding 10 — Cat From Hell GC deadlock is a coordinator stall, not signal delivery (2026-07-14, corrects Finding 8)

Finding 8 assumed CFH hangs because the GC "stop the world" SIGUSR1 never reaches
threads parked in host HLE waits. An interruptible-wait fix was implemented
(bounded-poll `Semaphore::acquire` + a `Core::CPU::PumpGuestSignals` hook that
runs `DispatchPendingGuestSignal` from the wait) and instrumented. It disproved
its own premise and was reverted:

- `QueueGuestSignal` succeeds for every SIGUSR1 (`calls=140 ok=140 no_thread=0
  busy=0`): the signals *are* queued as pending guest signals.
- The pump never dispatched anything (`0` dispatches, polling threads always saw
  `phase==Idle`): the queued signals are delivered through the existing
  `was_in_jit` signal-return path, not through a host-wait pump. The suspend
  handlers run and the threads park suspended.
- Decisive counter: **signal 30 (suspend) raised 118 times, signal 31 (resume)
  raised 0 times.** The GC suspends every mutator but never issues a resume, so
  the GC coordinator is stuck *before* finishing collection - a higher-level GC
  stall (a thread that never acknowledges, or the coordinator waiting on state
  owned by a suspended thread), not a missing signal delivery.

The pump added ~4 ms polling to every semaphore acquire for zero benefit here, so
it was reverted. Root-causing the coordinator stall needs GC-internals tracing
(which thread fails to acknowledge suspension and why) and is a separate, deeper
investigation. The callback-return fix (Finding 9) stands and is unaffected.

### Finding 11 — Black UI/text in TR remasters: not textures/shader/blend (2026-07-15)

Tomb Raider I-III / IV-VI Remastered now boot to the menu and render ~90%
correctly (callback-return fix, Finding 9), but menu text labels and some icons
appear as solid black boxes. Ruled out, each with hard evidence:

- **Formats:** all game texture assets are BC7_UNORM / 32bpp BGRA8 / DXT1 (parsed
  the DDS headers); the emulator creates 0 unsupported-format textures.
- **Render-to-texture:** a per-draw frame dump shows the menu uses only ~4 static
  textures and 2 framebuffers, no RTT.
- **Texture content:** GPU-readback + blit(BC7->RGBA) + PNG dump of the atlases
  showed the 2048x2048 texture IS the font atlas with every glyph intact; the
  256x256 / 512x512 are solid-white fill textures; the 1920x1080 is the (working)
  background. No UI texture has black source data.
- **Fragment shader:** enabling `GPU.dump_shaders` dumped GCN + SPIR-V. The text
  shader's SPIR-V image sample is well-formed (`OpImageSampleImplicitLod
  %f32vec4 %sampler %coord None`); the "type error" annotation in the IR dump is
  benign (present in the working background shader too).
- **Blend state:** the text draws use `src=One, dst=OneMinusSrcAlpha`, correct
  premultiplied-alpha blending for the shader's premultiplied output.

Since the atlas, fragment shader and blend are all correct, the black must come
from the fragment shader's *inputs*: the per-vertex tint color (shader computes
`out = fontSample * vertexColor * alpha`) and/or the UV coordinates / sampler.
Those originate in the **vertex shader / vertex attributes** - the one stage not
yet inspected. Next step: dump the text draw's vertex-shader output (UV + color).

**Reusable tooling built this session:** GPU-image readback -> PNG (decodes any
format including BC7 for offline viewing), shader dump via `GPU.dump_shaders`
(GCN .bin/.asl + SPIR-V .spv in <UserDir>/shader/dumps), and per-draw blend-state
dump. All instrumentation was reverted; the tree is clean.

### Finding 12 — Cat From Hell 1 & 2 GC deadlock, deeper analysis (2026-07-15)

Cat From Hell 2 (CUSA56663) hits the exact same GC "stop the world" deadlock as
CFH1 (Finding 10) - confirming it is a reproducible, cross-title issue for this
Unity/il2cpp engine, not per-game. Instrumented deeper:

- **Every suspend signal IS delivered.** Logging `posix_pthread_kill` (send) vs
  `DispatchPendingGuestSignal` (handler run) shows the counts match exactly per
  thread (e.g. GC Finalizer 32 sent / 32 dispatched; Job.Worker 1/2 15/15;
  PreloadManager 14/14; Timer-Scheduler 13/13). So signal delivery is NOT the
  problem (the pump idea from Finding 10 stays disproven).
- **The suspend handlers block correctly:** a deadlock `sample` shows every
  thread - including Game:Main - parked in `posix_sem_wait` (dispatch_semaphore).
  The GC uses semaphores for suspend/resume, not SIGUSR2 (hence "0 signal 31").
- **The smell:** GC Finalizer is re-suspended 32 times, and the process ends
  fully quiescent (log stops growing, all threads in sem_wait, 0 CPU).

Interpretation: this is a **GC-stop-the-world <-> job-system <-> FEX
async-signal-timing deadlock**. Game:Main blocks in sem_wait for a worker result;
the worker is GC-suspended (parked in its resume-wait); the GC coordinator waits
on Game:Main reaching a safe point - a cycle. FEX async signals are delivered on
the target thread only at safe points (JIT / SleepThread), which shifts *when*
each thread suspends relative to the GC's expectations, and the GC's assumption
that a signalled thread halts promptly does not hold across HLE waits.

A real fix needs deep changes to FEX async-signal delivery + how HLE blocking
waits cooperate with cooperative thread suspension - high risk to the whole
integration, not a bounded change. Characterized thoroughly; deferred. All
instrumentation reverted; tree clean.

### Finding 13 — CUSA00109 xzone fix and corrected crash attribution (2026-07-15)

The newly supplied complete `Tomb Raider - Definitive Edition` v1.00 dump exposed
an Apple-host VM-layout failure first. shadPS4 reserved user VA up to
`0x5fffffffffff`; subsequent Metal allocations landed above macOS xzone's segment
table and aborted in `kk_CmdBeginRendering`. The ARM64 Apple address-space ceiling
is now `0xfffffffffff`. This leaves 16 TiB of guest user VA while keeping native
Metal/libmalloc allocations in Darwin's supported range. Repeated cold starts no
longer hit the malloc assertion and proceed through hundreds of pipelines.

The later crash at saved RIP `0x700060b140` was not a native GPU-driver pointer
fault. A shared FEX code-buffer generation was executing but
`IsAddressInCodeBuffer(thread, host_pc)` only recognized the thread's reported
current generation. Calling FEX's guarded `RestoreRIPFromHostPC` before rejecting
the address recovered guest RIP `0x70007b6f59`. Runtime bytes and offline SELF
disassembly agree:

```text
mov r8, [rcx + rax*8]
mov cl, [r8 + 0x500]   ; fault, live R8 = 0
test cl, 2
```

The function is Tomb Raider's resource-command parser. The indexed table at
`[r12+0x150]` contains a null entry (observed command index `0x1f250`). Normal FEX
and single-instruction blocks both reproduce bad resource state. An earlier global
scalar software-TSO test made the title unusably slow/black because protected
GPU-page reads entered the readback fault path. Scalar TSO is now intentionally
retested after invalid sparse ranges were hardened; vector and REP MOVS/STOS atomic
modes remain disabled. The dump was also
validated: `sce_sys/playgo-*` is correct and every required tiger archive is opened.

Current result: the Darwin malloc crash is fixed and FEX diagnostics now identify
the true guest instruction. Reaching a visible stable menu still requires finding
why the resource producer leaves that table slot null (likely sparse/direct-memory
publication), not a null-skip in the consumer.

### Finding 14 — TR I-III Remastered corruption was a bad dump (2026-07-15)

The earlier black/pink text and icon rectangles in CUSA43774 were reproduced from
an incomplete/corrupt extraction. Re-extracting the title correctly and applying
v1.04 produced normal menu text and icons without a renderer change. Therefore the
temporary texture/shader diagnostic modifications were removed. Treat comparable
atlas-wide black or magenta UI corruption as a content-integrity question first;
compare the extracted files before adding format-specific Vulkan workarounds.

### Finding 15 — Runtime Apple M1-M5 feature detection (2026-07-15)

The hand-built macOS FEX `HostFeatures` originally assumed the common M1 baseline.
That avoided the privileged MIDR/ID-register reads which fault in macOS EL0, but it
also fixed both cache-line size and code-generation capabilities for every Apple
CPU generation. The backend now queries `hw.cachelinesize` and Apple's
`hw.optional.arm.FEAT_*` sysctls. Missing or zero-valued keys stay disabled, so the
same build runs conservatively on M1 and can use newer M4/M5 features without a
fragile model-name table.

The current M3 Pro reports a 128-byte cache line, LSE, LRCPC, LRCPC2, AES, CRC,
SHA1/SHA256, PMULL, FCMA, FlagM/FlagM2, FRINTTS, AFP, and RPRES. CSSC is reported
off and MOPS is absent, so both remain off. Guest `SupportsAVX`/`SupportsSSE4a`
remain enabled because they describe the emulated PS4 Jaguar ISA, not native Apple
vectors. SVE, ECV, WFXT, and `dc zva` were not guessed or force-enabled.

The optimization changes only FEX host code selection; multiblock and block-size
behavior are unchanged. The memory model was corrected separately to scalar
software TSO. The experimental FEX disk code cache was
not enabled. A full shadPS4 ARM64 build passed, and a CUSA43774 smoke test reached
normal file streaming, audio initialization, and shader compilation without an
illegal instruction or FEX critical.

### Finding 16 — `SetHardwareTSOSupport` was a declaration, not an activator (2026-07-15)

The previous macOS setup called `SetHardwareTSOSupport(true)` based only on the
Apple Silicon CPU family. FEX uses that flag to disable its atomic software-TSO
passes; it assumes the embedding application or operating system already enabled
hardware TSO. Native shadPS4 on macOS performed no such activation, so x86 guest
threads could observe ARM's weaker ordering.

Initialization now sets `CONFIG_TSOENABLED=1` before `CreateNewContext`, keeps
`CONFIG_VECTORTSOENABLED=0` and `CONFIG_MEMCPYSETTSOENABLED=0`, and calls
`SetHardwareTSOSupport(false)`. The scalar mode targets the resource-publication
races seen in several games. This is an intentional retry after invalid sparse
guest ranges were hardened; the startup log prints the active choice and the
vector/REP atomic modes remain off.

### Finding 17 — KosmicKrisp persistent shader cache, not unsafe async draws (2026-07-15)

The current KosmicKrisp compiler creates complete Metal graphics pipelines and
states in its own source that Graphics Pipeline Library and shader objects are
not supported yet. Consequently shadPS4 cannot safely return a usable pipeline
future from `GetGraphicsPipeline()`: it must either wait or skip the draw. The
latter produces missing UI/effects and compute work cannot be skipped correctly.

Two inactive Mesa cache paths were found instead. The physical device calculated
a pipeline-cache UUID but never created its `disk_cache`, and the device had a
`fail_mem_cache` cleanup label but never created `vk.mem_cache`. Both caches are
now initialized and destroyed with the corresponding Mesa objects. KosmicKrisp's
shader serialization stores generated MSL and shader metadata, avoiding repeated
SPIR-V parsing, NIR lowering, and MSL emission on later launches. Final Metal
pipeline compilation remains synchronous and device-specific.

In shadPS4, `PipelineCache::WarmUp()` previously ran before the explicit
`vk::PipelineCache` was created. The order is corrected so all preloaded and
runtime pipelines use the same Vulkan cache. The full build passed; CUSA43774 ran
through pipeline compilation and normal game work. A controlled 26-pipeline test
created 2.8 MiB of cache data and improved immediate warm-up from 1.17 s cold to
1.05 s cached. This is a safe cache improvement, not an async draw-skip mode.

### Finding 18 — A 40-bit GPU address is not necessarily a guest VMA (2026-07-15)

`CopySparseMemory` asserted on `0x70500000` during an image upload. The existing
renderer hardening only rejected values at or above the 40-bit GPU limit, so this
low garbage/stale address reached the buffer cache even though it was outside the
actual guest VMA layout. T#/V# binding now checks both constraints. The sparse-copy
fallback revalidates the complete range under the VMA lock and zero-fills invalid
ranges, covering an unmap race after binding without dereferencing host address
zero or terminating the command processor.

Do not automatically turn `0x70500000` into `0x7050000000`. The T# decoder in
`AmdGpu::Image::Address()` already applies the GCN 256-byte unit shift; applying a
second shift would conceal the producer bug and break correct descriptors. The
new rejection log records the encoded T# base so a subsequent run can distinguish
a stale descriptor from an incorrectly constructed descriptor. The full ARM64
build passed after this hardening.

## 19. Validation ledger after `poc1.4` (2026-07-16)

This section records runtime results separately from build success. The active
checkpoint is `poc1.46` (`b58b35f`); later uncommitted experiments are not treated
as part of that checkpoint.

### 19.1 Changes that remained useful

`poc1.41` changed AVPlayer frame ownership so a decoded frame is not immediately
cleared or reused while the renderer can still read it. The original recording
alternated between a valid frame, a Y-only green frame and a fully green frame,
which matched partial Y/UV buffer reuse. After the lifetime correction the user
confirmed that the green flicker was gone.

The upstream integration was followed by restored ARM64 fixed mappings
(`poc1.42`), present/WaitRegMem loading-stall fixes (`poc1.43`), and three narrow
HLE/renderer corrections before `poc1.46`:

- marshal AJM instance flags as raw guest ABI data instead of an AArch64 aggregate;
- reconstruct `snprintf` arguments from the FEX guest varargs state;
- ignore stale serialized shader-permutation indices instead of terminating warm-up.

These changes build together in the current ARM64 application. They are kept
independent of the rejected experiments below.

### 19.2 Rejected KosmicKrisp and game-specific renderer experiments

The temporary KosmicKrisp update (`5f63f9c`) and the green-Bink workaround
(`fecd237`) were both reverted by `e93e355` and `333d320`. New Super Lucky's Tale
continued to show a green Bink movie, and the linked official Windows report shows
the same symptom. The current tree therefore uses the pre-update driver state and
does not carry the speculative workaround.

Dark Souls Remastered's enlarged/corrupt player geometry was likewise reproduced
in Windows compatibility reports. The attempted renderer changes did not repair
the image and were reverted. It remains an upstream cross-platform rendering
problem rather than evidence of an ARM64/FEX or KosmicKrisp-only fault.

### 19.3 Trophy extraction did not resolve the CUSA02607 black screen

The trophy release key was written to the user's shadPS4 `keys.json`, not to the
source tree or executable. On the next CUSA02607 run, shadPS4 successfully
extracted 60 files for `NPWR09014_00`, including `TROPCONF.XML` and its icons.
The key value is intentionally not recorded here.

SWORD ART ONLINE Re: Hollow Fragment v1.05 still stayed black. VideoOut buffers
were registered at 1280x720, but the log never reached a first flip. A process
sample showed heavy page-fault handling and `BufferCache::ReadMemory`, so two
bounded PageManager experiments were tried:

1. skip `ReadMemory` for protected writes when readbacks are disabled;
2. treat protected reads as invalidation when readbacks are disabled.

Neither changed the black screen. Both source changes were reverted and the
binary was rebuilt from the restored PageManager. This rules out those two
readback shortcuts; it does not rule out missing system fonts, stale title data,
or a different ARM64/FEX synchronization problem.

### 19.4 libc mspace prototype caused a real regression

To support titles importing the PS4 libc mspace API, an experimental copy of
`dlmalloc` was built into shadPS4 and registered for create, destroy, allocation,
free, calloc, realloc, alignment and usable-size calls. The implementation used
the caller-provided guest range as its heap and disabled fallback host mappings.
It compiled successfully, but compile success concealed an ABI/behavior mismatch.

SWORD ART ONLINE: FATAL BULLET (`CUSA10168`, v1.14) then failed consistently while
initializing `libSceNpToolkit2`. The decisive guest instruction was:

```text
guest RIP 0x7012ca4984: call qword ptr [rax + 0x10]
RAX = 0x0
fault address = 0x10
```

Rebuilding after only the PageManager revert did not help. Removing the new
mspace registrations and `dlmalloc` object from the build restored the game's
startup, which the user confirmed. The mspace prototype is therefore rejected,
not merely hidden by logging. A future implementation must reproduce PS4 flags,
metadata placement, locking, statistics, callback ownership and failure semantics
and must pass Fatal Bullet's NpToolkit2 initialization before promotion.

The separate ARM64 Fiber implementation still builds but has not yet earned a
runtime compatibility claim. It must not be grouped with the removed mspace code
when bisecting later failures. (Superseded on 2026-07-18: the fiber
implementation earned its first runtime validation after the cross-thread
context fix below.)

## 2026-07-18: SAO Hollow Realization (CUSA05033) load deadlock — three root causes

The reproducible hang after the optional `l1qg` ENOENTs turned out to be three
independent defects stacked on top of each other. Each fix was general; no
title-specific handling, timeouts, or suppressed checks were added.

### 1. `_umtx_op` implemented with Linux futex semantics (umtx.cpp)

`UMTX_OP_MUTEX_WAIT` treated `val` as an expected futex word (libthr passes 0),
so it slept exactly when the umutex was FREE and returned immediately when it
was owned — and it never set `UMUTEX_CONTESTED`, so libthr's unlock fast path
never issued `MUTEX_WAKE2`. A waiter that lost the release race slept forever
on a free mutex. `CV_WAIT` never set `c_has_waiters`, which libthr checks in
userland before issuing `CV_SIGNAL`, so every signal was skipped. The CV
timeout was also parsed as `_umtx_time` although the kernel ABI passes a plain
`timespec` with `CVWAIT_ABSTIME`/`CVWAIT_CLOCKID` in `val`, and absolute
`_umtx_time` deadlines ignored `_clockid`. Fixed to FreeBSD-9 kernel
semantics with clockid-aware absolute timeouts. Diagnostics: a 1024-entry
wait/wake ring buffer plus a "waiter blocked >30s" dump (log-flushed,
semantics unchanged). Note the Rosetta/x86 build never hit any of this: its
`_umtx_op` is still the return-0 stub, so those paths busy-wait instead of
blocking.

### 2. Vector/memcpy stores were not TSO-ordered (fex_backend.cpp)

`VECTORTSOENABLED` and `MEMCPYSETTSOENABLED` were 0 (FEX defaults), although
x86-TSO orders SSE/AVX and `rep movs` stores exactly like scalar stores.
`STRICTINPROCESSSPLITLOCKS` was also off, allowing unaligned atomics that
cross a 16-byte/cacheline boundary to tear in-process. All three are now
enabled (env overrides `FEX_VECTORTSO`/`FEX_MEMCPYSETTSO` exist for perf
experiments). This closed a real corruption class but was NOT the final cause
of this title's hang — do not treat the interim "ran 1:45h without a worker
death" observation as proof that TSO alone fixed CUSA05033.

### 3. sceFiber contexts were thread-local (fiber_fex.cpp) — the actual hang

Deterministic symptom: on every run, exactly one random `RenderMixing[n]`
worker silently disappeared during the first voice-bank load, after which
`Game:Main` waited forever on a "PhyreSema" completion semaphore. The
diagnostic chain that identified it:

- A new `RunGuestThread` check proved the guest thread stopped at
  rip=`0x700106c04d` inside non-executable BSS instead of returning to the
  HLT sentinel. The apparent "exit status" `0xdeadbeef54321abc` is simply the
  game's `__stack_chk_guard` VALUE, which function prologues/epilogues keep
  in rax — not an emulator or game exit marker.
- FEX's `GuestSignal_SIGSEGV` path in ExitOnHLT mode exits the dispatch loop
  cleanly, so a guest fault looked exactly like a normal routine return and
  the pthread layer joined the "cleanly exited" worker.
- The FEX call-ret shadow-stack dump placed the final calls inside the game's
  fiber framework, and sceFiber API tracing showed the voice fibers
  (`vm_69_0000`, `vm_11_0000`, ...) being run round-robin across all four
  RenderMixing threads.

Root cause: `fiber_fex.cpp` kept suspended fiber contexts in a
`thread_local` map. sceFiber explicitly allows resuming a fiber on a
different thread (the context belongs to the fiber; the x86 reference stores
it via `fiber->context`). On a cross-thread resume the new thread found no
context and `Start()` re-ran the fiber from its entry on its still-live
context stack, corrupting it and eventually jumping to a data address.

Fix: suspended contexts now live in a global per-fiber store (mutex-guarded;
the Idle->Run state CAS already prevents one fiber from running on two
threads; resume consumes the context). XMM0-15 and MXCSR are now saved and
restored alongside the 16 GPRs — the old GPR-only save was its own
corruption risk for float-heavy audio code.

### Permanent diagnostics added

- `RunGuestThread` validates that a guest thread ended on the HLT return
  sentinel; otherwise it logs GPRs, a stack window, code bytes around rip and
  the pending call-ret chain, then flushes the log (a silent guest-thread
  death is no longer possible to miss).
- Fiber `Activate` validates that the resume target is executable and logs
  the fiber identity if not; sceFiber APIs are trace-logged (throttled).
- Guest thread exits, semaphore creation/deletion/cancel and all failing
  semaphore waits are logged; unaligned-atomic backpatches are counted.

### Verification

All four RenderMixing threads processed >150k fiber cycles across multiple
voice banks without losing a single worker (previously exactly one worker
died per run at the first voice bank). The title reaches gameplay on native
ARM64/FEX — user-confirmed in-game on 2026-07-18. This is also the first real
runtime validation of the ARM64 fiber implementation.

## 2026-07-18: CUSA32809 (SAO Last Recollection) — canonical carveout emulation

The title reserves memory pools at the canonical PS4 fixed addresses
(`sceKernelMemoryPoolReserve` at 0x1000000000/0x2000000000/0x3000000000 with
FIXED flags) and then commits and **directly dereferences** those constants,
ignoring the relocated address returned in `addrOut`. On macOS the kernel
hard-reserves the GPU carveout 0x1000000000-0x6FFFFFFFFF in every process
(native arm64 AND x86_64 under Rosetta; probed: `mmap(MAP_FIXED)` fails,
`mach_vm_deallocate` is a no-op, `mach_vm_map(VM_FLAGS_OVERWRITE)` returns
KERN_PROTECTION_FAILURE), so those addresses can never be backed directly.

Three layers now make canonical carveout accesses work by redirecting each to
the address the memory manager actually mapped (the linear shift in
`MemoryManager::TranslateCanonicalGuestAddress`):

1. **Memory API bookkeeping** (`memory.cpp`): the ARM64 relocation was
   one-sided — applied at reserve/map time but not to later guest calls that
   pass the same canonical constants. Translation now also runs at
   PoolCommit/PoolDecommit/Protect/UnmapMemory (NOT QueryProtection, which FEX
   feeds guest RIPs for exec-range checks). Removes the former
   `PoolCommit: invalid address` assert.

2. **JIT load/store faults** (`a64_ls_emulator.cpp` + `fex_backend.cpp`):
   guest JIT code that dereferences a canonical carveout address faults; the
   handler decodes the single faulting A64 load/store and performs it against
   the relocated address, then advances the host PC by 4. The decoder covers
   scalar/pair/atomic (LDR/STR/LDUR/LDP/STP/LDAR/STLR/LDAPR/LDAPUR/STLUR/LSE
   atomics/CAS) and the NEON register and structure families (LD1-4/ST1-4
   single & multiple, LD1R), and is verified instruction-by-instruction
   against native execution in a JIT self-test (45 encodings, all matching).

3. **HLE pointer arguments** (`fex_hle.h` + `memory.cpp`): a canonical pointer
   passed to a native HLE function is translated at the marshalling boundary
   (only pointer/reference-typed args, only when it resolves to a live
   relocated mapping), so native shadPS4 code dereferences the mapping instead
   of the unmapped constant. Fixed e.g. `posix_pthread_mutex_init`.

Result: the title now boots through the entire pool-reservation/commit phase,
libc/RTC/thread init and deep into engine startup, reaching the AvPlayer video
decode path — from an immediate crash at the very first pool commit.

Remaining (separate, not carveout): it then faults in native FFmpeg swscale
(`rgb15leToUV_half_c` writing to a ~5 GB host address, below the carveout),
which is a distinct video-path issue in the AvPlayer stack, not related to the
address relocation. Also note: the title uses these pools as hot working
memory, so it takes millions of fault-emulated accesses during boot — correct
but slow; a performant solution would need JIT-level guest-address translation
(a FEX feature that does not exist), not per-fault emulation. The carveout
handlers only ever run for titles that touch the carveout, so titles that map
normally (e.g. CUSA05033) are unaffected.
