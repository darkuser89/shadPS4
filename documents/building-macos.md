<!--
SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

## Build shadPS4 for macOS

### Install the necessary tools to build shadPS4:

First, make sure you have **Xcode 26.0 or newer** installed.

For installing other tools and library dependencies we will be using [Homebrew](https://brew.sh/).

First, install Homebrew:
```
# Installs Homebrew to /opt/homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Adds Homebrew to your path
echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> ~/.zprofile
eval $(/opt/homebrew/bin/brew shellenv)
```

Then, use Homebrew to install the required build tools:
```
brew install clang-format cmake
```

Finally, install the dependencies required for building the KosmicKrisp Vulkan driver. You can skip this by setting `-DENABLE_SYSTEM_VULKAN=ON` when configuring, but you are responsible for having a compatible Vulkan setup installed.
```
brew install meson ninja pkg-config llvm spirv-tools spirv-llvm-translator libclc
pip3 install --break-system-packages mako packaging pyyaml
```

### Cloning and compiling:

Clone the repository recursively:
```
git clone --recursive https://github.com/shadps4-emu/shadPS4.git
cd shadPS4
```

Generate the build directory in the shadPS4 directory:
```
cmake -S . -B build/ -DCMAKE_OSX_ARCHITECTURES=x86_64
```

Enter the directory:
```
cd build/
```

Use cmake to build the project:
```
cmake --build . --parallel$(sysctl -n hw.ncpu)
```

Now run the emulator:
```
./shadps4 /"PATH"/"TO"/"GAME"/"FOLDER"/eboot.bin
```

### Native ARM64 build with FEXCore (this fork, Apple Silicon)

This fork can build shadPS4 as a native arm64 binary that executes the x86-64
guest code through an embedded FEXCore JIT instead of running the whole
emulator under Rosetta 2. It requires a macOS-ported, already-built FEX tree
(see `FEX_ROADMAP.md` and `INTEGRATION.md` for status and internals).

Configure with:
```
cmake -S . -B build-arm64/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DENABLE_QT_GUI=OFF -DENABLE_UPDATER=OFF \
  -DENABLE_FEX_CPU=ON \
  -DFEXCORE_DIR=/path/to/FEX_MacOs_FEX-2607
cmake --build build-arm64 --target shadps4
```

`FEXCORE_DIR` must point at the root of the built FEX source+build tree.
The host binary, FEXCore and native plugins are ARM64; `eboot.bin`, PRX and
PS4 system modules remain x86-64 guest code.

Runtime environment knobs (debugging/performance experiments only — the
defaults are the correct configuration):

- `FEX_MAXINST`, `FEX_MULTIBLOCK` — JIT block formation controls for
  isolating translation regressions.
- `FEX_VECTORTSO=0`, `FEX_MEMCPYSETTSO=0` — disable the TSO ordering of
  vector/rep-string stores. Faster, but loses x86 memory-model correctness;
  known to corrupt lock-free job queues (see INTEGRATION.md, 2026-07-18).
