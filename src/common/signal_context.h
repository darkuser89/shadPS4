// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Common {

void* GetXmmPointer(void* ctx, u8 index);

void* GetRip(void* ctx);

void IncrementRip(void* ctx, u64 length);

bool IsWriteError(void* ctx);

/// Returns true when the native fault is an ARM64 data-alignment fault.
bool IsAlignmentError(void* ctx);

} // namespace Common
