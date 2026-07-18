// SPDX-FileCopyrightText: Copyright 1997-2026 Sam Lantinga <slouken@libsdl.org>
// SPDX-License-Identifier: Zlib

// Build the mspace-only half of SDL's dlmalloc copy. SDL itself builds this
// file with MSPACES disabled, so these symbols do not collide with SDL's
// allocator. Keeping the allocator metadata in the guest-provided region is
// important: allocations returned to PS4 code must remain guest addresses.
#include "SDL_internal.h"

// SDL selects the system malloc on macOS. Undefine this only for the second,
// mspace-only inclusion below so the embedded dlmalloc implementation is built.
#undef HAVE_MALLOC
#define ONLY_MSPACES 1
#define USE_LOCKS 1
#define USE_SPIN_LOCKS 1

// SDL_malloc.c also contains SDL's public allocator dispatch at the end. The
// mspace-only build intentionally has no global dlmalloc entry points, so give
// that unused dispatch private names backed by libc and avoid colliding with
// the real SDL library linked by shadPS4.
static void* dlmalloc(size_t size) {
    return malloc(size);
}
static void* dlcalloc(size_t count, size_t size) {
    return calloc(count, size);
}
static void* dlrealloc(void* memory, size_t size) {
    return realloc(memory, size);
}
static void dlfree(void* memory) {
    free(memory);
}
#undef SDL_GetOriginalMemoryFunctions
#undef SDL_GetMemoryFunctions
#undef SDL_SetMemoryFunctions
#undef SDL_GetNumAllocations
#undef SDL_malloc
#undef SDL_calloc
#undef SDL_realloc
#undef SDL_free
#define SDL_GetOriginalMemoryFunctions shad_mspace_GetOriginalMemoryFunctions
#define SDL_GetMemoryFunctions shad_mspace_GetMemoryFunctions
#define SDL_SetMemoryFunctions shad_mspace_SetMemoryFunctions
#define SDL_GetNumAllocations shad_mspace_GetNumAllocations
#define SDL_malloc shad_mspace_malloc
#define SDL_calloc shad_mspace_calloc
#define SDL_realloc shad_mspace_realloc
#define SDL_free shad_mspace_free
#include "../../../../externals/sdl3/src/stdlib/SDL_malloc.c"
