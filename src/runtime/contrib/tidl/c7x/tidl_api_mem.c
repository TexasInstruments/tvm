// A simple wrapper for malloc to collect allocation statistics
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (HOST_EMULATION)
   #include <malloc.h>
#endif
#include "tidl_api.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

static uint32_t malloc_size = 0;
static uint32_t malloc_requests = 0;

EXTERN_C
void *tidl_malloc(size_t size)
{
  malloc_size += size;
  ++malloc_requests;
#ifndef HOST_EMULATION
  void *ptr = appMemAlloc(APP_MEM_HEAP_DDR, size, 128);
#else
  void *ptr = malloc(size);
#endif
  memset(ptr, 0, size);
  return ptr;
}

EXTERN_C
void *tidl_memalign(size_t align, size_t size)
{
  malloc_size += size;
  ++malloc_requests;
#ifndef HOST_EMULATION
  void *ptr = appMemAlloc(APP_MEM_HEAP_DDR, size, align);
#else
#if defined(MSVC_BUILD)
  void *ptr = _aligned_malloc(size, align);
#else
  void *ptr = memalign(align, size);
#endif
#endif
  memset(ptr, 0, size);
  return ptr;
}

EXTERN_C
void tidl_free(void *ptr, size_t size)
{
#ifndef HOST_EMULATION
  appMemFree(APP_MEM_HEAP_DDR, ptr, size);
#else
  free(ptr);
#endif
}

EXTERN_C
void tidl_malloc_report()
{
  printf("TIDL dynamic allocation: %u bytes in %d requests\n", 
     malloc_size, malloc_requests);
}
