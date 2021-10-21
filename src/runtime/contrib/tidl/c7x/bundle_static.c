/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/packed_func.h>

#include "bundle.h"
#include "tidl_api.h"

/** \brief Tiered memory management
 * Tier 1: Small size allocation (<= 1KB), handled by TVM managed memory (heap)
 *         - Page-based allocation, minimum allocation is a page
 *         - Define page to be 128 bytes to avoid waste  (adjustable)
 *         - Define total heap size to be 1MB (adjustable)
 *         - Alloc total memory size from appMem and give to TVM memory manager
 * Tier 2: Large size allocation (> 1KB), handled directly by appMem{Alloc,Free}
 *         - Book-keep (ptr, size) alloc info, to be used in appMemFree(ptr, size)
 *           (TVM heap does not have this requirement)
 *         - Define max allocation entries to be 2048 (adjustable)
 * TODO: If there are still memory allocation when running the TVM model,
 *       we should get the maximum allocation size across all layers,
 *       and pre-allocate this memory at TVM runtime create time.
 *       We may need a flag to indicate allocation during running the network.
 */

#if defined(__C7100__) && ! defined(HOST_EMULATION)
  #define CRT_MEMORY_NUM_PAGES (1 * 1024 * 8)
  #define CRT_MEMORY_PAGE_SIZE_LOG2 7
  #define CRT_MEMORY_SIZE (CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2))
  #define CRT_MEMORY_MAX_ALLOC_SIZE (1 << 10)
  static uint8_t *g_crt_memory = NULL;
  static MemoryManagerInterface* g_memory_manager = NULL;
#else
  #define CRT_MEMORY_NUM_PAGES 65536
  #define CRT_MEMORY_PAGE_SIZE_LOG2 10
  #define CRT_MEMORY_SIZE (CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2))
  static uint8_t g_crt_memory[CRT_MEMORY_SIZE];
  static MemoryManagerInterface* g_memory_manager;
#endif

#define CRT_MEMORY_DEFAULT_ALIGN 128
#define MAX_PTR_SIZE_MAP_SIZE (8192U)
typedef struct {
  void *ptrs[MAX_PTR_SIZE_MAP_SIZE];
  int  sizes[MAX_PTR_SIZE_MAP_SIZE];
  int  size;
} AllocPtrSizeMap_t;
static AllocPtrSizeMap_t *tvmcrt_alloc_size_map = NULL;

static void tvmcrt_free_all();

/** \brief Error handling */
static jmp_buf tvmcrt_jmpbuf;
/** \brief Called by functions here and memory allocation in c7x_tvm_runtime.h */
void    tvmcrt_exit(int ecode)
{
  longjmp(tvmcrt_jmpbuf, -1);
}

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                                              \
  do {                                                                               \
    tvm_crt_error_t ret = (func);                                                    \
    if (ret != kTvmErrorNoError) {                                                   \
      fprintf(stderr, "%s: %d: error: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
      tvmcrt_exit(ret);                                                              \
    }                                                                                \
  } while (0)


void* tvm_runtime_create(const char* json_data, const char* params_data,
                                 const uint64_t params_size) {
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMByteArray params;
  params.data = params_data;
  params.size = params_size;

  TVMContext ctx;
  ctx.device_type = (DLDeviceType)device_type;
  ctx.device_id = device_id;

  // get pointers
#if defined(__C7100__) && ! defined(HOST_EMULATION)
  g_crt_memory = (uint8_t *) appMemAlloc(APP_MEM_HEAP_DDR, CRT_MEMORY_SIZE,
                                         (1 << CRT_MEMORY_PAGE_SIZE_LOG2));
  tvmcrt_alloc_size_map = (AllocPtrSizeMap_t *) appMemAlloc(APP_MEM_HEAP_DDR,
                                              sizeof(AllocPtrSizeMap_t), CRT_MEMORY_DEFAULT_ALIGN);
  if (g_crt_memory == NULL || tvmcrt_alloc_size_map == NULL)
  {
    printf("tvm_runtime_create: failed to alloc mem, %d\n", CRT_MEMORY_SIZE);
    if (g_crt_memory != NULL)
      appMemFree(APP_MEM_HEAP_DDR, g_crt_memory, CRT_MEMORY_SIZE);
    if (tvmcrt_alloc_size_map != NULL)
      appMemFree(APP_MEM_HEAP_DDR, tvmcrt_alloc_size_map, sizeof(AllocPtrSizeMap_t));
    return NULL;
  }
  memset(g_crt_memory, 0, CRT_MEMORY_SIZE);
  memset(tvmcrt_alloc_size_map, 0, sizeof(AllocPtrSizeMap_t));
#endif

  TVMGraphRuntime* graph_runtime = NULL;

  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVM_CCALL(MemoryManagerCreate(&g_memory_manager, g_crt_memory, CRT_MEMORY_SIZE,
                                  CRT_MEMORY_PAGE_SIZE_LOG2));
    // TVMInitializeRuntime leaks 2 memory allocs, each TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES
    TVM_CCALL(TVMInitializeRuntime());
    TVMPackedFunc pf;
    TVMArgs args = TVMArgs_Create(NULL, NULL, 0);

    // Workaround for a bug in the C runtime. The number of values returned is
    // not set by TVMPackedFunc_Call and is left uninitialized. However, this
    // value is checked in TVMArgs_AsModuleHandle
    pf.ret_value.values_count = 1;

    TVM_CCALL(TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args));
    TVM_CCALL(TVMPackedFunc_Call(&pf));

    TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

    // create runtime modules
    TVM_CCALL(TVMGraphRuntime_Create(json_data, mod_syslib, &ctx, &graph_runtime));
    TVM_CCALL(TVMGraphRuntime_LoadParams(graph_runtime, params.data, params.size));
  }
  else
  {
    tvmcrt_free_all();
    graph_runtime = NULL;
  }

  return graph_runtime;
}

TVM_DLL int tvm_runtime_destroy(void* runtime) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_Release(&graph_runtime);
  tvmcrt_free_all();
  return 0;
}

TVM_DLL int tvm_runtime_set_input(void* runtime, const char* name, DLTensor* tensor) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
    TVMGraphRuntime_SetInput(graph_runtime, name, tensor);
  }
  else
  {
    printf("tvm_runtime_set_input: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_set_input_raw(void* runtime, const char* name, void* tensor_raw) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
    TVMGraphRuntime_SetInputRaw(graph_runtime, name, tensor_raw);
  }
  else
  {
    printf("tvm_runtime_set_input_raw: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_run(void* runtime) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
    TVMGraphRuntime_Run(graph_runtime);
  }
  else
  {
    printf("tvm_runtime_run: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_get_output(void* runtime, int32_t index, DLTensor* tensor) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
    TVMGraphRuntime_GetOutput(graph_runtime, index, tensor);
  }
  else
  {
    printf("tvm_runtime_get_output: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_get_output_raw(void* runtime, int32_t index, void* tensor_raw) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
    TVMGraphRuntime_GetOutputRaw(graph_runtime, index, tensor_raw);
  }
  else
  {
    printf("tvm_runtime_get_output_raw: failed\n");
    return -1;
  }

  return 0;
}

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  //vfprintf(stderr, msg, args);
  vprintf(msg, args);
  va_end(args);
}

/** \brief Redirect all fprintf to stdout
 */
void TVM_C7x_fprintf(FILE *stream, const char *format, ...) {
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
  fprintf(stderr, "TVMPlatformAbort: %d\n", error_code);
#ifdef ENABLE_TVM_PLATFORM_ABORT_BACKTRACE
  tvm_platform_abort_backtrace();
#endif
  tvmcrt_exit(-1);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLContext ctx, void** out_ptr) {
  *out_ptr = NULL;

  // Tier 1: try CRT_MEMORY first
  if (num_bytes <= CRT_MEMORY_MAX_ALLOC_SIZE && g_memory_manager != NULL)
  {
    g_memory_manager->Allocate(g_memory_manager, num_bytes, ctx, out_ptr);
  }

  // Tier 2: directly allocate from appMem, bookkeep (ptr, size) for Free() later
  if (*out_ptr == NULL && tvmcrt_alloc_size_map != NULL)
  {
    *out_ptr = appMemAlloc(APP_MEM_HEAP_DDR, num_bytes, CRT_MEMORY_DEFAULT_ALIGN);
    if (*out_ptr != NULL)
    {
      if (tvmcrt_alloc_size_map->size < MAX_PTR_SIZE_MAP_SIZE)
      {
        tvmcrt_alloc_size_map->ptrs[tvmcrt_alloc_size_map->size] = *out_ptr;
        tvmcrt_alloc_size_map->sizes[tvmcrt_alloc_size_map->size] = num_bytes;
        tvmcrt_alloc_size_map->size += 1;
      }
      else
      {
        printf("tvmcrt_alloc_size_map overflow: %d\n", tvmcrt_alloc_size_map->size);
        appMemFree(APP_MEM_HEAP_DDR, *out_ptr, num_bytes);
        *out_ptr = NULL;
      }
    }
  }

  if (*out_ptr != NULL)
  {
    memset(*out_ptr, 0, num_bytes);
    return kTvmErrorNoError;
  }

  TVMPlatformAbort(kTvmErrorPlatformNoMemory);
  return kTvmErrorPlatformNoMemory;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLContext ctx) {
  tvm_crt_error_t err = kTvmErrorNoError;

  if (ptr >= g_crt_memory && ptr < g_crt_memory + CRT_MEMORY_SIZE)
  {
    err = g_memory_manager->Free(g_memory_manager, ptr, ctx);
  }
  else
  {
    int i, size = 0;
    for (i = 0; i < tvmcrt_alloc_size_map->size; i++)
    {
      if (tvmcrt_alloc_size_map->ptrs[i] == ptr)
      {
        size = tvmcrt_alloc_size_map->sizes[i];
        tvmcrt_alloc_size_map->ptrs[i] = tvmcrt_alloc_size_map->ptrs[
                                              tvmcrt_alloc_size_map->size - 1];
        tvmcrt_alloc_size_map->sizes[i] = tvmcrt_alloc_size_map->sizes[
                                              tvmcrt_alloc_size_map->size - 1];
        tvmcrt_alloc_size_map->size -= 1;
        break;
      }
    }
    if (size > 0)
      appMemFree(APP_MEM_HEAP_DDR, ptr, size);
    else
    {
      printf("Warning: tvmcrt: Ptr %p size unknown, not freed\n", ptr);
    }
  }

  return err;
}

tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorFunctionCallNotImplemented; }

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  return kTvmErrorFunctionCallNotImplemented;
}

/** \brief Clean up all memory allocations in case anything goes wrong
 */
static void
tvmcrt_free_all()
{
  if (tvmcrt_alloc_size_map != NULL)
  {
    int i = 0;
    for (i = 0; i < tvmcrt_alloc_size_map->size; i++)
    {
      void *ptr = tvmcrt_alloc_size_map->ptrs[i];
      int  size = tvmcrt_alloc_size_map->sizes[i];
      if (size > 0)
        appMemFree(APP_MEM_HEAP_DDR, ptr, size);
    }
    appMemFree(APP_MEM_HEAP_DDR, tvmcrt_alloc_size_map, sizeof(AllocPtrSizeMap_t));
    tvmcrt_alloc_size_map = NULL;
  }
#if defined(__C7100__) && ! defined(HOST_EMULATION)
  if (g_crt_memory != NULL)
  {
    appMemFree(APP_MEM_HEAP_DDR, g_crt_memory, CRT_MEMORY_SIZE);
    g_crt_memory = NULL;
  }
#endif
  g_memory_manager = NULL;
}
