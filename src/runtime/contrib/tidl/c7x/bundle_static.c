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
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/packed_func.h>
#include <tvm/runtime/crt/page_allocator.h>

#include "bundle.h"
#include "tidl_api_mem.h"

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
 */

#if defined(__C7100__) || defined(__C7120__) || defined(__C7504__) || defined(__C7524__)
#define C7X_TARGET
#endif

#define CRT_MEMORY_NUM_PAGES (1 * 1024 * 8)
#define CRT_MEMORY_PAGE_SIZE_LOG2 7
#define CRT_MEMORY_SIZE (CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2))
#define CRT_MEMORY_MAX_ALLOC_SIZE (1 << 10)
static MemoryManagerInterface* g_memory_manager = NULL;

#if defined(C7X_TARGET) && ! defined(HOST_EMULATION)
  static uint8_t *g_crt_memory = NULL;
#else
  static uint8_t g_crt_memory[CRT_MEMORY_SIZE];
#endif


#define CRT_MEMORY_DEFAULT_ALIGN (8U)
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
      fprintf(stderr, "%s: %d: error: %s\n", __FILE__, __LINE__, CRT_TVMGetLastError()); \
      tvmcrt_exit(ret);                                                              \
    }                                                                                \
  } while (0)


TVM_DLL void* tvm_runtime_create(const char* json_data, const char* params_data,
                                 const uint64_t params_size) {
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMByteArray params;
  params.data = params_data;
  params.size = params_size;

  DLDevice dev;
  dev.device_type = (DLDeviceType)device_type;
  dev.device_id = device_id;

  // get pointers
#if defined(C7X_TARGET) && ! defined(HOST_EMULATION)
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

  TVMGraphExecutor* graph_executor = NULL;

  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVM_CCALL(PageMemoryManagerCreate(&g_memory_manager, g_crt_memory, CRT_MEMORY_SIZE,
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
    TVM_CCALL(TVMGraphExecutor_Create(json_data, mod_syslib, &dev, &graph_executor));
    TVM_CCALL(TVMGraphExecutor_LoadParams(graph_executor, params.data, params.size));

    // allocate ddr scratch memory used in the generated C7x code for layers
    extern size_t get_ddr_scratch_mem_size();
    void  *scratch_mem_addr = NULL;
    size_t scratch_mem_size = get_ddr_scratch_mem_size();
    if (scratch_mem_size > 0)
      TVMPlatformMemoryAllocate(scratch_mem_size, dev, &scratch_mem_addr);
    tvm_tidl_ddr_scratch_set(scratch_mem_addr, scratch_mem_size);
  }
  else
  {
    tvmcrt_free_all();
    graph_executor = NULL;
  }

  return graph_executor;
}

TVM_DLL int tvm_runtime_destroy(void* executor) {
  TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
  TVMGraphExecutor_Release(&graph_executor);
  tvmcrt_free_all();
  return 0;
}

TVM_DLL int tvm_runtime_set_input(void* executor, const char* name, DLTensor* tensor) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
    TVMGraphExecutor_SetInput(graph_executor, name, tensor);
  }
  else
  {
    printf("tvm_runtime_set_input: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_set_input_raw(void* executor, const char* name, void* tensor_raw) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
    TVMGraphExecutor_SetInputRaw(graph_executor, name, tensor_raw);
  }
  else
  {
    printf("tvm_runtime_set_input_raw: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_run(void* executor) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
    TVMGraphExecutor_Run(graph_executor);
  }
  else
  {
    printf("tvm_runtime_run: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_get_output(void* executor, int32_t index, DLTensor* tensor) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
    TVMGraphExecutor_GetOutput(graph_executor, index, tensor);
  }
  else
  {
    printf("tvm_runtime_get_output: failed\n");
    return -1;
  }

  return 0;
}

TVM_DLL int tvm_runtime_get_output_raw(void* executor, int32_t index, void* tensor_raw) {
  if (! setjmp(tvmcrt_jmpbuf))
  {
    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
    TVMGraphExecutor_GetOutputRaw(graph_executor, index, tensor_raw);
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

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  *out_ptr = NULL;

  // Tier 1: try CRT_MEMORY first
  if (num_bytes <= CRT_MEMORY_MAX_ALLOC_SIZE && g_memory_manager != NULL)
  {
    g_memory_manager->Allocate(g_memory_manager, num_bytes, dev, out_ptr);
  }

#if defined(C7X_TARGET) && ! defined(HOST_EMULATION)
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
#else
  if (*out_ptr == NULL)
  {
    *out_ptr = malloc(num_bytes);
  }
#endif

  if (*out_ptr != NULL)
  {
    memset(*out_ptr, 0, num_bytes);
    return kTvmErrorNoError;
  }

  TVMPlatformAbort(kTvmErrorPlatformNoMemory);
  return kTvmErrorPlatformNoMemory;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  tvm_crt_error_t err = kTvmErrorNoError;

  if ((uint8_t*)ptr >= g_crt_memory && (uint8_t*)ptr < g_crt_memory + CRT_MEMORY_SIZE)
  {
    err = g_memory_manager->Free(g_memory_manager, ptr, dev);
  }
  else
  {
#if defined(C7X_TARGET) && ! defined(HOST_EMULATION)
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
#else
    free(ptr);
#endif
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
#if defined(C7X_TARGET) && ! defined(HOST_EMULATION)
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
  if (g_crt_memory != NULL)
  {
    appMemFree(APP_MEM_HEAP_DDR, g_crt_memory, CRT_MEMORY_SIZE);
    g_crt_memory = NULL;
  }
#endif
  g_memory_manager = NULL;
}

/** \brief TVM C codegen (using flat memory allocation) lowers "Reshape" op to "__nop",
 *         because input and output share the same memory. (graph_executor_codegen.cc)
 *         So we provide the "__nop" function for the graph executor runtime.
 *         To be removed once we move to the aot executor codegen and runtime.
 */
int32_t __nop(void* __restrict__ args, void* __restrict__ arg_type_ids, int num_args, void* __restrict__ out_ret_value, void* __restrict__ out_ret_tcode, void* __restrict__ resource_handle) {
  return 0;
}
