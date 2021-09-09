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
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/packed_func.h>

#include "bundle.h"
#include "tidl_api.h"

#define CRT_MEMORY_NUM_PAGES 65536
#define CRT_MEMORY_PAGE_SIZE_LOG2 10
#define CRT_MEMORY_SIZE (CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2))

#if defined(__C7100__) && ! defined(HOST_EMULATION)
static uint8_t *g_crt_memory;
static MemoryManagerInterface* g_memory_manager;
#else
static uint8_t g_crt_memory[CRT_MEMORY_SIZE];
static MemoryManagerInterface* g_memory_manager;
#endif

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                                              \
  do {                                                                               \
    tvm_crt_error_t ret = (func);                                                    \
    if (ret != kTvmErrorNoError) {                                                   \
      fprintf(stderr, "%s: %d: error: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
      exit(ret);                                                                     \
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
  g_crt_memory = (uint8_t *) appMemAlloc(APP_MEM_HEAP_DDR, CRT_MEMORY_SIZE, (1 << CRT_MEMORY_PAGE_SIZE_LOG2));
  //printf("###TVMPlatformMemory CRT_MEMORY_SIZE=0x%x, g_crt_memory=%p\n", CRT_MEMORY_SIZE, g_crt_memory);
  memset(g_crt_memory, 0, CRT_MEMORY_SIZE);
#endif
  TVM_CCALL(MemoryManagerCreate(&g_memory_manager, g_crt_memory, CRT_MEMORY_SIZE,
                                CRT_MEMORY_PAGE_SIZE_LOG2));
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

  // run modules
  TVMGraphRuntime* graph_runtime = NULL;
  TVM_CCALL(TVMGraphRuntime_Create(json_data, mod_syslib, &ctx, &graph_runtime));
  TVM_CCALL(TVMGraphRuntime_LoadParams(graph_runtime, params.data, params.size));

  return graph_runtime;
}

TVM_DLL void tvm_runtime_destroy(void* runtime) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_Release(&graph_runtime);
#if defined(__C7100__) && ! defined(HOST_EMULATION)
  appMemFree(APP_MEM_HEAP_DDR, g_crt_memory, CRT_MEMORY_SIZE);
#endif
}

TVM_DLL void tvm_runtime_set_input(void* runtime, const char* name, DLTensor* tensor) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_SetInput(graph_runtime, name, tensor);
}

TVM_DLL void tvm_runtime_set_input_raw(void* runtime, const char* name, void* tensor_raw) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_SetInputRaw(graph_runtime, name, tensor_raw);
}

TVM_DLL void tvm_runtime_run(void* runtime) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_Run(graph_runtime);
}

TVM_DLL void tvm_runtime_get_output(void* runtime, int32_t index, DLTensor* tensor) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_GetOutput(graph_runtime, index, tensor);
}

TVM_DLL void tvm_runtime_get_output_raw(void* runtime, int32_t index, void* tensor_raw) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_GetOutputRaw(graph_runtime, index, tensor_raw);
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
  exit(-1);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLContext ctx, void** out_ptr) {
  tvm_crt_error_t err = g_memory_manager->Allocate(g_memory_manager, num_bytes, ctx, out_ptr);
  if (*out_ptr != NULL)  memset(*out_ptr, 0, num_bytes);
  //printf("###TVMPlatformMemoryAllocate 0x%x bytes, ptr=%p\n", num_bytes, *out_ptr);
  return err;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLContext ctx) {
  return g_memory_manager->Free(g_memory_manager, ptr, ctx);
}

tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorFunctionCallNotImplemented; }

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  return kTvmErrorFunctionCallNotImplemented;
}
