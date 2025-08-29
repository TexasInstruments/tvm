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

#ifndef _TIDL_API_MEM_H
#define _TIDL_API_MEM_H

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(HOST_EMULATION)
  #include <malloc.h>
  #define EXTRA_MEM_FOR_ALIGN (1024)
  #define L1_TOTAL_MEMORY_SIZE  (16 * 1024)
  #define L2_TOTAL_MEMORY_SIZE  (512 * 1024)
  #define L3_TOTAL_MEMORY_SIZE  (8 * 1024 * 1024)
  #define L1_MEM_SIZE  (16*1024 +  EXTRA_MEM_FOR_ALIGN)
  #define L2_MEM_SIZE  (448*1024+  EXTRA_MEM_FOR_ALIGN)
  #define L3_MEM_SIZE  (7968 * 1024)
  #define L4_MEM_SIZE  (1.5*1024 * 1024 * 1024)
#endif

extern void *g_l1_mem_addr;
extern void *g_l2_mem_addr;
extern void *g_l3_mem_addr;
extern uint32_t g_l1_mem_size;
extern uint32_t g_l2_mem_size;
extern uint32_t g_l3_mem_size;

extern void     tvm_tidl_l2_scratch_reset();
extern uint8_t* tvm_tidl_l2_scratch_alloc(int32_t size);
extern int32_t  tvm_tidl_l2_scratch_avail_size();

extern void     tvm_tidl_ddr_scratch_set(void *ptr, size_t size);
extern void     tvm_tidl_ddr_scratch_reset();
extern uint8_t* tvm_tidl_ddr_scratch_alloc(int32_t size);
extern int32_t  tvm_tidl_ddr_scratch_avail_size();

//---------------------------------------------------------------------
// This is an auxilliary API for handling memory allocation requests
// from TIDL, with usage tracking.
extern void *tidl_malloc(size_t size);
extern void *tidl_memalign(size_t alignment, size_t size);
extern void tidl_free(void *ptr, size_t size);

//---------------------------------------------------------------------
// This is for integration into PSDK C7x firmware
//   appMem*() routines are from PSDK RTOS vision_apps
#ifndef HOST_EMULATION
/** \brief Heap located in DDR */
#define APP_MEM_HEAP_DDR (0u)
/** \brief Heap located in DDR and is used as scratch */
#define APP_MEM_HEAP_DDR_SCRATCH (4u)

extern void *appMemAlloc(uint32_t heap_id, uint32_t size, uint32_t align);
extern int32_t appMemFree(uint32_t heap_id, void *ptr, uint32_t size);
#endif


#ifdef __cplusplus
}
#endif

#endif  // _TIDL_API_MEM_H
