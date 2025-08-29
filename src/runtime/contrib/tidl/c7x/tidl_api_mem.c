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

// A simple wrapper for malloc to collect allocation statistics
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(HOST_EMULATION)
   #include <malloc.h>
#endif
#include "tidl_api.h"
#include "tidl_api_mem.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

/* Align to 8 bytes to account for 64b data types */
#define L2_ALIGN_SIZE (8U)
#define L2_ALIGN_CEIL(VAL, ALIGN) ((((VAL)+(ALIGN)-1)/(ALIGN)) * (ALIGN))

static uint8_t *p_l2_scratch = NULL;
static int32_t l2_scratch_avail_size;

static uint8_t *p_ddr_scratch = NULL;
static int32_t ddr_scratch_avail_size;

static void   *g_ddr_scratch_mem_addr = NULL;
static int32_t g_ddr_scratch_mem_size = 0;

#if defined(HOST_EMULATION)
uint8_t g_l2_mem[L2_MEM_SIZE];
uint8_t g_l3_mem[L3_MEM_SIZE];
uint32_t g_l3_mem_size;
void    *g_l3_mem_addr;
#endif

EXTERN_C void tvm_tidl_l2_scratch_reset()
{
  #if defined(HOST_EMULATION)
  p_l2_scratch = &g_l2_mem[0];
  l2_scratch_avail_size = L2_MEM_SIZE;
  g_l3_mem_addr = (void *) &g_l3_mem[0];
  g_l3_mem_size = L3_MEM_SIZE;
  #else
  p_l2_scratch = (uint8_t *) g_l2_mem_addr;
  l2_scratch_avail_size = g_l2_mem_size;
  #endif
}

EXTERN_C uint8_t *tvm_tidl_l2_scratch_alloc(int32_t size)
{
  if (size <= 0 || l2_scratch_avail_size < size)  return NULL;

  uint8_t *alloc_ptr = p_l2_scratch;
  int32_t aligned_alloc_size = L2_ALIGN_CEIL(size, L2_ALIGN_SIZE);
  p_l2_scratch          += aligned_alloc_size;
  l2_scratch_avail_size -= aligned_alloc_size;
  return alloc_ptr;
}

EXTERN_C int32_t tvm_tidl_l2_scratch_avail_size()
{
  return l2_scratch_avail_size;
}

// This can be overriden to provide actual required ddr scratch mem size
//   e.g. from generated c7x code
EXTERN_C size_t __attribute__((weak)) get_ddr_scratch_mem_size()
{
  return 0;
}

EXTERN_C void tvm_tidl_ddr_scratch_set(void *ptr, size_t size)
{
  g_ddr_scratch_mem_addr = ptr;
  g_ddr_scratch_mem_size = size;
}

EXTERN_C void tvm_tidl_ddr_scratch_reset()
{
  p_ddr_scratch = (uint8_t *) g_ddr_scratch_mem_addr;
  ddr_scratch_avail_size = g_ddr_scratch_mem_size;
}

EXTERN_C uint8_t *tvm_tidl_ddr_scratch_alloc(int32_t size)
{
  if (size <= 0 || ddr_scratch_avail_size < size)  return NULL;

  uint8_t *alloc_ptr = p_ddr_scratch;
  int32_t aligned_alloc_size = L2_ALIGN_CEIL(size, L2_ALIGN_SIZE);
  p_ddr_scratch          += aligned_alloc_size;
  ddr_scratch_avail_size -= aligned_alloc_size;
  return alloc_ptr;
}

EXTERN_C int32_t tvm_tidl_ddr_scratch_avail_size()
{
  return ddr_scratch_avail_size;
}

EXTERN_C
void *tidl_malloc(size_t size)
{
#ifndef HOST_EMULATION
  void *ptr = appMemAlloc(APP_MEM_HEAP_DDR, size, L2_ALIGN_SIZE);
#else
  void *ptr = malloc(size);
#endif
  if (ptr != NULL)  memset(ptr, 0, size);
  return ptr;
}

EXTERN_C
void *tidl_memalign(size_t align, size_t size)
{
#ifndef HOST_EMULATION
  void *ptr = appMemAlloc(APP_MEM_HEAP_DDR, size, align);
#else
#if defined(MSVC_BUILD)
  void *ptr = _aligned_malloc(size, align);
#else
  void *ptr = memalign(align, size);
#endif
#endif
  if (ptr != NULL)  memset(ptr, 0, size);
  return ptr;
}

EXTERN_C
void tidl_free(void *ptr, size_t size)
{
  if (ptr == NULL)  return;
#ifndef HOST_EMULATION
  appMemFree(APP_MEM_HEAP_DDR, ptr, size);
#else
  free(ptr);
#endif
}

