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
#if (HOST_EMULATION)
   #include <malloc.h>
#endif
#include "tidl_api.h"
#include "tidl_api_mem.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#define L2_ALIGN_SIZE (128U)
#define L2_ALIGN_CEIL(VAL, ALIGN) ((((VAL)+(ALIGN)-1)/(ALIGN)) * (ALIGN))

static uint32_t malloc_size = 0;
static uint32_t malloc_requests = 0;

static uint8_t *p_l2_scratch = NULL;
static int32_t l2_scratch_avail_size;

EXTERN_C void tvm_tidl_l2_scratch_reset()
{
  #if (HOST_EMULATION)
  if (!p_l2_scratch)  p_l2_scratch = tidl_memalign(L2_ALIGN_SIZE, L2_MEM_SIZE);
  l2_scratch_avail_size = L2_MEM_SIZE;
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
  if (ptr != NULL)  memset(ptr, 0, size);
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

EXTERN_C
void tidl_malloc_report()
{
  printf("TIDL dynamic allocation: %u bytes in %d requests\n", 
     malloc_size, malloc_requests);
}
