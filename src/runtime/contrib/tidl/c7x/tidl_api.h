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

/*------------------------------------------------------------------------------*/
// TIDL_API.H
//   This file defines a simple interface for a client application to
//   instantiate and invoke TIDL.
/*------------------------------------------------------------------------------*/
#ifndef TIDL_API_H_
#define TIDL_API_H_

#include "dlpack/dlpack.h"

#ifdef __cplusplus
extern "C" {
#endif
//---------------------------------------------------------------------
// Instantiate a TIDL graph
extern void* init_tidl_subgraph(void *Network,
                                uint32_t network_size,
				void *IOParams,
				void *udmaDrvObjPtr,
                                int   is_nchw);

// Invoke a TIDL graph
extern int32_t process_tidl_subgraph(void *instance,
				     DLTensor* in_tensors[],
				     DLTensor* out_tensors[]);

// Free TIDL graph
extern int32_t free_tidl_subgraph(void *instance);

//---------------------------------------------------------------------
// This is an auxilliary API for handling memory allocation requests
// from TIDL, with usage tracking.
extern void *tidl_malloc(size_t size);
extern void *tidl_memalign(size_t alignment, size_t size);
extern void tidl_free(void *ptr, size_t size);
extern void tidl_malloc_report();

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

#endif
