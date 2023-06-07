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

/* Need to be in sync with tvm_tidl_rt_info defined in tiovx/include/TI/j7_tvm.h */
typedef struct {
  int32_t  tvm_rt_debug_level;
  int32_t  tidl_trace_log_level;
  int32_t  tidl_trace_write_level;
  float    max_preempt_delay;
  uint64_t tvm_rt_trace_ptr;
  int32_t  tvm_rt_trace_size;
  int32_t  tvm_rt_trace_node;
  int32_t  tvm_rt_target_priority;
  int32_t  tvm_rt_core_num;
} tvm_tidl_rt_info;

//---------------------------------------------------------------------
// Instantiate a TIDL graph
extern void* init_tidl_subgraph(void *Network,
                                uint32_t network_size,
				void *IOParams,
				void *udmaDrvObjPtr,
                                int   is_nchw,
                                void *rt_info);

// Invoke a TIDL graph
extern int32_t process_tidl_subgraph(void *instance,
				     DLTensor* in_tensors[],
				     DLTensor* out_tensors[]);

// Free TIDL graph
extern int32_t free_tidl_subgraph(void *instance);

#ifdef __cplusplus
}
#endif

#endif
