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


/* This file provides TVM Runtime Trace capability.  It can be
   included in TVM C++ graph_executor runtime on Arm
   or TVM C graph_executor runtime on C7x.

   Use (#define TVM_RT_TRACE_CRT) to specify C runtime, otherwise,
   it is for C++ runtime.
*/

#include <float.h>

/* A simple TVM trace:

TRACE -> total_size (int32_t) VERSION (int32_t) CORE (int32_t) (NODE)* END_GRAPH (int32_t)
NODE -> node_id (int32_t) node_name (str) TIME (uint64_t) (OUTPUT)* END_NODE (int32_t)
TIME -> cycles (on C7x) | nanoseconds (on Arm)
OUTPUT -> out_id ndim type_code elem_bytes num_elements min max sum fh_sum REAL_TENSOR
REAL_TENSOR -> 0
             | num_tensor_values (float)*
*/
#define TVM_RT_TRACE_VERSION   0x20220728
#define TVM_RT_TRACE_END_NODE  0xF000E001
#define TVM_RT_TRACE_END_GRAPH 0xF000E002
#define TVM_RT_DEVICE_J7       0x0
#define TVM_RT_CORE_ARM        0x0
#define TVM_RT_CORE_C7X        0x1
#define TVM_RT_TRACE_DEFAULT_SIZE (2 * 1024 * 1024)

static uint8_t *tvm_rt_trace_ptr = NULL;
static int32_t  tvm_rt_trace_size = 0;
static int32_t  tvm_rt_trace_cur_size = 0;

#if defined(TVM_RT_TRACE_CRT)
  #define TVM_RT_PREFIX "TVM CRT: "
  extern int32_t tvm_rt_get_debug_level();
  extern void*   tvm_rt_get_trace_ptr();
  extern int32_t tvm_rt_get_trace_size();
  extern int32_t tvm_rt_get_trace_node();
#else
  #define TVM_RT_PREFIX "TVM RT: "
#endif

#if defined(__C7000__)
  #include <c7x.h>
  #define _TSC_read() (__TSC)
#else
  #include <time.h>
  static uint64_t _TSC_read()
  {
    struct timespec tp;
    if (clock_gettime(CLOCK_MONOTONIC, &tp) != 0)
      clock_gettime(CLOCK_REALTIME, &tp);
    return ((uint64_t)tp.tv_nsec + (uint64_t)tp.tv_sec * 1e9);  /* nanoseconds */
  }
#endif


static void tvm_rt_trace_write_nbytes(void *pv, int32_t len)
{
  if (tvm_rt_trace_ptr != NULL && tvm_rt_trace_cur_size + len < tvm_rt_trace_size)
  {
    memcpy(tvm_rt_trace_ptr + tvm_rt_trace_cur_size, pv, len);
    tvm_rt_trace_cur_size += len;
  }
  else
    tvm_rt_trace_size = 0;  /* trace buffer full, no more writing */
}
static void tvm_rt_trace_write_int(int32_t v)
{
  tvm_rt_trace_write_nbytes(&v, sizeof(int32_t));
}
static void tvm_rt_trace_write_float(float v)
{
  tvm_rt_trace_write_nbytes(&v, sizeof(float));
}
static void tvm_rt_trace_write_str(const char *str)
{
  tvm_rt_trace_write_nbytes((void*)str, strlen(str)+1);
}

static void tvm_rt_trace_init()
{
  #if defined(TVM_RT_TRACE_CRT)
  tvm_rt_trace_size = tvm_rt_get_trace_size();
  tvm_rt_trace_ptr = (uint8_t*) tvm_rt_get_trace_ptr();
  #else
  tvm_rt_trace_size = TVM_RT_TRACE_DEFAULT_SIZE;
  if (char *env_var = getenv("TVM_RT_TRACE_SIZE"))
    tvm_rt_trace_size = atoi(env_var);
  tvm_rt_trace_ptr = new uint8_t[tvm_rt_trace_size];
  #endif

  tvm_rt_trace_cur_size = 0;
  tvm_rt_trace_write_int(16); /* placeholder for total written size */
  tvm_rt_trace_write_int(TVM_RT_TRACE_VERSION);
  tvm_rt_trace_write_int(TVM_RT_DEVICE_J7);
  #if defined(__C7000__)
  tvm_rt_trace_write_int(TVM_RT_CORE_C7X);
  #else
  tvm_rt_trace_write_int(TVM_RT_CORE_ARM);
  #endif
}

static void tvm_rt_trace_node_begin(int32_t idx, const char* name, uint64_t node_time)
{
  tvm_rt_trace_write_int(idx);
  tvm_rt_trace_write_str(name);
  tvm_rt_trace_write_nbytes(&node_time, sizeof(uint64_t));
}

static void tvm_rt_trace_finalize(uint64_t graph_time)
{
  tvm_rt_trace_write_int(-1);
  tvm_rt_trace_write_str("Graph");
  tvm_rt_trace_write_nbytes(&graph_time, sizeof(uint64_t));
  tvm_rt_trace_write_int(TVM_RT_TRACE_END_NODE);
  tvm_rt_trace_write_int(TVM_RT_TRACE_END_GRAPH);

  if (tvm_rt_trace_ptr != NULL)
    memcpy(tvm_rt_trace_ptr, &tvm_rt_trace_cur_size, sizeof(int32_t));

  #if !defined(TVM_RT_TRACE_CRT)
  if (FILE *tf = fopen("./tvm_arm.trace", "wb"))
  {
    fwrite(tvm_rt_trace_ptr, tvm_rt_trace_cur_size, 1, tf);
    fclose(tf);
  }
  delete [] tvm_rt_trace_ptr;
  #endif
}

static float get_tensor_data(const DLTensor *tensor, int i)
{
  if (tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32)
    return ((float *)tensor->data)[i];
  if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 64)
    return ((int64_t *)tensor->data)[i];
  if (tensor->dtype.code == kDLUInt && tensor->dtype.bits == 64)
    return ((uint64_t *)tensor->data)[i];
  if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 32)
    return ((int *)tensor->data)[i];
  if (tensor->dtype.code == kDLUInt && tensor->dtype.bits == 32)
    return ((unsigned int *)tensor->data)[i];
  if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 16)
    return ((short *)tensor->data)[i];
  if (tensor->dtype.code == kDLUInt && tensor->dtype.bits == 16)
    return ((unsigned short *)tensor->data)[i];
  if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 8)
    return ((char *)tensor->data)[i];
  if (tensor->dtype.code == kDLUInt && tensor->dtype.bits == 8)
    return ((unsigned char *)tensor->data)[i];

  return -FLT_MAX;
}

static void tvm_rt_trace_write_tensor(const DLTensor *tensor, int id, int debug_level, int dump_all)
{
  int32_t elem_bytes = tensor->dtype.bits / 8;
  int32_t size = (int32_t) Shape_Accumulate(tensor->shape, tensor->ndim);
  if (debug_level > 3)
    printf(TVM_RT_PREFIX " Out[%d]: ndim=%d, type_code=%d, elem_bytes=%d, num_elements=%d\n",
           id, tensor->ndim, tensor->dtype.code, elem_bytes, size);
  tvm_rt_trace_write_int(id);
  tvm_rt_trace_write_int(tensor->ndim);
  tvm_rt_trace_write_int(tensor->dtype.code);
  tvm_rt_trace_write_int(elem_bytes);
  tvm_rt_trace_write_int(size);
  if (get_tensor_data(tensor, 0) != -FLT_MAX)
  {
    float minval =  FLT_MAX;
    float maxval = -FLT_MAX;
    float sum = 0.0f;
    float h_sum = 0.0f;
    int32_t h_size = size / 2;
    for (int i = 0; i < size; i++)
    {
      float val = get_tensor_data(tensor, i);
      if (val < minval)  minval = val;
      if (val > maxval)  maxval = val;
      sum += val;
      if (i == h_size)  h_sum = sum;
    }
    if (debug_level > 3)
      printf(TVM_RT_PREFIX "         min=%f, max=%f, sum=%f (fh=%f)\n", minval, maxval, sum, h_sum);
    tvm_rt_trace_write_float(minval);
    tvm_rt_trace_write_float(maxval);
    tvm_rt_trace_write_float(sum);
    tvm_rt_trace_write_float(h_sum);

    /* Optionally dump all output tensor values for a specified node */
    if (dump_all)
    {
      if (tvm_rt_trace_cur_size + (size+1) * (int)sizeof(float) > tvm_rt_trace_size)
      {
        printf(TVM_RT_PREFIX "Not enough trace memory for dumping output [%d]\n", id);
        size = 0;
      }
      tvm_rt_trace_write_int(size);
      for (int i = 0; i < size; i++)
        tvm_rt_trace_write_float(get_tensor_data(tensor, i));
    }
    else
    {
      tvm_rt_trace_write_int(0);
    }
  }
}

