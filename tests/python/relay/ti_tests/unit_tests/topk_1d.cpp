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

/*!
 * \file topk_1d.cpp
 *
 * A simple C7x example implementation of topk from n elements where
 * - (n >> k) and k val/ind fit in C7x local memory
 * - n elements can be treated as flattened 1-D tensor (i.e. axis ignored)
 *
 * The implementation uses a k-size min-root heap to keep the k largest values
 * in the heap when iterating through the input tensor, so that we do not
 * need to sort the whole tensor.
 * Depending on your n and k, you might want different implementation.
 *
 * Note: this is not a generic implementation that supports all topk cases.
 * For other cases, e.g. different axis/is_ascend/data type, this example
 * can be modified or generalized using template.
 */

#include <dlpack/dlpack.h>
#include <c7x_tvm_runtime.h>
#include <algorithm>


extern "C" int tvm_contrib_sort_topk_both(
    DLTensor* __restrict__ t_in, DLTensor* __restrict__ t_values, DLTensor* __restrict__ t_indices,
    int k, int axis, void* ret_type, bool is_ascend)
{
  AllocL2Context L2Context;

  // ignoring is_ascend, ret_type, axis
  // computing one dimensional flattened topk biggest values
  float * __restrict__ in  = (float *) t_in->data;
  float * __restrict__ values = nullptr;
  if (t_values != nullptr)  values = (float *) t_values->data;
  int * __restrict__ indices = nullptr;
  if (t_indices != nullptr)  indices = (int *) t_indices->data;

  int len = 1;
  for (int i = 0; i < t_in->ndim; i++)
    len *= t_in->shape[i];

  // Use a min-root heap to help keep the k largest values (and indices)
  typedef struct { float val; int ind; } val_ind_t;
  val_ind_t * __restrict__ topk = (val_ind_t *) L2Context.allocate(k * sizeof(val_ind_t));

  SEConfig<float, 1> SE_Config1(len, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  __SE0_OPEN((void *)(in), SE_Config1.params());

  // init the heap with first k val/ind
  for (int i = 0; i < k; i++)
  {
    topk[i].val = __SE0ADV(float);
    topk[i].ind = i;
  }
  auto cmp = [](val_ind_t &p1, val_ind_t &p2) { return p1.val > p2.val; };
  std::make_heap(topk, topk + k, cmp);

  // iterate through the rest of the val/ind
  // if new val is greater than current min in heap, replace min with new val, re-heapify
  float curr_min = topk[0].val;
  for (int i = k; i < len; i++)
  {
    float val = __SE0ADV(float);
    if (val > curr_min)
    {
      int curr = 0;
      while (1)
      {
        int smallest = curr;
        int left  = 2 * curr + 1;
        int right = 2 * curr + 2;

        float smallest_val = val;
        if (left < k && topk[left].val < smallest_val)
        {
          smallest = left;
          smallest_val = topk[left].val;
        }
        if (right < k && topk[right].val < smallest_val)
        {
          smallest = right;
        }

        if (smallest != curr)
        {
          topk[curr] = topk[smallest];
          // recur and heapfiy on the child
          curr = smallest;
        }
        else
        {
          break;
        }
      }
      topk[curr].val = val;
      topk[curr].ind = i;
      curr_min = topk[0].val;
    }
  }
  __SE0_CLOSE();

  // save top k val/ind in the results (note the min-root heap)
  #if 1  // disable this if top k values do not need to be sorted
  for (int i = 0; i < k-1; i++)
    std::pop_heap(topk, topk + k-i, cmp);
  #endif
  if (values != nullptr)
  {
    for (int i = 0; i < k; i++)
      values[i] = topk[i].val;
  }
  if (indices != nullptr)
  {
    for (int i = 0; i < k; i++)
      indices[i] = topk[i].ind;
  }

  return 0;
}


extern "C" int tvm_contrib_sort_topk_indices(
    DLTensor* __restrict__ t_in, DLTensor* __restrict__ t_indices,
    int k, int axis, void* ret_type, bool is_ascend)
{
  return tvm_contrib_sort_topk_both(t_in, nullptr, t_indices, k, axis, ret_type, is_ascend);
}


extern "C" int tvm_contrib_sort_topk_values(
    DLTensor* __restrict__ t_in, DLTensor* __restrict__ t_values,
    int k, int axis, void* ret_type, bool is_ascend)
{
  return tvm_contrib_sort_topk_both(t_in, t_values, nullptr, k, axis, ret_type, is_ascend);
}
