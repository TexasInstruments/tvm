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


/* This file provides TVM C runtime support for various 
   sort operations in TVM+TIDL generated code  */

#include <stdio.h>
#include <math.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include "c7x_tvm_runtime.h"
#include <c7x_scalable.h>

using namespace c7x;


template <typename DataType>
bool compare_descend(DataType a, DataType b) {
  return a > b;
}

template <typename DataType>
bool compare_ascend(DataType a, DataType b) {
  return a < b;
}


template <typename DataType, typename IndexType>
void argsort_impl(DataType* __restrict__ input, int *sort_num, IndexType* __restrict__ output, bool (*compare)(DataType, DataType)) {
  // auxiliary stack to avoid recursion (on C7x RTOS task)
  // Average computation-complexity case (equal partitions) actually has
  // the maximum stack depth, worst computation-complexity case (one element
  // in one partition, the rest in the other) has less stack depth.
  // Maximum stack depth is ceiling(log2(n))
  // int stack_size = ((int)log2f(*sort_num) + 1) * 2 * sizeof(int);
  // stack depth of 32 can sort 2^31 boxes (>> what's possible in real networks)
  int stack[32 * 2];

  int l = 0;
  int r = (*sort_num) - 1;

  // initialize the indices, sort indices along with value (argsort)
  for (int i = l; i <= r; i++)
    output[i] = i;

  int top = 0;
  stack[top++] = l;
  stack[top++] = r;

  while (top > 0)
  {
    r = stack[--top];
    l = stack[--top];

    // chunk sorted
    if (l >= r)
      continue;

    // pivot
    DataType pivot = input[r];
    IndexType   id    = output[r];

    // partition
    int cnt = l;
    for (int i = l; i < r; i++)
    {
      if (compare(input[i], pivot))
      {
        DataType ftmp = input[i];
        input[i] = input[cnt];
        input[cnt] = ftmp;
        int itmp = output[i];
        output[i] = output[cnt];
        output[cnt] = itmp;
        cnt++;
      }
    }
    input[r] = input[cnt];
    input[cnt] = pivot;
    output[r] = output[cnt];
    output[cnt] = id;

    // insert new chunks to stack, short one on top to reduce stack depth
    int len1 = cnt - l;
    int len2 = r - cnt;
    if (len2 > len1 && len2 > 1)
    {
      stack[top++] = cnt+1;
      stack[top++] = r;
    }
    if (len1 > 1)
    {
      stack[top++] = l;
      stack[top++] = cnt-1;
    }
    if (len2 <= len1 && len2 > 1)
    {
      stack[top++] = cnt+1;
      stack[top++] = r;
    }
  }
}

extern "C" void tvm_tidl_argsort_nms(float *input, int *sort_num, int *output) {
  argsort_impl(input, sort_num, output, compare_descend);
}

template<typename DataType, typename IndexType>
void argsort(DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend) {
  if (axis < 0) {
    axis = input->ndim + axis;
  }
  int64_t axis_mul_before = 1;
  int64_t axis_mul_after = 1;
  AllocDDRContext DDRContext;
  // adapted from runtime/contrib/sort/sort.cc:tvm.contrib.sort.argsort

  bool (*compare)(DataType, DataType);
  compare = is_ascend ? compare_ascend<DataType> : compare_descend<DataType>;

  DataType* __restrict__ in_scratch = (DataType *) DDRContext.allocate(input->shape[axis] * sizeof (DataType));
  IndexType* __restrict__ out_scratch = (IndexType *) DDRContext.allocate(input->shape[axis] * sizeof (IndexType));
  DataType* __restrict__ data = (DataType *) input->data;
  IndexType* __restrict__ out = (IndexType *) output->data;
  for (int64_t i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }
  SEConfig<DataType, 1> SE_Config(1, input->shape[axis], 1, 1, 1, 1, axis_mul_after, 0, 0, 0, 0);
  SAConfig<IndexType, 1> SA_Config(1, input->shape[axis], 1, 1, 1, 1, axis_mul_after, 0, 0, 0, 0);
  for (int i = 0; i < axis_mul_before; ++i) {
    for (int j = 0; j < axis_mul_after; ++j) {
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      __SE0_OPEN((void *)(data + base_idx), SE_Config.params());
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        in_scratch[k] = strm_eng<0, DataType>::get_adv();
      }
      __SE0_CLOSE();
      int sort_num = input->shape[axis];
      argsort_impl<DataType, IndexType>(in_scratch, &sort_num, out_scratch, compare);
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        out[full_idx] = out_scratch[k];
      }
    }
  }
}

extern "C" void tvm_contrib_sort_argsort(DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend) {
  DLDataType input_type = input->dtype;
  DLDataType output_type = output->dtype;
  if (input_type.code == kDLFloat && input_type.bits == 32) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      argsort<float, int32_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<float, int64_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      argsort<float, float>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<float, double>(input, output, axis, is_ascend);
    }
  } else if (input_type.code == kDLFloat && input_type.bits == 64) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      argsort<double, int32_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<double, int64_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      argsort<double, float>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<double, double>(input, output, axis, is_ascend);
    }
  } else if (input_type.code == kDLInt && input_type.bits == 32) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      argsort<int32_t, int32_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<int32_t, int64_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      argsort<int32_t, float>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<int32_t, double>(input, output, axis, is_ascend);
    }
  } else if (input_type.code == kDLInt && input_type.bits == 64) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      argsort<int64_t, int32_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<int64_t, int64_t>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      argsort<int64_t, float>(input, output, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      argsort<int64_t, double>(input, output, axis, is_ascend);
    }
  }
}

template <typename DataType, typename IndexType>
void topk_impl(
    std::pair<DataType, IndexType> * __restrict__ t_in, int sort_num, int k, bool is_ascend)
{
  AllocL2Context L2Context;
  int len = sort_num;

  typedef std::pair<DataType, IndexType> val_ind_t;
  
  // Use a heap to help keep the k largest values (and indices)
  val_ind_t * __restrict__ topk = (val_ind_t *) L2Context.allocate(k * sizeof(val_ind_t));

  // init the heap with first k val/ind
  for (int i = 0; i < k; i++)
  {
    topk[i] = t_in[i];

  }
  auto cmp = is_ascend ? [](val_ind_t &p1, val_ind_t &p2) { return p1.first < p2.first; }
                       : [](val_ind_t &p1, val_ind_t &p2) { return p1.first > p2.first; };
  auto cmp2 = is_ascend ? [](DataType p1, DataType p2) { return p1 > p2; }
                       : [](DataType p1, DataType p2) { return p1 < p2; };
  std::make_heap(topk, topk + k, cmp);

  // iterate through the rest of the val/ind
  // if new val is greater/less than current min/max in heap, replace min/max with new val, re-heapify
  DataType curr_minmax = topk[0].first;
  for (int i = k; i < len; i++)
  {
    DataType val = t_in[i].first;
    if (!cmp2(val, curr_minmax))
    {
      int curr = 0;
      while (1)
      {
        int minmax_ind = curr;
        int left  = 2 * curr + 1;
        int right = 2 * curr + 2;

        DataType minmax_val = val;
        if (left < k && cmp2(topk[left].first, minmax_val))
        {
          minmax_ind = left;
          minmax_val = topk[left].first;
        }
        if (right < k && cmp2(topk[right].first, minmax_val))
        {
          minmax_ind = right;
        }

        if (minmax_ind != curr)
        {
          topk[curr] = topk[minmax_ind];
          // recur and heapfiy on the child
          curr = minmax_ind;
        }
        else
        {
          break;
        }
      }
      topk[curr].first = val;
      topk[curr].second = i;
      curr_minmax = topk[0].first;
    }
  }
  __SE0_CLOSE();

  // save top k val/ind in the results
  #if 1  // disable this if top k values do not need to be sorted
  for (int i = 0; i < k-1; i++)
    std::pop_heap(topk, topk + k-i, cmp);
  #endif
  for (int i = 0; i < k; i++) {
    t_in[i] = topk[i];
  }
}

template <typename DataType, typename IndexType>
void topk(DLTensor* input, DLTensor* out_values, DLTensor* out_indices, int k, int axis, bool is_ascend) {
  AllocDDRContext DDRContext;
  // adapted from runtime/contrib/sort/sort.cc:tvm.contrib.sort.topk
  if (axis < 0) {
    axis = input->ndim + axis;
  }

  std::pair<DataType, IndexType>* __restrict__ scratch = 
    (std::pair<DataType, IndexType> *) DDRContext.allocate(
      input->shape[axis] * sizeof (std::pair<DataType, IndexType>)
      );
  DataType* __restrict__ data = (DataType *) input->data;
  DataType* __restrict__ values_ptr = out_values == nullptr ? nullptr : (DataType *) out_values->data;
  IndexType* __restrict__ indices_ptr = out_indices == nullptr ? nullptr : (IndexType *) out_indices->data;
  
  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }
  if (k < 1) {
    k = input->shape[axis];
  }

  // bool (*compare)(DataType, DataType);
  // compare = is_ascend ? compare_ascend<DataType> : compare_descend<DataType>;
  SEConfig<DataType, 1> SE_Config(1, input->shape[axis], 1, 1, 1, 1, axis_mul_after, 0, 0, 0, 0);

  for (int i = 0; i < axis_mul_before; ++i) {
    for (int j = 0; j < axis_mul_after; ++j) {
      int64_t src_base_idx = i * input->shape[axis] * axis_mul_after + j;
      int64_t dst_base_idx = i * k * axis_mul_after + j;
      #if 1
      __SE0_OPEN((void *)(data + src_base_idx), SE_Config.params());
      for (int64_t kk = 0; kk < input->shape[axis]; ++kk) {
        scratch[kk] = std::make_pair(strm_eng<0, DataType>::get_adv(), kk); 
      }
      __SE0_CLOSE();
      #else
      for (int64_t kk = 0; kk < input->shape[axis]; ++kk) {
        int64_t full_idx = src_base_idx + kk * axis_mul_after;
        in_scratch[kk] = data[full_idx];
      }
      #endif
      topk_impl<DataType, IndexType>(scratch, input->shape[axis], k, is_ascend);
      int64_t cnt = k > 0 ? k : input->shape[axis];
      for (int64_t kk = 0; kk < cnt; ++kk) {
        if (indices_ptr != nullptr) {
          indices_ptr[dst_base_idx + kk * axis_mul_after] = scratch[kk].second;
        }
        if (values_ptr != nullptr) {
          values_ptr[dst_base_idx + kk * axis_mul_after] = scratch[kk].first;
        }
      }
    }
  }

}
void topk_chooser(DLTensor* input, DLTensor* values_out, DLTensor* indices_out, int k, int axis, bool is_ascend) {
  DLDataType input_type = input->dtype;
  DLDataType output_type = indices_out == nullptr ? (DLDataType) {kDLInt, 32, 1} : indices_out->dtype;

  if (input_type.code == kDLFloat && input_type.bits == 32) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      topk<float, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<float, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      topk<float, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<float, double>(input, values_out, indices_out, k, axis, is_ascend);
    }
  } else if (input_type.code == kDLFloat && input_type.bits == 64) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      topk<double, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<double, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      topk<double, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<double, double>(input, values_out, indices_out, k, axis, is_ascend);
    }
  } else if (input_type.code == kDLInt && input_type.bits == 32) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      topk<int32_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<int32_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      topk<int32_t, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<int32_t, double>(input, values_out, indices_out, k, axis, is_ascend);
    }
  } else if (input_type.code == kDLInt && input_type.bits == 64) {
    if (output_type.code == kDLInt && output_type.bits == 32) {
      topk<int64_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<int64_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLFloat && output_type.bits == 32) {
      topk<int64_t, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (output_type.code == kDLInt && output_type.bits == 64) {
      topk<int64_t, double>(input, values_out, indices_out, k, axis, is_ascend);
    }
  }
}

extern "C" void tvm_contrib_sort_topk_both(DLTensor* input, DLTensor* values_out, DLTensor* indices_out, int k, int axis, char **ret_type, bool is_ascend) {
  topk_chooser(input, values_out, indices_out, k, axis, is_ascend);
}

extern "C" void tvm_contrib_sort_topk_indices(DLTensor* input, DLTensor* indices_out, int k, int axis, char **ret_type, bool is_ascend) {
  topk_chooser(input, nullptr, indices_out, k, axis, is_ascend);
}

extern "C" void tvm_contrib_sort_topk_values(DLTensor* input, DLTensor* values_out, int k, int axis, char **ret_type, bool is_ascend) {
  topk_chooser(input, values_out, nullptr, k, axis, is_ascend);
}
