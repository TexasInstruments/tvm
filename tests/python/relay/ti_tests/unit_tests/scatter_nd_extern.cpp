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
 * \file scatter_nd_extern.cpp
 *
 * A simple C7x implementation of scatter_nd
 * - Leverage DMA and L2 local storage
 */

#include <dlpack/dlpack.h>
#include <c7x_tvm_runtime.h>
#include <c7x_scalable.h>

using namespace c7x;

class Tiler1D {
 public:
  Tiler1D(int total, int num_blocks) : total_(total), num_blocks_(num_blocks), curr_block_(0) {
    block_size_ = total_ / num_blocks_;
    if (total_ % num_blocks_ != 0)  num_blocks_ += 1;
  }
  void next_block() { curr_block_ += 1; }
  bool done() { return curr_block_ >= num_blocks_; }
  bool is_last_block() { return curr_block_ >= num_blocks_ - 1; }
  int curr_block_begin() { return curr_block_ * block_size_; }
  int curr_block_size() { return is_last_block() ? (total_ - curr_block_begin()) : block_size_; }
  int regular_block_size() { return block_size_; }
  int real_num_blocks() { return num_blocks_; }
 private:
  int total_;
  int num_blocks_;
  int curr_block_;
  int block_size_;
};


void copy_data(float *from, float *to, int size)
{
  const int veclen = max_simd<float>::value;
  SAConfig<float, veclen> SA_Config1(size, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SEConfig<float, veclen> SE_Config2(size, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  __SA0_OPEN(SA_Config1.params());
  __SE0_OPEN((void *)(from), SE_Config2.params());
  for (int i = 0; i < (size + veclen - 1) / veclen; ++i) {
    float_vec value = strm_eng<0, float_vec>::get_adv();
    __vpred pred = strm_agen<0, float_vec>::get_vpred();
    float_vec* addr = strm_agen<0, float_vec>::get_adv(to);
    __vstore_pred(pred, addr, value);
  }
  __SA0_CLOSE();
  __SE0_CLOSE();
}


void update_local(float *local, int begin, int end, int *indices_ptr, float *updates_ptr,
                  int fused_updates_dimension, int fused_data_dimension,
                  int num_indices, int indices_offsets[])
{
  const int veclen = max_simd<float>::value;
  int vec_slice_iters = (fused_data_dimension + veclen - 1) / veclen;

  float_vec dummy;
  float *local_slice;
  SAConfig<int, 1>        SA_Config3(fused_updates_dimension * num_indices, 1, 1, 1, 1, 1,
		                     0, 0, 0, 0, 0);
  SEConfig<float, veclen> SE_Config4(fused_data_dimension, fused_updates_dimension, 1, 1, 1, 1,
		                     fused_data_dimension, 0, 0, 0, 0);
  SAConfig<float, veclen> SA_Config5(fused_data_dimension, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SEConfig<float, veclen> SE_Config6(fused_data_dimension, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SAConfig<float, veclen> SA_Config7(veclen, vec_slice_iters, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SEConfig<float, veclen> SE_Config8(veclen, vec_slice_iters, 1, 1, 1, 1, 0, 0, 0, 0, 0);

  __SA1_OPEN(SA_Config3.params());
  __SE1_OPEN((void *)(updates_ptr), SE_Config4.params());
  for (int i = 0; i < fused_updates_dimension; ++i) {
    int index = 0;
    if (num_indices == 1) {
      index = *strm_agen<1, int>::get_adv(indices_ptr);
    } else {
      for (int l = 0; l < num_indices; l++) {
        index += indices_offsets[l] * (*strm_agen<1, int>::get_adv(indices_ptr));
      }
    }

    if (begin <= index && index < end) {
      local_slice = local + (index-begin)*fused_data_dimension;
      __SA0_OPEN(SA_Config5.params());
      __SE0_OPEN(local_slice, SE_Config6.params());
    } else {
      local_slice = (float *) &dummy;
      __SA0_OPEN(SA_Config7.params());
      __SE0_OPEN(local_slice, SE_Config8.params());
    }

    for (int j = 0; j < vec_slice_iters; ++j) {
      float_vec data   = strm_eng<0, float_vec>::get_adv();
      float_vec update = strm_eng<1, float_vec>::get_adv();
      __vpred pred     = strm_agen<0, float_vec>::get_vpred();
      float_vec* addr  = strm_agen<0, float_vec>::get_adv(local_slice);
      __vstore_pred(pred, addr, data + update);
    }
    __SA0_CLOSE();
    __SE0_CLOSE();
  }
  __SA1_CLOSE();
  __SE1_CLOSE();
}


extern "C" int scatter_nd_ext(DLTensor *data, DLTensor *indices, DLTensor *updates, DLTensor *out)
{
  AllocL2Context L2Context;
  int32_t l2_size = L2Context.avail_size();

  int data_shape = 1;
  for(int i = 0; i < data->ndim; i++) {
    data_shape *= data->shape[i];
  }

  int* indices_ptr = (int*) indices->data;
  float* data_ptr = (float*) data->data;
  float* updates_ptr = (float*) updates->data;
  float* out_ptr = (float*) out->data;

  int num_indices = indices->shape[indices->ndim - 1];
  int fused_updates_dimension = 1;
  for (int i = 0; i < indices->ndim - 1; i++) {
    fused_updates_dimension *= updates->shape[i];
  }

  int outer_data_dimension = 1;
  for(int i = 0; i < num_indices; i++) {
    outer_data_dimension *= data->shape[i];
  }

  int fused_data_dimension = 1;
  for(int i = num_indices; i < data->ndim; i++) {
    fused_data_dimension *= data->shape[i];
  }

  int indices_offsets[4];
  indices_offsets[num_indices - 1] = 1;
  for (int l = num_indices - 2; l >= 0; l--) {
    indices_offsets[l] = data->shape[l] * indices_offsets[l+1];
  }

  int32_t num_blocks = 1;
  float *local_data = NULL;
  uint32_t data_size_bytes = outer_data_dimension * fused_data_dimension * sizeof(float);
  if (data_size_bytes <= l2_size) {  /* performing update to data in L2 */
    local_data = (float *) L2Context.allocate(data_size_bytes);
  } else {  /* performing update to data in L3/MSMC */
    extern uint32_t g_l3_mem_size;
    extern void *   g_l3_mem_addr;
    while (data_size_bytes / num_blocks > g_l3_mem_size)
      num_blocks += 1;
    local_data = (float *) g_l3_mem_addr;
  }

  Tiler1D tiler(outer_data_dimension, num_blocks);
  for(; !tiler.done(); tiler.next_block()) {
    int begin = tiler.curr_block_begin();
    int end   = begin + tiler.curr_block_size();
    int data_size = (end - begin) * fused_data_dimension;

    copy_data(data_ptr + begin * fused_data_dimension, local_data, data_size);
    update_local(local_data, begin, end, indices_ptr, updates_ptr,
                 fused_updates_dimension, fused_data_dimension, num_indices, indices_offsets);
    copy_data(local_data, out_ptr + begin * fused_data_dimension, data_size);
  }

  return 0;
}


#if 0
extern "C" int scatter_nd_ext(DLTensor *data, DLTensor *indices, DLTensor *updates, DLTensor *out)
{
  AllocL2Context L2Context;

  int data_shape = 1;
  for(int i = 0; i < data->ndim; i++) {
    data_shape *= data->shape[i];
  }

  const int veclen = max_simd<float>::value;
  SAConfig<float, veclen> SA_Config1(data_shape, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SEConfig<float, veclen> SE_Config2(data_shape, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  __SA0_OPEN(SA_Config1.params());
  __SE0_OPEN((void *)(data->data), SE_Config2.params());
  for (int i = 0; i < (data_shape + veclen - 1) / veclen; ++i) {
    float_vec value = strm_eng<0, float_vec>::get_adv();
    __vpred pred = strm_agen<0, float_vec>::get_vpred();
    float_vec* addr = strm_agen<0, float_vec>::get_adv(out->data);
    __vstore_pred(pred, addr, value);
  }
  __SA0_CLOSE();
  __SE0_CLOSE();

  int* indices_ptr = (int*) indices->data;
  float* data_ptr = (float*) data->data;
  float* updates_ptr = (float*) updates->data;
  float* out_ptr = (float*) out->data;

  int num_indices = indices->shape[indices->ndim - 1];
  int fused_updates_dimension = 1;
  for (int i = 0; i < indices->ndim - 1; i++) {
    fused_updates_dimension *= updates->shape[i];
  }

  int fused_data_dimension = 1;
  for(int i = num_indices; i < data->ndim; i++) {
    fused_data_dimension *= data->shape[i];
  }

  int offsets[4];
  offsets[num_indices - 1] = fused_data_dimension;
  for (int l = num_indices - 2; l >= 0; l--) {
    offsets[l] = data->shape[l] * offsets[l+1];
  }

  SEConfig<int, 1> SE_Config3(fused_updates_dimension * num_indices, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SEConfig<float, 1> SE_Config4(fused_updates_dimension * fused_data_dimension, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  __SE0_OPEN((void *)(indices_ptr), SE_Config3.params());
  __SE1_OPEN((void *)(updates_ptr), SE_Config4.params());
  for (int i = 0; i < fused_updates_dimension; ++i) {
    int index = 0;
    for (int l = 0; l < num_indices; l++) {
      index += offsets[l] * strm_eng<0, int>::get_adv();
    }

    for (int j = 0; j < fused_data_dimension; ++j) {
      out_ptr[index + j] += strm_eng<1, float>::get_adv();
    }
  }
  __SE0_CLOSE();
  __SE1_CLOSE();

  return 0;
}
#endif

