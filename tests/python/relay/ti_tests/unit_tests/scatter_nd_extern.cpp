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
 * \file sigmoid_approx.cpp
 *
 * A simple C7x implementation of sigmoid using a lookup table
 * - Auto-generated look up table in sigmoid_approx.h has
 *   801 entries covering [0.0, 6.25] with step size 1.0/128
 *   and 0.002 precision
 * - sigmoid(x) = 1.0               for 6.25 < x 
 * - sigmoid(x) = lut[x*128]        for 0 <= x <= 6.25
 * - sigmoid(x) = 1.0 - sigmoid(-x) for x < 0
 */

#include <dlpack/dlpack.h>
#include <c7x_tvm_runtime.h>
#include <c7x_scalable.h>

using namespace c7x;


extern "C" int scatter_nd_ext(DLTensor *data, DLTensor *indices, DLTensor *updates, DLTensor *out)
{

  AllocL2Context L2Context;

  int data_shape = 1;
  for(int i = 0; i < data->ndim; i++) {
    data_shape *= data->shape[i];
  }
  SEConfig<float, 1> SE_Config0(64, 58080, 1, 1, 1, 1, 64, 0, 0, 0, 0);
  SAConfig<float, 1> SA_Config1(data_shape, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  SEConfig<float, 1> SE_Config2(data_shape, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
  __SA0_OPEN(SA_Config1.params());
  __SE0_OPEN((void *)(data->data), SE_Config2.params());
  for (int i = 0; i < data_shape; ++i) {
    *__SA0ADV(float, out->data) = __SE0ADV(float);
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

  for (int i = 0; i < fused_updates_dimension; ++i) {
    int offset = fused_data_dimension;
    int index = 0;
    for(int l = num_indices - 1; l >= 0; l--){
      index += offset * indices_ptr[i * num_indices + l];
      offset *= data->shape[l];
    }
    for (int j = 0; j < fused_data_dimension; ++j) {
      out_ptr[index + j] += updates_ptr[i * fused_data_dimension + j];
    }
  }

  return 0;
}

