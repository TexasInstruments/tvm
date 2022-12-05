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
#include "sigmoid_approx.h"
#include <c7x_scalable.h>

using namespace c7x;


extern "C" int sigmoid_approx(DLTensor *t_in, DLTensor *t_out)
{
  AllocL2Context L2Context;

  // load look up table from ddr into L2
  // use streaming engine to load a vector of veclen floats at a time, then store into L2
  float * __restrict__ local_lut = (float *) L2Context.allocate(sigmoid_lut_len * sizeof(float));
  const int veclen = max_simd<float>::value;
  SEConfig<float, veclen> SE_Config1(sigmoid_lut_len, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0);
  SAConfig<float, veclen> SA_Config1(sigmoid_lut_len, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0);
  __SE0_OPEN((void *)(sigmoid_lut), SE_Config1.params());
  __SA0_OPEN(SA_Config1.params());
  for (int i = 0; i < (sigmoid_lut_len + veclen - 1)/ veclen;  i++)
  {
    float_vec value = strm_eng<0, float_vec>::get_adv(); 
    __vpred pred = strm_agen<0, float_vec>::get_vpred();
    float_vec* addr = strm_agen<0, float_vec>::get_adv(local_lut);
    __vstore_pred(pred, addr, value);
  }
  __SA0_CLOSE();
  __SE0_CLOSE();

  // setup
  float * __restrict__ in  = (float *) t_in->data;
  float * __restrict__ out = (float *) t_out->data;
  int len = 1;
  for (int i = 0; i < t_in->ndim; i++)
    len *= t_in->shape[i];

  // compute sigmoid, scalar operation (ii=3)
  // use streaming engine to load one float value at a time, then compute and store into results
  SEConfig<float, 1> SE_Config2(len, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0);
  SAConfig<float, 1> SA_Config2(len, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0);
  __SE0_OPEN((void *)(in), SE_Config2.params());
  __SA0_OPEN(SA_Config2.params());
  for (int i = 0; i < len; i++)
  {
    float x = __SE0ADV(float);
    float y = (x < 0) ? -x : x;
    int index = __min((int) (y * sigmoid_lut_steps_in_one), (int)(sigmoid_lut_len - 1));
    float z = local_lut[index];
    if (x < 0) z = 1.0f - z;
    *__SA0ADV(float, out) = z;
  }
  __SA0_CLOSE();
  __SE0_CLOSE();

  return 0;
}

