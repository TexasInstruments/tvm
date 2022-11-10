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
 * \file resize_nchw_1x2.cpp
 *
 * A simple C7x implementation of 1x2 upsampling/resizing
 * - NCHW layout, half_pixel coordinate mode, linear method
 * - input_height == output_height, input_width * 2 == output_width
 * - one line of input image and output image fits in L2 scratch memory
 */
#define V1 (0x1)
#define V2 (0x2)
#define VSELECTED V2


#include <dlpack/dlpack.h>
#include <c7x_tvm_runtime.h>


extern "C" int resize_nchw_1x2(DLTensor * __restrict__ t_in, DLTensor * __restrict__ t_out)
{
  // setup
  float * __restrict__ in  = (float *) t_in->data;
  float * __restrict__ out = (float *) t_out->data;
  int N = t_in->shape[0];
  int C = t_in->shape[1];
  int H = t_in->shape[2];
  int W = t_in->shape[3];
  int CHW = C * H * W;
  int CHW2 = C * H * W * 2;
  int HW  = H * W;
  int HW2  = H * W * 2;
  int W2 = W * 2;

  // Version 1, natural C code, 3246 us
  // half-pixel coordinate mode, linear interpolation method
  // each output pixel is an interpolation between two input pixels
#if (VSELECTED & (V1))
  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h = 0; h < H; h++)
      {
        out[n*CHW2 + c*HW2 + h*W2 + 0] = in[n*CHW + c*HW + h*W + 0];

        // software pipelining, ii=11 for unroll factor 8, ii=3 for peeled scalar loop
        //   i.e. in the unrolled loop, we output 16 pixels for every 11 cycle, of course
        //   the pipeline will be stalled when there are cache misses
        #pragma UNROLL(8)
        for (int w = 0; w < W-1; w++)
        {
          float p0 = in[n*CHW + c*HW + h*W + w  ];
          float p1 = in[n*CHW + c*HW + h*W + w+1];
          out[n*CHW2 + c*HW2 + h*W2 + 2*w+1] = 0.75f * p0 + 0.25f * p1;
          out[n*CHW2 + c*HW2 + h*W2 + 2*w+2] = 0.25f * p0 + 0.75f * p1;
        }

        out[n*CHW2 + c*HW2 + h*W2 + 2*W-1] = in[n*CHW + c*HW + h*W + W-1];
      }
    }
  }
#endif

  // Version 2, streaming engine (SE) and streaming address (SA) generator code, 861 us
  // In this version, we use the streaming engine to bring the input pixels in.
  // Note we that provide a simple template wrapper in c7x_tvm_runtime.h to help setup
  // SE/SA parameters (ICNTs and DIMs).  Details on SE/SA please refer to "C7000 Optimizing
  // C/C++ Compiler user's guide".  In short, SE leverages the access pattern provided
  // by user (ICNTs, DIMs) and prefetches data in the background into special registers
  // for C7x instructions to use.  SA leverages the access pattern to precompute the
  // address in the background for C7x instructions to use.
  // 
  //   There are N*CHW input pixels, we configure a streaming engine to (pre-)fetch
  //   one pixel at a time,
  //   - The first __SE0 will fetch the first pixel in a row
  //   - The next __SE0ADV fetches the current pixel and then moves/pops the SE to next pixel
  //   - The next __SE0 fetches the current pixel
  //     - note this is the same pixel that __SE0ADV will fetch in the next iteration
  //   - The last __SE0ADV will fetch the last pixel in a row and then moves/pops the SE
  //       to the first pixel in the next row
  //   There are N*CHW2 output pixels, we configure a streaming address generator to
  //   pre-compute the address,
  //   - __SA0ADV will provide the current pixel address and then move/pop the SA to the
  //     address of next pixel
#if (VSELECTED & (V2))
  SEConfig<float, 1> SE_Config2(N*CHW,  1, 1, 1, 1, 1,  0, 0, 0, 0, 0);
  SAConfig<float, 1> SA_Config2(N*CHW2, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0);
  __SE0_OPEN((void *)(in), SE_Config2.params());
  __SA0_OPEN(SA_Config2.params());
  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h = 0; h < H; h++)
      {
        *__SA0ADV(float, out) = __SE0(float);

        // software pipelining, ii=2
        //   i.e. we are producing 2 output pixels in every 2 cycles, SE prefetches data
        //   and can effectively hide the memory access latency
        for (int w = 0; w < W-1; w++)
        {
          float p0 = __SE0ADV(float);
          float p1 = __SE0(float);
          *__SA0ADV(float, out) = 0.75f * p0 + 0.25f * p1;
          *__SA0ADV(float, out) = 0.25f * p0 + 0.75f * p1;
        }

        *__SA0ADV(float, out) = __SE0ADV(float);
      }
    }
  }
  __SA0_CLOSE();
  __SE0_CLOSE();
#endif

  return 0;
}

