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


/* This file provides TVM C runtime support for argsort_nms
   in TVM+TIDL generated code for object-detection models */

#include <stdio.h>
#include <math.h>

extern void *tidl_malloc(size_t size);
extern void tidl_free(void *ptr, size_t size);

/* Argsort on flattened input, sorted result in Descending order */
void tvm_tidl_argsort_nms(float *input, int *sort_num, int *output)
{
  int l = 0;
  int r = (*sort_num) - 1;

  // initialize the indices, sort indices along with value (argsort)
  for (int i = l; i <= r; i++)
    output[i] = i;

  // auxiliary stack to avoid recursion (on C7x RTOS task)
  // Average computation-complexity case (equal partitions) actually has
  // the maximum stack depth, worst computation-complexity case (one element
  // in one partition, the rest in the other) has less stack depth.
  // Maximum stack depth is ceiling(log2(n))
  int stack_size = ((int)log2f(*sort_num) + 1) * 2 * sizeof(int);
  int *stack = (int *) tidl_malloc(stack_size);
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
    float pivot = input[r];
    int   id    = output[r];

    // partition
    int cnt = l;
    for (int i = l; i < r; i++)
    {
      if (input[i] > pivot)
      {
        float ftmp = input[i];
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

  tidl_free(stack, stack_size);
}
