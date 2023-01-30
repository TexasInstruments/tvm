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

/* Testing DLR C/C++ inference API */

#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include "dlr.h"
#include "dlpack/dlpack.h"

#define TSIZE (3*224*224)
uint8_t airshow_raw[TSIZE];
float __attribute__((aligned(128))) airshow[TSIZE];

void check_status(int status, DLRModelHandle *p_model, const char *msg)
{
  if (status == 0)  return;
  printf("%s failed: %d\n", msg, status);
  if (p_model != NULL)  DeleteDLRModel(p_model);
  exit(status);
}

int main(int argc, char *argv[])
{
  int status;

  // Read input
  FILE *fin = fopen("./airshow_3x224x224.y", "rb");
  if (fin != nullptr) {
    fread(airshow_raw, TSIZE, 1, fin);
    for (int i = 0; i < TSIZE; i++)
      airshow[i] = ((float)airshow_raw[i] - 128.0f) / 128.0f;
    fclose(fin);
  }

  // Step 1: Create DLR model from compiled model artifacts
  DLRModelHandle model;
  const char *model_path = "../artifacts/mv2_onnx_J7_target_tidl_c7x";
  if (argc > 1)  model_path = argv[1];
  status = CreateDLRModel(&model, model_path, 1, 0);
  check_status(status, &model, "CreateDLRModel");

  // Step 2: Set input tensor
  const char *model_input_name = "data";
  int64_t shape[4] = {1, 3, 224, 224};
  DLTensor in_tensor = { (void*) airshow,
                         {kDLCPU, 0},
                         4,
                         {kDLFloat, 32, 1},
                         shape,
                         NULL,
                         0
                       };
  status = SetDLRInputTensorZeroCopy(&model, model_input_name, &in_tensor);
  check_status(status, &model, "SetDLRInputTensorZeroCopy");

  // Step 3: Run inference
  status = RunDLRModel(&model);
  check_status(status, &model, "RunDLRModel");

  // Step 4: Get output
  float *probs;
  status = GetDLROutputPtr(&model, 0, (const void**) &probs);
  check_status(status, &model, "GetDLROutputPtr");

  int64_t size;
  int dim;
  int64_t out_shape[8];
  char* type_name;
  status = GetDLROutputSizeDim(&model, 0, &size, &dim);
  check_status(status, &model, "GetDLROutputSizeDim");
  status = GetDLROutputShape(&model, 0, out_shape);
  check_status(status, &model, "GetDLROutputShape");
  status = GetDLROutputType(&model, 0, (const char**) &type_name);
  check_status(status, &model, "GetDLROutputType");

  printf("\nModel output 0 size=%" PRId64 ", dim=%d\n", size, dim);
  printf("Model output 0 shape: ");
  for (int i = 0; i < dim; i++)  printf("%" PRId64 "x", out_shape[i]);
  printf("\n");
  printf("Model output 0 type: %s\n", type_name);

  // Step 5: Interpret results
  int imax = 0;
  for (int i = 0; i < size; i++)
    if (probs[i] > probs[imax])
      imax = i;
  printf("Top 1 index = %d, probability = %f\n\n", imax, probs[imax]);

  // Step 6: Tear down
  status = DeleteDLRModel(&model);
  check_status(status, NULL, "DeleteDLRModel");

  if (imax != 895) {
    printf("Fail %d not in [895]\n", imax);
    status = -1;
  } else {
    printf("Pass\n");
    status = 0;
  }

  return status;
}
