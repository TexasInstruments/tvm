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
 *  Copyright (c) 2017 by Contributors
 * \file Use external cblas library call.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>
#include <dlfcn.h>
#include <memory.h>

extern "C" {
//#include <cblas.h>
  extern void TidlRunSubgraph(int total_subgraphs, 
                              int subgraph_id, 
                              int batch_size,
                              int num_inputs, 
                              int num_outputs, float **inputTensors, float **outputTensors);
}

namespace tvm {
namespace contrib {

using namespace runtime;

typedef void (*tidl_subgraph_t)(int, int, int, int, int, float **, float **);
//Singletons required for making calls to TIDL-API
static void *tidl_handle = NULL;
static tidl_subgraph_t tidl_subgraph = (tidl_subgraph_t)NULL;

#define MAX_INPUT_TENSORS  4
#define MAX_OUTPUT_TENSORS 4
#define MAX_BATCH_SIZE     256

//===================================================
//Adding TIDL specific inference via call to TIDL-API
//===================================================
template<typename DataType, typename OutType>
void my_arginference(DLTensor* input, DLTensor* output, int32_t num_labels, std::string inference_attr) {
  auto data_ptr = reinterpret_cast<DataType *>(static_cast<char *>(input->data) + input->byte_offset);
  auto out_ptr  = reinterpret_cast<OutType *>(static_cast<char *>(output->data) + output->byte_offset);
  float *inputTensors[MAX_INPUT_TENSORS*MAX_BATCH_SIZE];
  float *outputTensors[MAX_OUTPUT_TENSORS*MAX_BATCH_SIZE];
  float *input_ptr  = static_cast<float *>(data_ptr);
  float *output_ptr = static_cast<float *>(out_ptr);
  int batch_size = input->shape[0];
  int image_size = input->shape[1] * input->shape[2] * input->shape[3];

  CHECK(batch_size <= MAX_BATCH_SIZE);

  for(int i = 0; i < batch_size; i++) {
    inputTensors[i]  = &input_ptr[i * image_size];
    outputTensors[i] = &output_ptr[i * num_labels];
  }

  if(!tidl_handle)
  {
    // reset errors
    dlerror();
    tidl_handle = dlopen("libtidl_api.so", RTLD_NOW | RTLD_GLOBAL );
    const char *dlsym_error1 = dlerror();
    if (dlsym_error1) {
      LOG(FATAL) << "Cannot open libtidl_api.so! " << dlsym_error1 << '\n';
      return;
    }
    dlerror();
    tidl_subgraph = (tidl_subgraph_t) dlsym(tidl_handle, "TidlRunSubgraph");
    const char *dlsym_error2 = dlerror();
    if (dlsym_error2) {
       LOG(FATAL) << "Cannot load symbol 'TidlRunSubgraph': " << dlsym_error2 << '\n';
       dlclose(tidl_handle); 
       return;
    }
  }
  // use it to do the calculation
  std::cout << "Calling tidl_subgraph...\n";
  tidl_subgraph(1, 0, batch_size, 1, 1, inputTensors, outputTensors);

  // close the library
  dlclose(tidl_handle);
}
//------------------------------------------------------------------------------------------------------------
TVM_REGISTER_GLOBAL("tvm.contrib.tidl.my_inference")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *input  = args[0]; // Input tensor
  DLTensor *output = args[1]; // Otput tensor
  int32_t num_labels = args[2];
  std::string inference_attr = args[3];

  auto data_dtype = TVMType2String(input->dtype);
  auto out_dtype  = TVMType2String(output->dtype);
  my_arginference<float, float>(input, output, num_labels, inference_attr);
});
//------------------------------------------------------------------------------------------------------------


}  // namespace contrib
}  // namespace tvm
