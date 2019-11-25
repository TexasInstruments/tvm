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
#define MAX_BATCH_SIZE     8
//=============================
//Adding my specific matrix add
//=============================
#if 0
typedef struct {
  /*!
   * \brief The opaque data pointer points to the allocated data. This will be
   * CUDA device pointer or cl_mem handle in OpenCL. This pointer is always
   * aligns to 256 bytes as in CUDA.
   *
   * For given DLTensor, the size of memory required to store the contents of
   * data is calculated as follows:
   *
   * \code{.c}
   * static inline size_t GetDataSize(const DLTensor* t) {
   *   size_t size = 1;
   *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
   *     size *= t->shape[i];
   *   }
   *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
   *   return size;
   * }
   * \endcode
   */
  void* data;
  /*! \brief The device context of the tensor */
  DLContext ctx;
  /*! \brief Number of dimensions */
  int ndim;
  /*! \brief The data type of the pointer*/
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
  /*!
   * \brief strides of the tensor,
   *  can be NULL, indicating tensor is compact.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
} DLTensor;
#endif

int ColumnStride(DLTensor *tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[0], tensor->strides[1]);
  } else {
    return tensor->shape[1];
  }
}

int ElementStride(DLTensor *tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[0], tensor->strides[1]);
  } else {
    return 1;
  }
}


void AddFP32(TVMArgs args, TVMRetValue *ret) {
  DLTensor *A = args[0];
  DLTensor *B = args[1];
  DLTensor *C = args[2];
  std::string kernel_attr = args[3];
  int bit_depth = sizeof(float) * 8;
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(B), 1);
  CHECK_EQ(ElementStride(C), 1);
  CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

  float *ptr_a = reinterpret_cast<float *>(static_cast<char *>(A->data) + A->byte_offset);
  float *ptr_b = reinterpret_cast<float *>(static_cast<char *>(B->data) + B->byte_offset);
  float *ptr_c = reinterpret_cast<float *>(static_cast<char *>(C->data) + C->byte_offset);

  for(int64_t i = 0; i < C->shape[0]; i ++) {
    for(int64_t j = 0; j < C->shape[1]; j ++) {
      *ptr_c++ = *ptr_a++ + *ptr_b++; 
    }
  }

}

void AddFP64(TVMArgs args, TVMRetValue *ret) {
  DLTensor *A = args[0];
  DLTensor *B = args[1];
  DLTensor *C = args[2];
  std::string kernel_attr = args[3];
  int bit_depth = sizeof(double) * 8;
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(B), 1);
  CHECK_EQ(ElementStride(C), 1);
  CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  double *ptr_a = reinterpret_cast<double *>(static_cast<char *>(A->data) + A->byte_offset);
  double *ptr_b = reinterpret_cast<double *>(static_cast<char *>(B->data) + B->byte_offset);
  double *ptr_c = reinterpret_cast<double *>(static_cast<char *>(C->data) + C->byte_offset);

  for(int64_t i = 0; i < C->shape[0]; i ++) {
    for(int64_t j = 0; j < C->shape[1]; j ++) {
      *ptr_c++ = *ptr_a++ + *ptr_b++; 
    }
  }

}

// matrix elementwise addition
TVM_REGISTER_GLOBAL("tvm.contrib.tidl.my_matadd")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  CHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));

  if (TypeMatch(A->dtype, kDLFloat, 32))
    AddFP32(args, ret);
  else
    AddFP64(args, ret);
});

//==========================
//Adding my specific argsort
//==========================
template<typename DType>
bool CompareAscend(const std::pair<int64_t, DType>& lhs,
                   const std::pair<int64_t, DType>& rhs) {
  return lhs.second < rhs.second;
}

template<typename DType>
bool CompareDescend(const std::pair<int64_t, DType>& lhs,
                    const std::pair<int64_t, DType>& rhs) {
  return lhs.second > rhs.second;
}

template<typename DataType, typename OutType>
void my_argsort(DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend, std::string test_new_attr) {
  auto data_ptr = static_cast<DataType *>(input->data);
  auto out_ptr = static_cast<OutType *>(output->data);
  std::vector<std::pair<int64_t, DataType> > sorter;

  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }

  for (int i = 0 ; i < axis_mul_before; ++i) {
    for (int j = 0 ; j < axis_mul_after; ++j) {
      sorter.clear();
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        sorter.emplace_back(std::make_pair(k, data_ptr[full_idx]));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<DataType>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<DataType>);
      }
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        out_ptr[base_idx + k * axis_mul_after] = static_cast<OutType>(sorter[k].first);
      }
    }
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.tidl.my_sort")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *input = args[0];
  DLTensor *output = args[1];

  int32_t axis = args[2];
  bool is_ascend = args[3];
  std::string test_new_attr = args[4];

  if (axis < 0) {
    axis = input->ndim + axis;
  }
  CHECK_LT(axis, input->ndim) << "Axis out of boundary for "
                                 "input ndim " << input->ndim;

  auto data_dtype = TVMType2String(input->dtype);
  auto out_dtype = TVMType2String(output->dtype);

  if ((data_dtype == "float32") && (out_dtype == "int32"))
  {
    my_argsort<float, int32_t>(input, output, axis, is_ascend, test_new_attr);
  } else {
    LOG(FATAL) << "Unsupported type combination: data dtype:" << data_dtype << " output dtype: " << out_dtype;
  }
});
//------------------------------------------------------------------------------------------------------------
//=================================================
//Adding my specific inference via call to TIDL-API
//=================================================
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

  for(int i = 0; i < batch_size; i++) {
    inputTensors[i]  = &input_ptr[i * image_size];
    outputTensors[i] = &output_ptr[i * num_labels];
  }

#ifdef VERBOSE
  for(int32_t k = 0; k < input->shape[0]; k ++) {
    std::cout << "NEW IMAGE:" << std::endl;
    for(int32_t i = 0; i < input->shape[1]; i ++) {
      for(int32_t j = 0; j < input->shape[2]; j ++) {
        for(int32_t c = 0; c < input->shape[3]; c ++) std::cout << *data_ptr++ << " "; 
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
#endif

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
  //CHECK_EQ(input->ndim, 4); // e.g. [1, 3, 224, 224], NCHW
  auto data_dtype = TVMType2String(input->dtype);
  auto out_dtype  = TVMType2String(output->dtype);
  my_arginference<float, float>(input, output, num_labels, inference_attr);
});
//------------------------------------------------------------------------------------------------------------


}  // namespace contrib
}  // namespace tvm
