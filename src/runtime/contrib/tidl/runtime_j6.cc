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
 * \file runtime/contrib/tidl/tidl_runtime.cc
 * \brief TIDLJ6Module is the runtime module for TIDL backend.
 */

#include <stdlib.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <dmlc/logging.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_util.h"
#include <time.h>

#include "runtime_j6.h"

//#define TVM_RUNTIME_DBG_TIDL_TIMER
#ifdef TVM_RUNTIME_DBG_TIDL_TIMER
struct timespec t0,t1;
#define tick()  clock_gettime(CLOCK_MONOTONIC, &t0);
#define tock() (clock_gettime(CLOCK_MONOTONIC, &t1), \
                        t1.tv_sec - t0.tv_sec + (t1.tv_nsec - t0.tv_nsec) / 1e9)
#endif

extern "C" {
extern void TidlRunSubgraph(
    int total_subgraphs,   // total number of subgraphs to be executed on TIDL
    int subgraph_id,       // subgraph id: 0, 1, 2, etc.
    int batch_size,        // batch value of input tensor
    int num_inputs,        // number of inputs to the subgraph
    int num_outputs,       // number of outputs of the subgraph
    float **inputTensors,  // array of input tensors
    float **outputTensors  // array of output tensors
    );
}
typedef void (*tidl_subgraph_t)(int, int, int, int, int, float **, float **);

inline std::string ToJSON(int total_subgraphs,
                          const std::unordered_map<std::string, int>& num_inputs_,
                          const std::unordered_map<std::string, int>& num_outputs_)
{
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  writer.BeginObject();
  writer.WriteObjectKeyValue("total subgraphs",  total_subgraphs);
  writer.WriteObjectKeyValue("subgraph inputs",  num_inputs_);
  writer.WriteObjectKeyValue("subgraph outputs", num_outputs_);
  writer.EndObject();
  return os.str();
}

inline void FromJSON(const std::string& str,
                     int &total_subgraphs,
                     std::unordered_map<std::string, int>& num_inputs,
                     std::unordered_map<std::string, int>& num_outputs)
{
  std::istringstream is(str);
  dmlc::JSONReader reader(&is);
  dmlc::JSONObjectReadHelper helper;
  // Read total subgraphs
  helper.DeclareField("total subgraphs", &total_subgraphs);
  // Read num_inputs
  helper.DeclareField("subgraph inputs", &num_inputs);
  // Read num_outputs
  helper.DeclareField("subgraph outputs",&num_outputs);
  helper.ReadAllFields(&reader);
}

namespace tvm {
namespace runtime {

/*! \brief A module for TIDL runtime. */
class TIDLJ6Module : public runtime::ModuleNode {
 public:
  explicit TIDLJ6Module(int total_subgraphs, 
                        const std::unordered_map<std::string, int>& num_inputs,
                        const std::unordered_map<std::string, int>& num_outputs) {
    this->total_subgraphs_ = total_subgraphs;
    this->num_inputs_  = num_inputs;
    this->num_outputs_ = num_outputs;
    this->tidl_handle  = NULL;
  }

  /*! 
   * \brief Initialize TIDL runtime by loading subgraph execution function from 
   * TIDL library. 
   */
  void TidlInit() {
    if(!tidl_handle) {
      // Load TIDL shared library
      dlerror();
      tidl_handle = dlopen("libtidl_api.so", RTLD_NOW | RTLD_GLOBAL );
      const char *dlsym_error1 = dlerror();
      if (dlsym_error1) {
        LOG(FATAL) << "Cannot open libtidl_api.so! " << dlsym_error1 << '\n';
      }
      // Load TIDL subgraph execution function
      dlerror();
      tidl_subgraph = (tidl_subgraph_t) dlsym(tidl_handle, "TidlRunSubgraph");
      const char *dlsym_error2 = dlerror();
      if (dlsym_error2) {
         LOG(FATAL) << "Cannot load symbol 'TidlRunSubgraph': " << dlsym_error2 << '\n';
         dlclose(tidl_handle);
      }
    }
  }

  /*! 
   * \brief Provides a packed function implementation for TVM runtime to execute,  
   *  when TVM runtime wants to execute a subgraph with "tidl_" tag.
   * \param name Subgraph name which contains "tidl_" prefix if the subgraph is 
   *  to run on TIDL.
   */
  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    if (name.find("tidl_") == std::string::npos) {
      return PackedFunc(nullptr);
    }

    TidlInit();

    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
#ifdef TVM_RUNTIME_DBG_TIDL_TIMER
      tick();
#endif
      std::string subgraph_name = (std::string)name;
      // Get subgraph id which is after "tidl_" (5 characters)
      int subgraph_id = std::stoi(subgraph_name.erase(0,5));
      // Get batch size of input data
      DLTensor *arg0 = (DLTensor *)args[0];
      const int batch_size = arg0->shape[0];
      // Prepare input and output tensors for TIDL to execute on
      int num_inputs  = num_inputs_[name];
      int num_outputs = num_outputs_[name];
      std::vector<float *> inputs;
      std::vector<float *> outputs;
      for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < num_inputs; i++) {
          DLTensor *arg = (DLTensor *)args[i];
          int tensor_size = 1;
          for (int dim = 1; dim < arg->ndim; dim++) {
            tensor_size *= arg->shape[dim];
          }
          float *input_ptr = reinterpret_cast<float*>(arg->data);
          inputs.push_back(&(input_ptr[batch*tensor_size]));
        }
      
        for (int i = 0; i < num_outputs; i++) {
          const int index_in_args = num_inputs + i;
          DLTensor *arg = (DLTensor *)args[index_in_args];
          int tensor_size = 1;
          for (int dim = 1; dim < arg->ndim; dim++) {
            tensor_size *= arg->shape[dim];
          }
          float *output_ptr = reinterpret_cast<float*>(arg->data);
          outputs.push_back(&(output_ptr[batch*tensor_size]));
        }
      }
      // Execute the subgraph on TIDL
      tidl_subgraph(total_subgraphs_, subgraph_id, batch_size, 
                    num_inputs, num_outputs, &inputs[0], &outputs[0]);

#ifdef TVM_RUNTIME_DBG_TIDL_TIMER
      double time_secs = tock();
      printf("Time spent on TIDL: %f seconds.\n", time_secs);
#endif
    });
  }

  const char* type_key() const { return "tidl"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    SaveBinaryToFile(file_name, ToJSON(total_subgraphs_, num_inputs_, num_outputs_));
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(ToJSON(total_subgraphs_, num_inputs_, num_outputs_));
  }

  static Module LoadFromFile(const std::string& path) {
    int total_subgraphs;
    std::unordered_map<std::string, int> num_inputs;
    std::unordered_map<std::string, int> num_outputs;
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string graph_info(size, ' ');
    filep.seekg(0);
    filep.read(&graph_info[0], size);
    FromJSON(graph_info, total_subgraphs, num_inputs, num_outputs);

    return TIDLJ6ModuleCreate(total_subgraphs, num_inputs, num_outputs);
  }

  static Module LoadFromBinary(void* strm) {
    int total_subgraphs;
    std::unordered_map<std::string, int> num_inputs;
    std::unordered_map<std::string, int> num_outputs;
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string graph_info;
    stream->Read(&graph_info);
    FromJSON(graph_info, total_subgraphs, num_inputs, num_outputs);
    return TIDLJ6ModuleCreate(total_subgraphs, num_inputs, num_outputs);
  }

 private:
  int total_subgraphs_;
  std::unordered_map<std::string, int> num_inputs_;
  std::unordered_map<std::string, int> num_outputs_;
  void *tidl_handle;
  tidl_subgraph_t tidl_subgraph;
};

Module TIDLJ6ModuleCreate(int total_subgraphs, 
                          const std::unordered_map<std::string, int>& num_inputs,
                          const std::unordered_map<std::string, int>& num_outputs) {
  auto n = make_object<TIDLJ6Module>(total_subgraphs, num_inputs, num_outputs);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_tidlj6")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TIDLJ6Module::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tidlj6")
.set_body_typed(TIDLJ6Module::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
