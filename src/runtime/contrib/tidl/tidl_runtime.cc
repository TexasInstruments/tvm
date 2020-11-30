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
 * \brief TIDLModule is the runtime module for TIDL backend.
 */

#include <dlfcn.h>
#include <dmlc/logging.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>
#include "../../../support/base64.h"

#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../file_util.h"
#include "tidl_runtime.h"
#include "itidl_rt.h"

static int tidlrt_debuglevel = 0;
static int tidlrt_perfstats = 0;

static void __attribute__((constructor)) lib_init()
{
	char *debug_str, *perf_str;

	debug_str = getenv("TIDL_RT_DEBUG");
	if(!debug_str)
		tidlrt_debuglevel = 0;
	else
		tidlrt_debuglevel = atoi(debug_str);

	perf_str = getenv("TIDL_RT_PERFSTATS");
	if(!perf_str)
		tidlrt_perfstats = 0;
	else
		tidlrt_perfstats = atoi(perf_str);
}

#define TIDL_LOG   LOG_IF(INFO, (tidlrt_debuglevel > 0))

// #define TVM_RUNTIME_DBG_TIDL_TIMER
#ifdef TVM_RUNTIME_DBG_TIDL_TIMER
struct timespec t0, t1;
#define tick() clock_gettime(CLOCK_MONOTONIC, &t0);
#define tock() \
  (clock_gettime(CLOCK_MONOTONIC, &t1), t1.tv_sec - t0.tv_sec + (t1.tv_nsec - t0.tv_nsec) / 1e9)
#endif

namespace tvm {
namespace runtime {

/*! \brief A module for TIDL runtime. */
class TIDLJ6Module : public runtime::ModuleNode {
 public:

  explicit TIDLJ6Module(int total_subgraphs,
                      const std::unordered_map<std::string, int>& num_inputs,
                      const std::unordered_map<std::string, int>& num_outputs) {
    this->total_subgraphs_ = total_subgraphs;
    this->num_inputs_ = num_inputs;
    this->num_outputs_ = num_outputs;
    this->tidl_handle = NULL;
  }

  typedef void (*tidl_subgraph_t)(int, int, int, int, int, float**, float**);

  /*! 
   * \brief Initialize TIDL runtime by loading subgraph execution function from 
   * TIDL library. 
   */
  void TidlInit() {
    if (!tidl_handle) {
      // Load TIDL shared library
      dlerror();
      tidl_handle = dlopen("libtidl_api.so", RTLD_NOW | RTLD_GLOBAL);
      const char* dlsym_error1 = dlerror();
      if (dlsym_error1) {
        LOG(FATAL) << "Cannot open libtidl_api.so! " << dlsym_error1 << '\n';
      }
      // Load TIDL subgraph execution function
      dlerror();
      tidl_subgraph = (tidl_subgraph_t)dlsym(tidl_handle, "TidlRunSubgraph");
      const char* dlsym_error2 = dlerror();
      if (dlsym_error2) {
        LOG(FATAL) << "Cannot load symbol 'TidlRunSubgraph': " << dlsym_error2 << '\n';
        dlclose(tidl_handle);
      }
    }
  }

  /*!
   * \brief Provides a packed function for TVM runtime to execute,
   *  when TVM runtime wants to execute a subgraph with "tidl_" tag.
   * \param name Subgraph name which contains "tidl_" prefix if the subgraph is
   *  to run on TIDL.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name.find("tidl_") == std::string::npos) {
      return PackedFunc(nullptr);
    }

    if (name.find("tidl_get_custom_data_") != std::string::npos) {
      return PackedFunc(nullptr);
    }

    TidlInit();

    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      std::string subgraph_name = (std::string)name;
      // Get subgraph id which is after "tidl_" (5 characters)
      int subgraph_id = std::stoi(subgraph_name.erase(0, 5));
      // Get batch size of input data
      void* arg0 = args[0];
      DLTensor* tensor = reinterpret_cast<DLTensor*>(arg0);
      const int batch_size = tensor->shape[0];
      // Prepare input and output tensors for TIDL to execute on
      int num_inputs = num_inputs_[name];
      int num_outputs = num_outputs_[name];
      std::vector<float*> inputs;
      std::vector<float*> outputs;
      for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < num_inputs; i++) {
          inputs.push_back(GetTensorAddress(args[i], batch));
        }

        for (int i = 0; i < num_outputs; i++) {
          outputs.push_back(GetTensorAddress(args[num_inputs + i], batch));
        }
      }
      // Execute the subgraph on TIDL
      tidl_subgraph(total_subgraphs_, subgraph_id, batch_size, num_inputs,
                    num_outputs, &inputs[0], &outputs[0]);
    });
  }

  const char* type_key() const { return "tidl"; }

  float* GetTensorAddress(void* arg, int batch) {
    DLTensor* tensor = reinterpret_cast<DLTensor*>(arg);
    int tensor_size = 1;
    for (int dim = 1; dim < tensor->ndim; dim++) {
      tensor_size *= tensor->shape[dim];
    }
    float* tensor_ptr = reinterpret_cast<float*>(tensor->data);
    return (&(tensor_ptr[batch * tensor_size]));
  }

  void ToJSON(std::ostringstream& os,
              int total_subgraphs,
              const std::unordered_map<std::string, int>& num_inputs_,
              const std::unordered_map<std::string, int>& num_outputs_)
  {
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("total subgraphs", total_subgraphs);
    writer.WriteObjectKeyValue("subgraph inputs", num_inputs_);
    writer.WriteObjectKeyValue("subgraph outputs", num_outputs_);
    writer.EndObject();
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();

    std::ostringstream os;
    os << "J6";
    ToJSON(os, total_subgraphs_, num_inputs_, num_outputs_);
    SaveBinaryToFile(file_name, os.str());
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    std::ostringstream os;
    os << "J6";
    ToJSON(os, total_subgraphs_, num_inputs_, num_outputs_);
    stream->Write(os.str());
  }

  static Module FromJSON(std::istringstream& graph_stream) {

    int total_subgraphs;
    std::unordered_map<std::string, int> num_inputs;
    std::unordered_map<std::string, int> num_outputs;

    dmlc::JSONReader reader(&graph_stream);
    dmlc::JSONObjectReadHelper helper;
    // Read total subgraphs
    helper.DeclareField("total subgraphs", &total_subgraphs);
    // Read num_inputs
    helper.DeclareField("subgraph inputs", &num_inputs);
    // Read num_outputs
    helper.DeclareField("subgraph outputs",&num_outputs);
    helper.ReadAllFields(&reader);

    return TIDLJ6ModuleCreate(total_subgraphs, num_inputs, num_outputs);
  }


 private:
  int total_subgraphs_;
  std::unordered_map<std::string, int> num_inputs_;
  std::unordered_map<std::string, int> num_outputs_;
  void* tidl_handle;
  tidl_subgraph_t tidl_subgraph;
};

int32_t TIDLVprintf(const char * format, va_list arg)
{
  printf("TIDLVprintf format=%s\n", format);
  printf(format, arg);
  return 0;
}

class TIDLJ7Module : public runtime::ModuleNode {
 public:

  // I wanted to use a std::unique_ptr API for the unordered map, but there are
  // problems with the make_object API and non-copyable objects. I changed this
  // to pass by value just to make progress.
  explicit TIDLJ7Module(std::unordered_map<std::string, TIDLSubgraphInfo> infos)
           : infos(infos), subgraph_id(-1) {}

  ~TIDLJ7Module() {
    for (auto rt_arg : tidlrt_args)   delete rt_arg;

    if (tidlrt_handle != nullptr) {
      TIDL_LOG << "#TVM# TIDLRT_delete tidl_" << subgraph_id << ": "
               << tidlrt_handle << "...";
      if (TIDLRT_delete_(tidlrt_handle) != 0)
        LOG(FATAL) << "TIDLRT_delete failed\n";
    }
    if (tidl_handle != nullptr)  dlclose(tidl_handle);
  }

  /*!
   * \brief Initialize TIDL runtime by loading subgraph execution function from
   * TIDL library.
   */
  void LoadTIDLRT() {
    if(!tidl_handle) {
      // Load TIDL shared library
      dlerror();
      /*
       * The current TIOVX stack is designed in a way that everytime this
       * library is loaded, a new set of files are opened for IPC among
       * cores, without closing the existing ones. Soon the process runs
       * out of file descriptors and the most recently opened models fail
       * in TIDL_create.
       *
       * As a temporary workaround, opening the library two times and throwing
       * one handle away, so as to bump up the reference count on dlOpen()
       * and to prevent a future dlClose() from actually unloading the library.
       *
       * Very ugly hack, but need to maintain this as of now.
       */
      tidl_handle = dlopen("libvx_tidl_rt.so", RTLD_NOW | RTLD_GLOBAL );
      tidl_handle = dlopen("libvx_tidl_rt.so", RTLD_NOW | RTLD_GLOBAL );
      const char *dlsym_error1 = dlerror();
      if (dlsym_error1) {
        LOG(FATAL) << "Cannot open libvx_tidl_rt.so! " << dlsym_error1 << '\n';
      }

      TIDLRT_create_ = LoadSymbol<decltype(TIDLRT_create_)>  ("TIDLRT_create");
      TIDLRT_delete_ = LoadSymbol<decltype(TIDLRT_delete_)>  ("TIDLRT_delete");
      TIDLRT_invoke_ = LoadSymbol<decltype(TIDLRT_invoke_)>  ("TIDLRT_invoke");
      TIDLRT_deactive_ = LoadSymbol<decltype(TIDLRT_deactive_)>(
                                                          "TIDLRT_deactivate");
      TIDLRT_setParamsDefault_ = LoadSymbol<decltype(TIDLRT_setParamsDefault_)>(
                                                    "TIDLRT_setParamsDefault");
      TIDLRT_setTensorDefault_ = LoadSymbol<decltype(TIDLRT_setTensorDefault_)>(                                                    "TIDLRT_setTensorDefault");
      TIDLRT_getDdrStats_ = LoadSymbol<decltype(TIDLRT_getDdrStats_)>(
                                                    "TIDLRT_getDdrStats");
    }
  }

  // Loads a symbol from the tidl shared library
  template <typename T>
  T LoadSymbol(const char* symbol) {
    T sym = reinterpret_cast<T>(dlsym(tidl_handle, symbol));
    const char* error = dlerror();
    if (error) LOG(FATAL) << "Cannot load symbol " << symbol << ": " << error
                          << '\n';

    return sym;
  }

  /*!
   * \brief Provides a packed function for TVM runtime to execute,
   *  when TVM runtime wants to execute a subgraph with "tidl_" tag.
   * \param name Subgraph name which contains "tidl_" prefix if the subgraph is
   *  to run on TIDL.
   */
  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    if (name.find("tidl_") == std::string::npos) {
      return PackedFunc(nullptr);
    }

    if (name.find("tidl_get_custom_data_") != std::string::npos) {
      if(name.substr(21) == "ddrstats")
        return PackedFunc([this](tvm::TVMArgs args, tvm::TVMRetValue *rv) {
          uint64_t read = 0, write = 0;
          TIDLRT_getDdrStats_(&read, &write);
          std::pair<uint64_t, uint64_t> *v = new std::pair<uint64_t, uint64_t>(read, write);
          *rv = static_cast<void *>(v);
        });

      if(tidlrt_perfstats == 0)
        return PackedFunc(nullptr);

      int tmp_id = std::stoi(name.substr(21));
      if(tmp_id == subgraph_id && subgraph_id != -1)
        return PackedFunc([this](tvm::TVMArgs args, tvm::TVMRetValue *rv) {
          std::vector<uint64_t> *v = new std::vector<uint64_t>();

          v->push_back(uint64_t(stats.cpIn_time_start));
          v->push_back(uint64_t(stats.cpIn_time_end));
          v->push_back(uint64_t(stats.proc_time_start));
          v->push_back(uint64_t(stats.proc_time_end));
          v->push_back(uint64_t(stats.cpOut_time_start));
          v->push_back(uint64_t(stats.cpOut_time_end));
          *rv = static_cast<void *>(v);
        });

      return PackedFunc(nullptr);
    }

    auto info_it = infos.find(name);
    if (info_it == infos.end()) {
      // try to find it in next TIDLJ7Module
      return PackedFunc(nullptr);
    }
    auto& info = info_it->second;

    // Get subgraph id which is after "tidl_" (5 characters).
    subgraph_id = std::stoi(name.substr(5));

    // Load TIDLRT library.  Each subgraph/TIDLJ7Module will call this once,
    //     it is okay to dlopen() same library multiple times
    LoadTIDLRT();

    // Call TIDLRT_create() to initialize the subgraph
    sTIDLRT_Params_t params;
    TIDLRT_setParamsDefault_(&params);
    params.stats = (tidlrt_perfstats == 0) ? nullptr : &stats;
    params.netPtr = (void *) info.net_data.data();
    params.ioBufDescPtr = (void *) info.params_data.data();
    params.net_capacity = info.net_data.size();
    params.io_capacity  = info.params_data.size();
    params.traceLogLevel   = std::min(tidlrt_debuglevel, 3);
    params.traceWriteLevel = (tidlrt_debuglevel > 3) ? 1 : 0;
    params.TIDLVprintf = TIDLVprintf;
    TIDL_LOG << "#TVM# net size: " << info.net_data.size();
    TIDL_LOG << "#TVM# ioparams size: " << info.params_data.size();

    if (TIDLRT_create_(&params, &tidlrt_handle) != 0) {
      LOG(FATAL) << "Failed to initialize TIDLRT for subgraph " << subgraph_id
                 << '\n';
      return PackedFunc(nullptr);
    }
    TIDL_LOG << "#TVM# TIDLRT_create tidl_" << subgraph_id << ": "
             << tidlrt_handle;

    // Initialize sTIDLRT_Tensor_t* vector for inputs/outputs
    InitArgs(info);

    return PackedFunc([this, info](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      // args contains both inputs and outputs. First convert all args to
      // sTIDLRT_Tensor_t objects then use the number of inputs to find the
      // output index.
      ConvertArgs(args, info);

      // TIDLRT_invoke() sequence: TIDL_activate, TIDL_invoke, TIDL_deactivate
      if (TIDLRT_invoke_(tidlrt_handle, &tidlrt_args[0],
                                        &tidlrt_args[info.NumInputs()]) != 0)
        LOG(FATAL) << "TIDLRT_invoke failed\n";

      // If TIDLRT_invoke() changes and does not call TIDL_deactivate(),
      // we need to call TIDLRT_deactivate() explicitly here to handle
      // multiple models or multiple TIDL subgraphs.  We can offer
      // an environment variable to not call TIDL_deactivate()
      // for the case of single model/subgraph.
    });
  }

  // Initialize TIDLRT_Tensor_t for TIDLRT inputs and outputs
  // Certain fields can be set up at initialization time
  void InitArgs(const TIDLSubgraphInfo& info) {
    tidlrt_args.resize(info.NumInputs() + info.num_outputs, nullptr);

    for (int i = 0; i < (int) tidlrt_args.size(); i++) {
      sTIDLRT_Tensor_t * &rt_arg = tidlrt_args[i];
      rt_arg = new sTIDLRT_Tensor_t;
      memset(rt_arg, 0, sizeof(sTIDLRT_Tensor_t));
      TIDLRT_setTensorDefault_(rt_arg);

      // Set tensor name. Only inputs have names from TVM
      if (i < (int) info.input_names.size())
      {
        int name_size = std::min((int) info.input_names[i].size(),
                                 TIDLRT_STRING_SIZE - 1);
        strncpy((char *) rt_arg->name, info.input_names[i].c_str(), name_size);
        rt_arg->name[name_size] = '\0';
        TIDL_LOG << "#TVM# input name: " << (char *) rt_arg->name;
      }
      else
      {
        sprintf((char *) rt_arg->name, "tidl_%d_o%d",
                subgraph_id, i - (int) info.NumInputs());
        TIDL_LOG << "#TVM# output name: " << (char *) rt_arg->name;
      }

      // Skip padValues for now (assuming TVM tensor is continuous)
      rt_arg->padValues[0] =
      rt_arg->padValues[1] =
      rt_arg->padValues[2] =
      rt_arg->padValues[3] = 0;
      TIDL_LOG << "#TVM#  padValues: " << rt_arg->padValues[0] << " "
               << rt_arg->padValues[1] << " " << rt_arg->padValues[2] << " "
               << rt_arg->padValues[3];

      // Skip dataOffset (assuming TVM tensor starts from offset 0)
      rt_arg->dataOffset = 0;

      // If ConvertLayout("NCHW") pass is called on the entire graph
      // and only transpose or NCHW operators are allowed as TIDL
      // nodes, then TIDLRT do not need to perform any layout conversions.
      rt_arg->layout = info.is_nchw ? TIDLRT_LT_NCHW : TIDLRT_LT_NHWC;
      TIDL_LOG << "#TVM# layout: " << rt_arg->layout;

      // Skip zeroPoint and scale for now since those are for quantized models.
      rt_arg->zeroPoint = 0;
      rt_arg->scale = 1.0f;
      TIDL_LOG << "#TVM# zeroPoint: " << rt_arg->zeroPoint << ", scale: "
               << rt_arg->scale;

      // Set memtype
      rt_arg->memType = TIDLRT_MEM_USER_SPACE;
      TIDL_LOG << "#TVM# memType: " << rt_arg->memType;
    }

    tidlrt_args_all_fields_initialized = false;
  }

  // Convert arguments from TVM to sTIDLRT_Tensor_t objects
  void ConvertArgs(tvm::TVMArgs args, const TIDLSubgraphInfo& info) {
    for (int i = 0; i < args.size(); i++) {
      TIDL_LOG << "#TVM# Convert arg/tensor " << i;
      sTIDLRT_Tensor_t* rt_arg = tidlrt_args[i];

      // There are multiple type codes for a TVMArg. I think we only need to
      // support DLTensor, but TensorRT supports NDArray types as well.
      switch (args[i].type_code()) {
        case kTVMDLTensorHandle: {
          DLTensor* tensor_arg = args[i];

          /* -----------------------------------------------------------------
           * One time setup, do not change in subsequent subgraph calls
           * -----------------------------------------------------------------*/
          if (! tidlrt_args_all_fields_initialized) {
            // Set element type
            rt_arg->elementType = GetTIDLRTElementType(tensor_arg->dtype);
            TIDL_LOG << "#TVM# elementType: " << rt_arg->elementType;

            // Set the number of dimensions
            rt_arg->numDim = tensor_arg->ndim;
            TIDL_LOG << "#TVM# numDim: " << rt_arg->numDim;
            if (rt_arg->numDim > TIDLRT_DIM_MAX) {
              LOG(FATAL) << "Number of dimensions (" << rt_arg->numDim
                         << ") is greater than TIDLRT_DIM_MAX ("
                         << TIDLRT_DIM_MAX << ")\n";
              // Truncate to tidlrt max dimensions
              rt_arg->numDim = TIDLRT_DIM_MAX;
            }

            // Set the dimensions
            int missing_dims = TIDLRT_DIM_MAX - rt_arg->numDim;
            for (int s = 0; s < missing_dims; s++) {
              rt_arg->dimValues[s] = 1;
              TIDL_LOG << "#TVM#  auto-added dimValues[" << s << "]: 1";
            }
            for (int s = 0; s < rt_arg->numDim; s++) {
              int64_t shape = tensor_arg->shape[s];
              rt_arg->dimValues[missing_dims + s] = shape;
              TIDL_LOG << "#TVM#  dimValues[" << missing_dims+s << "]: "<<shape;

              // Perhaps this is paranoid, but make sure we don't overflow the
              // 32-bit integer in the TIDLRT tensor object.
              typedef typename std::remove_extent<decltype(rt_arg->dimValues)
                                                 >::type rt_dimValue_type;
              if (shape > std::numeric_limits<rt_dimValue_type>::max())
                LOG(FATAL) << "Tensor shape of " << shape
                           << " is not supported in TIDL RT\n";
            }

            // Skip pitch for now (assuming TVM tensor is continuous)
            rt_arg->pitch[2] = rt_arg->dimValues[3];
            rt_arg->pitch[1] = rt_arg->dimValues[2] * rt_arg->dimValues[3];
            rt_arg->pitch[0] = rt_arg->dimValues[1] * rt_arg->dimValues[2] *
                               rt_arg->dimValues[3];
            TIDL_LOG << "#TVM#  pitch: " << rt_arg->pitch[0] << " "
                     << rt_arg->pitch[1] << " " << rt_arg->pitch[2];
          }

          /* -----------------------------------------------------------------
           * Per-invocation setup, could change in each subgraph call
           * -----------------------------------------------------------------*/

          // Set the data pointer
          rt_arg->ptr = tensor_arg->data;
          TIDL_LOG << "#TVM# ptr: " << rt_arg->ptr;

          break;
        }

        // TensorRT handles this, but Jianzhong didn't.
        case kTVMNDArrayHandle:
        default:
          LOG(FATAL) << "Invalid TVMArgs type (" << args[i].type_code() <<")\n";
      }
    }

    tidlrt_args_all_fields_initialized = true;
  }

  // Return the TIDLRT element type for a given DLDataType
  int32_t GetTIDLRTElementType(const DLDataType dtype) {
    if (dtype.lanes != 1) {
      LOG(FATAL) << "Vector types are not supported in TIDL Tensors.\n";
      return -1;
    }

    switch (dtype.code) {
       case kDLInt:
         switch (dtype.bits) {
           case 8:
             return TIDLRT_Int8;
           case 16:
             return TIDLRT_Int16;
           case 32:
             return TIDLRT_Int32;
           default:
             LOG(FATAL) << "Invalid bit size (" << dtype.bits << ") for int\n";
             return -1;
         }

       case kDLUInt:
         switch (dtype.bits) {
           case 8:
             return TIDLRT_Uint8;
           case 16:
             return TIDLRT_Uint16;
           case 32:
             return TIDLRT_Uint32;
           default:
             LOG(FATAL) << "Invalid bit size (" << dtype.bits << ") for uint\n";
             return -1;
         }

      case kDLFloat:
        switch (dtype.bits) {
          case 32:
            return TIDLRT_Float32;
          default:
            LOG(FATAL) << "Invalid bit size (" << dtype.bits << ") for float\n";
            return -1;
        }

      default:
        LOG(FATAL) << "Invalid type code\n";
        return -1;
    }
  }

  const char* type_key() const { return "tidl"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    std::string bin;
    dmlc::MemoryStringStream mstrm(&bin);
    SaveToBinary(&mstrm);
    SaveBinaryToFile(file_name, bin);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    std::ostringstream os;
    os << "J7";
    dmlc::JSONWriter writer(&os);
    writer.Write(infos);
    stream->Write(os.str());
  }

  static Module FromJSON(std::istringstream& graph_stream) {
    dmlc::JSONReader reader(&graph_stream);
    std::unordered_map<std::string, TIDLSubgraphInfo> infos;
    reader.Read(&infos);
    return TIDLJ7ModuleCreate(infos);
  }

private:

  sTIDLRT_PerfStats_t stats;
  // I used an unordered map with strings as the index because there is built in
  // support to serialize/deserialize this to/from JSON.
  std::unordered_map<std::string, TIDLSubgraphInfo> infos;

  int subgraph_id;
  void* tidlrt_handle;
  std::vector<sTIDLRT_Tensor_t*> tidlrt_args;
  bool tidlrt_args_all_fields_initialized;

  // TIDLRT API from TIDLRT shared library
  void* tidl_handle = nullptr;
  decltype(&TIDLRT_create) TIDLRT_create_ = nullptr;
  decltype(&TIDLRT_delete) TIDLRT_delete_ = nullptr;
  decltype(&TIDLRT_invoke) TIDLRT_invoke_ = nullptr;
  decltype(&TIDLRT_deactivate) TIDLRT_deactive_ = nullptr;
  decltype(&TIDLRT_setParamsDefault) TIDLRT_setParamsDefault_ = nullptr;
  decltype(&TIDLRT_setTensorDefault) TIDLRT_setTensorDefault_ = nullptr;
  decltype(&TIDLRT_getDdrStats) TIDLRT_getDdrStats_ = nullptr;
};


// Loads a TIDL (J6 or J7) module from a file. As far as I know this function is
// never called because the module is always embedded as a binary in the DSO
// module.
static Module LoadFromFile(const std::string& path) {
  std::ifstream filep(path);
  filep.seekg(0, std::ios::end);
  size_t size = filep.tellg();
  std::string graph_info(size, ' ');
  filep.seekg(0);
  filep.read(&graph_info[0], size);

  std::istringstream graph_stream(graph_info);
  char keyword[3];
  graph_stream.get(keyword, 3);
  if (!strcmp(keyword, "J6"))
    return TIDLJ6Module::FromJSON(graph_stream);
  else if (!strcmp(keyword, "J7"))
    return TIDLJ7Module::FromJSON(graph_stream);
  else {
    LOG(FATAL) << "Unsupported platform found when loading TIDL binary (" << keyword << ")\n";
    return Module();
  }
}

// Loads a TIDL (J6 or J7) module from a binary
static Module LoadFromBinary(void* strm) {
  // Read the stream into a string. Seems like we should be able to avoid this ...
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string graph_info;
  stream->Read(&graph_info);

  // Create an input string stream and read the keyword to determine J6 or J7
  std::istringstream graph_stream(graph_info);
  char keyword[3];
  graph_stream.get(keyword, 3);
  if (!strcmp(keyword, "J6"))
    return TIDLJ6Module::FromJSON(graph_stream);
  else if (!strcmp(keyword, "J7"))
    return TIDLJ7Module::FromJSON(graph_stream);
  else {
    LOG(FATAL) << "Unsupported platform found when loading TIDL binary (" << keyword << ")\n";
    return Module();
  }
}

// Decodes a base64 string. Note this is copied from serialization.cc
inline std::string Base64Decode(std::string s) {
  dmlc::MemoryStringStream mstrm(&s);
  support::Base64InStream b64strm(&mstrm);
  std::string output;
  b64strm.InitPosition();
  dmlc::Stream* strm = &b64strm;
  strm->Read(&output);
  return output;
}

// Encodes a base64 string. Note this is copied from serialization.cc
inline std::string Base64Encode(std::string s) {
  std::string blob;
  dmlc::MemoryStringStream mstrm(&blob);
  support::Base64OutStream b64strm(&mstrm);
  dmlc::Stream* strm = &b64strm;
  strm->Write(s);
  b64strm.Finish();
  return blob;
}

// Save a TIDLSubgraphInfo to JSON
void TIDLSubgraphInfo::Save(dmlc::JSONWriter* writer) const {
  writer->BeginObject();
  writer->WriteObjectKeyValue("input_names", input_names);
  writer->WriteObjectKeyValue("num_outputs", num_outputs);
  writer->WriteObjectKeyValue("is_nchw", is_nchw);
  writer->WriteObjectKeyValue("net_data", Base64Encode(net_data));
  writer->WriteObjectKeyValue("params_data", Base64Encode(params_data));
  writer->EndObject();
}

// Load a TIDLSubgraphInfo from JSON
void TIDLSubgraphInfo::Load(dmlc::JSONReader* reader) {
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("input_names", &input_names);
  helper.DeclareField("num_outputs", &num_outputs);
  helper.DeclareField("is_nchw", &is_nchw);

  std::string net_base64;
  helper.DeclareField("net_data", &net_base64);

  std::string params_base64;
  helper.DeclareField("params_data", &params_base64);
  helper.ReadAllFields(reader);

  net_data = Base64Decode(net_base64);
  params_data = Base64Decode(params_base64);
}

// Factory method to create a J6 TIDL Module
Module TIDLJ6ModuleCreate(int total_subgraphs,
                          const std::unordered_map<std::string, int>& num_inputs,
                          const std::unordered_map<std::string, int>& num_outputs) {
  auto n = make_object<TIDLJ6Module>(total_subgraphs, num_inputs, num_outputs);
  return Module(n);
}

// Factory method to create a J7 TIDL Module
Module TIDLJ7ModuleCreate(std::unordered_map<std::string, TIDLSubgraphInfo> infos) {
  auto n = make_object<TIDLJ7Module>(infos);
  return Module(n);
}

// Register python API for loading the TIDL modules
TVM_REGISTER_GLOBAL("runtime.module.loadfile_tidl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tidl")
.set_body_typed(LoadFromBinary);


}  // namespace runtime
}  // namespace tvm
