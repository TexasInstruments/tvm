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
 * \file src/relay/backend/contrib/tidl/codegen.cc
 * \brief Implementation of TIDL codegen APIs.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stack>
#include <cstdint>
#include <streambuf>

#include "../../utils.h"
#include "../../../../runtime/contrib/tidl/tidl_runtime.h"
#include "../codegen_c/codegen_c.h"
#include "picojson.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief TIDLContextNode contains the information that a pass can rely on,
 * such as analysis results.
 * \sa TIDLContext
 */
class TIDLContextNode : public Object {
 public:
  /*!
   * \brief The error reporter used to notify users why an optimization fails.
   */
  ErrorReporter err_reporter;

  std::string artifacts_directory;

  std::string platform;
  int         c7x_codegen_enabled;
  int         gen_c7x_mod_enabled;

  TIDLContextNode() : artifacts_directory(""), platform("J7"),
                      c7x_codegen_enabled(0), gen_c7x_mod_enabled(0) {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("artifacts_directory", &artifacts_directory);
    v->Visit("platform", &platform);
    v->Visit("c7x_codegen_enabled", &c7x_codegen_enabled);
    v->Visit("gen_c7x_mod_enabled", &gen_c7x_mod_enabled);
  }

  static constexpr const char* _type_key = "tidl.TIDLContext";
  static constexpr bool _type_has_method_sequal_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(TIDLContextNode, Object);
};


/*!
 * \brief TIDLContext that is used to configure the pass behavior.
 *
 * \code
 *
 *  auto new_ctx = TIDLContext::Create();
 *  ctx->opt_level = 2;
 *  ctx->fallback_device = kDLCPU;
 *  With<TIDLContext> scope(ctx);
 *  // pass context in effect.
 *
 * \endcode
 * \sa TIDLContextNode
 */
class TIDLContext : public ObjectRef {
 public:
  TIDLContext() {}
  explicit TIDLContext(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief const accessor.
   * \return const access pointer.
   */
  const TIDLContextNode* operator->() const {
    CHECK(get() != nullptr);
    return static_cast<const TIDLContextNode*>(get());
  }
  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  TIDLContextNode* operator->() {
    CHECK(get() != nullptr);
    return static_cast<TIDLContextNode*>(get_mutable());
  }
  /*!
   * \brief Construct a TIDLContext containing the default configurations.
   * \return The new TIDLContext.
   */
  TVM_DLL static TIDLContext Create();
  /*!
   * \brief Get the default pass context in the current scope.
   * \return The pass context.
   */
  TVM_DLL static TIDLContext Current();
#if 0
  /*!
   * \brief Apply the tracing functions of the context to the module, with the info.
   * \param module The IRModule to trace.
   * \param info The pass information.
   * \param is_before Indicated whether the tracing is before or after a pass.
   */
  TVM_DLL void Trace(const IRModule& module, const TIDLInfo& info, bool is_before) const;
  #endif

  // accessor.
  using ContainerType = TIDLContextNode;
  class Internal;

 private:
  // The entry of a pass context scope.
  TVM_DLL void EnterWithScope();
  // The exit of a pass context scope.
  TVM_DLL void ExitWithScope();

  // Classes to get the Python `with` like syntax.
  friend class Internal;
  friend class With<TIDLContext>;
};

class TIDLContext::Internal {
 public:
  static void EnterScope(TIDLContext ctx) {
    ctx.EnterWithScope();
  }

  static void ExitScope(TIDLContext ctx) {
    ctx.ExitWithScope();
  }
};

struct TIDLContextThreadLocalEntry {
  /*! \brief The default pass context. */
  TIDLContext default_context;

  /*! \brief The current pass context. */
  std::stack<TIDLContext> context_stack;

  TIDLContextThreadLocalEntry() {
    default_context = TIDLContext(make_object<TIDLContextNode>());
  }
};

/*! \brief Thread local store to hold the pass context. */
typedef dmlc::ThreadLocalStore<TIDLContextThreadLocalEntry>
    TIDLContextThreadLocalStore;

void TIDLContext::EnterWithScope() {
  TIDLContextThreadLocalEntry* entry =
      TIDLContextThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void TIDLContext::ExitWithScope() {
  TIDLContextThreadLocalEntry* entry =
      TIDLContextThreadLocalStore::Get();
  CHECK(!entry->context_stack.empty());
  CHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

TIDLContext TIDLContext::Current() {
  TIDLContextThreadLocalEntry* entry =
      TIDLContextThreadLocalStore::Get();
  if (!entry->context_stack.empty()) {
    return entry->context_stack.top();
  } else {
    return entry->default_context;
  }
}


TIDLContext TIDLContext::Create() {
  return TIDLContext(make_object<TIDLContextNode>());
}

TVM_REGISTER_GLOBAL("tidl.GetCurrentTIDLContext")
.set_body_typed(TIDLContext::Current);

TVM_REGISTER_GLOBAL("tidl.EnterTIDLContext")
.set_body_typed(TIDLContext::Internal::EnterScope);

TVM_REGISTER_GLOBAL("tidl.ExitTIDLContext")
.set_body_typed(TIDLContext::Internal::ExitScope);

TVM_REGISTER_NODE_TYPE(TIDLContextNode);

TVM_REGISTER_GLOBAL("tidl.CreateTIDLContext")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  auto ctx = TIDLContext::Create();
  runtime::String artifacts_directory = args[0];
  runtime::String platform = args[1];
  int             c7x_codegen_enabled = args[2];
  int             gen_c7x_mod_enabled = args[3];
  ctx->artifacts_directory = artifacts_directory;
  ctx->platform = platform;
  ctx->c7x_codegen_enabled = c7x_codegen_enabled;
  ctx->gen_c7x_mod_enabled = gen_c7x_mod_enabled;
  *ret = ctx;
});

/*!
 * \brief Generates a TIDLModule from a Relay expression. The generated TIDLModule
 * does not contain the TIDL representation, since the conversion from Relay to
 * TIDL representation needs to be done before codegen. The TIDLModule only
 * contains total number of subgraphs, and number of inputs and outputs for each
 * subgraph.
 */
class TIDLJ7ModuleCodeGen : public CSourceModuleCodegenBase {
 public:
  /*!
   * \brief Gets a TIDL SubgraphInfo object from a Relay function.
   * \param func A relay function that will be executed by TIDL as a subgraph.
   * \return A TIDLSubgraphInfo object.
   */
  std::pair<std::string, runtime::TIDLSubgraphInfo> GetSubgraphInfo(const Function& func) {
    TIDLContext ctx = TIDLContext::Current();
    runtime::TIDLSubgraphInfo subgraph_info;

    // Get the subgraph name and id. We should eventually have API calls to
    // manipulate the subgraph id and create file names for the subgraph net and
    // params files.
    auto subgraph_name = GetExtSymbol(func);
    CHECK(subgraph_name.substr(0, 5) == "tidl_");
    int subgraph_id = std::stoi(subgraph_name.substr(5));
    const std::string tempdir_name = ctx->artifacts_directory + "/tempDir";

    std::stringstream subgraph_prefix_stream;
    subgraph_prefix_stream << tempdir_name << "/subgraph" << subgraph_id;
    std::string subgraph_prefix = subgraph_prefix_stream.str();

    // Read in the subgraph info file
    // Read and parse the file using the picoJSON parser.
    // Format of relay.nfo file is:
    // {   ...
    //     "subgraphs" : [
    //           { "name"   : "tidl_0",
    //             "is_nchw": 0,
    //             ...
    //           }
    //           ...
    subgraph_info.is_nchw = -1;

    const std::string info_filename = tempdir_name + "/relay.nfo";
    std::ifstream info_file_stream(info_filename);
    if (!info_file_stream.is_open())
      LOG(FATAL) << "Failed to open TIDL info file " << info_filename << '\n';

    picojson::value json;
    auto in_it = std::istreambuf_iterator<char>(info_file_stream);
    std::string err;
    in_it = picojson::parse(json, in_it, std::istreambuf_iterator<char>(), &err);
    info_file_stream.close();
    if (!err.empty())
      LOG(FATAL) << "picoJSON error parsing TIDL info file " << info_filename <<
                    '[' << err << ']' << '\n';

    else if (json.is<picojson::object>()) {
      const picojson::object& obj = json.get<picojson::object>();
      auto it = obj.find("subgraphs");
      if (it != obj.end() && it->second.is<picojson::array>()) {
        const picojson::array& subgraphs = it->second.get<picojson::array>();
        for (const auto &sg : subgraphs) {
          if (!sg.is<picojson::object>())
            continue;
          const picojson::object& sg_obj = sg.get<picojson::object>();
          auto it1 = sg_obj.find("name");
          if (it1 == sg_obj.end() || it1->second.to_str() != subgraph_name)
            continue;
          auto it2 = sg_obj.find("is_nchw");
          if (it2 == sg_obj.end())
            continue;
          subgraph_info.is_nchw = (it2->second.to_str() == "1");
          auto it3 = sg_obj.find("inouts_zp");
          if (it3 == sg_obj.end())
            continue;
          const picojson::array& zps = it3->second.get<picojson::array>();
          for (const auto &zp : zps)
            subgraph_info.inouts_zp.push_back((int32_t) zp.get<double>());
          auto it4 = sg_obj.find("inouts_scale_inv");
          if (it4 == sg_obj.end())
            continue;
          const picojson::array& scale_invs = it4->second.get<picojson::array>();
          for (const auto &scale_inv : scale_invs)
            subgraph_info.inouts_scale_inv.push_back((float) scale_inv.get<double>());
        }
      }
    }
    if (subgraph_info.is_nchw == -1)
       LOG(FATAL) << "Could not determine layout for subgraph " << subgraph_name << "\n";
    //std::cout << "Parsed info for " << subgraph_name << ", is_nchw=" << subgraph_info.is_nchw << "\n";

    // Read in the net binary file
    std::string net_filename = subgraph_prefix + "_net.bin";
    std::ifstream net_file_stream(net_filename, std::ios::binary | std::ios::in);
    if (!net_file_stream.is_open())
      LOG(FATAL) << "Failed to open TIDL network file " << net_filename << '\n';

    subgraph_info.net_data.assign(std::istreambuf_iterator<char>(net_file_stream),
                                   std::istreambuf_iterator<char>());

    // Read in the params binary file
    std::string params_filename = subgraph_prefix + "_params_1.bin";
    std::ifstream params_file_stream(params_filename, std::ios::binary | std::ios::in);
    if (!params_file_stream.is_open())
      LOG(FATAL) << "Failed to open TIDL params file " << params_filename << '\n';

    subgraph_info.params_data.assign(std::istreambuf_iterator<char>(params_file_stream),
                                     std::istreambuf_iterator<char>());

    // Add all the input tensor names to the subgraph.
    for (auto& var : func->params)
      subgraph_info.input_names.push_back(var->name_hint());


    // TODO: Determine if we need output names.
    subgraph_info.num_outputs = func->ret_type.as<TensorTypeNode>() ? 1
                                     : func->ret_type.as<TupleTypeNode>()->fields.size();

    return std::make_pair(subgraph_name, subgraph_info);
  }

  /*!
   * \brief Create TIDL module from Relay funtion or IRModule.
   * \param ref An object ref that could be either a Relay function or IRModule.
   * \return The TIDL runtime module.
   */
  virtual runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    std::unordered_map<std::string, runtime::TIDLSubgraphInfo> subgraph_infos;
    if (ref->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(ref);
      subgraph_infos.insert(GetSubgraphInfo(func));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        auto func = Downcast<Function>(it.second);
        subgraph_infos.insert(GetSubgraphInfo(func));
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }
    return runtime::TIDLJ7ModuleCreate(subgraph_infos);
  }
};


class J7CSourceCodegen : public TIDLJ7ModuleCodeGen {
 public:

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    CHECK(ref->IsInstance<FunctionNode>());

    Function func = Downcast<Function>(ref);
    TIDLContext ctx = TIDLContext::Current();

    const std::pair<std::string, runtime::TIDLSubgraphInfo>& subgraph_info = GetSubgraphInfo(func);

    const std::string& subgraph_name = subgraph_info.first;
    uint32_t           subgraph_id   = std::stoi(subgraph_name.substr(5));

    const std::string tempdir = ctx->artifacts_directory + "/tempDir";

    EmitHeaders(subgraph_name, subgraph_id, tempdir);
    EmitWrapperFunction(subgraph_name, subgraph_info.second);
    EmitDestroyFunction(subgraph_name);
    EmitInitFunction(subgraph_name, subgraph_id, subgraph_info.second);

    std::string code = code_stream_.str();

    // Record the external symbol for runtime lookup.
    String sym = GetExtSymbol(func);

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", Array<String>{sym}, Array<String>{});
  }

 private:
  std::ostringstream code_stream_;

  void EmitHeaders(const std::string subgraph_name, uint32_t subgraph_id, const std::string& tempdir);
  void EmitWrapperFunction(const std::string& prefix, const runtime::TIDLSubgraphInfo& subgraph_info);
  void EmitInitFunction(const std::string& prefix, uint32_t subgraph_id, const runtime::TIDLSubgraphInfo& subgraph_info);
  void EmitDestroyFunction(const std::string& prefix);
};


void J7CSourceCodegen::EmitHeaders(const std::string subgraph_name, uint32_t subgraph_id, const std::string& tempdir)
{
    const char* header_files = R"headers(
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"

#include "tidl_api.h"
)headers";

    // Create headers
    code_stream_ << "extern \"C\" {\n";
    code_stream_ << header_files;

    code_stream_ << "#include \"subgraph" << subgraph_id << "_net.c\"\n";
    code_stream_ << "#include \"subgraph" << subgraph_id << "_params.c\"\n";
    code_stream_ << "void* " << subgraph_name << "_instance;\n\n";
    code_stream_ << "extern void* getUDMADrvObjPtr();\n\n";
    code_stream_ << "} /* extern \"C\" */\n\n";
}


void J7CSourceCodegen::EmitInitFunction(const std::string& prefix, uint32_t subgraph_id,
                                        const runtime::TIDLSubgraphInfo& subgraph_info)
{
    const char* TS = "    ";
    code_stream_ << "extern \"C\" int " << prefix << "_init(void* rt_info) {\n"
                 << TS  << prefix << "_instance = init_tidl_subgraph((void *) subgraph" << subgraph_id << "_net_bin,\n"
                 << TS << TS << TS << "subgraph" << subgraph_id << "_net_bin_len,\n"
                 << TS << TS << TS << "(void* ) subgraph" << subgraph_id << "_params_1_bin,\n"
                 << TS << TS << TS << "getUDMADrvObjPtr(),\n"
                 << TS << TS << TS << subgraph_info.is_nchw << " /* is_nchw */,\n"
                 << TS << TS << TS << "rt_info);\n\n"
                 << TS << "return (" << prefix << "_instance == NULL) ? -1 : 0;\n"
                 << "}\n\n";
}


void J7CSourceCodegen::EmitWrapperFunction(const std::string& prefix, const runtime::TIDLSubgraphInfo& subgraph_info)

{
    const char* TS = "    ";

    code_stream_ << "extern \"C\" int " << prefix << "(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value, int* out_ret_tcode, void* resource_handle) {\n";

    uint32_t num_args = subgraph_info.NumInputs() + subgraph_info.NumOutputs();
    for (uint32_t i = 0; i < num_args; i++)
        code_stream_ << TS << "void* arg" << i << " = (((TVMValue*)args)[" << i << "].v_handle);\n";
    code_stream_ << "\n";

    int index = 0;
    code_stream_ << TS << "DLTensor* input_tensors[] = {";
    for (uint32_t i = 0; i < subgraph_info.NumInputs(); i++)
        code_stream_ << "(DLTensor*) arg" << index++ << ",";
    code_stream_ << "};\n";

    code_stream_ << TS << "DLTensor* output_tensors[] = {";
    for (uint32_t i = 0; i < subgraph_info.NumOutputs(); i++)
        code_stream_ << "(DLTensor*) arg" << index++ << ",";
    code_stream_ << "};\n";

    code_stream_ << TS << "process_tidl_subgraph(" << prefix << "_instance, input_tensors, output_tensors);\n\n";

    code_stream_ << TS << "return 0;\n";
    code_stream_ << "}\n\n";
}


void J7CSourceCodegen::EmitDestroyFunction(const std::string& prefix)
{
    code_stream_ << "extern \"C\" void " << prefix << "_destroy(void) {\n"
                 << "    free_tidl_subgraph(" << prefix << "_instance);\n"
                 << "}\n\n";
}

/*!
 * \brief Generates a TIDLJ7C7xModule from a Relay expression (call "tidl_tvm_0").
 * The generated TIDLJ7C7xModule dispatches outlined C7x TVM graph via TVM RT.
 * The dispatch passes TVM tensors as is from Arm TVM runtime to C7x TVM runtime.
 * Only the data pointer and the size of TVM tensors are needed by TVM RT.
 */
class TIDLJ7C7xModuleCodeGen : public CSourceModuleCodegenBase {
 public:
  /*!
   * \brief Gets a TIDL SubgraphInfo object from a Relay function.
   * \param func A relay function that will be executed by TIDL as a subgraph.
   * \return A TIDLSubgraphInfo object.
   */
  std::pair<std::string, runtime::C7xTVMGraphInfo> GetC7xTVMGraphInfo(const Function& func) {
    TIDLContext ctx = TIDLContext::Current();
    runtime::C7xTVMGraphInfo c7xgraph_info;

    // Get the subgraph name and tempdir name.
    auto subgraph_name = GetExtSymbol(func);
    const std::string tempdir_name = ctx->artifacts_directory + "/tempDir";

    // Read in the deploy_mod binary file
    std::string c7xmod_filename = tempdir_name + "/c7x_deploy_tvm.out";
    std::ifstream c7xmod_file_stream(c7xmod_filename, std::ios::binary | std::ios::in);
    if (!c7xmod_file_stream.is_open())
      LOG(FATAL) << "Failed to open C7x TVM deployable mod file " << c7xmod_filename << '\n';
    c7xgraph_info.c7x_deploy_mod.assign(std::istreambuf_iterator<char>(c7xmod_file_stream),
                                        std::istreambuf_iterator<char>());

    // Add all the input tensor names, and tensor sizes
    for (auto& var : func->params)
    {
      // Drop off "_c7x" suffix that was added as a workaround (see tidl_build_c7x_mod.py)
      std::string input_name = var->name_hint();  // "data_c7x"
      input_name.resize(input_name.size() - 4);   // "data"
      c7xgraph_info.input_names.push_back(input_name);
      c7xgraph_info.tensor_sizes.push_back(
                                 ComputeTensorTypeSize(var->checked_type().as<TensorTypeNode>()));
    }

    // Count outputs, and tensor sizes
    if (const TensorTypeNode *ttype = func->ret_type.as<TensorTypeNode>())
    {
      c7xgraph_info.tensor_sizes.push_back(ComputeTensorTypeSize(ttype));
    }
    else
    {
      const TupleTypeNode* ttypes = func->ret_type.as<TupleTypeNode>();
      for (auto& field : ttypes->fields)
        c7xgraph_info.tensor_sizes.push_back(ComputeTensorTypeSize(field.as<TensorTypeNode>()));
    }

    return std::make_pair(subgraph_name, c7xgraph_info);
  }

  /*!
   * \brief Create TIDL module from Relay funtion or IRModule.
   * \param ref An object ref that could be either a Relay function or IRModule.
   * \return The TIDL runtime module.
   */
  virtual runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    std::unordered_map<std::string, runtime::C7xTVMGraphInfo> subgraph_infos;
    if (ref->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(ref);
      subgraph_infos.insert(GetC7xTVMGraphInfo(func));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        auto func = Downcast<Function>(it.second);
        subgraph_infos.insert(GetC7xTVMGraphInfo(func));
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }
    return runtime::TIDLJ7C7xModuleCreate(subgraph_infos);
  }

 private:
  // Didn't find a readily available function in TVM, closest one is GetMemorySize()
  int32_t ComputeTensorTypeSize(const TensorTypeNode* ttype)
  {
    int32_t size = 1;
    for (IndexExpr dim : ttype->shape) {
      const int64_t* pval = tir::as_const_int(dim);
      ICHECK(pval != nullptr) << "TIDLJ7C7x: Cannot support symbolic tensor shape " << ttype->shape;
      ICHECK_GE(*pval, 0) << "TIDLJ7C7x: Cannot support tensor with negative shape" << *pval;
      size *= pval[0];
    }

    return (size * ((ttype->dtype.bits() * ttype->dtype.lanes() + 7) / 8));
  }
};


/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compile it into a TIDL runtime module.
 *
 *    c7x_codegen_enabled == 0: Disable C7x code generation, all TIDL-unsupported layers run on Arm
 *                         building an Arm deployable module
 *    c7x_codegen_enabled >  0: Enable  C7x code generation, all TIDL-unsupported layers run on C7x
 *      In the compilation flow, first we build a C7x deployable module (c7x_deploy_mod.out),
 *      then we embed C7x deployable module as a single node ("tidl_tvm_0") into Arm wrapper
 *      deployable module (deploy_graph.json, deploy_lib.so)
 *      - gen_c7x_mod_enabled = 1: building a C7x deployable module
 *      - gen_c7x_mod_enabled = 0: building an Arm wrapper deployable module
 */
runtime::Module TIDLCompiler(const ObjectRef& ref) {
  TIDLContext ctx = TIDLContext::Current();
  if (ctx->platform == "J7" || ctx->platform == "J721S2" || ctx->platform == "J784S4" ||
      ctx->platform == "J722S" || ctx->platform == "AM62A") {
    if (ctx->c7x_codegen_enabled > 0)
    {
      if (ctx->gen_c7x_mod_enabled == 1)
      {
        J7CSourceCodegen csource;
        return csource.CreateCSourceModule(ref);
      }
      else
      {
        TIDLJ7C7xModuleCodeGen tidl;
        return tidl.CreateCSourceModule(ref);
      }
    }
    else
    {
      TIDLJ7ModuleCodeGen tidl;
      return tidl.CreateCSourceModule(ref);
    }
  } else {
    LOG(FATAL) << "Illegal TIDL platform " << ctx->platform;
    return runtime::Module();
  }
}

TVM_REGISTER_GLOBAL("relay.ext.tidl").set_body_typed(TIDLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
