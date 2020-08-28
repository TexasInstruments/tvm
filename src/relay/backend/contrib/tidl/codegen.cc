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

#include "../../../../runtime/contrib/tidl/tidl_runtime.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

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

  TIDLContextNode() : artifacts_directory(""), platform("AM57") {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("artifacts_directory", &artifacts_directory);
    v->Visit("platform", &platform);
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
  ctx->artifacts_directory = artifacts_directory;
  ctx->platform = platform;
  *ret = ctx;
});

/*!
 * \brief Generates a TIDLModule from a Relay expression. The generated TIDLModule
 * does not contain the TIDL representation, since the conversion from Relay to
 * TIDL representation needs to be done before codegen. The TIDLModule only
 * contains total number of subgraphs, and number of inputs and outputs for each
 * subgraph.
 */
class TIDLJ6ModuleCodeGen : public CSourceModuleCodegenBase {
 public:
  /*!
   * \brief Get the number of inputs and number of outputs for a subgraph.
   * \param func A relay function that will be executed by TIDL as a subgraph.
   * \return The TIDL runtime module.
   */
  void GetSubgraphInfo(const Function& func) {
    auto subgraph_name = GetExtSymbol(func);
    const int num_inputs = func->params.size();
    subgraph_num_inputs[subgraph_name] = num_inputs;
    const int num_outputs = func->ret_type.as<TensorTypeNode>() ? 1
                          : func->ret_type.as<TupleTypeNode>()->fields.size();
    subgraph_num_outputs[subgraph_name] = num_outputs;
  }

  /*!
   * \brief Create TIDL module from Relay funtion or IRModule.
   * \param ref An object ref that could be either a Relay function or IRModule.
   * \return The TIDL runtime module.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    int total_subgraphs = 0;
    if (ref->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(ref);
      total_subgraphs = 1;
      GetSubgraphInfo(func);
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      total_subgraphs = mod->functions.size();
      for (const auto& it : mod->functions) {
        auto func = Downcast<Function>(it.second);
        GetSubgraphInfo(func);
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module.";
    }
    return runtime::TIDLJ6ModuleCreate(total_subgraphs, subgraph_num_inputs,
                                       subgraph_num_outputs);
  }

 private:
  /*! \brief Map of subgraph name to number of inputs/outputs */
  std::unordered_map<std::string, int> subgraph_num_inputs;
  std::unordered_map<std::string, int> subgraph_num_outputs;
};

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

    std::stringstream subgraph_prefix_stream;
    subgraph_prefix_stream << ctx->artifacts_directory << "/tempDir/" << "subgraph"
                           << subgraph_id;
    std::string subgraph_prefix = subgraph_prefix_stream.str();

    // Read in the subgraph info file
    std::string info_filename = subgraph_prefix + ".nfo";
    std::ifstream info_file_stream(info_filename.c_str());
    if (!info_file_stream.is_open())
      LOG(FATAL) << "Failed to open TIDL info file " << info_filename << '\n';
    subgraph_info.is_nchw = 1;
    dmlc::JSONReader reader(&info_file_stream);
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("is_nchw", &subgraph_info.is_nchw);
    helper.ReadAllFields(&reader);
    info_file_stream.close();

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
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
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

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compile it into a TIDL runtime module.
 *
 */
runtime::Module TIDLCompiler(const ObjectRef& ref) {
  TIDLContext ctx = TIDLContext::Current();
  if (ctx->platform == "AM57") {
    TIDLJ6ModuleCodeGen tidl;
    return tidl.CreateCSourceModule(ref);
  } else if (ctx->platform == "J7") {
    TIDLJ7ModuleCodeGen tidl;
    return tidl.CreateCSourceModule(ref);
  } else {
    LOG(FATAL) << "Illegal TIDL platform " << ctx->platform;
    return runtime::Module();
  }
}

TVM_REGISTER_GLOBAL("relay.ext.tidl").set_body_typed(TIDLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
