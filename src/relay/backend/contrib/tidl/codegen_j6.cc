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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/ir/module.h>

#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../codegen_c/codegen_c.h"
#include "../../../../runtime/contrib/tidl/runtime_j6.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief Generates a TIDLJ6Module from a Relay expression. The generated TIDLJ6Module
 * does not contain the TIDL representation, since the conversion from Relay to
 * TIDL representation needs to be done before codegen. The TIDLJ6Module only
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
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
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
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compile it into a TIDL runtime module.
 *
 */
runtime::Module TIDLJ6Compiler(const ObjectRef& ref) {
  TIDLJ6ModuleCodeGen tidl;
  return tidl.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.tidlj6").set_body_typed(TIDLJ6Compiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
