/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tidl/tidl_runtime.h
 * \brief TIDLModule is the runtime module for TIDL backend.
 */

#ifndef TVM_RUNTIME_CONTRIB_TIDL_TIDL_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_TIDL_TIDL_RUNTIME_H_

#include <tvm/ir/module.h>

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Create a TIDLModule.
 * \param total_subgraphs Total number of subgraphs
 * \param num_inputs  Map of subgraph name to number of inputs
 * \param num_outputs Map of subgraph name to number of outputs
 * \return TIDLModule created from subgraphs.
 */
Module TIDLJ6ModuleCreate(int total_subgraphs,
                          const std::unordered_map<std::string, int>& num_inputs,
                          const std::unordered_map<std::string, int>& num_outputs);

// Class to pass information between the TIDL compiler and runtime module
// classes. This is only used by J7 for now, but J6 could easily be refactored
// to use the same approach.
class TIDLSubgraphInfo {
 public:
  std::string net_data;
  std::string params_data;
  std::vector<std::string> input_names;
  std::size_t num_outputs;
  int32_t     is_nchw;
  std::vector<int32_t> inouts_zp;
  std::vector<float>   inouts_scale_inv;

  std::size_t NumInputs() const { return input_names.size(); }
  std::size_t NumOutputs() const { return num_outputs; }

  void Save(dmlc::JSONWriter* writer) const;
  void Load(dmlc::JSONReader* reader);
};

Module TIDLJ7ModuleCreate(std::unordered_map<std::string, TIDLSubgraphInfo> infos);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TIDL_TIDL_RUNTIME_H_
