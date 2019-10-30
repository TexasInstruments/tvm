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
 * \file tvm/relay/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAY_ATTRS_TIDL_H_
#define TVM_RELAY_ATTRS_TIDL_H_

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in TIDL sort operators */
struct TidlSortAttrs : public tvm::AttrsNode<TidlSortAttrs> {
  int axis;
  bool is_ascend;
  DataType dtype;
  std::string test_new_attr;

  TVM_DECLARE_ATTRS(TidlSortAttrs, "relay.attrs.TidlSortAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1)
      .describe("Axis along which to sort the input tensor."
                "If not given, the flattened array is used.");
    TVM_ATTR_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order."
                "By default, sort in ascending order");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
      .describe("DType of the output indices.");
    TVM_ATTR_FIELD(test_new_attr).set_default("Default value for new attr")
      .describe("dummy attr.");
  }
};

/*! \brief Attributes used in TIDL sort operators */
struct TidlMatAddAttrs : public tvm::AttrsNode<TidlMatAddAttrs> {
  int axis;
  std::string kernel_attr;
#if 1
  TVM_DECLARE_ATTRS(TidlMatAddAttrs, "relay.attrs.TidlMatAddAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1)
      .describe("Axis along which to sort the input tensor."
                "If not given, the flattened array is used.");
    TVM_ATTR_FIELD(kernel_attr).set_default("djdbg_no_kernel_attr")
      .describe("Kernel name for selection of precompiled model.");
  }
#endif

};

/*! \brief Attributes used in TIDL sort operators */
struct TidlInferenceAttrs : public tvm::AttrsNode<TidlInferenceAttrs> {
  int num_labels;
  std::string inference_attr;

  TVM_DECLARE_ATTRS(TidlInferenceAttrs, "relay.attrs.TidlInferenceAttrs") {
    TVM_ATTR_FIELD(num_labels).set_default(1001)
      .describe("Number of labels.");
    TVM_ATTR_FIELD(inference_attr).set_default("Default value for new attr")
      .describe("dummy attr.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_TIDL_H_
