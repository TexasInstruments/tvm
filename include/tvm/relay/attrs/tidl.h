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

/*! \brief Attributes used in TIDL inference operator */
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
