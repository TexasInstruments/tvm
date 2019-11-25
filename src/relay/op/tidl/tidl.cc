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
 *  Copyright (c) 2019 by Contributors
 * \file argsort.cc
 * \brief Tidl operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <topi/broadcast.h>
//#include "../type_relations.h"
#include <tvm/relay/attrs/nn.h>
#include <topi/nn.h>
#include <topi/nn/bias_add.h>
#include "../op_common.h"
#include <tvm/relay/attrs/tidl.h>

namespace tvm {
namespace relay {

/*!
 * \brief Define new custom operator for TIDL backend
 *
 */

TVM_REGISTER_NODE_TYPE(TidlInferenceAttrs);

bool TidlInferenceRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // `types` contains: [data, result]
  const TidlInferenceAttrs* param = attrs.as<TidlInferenceAttrs>();
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "Tidl: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  Array<IndexExpr> out_shape;
  out_shape.push_back(data->shape[0]);
  out_shape.push_back(param->num_labels);

  reporter->Assign(types[1], TensorTypeNode::make(out_shape, Float(32)));
  return true;
}

Expr MakeTidlInference(Expr data,
                 int num_labels,
                 std::string inference_attr) {
  auto attrs = make_node<TidlInferenceAttrs>();
  attrs->num_labels = num_labels;
  attrs->inference_attr = inference_attr;
  static const Op& op = Op::Get("TidlInference");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op._make.TidlInference")
.set_body_typed(MakeTidlInference);

RELAY_REGISTER_OP("TidlInference")
.describe(R"doc(Returns the softmax normalized classification result.)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type<TidlInferenceAttrs>()
.add_argument("data", "Tensor", "Input data.")
.set_support_level(6)
.add_type_rel("TidlInference", TidlInferenceRel);



}  // namespace relay
}  // namespace tvm


