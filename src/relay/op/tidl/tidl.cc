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

TVM_REGISTER_NODE_TYPE(TidlSortAttrs);

bool TidlSortRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // `types` contains: [data, result]
  const TidlSortAttrs* param = attrs.as<TidlSortAttrs>();
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "Tidl: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorTypeNode::make(data->shape, param->dtype));
  return true;
}

Expr MakeTidlSort(Expr data,
                 int axis,
                 bool is_ascend,
                 DataType dtype,
                 std::string test_new_attr) {
  auto attrs = make_node<TidlSortAttrs>();
  attrs->axis = axis;
  attrs->is_ascend = is_ascend;
  attrs->dtype = dtype;
  attrs->test_new_attr = test_new_attr;
  //std::cout << "DJDBG MakeTidlSort extra attribute:" << test_new_attr << std::endl;
  static const Op& op = Op::Get("TidlSort");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op._make.TidlSort")
.set_body_typed(MakeTidlSort);

RELAY_REGISTER_OP("TidlSort")
.describe(R"doc(Returns the indices that would sort an input array along the given axis.)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.TidlSortAttrs")
.add_argument("data", "Tensor", "Input data.")
.set_support_level(6)
.add_type_rel("TidlSort", TidlSortRel);

/*!
 * \brief Define new custom operator for TIDL backend
 *
 */

TVM_REGISTER_NODE_TYPE(TidlMatAddAttrs);

bool TidlMatAddRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // Simple data type relation checker - just for prototyping
  CHECK_EQ(types.size(), 3);
  CHECK_EQ( types[0].as<TensorTypeNode>()->dtype, types[1].as<TensorTypeNode>()->dtype);
  reporter->Assign(types[2], types[1]);
  return true;
}

Expr MakeTidlMatAdd(Expr lhs, Expr rhs, std::string kernel_attr) {  
  auto attrs = make_node<TidlMatAddAttrs>();
   static const Op& op = Op::Get("TidlMatAdd");
   //std::cout << "DJDBG MakeTidlMatAdd:" << kernel_attr << std::endl;
   attrs->kernel_attr = kernel_attr;
   return CallNode::make(op, {lhs, rhs}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.TidlMatAdd")
.set_body_typed(MakeTidlMatAdd);

RELAY_REGISTER_OP("TidlMatAdd")
.describe("TIDL mat addition")
.set_num_inputs(2)
.set_attrs_type_key("relay.attrs.TidlMatAddAttrs")
.add_argument("lhs", "Tensor", "The left hand side tensor.")  
.add_argument("rhs", "Tensor", "The right hand side tensor.") 
.set_support_level(1)
.add_type_rel("TidlMatAdd", TidlMatAddRel);
#if 0
//Instead of using python wrapper we can register here directly
.set_attr<TOpPattern>("TOpPattern", kOpaque)
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs, const Array<Tensor>& inputs,
                                        const Type& out_type, const Target& target) {
    const auto* param = attrs.as<TidlMatAddAttrs>();
    std::cout << "DJDBG: I am in this new compute function!!!:" << sizeof(param->kernel_attr) << std::endl;
    return tvm::Array<tvm::Tensor>{topi::nn::bias_add(inputs[0], inputs[1], param->axis)};
});
#endif
//return [topi.TidlMatAdd(inputs[0], inputs[1], kernel_attr=kernel_attr)]


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
  //std::cout << "DJDBG in MakeTidlInference!" << std::endl;
  auto attrs = make_node<TidlInferenceAttrs>();
  attrs->num_labels = num_labels;
  attrs->inference_attr = inference_attr;
  //std::cout << "DJDBG MakeTidlSort extra attribute:" << test_new_attr << std::endl;
  static const Op& op = Op::Get("TidlInference");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op._make.TidlInference")
.set_body_typed(MakeTidlInference);

RELAY_REGISTER_OP("TidlInference")
.describe(R"doc(Returns the softmax normalized classification result.)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.TidlInferenceAttrs")
.add_argument("data", "Tensor", "Input data.")
.set_support_level(6)
.add_type_rel("TidlInference", TidlInferenceRel);



}  // namespace relay
}  // namespace tvm


