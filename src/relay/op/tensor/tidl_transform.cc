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
 * \file tidl_transform.cc
 * \brief Transform operators.
 */

#include "transform.h"

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#include <sstream>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../../transforms/pass_utils.h"
#include "../../transforms/pattern_utils.h"
#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"


namespace tvm {
namespace relay {

bool TIDLScatterNDRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  // `types` contains: [data, indices, updates, result]
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  const auto* updates = types[2].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "TIDLScatterND: expect input data type to be TensorType but got " << types[0];
    return false;
  }
  if (indices == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "TIDLScatterND: expect indices type to be TensorType but got " << types[1];
    return false;
  }
  if (updates == nullptr) {
    ICHECK(types[2].as<IncompleteTypeNode>())
        << "TIDLScatterND: expect updates type to be TensorType but got " << types[2];
    return false;
  }
  ICHECK(indices->dtype.is_int() || indices->dtype.is_uint())
      << "TIDLScatterND: indices must be a tensor of integers.";

  const auto out_shape = data->shape;
  const IntImmNode* mdim = indices->shape[indices->shape.size() - 1].as<IntImmNode>();
  ICHECK(mdim) << "TIDLScatterND needs a static shape for the last axis of indices, got "
               << indices->shape;
  const size_t kdim = indices->shape.size() - 1;
  const size_t ndim = out_shape.size();
  ICHECK_LE(size_t(mdim->value), ndim)
      << "TIDLScatterND: Given updates with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), and indices "
         "with shape (Y_0, ..., Y_{K-1}, M), M must be less than or equal to N.";
  // Indices: (Y_0, .. Y_{K-1}, M) data: (Y_0, .. Y_{K-1}, ...), verify Y's.
  for (size_t i = 0; i < kdim; i++) {
    reporter->AssertEQ(indices->shape[i], updates->shape[i]);
  }

  std::vector<IndexExpr> oshape;
  for (auto& x : out_shape) {
    oshape.push_back(x);
  }

  // updates: (Y_0, .. Y_{K-1}, X_M, .. X_{N-1}) out: (X_0, .. X_{N-1}), verify X_M to X_{N-1}
  for (size_t i = mdim->value; i < ndim; i++) {
    reporter->AssertEQ(updates->shape[i - mdim->value + kdim], oshape[i]);
  }

  reporter->Assign(types[3], TensorType(data->shape, data->dtype));
  return true;
}

Expr MakeTIDLScatterND(Expr data, Expr indices, Expr updates, String mode) {
  auto attrs = make_object<ScatterNDAttrs>();
  attrs->mode = std::move(mode);
  static const Op& op = Op::Get("tidl_scatter_nd");
  return Call(op, {data, indices, updates}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.tidl_scatter_nd").set_body_typed(MakeTIDLScatterND);

// scatter_nd operator has extern schedules for CPU and GPU devices.
// Fusing extern schedules with Injective schedules leads to errors.
// So, converting the scatter_nd to Opaque to prevent compilation failures
RELAY_REGISTER_OP("tidl_scatter_nd")
    .describe(R"code(Scatter elements or slices from data and store to a tensor
whose shape is defined by indices.

Given data with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}) and indices with shape
(Y_0, ..., Y_{K-1}, M), the output will have shape (X_0, X_1, ..., X_{N-1}).
)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .add_argument("updates", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("TIDLScatterND", TIDLScatterNDRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
} //namespace relay
} //namespace tvm
