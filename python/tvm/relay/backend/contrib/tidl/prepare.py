# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Passes to prepare the Relay graph for partitioning"""

import tvm
from tvm import relay
import typing
import tvm.ir
from tvm.relay.expr_functor import ExprMutator
#from tvm.relay.expr import Tuple, GlobalVar
#from tvm.relay.function import Function

class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, call):
        if call.op.name == "multiply":
            if isinstance(call.args[1], tvm.relay.expr.Constant):
                data = call.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return call.args[0]
            if isinstance(call.args[0], tvm.relay.expr.Constant):
                data = call.args[0].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return call.args[1]
        return super().visit_call(call)


class RemoveTrainingOperators(ExprMutator):
    """
    Removes operators that apply to network training but not to inference.
    """
    # Dropout layer produces a tuple
    def visit_tuple_getitem(self, t):
        expr = t.tuple_value
        if t.index == 0  and \
           isinstance(expr, relay.expr.Call) and \
           expr.op.name in ["nn.dropout", "nn.dropout_raw"]:
            return expr.args[0]
        return super().visit_tuple_getitem(t)

class ConvertMaxMinToClip(ExprMutator):
    """
    Convert maximum(), minimun() sequence to clip operator
    ONNX v11 frontend has created the following Relay IR for ONNX Clip operator:
      %2 = maximum(%x.1, -4f /* ty=float32 */) /* ty=Tensor[(1, 3, 224, 224), float32] */;
      %3 = minimum(%2, 4f /* ty=float32 */) /* ty=Tensor[(1, 3, 224, 224), float32] */;
    Convert the above example to %3 = clip(%x.1, -4f, 4f)
    """
    def visit_call(self, call):
        if call.op.name == "minimum" and isinstance(call.args[1], relay.expr.Constant):
            arg0 = call.args[0]
            if isinstance(arg0, relay.expr.Call) and \
               arg0.op.name == "maximum" and isinstance(arg0.args[1], relay.expr.Constant):
                max_val = call.args[1].data.asnumpy()
                min_val = arg0.args[1].data.asnumpy()
                if min_val.shape == () and max_val.shape == ():
                    return tvm.relay.clip(super().visit(arg0.args[0]), min_val.item(),
                                                                       max_val.item())
        return super().visit_call(call)

class ConvertBroadcastAddtoBiasAdd(ExprMutator):

    def get_broadcasting_constant_axis(self, call : relay.expr.Call):
        """
        Get constant (flattened) and axis if op is broadcasting over a single channel
        return None if not applicable
        """
        arg0 = call.args[0]
        arg1 = call.args[1]
        if isinstance(arg1, relay.Constant):
            const_shape = arg1.checked_type.shape
            tensor_shape = arg0.checked_type.shape

            # Ensure there is only one axis with size != 1 and find its size
            const_size = 1
            for i in range(len(const_shape)-1, -1, -1):
                if const_shape[i] != 1:
                    if const_size == 1:
                        const_size = const_shape[i]
                    else:  # there can only be one dimension to be scaled channel-wise
                        return None, None

            if const_size == 1:
                return None, None

            # Ensure the constant axis size matches the tenor size
            # If it matches, flatten and return the constant and the axis
            val = None
            for i in range(len(tensor_shape)-1, -1, -1):
                if tensor_shape[i] == const_size:
                    val, axis = arg1.data.numpy().flatten(), i
            if val is not None:
                return relay.const(val), axis

        return None, None

    def visit_call(self, call):
        if call.op.name == "add":
            val, axis = self.get_broadcasting_constant_axis(call)
            if val is not None and axis is not None:
                return relay.nn.bias_add(super().visit(call.args[0]), val, axis=axis)

        return super().visit_call(call)

class ConvertArgMaxToKeepDims(ExprMutator):
    """
    Convert argmax() that reduces dims to argmax that keeps dims plus squeeze()
      so that TIDL can import the argmax layer
    TFLite deeplabv3_mnv2_ade20k 8bit quantized example:
      %203 = @tidl_0(%MobilenetV2/MobilenetV2/input) /* ty=Tensor[(1, 512, 512, 32), uint8] */;
      argmax(%203, axis=[3]) /* ty=Tensor[(1, 512, 512), int32] */
    Convert the above example to 
      %204 = argmax(%203, axis=[3], keepdims=True) /* ty=Tensor[(1, 512, 512, 1), int32] */
      squeeze(%204, axis=[3]) /* ty=Tensor[(1, 512, 512), int32] */
    """
    def visit_call(self, call):
        if call.op.name == "argmax" and \
           (not call.attrs.keepdims) and (not call.attrs.exclude) and \
           call.attrs.axis != None and len(call.attrs.axis) == 1:
            argmax = tvm.relay.argmax(call.args[0], axis=call.attrs.axis,
                                      keepdims=True, exclude=False)
            return tvm.relay.squeeze(argmax, axis=call.attrs.axis)
        return super().visit_call(call)

class RemoveIdentityClip(ExprMutator):
    """
    Removes clip operators that clip uint8 input to uint8 range (0, 255), which is identity op
    """
    def visit_call(self, call):
        if call.op.name == 'clip':
            if call.args[0].checked_type.dtype == 'uint8' and \
               call.checked_type.dtype == 'uint8' and \
               call.attrs.a_min == 0 and call.attrs.a_max == 255:
                return super().visit_call(call.args[0])
            if call.args[0].checked_type.dtype == 'int8' and \
               call.checked_type.dtype == 'int8' and \
               call.attrs.a_min == -128 and call.attrs.a_max == 127:
                return super().visit_call(call.args[0])
        return super().visit_call(call)


def prepare_graph_for_partitioning(mod_orig: tvm.IRModule, 
                                  has_qnn_ops: bool, 
                                  params : typing.Dict[str, tvm.nd.NDArray]) -> tvm.IRModule:
    """Prepare the graph for partitioning"""

    mod = relay.transform.RemoveUnusedFunctions()(mod_orig)

    # Bind params so that weights will appear as constants instead of variables
    mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)
    mod = relay.transform.FoldConstant()(mod)
    mod['main'] = RemoveMultiplyByOne().visit(mod['main'])
    mod['main'] = RemoveTrainingOperators().visit(mod['main'])
    mod['main'] = ConvertMaxMinToClip().visit(mod['main'])
    mod['main'] = ConvertArgMaxToKeepDims().visit(mod['main'])
    mod = relay.transform.InferType()(mod)
    mod['main'] = ConvertBroadcastAddtoBiasAdd().visit(mod['main'])

    if has_qnn_ops:
        mod = relay.transform.InferType()(mod)
        mod['main'] = RemoveIdentityClip().visit(mod['main'])
    # Removing redundant outputs
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    mod = relay.transform.DynamicToStatic()(mod)

    return mod