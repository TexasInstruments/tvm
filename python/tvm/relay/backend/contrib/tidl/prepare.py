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

def get_call_op_name(call):
  if isinstance(call, relay.Call) and isinstance(call.op, tvm.ir.Op):
    return call.op.name
  return None

# Function to preserve the span information
def get_expr_with_span(call, span):
    return relay.expr.Call(call.op, call.args, call.attrs, call.type_args, span=span)

class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, call):
        if get_call_op_name(call) == "multiply":
            if isinstance(call.args[1], tvm.relay.expr.Constant):
                data = call.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return super().visit(call.args[0])
            if isinstance(call.args[0], tvm.relay.expr.Constant):
                data = call.args[0].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return super().visit(call.args[1])
        return super().visit_call(call)


class RemovePadByZero(ExprMutator):
    """
    Removes pad operators that have pad of 0 in all tuples.
    Cases seen in dmnet after coverted to .onnx.
    """
    def visit_call(self, call):
        if get_call_op_name(call) == "nn.pad":
            if (all([ x[0] == 0 and x[1] == 0 for x in call.attrs.pad_width])):
                return super().visit(call.args[0])
        return super().visit_call(call)

class RemoveCopy(ExprMutator):
    """
    Removes copy operator
    """
    def visit_call(self,call):
        if get_call_op_name(call) == "copy":
            return super().visit(call.args[0])
        return super().visit_call(call)


class RemoveIdentityReshape(ExprMutator):
    """
    Removes reshape operators to same shape as input.
    Cases seen in dmnet after coverted to .onnx.
    """
    def visit_call(self, call):
        if get_call_op_name(call) == "reshape":
            # Extract shape from first arg
            orig_shape = call.args[0].checked_type.shape
            desired_shape = call.attrs.newshape
            # Compare input tensor's shape to reshape attribute and if the
            # shapes are the same, remove the reshape operator.
            if len(orig_shape) == len(desired_shape) and \
               all([ o == d for o, d in zip(orig_shape, desired_shape)]):
                    return super().visit(call.args[0])
        return super().visit_call(call)

class RemoveTrainingOperators(ExprMutator):
    """
    Removes operators that apply to network training but not to inference.
    """
    # Dropout layer produces a tuple
    def visit_tuple_getitem(self, t):
        expr = t.tuple_value
        if t.index == 0  and get_call_op_name(expr) in ["nn.dropout", "nn.dropout_raw"]:
            return super().visit(expr.args[0])
        return super().visit_tuple_getitem(t)


class RemoveIdentityResize(ExprMutator):
    """
    Removes resize2d to the same size as input
    """
    def visit_call(self, call):
        if get_call_op_name(call) == "image.resize2d":
            new_h, new_w = call.attrs.size
            old_shape = call.args[0].checked_type.shape
            old_h, old_w = old_shape[2:4] if call.attrs.layout == "NCHW" else old_shape[1:3]
            if old_h == new_h and old_w == new_w:
               print("\n[TVM Optimize]: Removing identity resize2d operator")
               return super().visit(call.args[0])
        return super().visit_call(call)


class ConvertMaxMinToClip(ExprMutator):
    """
    Convert maximum(), minimun() sequence to clip operator
    ONNX v11 frontend has created the following Relay IR for ONNX Clip operator:
      %2 = maximum(%x.1, -4f /* ty=float32 */) /* ty=Tensor[(1, 3, 224, 224), float32] */;
      %3 = minimum(%2, 4f /* ty=float32 */) /* ty=Tensor[(1, 3, 224, 224), float32] */;
    Convert the above example to %3 = clip(%x.1, -4f, 4f)
    """
    def visit_call(self, call):
        if get_call_op_name(call) == "minimum" and isinstance(call.args[1], relay.expr.Constant):
            arg0 = call.args[0]
            if get_call_op_name(arg0) == "maximum" and isinstance(arg0.args[1], relay.expr.Constant):
                max_val = call.args[1].data.asnumpy()
                min_val = arg0.args[1].data.asnumpy()
                if min_val.shape == () and max_val.shape == ():
                    clip = tvm.relay.clip(super().visit(arg0.args[0]), min_val.item(), max_val.item())
                    return get_expr_with_span(clip, call.span) if hasattr(call, 'span') else clip
        return super().visit_call(call)

# (deprecated) add layers with addition to a constant were converted to biasadd to get later converted to broadcast layer by tidl
# now support for all such add layers is added and the below class is not required anymore
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
        if get_call_op_name(call) == "add":
            val, axis = self.get_broadcasting_constant_axis(call)
            if val is not None and axis is not None:
                return relay.nn.bias_add(super().visit(call.args[0]), val, axis=axis)

        return super().visit_call(call)

class RemoveIdentityClip(ExprMutator):
    """
    Removes clip operators that clip uint8 input to uint8 range (0, 255), which is identity op
    """
    def visit_call(self, call):
        if get_call_op_name(call) == 'clip':
            if call.args[0].checked_type.dtype == 'uint8' and \
               call.checked_type.dtype == 'uint8' and \
               call.attrs.a_min == 0 and call.attrs.a_max == 255:
                return super().visit_call(call.args[0])
            if call.args[0].checked_type.dtype == 'int8' and \
               call.checked_type.dtype == 'int8' and \
               call.attrs.a_min == -128 and call.attrs.a_max == 127:
                return super().visit_call(call.args[0])
        return super().visit_call(call)

class ConvertConvStride(ExprMutator):
    """
    Converts a conv2d with stride=[1,2] to conv2d with stride=[1,1], followed by a maxpool with stride [1,2].
    """
    def visit_call(self, call):
        if get_call_op_name(call) == 'nn.conv2d':
            if list(call.attrs.strides) == [1,2]:
                attrs = {key: call.attrs[key] for key in call.attrs.keys() if key != 'strides'}
                attrs["strides"] = [1,1]
                conv2d = relay.nn.conv2d(super().visit(call.args[0]), super().visit(call.args[1]), **attrs)
                conv2d = get_expr_with_span(conv2d, call.span) if hasattr(call, 'span') else conv2d
                maxpool = relay.nn.max_pool2d(conv2d, pool_size=[1,1], strides=[1,2])
                return get_expr_with_span(maxpool, call.span) if hasattr(call, 'span') else maxpool
        return super().visit_call(call)

class TransposeScatterND(ExprMutator):
    """
    Converts all instances of  a = transpose(in, [n-1, 0, 1,...,n-2]), followed by
    scatter_nd(..., a, ...) to tidl_scatter_nd(..., in, ...)
    """
    def visit_call(self, call):
        if get_call_op_name(call) == 'scatter_nd' and get_call_op_name(call.args[1]) == 'transpose':
            for i, axis in enumerate(call.args[1].attrs.axes[1:]):
                if axis != i:
                    return super().visit_call(call)
            if call.args[1].attrs.axes[0] != len(call.args[1].attrs.axes) - 1:
                return super().visit_call(call)
            data = super().visit(call.args[0])
            indices = super().visit(call.args[1].args[0])
            updates = super().visit(call.args[2])
            tidl_scatter_nd = relay.tidl_scatter_nd(data, indices, updates, call.attrs.mode)
            return get_expr_with_span(tidl_scatter_nd, call.span) if hasattr(call, 'span') else tidl_scatter_nd

        return super().visit_call(call)

class MergePadLayer(ExprMutator):
    """
    Merges pad layer with the following conv2d layer
    """
    def visit_call(self, call):
        if get_call_op_name(call) == 'nn.conv2d':
            if get_call_op_name(call.args[0]) == 'nn.pad':
                pad_width = call.args[0].attrs.pad_width
                attrs = {key: call.attrs[key] for key in call.attrs.keys()}
                attrs['padding'] = (pad_width[2][0], pad_width[3][0], pad_width[2][1], pad_width[3][1]) # tlbr
                conv2d = relay.nn.conv2d(super().visit(call.args[0].args[0]), super().visit(call.args[1]), **attrs)
                return get_expr_with_span(conv2d, call.span) if hasattr(call, 'span') else conv2d
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
    mod['main'] = RemovePadByZero().visit(mod['main'])
    mod['main'] = RemoveCopy().visit(mod['main']) # removing copy layer as(idenity layer is remppaed to copy and it just puts input to output)
    mod = relay.transform.InferType()(mod)
    mod['main'] = RemoveIdentityReshape().visit(mod['main'])
    mod['main'] = RemoveTrainingOperators().visit(mod['main'])
    mod['main'] = ConvertMaxMinToClip().visit(mod['main'])
    mod = relay.transform.InferType()(mod)
    mod['main'] = RemoveIdentityResize().visit(mod['main'])
    mod = relay.transform.InferType()(mod)
    # mod['main'] = ConvertBroadcastAddtoBiasAdd().visit(mod['main'])
    mod['main'] = ConvertConvStride().visit(mod['main'])
    mod['main'] = TransposeScatterND().visit(mod['main'])
    mod['main'] = MergePadLayer().visit(mod['main'])


    if has_qnn_ops:
        mod = relay.transform.InferType()(mod)
        mod['main'] = RemoveIdentityClip().visit(mod['main'])
    # Removing redundant outputs
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    mod = relay.transform.DynamicToStatic()(mod)

    return mod

odpostproc_operator_registered = False
def prune_graph_for_ODPostProc_inputs(mod: tvm.IRModule,
        ODPostProc_inputs: typing.List[str],
        od_output_shapes: typing.List[typing.Tuple],
        od_output_dtypes: typing.List[str]) -> tvm.IRModule:
    """Prune the graph to compute the ODPostProc inputs only.
       PostProc part of the graph will be pruned and the actual processing
       will be added by TIDL import using Meta Arch automatically.
    """
    import functools
    from tvm import topi
    from tvm import te
    from tvm.target import generic_func, override_native_generic_func
    from tvm.relay.op import op as reg
    from tvm.relay.op import strategy as _strategy
    from tvm.relay.op.op import OpStrategy, OpPattern

    def traverse_expr(node, node_dict, names):
        if isinstance(node, relay.expr.Call) and node.span.source_name.name in names:
            # post visit: the last node with the same name wins
            node_dict[node.span.source_name.name] = node

    def tidl_odpostproc_type_rel(arg_types, attrs):
        nonlocal od_output_shapes
        nonlocal od_output_dtypes
        output_types = [ relay.TensorType(out, dtype)
                         for out, dtype in zip(od_output_shapes, od_output_dtypes) ]
        return relay.TupleType(output_types)

    def tidl_odpostproc(inputs):
        return relay.expr.Call(reg.get("tidl_odpostproc"), inputs)

    def tidl_odpostproc_macs(call):
        return 2**48  # a big MAC count, always offloaded to TIDL (i.e. not pruned)

    def register_tidl_postproc_as_supported():
        @tvm.ir.register_op_attr("tidl_odpostproc", "target.tidl")
        def _func_wrapper(expr):
            return True
        return _func_wrapper

    @override_native_generic_func("tidl_odpostproc_strategy")
    def tidl_odpostproc_strategy(attrs, inputs, out_type, target):
        "pseduo strategy for tidl_odpostproc on host, produce all 0s"

        def pseudo_tidl_odpostproc_compute(attrs, inputs, out_type):
            out_shapes = [t.shape for t in out_type.fields]
            out_dtypes = [t.dtype for t in out_type.fields]

            def gen_ir(shape, dtype, out):
                buf_size = 1
                for dim_size in shape:
                    buf_size *= dim_size
                ib = tvm.tir.ir_builder.create()
                out_buf = ib.buffer_ptr(out)
                with ib.for_range(0, buf_size, "fused") as fused:
                    out_buf[fused] = 0.0 if dtype == "float32" else 0
                return ib.get()

            outputs = []
            for i in range(len(out_shapes)):
                out_buf = tvm.tir.decl_buffer(out_shapes[i], out_dtypes[i],
                                              f"od_out_buf_{i}", data_alignment=8)
                out_i = te.extern([out_shapes[i]], [],
                                  lambda ins, outs: gen_ir(out_shapes[i], out_dtypes[i], outs[0]),
                                  dtype=[out_dtypes[i]],
                                  in_buffers=[],
                                  out_buffers=[out_buf],
                                  name="pseudo_tidl_odpostproc", tag="pseudo_tidl_odpostproc_host",
                                 )
                outputs.append(out_i)
            return outputs

        from tvm.relay.op.op import OpStrategy, OpPattern
        strategy = OpStrategy()
        strategy.add_implementation(
            pseudo_tidl_odpostproc_compute,
            _strategy.wrap_topi_schedule(topi.generic.schedule_extern),
            name="pseudo_tidl_odpostproc",
        )
        return strategy

    # Create an operator that takes ODPostProc input nodes and returns TIDL processed outputs
    global odpostproc_operator_registered
    if not odpostproc_operator_registered:
        op_name = "tidl_odpostproc"
        reg.register(op_name)
        reg.get(op_name).set_num_inputs(len(ODPostProc_inputs))
        for i in range(len(ODPostProc_inputs)):
            reg.get(op_name).add_argument(f"ODPostProc_input{i}", "Tensor", "ODPostProc input")
        reg.get(op_name).add_type_rel(op_name, tidl_odpostproc_type_rel)
        reg.get(op_name).set_support_level(10)
        reg.get(op_name).set_attr("FMacCount", tidl_odpostproc_macs)
        reg.register_strategy(op_name, tidl_odpostproc_strategy)
        reg.register_pattern("tidl_odpostproc", OpPattern.OPAQUE)
        register_tidl_postproc_as_supported()
        odpostproc_operator_registered = True

    # Find relay graph nodes that correspond to ODPostProc_inputs names
    names_to_nodes = {}
    traverse_func = functools.partial(traverse_expr,
                                      node_dict=names_to_nodes, names=ODPostProc_inputs)
    relay.analysis.post_order_visit(mod['main'], traverse_func)
    ODPostProc_inputs_nodes = [ names_to_nodes[k] for k in ODPostProc_inputs ]

    # Create a "pseudo" relay layer to represent TIDL OD PostProcessing, return tuple
    new_body = tidl_odpostproc(ODPostProc_inputs_nodes)
    od_outputs = [ relay.expr.TupleGetItem(new_body, i) for i in range(len(od_output_shapes)) ]
    new_body = relay.expr.Tuple(od_outputs)

    main_func = mod['main']
    new_func = relay.Function(params=main_func.params, body=new_body, attrs=main_func.attrs)
    mod['main'] = new_func
    return mod
