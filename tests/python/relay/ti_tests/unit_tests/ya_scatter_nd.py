#!/usr/bin/env python3

""" Example file to show process of defining a new relay op, adding it to a relay module,
    defining a schedule for it to run during TIDL calibration, and defining a schedule to
    call an external library function on c7x
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, platform, artifacts_folders, check_occurrence
from unit_utils import build_and_set_ext_lib

model_name = "ya_scatter_nd"
artifacts_dir, artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [("data", (2240, 64)),
                ("indices_i2240", (1, 58080, 1)),
                ("updates", (1, 58080, 64))]
weight_shapes = []


def register_new_op():
    import tvm
    from tvm import topi
    from tvm import relay, te
    from tvm.tir import expr, decl_buffer, ir_builder
    from tvm.target import generic_func, override_native_generic_func
    from tvm.relay.op import op as reg
    from tvm.relay.op import strategy as _strategy
    from tvm.relay.op.op import OpStrategy, OpPattern
    from tvm.runtime import DataType

    def _verify_tidl_scatter_nd_inputs(data, indices, updates):
        mdim = int(indices.shape[-1])
        assert mdim <= len(data.shape), (
            f"The last dimension of the indices ({mdim}) must be less than or equal to "
            f"the length of the shape of the output ({len(data.shape)})."
        )
        for i in range(len(indices.shape) - 1):
            if isinstance(indices.shape[i], expr.Var) or isinstance(updates.shape[i], expr.Var):
                continue
            assert indices.shape[i] == updates.shape[i], (
                f"Dimension of indices[{i}] ({indices.shape[i]}) must equal dimension of "
                f"updates[{i}] ({updates.shape[i]})."
            )
        for i in range(mdim, len(data.shape)):
            data_ind = i - mdim + len(indices.shape) - 1
            if isinstance(updates.shape[data_ind], expr.Var) or isinstance(data.shape[i], expr.Var):
                continue
            assert updates.shape[data_ind] == data.shape[i], (
                f"Dimension of updates[{data_ind}] ({updates.shape[data_ind]}) must equal dimension "
                f"of out_shape[{i}] ({data.shape[i]})."
            )

        assert (
            "int" in indices.dtype
        ), f"Indices must be a tensor of integers, but its elements are {indices.dtype}."

    # define a generic strategy for TIDL calibration on x86

    @override_native_generic_func("ya_scatter_nd_strategy")
    def ya_scatter_nd_strategy(attrs, inputs, out_type, target):
        def ya_scatter_nd_compute(attrs, inputs, out_type):
            """Scatter elements from a n-dimension array.

            Given updates with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), indices with shape
            (Y_0, ..., Y_{K-1}, M), and output copied from data with shape (X_0, X_1, ..., X_{N-1}),
            scatter_nd computes

            .. code-block::

                output[indices[y_0, ..., y_{K-1}, 0],
                    ...,
                    indices[y_0, ..., y_{K-1}, M-1],
                    x_M,
                    ...,
                    x_{N-1}
                    ] = f(output[...], updates[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}])

            where the update function f is determinted by the mode.

            Parameters
            ----------
            data : tvm.te.Tensor
                The source array.

            indices : tvm.te.Tensor
                The indices of the values to extract.

            updates : tvm.te.Tensor
                The updates to apply at the Indices

            mode : string
                The update mode for the algorithm, either "update" or "add"
                If update, the update values will replace the input data
                If add, the update values will be added to the input data

            Returns
            -------
            ret : tvm.te.Tensor
            """
            data = inputs[0]
            indices = inputs[1]
            updates = inputs[2]
            mode = attrs.mode

            _verify_tidl_scatter_nd_inputs(data, indices, updates)

            def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr, dtype):
                ib = ir_builder.create()

                data = ib.buffer_ptr(data_ptr)
                indices = ib.buffer_ptr(indices_ptr)
                updates = ib.buffer_ptr(updates_ptr)
                out = ib.buffer_ptr(out_ptr)

                fused_shape = 1
                for i in data_ptr.shape:
                    fused_shape *= i
                with ib.for_range(0, fused_shape) as i:
                    out[i] = data[i]

                # We combine all the indices dimensions but the first one into a single
                # dimension so we can iterate it in single loop instead of an arbitrary
                # number of loops. We do the same thing for all the data dimensions.
                fused_indices_dimension = indices_ptr.shape[-1]
                # for i in indices_ptr.shape[:-1]:
                #     fused_indices_dimension *= i

                fused_updates_dimension = 1
                for i in updates_ptr.shape[:len(indices_ptr.shape) - 1]:
                    fused_updates_dimension *= i

                # Generic compute has a bug here. It calculates the fused data dimension using
                # `data_ptr.shape[len(indices_ptr.shape) - 1 :]`, which does not properly calculate
                # insertion points for scatter, this fixes it as indices_ptr.shape[0] is the effective
                # "depth" of our insertions. `data_ptr.shape[indices_ptr.shape[0].value:]` is guaranteed
                # to be equal to `updates_ptr.shape[len(indices_ptr.shape) - 1 :]`
                fused_data_dimension = 1
                for i in data_ptr.shape[indices_ptr.shape[-1].value:]:
                    fused_data_dimension *= i

                def get_index(i):
                    offset = fused_data_dimension
                    # This is x_M, .. x_{N-1} part of the index into out.
                    index = 0
                    # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}] part
                    # of the index into out.
                    for l in reversed(range(indices_ptr.shape[-1].value)):
                        # indices[i * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                        index += offset * \
                            indices[i * fused_indices_dimension + l]
                        # ib.emit(
                        #     AssertStmt(
                        #         indices[i + l * fused_indices_dimension] < data_ptr.shape[l],
                        #         StringImm("index out of bounds"),
                        #         Evaluate(0),
                        #     )
                        # )
                        offset *= data_ptr.shape[l]
                    return index

                def scatter(index, i, j):
                    # update =
                    if mode == "add":
                        out[index + j] += updates[i * fused_data_dimension + j]
                    elif mode == "update":
                        out[index + j] = updates[i * fused_data_dimension + j]
                    else:
                        raise NotImplementedError(
                            "scatter_nd mode not in [update, add]:", mode)

                # Small change from generic version to not generate an inner loop when unnecesary
                # allowing the streaming engine to work for the innermost loop

                elem_bytes = DataType(dtype).bits // 8
                vector_length = 16 * 4 // elem_bytes
                outer_max = fused_data_dimension // vector_length
                if fused_data_dimension == 1:
                    with ib.for_range(0, fused_updates_dimension, name="i") as i:
                        index = get_index(i)
                        scatter(index, i, 0)

                else:
                    with ib.for_range(0, fused_updates_dimension, name="i") as i:
                        index = get_index(i)
                        with ib.for_range(0, outer_max, name="outer") as outer:
                            with ib.for_range(0, vector_length, name="inner", kind="vectorize") as inner:
                                scatter(index, i, outer *
                                        vector_length + inner)
                        if outer_max * vector_length < fused_data_dimension:
                            with ib.for_range(0, fused_data_dimension - (outer_max * vector_length), name="j") as j:
                                scatter(index, i, j +
                                        outer_max * vector_length)

                return ib.get()

            out_buf = decl_buffer(data.shape, data.dtype, "out_buf")
            sched = te.extern(
                [data.shape],
                [data, indices, updates],
                lambda ins, outs: gen_ir(
                    ins[0], ins[1], ins[2], outs[0], data.dtype),
                dtype=data.dtype,
                out_buffers=[out_buf],
                name="ya_scatter_nd_c7x",
                tag="ya_scatter_nd_c7x",
            )
            return [sched]

        strategy = OpStrategy()
        strategy.add_implementation(
            ya_scatter_nd_compute,
            _strategy.wrap_topi_schedule(topi.generic.schedule_extern),
            name="ya_scatter_nd.generic",
        )
        return strategy

    def ya_scatter_nd_macs(call):
        return 0  # a small MAC count, not offloaded to tidl

    # scatter_nd
    def ya_scatter_nd_type_rel(arg_types, attrs):
        inputa_type = arg_types[0]
        return relay.TensorType(inputa_type.shape, inputa_type.dtype)

    def register_ya_scatter_nd_as_supported():
        @tvm.ir.register_op_attr("ya_scatter_nd", "target.tidl")
        def _func_wrapper(expr):
            return True
        return _func_wrapper

    op_name = "ya_scatter_nd"
    reg.register(op_name)
    reg.get(op_name).set_num_inputs(3)
    reg.get(op_name).add_argument(f"data", "Tensor", "")
    reg.get(op_name).add_argument(f"indices", "Tensor", "")
    reg.get(op_name).add_argument(f"updates", "Tensor", "")
    reg.get(op_name).add_type_rel(op_name, ya_scatter_nd_type_rel)
    reg.get(op_name).set_support_level(10)
    reg.get(op_name).set_attr("FMacCount", ya_scatter_nd_macs)
    reg.get(op_name).set_attrs_type_key("DictAttrs")
    reg.register_strategy(op_name, ya_scatter_nd_strategy)
    reg.register_pattern(op_name, OpPattern.OPAQUE)
    register_ya_scatter_nd_as_supported()


def add_c7x_ya_scatter_nd_strategy():
    import tvm
    from tvm import relay
    from tvm import topi
    from tvm import te
    from tvm.relay.op import op as reg
    from tvm.relay.op.op import OpStrategy, OpPattern
    from tvm.tir import decl_buffer

    def compute_c7x_ya_scatter_nd(attrs, inputs, out_type):
        data = inputs[0]
        indices = inputs[1]
        updates = inputs[2]

        out = tvm.te.extern(
            [data.shape],
            [data, indices, updates],
            lambda ins, outs: tvm.tir.call_packed(
                "scatter_nd_ext", ins[0], ins[1], ins[2], outs[0]),
            dtype=data.dtype,
            name="ya_scatter_nd_c7x",
            tag="ya_scatter_nd_c7x",
        )
        return [out]

    def wrap_c7x_ya_scatter_nd_schedule(topi_schedule):
        def wrapper(attrs, outs, target):
            with target:
                return topi_schedule(outs)
        return wrapper

    def ya_scatter_nd_strategy_c7x(attrs, inputs, out_type, target):
        strategy = OpStrategy()
        strategy.add_implementation(
            compute_c7x_ya_scatter_nd,
            wrap_c7x_ya_scatter_nd_schedule(topi.generic.schedule_extern),
            name="ya_scatter_nd_c7x",
            plevel=15
        )
        return strategy

    reg.get("ya_scatter_nd").get_attr("FTVMStrategy").register(ya_scatter_nd_strategy_c7x,
                                                               "c7x", allow_override=True)
    reg.register_pattern("ya_scatter_nd", OpPattern.OPAQUE, level=15)


def compile_model():
    """Create a relay model, generate reference inputs/outputs, compile it"""
    import tvm
    from tvm import relay
    from tvm.contrib.tidl.compile import compile_relay
    from tvm.relay.expr_functor import ExprMutator

    register_new_op()
    add_c7x_ya_scatter_nd_strategy()

    # define graph/model in relay
    input_vars = [relay.var(name, relay.TensorType(shape,
                                                   "int32" if name.endswith("i2240") else "float32"))
                  for name, shape in input_shapes]

    src_name = "scatter_nd_extern"
    src_dir = os.path.dirname(os.path.realpath(__file__))
    if not build_and_set_ext_lib(src_name, src_dir, artifacts_data_dir):
        return False

    ind_trans = relay.transpose(input_vars[1], axes=[2, 0, 1])
    output1 = relay.scatter_nd(
        input_vars[0], ind_trans, input_vars[2], mode="add")

    func: relay.function.Function = relay.Function(input_vars, output1)
    mod: tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)


    # create a relay pass to replace sequence of transpose->scatter_nd with custom op
    class TransposeYAScatterND(ExprMutator):
        """
        Converts all instances of  a = transpose(in, [n-1, 0, 1,...,n-2]), followed by
        scatter_nd(..., a, ...) to tidl_scatter_nd(..., in, ...)
        """

        def visit_call(self, call):
            from tvm.relay.op import op as reg
            if call.op.name == 'scatter_nd' and isinstance(call.args[1], relay.expr.Call) and \
                    call.args[1].op.name == 'transpose':
                for i, axis in enumerate(call.args[1].attrs.axes[1:]):
                    if axis != i:
                        return super().visit_call(call)
                if call.args[1].attrs.axes[0] != len(call.args[1].attrs.axes) - 1:
                    return super().visit_call(call)
                data = super().visit(call.args[0])
                indices = super().visit(call.args[1].args[0])
                updates = super().visit(call.args[2])
                return relay.expr.Call(reg.get("ya_scatter_nd"), [data, indices, updates], call.attrs)

            return super().visit_call(call)

    # apply pass to relay module
    mod['main'] = TransposeYAScatterND().visit(mod['main'])

    # gen reference inputs/outputs
    gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None

    inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                       gen_new_data=gen_new_data)

    # Compile relay module
    status = compile_relay(mod, weights, inputs, platform,
                           compile_for_device=True, enable_tidl_offload=True, enable_c7x_codegen=True,
                           artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
    if status != 1:
        print("TIDL compilation failed")
        return False

    relay_file = os.path.join(artifacts_dir, "tempDir/relay_graph.import.txt")
    num_ya_scatter_nd = check_occurrence("ya_scatter_nd", relay_file)
    if num_ya_scatter_nd != 1:
        print(
            f"FAIL: num_ya_scatter_nd {num_ya_scatter_nd} != 1 (expected)")
        return False

    return True


def run_model():
    import sys
    sys.path.append("..")
    from infer_model import run_model

    inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                            gen_new_data=False)

    tvm_outputs = run_model(artifacts_dir, inputs, use_dlr=True)

    if not check_reference(tvm_outputs, artifacts_data_dir, maxdiff_ratio=0.03):
        return False

    return True


if __name__ == "__main__":
    if not os.path.exists(artifacts_data_dir):
        os.makedirs(artifacts_data_dir)

    if is_on_target():
        status = run_model()
    else:
        status = compile_model()

    sys.exit(0 if status else 1)
