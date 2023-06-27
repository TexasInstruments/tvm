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
# pylint: disable=invalid-name
"""ScatterND operator"""
from tvm import te, tir, get_global_func  # hide redefinition of min and max
from tvm.tir import expr
from tvm.runtime import DataType
from ..tir import decl_buffer, ir_builder, AssertStmt, StringImm, Evaluate, ceildiv, Var, IntImm



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


def tidl_scatter_nd(data, indices, updates, mode):
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
            index = 0  # This is x_M, .. x_{N-1} part of the index into out.
            # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}] part
            # of the index into out.
            for l in reversed(range(indices_ptr.shape[-1].value)):
                # indices[i * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                index += offset * indices[i * fused_indices_dimension + l]
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
            #update = 
            if mode == "add":
                out[index + j] += updates[i * fused_data_dimension + j]
            elif mode == "update":
                out[index + j] = updates[i * fused_data_dimension + j]
            else:
                raise NotImplementedError("scatter_nd mode not in [update, add]:", mode)

        # Small change from generic version to not generate an inner loop when unnecesary
        # allowing the streaming engine to work for the innermost loop

        GetCurrentTIDLContext = get_global_func("tidl.GetCurrentTIDLContext")
        ctx = GetCurrentTIDLContext()
        elem_bytes = DataType(dtype).bits // 8
        vector_length = 8 * 4 // elem_bytes if ctx.platform == "AM62A" else 16 * 4 // elem_bytes
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
                        scatter(index, i, outer * vector_length + inner)
                if outer_max * vector_length < fused_data_dimension:
                    with ib.for_range(0, fused_data_dimension - (outer_max * vector_length), name="j") as j:
                        scatter(index, i, j + outer_max * vector_length)
                
        return ib.get()

    out_buf = decl_buffer(data.shape, data.dtype, "out_buf")
    sched = te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0], data.dtype),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="tidl_scatter_nd_c7x",
        tag="tidl_scatter_nd_c7x",
    )
    return sched
