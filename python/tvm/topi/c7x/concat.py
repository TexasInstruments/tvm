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
# pylint: disable=invalid-name, unused-variable
"""Schedule for C7x injective operators"""
from typing import Union, List, Optional

import tvm
from tvm import te
from ..utils import get_const_int
import numpy as np

def compute_concatenation(data : tuple[tvm.te.Tensor,...], axis : Optional[int] = 0 ):
    """ C7x TIR-based compute for the concat operator.
    The purpose of this compute function is to remove the if-then-else that is
    normally in the inner loop for concat. The if-then-else is not conducive to
    vectorization, the use of the SE, and efficient software pipelining. This
    method requires n loop nests for n input tensors.

    At this time (2025/05), TVM uses SE and SA on this computation, but the use
    of SE and SA inhibits vectorization in the C7000 compiler. So there is
    room for improvement.

    Parameters
    ----------
    data : tuple of tvm.te.Tensor
        The arrays to concatenate

    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    output : tvm.te.Tensor
        The resulting concatenated tensor
 
    """
    # This implementation borrows heavily from python/tvm/topi/x86/concat.py.

    # Step 1:
    #   Find the dimensions below the axis to be concatenated and obtain the
    #   product of those dimension sizes. For example, if the shape of the
    #   (say) two input tensors are (all) [5, 4, 3, 2] and the concat axis is 0,
    #    the product will be 120. If the shape of the input tensors are
    #   [5, 4, 3, 2] and the concat axis is 1, the product will be 24
    #   because of (4 * 3 * 2).
    #   Note that the shape of each tensor must be the same except for the axis
    #   that is being concatenated.
    # Step 2:
    #   Compute the cumulative sum of the inner dimensions. This will be
    #   used later to compute addressing positions in the output tensor
    #   to facilitate the relative concatenation of each input tensor.
    #   For example, for three input tensors of shape [5, 4, 3, 2], and
    #   a concatenation axis of 1, inner_cumsum will be [0, 24, 48]
    # Step 3:
    #   If the given axis is negative, find the axis to use by counting from
    #   the inner (rightmost) dimension
    # Step 4:
    #    Find each tensor's size on the concatenation axis and then
    #    compute the size of the concatenated dimension.
    # Step 5:
    #    Find the shape of the output tensor by merging the shape to
    #    the left of the merge axis, the joined axis size, and the
    #    shape to the right of the merge axis.
    # Step 6:
    #    Get the product of the dimensions to the left and to the right
    #    of the concatenation axis. The "left" value will include the
    #    summed dimension of the axis to be concatenated.
    # Step 7:
    #    Step 7, generate a loop nests for each tensor. Each loop nest
    #    will perform a copy of the input tensor to the correct place in
    #    the output tensor.
    #    If the axis to be concatenated is not the first dimension, we need
    #    to generate a set of doubly-nested loops, one for the outer set of
    #    dimensions "above" the concatenation axis and an inner loop for
    #    the set of dimensions including and below the concatenation axis.
    #    The top-most "then" case of the if outer > 1 creates an outer loop
    #    for the product of the outer dimensions (left of the concatenation
    #    axis). In the example above, outer is 5.
    # Step 7a:
    #    Calculate the offset within the input tensor due to iterating
    #    through the outer dimensions of the tensor.
    # Step 7b:
    #    Calculate the offset within the output tensor due to iterating
    #    through the outer dimensions of the tensor.
    # Step 7c:
    #    Create an inner for loop that strides through the product of the inner
    #    dimensions (right-of and including the concatenation axis)
    # Step 7d:
    #    Create the inner loop that strides through the inner dimensions,
    #    including the concatenation axis. Create the expression that performs
    #    the copy from the input tensor to the appropriate spot in the output
    #    tensor. 

    dtype = data[0].dtype

    # Step 1, Find product of inner dimensions below the axis to concatenate
    inner_dim_prod = [int(np.prod(i.shape[axis:])) for i in data]

    # Step 2, Compute cumulative sum of the inner dimensions.
    inner_cumsum = [0, *np.cumsum(inner_dim_prod, dtype="int64")[0:-1]]

    # Step 3, If negative axis, find axis to use from end of shape
    if axis < 0:
        axis += len(data[0].shape)

    # Step 4, compute the size of the concatenated dimension
    concat_axis_sizes = [int(t.shape[axis]) for t in data]
    join_size = int(np.sum(concat_axis_sizes))

    # Step 5, Construct the output tensor shape
    out_shape = data[0].shape[:axis] + [join_size] + data[0].shape[axis + 1 :]

    # Step 6, find the product of the dimensions to the left and right of the
    # concatenation axis.
    right_val = np.prod(out_shape[axis:])
    left_val = np.prod(out_shape[:axis])

    outer = get_const_int(int(left_val))
    inner = get_const_int(int(right_val))

    def gen_ir(data_bufs, out_buf, outer, inner):
        i_b = tvm.tir.ir_builder.create()
        data_bufs1 = [i_b.buffer_ptr(data_buf) for data_buf in data_bufs]
        out_buf = i_b.buffer_ptr(out_buf)

        # Step 7, generate a loop nest for each tensor.

        # Handle case where concatenation axis is *not* topmost dimension
        if outer > 1:
            for i in range(len(data)):
                with i_b.for_range(0, outer, name="outer_cntr", kind="serial") as outer_cntr:
                    # Step 7a&b: Calculate the outer dimensional offset within
                    # the input tensor (offset) and output tensor (pos).
                    offset = outer_cntr * inner_dim_prod[i]
                    pos    = outer_cntr * inner

                    # Step 7c&d: Create inner loop where we copy the input tensor
                    with i_b.for_range(0, inner_dim_prod[i], name="j", kind = "serial") as j:
                        out_buf[pos + inner_cumsum[i] + j] = data_bufs1[i][offset + j]
        # Handle case where concatenation axis is topmost dimension
        else:
            for i in range(len(data)):
                with i_b.for_range(0, inner_dim_prod[i], name="j", kind="serial") as j:
                    out_buf[inner_cumsum[i] + j] = data_bufs1[i][j]

        return i_b.get()

    return te.extern(
        [out_shape],
        list(data),
        lambda ins, outs: gen_ir(ins, outs[0], outer, inner),
        tag="concat",
        dtype=dtype,
        name="concatenate_ext_c7x",
    )

