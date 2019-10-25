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
"""External function interface to TIDL-API libraries."""
"""my_sort operator"""
import tvm
from tvm import api
from .util import get_const_tuple
from __future__ import absolute_import as _abs
from .. import api as _api, intrin as _intrin

def my_matadd(lhs, rhs, transa=False, transb=False, **kwargs):
    """Create an extern op that compute matrix addition of A and rhs with CrhsLAS

    This function serves as an example on how to call external libraries.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    n = lhs.shape[1] if transa else lhs.shape[0]
    m = rhs.shape[0] if transb else rhs.shape[1]
    return _api.extern(
        (n, m),
        [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.tidl.my_matadd", ins[0], ins[1], outs[0]
        ),
        name="C",
        **kwargs
    )


def my_sort(data, axis=-1, is_ascend=1, dtype="float32", test_new_attr="my_sort_test_new_attr"):
    """Performs sorting along the given axis and returns an array
    of indices having the same shape as an input array that index
    data in sorted order.

    Parameters
    ----------
    data : tvm.Tensor
        The input tensor.

    axis : int, optional
	    Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.Tensor
        Sorted index tensor.

    Example
    --------
    .. code-block:: python

        # An example to use argsort
        dshape = (1, 5, 6)
        data = tvm.placeholder(dshape, name="data")
        axis = 0
        is_ascend = False
        out = argsort(data, axis=axis, is_ascend=is_ascend)
        np_data = np.random.uniform(dshape)
        s = topi.generic.schedule_argsort(out)
        f = tvm.build(s, [data, out], "llvm")
        ctx = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f(tvm_data, tvm_out)
    """
    print("DJDBG tidl.py dedicated, is invoked (my_sort)!")
    data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_buf = api.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
    out = \
         tvm.extern(data.shape,
                    [data],
                    lambda ins, outs: tvm.call_packed(
                        "tvm.contrib.tidl.my_sort", ins[0],
                        outs[0], axis, is_ascend, test_new_attr),
                    dtype=dtype,
                    in_buffers=[data_buf],
                    out_buffers=out_buf,
                    name="argsort_cpu",
                    tag="argsort_cpu")
    return out

