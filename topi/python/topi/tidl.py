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
import tvm
from tvm import api
from .util import get_const_tuple

@tvm.target.generic_func
def TidlMatAdd(lhs, rhs, transa=False, transb=False, **kwargs):
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


@tvm.target.generic_func
def TidlSort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
   print("DJDBG: in TidlSort")

   data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
   out_buf = api.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
   out = \
         tvm.extern(data.shape,
                    [data],
                    lambda ins, outs: tvm.call_packed(
                       "tvm.contrib.tidl.my_sort", ins[0],
                        outs[0], axis, is_ascend),
                    dtype=dtype,
                    in_buffers=[data_buf],
                    out_buffers=out_buf,
                    name="my_sort_cpu",
                    tag="my_sort_cpu")

   return out
