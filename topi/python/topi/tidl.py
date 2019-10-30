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
def TidlMatAdd(lhs, rhs, kernel_attr="djdbg_kernel_none2"):
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
    #print("DJDBG: in TVM, TidlMatAdd:" + kernel_attr)
    n = lhs.shape[0]
    m = rhs.shape[1]
    return tvm.extern(
        (n, m),
        [lhs, rhs],
        lambda ins, outs: tvm.call_packed(
            "tvm.contrib.tidl.my_matadd", ins[0], ins[1], outs[0], kernel_attr
        ),
        name="my_matadd_name_cpu",
        tag="my_matadd_tag_cpu",
    )


@tvm.target.generic_func
def TidlSort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32", test_new_attr="default_for_test_new_attr"):
   #print("DJDBG: in TVM, TidlSort")

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
                    name="my_sort_name_cpu",
                    tag="my_sort_tag_cpu")

   return out

@tvm.target.generic_func
def TidlInference(data, num_labels=1001, inference_attr="default_for_test_inference_attr"):
   print("DJDBG: in TVM, TidlInference")

   data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
   out_shape = [data.shape[0], num_labels]
   out_buf = api.decl_buffer(out_shape, "float32", "out_buf", data_alignment=8)

   out = \
         tvm.extern(data.shape,
                    [data],
                    lambda ins, outs: tvm.call_packed(
                       "tvm.contrib.tidl.my_inference", ins[0],
                        outs[0], num_labels, inference_attr),
                    in_buffers=[data_buf],
                    out_buffers=out_buf,
                    name="my_inference_name_cpu",
                    tag="my_inference_tag_cpu")

   return out


