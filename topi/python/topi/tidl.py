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
def TidlInference(data, num_labels=1001, inference_attr="default_for_test_inference_attr"):
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


