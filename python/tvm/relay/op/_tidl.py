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
"Definition of classic algorithms"
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import topi
from topi.util import get_const_int
from ..op import OpPattern, register_compute, register_schedule, register_pattern

@register_schedule("TidlSort")
def schedule_TidlSort(_, outs, target):
    """Schedule definition of TidlSort"""
    with target:
        return topi.generic.schedule_tidlsort(outs)


@register_compute("TidlSort")
def compute_TidlSort(attrs, inputs, _, target):
    """Compute definition of tidlsort"""
    #print("DJDBG_TidlSort_Attrs:" + str(dir(attrs)))
    axis = get_const_int(attrs.axis)
    is_ascend = bool(get_const_int(attrs.is_ascend))
    dtype = attrs.dtype
    test_new_attr = attrs.test_new_attr
    return [topi.TidlSort(inputs[0], axis=axis, is_ascend=is_ascend, dtype=dtype, test_new_attr=test_new_attr)]


register_pattern("TidlSort", OpPattern.OPAQUE)

@register_compute("TidlMatAdd")
def compute_TidlMatAdd(attrs, inputs, out_type, target):
    """Compute definition of tidlmatadd"""
    #print("DJDBG_TidlMatAdd_Attrs:" + str(dir(attrs)))
    kernel_attr = attrs.kernel_attr
    with target:
        #print("DJDBG_compute: TidlMatAdd")
        return [topi.TidlMatAdd(inputs[0], inputs[1], kernel_attr=kernel_attr)]


@register_schedule("TidlMatAdd")
def schedule_TidlMatAdd(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        #print("DJDBG_schedule: TidlMatAdd")
        return topi.generic.schedule_tidlmatadd(outputs)

register_pattern("TidlMatAdd", OpPattern.OPAQUE)



