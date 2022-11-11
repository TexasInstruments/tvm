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
"""Definition of c7x operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import logging

from tvm import topi
from tvm.te import SpecializedCondition
from .generic import *
from .. import op as _op

logger = logging.getLogger('strategy')

@schedule_pool.register("c7x")
def schedule_pool_c7x(attrs, outs, target):
    """schedule pooling ops for c7x"""
    with target:
        return topi.c7x.schedule_pool(outs, attrs.layout)

@schedule_injective.register(["c7x"])
def schedule_injective_c7x(_, outs, target):
    """schedule injective ops for c7x"""
    with target:
        return topi.c7x.schedule_injective(outs)

@schedule_concatenate.register(["c7x"])
def schedule_concatenate(attrs, outs, target):
    """Schedule concatenate op for c7x"""
    with target:
        return topi.c7x.schedule_injective(outs)


def resize2d_strategy(attrs, inputs, out_type, target):
    """C7x resize2d strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        topi.c7x.resize.compute_resize2d,
        schedule_injective_c7x,
        name="c7x_resize2d",
        plevel=15,
    )
    return strategy
# The default resize2d strategy (compute+schedule) is defined in tvm.relay.op.image/_image.py:
#     @reg.register_compute("image.resize2d")
#     def compute_resize2d(attrs, inputs, out_type):
#         ... ... ...
#     reg.register_injective_schedule("image.resize2d")
# We need the default strategy first, then register the customized one for c7x
from tvm.relay.op import image as image
_op.get("image.resize2d").get_attr("FTVMStrategy").register(resize2d_strategy,
                                                            "c7x", allow_override=True)
