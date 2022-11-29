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
from .. import strategy as _strategy

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


@scatter_nd_strategy.register(["c7x"])
def scatter_nd_strategy_c7x(attrs, inputs, out_type, target):
    """scatter_nd generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter_nd(topi.c7x.scatter_nd),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="scatter_nd.c7x",
    )
    return strategy


def max_pool2d_1x1_pool_size_strategy(attrs, inputs, out_type, target):
    """C7x max_pool2d strategy"""
    strategy = _op.OpStrategy()
    if attrs.pool_size[0] == 1 and attrs.pool_size[1] == 1 and attrs.layout == 'NCHW' and \
       attrs.dilation[0] == 1 and attrs.dilation[1] == 1 and \
       attrs.padding[0] == 0 and attrs.padding[1] == 0:
        strategy.add_implementation(
            topi.c7x.pooling.compute_max_pool2d_1x1_pool_size,
            schedule_injective_c7x,
            name="c7x_max_pool2d_1x1_pool_size",
            plevel=15,
        )
    else:
        strategy.add_implementation(
            _op.get("nn.max_pool2d").get_attr("FTVMCompute"),
            schedule_pool_c7x,
            name="c7x_max_pool2d",
            plevel=10,
        )
    return strategy
# We need the default strategy first, then register the customized one for c7x
from tvm.relay.op.nn import max_pool2d as max_pool2d
_op.get("nn.max_pool2d").get_attr("FTVMStrategy").register(max_pool2d_1x1_pool_size_strategy,
                                                           "c7x", allow_override=True)


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
