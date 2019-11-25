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

@register_schedule("TidlInference")
def schedule_TidlInference(_, outs, target):
    """Schedule definition of TidlInference"""
    with target:
        return topi.generic.schedule_tidlinference(outs)


@register_compute("TidlInference")
def compute_TidlInference(attrs, inputs, _, target):
    """Compute definition of tidlinference"""
    num_labels = get_const_int(attrs.num_labels)
    inference_attr = attrs.inference_attr
    return [topi.TidlInference(inputs[0], num_labels=num_labels, inference_attr=inference_attr)]


register_pattern("TidlInference", OpPattern.OPAQUE)


