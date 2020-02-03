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
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm, relay
import tvm
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

from tvm.relay.op.annotation import tidlAnnotation

import tidl_relay_import as tidl

model_path = './mobileNet1/mobilenet_v1_1.0_224_frozen_opt.pb'
out_node   = 'MobilenetV1/Predictions/Reshape'

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
layout = None
shape_dict = {'input': (1, 224, 224, 3)}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

print("=========Tensorflow protobuf imported to relay frontend========")
#print(mod.astext(show_meta_data=False))

# TIDL operator annotation 
#    - mark each operator either supported (True) or unsupported (False) by TIDL
op_annotations = tidlAnnotation.tidl_annotation(mod)

# Check if whole graph can offload to TIDL (no graph partitioning for now)
graph_supported_by_tidl = True
for node in op_annotations:
    print(f'Operator {node.op.name}: {op_annotations[node]}')
    graph_supported_by_tidl = graph_supported_by_tidl and op_annotations[node]

# Import whole graph to TIDL if it can offload to TIDL
if graph_supported_by_tidl:
    print(model_path + ' can be offloaded to TIDL.')
    print('============== importing this model to TIDL ================')
    if tidl.relay_ir_import(mod, params) == False:
        print('Importing this model to TIDL failed!')
    else:
        print('Importing this model to TIDL succeeded!')
else:
    print(model_path + ' can not be offloaded to TIDL.')

