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

######################################################################
# Overview for Supported Hardware Backend of TVM
# ----------------------------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/tvm_support_list.png
#      :align: center
#      :scale: 100%
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import Relay and TVM.
import onnx
import numpy as np
import inspect 

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime

import topi
from topi.util import get_const_tuple

from tvm.relay.expr_functor import ExprVisitor

from tvm.relay.op.annotation import tidlAnnotation

######################################################################
# Define Neural Network in Relay
# ------------------------------
image_shape = (1, 3, 224, 224)
onnx_model = onnx.load("../../tutorials/resnet18v2.onnx")
input_name = 'data'

shape_dict = {input_name: image_shape }
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# set show_meta_data=True if you want to show meta data
print("-------------------------------")
print(mod.astext(show_meta_data=False))

op_white_list = tidlAnnotation.tidl_annotation(mod)

graph_supported_by_tidl = True
for node in op_white_list:
    print(f'Operator {node.op.name}: {op_white_list[node]}')
    graph_supported_by_tidl = graph_supported_by_tidl and op_white_list[node]

if graph_supported_by_tidl:
    print("resnet18v2.onnx can be offloaded to TIDL.")
else:
    print("resnet18v2.onnx can not be offloaded to TIDL.")


from tvm.contrib.download import download_testdata
import coremltools as cm
import numpy as np
from PIL import Image

######################################################################
# Load pretrained CoreML model
# ----------------------------
# We will download and load a pretrained mobilenet classification network
# provided by apple in this example
model_url = 'https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel'
model_file = 'mobilenet.mlmodel'
model_path = download_testdata(model_url, model_file, module='coreml')
# Now you have mobilenet.mlmodel on disk
mlmodel = cm.models.MLModel(model_path)

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 224))
# Mobilenet.mlmodel's input is BGR format
img_bgr = np.array(img)[:,:,::-1]
x = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]

######################################################################
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
target = 'llvm'
shape_dict = {'image': x.shape}

# Parse CoreML model and convert into Relay computation graph
mod, params = relay.frontend.from_coreml(mlmodel, shape_dict)
print("============= CoreML model in Relay format ==============")
print(mod.astext(show_meta_data=False))

op_white_list = tidlAnnotation.tidl_annotation(mod)

graph_supported_by_tidl = True
for node in op_white_list:
    print(f'Operator {node.op.name}: {op_white_list[node]}')
    graph_supported_by_tidl = graph_supported_by_tidl and op_white_list[node]

if graph_supported_by_tidl:
    print(model_file + " can be offloaded to TIDL.")
else:
    print(model_file + " can not be offloaded to TIDL.")

