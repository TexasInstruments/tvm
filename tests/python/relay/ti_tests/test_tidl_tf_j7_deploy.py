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
"""Unit tests for graph partitioning."""
import os
import sys
import tvm
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.download import download_testdata

######################################################################
# Load a test image
# ---------------------------------------------
from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np

#img_file = download_testdata(
#         'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true',
#         'cat.png', module='data')
img_file = "airshow.jpg"
if len(sys.argv) > 1:
  img_file = sys.argv[1]
resized_image = Image.open(img_file).resize((224, 224))
#plt.imshow(resized_image)
#plt.show()
image_data = np.asarray(resized_image).astype("float32")

# after expand_dims, we have format NHWC
image_data = np.expand_dims(image_data, axis=0)

# preprocess image as described here:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
print('input', image_data.shape)

# TF/TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"


######################################################################
# Run the model with relay runtime
# ---------------------------------------------
def run_module():
    # load deployable module
    loaded_json = open("artifacts_MobileNetV1/deploy_graph.json").read()
    loaded_lib = tvm.runtime.load_module("artifacts_MobileNetV1/deploy_lib.so")
    loaded_params = bytearray(open("artifacts_MobileNetV1/deploy_param.params",
                                   "rb").read())

    # create a runtime executor module
    module = runtime.create(loaded_json, loaded_lib, tvm.cpu())

    # load params into the module
    module.load_params(loaded_params)

    # feed input data
    module.set_input(input_tensor, tvm.nd.array(image_data))

    # run
    module.run()

    # get output
    tvm_output = module.get_output(0).asnumpy()

    return tvm_output


if __name__ == '__main__':

    output = run_module()

    print("Execution finished")
    print(np.argmax(output[0,:]), end=' ')
    print(" ")

