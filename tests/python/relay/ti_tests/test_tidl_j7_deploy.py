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
import numpy as np
import tvm
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.download import download_testdata

######################################################################
# Load a test image
# ---------------------------------------------
def load_image(batch_size, mean, scale, needs_nchw):
    from PIL import Image
    #from matplotlib import pyplot as plt

    img_file = "./airshow.jpg"
    if len(sys.argv) > 1:
        img_file = sys.argv[1]

    orig_img = Image.open(img_file)  # HWC
    resized_img = orig_img.resize((256, 256))
    cropped_img = resized_img.crop((16, 16, 240, 240))
    # plt.imshow(cropped_img)
    # plt.show()

    # Normalize input data to (-1, 1)
    norm_img = np.asarray(cropped_img).astype("float32")
    norm_img[:, :, 0] = (norm_img[:, :, 0] - mean[0]) * scale[0]
    norm_img[:, :, 1] = (norm_img[:, :, 1] - mean[1]) * scale[1]
    norm_img[:, :, 2] = (norm_img[:, :, 2] - mean[2]) * scale[2]
    #norm_img = norm_img / np.amax(np.abs(norm_img))
    # Set batch_size of input data: NHWC
    input_data = np.concatenate([norm_img[np.newaxis, :, :]]*batch_size)
    if needs_nchw:
        input_data = input_data.transpose(0, 3, 1, 2)  # NCHW
    return input_data

######################################################################
# Run the model with relay runtime
# ---------------------------------------------
def run_module(model_name, input_tensor, mean, scale, is_nchw):
    # load deployable module
    artifacts_dir = "artifacts_" + model_name + "/"
    loaded_json = open(artifacts_dir + "deploy_graph.json").read()
    loaded_lib = tvm.runtime.load_module(artifacts_dir + "deploy_lib.so")
    loaded_params = bytearray(open(artifacts_dir + "deploy_param.params", "rb").read())

    # create a runtime executor module
    module = runtime.create(loaded_json, loaded_lib, tvm.cpu())

    # load params into the module
    module.load_params(loaded_params)

    # feed input data
    input_data = load_image(1, mean, scale, is_nchw)
    module.set_input(input_tensor, tvm.nd.array(input_data))

    # run
    module.run()

    # get output
    tvm_output = module.get_output(0).asnumpy()

    print(model_name + " execution finished")
    return tvm_output

def print_top5(output):
    top5 = []
    values = []
    for i in range(5):
        top = np.argmax(output[0, :])
        top5.append(top)
        values.append(output[0, top])
        output[0, top] = 0
    print(top5)
    print(values)

if __name__ == '__main__':

    output1 = run_module("MobileNetV1", "input", [128, 128, 128],
                         [0.0078125, 0.0078125, 0.0078125], False)

    output2 = run_module("ONNX_MobileNetV2", "data", [123.675, 116.28, 103.53],
                         [0.017125, 0.017507, 0.017429], True)

    print("TensorFlow MobileNetV1 output: (index of 1001)")
    print_top5(output1)
    print("ONNX MobileNetV2 output: (index of 1000)")
    print_top5(output2)

