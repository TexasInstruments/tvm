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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dlr', action='store_true')
parser.add_argument('--cv', action='store_true')
parser.add_argument('--target', action='store_true')
parser.add_argument('input', nargs='?')
args = parser.parse_args()

from tvm.contrib.download import download_testdata
if args.dlr:
    from dlr import DLRModel
else:
    import tvm
    from tvm.contrib import graph_runtime as runtime

######################################################################
# Load a test image
# ---------------------------------------------
def load_image_pillow(batch_size, img_file, mean, scale, needs_nchw, resize_wh, crop_wh):
    from PIL import Image
    #from matplotlib import pyplot as plt

    orig_img = Image.open(img_file)  # HWC
    resized_img = orig_img.resize((resize_wh[0], resize_wh[1]))
    assert(resize_wh[0] >= crop_wh[0] and resize_wh[1] >= crop_wh[1]), \
           "resize size needs to be bigger than crop size"
    if resize_wh[0] > crop_wh[0] or resize_wh[1] > crop_wh[1]:
        wh_start = [ (x - y) / 2 for x, y in zip(resize_wh, crop_wh)]
        wh_end   = [ x + y for x, y in zip(wh_start, crop_wh)]
        cropped_img = resized_img.crop((wh_start[0], wh_start[1], wh_end[0], wh_end[1]))
    else:
        cropped_img = resized_img
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

def load_image_cv(batch_size, img_file, mean, scale, needs_nchw, resize_wh, crop_wh):
    import cv2

    img = cv2.imread(img_file)

    # Resize to 299xH or Wx299
    orig_height, orig_width, _ = img.shape
    new_height = orig_height * 299 // min(img.shape[:2])
    new_width = orig_width * 299 // min(img.shape[:2])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Center crop to 299x299
    height, width, _ = img.shape
    startx = width//2 - (224//2)
    starty = height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]

    # OpenCV loads as BGR, convert to RGB by swapping channels
    img = img[:,:,::-1]

    # convert HWC to NCHW
    img = np.expand_dims(np.transpose(img, (2,0,1)),axis=0).astype(np.float32)

    for mean, scale, ch in zip(mean, scale, range(img.shape[1])):
        img[:,ch,:,:] = ((img[:,ch,:,:] - mean) * scale)

    if not needs_nchw:
        img = img.transpose(0, 2, 3, 1)

    return img

######################################################################
# Run the model with relay runtime
# ---------------------------------------------
def run_module(model_name, input_tensor, mean, scale, is_nchw, img_file, resize_wh, crop_wh):
    # load deployable module
    artifacts_dir = "artifacts_" + model_name + ("_target" if args.target else "_host") + "/"

    if args.cv:
        input_data = load_image_cv(1, img_file, mean, scale, is_nchw, resize_wh, crop_wh)
    else:
        input_data = load_image_pillow(1, img_file, mean, scale, is_nchw, resize_wh, crop_wh)

    if args.dlr:
        module = DLRModel(artifacts_dir)
        results = module.run({input_tensor : input_data})
        tvm_output = results[0]
    else:
        loaded_json = open(artifacts_dir + "deploy_graph.json").read()
        loaded_lib = tvm.runtime.load_module(artifacts_dir + "deploy_lib.so")
        loaded_params = bytearray(open(artifacts_dir + "deploy_param.params", "rb").read())

        # create a runtime executor module
        module = runtime.create(loaded_json, loaded_lib, tvm.cpu())

        # load params into the module
        module.load_params(loaded_params)

        # feed input data
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

    img_file = download_testdata(
         'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true',
         'cat.png', module='data')
    #img_file = "./airshow.jpg"
    if args.input is not None:
        img_file = args.input

    output1 = run_module("MobileNetV1", "input", [128, 128, 128],
                         [0.0078125, 0.0078125, 0.0078125], False,
                         img_file, [256,256], [224,224])

    output2 = run_module("ONNX_MobileNetV2", "data", [123.675, 116.28, 103.53],
                         [0.017125, 0.017507, 0.017429], True,
                         img_file, [256,256], [224,224])

    if args.input is None:
        img_file = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                     'gluoncv/detection/street_small.jpg?raw=true',
                                     'street_small.jpg', module='data')
    output3 = run_module("deeplabv3", "sub_7", [128, 128, 128],
                         [0.0078125, 0.0078125, 0.0078125], False,
                         img_file, [257,257], [257,257])

    print("TensorFlow MobileNetV1 output: (index of 1001)")
    print_top5(output1)
    print("ONNX MobileNetV2 output: (index of 1000)")
    print_top5(output2)
    print("deeplabv3 output shape: {}".format(output3.shape))
    output3 = np.squeeze(output3, axis=0)
    np.savetxt("deeplabv3.results", np.argmax(output3, axis=2).astype(int), "%2d", "")
