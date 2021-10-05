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

parser = argparse.ArgumentParser(epilog='e.g. python3 ./test_tidl_j7_deploy.py input_img.jpg')
parser.add_argument('--dlr', action='store_true',
                    default=True,
                    help='run inference with DLR runtime (Default)')
parser.add_argument('--tvm', action='store_false',
                    dest='dlr',
                    help='run inference with TVM runtime')
parser.add_argument('--cv', action='store_true',
                    default=True,
                    help='pre-process image with OpenCV (Default)')
parser.add_argument('--pil', action='store_false',
                    dest='cv',
                    help='pre-process image with Pillow/PIL (Python Image Library)')
parser.add_argument('--target', action='store_true',
                    default=True,
                    help='run inference on target device (ARM core) (Default)')
parser.add_argument('--host', action='store_false',
                    dest='target',
                    help='run inference on host with host emulation (e.g. x86_64)')
parser.add_argument('input', nargs='?')
args = parser.parse_args()

if args.dlr:
    from dlr import DLRModel
else:
    import tvm
    from tvm.contrib import graph_runtime as runtime

######################################################################
# Load a test image
# ---------------------------------------------
def load_image_pillow(batch_size, img_file, mean, scale, needs_nchw, resize_wh, crop_wh,
                      needs_quant):
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

    if needs_quant:
        norm_img = np.asarray(cropped_img).astype("uint8")
    else:
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

def load_image_cv(batch_size, img_file, mean, scale, needs_nchw, resize_wh, crop_wh, needs_quant):
    import cv2

    img = cv2.imread(img_file)
    resized_img = cv2.resize(img, (resize_wh[0], resize_wh[1]), interpolation=cv2.INTER_CUBIC)
    assert(resize_wh[0] >= crop_wh[0] and resize_wh[1] >= crop_wh[1]), \
           "resize size needs to be bigger than crop size"
    if resize_wh[0] > crop_wh[0] or resize_wh[1] > crop_wh[1]:
        wh_start = [ int((x - y) / 2) for x, y in zip(resize_wh, crop_wh)]
        wh_end   = [ int(x + y) for x, y in zip(wh_start, crop_wh)]
        img = resized_img[wh_start[1]:wh_end[1], wh_start[0]:wh_end[0]]
    else:
        img = resized_img

    # OpenCV loads as BGR, convert to RGB by swapping channels
    img = img[:,:,::-1]

    # convert HWC to NCHW
    img = np.expand_dims(np.transpose(img, (2,0,1)),axis=0)

    if needs_quant:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32)
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
    quant = model_name.endswith('_quant')

    if args.cv:
        input_data = load_image_cv(1, img_file, mean, scale, is_nchw, resize_wh, crop_wh, quant)
    else:
        input_data = load_image_pillow(1, img_file, mean, scale, is_nchw, resize_wh, crop_wh, quant)

    if args.dlr:
        module = DLRModel(artifacts_dir)
        results = module.run({input_tensor : input_data})
        tvm_outputs = results

        # get optional tidl info, run with TIDL_RT_PERFSTATS=1
        #perf_data = module.get_TI_benchmark_data()
        #print(perf_data)

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
        tvm_outputs = []
        for i in range(module.get_num_outputs()):
            tvm_outputs.append(module.get_output(i).asnumpy())

        # get optional tidl info, run with TIDL_RT_PERFSTATS=1
        #import ctypes
        #for c in range(16):
        #    try:
        #        func = loaded_lib.get_function(f'tidl_get_custom_data_{c}')
        #    except AttributeError as e:
        #        break
        #    vec_void = func()
        #    # vec_void is pointer to C++ std::vector<uint64_t>, hack into memory layout
        #    data_ptr = ctypes.cast(vec_void, ctypes.POINTER(ctypes.POINTER(ctypes.c_ulonglong)))
        #    data = data_ptr[0]
        #    print(f"tidl_{c}: cp_in {data[1]-data[0]} process {data[3]-data[2]} cp_out {data[5]-data[4]}")

    print(model_name + " execution finished")
    return tvm_outputs

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

    #img_file = "~/.tvm_test_data/data/airshow.jpg"
    assert(args.input is not None), "Please specify an input image"
    assert(os.path.exists(args.input)), f"Input file, {args.input}, does not exists"
    img_file = args.input
    print(f"Input image file: {img_file}")

    outputs1 = run_module("mobilenetv3_large", "data", [128, 128, 128],
                          [0.0078125, 0.0078125, 0.0078125], True,
                          img_file, [256,256], [224,224])

    outputs2 = run_module("mobilenetv3_large_c7x", "data", [128, 128, 128],
                          [0.0078125, 0.0078125, 0.0078125], True,
                          img_file, [256,256], [224,224])


    print("MxNet MobileNetV3 output: (index of 1000)")
    print_top5(outputs1[0])

    print("(With C7x codegen) MxNet MobileNetV3 output: (index of 1000)")
    print_top5(outputs2[0])

