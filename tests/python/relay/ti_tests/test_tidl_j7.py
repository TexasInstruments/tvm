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
"""Unit tests for TIDL compilation."""

import os
import numpy as np
import pytest
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.contrib.tar import untar
import tensorflow as tf
import onnx
from tvm.relay.testing import tf as tf_testing
from tvm.relay.backend.contrib import tidl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', action='store_true',
                    default=True,
                    help='generate code for target device (ARM core) (Default)')
parser.add_argument('--host', action='store_false',
                    dest="target",
                    help='generate code for host emulation (e.g. x86_64 core)')
parser.add_argument('--deny', dest='denylist', action='append',
                    help='force Relay operator to be unsupported by TIDL')
parser.add_argument('--nooffload', action='store_true',
                    help='produce a host-only deployable module without TIDL offload')
args = parser.parse_args()

def disable_outputs():
    """ Redirect stdout/stderr to /dev/null """
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    return [ null_fd, saved_stdout_fd, saved_stderr_fd ]

def restore_outputs(fds):
    """ Restore stdout/stderr to saved fds """
    os.dup2(fds[2], 2)
    os.dup2(fds[1], 1)
    os.close(fds[0])

def get_compiler_path():
    arm_gcc_path = os.getenv("ARM_GCC_PATH")
    if arm_gcc_path is None:
        print("Environment variable ARM_GCC_PATH is not set! Model won't be compiled!")
        return None
    else:
        arm_gcc = os.path.join(arm_gcc_path, "aarch64-none-linux-gnu-g++")
        if os.path.exists(arm_gcc):
            return arm_gcc
        else:
            print("ARM GCC aarch64-none-linux-gnu-g++ does not exist! Model won't be compiled!")
            return None

def get_tidl_tools_path():
    tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    if tidl_tools_path is None:
        print("Environment variable TIDL_TOOLS_PATH is not set! Model won't be compiled!")
        return None
    else:
        return tidl_tools_path

def model_compile(model_name, mod_orig, params, model_input_list, num_tidl_subgraphs=1):
    """ Compile a model in Relay IR graph

    Parameters
    ----------
    model_name : string
        Name of the model
    mod_orig : tvm.relay.Module
        Original Relay IR graph
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    model_input_list : list of dictionary for multiple calibration data
        A dictionary where the key in input name and the value is input tensor
    num_tidl_subgraphs : int
        Number of subgraphs to offload to TIDL
    Returns
    -------
    status: int
        Status of compilation:
            1  - compilation for TIDL offload succeeded
            -1 - compilation for TIDL offload failed - failure for CI testing
            0  - no compilation due to missing TIDL tools or GCC ARM tools
    """

    tidl_platform = "J7"   # or "AM57"
    tidl_version = (7, 0)  # corresponding Processor SDK version
    tidl_artifacts_folder = "./artifacts_" + model_name +  ("_target" if args.target else "_host")
    os.makedirs(tidl_artifacts_folder, exist_ok = True)
    for root, dirs, files in os.walk(tidl_artifacts_folder, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    tidl_compiler = tidl.TIDLCompiler(tidl_platform, tidl_version,
                                      num_tidl_subgraphs=num_tidl_subgraphs,
                                      artifacts_folder=tidl_artifacts_folder,
                                      tidl_tools_path=get_tidl_tools_path(),
                                      tidl_tensor_bits=8,
                                      tidl_calibration_options={'iterations': 10},
                                      tidl_denylist=args.denylist)

    if args.nooffload:
        mod, status = mod_orig, 0
    else:
        mod, status = tidl_compiler.enable(mod_orig, params, model_input_list)

    if args.target:
        arm_gcc = get_compiler_path()
        if arm_gcc is None:
            print("Skip build because ARM_GCC_PATH is not set")
            return 0  # No graph compilation

    if status == 1: # TIDL compilation succeeded
        print("Graph execution with TIDL")
    else: # TIDL compilation failed or no TIDL compilation due to missing tools
        print("Graph execution without TIDL")

    if args.target:
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    else:
        target = "llvm"

    with tidl.build_config(tidl_compiler=tidl_compiler):
        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    tidl.remove_tidl_params(params)

    path_lib = os.path.join(tidl_artifacts_folder, "deploy_lib.so")
    path_graph = os.path.join(tidl_artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(tidl_artifacts_folder, "deploy_param.params")
    if args.target:
        lib.export_library(path_lib, cc=arm_gcc)
    else:
        lib.export_library(path_lib)
    with open(path_graph, "w") as fo:
        fo.write(graph)
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    print("Artifacts can be found at " + tidl_artifacts_folder)
    return status

def gluoncv_compile_model(model_name, img_file, img_size=None, img_norm="ssd", batch_size=4):
    try:
        from gluoncv import model_zoo, data

        #======================== Obtain input data ========================
        if img_norm == "rcnn":
            img_norm, _ = data.transforms.presets.rcnn.load_test(img_file)
        else:
            img_norm, _ = data.transforms.presets.ssd.load_test(img_file, short=img_size)
        input_data = img_norm.asnumpy()
        input_data = np.concatenate([input_data]*batch_size)

        #======================== Load the model ===========================
        input_name = "data"
        model = model_zoo.get_model(model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, {input_name:input_data.shape})

        #======================== Compile the model ========================
        status = model_compile(model_name, mod, params, [{input_name:input_data}])
        assert status != -1, "TIDL compilation failed"   # For CI test

    except ModuleNotFoundError:
        print("gluoncv not installed. Skipping Gluon-CV model compilation.")

def test_tidl_classification():
    classification_models = ['mobilenet1.0', 'mobilenetv2_1.0', 'resnet101_v1', 'densenet121']
    image_size = 224

    #======================== Load testing image ===========================
    img_file = download_testdata('https://github.com/dmlc/mxnet.js/blob/master/' +
                                 'data/cat.png?raw=true', 'cat.png', module='data')

    for model_name in classification_models:
        #======================== Load and compile the model ========================
        gluoncv_compile_model(model_name, img_file, img_size=image_size)

def test_tidl_object_detection():
    object_detection_models = ['ssd_512_mobilenet1.0_voc', 'yolo3_mobilenet1.0_coco']
    image_size = 512

    #======================== Load testing image ===========================
    img_file = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')

    for model_name in object_detection_models:
        #======================== Load and compile the model ========================
        gluoncv_compile_model(model_name, img_file, img_size=image_size)

@pytest.mark.skip('skip because of incompatible gluoncv version')
def test_tidl_segmentation():
    model_name = 'mask_rcnn_resnet18_v1b_coco'

    #======================== Load testing image =======================
    img_file = download_testdata('https://raw.githubusercontent.com/dmlc/web-data/master/' +
                                 'gluoncv/segmentation/voc_examples/1.jpg', 'example.jpg',
                                 module='data')

    #======================== Load and compile the model ========================
    gluoncv_compile_model(model_name, img_file, img_norm="rcnn")

def create_tf_relay_graph(model, input_node, input_shape, layout):

    if model == "MobileNetV1":
        model_tar = download_testdata(
                'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/' +
                'mobilenet_v1_1.0_224.tgz', 'mobilenet_v1_1.0_224.tgz', module='models')
        model_dir = os.path.dirname(model_tar)
        model = os.path.join(model_dir, 'mobilenet_v1_1.0_224_frozen.pb');
        if not os.path.exists(model):
            untar(model_tar, model_dir)
        #model    = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
        #model    = "./mobileNet1/mobilenet_v1_1.0_224_final.pb"
        out_node = 'MobilenetV1/Predictions/Softmax'
        #out_node = 'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D'
        #out_node = 'MobilenetV1/MobilenetV1/Conv2d_0/Relu6'
    elif model == "MobileNetV2":
        model    = "./mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
        #model    = "./mobileNet2/mobilenet_v2_1.0_224_final.pb"
        out_node = 'MobilenetV2/Predictions/Softmax'
    elif model == "InceptionV1":
        model    = "./inception1/inception_v1_fbn.pb"
        out_node = "softmax/Softmax"
    elif model == "InceptionV3":
        model    = "./inception3/inception_v3_2016_08_28_frozen-with_shapes.pb"
        out_node = "InceptionV3/Predictions/Softmax"

    with tf.gfile.GFile(model, 'rb') as f:
        # Import tensorflow graph definition to relay frontend.
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        # Add shapes to the graph.
        with tf.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)

        shape_dict = {input_node : input_shape}
        print("Input node shape dict:" + str(shape_dict))
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                     layout = layout,
                                                     shape  = shape_dict,
                                                     outputs= None)
        print("Tensorflow model imported to Relay IR.")

    return mod, params

def create_tflite_relay_graph(model, input_node, input_shape, layout):
    if model == "MobileNetV1":
        model = "./mobileNet1/mobilenet_v1_1.0_224.tflite"
    elif model == "MobileNetV2":
        model = "./mobileNet2/mobilenet_v2_1.0_224.tflite"
    elif model == "deeplabv3":
        model = download_testdata(
            'https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite?raw=true',
            'deeplabv3_257_mv_gpu.tflite', module='models')

    # get TFLite model from buffer
    tflite_model_buf = open(model, "rb").read()
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    ##############################
    # Load Neural Network in Relay
    # ----------------------------

    # TFLite input tensor name, shape and type
    input_dtype = "float32"

    # parse TFLite model and convert into Relay computation graph
    mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_node: input_shape},
                                         dtype_dict={input_node: input_dtype})
    print("TensorflowLite model imported to Relay IR.")

    return mod, params

def load_image(batch_size, img_file, resize_wh, crop_wh, mean, scale, needs_nchw):
    from PIL import Image
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

def test_tidl_tf_mobilenets(model_name, img_file_list, format="tf"):
    data_layout = "NHWC"
    input_node = "input"
    input_shape = (1, 224, 224, 3)
    input_data_list = [ load_image(1, img_file, [256, 256], [224, 224],
                            [128, 128, 128], [0.0078125, 0.0078125, 0.0078125], False)
                            for img_file in img_file_list ]
    if input_data_list[0].shape != input_shape:
        sys.exit("Input data shape is not correct!")
    print("input_data shape: {}".format(input_data_list[0].shape))

    #============= Create a Relay graph for MobileNet model ==============
    if format == "tflite":
        create_relay_graph_func = create_tflite_relay_graph
    else:
        create_relay_graph_func = create_tf_relay_graph
    fds = disable_outputs()
    tf_mod, tf_params = create_relay_graph_func(model = model_name, input_node = input_node,
                                                input_shape = input_shape, layout = data_layout)
    restore_outputs(fds)

    #======================== TIDL code generation ====================
    input_dict_list = [ {input_node:input_data} for input_data in input_data_list ]
    status = model_compile(model_name, tf_mod, tf_params, input_dict_list)
    assert status != -1, "TIDL compilation failed"   # For CI test

def test_tidl_onnx(model_name, img_file_list):
    data_layout = "NCHW"
    input_node = "data"
    input_shape = (1, 3, 224, 224)
    input_data_list = [ load_image(1, img_file, [256, 256], [224, 224],
                            [123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429], True)
                            for img_file in img_file_list ]
    if input_data_list[0].shape != input_shape:
        sys.exit("Input data shape is not correct!")
    print("input_data shape: {}".format(input_data_list[0].shape))

    #============= Create a Relay graph for MobileNet model ==============
    if model_name == "ONNX_MobileNetV2":
        model = download_testdata(
                'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/' + 
                'model/mobilenetv2-7.onnx', 'mobilenetv2-7.onnx', module='models')
        #model = "./onnx_mobileNet2/mobilenetv2-1.0.onnx"

    onnx_mod, onnx_params = relay.frontend.from_onnx(onnx.load(model),
                                                     shape={input_node:input_shape})

    #======================== TIDL code generation ====================
    input_dict_list = [ {input_node:input_data} for input_data in input_data_list ]
    status = model_compile(model_name, onnx_mod, onnx_params, input_dict_list)
    assert status != -1, "TIDL compilation failed"   # For CI test

def test_tidl_pytorch(model_name, img_file_list):
    import torch
    import torchvision.models as models

    input_node = "data"
    input_shape = (1, 3, 224, 224)
    input_data_list = [ load_image(1, img_file, [256, 256], [224, 224],
                            [123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429], True)
                            for img_file in img_file_list ]
    if input_data_list[0].shape != input_shape:
        sys.exit("Input data shape is not correct!")
    print("input_data shape: {}".format(input_data_list[0].shape))

    #============= Create a Relay graph from Pytorch model ===============
    # from https://pytorch.org/docs/stable/torchvision/models.html
    model = None
    if model_name == "pytorch_mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
    #other models:
    # resnet50 = models.resnet50(pretrained=True)
    # fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=True)
    # deeplab_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)
    # maskrcnn_resnet50 = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    assert model, "Unknown pytorch model"
    script_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, script_data).eval()
    shape_list = [(input_node, input_shape)]
    pytorch_mod, pytorch_params = relay.frontend.from_pytorch(scripted_model, shape_list)

    #======================== TIDL code generation ====================
    input_dict_list = [ {input_node:input_data} for input_data in input_data_list ]
    status = model_compile(model_name, pytorch_mod, pytorch_params, input_dict_list, 
                           num_tidl_subgraphs=2)
    assert status != -1, "TIDL compilation failed"   # For CI test

def test_tidl_tflite_deeplabv3(img_file_list):
    model_name = "deeplabv3"
    data_layout = "NHWC"
    input_node = "sub_7"
    input_shape = (1, 257, 257, 3)
    input_data_list = [ load_image(1, img_file, [257, 257], [257, 257],
                            [128, 128, 128], [0.0078125, 0.0078125, 0.0078125], False)
                            for img_file in img_file_list ]
    if input_data_list[0].shape != input_shape:
        sys.exit("Input data shape is not correct!")
    print("input_data shape: {}".format(input_data_list[0].shape))

    #============= Create a Relay graph for model ==============
    tf_mod, tf_params = create_tflite_relay_graph(model = model_name, input_node = input_node,
                                                  input_shape = input_shape, layout = data_layout)

    #======================== TIDL code generation ====================
    input_dict_list = [ {input_node:input_data} for input_data in input_data_list ]
    status = model_compile(model_name, tf_mod, tf_params, input_dict_list,
                           num_tidl_subgraphs=2)
    assert status != -1, "TIDL compilation failed"   # For CI test

if __name__ == '__main__':
    img_cat = download_testdata(
         'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true',
         'cat.png', module='data')
    img_cat2 = download_testdata(
         'https://git.ti.com/cgit/tidl/tidl-api/plain/examples/test/testvecs/input/objects/cat-pet-animal-domestic-104827.jpeg',
         'cat2.jpeg', module='data')
    img_airshow = download_testdata(
         'https://git.ti.com/cgit/tidl/tidl-utils/plain/test/testvecs/input/airshow.jpg',
         'airshow.jpg', module='data')
    test_tidl_tf_mobilenets("MobileNetV1", [ img_cat, img_cat2, img_airshow ])
    #test_tidl_tf_mobilenets("MobileNetV1", [ img_cat ], "tflite")
    #test_tidl_tf_mobilenets("MobileNetV2", [ img_cat ])
    #test_tidl_tf_mobilenets("MobileNetV2", [ img_cat ], "tflite")
    test_tidl_onnx("ONNX_MobileNetV2", [ img_airshow, img_cat2, img_cat ])

    img_street = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')
    #img_kidbike = "./deeplab_kidbike_input.png"
    test_tidl_tflite_deeplabv3([ img_street ])
    #test_tidl_classification()
    #test_tidl_object_detection()
    #test_tidl_segmentation()

    test_tidl_pytorch("pytorch_mobilenetv2", [ img_airshow, img_cat, img_cat2 ])

