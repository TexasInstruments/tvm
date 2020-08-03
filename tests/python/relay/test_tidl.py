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
import sys
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

tidl_platform = "AM57"  # use command line argument, J7, to change

def get_arm_compiler():
    """ Get ARM compiler if it is available """
    arm_gcc_path = os.getenv("ARM_GCC_PATH")
    if arm_gcc_path is None:
        print("Environment variable ARM_GCC_PATH is not set! Model won't be compiled!")
        arm_compiler = None
    else:
        arm_gcc = os.path.join(arm_gcc_path, "arm-linux-gnueabihf-g++")
        if os.path.exists(arm_gcc):
            arm_compiler = arm_gcc
        else:
            print("ARM GCC arm-linux-gnueabihf-g++ does not exist! Model won't be compiled!")
            arm_compiler = None
    return arm_compiler

def get_tidl_tools_path():
    """ Get TIDL tools if they are available """
    tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    if tidl_tools_path is None:
        print("Environment variable TIDL_TOOLS_PATH is not set! Model won't be compiled!")

    return tidl_tools_path

def model_compile(model_name, mod_orig, params, model_input, num_tidl_subgraphs=1):
    """ Compile a model in Relay IR graph

    Parameters
    ----------
    model_name : string
        Name of the model
    mod_orig : tvm.relay.Module
        Original Relay IR graph
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    model_input : dictionary
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
    from tvm.relay.backend.contrib import tidl

    tidl_artifacts_folder = "./artifacts/" + model_name
    if os.path.isdir(tidl_artifacts_folder):
        filelist = [f for f in os.listdir(tidl_artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(tidl_artifacts_folder, file))
    else:
        os.makedirs(tidl_artifacts_folder)

    if tidl_platform == "AM57":
        tidl_compiler = tidl.TIDLCompiler("AM57", (6, 3),
                                      num_tidl_subgraphs=num_tidl_subgraphs,
                                      artifacts_folder=tidl_artifacts_folder,
                                      tidl_tools_path=get_tidl_tools_path())
    else:
        tidl_compiler = tidl.TIDLCompiler("J7", (7, 0),
                                      num_tidl_subgraphs=num_tidl_subgraphs,
                                      artifacts_folder=tidl_artifacts_folder,
                                      tidl_tools_path=get_tidl_tools_path(),
                                      tidl_tensor_bits=16)
    mod, status = tidl_compiler.enable(mod_orig, params, model_input)

    if tidl_platform == "AM57":
        arm_gcc = get_arm_compiler()
        if arm_gcc is None:
            print("Skip build because ARM_GCC_PATH is not set")
            return 0  # No graph compilation

    if status == 1: # TIDL compilation succeeded
        print("Graph execution with TIDL")
    else: # TIDL compilation failed or no TIDL compilation due to missing tools
        print("Graph execution without TIDL")

    if tidl_platform == "AM57":
        target = "llvm -mtriple=armv7l-linux-gnueabihf" # for AM57x/J6 devices
    else:
        target = "llvm"                                 # testing on J7 host 
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    path_lib = os.path.join(tidl_artifacts_folder, "deploy_lib.so")
    path_graph = os.path.join(tidl_artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(tidl_artifacts_folder, "deploy_param.params")
    if tidl_platform == "AM57":
        lib.export_library(path_lib, cc=arm_gcc)
    else:
        lib.export_library(path_lib)
    with open(path_graph, "w") as fo:
        fo.write(graph)
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    print("Artifacts can be found at " + tidl_artifacts_folder)
    return status

def get_input_nchw(input_shape):
    """ Get input data in 'NCHW' layout """
    batch_size = input_shape[0]
    img_file = download_testdata(
         'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true',
         'cat.png', module='data')
    from PIL import Image
    orig_image = Image.open(img_file)  # HWC
    resized_image = orig_image.resize((input_shape[2], input_shape[3]))
    # Normalize input data to (-1,1)
    # preprocess image as described here:
    # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
    image_norm = np.asarray(resized_image).astype("float32")
    image_norm[:, :, 0] = 2.0 / 255.0 * image_norm[:, :, 0] - 1
    image_norm[:, :, 1] = 2.0 / 255.0 * image_norm[:, :, 1] - 1
    image_norm[:, :, 2] = 2.0 / 255.0 * image_norm[:, :, 2] - 1

    image_norm = image_norm.transpose(2, 0, 1)  # CHW
    # Set batch_size of input data
    input_data = np.concatenate([image_norm[np.newaxis, :, :]]*batch_size)
    return input_data

@pytest.mark.skip('skip pytest because models must be pre-downloaded')
def test_tidl_onnx(batch_size=4):
    """ Test TIDL compilation for ONNX models """
    import onnx     # import ONNX here instead of top to avoid CI error

    model_metadata = {
        'resnet18v1': (224, 224, 'data'),
        'resnet18v2': (224, 224, 'data'),
        'resnet50': (224, 224, 'data'),
        'squeezenet1.1': (224, 224, 'data'),
        'resnet101v1': (224, 224, 'data'),
    }
    models = {
        'resnet18v1': 'resnet18v1',
        'resnet18v2': 'resnet18v2',
        'resnet50': 'resnet50-v1-7',
        'squeezenet1.1': 'squeezenet1.1',
        'resnet101v1': 'resnet101v1',
    }

    for key, value in model_metadata.items():
        model_file = 'onnx_models/' + models[key] + '.onnx'
        print('testing model:', key, ' loading from:', model_file)
        height, weight, input_node = value
        input_shape = (batch_size, 3, height, weight)
        # Get input data
        input_data = get_input_nchw(input_shape)
        # Load the model to Relay
        onnx_model = onnx.load(model_file)
        onnx_mod, onnx_params = relay.frontend.from_onnx(onnx_model, {input_node:input_shape})
        # Compile the model for TIDL
        model_name = 'onnx_' + key
        model_compile(model_name, onnx_mod, onnx_params, {input_node:input_data})

def get_tf_input(input_shape):
    """ Get input data for Tensorflow models """
    input_shape_nchw = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])
    input_data_nchw = get_input_nchw(input_shape_nchw)  # data in "NCHW" layout
    input_data = input_data_nchw.transpose(0, 2, 3, 1)  # TF uses "NHWC" layout
    return input_data

def create_relay_graph_from_tf(model, input_name, input_shape, output_name):
    """ Create Relay graph from Tensorflow model """
    import tensorflow as tf     # import TF here instead of top to avoid CI error
    from tvm.relay.testing import tf as tf_testing

    with tf.gfile.GFile(model, 'rb') as f:
        # Import tensorflow graph definition to relay frontend.
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, output_name)

    # Load the model to Relay
    mod, params = relay.frontend.from_tensorflow(graph_def, layout="NHWC",
                                                 shape={input_name:input_shape},
                                                 outputs=None)
    # Clear the default graph stack and reset the global default graph
    tf.compat.v1.reset_default_graph()

    return mod, params

@pytest.mark.skip('skip pytest because models must be pre-downloaded')
def test_tidl_tensorflow(batch_size=4):
    """ Test TIDL compilation for Tensorflow models """
    model_metadata = {
        'mobilenet100_v1': (224, 224, 'input', 'MobilenetV1/Predictions/Softmax'),
        'mobilenet100_v2': (224, 224, 'input', 'MobilenetV2/Predictions/Softmax'),
        'inception_v1': (224, 224, 'input', 'softmax/Softmax'),
        'inception_v3': (299, 299, 'input', 'InceptionV3/Predictions/Softmax'),
        'mobilenet130_v2': (224, 224, 'Placeholder', 'mobilenet130v2/probs'),
    }
    models = {
        'mobilenet100_v1': 'mobilenet_v1_1.0_224',
        'mobilenet100_v2': 'mobilenet_v2_1.0_224',
        'inception_v1': 'inception_v1',
        'inception_v3': 'inception_v3',
        'mobilenet130_v2': 'mobilenet_v2_130',
    }

    for key, value in model_metadata.items():
        model_file = 'tf_slim_models/' + models[key] + '_frozen.pb'
        print('testing model:', key, ' loading from:', model_file)
        height, weight, input_name, output_name = value
        input_shape = (batch_size, height, weight, 3)
        # Get input data
        input_data = get_tf_input(input_shape)
        # Load the model to Relay
        mod, params = create_relay_graph_from_tf(model_file, input_name, input_shape, output_name)
        # Compile the model for TIDL
        model_name = 'tf_slim_' + key
        model_compile(model_name, mod, params, {input_name:input_data})

@pytest.mark.skip('skip pytest because models must be pre-downloaded')
def test_tidl_pytorch(batch_size=4):
    """ Test TIDL compilation for Pytorch models """
    import torch    # import Pytorch here instead of top to avoid CI error

    model_metadata = {
        'inception_v3': (299, 299, 'input'),
        #'resnet152': (224, 224, 'input'),
    }
    models = {
        'inception_v3': 'inception_v3',
        'resnet152': 'resnet152',
    }

    for key, value in model_metadata.items():
        model_file = 'pytorch_models/' + models[key] + '.pth'
        print('testing model:', key, ' loading from:', model_file)
        height, weight, input_name = value
        input_shape = (batch_size, 3, height, weight)
        # Load the model to Relay
        torch_model = torch.jit.load(model_file, map_location='cpu').float().eval()
        mod, params = relay.frontend.from_pytorch(torch_model, [(input_name, input_shape)])
        # Get input data
        input_data = get_input_nchw(input_shape)
        # Compile the model for TIDL
        model_name = 'pytorch_' + key
        model_compile(model_name, mod, params, {input_name:input_data})

@pytest.mark.skip('skip pytest because models must be pre-downloaded')
def test_tidl_tflite(batch_size=4):
    """ Test TIDL compilation for Tensorflow Lite models """
    import tflite.Model # import TFLite here instead of top to avoid CI error

    model_metadata = {
        'mobilenet100_v1': (224, 224, 'input'),
        'mobilenet100_v2': (224, 224, 'input'),
        #'densenet': (224, 224, 'input'),
        'mnasnet': (224, 224, 'input'),
        'resnet_v2_101': (299, 299, 'input'),
    }
    models = {
        'mobilenet100_v1': 'mobilenet_v1_1.0_224',
        'mobilenet100_v2': 'mobilenet_v2_1.0_224',
        'densenet': 'densenet_2018_04_27',
        'mnasnet': 'mnasnet_1.0_224',
        'resnet_v2_101': 'resnet_v2_101_299',
    }
    for key, value in model_metadata.items():
        model_file = 'tflite_models/' + models[key] + '.tflite'
        print('testing model:', key, ' loading from:', model_file)
        height, weight, input_name = value
        input_shape = (batch_size, height, weight, 3)
        # Load the model to Relay
        with open(model_file, "rb") as f:
            tflite_model_buf = f.read()
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        mod, params = relay.frontend.from_tflite(tflite_model,
                                                 shape_dict={input_name: input_shape},
                                                 dtype_dict={input_name: "float32"})
        # Get input data
        input_data = get_tf_input(input_shape)
        # Compile the model for TIDL
        model_name = 'tflite_' + key
        model_compile(model_name, mod, params, {input_name:input_data})

def gluoncv_compile_model(model_name, img_file, batch_size, img_size=None, img_norm="ssd"):
    """ Compile Gluon-CV models """
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
        print('testing model: ', model_name, ' loading from gluoncv model_zoo')
        input_name = "data"
        model = model_zoo.get_model(model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, {input_name:input_data.shape})

        #======================== Compile the model ========================
        status = model_compile("gluon_"+model_name, mod, params, {input_name:input_data})
        assert status != -1, "TIDL compilation failed"   # For CI test

    except ModuleNotFoundError:
        print("gluoncv not installed. Skipping Gluon-CV model compilation.")

def test_tidl_gluon_classification(batch_size=4):
    """ Test TIDL compilation for Gluon-CV image classification models """
    classification_models = ['mobilenet1.0', 'mobilenetv2_1.0', 'resnet101_v1', 'densenet121']
    image_size = 224

    #======================== Load testing image ===========================
    img_file = download_testdata('https://github.com/dmlc/mxnet.js/blob/master/' +
                                 'data/cat.png?raw=true', 'cat.png', module='data')

    for model_name in classification_models:
        #======================== Load and compile the model ========================
        gluoncv_compile_model(model_name, img_file, batch_size, img_size=image_size)

def test_tidl_gluon_object_detection(batch_size=4):
    """ Test TIDL compilation for Gluon-CV object detection models """
    object_detection_models = ['ssd_512_mobilenet1.0_voc', 'yolo3_mobilenet1.0_coco']
    image_size = 512

    #======================== Load testing image ===========================
    img_file = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')

    for model_name in object_detection_models:
        #======================== Load and compile the model ========================
        gluoncv_compile_model(model_name, img_file, batch_size, img_size=image_size)

@pytest.mark.skip('skip because of incompatible gluoncv version')
def test_tidl_gluon_segmentation(batch_size=1):
    """ Test TIDL compilation for Gluon-CV segmentation models """
    model_name = 'mask_rcnn_resnet18_v1b_coco'

    #======================== Load testing image =======================
    img_file = download_testdata('https://raw.githubusercontent.com/dmlc/web-data/master/' +
                                 'gluoncv/segmentation/voc_examples/1.jpg', 'example.jpg',
                                 module='data')

    #======================== Load and compile the model ========================
    gluoncv_compile_model(model_name, img_file, batch_size, img_norm="rcnn")

def test_tidl_ops():
    #============= Constructing a simple graph to test individual ops ==============
    dtype = 'float32'
    data_layout = 'NCHW'
    input_shape = (1, 3, 224, 224) # NCHW
    input_layout= 'NCHW'
    input = relay.var('data', shape=(input_shape), dtype=dtype)
    output1 = relay.nn.relu(input)
    output2 = relay.clip(input, 0.0, 6.0)
    output = relay.concatenate((output1, output2), axis=1)
    func = relay.Function([input], output)
    mod = tvm.IRModule.from_expr(func)
    #print('---------- Original graph ----------')
    #print(mod.astext(show_meta_data=False))

    input_data = np.random.rand(1, 3, 224, 224) * 12.0 # "NCHW"
    weights = tvm.nd.array(np.random.rand(32, 3, 224, 224))  # "OIHW"
    params = {'weight1':weights}
    model_compile("relu6", mod, params, {"data":input_data}, num_tidl_subgraphs=1)

if __name__ == '__main__':
    if (len(sys.argv) > 1 and sys.argv[1] == "J7"):
        tidl_platform = "J7"
    print("test_tidl.py on platform " + tidl_platform)

    # Need to load tensorflow before tidl to avoid protobuf version conflict
    test_tidl_tensorflow()
    test_tidl_ops()
    test_tidl_onnx()
    test_tidl_pytorch()
    test_tidl_tflite()
    test_tidl_gluon_classification()
    test_tidl_gluon_object_detection()
    test_tidl_gluon_segmentation()
