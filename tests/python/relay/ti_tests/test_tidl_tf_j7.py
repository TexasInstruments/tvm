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
import tensorflow as tf
from tvm.relay.testing import tf as tf_testing
from tvm.relay.backend.contrib import tidl

def get_compiler_path():
    arm_gcc_path = os.getenv("ARM_GCC_PATH")
    if arm_gcc_path is None:
        print("Environment variable ARM_GCC_PATH is not set! Model won't be compiled!")
        return None
    else:
        arm_gcc = os.path.join(arm_gcc_path, "arm-linux-gnueabihf-g++")
        if os.path.exists(arm_gcc):
            return arm_gcc
        else:
            print("ARM GCC arm-linux-gnueabihf-g++ does not exist! Model won't be compiled!")
            return None

def get_tidl_tools_path():
    tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    if tidl_tools_path is None:
        print("Environment variable TIDL_TOOLS_PATH is not set! Model won't be compiled!")
        return None
    else:
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

    tidl_target = "tidl"
    tidl_platform = "J7"   # or "AM57"
    tidl_version = (7, 0)  # corresponding Processor SDK version
    tidl_artifacts_folder = "./artifacts_" + model_name
    if os.path.isdir(tidl_artifacts_folder):
        filelist = [f for f in os.listdir(tidl_artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(tidl_artifacts_folder, file))
    else:
        os.mkdir(tidl_artifacts_folder)

    tidl_compiler = tidl.TIDLCompiler(tidl_platform, tidl_version,
                                      num_tidl_subgraphs=num_tidl_subgraphs,
                                      artifacts_folder=tidl_artifacts_folder,
                                      tidl_tools_path=get_tidl_tools_path(),
                                      tidl_tensor_bits=16)
    mod, status = tidl_compiler.enable(mod_orig, params, model_input)

    #arm_gcc = get_compiler_path()
    #if arm_gcc is None:
    #    print("Skip build because ARM_GCC_PATH is not set")
    #    return 0  # No graph compilation

    if status == 1: # TIDL compilation succeeded
        print("Graph execution with TIDL")
    else: # TIDL compilation failed or no TIDL compilation due to missing tools
        print("Graph execution without TIDL")

    #target = "llvm -target=armv7l-linux-gnueabihf" # for AM57x or J6 devices
    target = "llvm"                                 # for host

    with tidl.build_config(artifacts_folder="tempDir", platform="j7"):
        graph, lib, params = relay.build_module.build(mod, target=target, params=params)

    path_lib = os.path.join(tidl_artifacts_folder, "deploy_lib.so")
    path_graph = os.path.join(tidl_artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(tidl_artifacts_folder, "deploy_param.params")
    #lib.export_library(path_lib, cc=arm_gcc)   # for AM57x/J6 devices
    lib.export_library(path_lib)                # for host
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
        status = model_compile(model_name, mod, params, {input_name:input_data})
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
        model    = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
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
        model    = "./mobileNet1/mobilenet_v1_1.0_224.tflite"
    elif model == "MobileNetV2":
        model    = "./mobileNet2/mobilenet_v2_1.0_224.tflite"

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

def test_tidl_tf_mobilenets(model_name, format="tf"):
    tidl_tools_path = get_tidl_tools_path()
    #dtype = "float32"
    data_layout = "NHWC"
    input_shape = (1, 224, 224, 3)
    x = np.load(os.path.join(tidl_tools_path, 'dog.npy'))  # "NCHW"
    x = x.transpose(0,2,3,1)  # TF uses "NHWC" layout
    if x.shape != input_shape:
        sys.exit("Input data shape is not correct!")
    # Normalize input data to (-1,1)
    input_data = x/np.amax(np.abs(x))
    input_node = "input"

    #============= Create a Relay graph for MobileNet model ==============
    if format == "tflite":
        create_relay_graph_func = create_tflite_relay_graph
    else:
        create_relay_graph_func = create_tf_relay_graph
    tf_mod, tf_params = create_relay_graph_func(model = model_name,
                                              input_node  = input_node,
                                              input_shape = input_shape,
                                              layout = data_layout)
    print("---------- Original TF Graph ----------")
    print(tf_mod.astext(show_meta_data=False))

    #======================== TIDL code generation ====================
    status = model_compile(model_name, tf_mod, tf_params,
                           {input_node:input_data})
    assert status != -1, "TIDL compilation failed"   # For CI test

if __name__ == '__main__':
    test_tidl_tf_mobilenets("MobileNetV1")
    #test_tidl_tf_mobilenets("MobileNetV1", "tflite")
    #test_tidl_tf_mobilenets("MobileNetV2")
    #test_tidl_tf_mobilenets("MobileNetV2", "tflite")
    #test_tidl_classification()
    #test_tidl_object_detection()
    #test_tidl_segmentation()
