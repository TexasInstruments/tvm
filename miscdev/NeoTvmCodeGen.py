# TVM code generation, creating deploy_* artifacts for full TIDL offload
import tvm
import os
import sys
import numpy as np
import tensorflow as tf
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tidl_import import tidl_check_model
from tvm.contrib import cc

if len(sys.argv[1:]) > 0:
  forced_tidl_offload = True
  print("FORCING ARM offload")
else:
  forced_tidl_offload = False

model = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
#model = "./mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
#model = "./mobileNet3/v3-large_224_1.0_float.pb"
#model = "./inceptionv1/inceptionnet_v1.pb"
#model  = "./inceptionv3/inception_v3_2016_08_28_frozen-with_shapes.pb"

artifacts_folder = "output"

image = './airshow.jpg'

input_node = 'Placeholder'
out_node = 'InceptionV1/Logits/Predictions/Softmax'
model_input_shape = (224,224,3)

#input_node = 'input'
#out_node   = 'InceptionV3/Predictions/Softmax'
#model_input_shape = (299,299,3)

input_node = "input"
out_node   = 'MobilenetV1/Predictions/Reshape_1'
model_input_shape = (224,224,3)

data_shape_input = list(model_input_shape)
data_shape_input.insert(0,1)
data_shape_input = tuple(data_shape_input) # Prepend batch size
print(data_shape_input)

#target = "llvm"
target = "llvm -target=armv7l-linux-gnueabihf"

plsdk_devkit = "/home/x0157990/ti-processor-sdk-linux-am57xx-evm-06.01.00.08/linux-devkit/sysroots/x86_64-arago-linux/usr/bin/"
tidl_calib_tool  = plsdk_devkit + "eve_test_dl_algo_ref.out"
tidl_import_tool = plsdk_devkit + "tidl_model_import.out"
arm_gcc          = plsdk_devkit + "arm-linux-gnueabihf-g++"

if forced_tidl_offload:
  tidl_offload = False
else:
  tidl_offload = tidl_check_model(model, image, 'tidl_subgraph', model_input_shape,
                                tidl_import_tool, tidl_calib_tool, artifacts_folder)


if tidl_offload:
  print("Offload this model to TIDL")
  x = relay.var("x", shape=data_shape_input)
  q = relay.TidlInference(x, num_labels=1001)
  func = relay.Function([x], q)
  net = relay.Module.from_expr(func)
  print(net)
  ######################################################################
  graph, lib, params = relay.build_module.build(net, target=target)

else: 
  from tvm.relay.testing import tf as tf_testing
  print("Run this model on ARM")
  #layout = 'NHWC'
  layout = None
  with tf.compat.v1.gfile.FastGFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
    ######################################################################
    # Import the graph to Relay
    # -------------------------
    # Import tensorflow graph definition to relay frontend.
    #
    # Results:
    #   sym: relay expr for given tensorflow protobuf.
    #   params: params converted from tensorflow params (tensor protobuf).
    shape_dict = {input_node : data_shape_input}
    print(shape_dict)
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
                                                 shape=shape_dict, outputs=None)

    print("Tensorflow protobuf imported to relay frontend.")
    with relay.build_config(opt_level=3):
      graph, lib, params = relay.build(mod, target=target, params=params)


#print(params)
print("...Start writting converted model:")
tmp_folder = os.getcwd() + "/" + artifacts_folder + "/"
path_lib = tmp_folder + "deploy_lib.so"
path_graph = tmp_folder + "deploy_graph.json"
path_params = tmp_folder + "deploy_param.params"
print("...library!")
lib.export_library(path_lib, cc.build_create_shared_func(compile_cmd=arm_gcc))
print("...graph!")
with open(path_graph, "w") as fo:
  fo.write(graph)
print("...params!")
with open(path_params, "wb") as fo:
  fo.write(relay.save_param_dict(params))

