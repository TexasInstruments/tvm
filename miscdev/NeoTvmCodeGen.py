# TVM code generation, creating deploy_* artifacts for full TIDL offload
import argparse
import tvm
import os
import sys
import numpy as np
import tensorflow as tf
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tidl_import import tidl_check_model
from tvm.contrib import cc

parser = argparse.ArgumentParser()
parser.add_argument("modelName", help="Model name")
parser.add_argument("--forced_arm_offload", "-a", help="Force ARM-only execution", action='store_true', default=False)
parser.add_argument("--forced_tidl_offload", "-t", help="Force TIDL offload", action='store_true', default=False)

try:
    args = parser.parse_args()
except SystemExit:
    quit()

forced_dim_expansion = True

if args.modelName == "mobileNet1":
  model      = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
  input_node = "input"
  out_node   = 'MobilenetV1/Predictions/Reshape_1'
  model_input_shape = (224,224,3)
elif args.modelName == "mobileNet2":
  model = "./mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
  input_node = "input"
  out_node   = 'MobilenetV2/Predictions/Reshape_1'
  model_input_shape = (224,224,3)
elif args.modelName == "mobileNet3":
  model = "./mobileNet3/v3-large_224_1.0_float.pb"
  input_node = "input"
  out_node   = 'MobilenetV3/Predictions/Reshape'
  model_input_shape = (224,224,3)
elif args.modelName == "tidl_inceptionv1":
  input_node = "Placeholder"
  model = "./inceptionv1/inception_v1_fbn.pb"
  out_node = "softmax/Reshape"
  model_input_shape = (224,224,3)
  args.forced_tidl_offload = True
  args.forced_arm_offload = False
elif args.modelName == "inceptionv1":
  model = "./inceptionv1/classify_image_graph_def-with_shapes.pb"
  input_node = "DecodeJpeg/contents"
  out_node = "softmax"
  model_input_shape = (299,299,3)
  forced_dim_expansion = False
elif args.modelName == "inceptionv3":
  model  = "./inceptionv3/inception_v3_2016_08_28_frozen-with_shapes.pb"
  input_node = 'input'
  out_node   = 'InceptionV3/Predictions/Softmax'
  model_input_shape = (299,299,3)
else:
  print("Model:" + str(args.modelName) + " not supported!")
  quit()

artifacts_folder = "./output/" + args.modelName
if not os.path.exists(artifacts_folder):
   #Create outputfolder
   os.makedirs(artifacts_folder)
else:
   #Remove all filder from output folder
   filelist = [ f for f in os.listdir(artifacts_folder)]
   for f in filelist:
      os.remove(os.path.join(artifacts_folder, f)) 

image = './airshow.jpg'

data_shape_input = list(model_input_shape)
if forced_dim_expansion:
  data_shape_input.insert(0,1)

data_shape_input = tuple(data_shape_input) # Prepend batch size
print(data_shape_input)

#target = "llvm"
target = "llvm -target=armv7l-linux-gnueabihf"

plsdk_devkit = "/home/x0157990/ti-processor-sdk-linux-am57xx-evm-06.01.00.08/linux-devkit/sysroots/x86_64-arago-linux/usr/bin/"
tidl_calib_tool  = plsdk_devkit + "eve_test_dl_algo_ref.out"
tidl_import_tool = plsdk_devkit + "tidl_model_import.out"
arm_gcc          = plsdk_devkit + "arm-linux-gnueabihf-g++"

if not args.forced_arm_offload:
  tidl_offload = tidl_check_model(model, image, 'tidl_subgraph', model_input_shape,
                                  tidl_import_tool, tidl_calib_tool, artifacts_folder)
else:
  tidl_offload = False

if args.forced_tidl_offload and not tidl_offload:
  print("This model can do only TIDL offload, but it failed")
  quit()

if args.forced_arm_offload and not args.forced_tidl_offload:
  tidl_offload = False
  print("FORCING ARM EXECUTION")

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
  with tf.compat.v1.gfile.GFile(model, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
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
    print("DJDBG:" + str(shape_dict))
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

