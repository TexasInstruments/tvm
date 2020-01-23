"""
* Copyright (c) {2019} Texas Instruments Incorporated
*
* All rights reserved not granted herein.
*
* Limited License.
*
* Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
* license under copyrights and patents it now or hereafter owns or controls to make,
* have made, use, import, offer to sell and sell ("Utilize") this software subject to the
* terms herein.  With respect to the foregoing patent license, such license is granted
* solely to the extent that any such patent is necessary to Utilize the software alone.
* The patent license shall not apply to any combinations which include this software,
* other than combinations with devices manufactured by or for TI ("TI Devices").
* No hardware patent is licensed hereunder.
*
* Redistributions must preserve existing copyright notices and reproduce this license
* (including the above copyright notice and the disclaimer and (if applicable) source
* code license limitations below) in the documentation and/or other materials provided
* with the distribution
*
* Redistribution and use in binary form, without modification, are permitted provided
* that the following conditions are met:
*
* *     No reverse engineering, decompilation, or disassembly of this software is
* permitted with respect to any software provided in binary form.
*
* *     any redistribution and use are licensed by TI for use only with TI Devices.
*
* *     Nothing shall obligate TI to provide you with source code for the software
* licensed and provided to you in object code.
*
* If software source code is provided to you, modification and redistribution of the
* source code are permitted provided that the following conditions are met:
*
* *     any redistribution and use of the source code, including any resulting derivative
* works, are licensed by TI for use only with TI Devices.
*
* *     any redistribution and use of any object code compiled from the source code
* and any resulting derivative works, are licensed by TI for use only with TI Devices.
*
* Neither the name of Texas Instruments Incorporated nor the names of its suppliers
*
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* DISCLAIMER.
*
* THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("modelName", help="Model name")
parser.add_argument("--forced_arm_offload", "-a", help="Force ARM-only execution", action='store_true', default=False)
parser.add_argument("--forced_tidl_offload", "-t", help="Force TIDL offload", action='store_true', default=False)
parser.add_argument("--batch_size", "-b", help="Batch size", type=int, default=4)
parser.add_argument("--input_node", "-i", help="Input node name", default=None)
parser.add_argument("--output_node", "-o", help="Output node name", default=None)
parser.add_argument("--num_labels", "-l", help="Number of classification classes", type=int, default=0)
parser.add_argument("--calibration_image", "-c", help="Calibration image", default="airshow.jpg")
parser.add_argument("--input_shape", "-s", help="Input shape: H W C, e.g. -s 224 224 3", nargs="+", type=int, default=-1)

try:
    args = parser.parse_args()
except SystemExit:
    quit()

forced_dim_expansion = True
batch_size      = args.batch_size
#tidl_input_node = 'x'
conv2d_kernel_type = None
customModel = False

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
elif args.modelName == "mobileNet1tflite":
  model      = "./mobileNet1tflite/mobilenet_v1_1.0_224.tflite"
  input_node = "input"
  out_node   = 'MobilenetV1/Predictions/Reshape_1'
  model_input_shape = (224,224,3)
elif args.modelName == "mobileNet2tflite":
  model      = "./mobileNet2tflite/mobilenet_v2_1.0_224.tflite"
  input_node = "input"
  out_node   = 'MobilenetV2/Predictions/Reshape_1'
  model_input_shape = (224,224,3)
elif args.modelName == "resNet18v1Onnx":
  model      = "./resNet18v1Onnx/resnet18v1.onnx"
  input_node = "data"
  out_node   = 'resnetv15_dense0_fwd'
  model_input_shape = (224,224,3)
elif args.modelName == "squeezeNetOnnx":
  model      = "./squeezeNetOnnx/squeezenet1.1.onnx"
  input_node = "data"
  out_node   = 'squeezenet0_flatten0_reshape0'
  model_input_shape = (224,224,3)
else:
  model = args.modelName
  print("Custom Model expected:" + args.modelName)
  input_node = args.input_node
  out_node   = args.output_node
  model_input_shape = tuple(args.input_shape)
  print(model)
  print(input_node)
  print(out_node)
  print(str(model_input_shape))
  customModel = True

if customModel:
  artifacts_folder = "./output4/custom"
else:
  artifacts_folder = "./output4/" + args.modelName

if not os.path.exists(artifacts_folder):
   #Create outputfolder
   os.makedirs(artifacts_folder)
else:
   #Remove all filder from output folder
   filelist = [ f for f in os.listdir(artifacts_folder)]
   for f in filelist:
      os.remove(os.path.join(artifacts_folder, f)) 

image = args.calibration_image

data_shape_input = list(model_input_shape)
if forced_dim_expansion:
  data_shape_input.insert(0,batch_size)

data_shape_input = tuple(data_shape_input) # Prepend batch size
print(data_shape_input)

#target = "llvm"
target = "llvm -target=armv7l-linux-gnueabihf"

if os.getenv("TIDL_PLSDK") is None:
  plsdk_devkit = os.getenv('HOME') + "/ti-processor-sdk-linux-am57xx-evm-06.02.00.75" + "/linux-devkit/sysroots/x86_64-arago-linux/usr/bin/"
else: 
  plsdk_devkit = os.getenv('TIDL_PLSDK') + "/linux-devkit/sysroots/x86_64-arago-linux/usr/bin/"
print("PLSDK path set to:" + plsdk_devkit)

tidl_calib_tool  = plsdk_devkit + "eve_test_dl_algo_ref.out"
tidl_import_tool = plsdk_devkit + "tidl_model_import.out"
arm_gcc          = plsdk_devkit + "arm-linux-gnueabihf-g++"

model_path = Path(model)
if not model_path.is_file():
  print('Error: model file ' + model + ' does NOT exist!')
  quit()

if not args.forced_arm_offload:
  tidl_offload, last_node_dim = tidl_check_model(model, image, 'tidl_subgraph', model_input_shape,
                                  tidl_import_tool, tidl_calib_tool, artifacts_folder, conv2d_kernel_type)
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
  output_dim = last_node_dim.split('x')
  if(int(output_dim[1]) != 1):
    print(int(output_dim[1]) + 2)
    print("Base on trace in tempDir folder, last node is not 1D vector! Instead it is:" + last_node_dim)
    quit()

  if(args.num_labels == 0):
    num_labels = int(output_dim[0])
  else:
    num_labels = args.num_labels

  print("num_labels set to " + str(num_labels))
  #x = relay.var(x, shape=data_shape_input)
  x = relay.var(input_node, shape=data_shape_input)
  q = relay.TidlInference(x, num_labels)
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
  with tf.gfile.GFile(model, 'rb') as f:
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
    print("Inut node shape dict:" + str(shape_dict))
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

