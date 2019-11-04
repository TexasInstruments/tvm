# TVM code generation, creating deploy_* artifacts for full TIDL offload
import tvm
import os
import numpy as np
import tensorflow as tf
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tidl_import import tidl_check_model

#model = './mobileNet2/mobilenet_v2_1.0_224_frozen.pb'
model = './mobileNet2/mobilenet_v2_1.0_224_frozen.pb'
#model = './mobileNet3/v3-large_224_1.0_float.pb'
#model = './inceptionv1/classify_image_graph_def-with_shapes.pb'

image = './airshow.jpg'
model_input_shape = (224,224,3)
#target = "llvm"
target = "llvm -target=armv7l-linux-gnueabihf"

tidl_calib_tool  = './eve_test_dl_algo_ref.out'
tidl_import_tool = './tidl_model_import.out'

tidl_offload = tidl_check_model(model, image, 'tidl_subgraph', model_input_shape,
                                tidl_import_tool, tidl_calib_tool)

if tidl_offload:
  print("Offload this model to TIDL")
  data_shape_input = list(model_input_shape)
  data_shape_input.insert(0, 1) # Prepend batch size
  x = relay.var("x", shape=data_shape_input)
  q = relay.TidlInference(x, num_labels=1001)
  func = relay.Function([x], q)
  net = relay.Module.from_expr(func)
  print(net)
  ######################################################################

  graph, lib, params = relay.build_module.build(net, target=target)

else: 
  from tvm.relay.testing import tf as tf_testing
  #import tvm.relay.testing.tf as tf_testing
  print("Run this model on ARM")
  layout = 'NHWC'
  with tf.gfile.FastGFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

    ######################################################################
    # Import the graph to Relay
    # -------------------------
    # Import tensorflow graph definition to relay frontend.
    #
    # Results:
    #   sym: relay expr for given tensorflow protobuf.
    #   params: params converted from tensorflow params (tensor protobuf).
    shape_dict = {'DecodeJpeg/contents': model_input_shape}
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
                                                 shape=shape_dict)

    print("Tensorflow protobuf imported to relay frontend.")
    ######################################################################
    # Relay Build
    # -----------
    # Compile the graph to llvm target with given input specification.
    #
    # Results:
    #   graph: Final graph after compilation.
    #   params: final params after compilation.
    #   lib: target library which can be deployed on target with TVM runtime.
    with relay.build_config(opt_level=3):
      graph, lib, params = relay.build(mod, target=target, params=params)


#print(params)
print("...Start writting conveted model:")
tmp_folder = os.getcwd() + "/"
path_lib = tmp_folder + "deploy_lib.tar"
path_graph = tmp_folder + "deploy_graph.json"
path_params = tmp_folder + "deploy_param.params"
print("...library!")
lib.export_library(path_lib)
print("...graph!")
with open(path_graph, "w") as fo:
  fo.write(graph)
print("...params!")
with open(path_params, "wb") as fo:
  fo.write(relay.save_param_dict(params))

