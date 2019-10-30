import tvm
import os
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
from tvm.relay.backend.interpreter import Value, TupleValue, TensorValue
from tvm.relay.backend.interpreter import RefValue, ConstructorValue
from tvm.contrib.download import download_testdata
######################################################################
# Create a simple network
# -----------------------
# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.
#data_shape_input = [1,3,6,4]
data_shape_input = [1,3,224,224]
x = relay.var("x", shape=data_shape_input)

q = relay.TidlInference(x, num_labels=5)

func = relay.Function([x], q)

net = relay.Module.from_expr(func)
print(net)
#data = np.random.uniform(-1, 1, size=data_shape_input).astype("float32")
######################################################################
# Load a test image
# ------------------------------------
# A single cat dominates the examples!
from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 224))
image_data = np.asarray(img).astype("float32")
# after expand_dims, we have format NHWC
image_data = np.expand_dims(image_data, axis=0)
# NHWC -> NCWH
image_data = image_data.transpose(0, 3, 1, 2)

# preprocess image as described here:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
# Rescale from (0..255) to (-1..+1) range
image_data[:, 0, :, :] = 2.0 / 255.0 * image_data[:, 0, :, :] - 1
image_data[:, 1, :, :] = 2.0 / 255.0 * image_data[:, 1, :, :] - 1
image_data[:, 2, :, :] = 2.0 / 255.0 * image_data[:, 2, :, :] - 1
print('input', image_data.shape)
data = image_data
######################################################################
print("---CREATE----EXECUTOR-----")
intrp = relay.create_executor("debug", ctx=tvm.cpu(0))
print("---EVALUATE---------------")
result = intrp.evaluate(func)(data)
print("-----------------")
print("--INTERP RESULT--")
print("INPUT:" + str(data))
print("RELAY:" + str(result))
#print("NUMPY:" + str(np.argsort(data, axis=1)))
print("-----------------")

#target = "llvm"
target = "llvm -target=armv7l-linux-gnueabihf"

graph, lib, params = relay.build_module.build(net, target)
print(params)

tmp_folder = os.getcwd() + "/"
path_lib = tmp_folder + "deploy_lib.tar"
path_graph = tmp_folder + "deploy_graph.json"
path_params = tmp_folder + "deploy_param.params"
lib.export_library(path_lib)
with open(path_graph, "w") as fo:
    fo.write(graph)
with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))

if "armv7" in target:
  print("Target is armv7, so execution on x86 not supported")
  quit()

print("RUNNING BUILT LIBRARY!!!")
ctx = tvm.context(target, 0)
module = runtime.create(graph, lib, ctx)
module.set_input("x", data)
module.run()
out = module.get_output(0).asnumpy()
print(data)
print(out)
