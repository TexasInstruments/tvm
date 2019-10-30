import tvm
import os
import numpy as np
from tvm.contrib import graph_runtime

data_shape_input = [1,3,224,224]
data = np.random.uniform(-1, 1, size=data_shape_input).astype("float32")
#data = np.random.uniform(0, 255, size=data_shape_input).astype("uint8")

tmp_folder = os.getcwd() + "/"
path_lib = tmp_folder + "deploy_lib.tar"
path_graph = tmp_folder + "deploy_graph.json"
path_params = tmp_folder + "deploy_param.params"

# load the module back.
loaded_json = open(path_graph).read()
loaded_lib = tvm.module.load(path_lib)
loaded_params = bytearray(open(path_params, "rb").read())

module = graph_runtime.create(loaded_json, loaded_lib, ctx=tvm.cpu(0))
module.load_params(loaded_params)
module.set_input("x", data)
module.run()
out = module.get_output(0).asnumpy()
print(data)
print(out)
