import tvm
import os
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
from tvm.relay.backend.interpreter import Value, TupleValue, TensorValue
from tvm.relay.backend.interpreter import RefValue, ConstructorValue

######################################################################
# Create a simple network
# -----------------------
# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.
data_shape_input = [1,4]
data_shape_output = [1,4]
x = relay.var("x", shape=data_shape_input)

y1 = relay.nn.relu(x)
y2 = relay.nn.relu(x)
y1 = relay.add(y1, relay.const(1.0, "float32"))
y2 = relay.add(y2, relay.const(1.0, "float32"))
y = relay.add(y1, y2)
q = relay.TidlSort(y)

#y = relay.add(x, relay.const(1.0, "float32"))
#q = relay.TidlSort(y)
#q = relay.argsort(y)

func = relay.Function([x], q)
data = np.random.uniform(-1, 1, size=data_shape_input).astype("float32")

intrp = relay.create_executor("debug", ctx=tvm.cpu(0))
result = intrp.evaluate(func)(data)
print("-----------------")
print("--INTERP RESULT--")
print(result)
print("-----------------")

net = relay.Module.from_expr(func)
print(net)
target = "llvm"
#target = "llvm -target=armv7l-linux-gnueabihf"
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

ctx = tvm.context(target, 0)
module = runtime.create(graph, lib, ctx)
module.set_input("x", data)
module.run()
out = module.get_output(0).asnumpy()
print(data)
print(out)
