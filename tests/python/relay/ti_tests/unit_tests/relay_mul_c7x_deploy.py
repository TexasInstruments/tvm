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
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(epilog='e.g. python3 ./relay_mul_j7_deploy.py')
parser.add_argument('--dlr', action='store_true',
                    default=True,
                    help='run inference with DLR runtime (Default)')
parser.add_argument('--tvm', action='store_false',
                    dest='dlr',
                    help='run inference with TVM runtime')
parser.add_argument('--target', action='store_true',
                    default=True,
                    help='run inference on target device (ARM core) (Default)')
parser.add_argument('--host', action='store_false',
                    dest='target',
                    help='run inference on host with host emulation (e.g. x86_64)')
args = parser.parse_args()

if args.dlr:
    from dlr import DLRModel
else:
    import tvm
    from tvm.contrib import graph_runtime as runtime


######################################################################
# Run the model with relay runtime
# ---------------------------------------------
def run_module(model_name, input_tensor, is_nchw):
    # load deployable module
    artifacts_dir = "artifacts_" + model_name + ("_target" if args.target else "_host") + "/"
    quant = model_name.endswith('_quant')

    a_shape = (672, 14, 14)
    b_shape = (672, 1, 1)
    a_data = np.random.randint(0, 10, a_shape).astype('float32')
    b_data = np.random.randint(0, 10, b_shape).astype('float32')

    input_variables = ["a", "b"]
    input_values = [a_data, b_data]
    input_dict = dict(zip(input_variables, input_values))

    if args.dlr:
        module = DLRModel(artifacts_dir)
        results = module.run(input_dict)
        tvm_outputs = results

    else:
        loaded_json = open(artifacts_dir + "deploy_graph.json").read()
        loaded_lib = tvm.runtime.load_module(artifacts_dir + "deploy_lib.so")
        loaded_params = bytearray(open(artifacts_dir + "deploy_param.params", "rb").read())

        # create a runtime executor module
        module = runtime.create(loaded_json, loaded_lib, tvm.cpu())

        # load params into the module
        module.load_params(loaded_params)

        # feed input data
        for key, value in input_dict.items():
            module.set_input(key, value)

        # run
        module.run()

        # get output
        tvm_outputs = []
        for i in range(module.get_num_outputs()):
            tvm_outputs.append(module.get_output(i).asnumpy())

    print(model_name + " execution finished")
    return tvm_outputs

if __name__ == '__main__':

    outputs_c7x = run_module("relay_mul_c7x", "a", True)
    print("(With C7x codegen) Relay Mul output:")
    print(outputs_c7x[0])
