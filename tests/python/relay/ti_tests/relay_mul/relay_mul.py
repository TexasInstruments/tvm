#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from typing import List
import numpy as np

def compile_model(model_name, args) -> List:
    """Create a relay model, compile it and save input/reference output to file"""
    # Add '..' to Python modules path to import compile_relay
    import sys
    sys.path.append("..")

    import tvm
    from tvm import relay

    from compile_model import compile_relay

    # define graph in relay
    a_shape = (672, 14, 14)
    b_shape = (672, 1, 1)
    a = relay.var('a', dtype="float32", shape=a_shape)
    b = relay.var('b', dtype="float32", shape=b_shape)
    c = relay.multiply(a, b)
    func = relay.Function([a, b], c)

    # create an IRModule containing relay function(s)
    mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

    a_data = np.random.randint(0, 10, a_shape).astype('float32')
    b_data = np.random.randint(0, 10, b_shape).astype('float32')
    params = {}

    # Run with TVM VM executor to create reference output
    reference_output = tvm.relay.create_executor(kind="vm", mod=mod).evaluate()(a_data, b_data)

    # Compile relay module to generate C7x code (max_num_subgraphs=0)
    input_variables = ["a", "b"]
    input_values = [a_data, b_data]
    input_dict_list = dict(zip(input_variables, input_values))

    artifacts_folder = "./artifacts_" + model_name +  ("_target" if args.target else "_host")
    status = compile_relay(mod, params, input_dict_list, "J7",
                          is_target=True, w_tidl=False, w_c7x=True, artifacts_folder=artifacts_folder, tidl_bits=8)

    if status != 1:
        print("TIDL compilation failed")
        sys.exit(1)

    def save_to_file(name: str, data: np.ndarray):
        file = open(name, "wb")
        np.save(file, data)
        file.close()

    save_to_file("a_data", a_data)
    save_to_file("b_data", b_data)
    save_to_file("reference_output", reference_output.asnumpy())

    return ["a_data", "b_data", "reference_output", artifacts_folder]

def copy_to_evm(model:str, ip: str, artifacts: List) -> bool:
    import subprocess
    command = ["scp", "-qr"]
    command.extend(artifacts)
    command.append(sys.argv[0])
    command.append("root@"+ip+":")
    print('Copying files to EVM: ' + ' '.join(command))
    try:
        subprocess.run(command, check=True)
    except:
        return False

    return True


######################################################################
# Run the model with relay runtime
# ---------------------------------------------
def run_model(model_name, is_nchw) -> bool:
    if args.dlr:
        from dlr import DLRModel
    else:
        import tvm
        from tvm.contrib import graph_runtime as runtime

    # load deployable module
    artifacts_dir = "artifacts_" + model_name + ("_target" if args.target else "_host") + "/"
    quant = model_name.endswith('_quant')

    def read_from_file(filename) -> np.ndarray:
        """Read numpy array from file"""
        file = open(filename, "rb")
        data = np.load(file)
        file.close()

        return data

    a_data = read_from_file("a_data")
    b_data = read_from_file("b_data")
    reference_output = read_from_file("reference_output")

    input_variables = ["a", "b"]
    input_values = [a_data, b_data]
    input_dict = dict(zip(input_variables, input_values))

    print(model_name + ": execution started")
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

    print(model_name + ": execution finished")

    # Check results
    return np.array_equal(reference_output, tvm_outputs[0])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true',
                        default=False,
                        help='Compile the model (Default)')
    parser.add_argument('--copy_to_evm',
                        default=None,
                        type=str,
                        help='Copy artifacts to EVM IP specified')
    parser.add_argument('--inference', action='store_true',
                        default=False,
                        help='Run inference with the model')
    parser.add_argument('--target', action='store_true',
                        default=True,
                        help='generate code for target device (Arm core) (Default)')
    parser.add_argument('--host', action='store_false',
                        dest="target",
                        help='generate code for host emulation (e.g. x86_64 core)')

    # Inference arguments
    parser.add_argument('--dlr', action='store_true',
                        default=True,
                        help='run inference with DLR runtime (Default)')
    parser.add_argument('--tvm', action='store_false',
                        dest='dlr',
                        help='run inference with TVM runtime')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Enable logging
    #logging.basicConfig(level=logging.DEBUG)
    #os.environ["TVM_LOG_DEBUG"] = "1"
    #os.environ["TIDL_C7X_CODEGEN_DEBUG"] = "1"
    #os.environ["TIDL_C7X_CODEGEN_DEBUG_BEGIN"] = "1"

    model = "relay_mul_c7x"

    args = parse_args()

    if args.compile:
        artifacts = compile_model(model, args)
        if args.copy_to_evm != None:
            if not copy_to_evm(model, args.copy_to_evm, artifacts):
                sys.exit(1)

    if args.inference:
        status = run_model(model, True)
        if status:
            print(f'{model}: PASSED')
        else:
            print(f'{model}: FAILED')
            sys.exit(1)

    sys.exit(0)
