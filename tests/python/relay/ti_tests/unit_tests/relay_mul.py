#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from typing import List
import numpy as np


def relay_compile(model_name, mod_orig, params, model_input_list, args, max_num_subgraphs=16):
    """ Compile a model in Relay IR graph

    Parameters
    ----------
    model_name : string
        Name of the model
    mod_orig : tvm.relay.Module
        Original Relay IR graph
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    model_input_list : list of dictionary for multiple calibration data
        A dictionary where the key in input name and the value is input tensor
    max_num_subgraphs : int
        Max number of subgraphs to offload to TIDL
    Returns
    -------
    status: int
        Status of compilation:
            1  - compilation for TIDL offload succeeded
            -1 - compilation for TIDL offload failed - failure for CI testing
            0  - no compilation due to missing TIDL tools or ARM64 GCC tools
    """

    from tvm.relay.backend.contrib import tidl
    from tvm import relay

    def get_tidl_tools_path():
        tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
        if tidl_tools_path is None:
            raise Exception("Environment variable TIDL_TOOLS_PATH is not set!")
        relay_import_lib = os.path.join(tidl_tools_path, "tidl_model_import_relay.so")
        if not os.path.exists(relay_import_lib):
            raise Exception("${TIDL_TOOLS_PATH}/tidl_model_import_relay.so does not exist!")
        return tidl_tools_path

    def get_arm_compiler():
        arm_gcc_path = os.getenv("ARM64_GCC_PATH")
        if arm_gcc_path is None:
            raise Exception("Environment variable ARM64_GCC_PATH is not set!")
        arm_gcc = os.path.join(arm_gcc_path, "bin", "aarch64-none-linux-gnu-g++")
        if not os.path.exists(arm_gcc):
            raise Exception("${ARM64_GCC_PATH}/aarch64-none-linux-gnu-g++ does not exist!")
        return arm_gcc

    def get_c7x_compiler_path():
        cgt7x_root = os.getenv("CGT7X_ROOT")
        if cgt7x_root is None:
            raise Exception("Environment variable CGT7X_ROOT is not set!")
        cl7x_bin = os.path.join(cgt7x_root, "bin", "cl7x")
        if not os.path.exists(cl7x_bin):
            raise Exception("${CGT7X_ROOT}/cl7x does not exist!")
        return cgt7x_root



    c7x_codegen = 1 if model_name.endswith('_c7x') else 0

    try:
        tidl_tools_path = get_tidl_tools_path()
        if args.target:
            arm_gcc = get_arm_compiler()
        if c7x_codegen == 1:
            cgt7x_root = get_c7x_compiler_path()
            #c7x_codegen = 9
    except Exception as ex:
        print(f"{__file__}: Skip compilation because: {ex}")
        return 0

    tidl_artifacts_folder = "./artifacts_" + model_name +  ("_target" if args.target else "_host")
    os.makedirs(tidl_artifacts_folder, exist_ok = True)
    for root, dirs, files in os.walk(tidl_artifacts_folder, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    tidl_compiler = tidl.TIDLCompiler(platform="J7", version="8.2",
                                      tidl_tools_path=tidl_tools_path,
                                      artifacts_folder=tidl_artifacts_folder,
                                      tensor_bits=16,
                                      max_num_subgraphs=max_num_subgraphs,
                                      deny_list=args.denylist,
                                      c7x_codegen=c7x_codegen,
                                      accuracy_level=0,
                                      advanced_options={'calibration_iterations': 10}
                                     )

    if args.nooffload:
        mod, status = mod_orig, 0
    else:
        mod, status = tidl_compiler.enable(mod_orig, params, model_input_list)

    if status == 1: # TIDL compilation succeeded
        print("Graph execution with TIDL")
    else: # TIDL compilation failed or no TIDL compilation due to missing tools
        print("Graph execution without TIDL")

    if args.target:
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    else:
        target = "llvm"

    with tidl.build_config(tidl_compiler=tidl_compiler):
        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    tidl.remove_tidl_params(params)

    path_lib = os.path.join(tidl_artifacts_folder, "deploy_lib.so")
    path_graph = os.path.join(tidl_artifacts_folder, "deploy_graph.json")
    path_params = os.path.join(tidl_artifacts_folder, "deploy_param.params")
    if args.target:
        lib.export_library(path_lib, cc=arm_gcc)
    else:
        lib.export_library(path_lib)
    with open(path_graph, "w") as fo:
        fo.write(graph)
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(params))

    print("Artifacts can be found at " + tidl_artifacts_folder)
    return status


def compile_model(model_name, args) -> List:
    """Create a relay model, compile it and save input/reference output to file"""
    import tvm
    from tvm import relay

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
    status = relay_compile(model_name, mod, params, input_dict_list, args, max_num_subgraphs=0)
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

    artifacts_dir = "artifacts_" + model_name + ("_target" if args.target else "_host")
    return ["a_data", "b_data", "reference_output", artifacts_dir]

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
    parser.add_argument('--deny', dest='denylist', action='append',
                        help='force Relay operator to be unsupported by TIDL, comma-separated string')
    parser.add_argument('--nooffload', action='store_true',
                        help='produce a host-only deployable module without TIDL offload')

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
            copy_to_evm(model, args.copy_to_evm, artifacts)

    if args.inference:
        status = run_model(model, True)
        if status:
            print(f'{model}: PASSED')
        else:
            print(f'{model}: FAILED')
            sys.exit(1)

    sys.exit(0)