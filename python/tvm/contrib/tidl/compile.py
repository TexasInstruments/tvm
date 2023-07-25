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
""" Compile a Relay IR module to TVM deployable module """


import os
from typing import Tuple, Dict, List, Any

import tvm
from tvm import relay
from tvm.runtime import NDArray
from tvm.contrib.tidl.c7x import supported_platform

def compile_relay(mod: tvm.IRModule,
                  params: Dict[str, NDArray],
                  calibration_input_list: List[Dict[str, NDArray]],
                  platform: str,
                  compile_for_device: bool,
                  enable_tidl_offload: bool,
                  enable_c7x_codegen: bool,
                  artifacts_folder: str,
                  tidl_tensor_bits: int = 8,
                  advanced_options: Dict[str, Any] = None) -> bool:
  """ Compile Relay IR module based on the parameters specified

  Parameters
  ----------
  mod :
      Input Relay IR module.
  params :
      The parameter dict used by Relay.
  platform :
      in ["J7", "J721S2", "J784S4", "AM62A"]
  calibration_input_list :
      A dictionary where the key is input name and the value is input tensor.
  compile_for_device:
      True => Compile module for inference on device (aarch64).
      False => Compile module for inference on host (x86).
  enable_tidl_offload:
      Set to True to enable TIDL offload.
  enable_c7x_codegen:
      True => Enable c7x code generation for layers not offloaded to TIDL. i.e. entire network runs on the C7x.
      False => Enable Arm code generation for layers not offloaded to TIDL. Unsupported layers are run on Arm (aarch64).
  tidl_tensor_bits:
      Number of bits used to represent TIDL tensors and weights.
  Return
  ------
  True for success, False for failure.
  """

  assert tidl_tensor_bits in [8, 16, 32]
  assert supported_platform(platform)

  tidl_tools_path, arm_gcc, status = setup_tool_paths(enable_tidl_offload, enable_c7x_codegen,
                                                      compile_for_device)
  if not status:
    return False

  prepare_output_directory(artifacts_folder)

  # If compiling for the device, generate aarch64 code for unsupported layers
  target = "llvm"
  if compile_for_device:
    target += " -device=arm_cpu -mtriple=aarch64-linux-gnu"

  # If TIDL offload is enabled, use TIDLCompiler to partition relay graph
  # for offload subgraphs to TIDL
  # If C7x code generation is enabled, use TIDLCompiler to generate C7x
  # code for TIDL unsupported layers
  if enable_tidl_offload or enable_c7x_codegen:
    from tvm.relay.backend.contrib.tidl import tidl

    # Calibration options corresponding to quantized tensor bits
    advanced_options_default = {
      8 : {
        #'calibration_iterations' : 10,
        'calibration_iterations' : 3,
        # Following options take effect only at accuracy level 9, are ignored otherwise
        'activation_clipping' : 1,
        'weight_clipping' : 1,
        'bias_calibration' : 1,
        'channel_wise_quantization' : 0,
      },
      16 : {
        'calibration_iterations' : 1,
      },
      32 : {
        'calibration_iterations' : 1,
      }
    }

    
    advanced_options_updated = advanced_options_default[tidl_tensor_bits]
    if advanced_options:
      advanced_options_updated.update(advanced_options)

    tidl_compiler = tidl.TIDLCompiler(platform=platform, # TI device category (E.g. J7)
                                      version="8.4", # Processor SDK version, currently unused
                                      tidl_tools_path=tidl_tools_path,
                                      artifacts_folder=artifacts_folder,
                                      tensor_bits=tidl_tensor_bits,
                                      max_num_subgraphs=((64 if enable_c7x_codegen else 16) if enable_tidl_offload else 0),
                                      deny_list="",
                                      c7x_codegen=(1 if enable_c7x_codegen else 0),
                                      accuracy_level=(1 if (tidl_tensor_bits == 8) else 0),
                                      advanced_options=advanced_options_updated)
    # Perform partitioning
    mod, _ = tidl_compiler.enable(mod, params, calibration_input_list)

    # Build the Relay module to run on the Graph Executor
    fmod: relay.backend.executor_factory.GraphExecutorFactoryModule
    with tidl.build_config(tidl_compiler=tidl_compiler):
      fmod = relay.build_module.build(mod, target=target, params=params)

    # Remove params used by TIDL subgraphs from deployable module params since they
    # are already included in TIDL subgraph artifacts
    params = fmod.get_params()
    tidl.remove_tidl_params(params)

  else:
    fmod = relay.build(mod, target=target, params=params)
    params = fmod.get_params()

  lib = fmod.get_lib()

  # Export the modules into a shared object
  path_lib = os.path.join(artifacts_folder, "deploy_lib.so")
  if compile_for_device:
    lib.export_library(path_lib, cc=arm_gcc)
  else:
    lib.export_library(path_lib)

  # Write the graph JSON and params to the artifacts directory
  path_graph = os.path.join(artifacts_folder, "deploy_graph.json")
  with open(path_graph, "w") as fo:
    fo.write(fmod.get_graph_json())

  path_params = os.path.join(artifacts_folder, "deploy_param.params")
  with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))

  print("Artifacts can be found at " + artifacts_folder)
  return True


def prepare_output_directory(output_dir: str):
  """Create and clean the output directory"""

  if os.environ.get("TIDL_REBUILD_ONLY", None) is not None:
    return

  # create the directory if its not already preset
  os.makedirs(output_dir, exist_ok=True)

  # Delete all files and directories from within dir
  for root, dirs, files in os.walk(output_dir, topdown=False):
    [os.remove(os.path.join(root, f)) for f in files]
    [os.rmdir(os.path.join(root, d)) for d in dirs]

def setup_tool_paths(enable_tidl_offload: bool,
                     enable_c7x_codegen: bool,
                     compile_for_device: bool) -> Tuple[str, str, bool]:
  """Read the appropriate environment variables and return tool paths"""
  tidl_tools_path = None
  arm_gcc = None

  try:
    if enable_tidl_offload or enable_c7x_codegen:
      tidl_tools_path = get_tidl_tools_path()
    if compile_for_device:
      arm_gcc = get_arm_compiler()
    if enable_c7x_codegen:
      check_c7x_compiler_path()

    return tidl_tools_path, arm_gcc, True

  except Exception as ex:
    print(f"{__file__}: Skip compilation because: {ex}")
    return None, None, False


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

def check_c7x_compiler_path():
  cgt7x_root = os.getenv("CGT7X_ROOT")
  if cgt7x_root is None:
    raise Exception("Set environment variable CGT7X_ROOT to location of C7000 Code Generation Tools")
  cl7x_bin = os.path.join(cgt7x_root, "bin", "cl7x")
  if not os.path.exists(cl7x_bin):
    raise Exception("${CGT7X_ROOT}/cl7x does not exist!")
