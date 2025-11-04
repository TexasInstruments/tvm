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
from typing import Tuple, Dict, List, Optional, Any

import tvm
from tvm import relay
from tvm.runtime import NDArray
from tvm.contrib.tidl.c7x import supported_platform
from tvm.relay.backend.contrib.tidl.tidl import find_dynamic_shape,unpack_composites

def convert_model_to_relay_IR(model_path: str, 
                              input_shape_dict: List[Dict[str, Any]]) -> Tuple[Optional[tvm.IRModule], Optional[Dict[str, NDArray]], Optional[str]]:
  """Convert model from ONNX/tflite frameworks to Relay IR format"""
  ### Checks ###
  if model_path is None or input_shape_dict is None:
    print("Model path and input details are not provided")
    return None, None, None
  if not os.path.exists(model_path):
    print("Model path does not exist")
    return None, None, None
  
  ### Get type of model ####
  model_type = os.path.splitext(model_path)[1][1:]
  if model_type not in ['tflite', 'onnx']:
      print("ERROR : Only tflite/onnx models can be converted to Relay IR internally. Please convert your model to Relay IR and pass converted 'mod', 'params' arguments to compile_model()")
      return None, None, None
  
  if model_type == 'onnx':
    import onnx
    try:
      onnx_model = onnx.load(model_path)
    except Exception as e:
      print(f"ERROR: Loading ONNX model failed with exception - {e}")
      return None, None, None
    
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=input_shape_dict
    )
  elif model_type == 'tflite':
    import tflite
    try:
      with open(model_path, "rb") as fp:
          tflite_model = tflite.Model.GetRootAsModel(fp.read(), 0)
    except Exception as e:
      print(f"ERROR: Loading Tflite model failed with exception - {e}")
      return None, None, None

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict=input_shape_dict
    )
  return mod, params, model_type

def compile_model(platform: str,
                  compile_for_device: bool,
                  enable_tidl_offload: bool,
                  delegate_options: Dict[str, Any],
                  calibration_input_list: List[Dict[str, NDArray]],
                  model_path: str = None,
                  input_shape_dict: List[Dict[str, Any]] = None,              
                  mod: tvm.IRModule = None,
                  params: Dict[str, NDArray] = None
                  ) -> bool:
  """ Compile model for TVM inference based on the parameters specified

  Parameters
  ----------
  platform :
      ["am68pa", "am68a", "am69a", "am67a", "am62a]
      Converted internally to one of the following
      ["J7", "J721S2", "J784S4", "J722S", "AM62A"]
  compile_for_device:
      True => Compile module for inference on device (aarch64).
      False => Compile module for inference on host (x86).
  enable_tidl_offload:
      Set to True to enable TIDL offload.
  delegate_options:
      TIDL offload related options specified in the form of a dictionary
  calibration_input_list :
      A dictionary where the key is input name and the value is input tensor.
  model_path : (Optional)
      Path to the model file. Supported formats: tflite, onnx
  input_shape_dict : (Optional)
      A list of dictionaries where each dictionary contains the input shape.
      Example: [{'input_1' : (1, 3, 224, 224)}]
  mod : (Optional)
      Input Relay IR module.
  params : (Optional)
      The parameter dict used by Relay.
  
  User expected to provide either (model_path, input_details) or (mod, params) of the optional arguments
  
  Return
  ------
  True for success, False for failure.
  """
  import copy
  delegate_options_copy = copy.deepcopy(delegate_options)

  enable_c7x_codegen = False
  if "advanced_options:c7x_codegen" in delegate_options_copy:
    c7x_codegen = delegate_options_copy["advanced_options:c7x_codegen"]
    enable_c7x_codegen = (c7x_codegen > 0)

  model_type = None
  if mod is None or params is None:
    mod, params, model_type = convert_model_to_relay_IR(model_path, input_shape_dict)
  
    if mod is None or params is None:
      print("Conversion to Relay IR format failed")
      return False

  if "artifacts_folder" not in delegate_options_copy:
    raise Exception("Required option 'artifacts_folder' is not set!")
  
  artifacts_folder = delegate_options_copy["artifacts_folder"]
  tidl_tensor_bits = delegate_options_copy.get("tensor_bits", 8)

  assert tidl_tensor_bits in [8, 16, 32]
  assert supported_platform(platform)

  tidl_tools_path, arm_gcc, status = setup_tool_paths(enable_tidl_offload, enable_c7x_codegen,
                                                      compile_for_device)
  if not status:
    return False
  
  reuse_tidl_artifacts = False
  if os.environ.get("REUSE_TIDL_ARTIFACTS", None) is not None:
    reuse_tidl_artifacts = True
    if not os.path.isdir(artifacts_folder):
      print(f'\n\nWARNING: Cannot reuse TIDL artifacts since artifacts folder "{artifacts_folder}" is not present\n')
      reuse_tidl_artifacts = False
  else:
    if os.listdir(artifacts_folder):
      raise Exception("'artifacts_folder' is not empty - please clear the folder and re-run !")

  # If compiling for the device, generate aarch64 code for unsupported layers
  target = "llvm"
  if compile_for_device:
    target += " -device=arm_cpu -mtriple=aarch64-linux-gnu"

  if enable_tidl_offload or enable_c7x_codegen:
    from tvm.relay.backend.contrib.tidl import tidl
    ti_offload_compiler = tidl.TIOffloadCompiler(
                                   platform=platform, # TI device category (E.g. J7)
                                   tidl_tools_path=tidl_tools_path,
                                   enable_tidl_offload=enable_tidl_offload,
                                   compile_for_device=(1 if compile_for_device else 0),
                                   reuse_tidl_artifacts = reuse_tidl_artifacts,
                                   delegate_options=delegate_options_copy)
    # Perform partitioning
    mod, _ = ti_offload_compiler.enable(mod, params, calibration_input_list)

    # Build the Relay module to run on the Graph Executor
    fmod: relay.backend.executor_factory.GraphExecutorFactoryModule
    # Compile the ARM deployable module that will contain the C7x deployable module
    with tidl.build_config(ti_offload_compiler=ti_offload_compiler):
      fmod = relay.build_module.build(mod, target=target, params=params)

    # Remove params used by TIDL subgraphs from deployable module params since they
    # are already included in TIDL subgraph artifacts
    params = fmod.get_params()
    tidl.remove_tidl_params(params)

  else:
    # Reject dynamic shape/network - in case TIDL offlaod is enabled, this check happens after OD post processing is handled
    if model_type == "onnx":
      if find_dynamic_shape(mod):
        print("\n\nDynamic shape/network not supported by TVM+TIDL yet!!!\n\n")
        return False
    mod = unpack_composites(mod,"tidl")
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

def setup_tool_paths(enable_tidl_offload: bool,
                     enable_c7x_codegen: bool,
                     compile_for_device: bool) -> Tuple[str, str, bool]:
  """Read the appropriate environment variables and return tool paths"""
  tidl_tools_path = None
  arm_gcc = None

  try:
    if enable_tidl_offload:
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
    raise Exception(f"${relay_import_lib} does not exist!")
  return tidl_tools_path

def get_arm_compiler():
  arm_gcc_path = os.getenv("ARM64_GCC_PATH")
  if arm_gcc_path is None:
    raise Exception("Environment variable ARM64_GCC_PATH is not set!")
  arm_gcc = os.path.join(arm_gcc_path, "bin", "aarch64-none-linux-gnu-g++")
  if not os.path.exists(arm_gcc):
    raise Exception(f"${arm_gcc} does not exist!")
  return arm_gcc

def check_c7x_compiler_path():
  cgt7x_root = os.getenv("CGT7X_ROOT")
  if cgt7x_root is None:
    raise Exception("Set environment variable CGT7X_ROOT to location of C7000 Code Generation Tools")
  cl7x_bin = os.path.join(cgt7x_root, "bin", "cl7x")
  if not os.path.exists(cl7x_bin):
    raise Exception(f"{cl7x_bin} does not exist!")
