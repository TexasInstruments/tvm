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
""" Compile a model defined in "models" to TVM deployable module """


import os
import sys

def compile_relay(mod_orig, params, input_list,
                  platform, is_target, w_tidl, w_c7x, artifacts_folder, tidl_bits=8):
  """ Compile a model in Relay IR graph for a single (platform, target, tidl, c7x) config

  Parameters
  ----------
  mod_orig : tvm.relay.Module
      Original Relay IR graph
  params : dict of str to tvm.NDArray
      The parameter dict to be used by relay
  platform : str
      in ["J7", "J721S2", "AM672A"]
  input_list : list of dictionary for multiple calibration data
      A dictionary where the key in input name and the value is input tensor
  is_target : bool
      building for target or host (emulation)
  w_tidl : bool
      with TIDL offload or not
  w_c7x : bool
      with c7x code generation for layers not offloaded to TIDL or not
  Return
  ------
  True for success, False for failure
  """
  from tvm.relay.backend.contrib.tidl import tidl
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


  try:
    tidl_tools_path = get_tidl_tools_path()
    if is_target:
      arm_gcc = get_arm_compiler()
    if w_c7x:
      cgt7x_root = get_c7x_compiler_path()
  except Exception as ex:
    print(f"{__file__}: Skip compilation because: {ex}")
    return False

  os.makedirs(artifacts_folder, exist_ok = True)
  path_lib = os.path.join(artifacts_folder, "deploy_lib.so")
  path_graph = os.path.join(artifacts_folder, "deploy_graph.json")
  path_params = os.path.join(artifacts_folder, "deploy_param.params")
  [os.path.exists(f) and os.remove(f) for f in [path_lib, path_graph, path_params]]

  tidl_compiler = tidl.TIDLCompiler(platform=platform, version="7.3",
                                    tidl_tools_path=tidl_tools_path,
                                    artifacts_folder=artifacts_folder,
                                    tensor_bits=tidl_bits,
                                    max_num_subgraphs=((64 if w_c7x else 16) if w_tidl else 0),
                                    deny_list="",
                                    c7x_codegen=(1 if w_c7x else 0),
                                    accuracy_level=(1 if (tidl_bits == 8) else 0),
                                    advanced_options={'calibration_iterations': 10}
                                   )

  if w_tidl or w_c7x:
    mod, status = tidl_compiler.enable(mod_orig, params, input_list)
  else:
    mod, status = mod_orig, 0

  target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu" if is_target else "llvm"

  with tidl.build_config(tidl_compiler=tidl_compiler):
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
  tidl.remove_tidl_params(params)

  if is_target:
    lib.export_library(path_lib, cc=arm_gcc)
  else:
    lib.export_library(path_lib)
  with open(path_graph, "w") as fo:
    fo.write(graph)
  with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))

  print("Artifacts can be found at " + artifacts_folder)
  return True


def compile_model(model_name, platform, is_target, w_tidl, w_c7x):
  """ Compile a model for a single (platform, target, tidl, c7x) config """
  from models import get_relay_model, get_tidl_bits
  from prepostproc import get_calib_inputs
  from utils import get_artifacts_folder

  mod, params = get_relay_model(model_name)
  input_list = get_calib_inputs(model_name)
  artifacts_folder = get_artifacts_folder(model_name, platform, is_target, w_tidl, w_c7x)
  ret = compile_relay(mod, params, input_list, platform, is_target, w_tidl, w_c7x,
                      artifacts_folder, tidl_bits=get_tidl_bits(model_name))
  return ret


def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', nargs='?')
  parser.add_argument('--platform', action='store',
                      default="J7",
                      help='Compile model for which platform (J7, J721S2)')
  parser.add_argument('--target', action='store_true',
                      default=True,
                      help="Compile for target")
  parser.add_argument('--host', action='store_false',
                      dest="target",
                      help="Compile for host (emulation)")
  parser.add_argument('--tidl', action='store_true',
                      default=True,
                      help="With TIDL offload")
  parser.add_argument('--notidl', action='store_false',
                      dest="tidl",
                      help="Without TIDL offload")
  parser.add_argument('--c7x', action='store_true',
                      default=False,
                      help="With C7x code generation")
  parser.add_argument('--noc7x', action='store_false',
                      dest="c7x",
                      help="Without C7x code generation")
  args = parser.parse_args()

  assert(args.model_name is not None), "Please specify a model name"
  assert(args.platform in ["J7", "J721S2"]), f"Platform {args.platform} is not supported"

  return args


if __name__ == "__main__":
  args = parse_args()
  ret = False
  try:
    ret = compile_model(args.model_name, args.platform, args.target, args.tidl, args.c7x)
  except Exception as ex:
    print(ex)
    ret = False

  print(f"compile_model {'succeed' if ret else 'fail'}ed: {args.model_name} {args.platform} "
        f"{'target' if args.target else 'host'} {'tidl' if args.tidl else 'notidl'} "
        f"{'c7x' if args.c7x else 'noc7x'}")
  sys.exit(0 if ret else 1)

