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
""" Run inference using DLR or TVM Runtime """


import os
import sys
from platform import processor


def run_model(artifacts_folder: str, input_dict, use_dlr: bool):
  """ Run model with given input using DLR or TVM Runtime.

  Parameters
  ----------
  artifacts_folder:
      Folder containing compilation artifacts.
  input_dict:
      Dictionary of input name (str) to input data (numpy.ndarray).
  use_dlr:
      If True, use DLR. If False, use the TVM runtime directly.

  Return
  ------
  results: List of result tensors.
  """

  if use_dlr:
    from dlr import DLRModel

    module = DLRModel(artifacts_folder)
    results = module.run(input_dict)

    # get optional tidl info, run with TIDL_RT_PERFSTATS=1
    if os.environ.get("TIDL_RT_PERFSTATS"):
      perf_data = module.get_TI_benchmark_data()
      print(perf_data)

  else:
    import tvm
    from tvm.contrib import graph_executor as runtime

    loaded_json = open(artifacts_folder + "/deploy_graph.json").read()
    loaded_lib = tvm.runtime.load_module(artifacts_folder + "/deploy_lib.so")
    loaded_params = bytearray(open(artifacts_folder + "/deploy_param.params", "rb").read())

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
    results = []
    for i in range(module.get_num_outputs()):
      results.append(module.get_output(i).asnumpy())

    # get optional tidl info, run with TIDL_RT_PERFSTATS=1
    if os.environ.get("TIDL_RT_PERFSTATS"):
      import ctypes
      for c in range(16):
        try:
          func = loaded_lib.get_function(f'tidl_get_custom_data_{c}')
        except AttributeError as e:
          break
        vec_void = func()
        # vec_void is pointer to C++ std::vector<uint64_t>, hack into memory layout
        pf_ptr = ctypes.cast(vec_void, ctypes.POINTER(ctypes.POINTER(ctypes.c_ulonglong)))
        pf = pf_ptr[0]
        print(f"tidl_{c}: cp_in {pf[1]-pf[0]} process {pf[3]-pf[2]} cp_out {pf[5]-pf[4]}")

  print(artifacts_folder + ": inference execution finished")
  return results


def infer_model(model_name, platform, is_target, is_dlr, w_tidl, w_c7x, batch_size=0):
  """ Run model model inference for a single (platform, target, tidl, c7x) config """
  from prepostproc import get_test_inputs, check_test_results
  from utils import get_artifacts_folder

  artifacts_folder = get_artifacts_folder(model_name, platform, is_target, w_tidl, w_c7x,
                                          batch_size)
  if not os.path.exists(artifacts_folder):
    raise Exception(f"{artifacts_folder} does not exist for inference")
  print(f"Running inference with deployable module in {artifacts_folder} ...")

  input_list = get_test_inputs(model_name, batch_size)
  res = run_model(artifacts_folder, input_list[0], is_dlr)
  passed = check_test_results(model_name, res, artifacts_folder)
  print("Pass" if passed else "Fail")
  return passed


def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', nargs='?')
  parser.add_argument('--platform', action='store',
                      default="J7",
                      help='Compile model for which platform (J7, J721S2)')
  parser.add_argument('--dlr', action='store_true',
                      default=True,
                      help="Use DLR runtime for inference")
  parser.add_argument('--tvm', action='store_false',
                      dest='dlr',
                      help="Use TVM runtime for inference")
  parser.add_argument('--tidl', action='store_true',
                      default=True,
                      help="Enable TIDL offload")
  parser.add_argument('--notidl', action='store_false',
                      dest="tidl",
                      help="Disable TIDL offload")
  parser.add_argument('--c7x', action='store_true',
                      default=False,
                      help="Enable C7x code generation")
  parser.add_argument('--noc7x', action='store_false',
                      dest="c7x",
                      help="Disable C7x code generation")
  parser.add_argument('--batch_size', action='store',
                      default=0, type=int,
                      help='Overwrite default batch size in the model, 0 means no overwrite')
  args = parser.parse_args()

  assert(args.model_name is not None), "Please specify a model name"
  assert(args.platform in ["J7", "J721S2", "AM62A"]), f"Platform {args.platform} is not supported"

  return args


if __name__ == "__main__":
  args = parse_args()
  is_target = (processor() == "aarch64")

  ret = False
  try:
    ret = infer_model(args.model_name, args.platform, is_target, args.dlr, args.tidl, args.c7x,
                      args.batch_size)
  except Exception as ex:
    print(ex)
    ret = False

  print(f"infer_model {'succeed' if ret else 'fail'}ed: {args.model_name} {args.platform} "
        f"{'target' if is_target else 'host'} {'dlr' if args.dlr else 'tvm'} "
        f"{'tidl' if args.tidl else 'notidl'} {'c7x' if args.c7x else 'noc7x'}"
        f"{(' bs'+str(args.batch_size)) if args.batch_size != 0 else ''}")
  sys.exit(0 if ret else 1)
