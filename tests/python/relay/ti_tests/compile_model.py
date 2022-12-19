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
""" Compile a model from models.py to a TVM deployable module """

import os
import sys

from tvm.contrib.tidl import compile
from tvm.contrib.tidl.c7x import supported_platform

def compile_model(model_name: str, platform: str,
                  compile_for_device: bool, enable_tidl_offload: bool, enable_c7x_codegen: bool,
                  batch_size:int=0):
  """ Compile a model based on the parameters specified.

  Parameters
  ----------
  model_name :
      name of the model as specified in models.py (E.g. mv2_onnx).
  platform :
      in ["J7", "J721S2"", "AM62A"]
  compile_for_device:
      True => Compile module for inference on device (aarch64).
      False => Compile module for inference on host (x86).
  enable_tidl_offload:
      Set to True to enable TIDL offload.
  enable_c7x_codegen:
      True => Enable c7x code generation for layers not offloaded to TIDL. i.e. entire network runs on the C7x.
      False => Enable Arm code generation for layers not offloaded to TIDL. i.e. Unsupported layers are run on Arm (aarch64).
  batch_size:
      0: use the batch size that comes with the model
      otherwise: override the default batch size
  Return
  ------
  True for success, False for failure.
  """

  from models import get_relay_model, get_tidl_bits
  from prepostproc import get_calib_inputs
  from utils import get_artifacts_folder

  if batch_size != 0:
    os.environ["TIDL_RELAY_MAX_BATCH_SIZE"] = str(batch_size)

  # Obtain model and convert to Relay
  mod, params = get_relay_model(model_name, batch_size)

  # Get inputs to use for calibraton (required for TIDL offload)
  calibration_input_list = get_calib_inputs(model_name, batch_size)

  # Generate a name for the artifacts folder based on the model and other parameters
  artifacts_folder = get_artifacts_folder(model_name, platform, compile_for_device,
                                          enable_tidl_offload, enable_c7x_codegen,
                                          batch_size)

  # Compile the model using TVM and place the output in the artifacts_folder
  result = compile.compile_relay(mod, params, calibration_input_list, platform, compile_for_device,
                                 enable_tidl_offload, enable_c7x_codegen,
                                 artifacts_folder, tidl_tensor_bits=get_tidl_bits(model_name))
  return result


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
  assert(supported_platform(args.platform)), f"Platform {args.platform} is not supported"

  return args


if __name__ == "__main__":
  args = parse_args()
  ret = False
  try:
    ret = compile_model(args.model_name, args.platform, args.target, args.tidl, args.c7x,
                        args.batch_size)
  except Exception as ex:
    print(ex)
    ret = False

  print(f"compile_model {'succeed' if ret else 'fail'}ed: {args.model_name} {args.platform} "
        f"{'target' if args.target else 'host'} {'tidl' if args.tidl else 'notidl'} "
        f"{'c7x' if args.c7x else 'noc7x'}"
        f"{(' bs'+str(args.batch_size)) if args.batch_size != 0 else ''}")
  sys.exit(0 if ret else 1)

