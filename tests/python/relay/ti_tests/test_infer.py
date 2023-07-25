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
""" Run inference tests for "models" """

import os
import sys
import subprocess
from platform import processor
from models import models

skipped_configs = [
  # od postproc offload only available with tidl
  [ 'mv2_od_onnx', '*', '*', '--notidl', '*', '*', 0 ],
  [ 'yolov5_od_onnx', '*', '*', '--notidl', '*', '*', 0 ],
  [ 'lidar_od_onnx', '*', '*', '--notidl', '*', '*', 0 ],

  # large feature map size in batch processing, unable to import to TIDL
  [ 'mv2_onnx', 'AM62A', '*', '--tidl', '*', '*', 2 ],
  [ 'mv2_pth', 'AM62A', '*', '--tidl', '*', '*', 2 ],

  # Not enough memory to run the model
  [ 'yolo3_mv1_mxnet', 'J7', '--target', '--notidl', '--c7x', '--dlr', 0 ],
  [ 'yolo3_mv1_mxnet', 'J721S2', '--target', '--notidl', '--c7x', '--dlr', 0 ],
  [ 'yolo3_mv1_mxnet', 'J784S4', '--target', '--notidl', '--c7x', '--dlr', 0 ],

  [ 'yolo3_mv1_mxnet', 'AM62A', '--target', '*', '--c7x', '--dlr', 0 ],
  [ 'swin_tiny_timm', 'AM62A', '--target', '*', '--c7x', '--dlr', 0 ],
    # TIDL_RT_OVX/TVM_RT_OVX hang on "Delete TIDL/VM graph..."
  [ 'yolov5_od_onnx', 'AM62A', '--target', '--tidl', '*', '--dlr', 0 ],

  [ 'lidar_od_onnx', '*', '--target', '--tidl', '--c7x', '--dlr', 0 ],
  [ 'lidar_od_onnx', 'AM62A', '--target', '--tidl', '--noc7x', '--dlr', 0 ],
    # liar: sometimes hangs on J7, reboot, run the network by itself, passes
  [ 'lidar_od_onnx', 'J7', '--target', '--tidl', '--noc7x', '--dlr', 0 ],
]

def test_infer(in_models, platforms, dlr_tvm, tidls, c7xs):
  is_target = (processor() == "aarch64")
  t_h = "--target" if is_target else "--host"
  failed_configs = []
  for model in in_models:
    for platform in platforms:
      for d_t in dlr_tvm:
        for t_nt in tidls:
          for c_nc in c7xs:
            # C7x codegen mode does not support host emulation yet
            if (not is_target) and c_nc == "--c7x":
              continue

            for n in (models[model]['batch_size'] if 'batch_size' in models[model] else [0]):
              print(f"\n\nInferring config: {[model, platform, t_h, t_nt, c_nc, d_t, n]} ...")
              if [model, platform, '*', t_nt, '*', '*', n] in skipped_configs or \
                 [model, '*', '*', t_nt, '*', '*', n] in skipped_configs or \
                 [model, platform, t_h, '*', c_nc, d_t, n] in skipped_configs or \
                 [model, platform, t_h, t_nt, '*', d_t, n] in skipped_configs or \
                 [model, '*', t_h, t_nt, c_nc, d_t, n] in skipped_configs or \
                 [model, platform, t_h, t_nt, c_nc, d_t, n] in skipped_configs:
                print("Skipped")
                continue
              try:
                subprocess.run(["python3", "infer_model.py", model, "--platform", platform,
                                d_t, t_nt, c_nc, "--batch_size", str(n)], check=True)
              except:
                failed_configs.append([model, platform, t_h, t_nt, c_nc, d_t, n])

  return failed_configs


def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', action='append',
                      dest='models',
                      help='Compile which model(s)')
  parser.add_argument('--platform', action='append',
                      dest='platforms',
                      help='Compile models for which platforms (J7, J721S2, AM62A)')
  parser.add_argument('--tvm', action='store_true',
                      default=False,
                      help="Using tvm runtime (default is using dlr runtime)")
  parser.add_argument('--tidl', action='store_true',
                      default=False,
                      help="With TIDL offload")
  parser.add_argument('--notidl', action='store_true',
                      default=False,
                      help="Without TIDL offload")
  parser.add_argument('--c7x', action='store_true',
                      default=False,
                      help="With C7x code generation")
  parser.add_argument('--noc7x', action='store_true',
                      default=False,
                      help="Without C7x code generation")
  args = parser.parse_args()

  return args

if __name__ == "__main__":
  args = parse_args()
  if not args.models:
    args.models = [k for k, v in models.items()]
  if not args.platforms:
    args.platforms = ["J7"]
  dlr_tvm = ["--dlr"] if not args.tvm else ["--tvm"]
  tidls = []
  tidls += ["--tidl"] if args.tidl else []
  tidls += ["--notidl"] if args.notidl else []
  if not tidls:
    tidls = ["--tidl", "--notidl"]
  c7xs = []
  c7xs += ["--c7x"] if args.c7x else []
  c7xs += ["--noc7x"] if args.noc7x else []
  if not c7xs:
    c7xs = ["--c7x", "--noc7x"]

  failed_configs = test_infer(args.models, args.platforms, dlr_tvm, tidls, c7xs)
  if not failed_configs:
    print("Pass")
    sys.exit(0)
  else:
    print("Failed configs:")
    for config in failed_configs:
      print(f"  {config}")
    sys.exit(1)

