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
""" Run compilation tests for "models" """

import os
import sys
import subprocess
from models import models


skipped_configs = [
  # od postproc offload only available with tidl
  [ 'mv2_od_onnx', '*', '*', '--notidl', '*', 0 ],
  [ 'yolov5_od_onnx', '*', '*', '--notidl', '*', 0 ],
  [ 'lidar_od_onnx', '*', '*', '--notidl', '*', 0 ],
  # large feature map size in batch processing, unable to import to TIDL
  [ 'mv2_onnx', 'AM62A', '*', '--tidl', '*', 2 ],
  [ 'mv2_pth', 'AM62A', '*', '--tidl', '*', 2 ],
  # PSDK 9.1 RC4 TIDL calib failure: tidlMultiInstanceTest Invalid Error Type!
  [ 'swin_tiny_timm', '*', '*', '--tidl', '*', 0 ],
]


def test_compile(in_models, platforms, targets, tidls, c7xs):
  failed_configs = []
  for model in in_models:
    for platform in platforms:
      for t_h in targets:
        for t_nt in tidls:
          for c_nc in c7xs:
            # C7x codegen mode does not support host emulation yet
            if t_h == "--host" and c_nc == "--c7x":
              continue

            for n in (models[model]['batch_size'] if 'batch_size' in models[model] else [0]):
              print(f"\n\nCompiling config: {[model, platform, t_h, t_nt, c_nc, n]} ...")
              if [model, platform, '*', t_nt, '*', n] in skipped_configs or \
                 [model, '*', '*', t_nt, '*', n] in skipped_configs or \
                 [model, platform, t_h, t_nt, c_nc, n] in skipped_configs:
                print("Skipped")
                continue
              try:
                subprocess.run(["python3", "compile_model.py", model, "--platform", platform,
                                t_h, t_nt, c_nc, "--batch_size", str(n)], check=True)
              except:
                failed_configs.append([model, platform, t_h, t_nt, c_nc, n])

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
  parser.add_argument('--target', action='store_true',
                      default=False,
                      help="Compile for target")
  parser.add_argument('--host', action='store_true',
                      default=False,
                      help="Compile for host (emulation)")
  parser.add_argument('--tidl', action='store_true',
                      default=False,
                      help="Enable TIDL offload")
  parser.add_argument('--notidl', action='store_true',
                      default=False,
                      help="Disable TIDL offload")
  parser.add_argument('--c7x', action='store_true',
                      default=False,
                      help="Enable C7x code generation")
  parser.add_argument('--noc7x', action='store_true',
                      default=False,
                      help="Disable C7x code generation")
  args = parser.parse_args()

  return args


if __name__ == "__main__":
  args = parse_args()
  if not args.models:
    args.models = [k for k, v in models.items()]
  if not args.platforms:
    args.platforms = ["AM62A", "J7"]
  targets = []
  targets += ["--target"] if args.target else []
  targets += ["--host"] if args.host else []
  if not targets:
    targets = ["--target", "--host"]
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

  failed_configs = test_compile(args.models, args.platforms, targets, tidls, c7xs)
  if not failed_configs:
    print("Pass")
    sys.exit(0)
  else:
    print("Failed configs:")
    for config in failed_configs:
      print(f"  {config}")
    sys.exit(1)
