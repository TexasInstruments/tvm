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

""" Utils for unit testing """
"""
Model: 1) relay model, inputs shape, weights shape
       2) strategy (if overwriting)
Reference: 1) random inputs/weights, save to file
           2) graph executor, evaluate, save results
Compile: 1) x86 compile relay
Infer: 1) EVM infer, use reference inputs, compare with reference outputs
       2) if check perf, run with TVM_RT_DEBUG
"""

import os
import numpy as np
from platform import processor
from typing import List

def supported_platform(platform): 
    return platform in ["J7", "J721S2", "AM62A"]

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--platform', action='store',
                      dest='platform',
                      default='J7',
                      help='Compile models for which platforms (J7, J721S2, AM62A)')
  args = parser.parse_args()
  assert supported_platform(args.platform), f"Platform {args.platform} is not supported"
  return args

args = parse_args()
platform = args.platform

def artifacts_folders(model_name):
  artifacts_dir = f"artifacts_{model_name}_{platform}"

  # Use a separate directory for data because compile_relay will delete the
  # contents of artifacts_dir
  artifacts_data_dir = artifacts_dir + '_data'
  return artifacts_dir, artifacts_data_dir

def is_on_target():
  return processor() == "aarch64"


def get_data_file(artifacts_dir, var):
  return os.path.join(artifacts_dir, var + "_data.npy")

def gen_reference(mod, artifacts_dir:str, input_shapes : List, weight_shapes : List,
                  gen_new_data=True):
  """Generate reference inputs and outputs for a model

  Parameters
  ----------
  mod: tvm.relay.Module
  artifacts_dir : str
  inputs: List of (name, shape)
  weights: List of (name, shape)

  Return
  ------
  inputs, weights, outputs: map of name to data
  """

  inputs = {}
  weights = {}
  outputs = {}

  if gen_new_data:
    for var, shape in input_shapes + weight_shapes:
      if var.endswith("_s7"):
        data = np.arange(-7.0, 7.0, 14.0 / np.prod(shape), dtype=float).reshape(shape)
      elif var.endswith("_s"):
        data = np.random.randint(-128, 127, size=shape).astype('float32') / 128.0
      elif var.endswith("_unique"):
        num_elements = np.prod(shape)
        rng = np.random.default_rng()
        data = rng.choice(np.arange(num_elements * 2), size=shape, replace=False).astype('float32') / num_elements * 2.0
      elif var.endswith("_i"):
        num_elements = np.prod(shape)
        rng = np.random.default_rng()
        data = rng.choice(np.arange(num_elements * 2), size=shape, replace=False).astype('int32')
      elif var.endswith("_i2240"):
        num_elements = 2240
        rng = np.random.default_rng()
        data = rng.choice(np.arange(num_elements), size=shape, replace=True).astype('int32')
      elif var.endswith("_ind"):
        data = np.stack([np.stack([np.stack([np.arange(4) for _ in range(4)]) for _ in range(4)]) for _ in range(1)]).astype('int32')
      elif var.endswith("_tanh"):
        data = np.arange(-3.0, 3.0, 6.0 / np.prod(shape),dtype=float).reshape(shape)
      else:
        data = np.random.randint(0, 255, size=shape).astype('float32') / 256.0
      np.save(get_data_file(artifacts_dir, var), data)

  for var, _ in input_shapes:
    inputs[var] = np.load(get_data_file(artifacts_dir, var))
  for var, _ in weight_shapes:
    weights[var] = np.load(get_data_file(artifacts_dir, var))

  num_outputs = 1
  if gen_new_data:
    import tvm
    output = tvm.relay.create_executor(kind="graph", mod=mod).evaluate()(**inputs, **weights)
    if isinstance(output, List):
      for i, out in enumerate(output):
        if isinstance(out, List):
          out = out[0]
        np.save(get_data_file(artifacts_dir, f"ref_out{i}"), out.numpy())
      num_outputs = len(output)
    else:
      np.save(get_data_file(artifacts_dir, "ref_out0"), output.numpy())

  outputs["ref_out"] = [np.load(get_data_file(artifacts_dir, f"ref_out{i}")) for i in range(num_outputs)]

  return inputs, weights, outputs


def check_reference(tvm_outputs, artifacts_dir:str, maxdiff_threshold=None,
                    maxdiff_ratio=0.00001) -> bool:
  """Check tvm inference results agains reference
  """
  # Check results
  failed_outputs = []
  failure = False
  for i, out in enumerate(tvm_outputs):
    if isinstance(out, List):
      out = out[0]
    print("\nInfer result:", out.shape, out.min(), out.max())
    np.save(get_data_file(artifacts_dir, f"tvm_out{i}"), out)
    ref_out = np.load(get_data_file(artifacts_dir, f"ref_out{i}"))
    print("Golden(x86):", ref_out.shape, ref_out.min(), ref_out.max())
    diff = ref_out - out
    print("Diff:", diff.min(), diff.max(), np.argmax(diff))

    maxdiff = np.fmax(np.fabs(diff.min()), np.fabs(diff.max()))
    if maxdiff_threshold is None:
      maxval  = np.fmax(np.fabs(ref_out.min()), np.fabs(ref_out.max()))
      maxdiff_threshold = maxval * maxdiff_ratio
    failed_outputs.append(maxdiff >= maxdiff_threshold)
    failure = failure or maxdiff >= maxdiff_threshold
  if failure:
    for i, fail in enumerate(failed_outputs):
      if fail:
        print(f"FAIL: maxdiff exceeded allowed for output {i}\n")
  else:
    print("PASS\n")
    return True


def check_occurrence(pattern:str, text_file:str) -> int:
  import re
  count = 0
  with open(text_file, "r") as f:
    for line in f:
      if re.search(pattern, line):
        count += 1
  return count


def build_and_set_ext_lib(src_name, src_dir, build_dir):
  import subprocess
  build_dir = os.path.abspath(build_dir)
  silicon_version = "7504" if platform == "AM62A" else "7100"
  try:
    subprocess.run(["make", "-f", "Makefile.ext_lib", f"NAME={src_name}",
                    f"SRC_DIR={src_dir}", f"BUILD_DIR={build_dir}", f"SILICON_VERSION={silicon_version}"], check=True)
  except:
    print(f"Build external library for {src_name} failed")
    return False
  ext_libs = os.environ.get("CGT7X_EXT_LIBS", "")
  ext_libs += f" -l {build_dir}/{src_name}.lib"
  os.environ["CGT7X_EXT_LIBS"] = ext_libs
  return True

def run_model_and_collect_trace(artifacts_dir, inputs, use_dlr=True):
  import sys
  sys.path.append("..")
  from infer_model import run_model
  sys.path.append("../../../../../python/tvm/contrib/tidl")
  from dump_tvm_trace import read_trace

  cwd = os.getcwd()
  cdebug = os.environ.get("TVM_RT_DEBUG", None)
  real_artifacts_dir = os.path.realpath(artifacts_dir)
  os.chdir(real_artifacts_dir)
  os.environ["TVM_RT_DEBUG"] = "2"

  tvm_outputs = run_model(real_artifacts_dir, inputs, use_dlr)
  c7x_trace = read_trace("tvm_c7x.trace")

  os.chdir(cwd)
  if cdebug is None:
    os.environ.pop("TVM_RT_DEBUG")
  else:
    os.environ["TVM_RT_DEBUG"] = cdebug

  return tvm_outputs, c7x_trace

