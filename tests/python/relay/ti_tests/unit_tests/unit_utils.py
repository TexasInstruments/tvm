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
      else:
        data = np.random.randint(0, 255, size=shape).astype('float32') / 256.0
      np.save(get_data_file(artifacts_dir, var), data)

  for var, _ in input_shapes:
    inputs[var] = np.load(get_data_file(artifacts_dir, var))
  for var, _ in weight_shapes:
    weights[var] = np.load(get_data_file(artifacts_dir, var))

  if gen_new_data:
    import tvm
    output = tvm.relay.create_executor(kind="graph", mod=mod).evaluate()(**inputs, **weights)
    if isinstance(output, List):
      output = output[0]
    np.save(get_data_file(artifacts_dir, "ref_out"), output.numpy())

  outputs["ref_out"] = np.load(get_data_file(artifacts_dir, "ref_out"))

  return inputs, weights, outputs


def check_reference(tvm_outputs, artifacts_dir:str, maxdiff_threshold=None,
                    maxdiff_ratio=0.00001) -> bool:
  """Check tvm inference results agains reference
  """
  # Check results
  print("\nInfer result:", tvm_outputs[0].shape, tvm_outputs[0].min(), tvm_outputs[0].max())
  np.save(get_data_file(artifacts_dir, "tvm_out"), tvm_outputs[0])
  ref_out = np.load(get_data_file(artifacts_dir, "ref_out"))
  print("Golden(x86):", ref_out.shape, ref_out.min(), ref_out.max())
  diff = ref_out - tvm_outputs[0]
  print("Diff:", diff.min(), diff.max(), np.argmax(diff))

  maxdiff = np.fmax(np.fabs(diff.min()), np.fabs(diff.max()))
  if maxdiff_threshold is None:
    maxval  = np.fmax(np.fabs(ref_out.min()), np.fabs(ref_out.max()))
    maxdiff_threshold = maxval * maxdiff_ratio
  if (maxdiff >= maxdiff_threshold):
    print("FAIL: maxdiff exceeded allowed threshold\n")
    return False
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
  try:
    subprocess.run(["make", "-f", "Makefile.ext_lib", f"NAME={src_name}",
                    f"SRC_DIR={src_dir}", f"BUILD_DIR={build_dir}"], check=True)
  except:
    print(f"Build external library for {src_name} failed")
    return False
  ext_libs = os.environ.get("CGT7X_EXT_LIBS", "")
  ext_libs += f" -l {build_dir}/{src_name}.lib"
  os.environ["CGT7X_EXT_LIBS"] = ext_libs
  return True

