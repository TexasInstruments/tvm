#!/usr/bin/env python3

""" Testing conversion of conv2d with 1x2 stride to conv2d with 1x1 stride -> max_pool2d with 1x2 stride
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence

model_name = "conv2d_1x2"
artifacts_dir = "artifacts_" + model_name

# Use a separate directory for data because compile_relay will delete the
# contents of artifacts_dir
artifacts_data_dir = artifacts_dir + '_data'

input_shapes = [("i0", (1, 8, 113 * 3, 341,)),]
weight_shapes = [("w1", (16, 8, 3, 3,)),]



def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes ]
  weight_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in weight_shapes ]
  output = relay.nn.conv2d(input_vars[0], weight_vars[0], strides=[1,2], padding=[1,1,1,1], kernel_size=[3,3])
  func : relay.function.Function = relay.Function(input_vars + weight_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # Compile relay module
  status = compile_relay(mod, weights, inputs, "J7",
                         compile_for_device=True, enable_tidl_offload=False, enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  relay_file = os.path.join(artifacts_dir, "tempDir/relay_graph.import.txt")
  num_maxpool = check_occurrence("max_pool2d", relay_file)

  if num_maxpool != 1:
    print(f"FAIL: num_maxpool {num_maxpool} != 1 (expected)")
    return False

  return True


def run_model():
  import sys
  sys.path.append("..")
  from infer_model import run_model

  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  tvm_outputs = run_model(artifacts_dir, inputs, use_dlr=True)

  return check_reference(tvm_outputs, artifacts_data_dir)


if __name__ == "__main__":
  if not os.path.exists(artifacts_data_dir):
    os.makedirs(artifacts_data_dir)

  if is_on_target():
    status = run_model()
  else:
    status = compile_model()

  sys.exit(0 if status else 1)
