#!/usr/bin/env python3

""" Testing broadcasting multiply operator
    Testing c7x specific scheduling (injective.py)
    Testing odd number of dma blocks and odd number of iterations in the block
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence, platform, artifacts_folders

model_name = "bmul_odd"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)
input_shapes = [ ("i0", (1, 671, 49, 49)), ("i1", (1, 671, 1, 1)) ]
weight_shapes = []


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes ]
  output = relay.multiply(input_vars[0], input_vars[1])
  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # Compile relay module
  status = compile_relay(mod, weights, inputs, platform,
                         compile_for_device=True, enable_tidl_offload=False, enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  c_file = os.path.join(artifacts_dir, "tempDir/model_1.c")
  num_DMAs = check_occurrence("create_DMA", c_file)
  num_SEs  = check_occurrence("SE0ADV\\(float8\\)" if platform in ["AM62A", "J722S"] else
                              "SE0ADV\\(float16\\)", c_file)
  if num_DMAs < 2 or num_SEs < 1:
    print(f"FAIL: num_DMAs {num_DMAs} < 2 (expected), {num_SEs} < 1 (expected)")
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
