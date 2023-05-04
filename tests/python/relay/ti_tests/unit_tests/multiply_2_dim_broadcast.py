#!/usr/bin/env python3

""" Tests 2 dimensional broadcast multiply. Ensure that dma pass
    can understand which dma calls are copy in vs out.
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence
from unit_utils import build_and_set_ext_lib, platform, artifacts_folders

model_name = "multiply_2_dim_broadcast"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)
input_shapes = [("i0", (1, 11, 256,)), ("i1", (64, 1, 256,))]
weight_shapes = []

# Use a separate directory for data because compile_relay will delete the
# contents of artifacts_dir


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
                         compile_for_device=True, enable_tidl_offload=False,
                         enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  c_file = os.path.join(artifacts_dir, "tempDir/model_1.c")
  num_DMAs = check_occurrence("create_DMA", c_file)
  num_SEs  = check_occurrence("SE.ADV\\(float8\\)" if platform == "AM62A" else "SE.ADV\\(float16\\)", c_file)
  if num_DMAs < 3 or num_SEs < 1:
    print(f"FAIL: num_DMAs {num_DMAs} < 3 (expected), {num_SEs} < 1 (expected)")
    return False

  return True


def run_model():
  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  from unit_utils import run_model_and_collect_trace
  tvm_outputs, trace = run_model_and_collect_trace(artifacts_dir, inputs)

  if not check_reference(tvm_outputs, artifacts_data_dir):
    return False



  return True


if __name__ == "__main__":
  if not os.path.exists(artifacts_data_dir):
    os.makedirs(artifacts_data_dir)

  if is_on_target():
    status = run_model()
  else:
    status = compile_model()

  sys.exit(0 if status else 1)
