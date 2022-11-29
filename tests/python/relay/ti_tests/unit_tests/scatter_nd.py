#!/usr/bin/env python3

""" Testing correctness of scatter_nd operation.
    Considers cases where rank of len(updates.shape) + 1 == len(indices.shape) and len(updates.shape) + 1 > len(indices.shape)
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence

model_name = "scatter_nd"
artifacts_dir = "artifacts_" + model_name
artifacts_dir2 = "artifacts_" + model_name + "_2"

# Use a separate directory for data because compile_relay will delete the
# contents of artifacts_dir
artifacts_data_dir = artifacts_dir + '_data'
artifacts_data_dir2 = artifacts_dir2 + '_data'

input_shapes = [("data", (64, 1, 1, 32, 32)), ("updates", (64, 1017))]
weight_shapes = []
input_shapes2 = [("data2", (64, 1, 1, 32, 32)), ("updates2", (64, 1017, 32))]


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes]
  input_vars2 = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes2]
  indices_np = np.stack([
          np.stack([np.repeat(i, 1017) for i in range(64)]),
          np.stack([np.repeat(0, 1017) for _ in range(64)]),
          np.stack([np.repeat(0, 1017) for _ in range(64)]),
          np.stack([np.repeat(np.arange(32), 32)[:1017] for _ in range(64)]),
          np.stack([np.tile(np.arange(32), 32)[:1017] for _ in range(64)]),
    ])
  indices_np2 = np.stack([
          np.stack([np.repeat(i, 1017) for i in range(64)]),
          np.stack([np.repeat(0, 1017) for _ in range(64)]),
          np.stack([np.repeat(0, 1017) for _ in range(64)]),
          np.stack([np.repeat(np.arange(32), 32)[:1017] for _ in range(64)]),
    ])
  indices = relay.const(indices_np, "int64")
  indices2 = relay.const(indices_np2, "int64")
  output = relay.scatter_nd(input_vars[0], indices, input_vars[1])
  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)
  output2 = relay.scatter_nd(input_vars2[0], indices2, input_vars2[1])
  func2 : relay.function.Function = relay.Function(input_vars2, output2)
  mod2 : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func2)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
 
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  inputs2, weights2, _ = gen_reference(mod2, artifacts_data_dir2, input_shapes2, weight_shapes,
                                     gen_new_data=gen_new_data)

  # Compile relay module
  status = compile_relay(mod, weights, inputs, "J7",
                         compile_for_device=True, enable_tidl_offload=True, enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  status = compile_relay(mod2, weights2, inputs2, "J7",
                         compile_for_device=True, enable_tidl_offload=True, enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir2, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  relay_file = os.path.join(artifacts_dir, "tempDir/relay_graph.import.txt")

  return True


def run_model():
  import sys
  sys.path.append("..")
  from infer_model import run_model

  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)
  inputs2, weights2, output2 = gen_reference(None, artifacts_data_dir2, input_shapes2, weight_shapes,
                                          gen_new_data=False)
  
  os.environ["TVM_RT_DEBUG"] = "2"
  tvm_outputs = run_model(artifacts_dir, inputs, use_dlr=True)

  if not check_reference(tvm_outputs, artifacts_data_dir, maxdiff_ratio=0.03):
    return False

  tvm_outputs = run_model(artifacts_dir2, inputs2, use_dlr=True)

  if not check_reference(tvm_outputs, artifacts_data_dir2, maxdiff_ratio=0.03):
    return False


  return True


if __name__ == "__main__":
  if not os.path.exists(artifacts_data_dir):
    os.makedirs(artifacts_data_dir)
  if not os.path.exists(artifacts_data_dir2):
    os.makedirs(artifacts_data_dir2)

  if is_on_target():
    status = run_model()
  else:
    status = compile_model()

  sys.exit(0 if status else 1)
