#!/usr/bin/env python3

""" Testing performance of tidl.scatter_nd operation with pre-transposed indices.
    When TVM imports scatterND op from other formats (e.g. ONNX), it transposes
    the indices and use in Relay's scatter_nd op.  TIDL only handles un-transposed
    indices.  Convert Relay scatter_nd op to a custom op, tidl.scatter_nd, that uses
    pre-transposed indices from the original external format.
    e.g. data 2240x64, indices 1x58080x1, updates 1x58080x64, reduction "add"
         each update is a slice of (64)

    TODO: test TIDL offload, however, TIDL has restrictions:
    - data 4D, indicies 1x1xPx4, updates 1x1xPx1
    - replace only, no reduction (e.g. add) support
    - backport scatter_nd reduction support from upstream TVM: commit 266ff51d2a
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, platform, artifacts_folders, check_occurrence

model_name = "scatter_nd_tidl"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [("data", (2240, 65)),
                ("indices_i2240", (1, 58080, 1)),
                ("updates", (1, 58080, 65)) ]
weight_shapes = []


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape,
                 "int32" if name.endswith("i2240") else "float32"))
                 for name, shape in input_shapes]

  ind_trans = relay.transpose(input_vars[1], axes=[2, 0, 1])
  output1 = relay.scatter_nd(input_vars[0], ind_trans, input_vars[2], mode="add")
  #output1 = relay.tidl_scatter_nd(input_vars[0], input_vars[1], input_vars[2], mode="add")
  func : relay.function.Function = relay.Function(input_vars, output1)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
 
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # Compile relay module
  status = compile_relay(mod, weights, inputs, platform,
                         compile_for_device=True, enable_tidl_offload=True, enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  relay_file = os.path.join(artifacts_dir, "tempDir/relay_graph.import.txt")
  num_tidl_scatter_nd = check_occurrence("tidl_scatter_nd", relay_file)
  if num_tidl_scatter_nd != 1:
    print(f"FAIL: num_tidl_scatter_nd {num_tidl_scatter_nd} != 1 (expected)")
    return False

  return True


def run_model():
  import sys
  sys.path.append("..")
  from infer_model import run_model

  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)
  
  tvm_outputs = run_model(artifacts_dir, inputs, use_dlr=True)

  if not check_reference(tvm_outputs, artifacts_data_dir, maxdiff_ratio=0.03):
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
