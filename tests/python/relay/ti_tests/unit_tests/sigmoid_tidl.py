#!/usr/bin/env python3

""" Testing offloading sigmoid to TIDL
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence, platform, artifacts_folders

model_name = "sigmoid_tidl"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [ ("i0_s7", (1, 1, 1, 512)) ]
weight_shapes = [ ]


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.relay.op import op as reg
  from tvm.contrib.tidl.compile import compile_relay

  # During TVM/TIDL partitioning, subgraphs with 0 MACs will be pruned and
  # not offloaded to TIDL.  TVM default for sigmoid's MAC is 0.  Set sigmoid's MAC
  # to be 100 (randomly chosen) so that the sigmoid layer can be offloaded to TIDL.
  def compute_sigmoid_macs(call):
    return 100
  reg.get("sigmoid").set_attr("FMacCount", compute_sigmoid_macs)

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes ]
  output = relay.sigmoid(input_vars[0])
  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # Compile relay module
  status = compile_relay(mod, weights, inputs, platform,
                         compile_for_device=True, enable_tidl_offload=True,
                         enable_c7x_codegen=True,
                         artifacts_folder=artifacts_dir, tidl_tensor_bits=8)
  if status != 1:
    print("TIDL compilation failed")
    return False

  relay_file = os.path.join(artifacts_dir, "tempDir/relay_graph.import.txt")
  num_tidl_subgraphs = check_occurrence("tidl_0", relay_file)
  if num_tidl_subgraphs < 1:
    print(f"FAIL: sigmoid not offloaded to TIDL")
    return False

  return True


def run_model():
  import sys
  sys.path.append("..")
  from infer_model import run_model

  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  tvm_outputs = run_model(artifacts_dir, inputs, use_dlr=True)

  return check_reference(tvm_outputs, artifacts_data_dir, maxdiff_threshold=0.025)


if __name__ == "__main__":
  if not os.path.exists(artifacts_data_dir):
    os.makedirs(artifacts_data_dir)

  if is_on_target():
    status = run_model()
  else:
    status = compile_model()

  sys.exit(0 if status else 1)
