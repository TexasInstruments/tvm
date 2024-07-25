#!/usr/bin/env python3

""" Testing topk operator calling external function in external library
    Note: this is only an example implementation to demonstrate the topk
          support via external library.  Details please see topk_1d.cpp.
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence
from unit_utils import build_and_set_ext_lib, platform, artifacts_folders

model_name = "argsort"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)
input_shapes = [("i0_unique", (4, 1028, 8, 8, 8)), ("i1_i", (4, 1028, 8, 8, 8))]
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
                 for name, shape in input_shapes[:1] ] + [relay.var(name, relay.TensorType(shape, "int32"))
                 for name, shape in input_shapes[1:] ]
  output1 = relay.argsort(input_vars[0], is_ascend=True, axis=1) 
  if isinstance(output1, relay.expr.TupleWrapper):
    output1 = output1.astuple()
  output2 = relay.argsort(input_vars[0], is_ascend=False, axis=-1) 
  if isinstance(output2, relay.expr.TupleWrapper):
    output2 = output2.astuple()
  output3 = relay.argsort(input_vars[1], is_ascend=False, axis=1) 
  if isinstance(output3, relay.expr.TupleWrapper):
    output3 = output3.astuple()
  output = relay.Tuple([output1, output2, output3])
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

  return True


def run_model():
  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  from unit_utils import run_model_and_collect_trace
  tvm_outputs, trace = run_model_and_collect_trace(artifacts_dir, inputs)

  if not check_reference(tvm_outputs, artifacts_data_dir):
    return False

  max_time = 1000000000 if platform not in ["AM62A", "J722S"] else 1500000000
  for i in range(3):
    argsort_time = trace['nodes'][i]['time']
    print(f"argsort node {i} time: {argsort_time} C7x cycles")
    if argsort_time > max_time:
      print(f"argsort node time exceeded expected threshold ({max_time} cycles)")
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
