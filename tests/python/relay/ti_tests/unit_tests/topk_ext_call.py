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

model_name = "topk_ext"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)
input_shapes = [ ("i0", (1, 1, 1, 4096)) ]
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
  #output = relay.topk(input_vars[0], k=32, ret_type="indices")
  #output = relay.topk(input_vars[0], k=32, ret_type="values")
  output = relay.topk(input_vars[0], k=32)  # default ret_type is "both"
  if isinstance(output, relay.expr.TupleWrapper):
    output = output.astuple()
  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # build and set external library
  src_dir = os.path.dirname(os.path.realpath(__file__))
  if not build_and_set_ext_lib("topk_1d", src_dir, artifacts_data_dir):
    return False

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
  import sys
  sys.path.append("..")
  from infer_model import run_model

  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  os.environ["TVM_RT_DEBUG"] = "2"
  tvm_outputs = run_model(artifacts_dir, inputs, use_dlr=True)

  if not check_reference(tvm_outputs, artifacts_data_dir):
    return False

  sys.path.append("../../../../../python/tvm/contrib/tidl")
  from dump_tvm_trace import read_trace
  trace = read_trace("tvm_c7x.trace")
  topk_time = trace['nodes'][0]['time']
  print(f"topk node time: {topk_time} C7x cycles")
  if topk_time > 100000:
    print(f"topk node time exceeded expected threshold (100,000 cycles)")
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
