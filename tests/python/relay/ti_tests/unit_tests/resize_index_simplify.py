#!/usr/bin/env python3

""" Testing c7x resize strategy that simplifies array index computation
    - half-pixel mode that rewrites index computation for better simplification
    - linear method that removes floorf() that prevents c7x software pipelining
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence
from unit_utils import build_and_set_ext_lib

#logging.basicConfig(level=logging.DEBUG)
#os.environ["TVM_LOG_DEBUG"] = "1"

model_name = "resize_index_simplify"
artifacts_dir = "artifacts_" + model_name
input_shapes = [ ("i0", (1, 21, 13, 13)) ]
weight_shapes = [ ("w0", (21, 21, 1, 1)) ]
artifacts_data_dir = artifacts_dir + '_data'


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes + weight_shapes ]
  t1 = relay.image.resize2d(input_vars[0], size=[33, 33], layout="NCHW")
  t2 = relay.nn.conv2d(t1, input_vars[1], kernel_size=[1,1], data_layout="NCHW",
                       kernel_layout="OIHW")
  output = relay.image.resize2d(t2, size=[257, 257], layout="NCHW",
                                coordinate_transformation_mode="align_corners")

  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # Compile relay module
  status = compile_relay(mod, weights, inputs, "J7",
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
  resize1_time = trace['nodes'][0]['time']
  resize2_time = trace['nodes'][2]['time']
  print(f"resize node 1 time: {resize1_time} C7x cycles")
  print(f"resize node 2 time: {resize2_time} C7x cycles")
  if resize1_time > 500000:
    print(f"resize node 1 time exceeded expected threshold (500,000 cycles)")
    return False
  if resize2_time > 8000000:
    print(f"resize node 2 time exceeded expected threshold (8,000,000 cycles)")
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
