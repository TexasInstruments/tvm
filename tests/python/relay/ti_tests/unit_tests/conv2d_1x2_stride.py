#!/usr/bin/env python3

""" Testing conversion of conv2d with 1x2 stride to conv2d with 1x1 stride -> max_pool2d with 1x2 stride
    - conv2d with 1x1 stride can be offloaded to TIDL
    - max_pool2d with 1x1 pool_size and 1x2 stride can be optimized by C7x code generation,
      testing optimized c7x max_pool2d strategy for 1x1 pool_size
    - add another max_pool2d layer with different pool_size,
      testing to make sure generic C7x max_pool2d also works
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence, platform, artifacts_folders

model_name = "conv2d_1x2"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [("i0", (1, 8, 224, 224)),]
weight_shapes = [("w1", (16, 8, 3, 3)),]


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
  t1 = relay.nn.conv2d(input_vars[0], weight_vars[0], strides=[1,2], padding=[1,1,1,1],
                       kernel_size=[3,3])
  output = relay.nn.max_pool2d(t1, pool_size=[2, 2], strides=[1,2])
  func : relay.function.Function = relay.Function(input_vars + weight_vars, output)
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
  num_maxpool = check_occurrence("max_pool2d", relay_file)

  if num_maxpool != 2:
    print(f"FAIL: num_maxpool {num_maxpool} != 1 (expected)")
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

  if not check_reference(tvm_outputs, artifacts_data_dir, maxdiff_ratio=0.03):
    return False

  sys.path.append("../../../../../python/tvm/contrib/tidl")
  from dump_tvm_trace import read_trace
  trace = read_trace("tvm_c7x.trace")
  tidl_conv2d_time    = trace['nodes'][0]['time']
  c7x_max_pool2d_time = trace['nodes'][1]['time']
  print(f"TIDL conv2d time: {tidl_conv2d_time} C7x cycles")
  print(f"C7x max_pol2d time: {c7x_max_pool2d_time} C7x cycles")
  if tidl_conv2d_time > 1000000:
    print(f"TIDL conv2d time exceeded expected threshold (1,000,000 cycles)")
    return False
  if c7x_max_pool2d_time > 1000000:
    print(f"C7x max_pool2d time exceeded expected threshold (1,000,000 cycles)")
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
