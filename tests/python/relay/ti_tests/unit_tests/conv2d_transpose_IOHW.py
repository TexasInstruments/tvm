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
#  %2 = nn.conv2d_transpose(%1,
#  meta[relay.Constant][0] /* ty=Tensor[(128, 112, 4, 4), float32] */,
#  Tensor[(1, 128, 6, 16), float32],
#  Tensor[(128, 112, 4, 4), float32],
#  channels=112,
#  kernel_size=[4, 4],
#  strides=[2, 2]
#  padding=[1, 1, 1, 1],
#  kernel_layout="IOHW") /* ty=Tensor[(1, 112, 12, 32), float32] */;

model_name = "conv2d_transpose_IOHW"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [("i0", (1, 128, 6, 16)),]
weight_shapes = [("w1", (128, 112, 4, 4)),]


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
  output = relay.nn.conv2d_transpose(input_vars[0], weight_vars[0], strides=[2,2], padding=[1,1,1,1],
                       kernel_size=[4,4],kernel_layout="IOHW")
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
  num_tidl = check_occurrence("def @tidl", relay_file)
  if num_tidl != 1:
    print(f"FAIL: num_tidl {num_tidl} != 1 (expected)")
    return False

  return True


def run_model():
  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  from unit_utils import run_model_and_collect_trace
  tvm_outputs, trace = run_model_and_collect_trace(artifacts_dir, inputs)

  if not check_reference(tvm_outputs, artifacts_data_dir, maxdiff_ratio=0.03):
    return False

  tidl_conv2d_time    = trace['nodes'][0]['time']
  print(f"TIDL conv2d time: {tidl_conv2d_time} C7x cycles")
  threshold1 = 1000000 if platform != "AM62A" else 1100000
  if tidl_conv2d_time > threshold1:
    print(f"TIDL conv2d time exceeded expected threshold ({threshold1} cycles)")
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
