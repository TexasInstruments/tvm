#!/usr/bin/env python3

""" Testing creating multiple models at inference time using priorities.
    - The model is from conv2d_1x2_stride.
    - Compile the model using 2 different configs: 1) Arm + TIDL, 2) C7x + TIDL
    - At inference time, create models using 4 different priorities (each config twice),
        run them one by one multiple times.

    - Limitation:
     c7x_codegen=1 models have to run before any c7x_codegen=0 models, otherwise, we run into
        this error at runtime:
            [C7x_1 ] 382546.643037 s: TIDL_initDmaUtils returned Error Code for handle: 9072400
     This is possibily caused by ivision activation optimization in OpenVX TIDL node. TIDL is
        only deactivated (when DMA resources are released) when a different ivision/TIDL handle
        (e.g. priority) is encountered.  OpenVX TVM node does not use ivision protocol.
        For now, document the limitation.
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence, platform, artifacts_folders

model_name = "priorities"
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

  # Compile relay module with 2 different configs
  mod_1_path = artifacts_dir+"_arm"
  mod_2_path = artifacts_dir+"_c7x"
  status1 = compile_relay(mod, weights, inputs, platform,
                       compile_for_device=True, enable_tidl_offload=True, enable_c7x_codegen=False,
                       artifacts_folder=mod_1_path, tidl_tensor_bits=8)
  status2 = compile_relay(mod, weights, inputs, platform,
                       compile_for_device=True, enable_tidl_offload=True, enable_c7x_codegen=True,
                       artifacts_folder=mod_2_path, tidl_tensor_bits=8)
  if status1 != 1 or status2 != 1:
    print("TIDL compilation failed")
    return False

  relay_file = os.path.join(mod_1_path, "tempDir/relay_graph.import.txt")
  num_maxpool = check_occurrence("max_pool2d", relay_file)
  relay_file = os.path.join(mod_2_path, "tempDir/relay_graph.import.txt")
  num_maxpool2 = check_occurrence("max_pool2d", relay_file)

  if num_maxpool != 2 or num_maxpool2 != 2:
    print(f"FAIL: num_maxpool {num_maxpool} != 2 (expected)")
    return False

  return True


def run_model():
  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  from dlr import DLRModel
  mod_1_path = artifacts_dir+"_arm"
  mod_2_path = artifacts_dir+"_c7x"
  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  print("\nCreating model 1")
  os.environ["TIDL_RT_TARGET_PRIORITY"] = "1"
  mod_1 = DLRModel(mod_1_path)
  print("\nCreating model 2")
  os.environ["TIDL_RT_TARGET_PRIORITY"] = "2"
  mod_2 = DLRModel(mod_2_path)
  print("\nCreating model 3")
  os.environ["TIDL_RT_TARGET_PRIORITY"] = "3"
  mod_3 = DLRModel(mod_1_path)
  print("\nCreating model 4")
  os.environ["TIDL_RT_TARGET_PRIORITY"] = "4"
  mod_4 = DLRModel(mod_2_path)

  print("\nRunning model 2")
  results2 = mod_2.run(inputs)
  print("\nRunning model 4")
  results4 = mod_4.run(inputs)
  print("\nRunning model 2 again")
  results2_2 = mod_2.run(inputs)
  print("\nRunning model 1")
  results1 = mod_1.run(inputs)
  print("\nRunning model 3")
  results3 = mod_3.run(inputs)
  print("\nRunning model 1 again")
  results1_2 = mod_1.run(inputs)

  if not check_reference(results1, artifacts_data_dir, maxdiff_ratio=0.03):
    return False
  if not check_reference(results2, artifacts_data_dir, maxdiff_ratio=0.03):
    return False
  if not check_reference(results3, artifacts_data_dir, maxdiff_ratio=0.03):
    return False
  if not check_reference(results4, artifacts_data_dir, maxdiff_ratio=0.03):
    return False
  if not check_reference(results1_2, artifacts_data_dir, maxdiff_ratio=0.03):
    return False
  if not check_reference(results2_2, artifacts_data_dir, maxdiff_ratio=0.03):
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
