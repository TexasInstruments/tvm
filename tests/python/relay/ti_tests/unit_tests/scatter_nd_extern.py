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

from unit_utils import is_on_target, gen_reference, check_reference, platform, artifacts_folders, build_and_set_ext_lib

model_name = "scatter_nd_extern"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [("data", (2240, 64)),
                ("indices_i2240", (1, 58080, 1)),
                ("updates", (1, 58080, 64)) ]
weight_shapes = []

def add_c7x_tidl_scatter_nd_strategy():
  import tvm
  from tvm import relay
  from tvm import topi
  from tvm import te
  from tvm.relay.op import op as reg
  from tvm.relay.op.op import OpStrategy, OpPattern
  from tvm.tir import decl_buffer

  def compute_c7x_tidl_scatter_nd(attrs, inputs, out_type):
    data = inputs[0]
    indices = inputs[1]
    updates = inputs[2]

    out = tvm.te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: tvm.tir.call_packed("scatter_nd_ext",ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        name="tidl_scatter_nd_c7x",
        tag="tidl_scatter_nd_c7x",
    )
    return [out]

  def wrap_c7x_tidl_scatter_nd_schedule(topi_schedule):
    def wrapper(attrs, outs, target):
      with target:
        return topi_schedule(outs)
    return wrapper

  def tidl_scatter_nd_strategy_c7x(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        compute_c7x_tidl_scatter_nd,
        wrap_c7x_tidl_scatter_nd_schedule(topi.generic.schedule_extern),
        name="scatter_nd_c7x",
        plevel=15
    )
    return strategy

  reg.get("tidl_scatter_nd").get_attr("FTVMStrategy").register(tidl_scatter_nd_strategy_c7x,
                                                       "c7x", allow_override=True)
  reg.register_pattern("tidl_scatter_nd", OpPattern.OPAQUE, level=15)



def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay
  add_c7x_tidl_scatter_nd_strategy()
  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape,
                 "int32" if name.endswith("i2240") else "float32"))
                 for name, shape in input_shapes]


  # build and set external library
  src_name = "scatter_nd_extern"
  src_dir  = os.path.dirname(os.path.realpath(__file__))
  if not build_and_set_ext_lib(src_name, src_dir, artifacts_data_dir):
    return False

  # ind_trans = relay.transpose(input_vars[1], axes=[2, 0, 1])
  # output1 = relay.scatter_nd(input_vars[0], ind_trans, input_vars[2], mode="add")
  output2 = relay.tidl_scatter_nd(input_vars[0], input_vars[1], input_vars[2], mode="add")
  # output = relay.Tuple([output1, output2])
  func : relay.function.Function = relay.Function(input_vars, output2)
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
