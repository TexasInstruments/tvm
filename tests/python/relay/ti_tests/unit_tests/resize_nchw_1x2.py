#!/usr/bin/env python3

""" Testing resize strategy for a special nchw 1x2 case
    - specialize the strategy to call an external function
    - all other cases uses the default TVM strategy
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

model_name = "resize_nchw_1x2"
artifacts_dir = "artifacts_" + model_name
input_shapes = [ ("i0", (1, 8, 64, 800)) ]
weight_shapes = []
artifacts_data_dir = artifacts_dir + '_data'


# Customize c7x resize strategy to call an extern function, "resize_1_2_nchnw"
#   for special case: nchw, half_pixel, linear, scale_h=1, scale_w=2.
# Otherwise, still use the C7x default resize strategy.
# Note that the example cpp implementation here only handles the float tensor.
#   It is not meant to be a generic implementation.
def add_c7x_resize_strategy():
  import tvm
  from tvm import relay
  from tvm import topi
  from tvm import te
  from tvm.relay.op import op as reg
  from tvm.relay.op import strategy as _strategy
  from tvm.relay.op.op import OpStrategy, OpPattern

  def compute_c7x_resize(attrs, inputs, out_type):
    A = inputs[0]
    out = tvm.te.extern(A.shape, [A],
        lambda ins, outs: tvm.tir.call_packed("resize_nchw_1x2", ins[0], outs[0]),
        name="resize_c7x"
    )
    return [out]

  def resize_strategy_c7x(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    if attrs['layout'] == "NCHW" and attrs['method'] == "linear" and \
       attrs['coordinate_transformation_mode'] == "half_pixel" and \
       inputs[0].shape[2] == attrs['size'][0] and inputs[0].shape[3] * 2 == attrs['size'][1]:
      strategy.add_implementation(
        compute_c7x_resize,
        _strategy.wrap_topi_schedule(topi.generic.schedule_extern),
        name="resize_c7x",
        plevel=15
      )
    else:
      strategy = _strategy.c7x.resize2d_strategy(attrs, inputs, out_type, target)
    return strategy

  reg.get("image.resize2d").get_attr("FTVMStrategy").register(resize_strategy_c7x,
                                                       "c7x", allow_override=True)
  reg.register_pattern("image.resize2d", OpPattern.OPAQUE, level=15)


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  add_c7x_resize_strategy()

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes ]
  output = relay.image.resize2d(input_vars[0], size=[64, 1600], layout="NCHW")
  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # build and set external library
  src_name = "resize_nchw_1x2"
  src_dir  = os.path.dirname(os.path.realpath(__file__))
  if not build_and_set_ext_lib(src_name, src_dir, artifacts_data_dir):
    return False

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
  resize_time = trace['nodes'][0]['time']
  print(f"resize node time: {resize_time} C7x cycles")
  if resize_time > 1000000:
    print(f"resize node time exceeded expected threshold (1,000,000 cycles)")
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
