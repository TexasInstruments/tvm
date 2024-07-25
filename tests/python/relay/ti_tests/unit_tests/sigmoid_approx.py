#!/usr/bin/env python3

""" Testing sigmoid operator calling external function in external library
    Testing external sigmoid being Opaque, i.e. not to be fused with other ops
    Approximating sigmoid computation with pre-computed lut
        (details see gen_sigmoid_lut() and sigmoid_approx.cpp)
"""

import os
import sys
import logging
from typing import List
import numpy as np

from unit_utils import is_on_target, gen_reference, check_reference, check_occurrence, platform, artifacts_folders
from unit_utils import build_and_set_ext_lib

model_name = "sigmoid_approx"
artifacts_dir , artifacts_data_dir = artifacts_folders(model_name)

input_shapes = [ ("i0_s7", (1, 1, 1, 14 * 256)) ]
weight_shapes = []

sigmoid_precision = 0.002


# add c7x sigmoid strategy to call an extern function, "sigmoid_approx"
# TVM strategy for a relay operator consists of "compute" and "schedule".
#   Here we overwrite the default sigmoid "compute" (f(x) = 1/(1+exp(-x)))
#   to simply calling an external function on the whole tensor, so that
#   we can use a look up table and avoid the expensive exp() computation.
# Details on TVM strategy for relay operator can be found:
# - https://tvm.apache.org/docs/arch/relay_op_strategy.html
# - https://tvm.apache.org/docs/dev/how_to/relay_add_op.html
def add_c7x_sigmoid_strategy():
  import tvm
  from tvm import relay
  from tvm import topi
  from tvm import te
  from tvm.relay.op import op as reg
  from tvm.relay.op.op import OpStrategy, OpPattern

  def compute_c7x_sigmoid(attrs, inputs, out_type):
    A = inputs[0]
    out = tvm.te.extern(A.shape, [A],
        lambda ins, outs: tvm.tir.call_packed("sigmoid_approx", ins[0], outs[0]),
        name="sigmoid_c7x"
    )
    return [out]

  def wrap_c7x_sigmoid_schedule(topi_schedule):
    def wrapper(attrs, outs, target):
      with target:
        return topi_schedule(outs)
    return wrapper

  def sigmoid_strategy_c7x(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        compute_c7x_sigmoid,
        wrap_c7x_sigmoid_schedule(topi.generic.schedule_extern),
        name="sigmoid_c7x",
        plevel=15
    )
    return strategy

  reg.get("sigmoid").get_attr("FTVMStrategy").register(sigmoid_strategy_c7x,
                                                       "c7x", allow_override=True)
  reg.register_pattern("sigmoid", OpPattern.OPAQUE, level=15)


# generate sigmoid lookup table of 801 entries with precision 0.002
def gen_sigmoid_lut(src_name, build_dir):
  cutoff = 6.25        # sigmoid(x) = 1.0 for x > cutoff
  steps_in_one = 128
  a = np.arange(0, cutoff, 1.0/steps_in_one, dtype=float)
  b = 1 / (1 + np.exp(-a))  # sigmoid definition
  # 0.002 precision shown as follows
  # >>> c = b[1:] - b[:-1]
  # >>> np.max(c)
  # 0.0019531150659531926
  # >>> 1.0 - np.max(b)
  # 0.0019418168866865981
  with open(os.path.join(build_dir, src_name+".h"), "w") as f:
    f.write("const float sigmoid_lut_cutoff = 6.25f;\n")
    f.write("const int sigmoid_lut_steps_in_one = 128;\n")
    f.write(f"const int sigmoid_lut_len = {len(b) + 1};\n")
    f.write("const float sigmoid_lut[] = {\n")
    for value in b:
      f.write(f"  {value},\n")
    f.write("  1.0f };\n");


def compile_model():
  """Create a relay model, generate reference inputs/outputs, compile it"""
  import tvm
  from tvm import relay
  from tvm.contrib.tidl.compile import compile_relay

  add_c7x_sigmoid_strategy()

  # define graph/model in relay
  input_vars = [ relay.var(name, relay.TensorType(shape, "float32"))
                 for name, shape in input_shapes ]
  t0 = relay.const(0.01, "float32")
  t1 = relay.add(input_vars[0], t0)
  output = relay.sigmoid(t1)
  func : relay.function.Function = relay.Function(input_vars, output)
  mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

  # gen reference inputs/outputs
  gen_new_data = os.environ.get("TIDL_REBUILD_ONLY", None) is None
  inputs, weights, _ = gen_reference(mod, artifacts_data_dir, input_shapes, weight_shapes,
                                     gen_new_data=gen_new_data)

  # build and set external library
  src_name = "sigmoid_approx"
  src_dir  = os.path.dirname(os.path.realpath(__file__))
  gen_sigmoid_lut(src_name, artifacts_data_dir)
  if not build_and_set_ext_lib(src_name, src_dir, artifacts_data_dir):
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
  inputs, weights, output = gen_reference(None, artifacts_data_dir, input_shapes, weight_shapes,
                                          gen_new_data=False)

  from unit_utils import run_model_and_collect_trace
  tvm_outputs, trace = run_model_and_collect_trace(artifacts_dir, inputs)

  if not check_reference(tvm_outputs, artifacts_data_dir, sigmoid_precision):
    return False

  sigmoid_time = trace['nodes'][1]['time']
  print(f"sigmoid node time: {sigmoid_time} C7x cycles")
  threshold = 25000 if platform not in ["AM62A", "J722S"] else 250000
  if sigmoid_time > threshold:
    print(f"sigmoid node time exceeded expected threshold ({threshold} cycles)")
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
