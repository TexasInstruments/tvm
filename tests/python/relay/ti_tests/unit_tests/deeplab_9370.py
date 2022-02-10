#!/usr/bin/env python3

import os
import logging

import tvm
from tvm import relay
import tvm.contrib.c7x as c7x

logging.basicConfig(level=logging.DEBUG)
os.environ["TVM_LOG_DEBUG"] = "1"

# define graph in relay

'''
 %219 = fn (%p03: Tensor[(1, 256, 1, 1), float32], %p12: Tensor[(1, 256, 33, 33), float32], src_layout="NHWC", hash="4e5df5e5308a13be", dst_layout="NCHW", Primitive=1, layout="NHWC") -> Tensor[(1, 512, 33, 33), float32] {
    %5 = layout_transform(%p03, src_layout="NCHW", dst_layout="NHWC") /* ty=Tensor[(1, 1, 1, 256), float32] */;
    %6 = image.resize(%5, size=[33, 33], layout="NHWC", coordinate_transformation_mode="align_corners", rounding_method="") /* ty=Tensor[(1, 33, 33, 256), float32] */;
    %7 = layout_transform(%p12, src_layout="NCHW", dst_layout="NHWC") /* ty=Tensor[(1, 33, 33, 256), float32] */;
    %8 = (%6, %7);
    %9 = concatenate(%8, axis=3) /* ty=Tensor[(1, 33, 33, 512), float32] */;
    layout_transform(%9, src_layout="NHWC", dst_layout="NCHW") /* ty=Tensor[(1, 512, 33, 33), float32] */
  };
'''

p03 = relay.var("data", relay.TensorType((1, 256, 1, 1), "float32"))
p12 = relay.var("data", relay.TensorType((1, 256, 33, 33), "float32"))
data_5 = relay.layout_transform(p03, src_layout="NCHW", dst_layout="NHWC")
data_6 = relay.image.resize(data_5, size=[33, 33], layout="NHWC", coordinate_transformation_mode="align_corners", rounding_method="")
data_7 = relay.layout_transform(p12, src_layout="NCHW", dst_layout="NHWC")
data_9 = relay.concatenate([data_6, data_7], axis=3)
data_10 =  relay.layout_transform(data_9, src_layout="NHWC", dst_layout="NCHW")


func : relay.function.Function = relay.Function([p03, p12], data_10)

# create an IRModule containing relay function(s)
mod : tvm.ir.module.IRModule = tvm.IRModule.from_expr(func)

# build from relay
with c7x.c7x_target_config():
    _, lib, _ = relay.build(mod, target="c7x", target_host="c7x", params={})

for imod in lib.imported_modules:
    #print(imod.get_source())
    imod.save('./deeplab_9370.cpp', 'c')

