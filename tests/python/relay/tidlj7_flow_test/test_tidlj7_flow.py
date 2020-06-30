# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import subprocess
from ctypes import *

#import onnx
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib import cc
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from tvm.relay.backend.contrib import tidl
import tensorflow as tf
from tvm.relay.testing import tf as tf_testing

from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform


def create_avg_pool2d_relay():
    infeats = relay.var("infeats", shape=[1,2,2,3], dtype='float32')
    avg_pool2d = relay.nn.avg_pool2d(infeats, pool_size=[2,2], layout='NHWC')
    function = relay.Function(relay.analysis.free_vars(avg_pool2d), avg_pool2d)
    module = tvm.ir.IRModule()
    module["main"] = function

    return module


def compile_module(mod):
#    Uncomment to get logging information on the strategy selected for compilation
#    logging.getLogger("compile_engine").setLevel(logging.INFO)
#    logging.getLogger("compile_engine").addHandler(logging.StreamHandler(sys.stdout))

    mod = relay.transform.RemoveUnusedFunctions()(mod)
    # Bind params so that weights will appear as constants instead of variables
#    print("----------Pre Bind-------------")
#    print(mod.astext(show_meta_data=False))
#    mod['main'] = bind_params_by_name(mod['main'], params)
#    print("----------Post Bind-------------")
#    print(mod.astext(show_meta_data=False))

    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.ConvertLayout('NCHW')(mod)

    #============= Annotate the graph ==============
    # Looks at annotated ops and marks them in the graph with compiler.begin
    # and compiler.end.
    # Merges annotated regions together that use the same external target,
    # and combines marked regions for each target
    #Merge sequence of ops into composite functions/ops
    mod = tidl.tidl._merge_sequential_ops(mod)
    mod = transform.AnnotateTarget("tidl")(mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    mod = tidl.UnpackComposites(mod, "tidl")

    # Create dummy files for testing
    ctx = tidl.TIDLContext.current()
    artifacts_directory = ctx.artifacts_directory
    with open(os.path.join(artifacts_directory, 'tidl_subgraph' + str(0) + '_net.bin'), 'w') as f:
        f.write("Network data")

    with open(os.path.join(artifacts_directory, 'tidl_subgraph' + str(0) + '_params.bin'), 'w') as f:
        f.write("Params Data")


    graph, lib, params = relay.build_module.build(mod, target='llvm')
    lib.export_library(os.path.join(artifacts_directory, 'deploy_lib.so'))
    return graph, params

def run_module(graph, params):
    data_shape = (1, 2, 2, 3)
    input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
    loaded_lib = tvm.runtime.load_module("tidlj7_artifacts/deploy_lib.so")
    graph_mod = graph_runtime.create(graph, loaded_lib, tvm.cpu())
    graph_mod.run(infeats=input_data)
    output = graph_mod.get_output(0).asnumpy()
    return output;

def compile_tidlrt(script_dir, artifacts_path):

    # We might want to get this path from cmake
    tidlrt_path="/Users/a0866938/work/repos/vision_apps/applibs/tidl_rt/inc"

    subprocess.run(["g++", "-std=c++14", "-fPIC", os.path.join(script_dir, './tidl_j7_test_api.cpp'), '-I', tidlrt_path, '-c', '-o', os.path.join(artifacts_path, 'tidl_j7_test_api.o')], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).check_returncode()

    subprocess.run(["g++", "-shared", "-o", os.path.join(artifacts_path, 'libtidlj7_api.so'), os.path.join(artifacts_path, 'tidl_j7_test_api.o')], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).check_returncode()

    return cdll.LoadLibrary("libtidlj7_api.so")


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.realpath(__file__))

    tidlrt = compile_tidlrt(script_dir, 'tidlj7_artifacts')

    mod = create_avg_pool2d_relay()

    with tidl.build_config(artifacts_folder='tidlj7_artifacts', platform='j7'):
        graph, params = compile_module(mod)

    output = run_module(graph, params)

    golden = np.ndarray(shape=(1,1,1,3), dtype=float, buffer=np.array([20.0, 30.0, 40.0]))
    if not np.array_equal(golden, output):
        print("Incorrect output")
        sys.exit(1)

    if tidlrt.TIDLRT_create_called() != 1:
        print("TIDLRT_create not called", file=sys.stderr)
        sys.exit(1)

    if tidlrt.TIDLRT_invoke_called() != 1:
        print("TIDLRT_invoke not called", file=sys.stderr)
        sys.exit(1)

    # Deactivate should only be called if more than 1 subgraph exist
    if tidlrt.TIDLRT_deactivate_called() != 0:
        print("TIDLRT_deactivate was called", file=sys.stderr)
        sys.exit(1)

    if tidlrt.TIDLRT_delete_called() != 1:
        print("TIDLRT_delete not called", file=sys.stderr)
        sys.exit(1)

    if tidlrt.TIDLRT_setParamsDefault_called() == 0:
        print("TIDLRT_setParamsDefault not called", file=sys.stderr)
        sys.exit(1)

    if tidlrt.TIDLRT_setTensorDefault_called() == 0:
        print("TIDLRT_setTensorDefault not called", file=sys.stderr)
        sys.exit(1)
