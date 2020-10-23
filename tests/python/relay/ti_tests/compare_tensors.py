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
"""TIDL tensor comparison"""

import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
    description="Comparing TIDL layer output from quantized execution to corresponding TVM layer output from floating point execution",
    epilog="Example: TIDL_RELAY_DEBUG_IMPORT=4 python3 ./test_tidl_j7.py --target; python3 ./compare_tensors.py artifacts_MobileNetV1_target 0 1")
parser.add_argument('artifacts_dir')
parser.add_argument('subgraph_id', type=int)
parser.add_argument('layer_id', type=int)
args = parser.parse_args()


if __name__ == '__main__':

    print(f"{args.artifacts_dir} {args.subgraph_id} {args.layer_id}")

    tempDir = os.path.join(args.artifacts_dir, "tempDir")
    tvm_tensor_file = os.path.join(tempDir, f"tidl_{args.subgraph_id}_layer{args.layer_id:04d}.npy")
    tidl_tensor_files = [ f for f in os.listdir(tempDir) if f.endswith("_float.bin") and
                   f.startswith(f"tidl_import_subgraph{args.subgraph_id}.txt{args.layer_id:04d}") ]
    tidl_tensor_file = os.path.join(tempDir, tidl_tensor_files[0])
    print(f"{tvm_tensor_file} {tidl_tensor_file}")

    # Load tvm tensor
    tvm_tensor = np.load(tvm_tensor_file)
    print(tvm_tensor.shape)

    # Load tidl tensor
    with open(tidl_tensor_file) as f:
        tidl_tensor = np.fromfile(f,dtype='float32',count=-1,sep="")

    # compare two tensors
    print(f"tvm:  min={np.amin(tvm_tensor)}, max={np.amax(tvm_tensor)}, average={np.average(tvm_tensor)}")
    print(f"tidl: min={np.amin(tidl_tensor)}, max={np.amax(tidl_tensor)}, average={np.average(tidl_tensor)}")

    # plot two tensors
    print(f"plotting tvm and tidl tensors together, close window or press q to continue...")
    plt.plot(tvm_tensor.flatten(),  'b.', tidl_tensor.flatten(), 'r+')
    plt.title('tvm: blue dot(.)   tidl: red plus (+)')
    plt.show()

    # plot differences
    print(f"plotting differences of tvm and tidl tensors, close window or press q to continue...")
    plt.plot(tvm_tensor.flatten() - tidl_tensor.flatten(), 'x', color='purple')
    plt.title('(tvm - tidl): purple cross (x)')
    plt.show()

