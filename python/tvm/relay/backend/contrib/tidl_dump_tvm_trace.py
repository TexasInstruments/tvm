
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
"""TIDL parse and dump TVM trace file"""

# This is a simple parsing and dumping utility for TVM trace.
# trace definition see src/runtime/crt/graph_executor/graph_executor.c

import os
import sys
import struct
import numpy as np

def read_trace(trace_filename : str):
  def read_int(f):
    return int.from_bytes(f.read(4), byteorder='little')
  def read_int64(f):
    return int.from_bytes(f.read(8), byteorder='little')
  def read_float(f):
    return struct.unpack('f', f.read(4))[0]
  def read_str(f):
    barray = bytes('', 'utf-8')
    byte = f.read(1)
    while byte != b'\0':
      barray += byte
      byte = f.read(1)
    return barray.decode()

  def read_output(f):
    output = {}
    oid = read_int(f)
    if oid == 0xF000E001:
      return None

    output['oid'] = oid
    output['ndim'] = read_int(f)
    output['type_code'] = read_int(f)
    output['elem_bytes'] = read_int(f)
    output['num_elements'] = read_int(f)
    output['min'] = read_float(f)
    output['max'] = read_float(f)
    output['sum'] = read_float(f)
    output['fh_sum'] = read_float(f)

    tensor_values = []
    num_tensor_values = read_int(f)
    if num_tensor_values:
      for i in range(num_tensor_values):
        tensor_values.append(read_float(f))
    output['tensor_values'] = tensor_values
    return output

  def read_node(f):
    node = {}
    nid = read_int(f)
    if nid == 0xF000E002:
      return None

    node['nid'] = nid
    node['name'] = read_str(f)
    node['time'] = read_int64(f)
    outputs = []
    output = read_output(f)
    while output:
      outputs.append(output)
      output = read_output(f)
    node['outputs'] = outputs
    return node

  trace = {}
  with open(trace_filename, "rb") as f:
    trace['size'] = read_int(f)
    trace['version'] = read_int(f)
    trace['device'] = read_int(f)
    trace['core'] = read_int(f)
    nodes = []
    node = read_node(f)
    while node:
      nodes.append(node)
      node = read_node(f)
    trace['nodes'] = nodes
  return trace

def dump_trace(trace, c7x_mhz):
  print(f"Trace size: {trace['size']} version: {hex(trace['version'])} device: {'J7' if trace['device'] == 0 else 'unknown'} core: {'Arm' if trace['core'] == 0 else 'C7x'}")
  div_to_ms = 1000
  if trace['core'] != 0:
    div_to_ms = c7x_mhz

  for node in trace['nodes']:
    print(f"node {node['nid']}: {node['name']}  {node['time']/div_to_ms} microseconds")
    for output in node['outputs']:
      print(f"  output {output['oid']}: ndim={output['ndim']} type_code={output['type_code']} elem_bytes={output['elem_bytes']} num_elements={output['num_elements']}")
      print(f"           min={output['min']} max={output['max']} sum={output['sum']} fh_sum={output['fh_sum']}")
      tensor_values = output['tensor_values']
      if tensor_values:
        tvf = os.path.join(os.path.dirname(sys.argv[1]), f"n{node['nid']}_o{output['oid']}.npy")
        np.save(tvf, np.asarray(tensor_values))
        print(f"           tensor values saved in {tvf}")

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('trace_filename', help='TVM trace file name, e.g. tvm_c7x.trace')
  parser.add_argument('--c7x_mhz', type=int, default=1000,
                      help='C7x core frequency in MHz, default to 1000')
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  trace = read_trace(args.trace_filename)
  dump_trace(trace, args.c7x_mhz)

