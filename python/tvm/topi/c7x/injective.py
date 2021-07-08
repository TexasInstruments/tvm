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
# pylint: disable=invalid-name, unused-variable
"""Schedule for injective operators"""
import tvm
from tvm import te
import logging

#----------------------------------------------------------------
# Experimental C7x-specific schedule for injective (elementwise) ops.
def schedule_injective(outs):
    """C7X CPU schedule for injective op: C = A op B

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.Target.current(allow_none=False)
    logging.debug(f"schedule_injective for c7x, target={target}")

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([E.op for E in outs])
    C = outs[0]

    op = s[C].op
    a = op.input_tensors[0]
    b = op.input_tensors[1]
    inner = s[C].op.axis[-1]

    axes = list(op.axis)  # type ir.container.Array[tir.expr.IterVar]
    # local buffers
    if 1: 
        aa = s.cache_read(a, "local", op)
        cc : te.tensor.Tensor = s.cache_write(C, "local")
        inner = s[cc].op.axis[-1]
        #print_schedule(s)

    # use streaming for local buffer access on compute loop
    if 0:
        s[cc].pragma(s[cc].op.axis[0], "stream")

    # block on channel axis
    if 1:
        (io, ii) = s[C].split(axes[0], factor=8)
        #print("after split")
        #print_schedule(s)

    # sink copies into loop nest
    if 1:
        s[aa].compute_at(s[C], io)
        s[cc].compute_at(s[C], io)
        #print("after sink")
        #print_schedule(s)

    # double buffer
    # currently integrated as part of custom DMA pass
    if 0:
        s[aa].double_buffer()
        s[cc].double_buffer()

    # mark local<->ext copies as using dma.
    if 1:
        s[aa].pragma(s[aa].op.axis[0], "dma")
        s[C].pragma(ii, "dma")

    # fuse inner loops
    if 1:
        inner : tir.expr.IterVar = s[cc].fuse(s[cc].op.axis[-2], s[cc].op.axis[-1])
        #print("after fuse")
        #print_schedule(s)

    # split by 16 for vectorization
    if 1:
        (xyo, inner) = s[cc].split(inner, 16)
        #print("after split")
        #print_schedule(s)

    # vectorize on inner axis
    if 1:
        s[cc].vectorize(inner)
        #print("after vectorize")
        #print_schedule(s)

    show(s)
    return s

#----------------------------------------------------------------
# Debug code to print and visualize the schedules
from tvm.contrib import tedd
import graphviz as gv

def show(s):
    dotstr = tedd.viz_dataflow_graph(s, output_dot_string = True)
    gv.Source(source=dotstr, format='svg', filename='df.dot').render()

    dotstr = tedd.viz_schedule_tree(s, output_dot_string = True)
    gv.Source(source=dotstr, format='svg', filename='tree.dot').render()

    dotstr = tedd.viz_itervar_relationship_graph(s, output_dot_string = True)
    gv.Source(source=dotstr, format='svg', filename='iter.dot').render()


def print_schedule(s):
   print("schedule------")
   for st in s.stages:
       print(f"stage: {st}")
       for iv in st.all_iter_vars:
           print(f"   iter: {iv}")

