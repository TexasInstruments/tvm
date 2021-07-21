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
import tvm.auto_scheduler.utils
from tvm.runtime import DataType
from functools import reduce
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
    # TODO: allow these to vary by target configuration
    max_block = 0x4000   #16k
    vector_length = 64   #16k

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([E.op for E in outs])
    C = outs[0]

    op = s[C].op
    a = op.input_tensors[0]
    b = op.input_tensors[1]

    # schedule transformations only apply to dimensioned operations
    if not s[C].op.axis:
        return s

    outer = s[C].op.axis[0]
    inner = s[C].op.axis[-1]
    block = outer       # block-level loop, if split
    baxis = 0           # axis index of block-level loop
    elem_bytes = int(DataType(C.dtype).bits / 8)

    # local buffers
    aa = a
    bb = b
    cc = C
    if 1: 
        if len(a.shape) > 1:
            aa = s.cache_read(a, "local", op)
        if len(b.shape) > 1:
            bb = s.cache_read(b, "local", op)
        if len(C.shape) > 1:
            cc = s.cache_write(C, "local")
            inner = s[cc].op.axis[-1]
        #print("after local buffers")
        #print_schedule(s)

    # split outer loop into bite-size chunks
    if 1:
        dims = list(tvm.auto_scheduler.utils.get_const_tuple(C.shape))
        # TODO: make sure dims are const
        # Find split point, including axis to split if needed
        baxis, nblocks, blocksize = find_split(dims, elem_bytes, max_block)
        if baxis != 0:
            if nblocks != dims[baxis-1]:
                #print(f"split at {baxis} by {nblocks}")
                old = dims[baxis-1]
                sdim = [nblocks, int(old / nblocks)]
                #print(f"split dim: {old} -> {sdim}")
                dims = dims[:baxis-1] + sdim + dims[baxis:]
                (outer, block) = s[C].split(s[C].op.axis[baxis-1], nparts=nblocks)
            else:
                #print(f"split at {baxis}")
                (outer, block) = (s[C].op.axis[baxis-1], s[C].op.axis[baxis])

            outers = dims[:baxis]
            inners = dims[baxis:]
            #print(f"after split: {outers} {inners}, blocksize={blocksize}")
            #print("after split")
            #print_schedule(s)

            # sink local-buffer copies into outer loop
            if 1:
                if aa != a:
                    s[aa].compute_at(s[C], outer)
                if bb != b:
                    s[bb].compute_at(s[C], outer)
                if cc != C:
                    s[cc].compute_at(s[C], outer)
                #print("after sink")
                #print_schedule(s)

    # mark local<->ext copies as using dma.
    if 1:
        if aa != a:
            s[aa].pragma(s[aa].op.axis[0], "dma")
        if bb != b:
            s[bb].pragma(s[bb].op.axis[0], "dma")
        if cc != C:
            # if no split above, block is outer loop
            s[C].pragma(block, "dma")

    # fuse inner loops
    if 1:
        innerloops = s[cc].op.axis[baxis:]
        if len(innerloops) > 1:
            inner = s[cc].fuse(*innerloops)
        #print("after fuse")
        #print_schedule(s)

    # split by 16 for vectorization
    if 1:
        #(xyo, inner) = s[cc].split(s[cc].op.axis[-1], int(vector_length/elem_bytes))
        (xyo, inner) = s[cc].split(inner, int(vector_length/elem_bytes))
        #print("after split")
        #print_schedule(s)

    # vectorize on inner axis
    if 1:
        s[cc].vectorize(inner)
        #print("after vectorize")
        #print_schedule(s)

    show(s)
    return s


def find_split(dims, elem_bytes, limit):
    ''' Given dimensions and max block size, find even split such 
        that inner dimensions fit in block.  Returned axis is outer axis 
        of element-level loop. Returned nblocks is number of blocks for 
        axis-1, which may or may not equal the original. Examples:
           split([672, 14, 14], 4, 8192) --> 
               axis=1, nblocks=84 -> [84] [8 14 14]
           split([10, 100, 200], 1, 200) --> 
               axis=2, nblocks=100 --> [10 100] [200]
    '''
    # Work from inner to outer until block gets too big
    blocksize = elem_bytes;
    for axis,dim in list(enumerate(dims))[::-1]:
        blocksize *= dim
        split = 1
        #print(f"dim {axis}=[{dim}];  blocksize={blocksize}")
        # If block too big, split. Find a split that evenly divides axis.
        if blocksize > limit:
             nblocks = int(blocksize/limit)
             while int(dim/nblocks) * nblocks != dim:
                 nblocks += 1
             return (axis+1, nblocks, int(blocksize/nblocks))
    return (0, 1, blocksize)   # no split needed

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

