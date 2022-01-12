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
"""Schedule for C7x injective operators"""
from typing import Tuple, List, Union
import logging

import tvm
from tvm import te
from tvm import tir
from tvm.runtime import DataType
from .. import utils

logging = logging.getLogger("c7x_injective")

#----------------------------------------------------------------
# Experimental C7x-specific schedule for injective (elementwise) ops.
def schedule_injective(outs: Union[te.tensor.Tensor, List[te.tensor.Tensor]]) -> te.Schedule:
    """C7x schedule for injective op.

    Parameters:
    outs: Output tensor or list of output tensors

    Returns:
    sch: The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    # Note: AutoInline can result in compute getting "inlined" into copy loops
    # Such loops cannot be annotated with the dma pragma. See is_copy.
    tvm.te.schedule.AutoInlineInjective(s)
    for out in outs:
        if not utils.is_empty_shape(out.shape):
            schedule_injective_from_existing(s, out)
    return s


def schedule_injective_from_existing(s: te.Schedule,
                                     C: te.Tensor) -> te.Schedule:
    """C7X CPU schedule for injective op: C = A op B

    Parameters
    ----------
    sch: The schedule to update.
    C:   Tensor
         The tensor representing the injective op.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """

    target = tvm.target.Target.current(allow_none=False)
    logging.debug(f"schedule_injective for c7x, target={target}")

    #print("initial schedule")
    #print_schedule(s)

    # schedule transformations only apply to dimensioned operations
    if not s[C].op.axis:
        return s

    # if C result type is bool, bits==1, we need rhs data type to determine block size
    elem_bytes = int(DataType(C.dtype).bits / 8)
    if elem_bytes == 0:
        return s

    # Transform to use double buffering, local buffers and DMA
    s, cc, baxis, inner = double_buffer_with_dma(s, C)

    # fuse inner loops
    innerloops = s[cc].op.axis[baxis:]
    if len(innerloops) > 1:
        inner = s[cc].fuse(*innerloops)
    #print("after fuse")
    #print_schedule(s)

    # split by 16 for vectorization
    vector_length = 64
    (xyo, inner) = s[cc].split(inner, int(vector_length/elem_bytes))
    #print("after split")
    #print_schedule(s)

    # vectorize on inner axis
    s[cc].vectorize(inner)
    #print("after vectorize")
    #print_schedule(s)

    #show(s)
    #print("final schedule")
    #print_schedule(s)
    return s


def double_buffer_with_dma(s: te.Schedule,
                           C: te.tensor.Tensor) -> Tuple[te.Schedule,
                                                         te.Tensor,
                                                         int,
                                                         tir.IterVar]:

    """Update schedule with DMA and double buffering

    Three steps to enable double buffering + DMA
    1. Annotate input and output tensors as local
    2. Split the outer loop to ensure the local tensors can fit in L2
    3. Annotate loops performing the copies with the dma pragma

    Parameters:
    s: The schedule to update.
    C: The tensor representing the injective op.

    Returns:
    s:  The updated schedule (if feasible)
    cc: Updated output tensor (local buffer)
    baxis: Axis that was split for blocking
    inner: Innermost axis of the stage
    """

    cc = C

    # TODO: allow these to vary by target configuration
    max_block = 0x4000   #16k

    op = s[C].op

    outer = op.axis[0]
    inner = op.axis[-1]
    block = outer       # block-level loop, if split
    baxis = 0           # axis index of block-level loop


    elem_bytes = int(DataType(C.dtype).bits / 8)
    dims       = list(tvm.auto_scheduler.utils.get_const_tuple(C.shape))

    # Cannot analyze if the dimensions are not constant. Return.
    constant_dim = True
    try:
        out_len = utils.prod(C.shape)
        const_size = utils.get_const_int(out_len)
    except ValueError:
        constant_dim = False

    if constant_dim is False:
        return (s, cc, baxis, inner)


    # Find split point, including axis to split if needed
    baxis, nblocks, blocksize = find_split(dims, elem_bytes, max_block)

    # If there is no split point (e.g. split results in odd iterations), return
    if blocksize == 0:
        return (s, cc, baxis, inner)

    # local buffers
    # Creates local copies of specified buffers
    # Inserts loops to copy-in to local inputs and copy-out from local outputs

    local_inputs = []

    for t in op.input_tensors:
        if len(t.shape) > 1:
            l = s.cache_read(t, "local", op)
            local_inputs.append(l)
    if len(dims) > 1:
        cc = s.cache_write(C, "local")
        inner = s[cc].op.axis[-1]
    logging.debug("after local buffers")
    print_schedule(s)

    # Split on the axis indicated by find_split to reduce local buffer size
    if baxis != 0:
        if nblocks != dims[baxis-1]:
            logging.debug(f"split at {baxis} by {nblocks}")
            old = dims[baxis-1]
            sdim = [nblocks, int(old / nblocks)]
            logging.debug(f"split dim: {old} -> {sdim}")
            dims = dims[:baxis-1] + sdim + dims[baxis:]
            (outer, block) = s[C].split(s[C].op.axis[baxis-1], nparts=nblocks)
            if baxis == len(s[C].op.axis):
                inner = block
        else:
            logging.debug(f"split at {baxis}")
            (outer, block) = (s[C].op.axis[baxis-1], s[C].op.axis[baxis])

        outers = dims[:baxis]
        inners = dims[baxis:]
        logging.debug(f"after split: {outers} {inners}, blocksize={blocksize}")
        logging.debug("after split")
        print_schedule(s)

        # sink local-buffer copies into outer loop
        for t in local_inputs:
            s[t].compute_at(s[C], outer)
        if cc != C:
            s[cc].compute_at(s[C], outer)
        logging.debug("after sink")
        print_schedule(s)

    # mark local<->ext copies as using dma.
    for t in local_inputs:
        # AutoInlineInjective can push compute into the copy loops - such loops
        # cannot be annotated with the dma pragma.
        #dump(t)
        if is_copy(t):
            s[t].pragma(s[t].op.axis[0], "dma")
    if cc != C:
        #dump(cc)
        # if no split above, block is outer loop
        s[C].pragma(block, "dma")

    return (s, cc, baxis, inner)

def is_copy(tensor):
    ''' Return False if any of the ops contributing to tensor has more than 2
        inputs, indicating that it is not just a copy operation.
    '''
    if isinstance(tensor.op, tvm.te.ComputeOp):
        # If there is more than one input, this is not a copy operation
        if len(tensor.op.input_tensors) > 1:
            return False

    for t in tensor.op.input_tensors:
        if is_copy(t) == False:
            return False

    return True

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
    blocksize = elem_bytes
    for axis,dim in list(enumerate(dims))[::-1]:
        blocksize *= dim
        #print(f"dim {axis}=[{dim}];  blocksize={blocksize}")
        # If block too big, split. Find a split that evenly divides axis.
        if blocksize > limit:
            nblocks = int(blocksize/limit)
            while int(dim/nblocks) * nblocks != dim:
                nblocks += 1
            iter_range = int(blocksize/nblocks/elem_bytes)

            # If block split results in odd number of iterations, do not split
            # Downstream passes cannot handle loops with odd iteration counts
            if iter_range % 2 != 0:
                #print(f"Invalid blocksize resulting in odd range: {iter_range}")
                return (0, 1, 0)

            return (axis+1, nblocks, int(blocksize/nblocks))

    return (0, 1, blocksize)   # no split needed



#----------------------------------------------------------------
# Debug code to dump TIR
def dump(tensor, indent=''):
    ''' Traverse TIR and print operands, operations '''
    logging.debug(f'{indent}tensor = {tensor.name}')
    logging.debug(f'{indent}op = {tensor.op.name}, tag={tensor.op.tag}, inputs={len(tensor.op.input_tensors)}')
    if isinstance(tensor.op, tvm.te.ComputeOp):
      logging.debug(f'{indent}body = {tensor.op.body}, expr type={type(tensor.op.body[0])}')

    if len(tensor.op.input_tensors) > 0:
        indent = indent + ' '
    for t in tensor.op.input_tensors:
        dump(t, indent)


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
    logging.debug("schedule------")
    for st in s.stages:
        logging.debug(f"stage: {st}")
        for iv in st.all_iter_vars:
            logging.debug(f"   iter: {iv}")
