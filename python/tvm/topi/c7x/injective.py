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
from tvm.contrib.tidl.c7x import register_c7x_local_mem
from .. import utils, tag

logging = logging.getLogger("c7x_injective")

# Operations that dma (from c7x_tvm_runtime.h) does not apply.
# Reason 1: input tensor is not completely used in the computation.
#   e.g. T_strided_slice from input tensor (1, 1, 851700, 6) to (1, 1, 100, 6).
#   Only first 600 elements are copied from input tensor to output tensor.
#   C7x dma runtime uses input tensor dimension and loop nest bounds to
#   configure the dma transfer, which will result 8517 transfers, while only 1
#   is really needed.  Besides, T_strided_slice is a simple tensor to tensor copy
#   with no computation and dma (with double buffering) brings no benefits.
# - T_concat should not be implemented with DMA, it does not have performance benefits.
#   We should write a separate schedule for T_concat, split the concat axis into several
#   loops nests, each loop nest with one or two input tensors and use streaming engine
#   for the input tensors. (TODO)
# - T_reshape causes lhs and rhs tensors using different indexing expressions in different
#   dimensions.  This makes creating a local copy of rhs using the lhs loop bounds problematic.
#   Disable for now (creating local buffers and dma).  We should revisit to see if the loop nest
#   can still be fused and vectorized.
ops_do_not_dma = ['T_strided_slice', 'T_concat', 'T_reshape']

# Do not DMA cases
# 1. specific operators that DMA does not bring performance benefits, they might
#    benefit from different schedules, e.g. T_concat
# 2. specific operators that only parts of input tensors are loaded and yet we use
#    the whole tensor shape to configure how many blocks will be transferred,
#    e.g. T_strided_slice
# 3. There is only 1 block or the blocking axis is outside the whole loop nest,
#    is it worthwhile to DMA each input as one block, and DMA the output as one block?
#    Should we simply skip DMA since there will not be double buffering?
# 4. lhs tensor and rhs tensor have differenet dimensions and therefore use different indexing
#    expressions, e.g. T_reshape
#
# DMA configurations for output and input: out[f(i,j,k)], in[g(i,j,k)]
#    DetectLinearEquation analyzes f(i,j,k) and g(i,j,k) to compute the coeff for each loop var,
#    e.g. f(i,j,k) = strides[0] * i + strides[1] * j + strides[2] * k + strides[3]
# 1. original loop nest range (ri, rj, rk) where each ri,rj,rk is a (min, ext) pair
# 2. assuming loop nested is blocked on j loop into (ri, orj, irj, rk))
#    blocked loop nest range  (irj, rk)
# 3. Total block size for output: (f(i,j,k) over (ri, rj, rk)), this should be the same
#    as output Var's size
# 4. Local block size for output over (irj, rk), this is computed as "src_shape/dst_shape"
#    with "src_strides/dst_strides" in CopyIntrinInjector() pass.  At some point
#    (before or after CopyIntrinInjector pass), it is also reflected in the local block
#    Var's shape, which is used in DMA config.
# 5. Total block size for input: (g(i,j,k) over (ri, rj, rk)), this may or may NOT be the
#    same as input Var's size, as we see in T_strided_slice's case
#    How do we catch this case:
#    - (Schedule) Insert dma pragma on copy in and copy out loops
#    - (InjectCopyIntrin) Compute the ICNTs, STRIDEs for the inner loops (copying loops),
#                         store information in the "c7x_dma_copy" intrinsic.
#                         Do not turn the loop nest into c7x_dma_copy intrinsic if ICNTs, STRIDEs
#                         are not integer constants.
#    - (DMAPass) Compute the ICNTs, STRIDEs for the outer loops (loops contains the c7x_dma_copy),
#                store the information in the "c7x_dma_setup" intrinsic.
# 6. Local block size for input over (irj, rk), similarly to output, it is reflected correctly
#    in the local block Var's shape, which is used in DMA config.

# Special ops that can be treated as injective and use the injective schedule,
# e.g. max_pool2d with 1x1 pool_size, the computation has reduced to injective op
# For such cases, we can explicitly call injective schedule
ops_treat_as_injective = ['c7x_max_pool2d_1x1' ]

# SE vector does not directly support strided access, c7x_max_pool2d_1x1 might have strided access
ops_do_not_vectorize = ['c7x_max_pool2d_1x1' ]

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

    def is_unsupported_op(tensor, is_invalid_op):
        ''' Return True if any of the ops contributing to tensor is not supported
            by the C7x implementation for injective ops.
        '''
        if isinstance(tensor.op, tvm.te.ComputeOp):
            if is_invalid_op(tensor.op):
                logging.debug(f"is_invalid_op true for op={tensor.op.name}")
                return True

            for t in tensor.op.input_tensors:
                if is_unsupported_op(t, is_invalid_op):
                    return True

        return False


    def vectorize(sch: te.Schedule, C: te.Tensor, cc: te.Tensor, baxis: int, inner: tir.IterVar) -> te.Schedule:
        '''Attempts to vectorize the schedule'''

        def contains_call(tensor: te.Tensor) -> bool:
            ''' Return True if tensor computation contains a call (that we cannot vectorize).'''
            seen_call = False
            def visit(expr):
                nonlocal seen_call
                if isinstance(expr, tvm.tir.Call): #e.g. expr.op.same_as(tvm.ir.Op.get("tir.floor")):
                    seen_call = True

            if isinstance(tensor.op, tvm.te.ComputeOp):
                for e in tensor.op.body:
                    if isinstance(e, tvm.tir.PrimExpr):
                        tvm.tir.stmt_functor.post_order_visit(e, visit)
                        if seen_call:
                            return True

                for t in tensor.op.input_tensors:
                    if contains_call(t):
                        return True

            return seen_call


        def largest_element(tensor: te.Tensor, largest_element_bytes: int) -> int:
            ''' Returns the largest element size in bytes across all tensors referenced
                in an expression. Used to determine the vectorization factor.
            '''
            elem_bytes = int(DataType(tensor.dtype).bits / 8)
            if elem_bytes > largest_element_bytes:
                largest_element_bytes = elem_bytes

            logging.debug(f"largest_element for {tensor}, size={elem_bytes}, l={largest_element_bytes}")
            if isinstance(tensor.op, tvm.te.ComputeOp):
                for t in tensor.op.input_tensors:
                    largest_element_bytes = largest_element(t, largest_element_bytes)

            return largest_element_bytes


        # Use the largest element in the expression to determine the vectorization factor
        largest_elem_bytes = largest_element(C, 0)

        # Fuse inner loops
        innerloops = sch[cc].op.axis[baxis:]
        if len(innerloops) > 1:
            inner = sch[cc].fuse(*innerloops)
        #logging.debug("after fuse")
        #print_schedule(s)
        GetCurrentTIDLContext = tvm.get_global_func("tidl.GetCurrentTIDLContext")
        ctx = GetCurrentTIDLContext()
        vector_length = 32 if ctx.platform == "AM62A" else 64
        split_factor = int(vector_length/largest_elem_bytes)
        inner_length = 1
        for axis_len in cc.shape[baxis:]:
            inner_length *= axis_len

        # Do not split/vectorize if the number of inner loop iterations is
        # less than the vectorization factor
        if inner_length < split_factor:
            return sch

        # Do not split/vectorize if the computation contains a call
        if contains_call(C):
            return sch

        # Split for vectorization
        (xyo, inner) = sch[cc].split(inner, split_factor)
        #logging.debug("after split")
        #print_schedule(s)

        # Annotate inner axis as vectorized
        sch[cc].vectorize(inner)
        #logging.debug("after vectorize")
        #print_schedule(s)

        return sch



    target = tvm.target.Target.current(allow_none=False)
    logging.debug(f"schedule_injective for c7x, target={target}")

    # Check for unsupported ops
    # Only apply tiling and dma on subset of injective ops: broadcast ops (including elemwise ops)
    # - For ops that are injective but not broadcast/elemwise, e.g. ones in ops_do_not_dma list,
    #   output tensor indexing axis might have different ordering/bounds than the input tensor.
    #   When using block size computed from output tensor to tile the input tensor, we might
    #   encouter problems such as:
    #   - After sinking with "compute_at", input tensor block size cannot be reduced
    #   - The loop nest tiled for output block cannot compute the tiled input tensor block
    #   - After vectorization, complicated index computation in input tensor blocks produces
    #     complicated vector masks
    # - For broadcast/elemwise ops, output tensor and input tensor always have the same axis
    #   ordering.
    if is_unsupported_op(tensor=C, is_invalid_op=(lambda t: not tag.is_broadcast(s[t].op.tag) and \
                                                  s[t].op.name not in ops_treat_as_injective)):
        return s

    # schedule transformations only apply to dimensioned operations
    if not s[C].op.axis:
        return s

    # if C result type is bool, bits==1, we need rhs data type to determine block size
    elem_bytes = int(DataType(C.dtype).bits / 8)
    if elem_bytes == 0:
        return s

    # Transform to use double buffering, local buffers and DMA
    s, cc, baxis, inner = double_buffer_with_dma(s, C)

    if (s[C].op.name in ops_do_not_vectorize):
        return s

    # Attempt to vectorize the schedule
    s = vectorize(s, C, cc, baxis, inner)

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

    def tensor_size(t: te.tensor.Tensor) -> int:
        size = 1
        for dim_size in t.shape:
            size *= utils.get_const_int(dim_size)
        return size

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
    # If the whole loop nest does not need split, do not dma, return
    if blocksize == 0 or baxis == 0:
        return (s, cc, baxis, inner)

    # local buffers
    # Creates local copies of specified buffers
    # Give each local copy a unique "local" + "<tag>" scope_name, so that
    #     they do not get "shared"/"fused" by tir.StorageRewrite pass
    # Inserts loops to copy-in to local inputs and copy-out from local outputs

    local_inputs = []

    for t in op.input_tensors:
        is_placeholder = isinstance(t.op, tvm.te.PlaceholderOp)
        if len(t.shape) > 1:
            scope_name = f"local{len(local_inputs)}"
            register_c7x_local_mem(scope_name)
            l = s.cache_read(t, scope_name, op)
            local_inputs.append((l, is_placeholder))
    if len(dims) > 1:
        register_c7x_local_mem("local")
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
        for t, _ in local_inputs:
            s[t].compute_at(s[C], outer)
        if cc != C:
            s[cc].compute_at(s[C], outer)
        logging.debug("after sink")
        print_schedule(s)

    # mark local<->ext copies as using dma.
    for t, is_placeholder in local_inputs:
        # AutoInlineInjective can push compute into the copy loops - such loops
        # cannot be annotated with the dma pragma.
        #dump(t)
        if is_placeholder:
            s[t].pragma(s[t].op.axis[0], "dma")
    if cc != C:
        #dump(cc)
        # if no split above, block is outer loop
        s[C].pragma(block, "dma")

    return (s, cc, baxis, inner)

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
