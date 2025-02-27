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
# pylint: disable=invalid-name
"""Support for C7x TVM Target """

import tvm
import logging


platform_dict = {
        "am68pa": "J7",
        "am68a": "J721S2",
        "am69a": "J784S4",
        "am67a": "J722S",
        "am62a": "AM62A"
    }

def supported_platform(platform): 
    return platform in ["J7", "J721S2", "J784S4", "J722S", "AM62A"] or platform in platform_dict


def platform_map(platform):
    return platform_dict[platform] if platform in platform_dict else platform

#----------------------------------------------------------------
# Sets up custom Target and PassConfig contexts for C7x. 
# Usage:
#   with c7x_target_config():
#      tvm.lower()
#      tvm.build()
# This would be a C7x-specific module in TVM. 
# FIXME: should properly use contextlib.ExitStack
class c7x_target_config():
   def __init__(self):
      self.target = tvm.target.Target("c7x -device=c7x")
      self.pass_context = c7x_pass_context()
   def __enter__(self):
      self.target.__enter__()
      self.pass_context.__enter__()
      return self
   def __exit__(self, type, value, traceback):
      self.target.__exit__(type, value, traceback)
      self.pass_context.__exit__(type, value, traceback)

# Create custom PassContext, used by tvm.lower and tvm.build.
# Used within a 'with' statement, the context is saved in persistent state and available
# via PassContext.current().
# See also vta/build_module.py
def c7x_pass_context():
   # optimization level; this affects set of Relay IR optimizations during BuildRelay()
   opt=3
   # Passes to disable
   disable = []
   # Leave allocations of "local" buffers alone
   disable.append("tir.LowerDeviceStorageAccessInfo")
   # Disable lowering TVM calls; this prevents the host kernel from using 
   # the packed func protocol to call the device kernel
   disable.append("tir.LowerTVMBuiltin")

   # pass-specific config options
   #   add passes to lowering pipeline as a list of (phase, pass) tuples, 
   #   where 'phase' indicates the position in the lowering pipeline (0-3). 
   #   See tvm.lower.
   config = {}
   config["tir.add_lower_pass"] = [
        # Move storage_scope attributes and allocate nodes out of their
        # enclosing scope
        #(1, LiftAllocToScopeBegin()),
        # Replace index expressions with stream-based access
        (1, C7xSEPass()),

        # Replace buffer copies that have "pragma_dma" with intrinsic calls. 
        # implemented via CopyIntrinInjector, which analyzes the copy loop and 
        # invokes fintrin to create the dma call. See vta/transform.py
        (2, tvm.tir.transform.InjectCopyIntrin(pragma_key="dma",
                                               fintrin=c7x_dma_injector)),
        # Insert C7x DMA intrinsics 
        (2, C7xDMAPass()),

        # Replace allocate statements with c7x-specific L2 allocation
        # Currently part of the DMA pass
        #(3, ir_lower_vtcm()),

        # Annotate functions with the "device_scope" attribute, which causes 
        # tvm.build to split each function into a host wrapper and device 
        # kernel. The host function is compiled with 'target_host' and the 
        # device function with 'target'
        (3, tvm.tir.transform.DecorateDeviceScope())
   ]
   # config options for individual passes
   config["tir.InjectDoubleBuffer"] = { "split_loop": 0 }
   pass_context = tvm.transform.PassContext(opt_level=opt, 
                                            disabled_pass=disable, 
                                            config=config)
   return pass_context

#----------------------------------------------------------------
# Register c7x-specific local memory tag
# So that tagged local memory has properly defined size and we don't need to modify TVM source
from tvm._ffi.registry import register_func

def register_c7x_local_mem(scope_name):
    def register_helper(func_name):
        @register_func(func_name)
        def mem_info_c7x():
            # pylint: disable=bad-whitespace
            return tvm.ir.make_node(
                "MemoryInfo",
                unit_bits=8,
                max_num_bits=448*1024*8,
                max_simd_bits=64,         # used for buffer alignment
                head_address=None         # starting address of mem
            )

    func_name = "tvm.info.mem." + scope_name
    if not tvm.get_global_func(func_name, allow_missing=True):
        register_helper(func_name)

#----------------------------------------------------------------
def merge_block(slist, body):
    '''
    Helper function to edit scoping constructs.
    Given one or more scoping constructs (attribute/allocate/let) and a body, 
    attach each construct to the body. For example:
       slist = [attr, alloc, let];    body = { ... }
    this becomes:
       attr { alloc { let { ... } } } 
    
    see tir::MergeNest, also vta/transform.py
    '''
    slist.reverse()
    for op in slist:
        if op.body == body:
            body = op
        elif isinstance(op, tvm.tir.Allocate):
            body = tvm.tir.Allocate(op.buffer_var, op.dtype, op.extents, 
                                    op.condition, body)
        elif isinstance(op, tvm.tir.AttrStmt):
            body = tvm.tir.AttrStmt(op.node, op.attr_key, op.value, body)
        elif isinstance(op, tvm.tir.LetStmt):
            body = tvm.tir.LetStmt(op.var, op.value, body)
        elif isinstance(op, tvm.tir.For):
            body = tvm.tir.For(
                op.loop_var,
                op.min,
                op.extent,
                op.kind,
                body,
                op.thread_binding,
                op.annotations,
            )
        else:
            raise RuntimeError("unexpected op")
    del slist[:]
    return body

#----------------------------------------------------------------
# C7x Streaming Pass

"""
This pass detects loads and stores within loop nests and replaces the index
expressions with c7x_stream_access intrinsics. It also inserts intrinsics 
to configure, open, and close the stream.

before:
       for (i, 0, Ni) {
         for (j, 0, Nj) {
           for (k, 0, Nk) "vectorized" {
             if (j*Nk + k < N)
               A[i*Nj*Nk + j*Nk + k] = B[i*Nj*Nk + j*Nk + k]
           }
         }
      }
after:
       let SE.Config0: @tir.call_extern("c7x_stream_config", ...)
       let SA.Config1: @tir.call_extern("c7x_stream_config", ...)
       ...
       @tir.call_extern("c7x_stream_open", SE.Config0, "SE0", B)
       @tir.call_extern("c7x_stream_open", SA.Config1, "SA0", A)
       for (i, 0, Ni) {
         for (j, 0, Nj) {
           for (k, 0, Nk) "vectorized" {
             // if guard removed
               A[@tir.c7x.stream_access(SA.Config1, "SA0", ...)] = 
               B[@tir.c7x.stream_access(SE.Config0, "SE0", ...)]
           }
         }
       }
       @tir.call_extern("c7x_stream_close", SE.Config0, "SE0")
       @tir.call_extern("c7x_stream_close", SA.Config1, "SA0")

  Note that the 'if' that guards access past the vector length is
  removed, relying on the store stream to apply predication.
"""
from functools import partial

def SETransform(f, mod, ctx):
    ''' 
    Implementation of C7xSEPass. The general outline is as follows.
    1. Find candidates pass
       a. find all loads and stores and create an SECandidate for each.
       b. Build an interference graph for loops
    2. Qualify 
       a. Sort candidiates by profitability
       b. Qualify each candidate to see if streaming is legal for that access
       c. Allocate a SE/SA, using the loop interference graph to avoid
          conflicts.
    3. Deploy pass
       a. For each qualified candidate:
          i. Insert open/close calls around the its outermost loop
          ii. Convert index expression of access to stream_access call
       b. Remove if guards that are no longer needed
    4. Wrap-up
       a. Insert stream_config calls at top of function

    valid case 1: the loop is annotated as to be "vectorized" and vectorized later
      for (ax1.c: int32, 0, 84) {
        for (ax2.c.ax3.c.fused.outer: int32, 0, 4) {
          for (ax2.c.ax3.c.fused.inner: int32, 0, 16) "vectorized" {
            if @tir.likely((((ax2.c.ax3.c.fused.outer*16) + ax2.c.ax3.c.fused.inner) < 49), dtype=bool) {
              T_multiply.local[(((ax1.c*49) + (ax2.c.ax3.c.fused.outer*16)) + ax2.c.ax3.c.fused.inner)] = ((float32*)placeholder.local0[(((ax1.c*49) + (ax2.c.ax3.c.fused.outer*16)) + ax2.c.ax3.c.fused.inner)]*(float32*)placeholder.local1[ax1.c])
            }
          }
        }

    valid case 2: loop annotated as to be "vectorized", but failed to vectorize later,
                  we still use vector predicate for scalar store
        for (ax1.c: int32, 0, 2) {
          for (ax2.c.ax3.c.fused.outer: int32, 0, 69) {
            for (ax2.c.ax3.c.fused.inner: int32, 0, 16) "vectorized" {
              if @tir.likely((((ax2.c.ax3.c.fused.outer*16) + ax2.c.ax3.c.fused.inner) < 1089), dtype=bool) {
                T_layout_trans.local[(((ax1.c*1089) + (ax2.c.ax3.c.fused.outer*16)) + ax2.c.ax3.c.fused.inner)] = (float32*)T_concat.local0[(((ax2.c.ax3.c.fused.outer*32) + (ax2.c.ax3.c.fused.inner*2)) + ax1.c)]
              }
            }
          }

    valid case 3: loop not annotated as to be "vectorized" (loop from original schedule),
                   no vector predicate
        for (j_6: int32, 0, 851760) {
          if (0f32 <= (float32*)hybrid_nms.v0[(j_6*6)]) {
            for (k_7: int32, 0, 6) {
              hybrid_rearrange_box_out_2[(((int32*)valid_indices[0]*6) + k_7)] = (float32*)hybrid_nms.v0[((j_6*6) + k_7)]
            }
            valid_indices[0] = ((int32*)valid_indices[0] + 1)
          }
          if ((int32*)valid_indices[0] <= j_6) {
            for (k_8: int32, 0, 6) {
              hybrid_rearrange_box_out_2[((j_6*6) + k_8)] = -1f32
            }
          }
        }

    valid case 4: load is inside if_then_else expression, should not SE
        for (ax1: int32, 0, 6132) {
          for (ax2: int32, 0, 6) {
            T_concat_2[((ax1*6) + ax2)] = @tir.if_then_else((2 <= ax2), (float32*)placeholder_6[(((ax1*4) + ax2) - 2)], @tir.if_then_else((1 <= ax2), (float32*)placeholder_8[(((ax1*80) + ax2) + 62)], (float32*)placeholder_7[(((ax1*80) + ax2) + 63)], dtype=float32), dtype=float32)
          }
        }


    # Conditions for memory accesses that can be turned into SE/SA
    # 1. within the innermost loop
    # 2. the indexing expression conforms to the innermost loop nest (e.g. a*i + b*j + c*k + d)
    # 3. the conforming innermost loop nest cannot contain IfThenElse except the veclen_guard,
    #    because SE/SA cannot be programmed to handle conditions other than veclen_guard
    # 4. Load cannot be under if_then_else expression
    #
    # Conditions for IfTenElse being a true veclen_guard and safely removed
    # 1. within the innermost loop, cannot be between two loops in the loop nest
    # 2. the innermost loop is vectorized
    # 3. there is a qualified SECandidate associated with this guard
    #
    # Place for inserting the SE/SAConfig and Open/Close
    # 1. configs can be hoisted to the beginning of function
    # 2. Open/Close can be around the level of the definition,
    #    upper (icnts=1, dims= 0) handles the case where outer loops are not used in indexing
    #
    # Valid parameters for SE/SA Config
    # 1. Using loop bounds and coefficients used in memory access expressions
    '''
    # map of var definitions to loop nesting level; delimits outer scope
    def_levels = {}
    # stack of ForStmts at current point
    loop_nest = []         
    # stack of (IfThenElseStmt, len(loop_nest)) at current point
    if_nest = []
    # stack of if_then_else expression at current point
    if_expr_nest = []
    # currently in-effect guard condition
    veclen_guard = None 
    # flat list of all SECandidates
    candidates = []
    # map each loop to candidates whose outer span is that loop
    loop_candidates = {}
    # map each load and store to the corresponding candidate
    op_candidates = {}
    # map each loop to all loops that overlap it (including itself)
    loop_overlaps = {}
    # numerical id for each loop, for debug
    loop_ids = {}
    # qualified guards that could be removed later
    qualified_guards = []

    class SECandidate():
        '''
        This class represents an instance of a load or store, that can possibly
        be converted to use streaming access.
        '''
        def __init__(self, op, nest, this_if_nest, trip, guard):
            # the load or store operator, and var operand
            self.op = op
            self.var = op.buffer.data
            # loops between var's def and access
            self.nest = nest
            # ifs between var's def and access
            self.if_nest = this_if_nest
            self.outer_loop = nest[0] if nest else None
            self.guard_condition = guard
            self.config = None
            self.config_var = None
            self.engine = None
            self.qualified = False
            # Sort criteria for priorization. 
            #  1. Prefer guarded accesses (enables guard to be subsumed)
            #  2. Prefer higher total trip counts
            self.sort_key = (int(self.guard_condition is not None), trip)

        def qualify(self):
            '''
            See if a load or store to qualifies to use stream-based access
            '''
            logging.debug(f"qualify: {self}")
            op = self.op
            index = op.indices[0]
            buf = op.buffer.data
            if not self.nest:
                return False
            # the loop nest cannot contain IfThenElse except the veclen_guard,
            #   because SE/SA cannot be programmed to handle conditions other than veclen_guard
            if len(self.if_nest) > 1 or \
               (len(self.if_nest) == 1 and
                (self.if_nest[0][1] != len(self.nest) or
                 _get_if_condition(self.if_nest[0][0]) != self.guard_condition)):
                logging.debug(f"  disqualified due to if_nest:{[(_get_if_condition(x[0]), x[1]) for x in self.if_nest]}")
                return False
            # Analyze the index expression to get the coefficients of the
            # loop index variables. Given: 
            #   A[i*Ki + j] and [i, j]
            # the result is [Ki, 1, 0]
            loop_vars = [l.loop_var for l in self.nest]
            extents = [l.extent for l in self.nest]
            coeffs = tvm.arith.detect_linear_equation(index, loop_vars)
            # The current model is that streaming only applies to accesses that 
            # advance on the innermost axis, with a coefficent of 1 or more.
            # coefficient 1 means continguous access, more means strided access.
            if not coeffs or len(coeffs) < 2 or coeffs[-2] == 0 or coeffs[-1] != 0:
                logging.debug(f"streamify fail, coeffs are {coeffs}")
                return False
            # extents[] should contain integer constants only, disqualify non-constant ones
            # e.g. mxnet yolo3_mobilenet1.0_coco: extents=[6, nkeep[0], 1], coeffs=[1, 6, 0]
            for cnt in extents:
                if not isinstance(cnt, tvm.tir.IntImm):
                    return False
            # if Load/SE, make sure the loop nest has no other accesses to the same var
            if isinstance(op, tvm.tir.BufferLoad) and self.outer_loop:
                for cand in loop_candidates[self.outer_loop]:
                    if cand != self and cand.var == self.var:
                        logging.debug(f"  load disqualified due to other accesses of the same var")
                        return False

            # Reverse the lists: inner-->outer
            coeffs = list(coeffs)[-2::-1]  # drop trailing 0
            extents = extents[::-1]

            # Get element type of buffer var. Initialize a config object
            # to represent the setup.
            buffer_type = buf.type_annotation
            assert isinstance(buffer_type, tvm.ir.PointerType)
            elem_type = buffer_type.element_type
            assert isinstance(elem_type, tvm.ir.PrimType)
            elem_type = elem_type.dtype
            kind = "SE" if isinstance(op, tvm.tir.BufferLoad) else "SA"
            config = SEConfig(kind, elem_type, extents, coeffs)
            #logging.debug(config)

            # If this access is guarded with a vector length condition, 
            # replace the inner icnt of the setup with the guarded length; the
            # guard will be implicitly applied by the stream.
            # TODO: currently assumes 2D guard (could be 1D)
            if self.guard_condition is not None:
                if not config.flatten01:
                    return False
                config.icnts[0] = self.guard_condition.b

            # Make sure the HW can support this stream setup
            if not config.validate():
                return False
            if self.guard_condition != None:
                if self.if_nest[0][0] not in qualified_guards:
                    qualified_guards.append(self.if_nest[0][0])
            self.config = config
            logging.debug(f"qualified: {self}")
            return True

        def allocate(self):
            '''
            Allocate a SE/SA resource to a candidate, considering other
            conflicting candidates.
            '''
            logging.debug(f"allocate: {self}")
            se_resources = ["SE" + str(i) for i in range(0,2)]
            sa_resources = ["SA" + str(i) for i in range(0,4)]
            resources = se_resources if isinstance(self.op, tvm.tir.BufferLoad) \
                        else sa_resources
            # Build a set of resources in use for all overlapping loops
            loop = self.outer_loop
            inuse = set([c.engine for l in loop_overlaps[loop] \
                                  for c in loop_candidates[l]])
            logging.debug(f"  inuse: {inuse}")
            self.engine = next((r for r in resources if r not in inuse), None)
            logging.debug(f"  result: {self}")
            return self.engine is not None

        def assign_config_var(self, idnum):
            ''' Define a config variable and associate it with the stream '''
            name = self.engine[0:2] + ".Config" + str(idnum)
            self.config_var = tvm.tir.Var(name, "handle");

        def open_call(self):
            call = tvm.tir.call_extern("handle", "c7x_stream_open", 
                                       self.config_var, self.engine, self.var)
            return tvm.tir.Evaluate(call)

        def close_call(self):
            call = tvm.tir.call_extern("handle", "c7x_stream_close", 
                                       self.config_var, self.engine);
            return tvm.tir.Evaluate(call)

        def access_call(self):
            '''
            Replace the index expression with a @tir.stream.access call.
            We use a builtin instead of a call_extern to enable vectorization.
            '''
            pred = "nopred" if self.guard_condition is None else "pred"
            adv = "adv"
            index = tvm.tir.Call("int", "tir.c7x.stream_access",
                                 [self.config_var, self.engine,
                                  pred, adv, self.op.indices[0]])
            return index
                
        def __lt__(self, other):
            return self.sort_key < other.sort_key
        def __str__(self):
            nest_ids = [loop_ids[l] for l in self.nest]
            if_ids = [ x[1] for x in self.if_nest ]
            return (f"candidate {self.var}, nest={nest_ids}, if_nest={if_ids}, guard={self.guard_condition} trip={self.sort_key[1]} engine={self.engine} config_var={self.config_var} config=[{self.config}]")

    class SEConfig:
        ''' Helper class to build and manage a single SE/SA configuration '''
        max_dim = 0x10000
        max_dims = 4
        def __init__(self, kind, dtype, extents, coeffs):
            self.MAX_NDIMS = 6
            self.kind = kind
            self.icnts = []
            self.dims = []
            self.veclen = 1
            self.dtype = dtype
            self.flatten01 = False
            for (cnt,dim) in zip(extents, coeffs):
                self.add_axis(cnt, dim)
        def add_axis(self, icnt, dim):
            ''' 
            Add a single axis (dimension) to the configuration. If the 
            current dimension equals the previous dimension's extent 
            (count * dim), flatten the two axes.
            '''
            #logging.debug(f"add_axis: icnt={icnt} dim={dim}")
            ndims = self.ndims()
            if ndims > 0 and \
               dim == self.icnts[ndims-1] * self.dims[ndims-1] and \
               self.icnts[ndims-1] * icnt < self.max_dim:
                self.icnts[ndims-1] *= icnt
                #logging.debug("folded")
                if ndims == 1:
                    self.flatten01 = True
            elif ndims == 0 and dim > 1:
                # For non-contiguous access (dim>1), insert the [1, icnt] and [1, dim]
                # This is because the SE/SA assume dim0 == 1.
                # For contiguous access (dim==1), insert the [icnt] and [1]
                self.icnts.extend([1, icnt])
                self.dims.extend([1, dim])
            else:
                self.icnts.append(icnt)
                self.dims.append(dim)
                #logging.debug("not folded")
        def ndims(self):
            return len(self.dims)
        def validate(self):
            # TODO
            # first dim must be 1
            # no dim >= 0x10000
            if self.ndims() > self.MAX_NDIMS:
                return False
            return True
        def get_config_call(self):
            ''' Return a config call to produce the curent configuration '''
            # pad to MAX_NDIMS axes
            # Updating to support MAX_NDIMS ICNTS and MAX_NDIMS-1 DIMS for SE/SA
            # DIM0 is always set to 1
            if self.ndims() < self.MAX_NDIMS:
               self.icnts = self.icnts + ([1] * (self.MAX_NDIMS-self.ndims()))
               self.dims  = self.dims  + ([0] * (self.MAX_NDIMS-self.ndims()))
            config_call = tvm.tir.call_extern("handle", "c7x_stream_config", 
                                self.kind, self.dtype,
                                *self.icnts[0:self.MAX_NDIMS], *self.dims[1:self.MAX_NDIMS])
            return config_call
        def __str__(self):
            return f"SEConfig: kind={self.kind} dtype={self.dtype} "+\
                   f"veclen={self.veclen} icnts={self.icnts} dims={self.dims}"

    #---------------------------------------------------------------
    def _find_candidates(f):
        ''' Find candidates pass ''' 
        loop_id_counter = 1
        trip = 0
        def _find_candidates_pre(op):
            nonlocal loop_id_counter, veclen_guard, trip
            # don't streamify dma loops
            if isinstance(op, tvm.tir.AttrStmt) and op.attr_key == "pragma_dma":
                return op
            elif isinstance(op, tvm.tir.For):
                loop_nest.append(op)
                loop_ids[op] = loop_id_counter
                loop_id_counter += 1
                loop_overlaps[op] = set()
                loop_candidates[op] = []
                # Keep track of which loops overlap which 
                # Estimate trip count of entire nest
                trip = 1
                for l in loop_nest:
                    loop_overlaps[op].add(l)
                    loop_overlaps[l].add(op)
                    if isinstance(l.extent, tvm.tir.IntImm):
                        trip *= l.extent.value 
                    else:
                        trip *= 10
                veclen_guard = None   # veclen_guard must be inside the innermost loop
            elif isinstance(op, tvm.tir.Allocate):
                def_levels.update({op.buffer_var: len(loop_nest)})
            elif isinstance(op, tvm.tir.Let):
                def_levels.update({op.var: len(loop_nest)})
            # look for vector length guard
            elif isinstance(op, tvm.tir.IfThenElse):
                if_nest.append((op, len(loop_nest)))
                if len(loop_nest) > 0 and loop_nest[-1].kind == tvm.tir.ForKind.VECTORIZED:
                    veclen_guard = _detect_veclen_guard(op)
                    logging.debug(f"_detect_veclen_guard: {_get_if_condition(op)}")
                    logging.debug(f"_detect_veclen_guard result: {veclen_guard}")
                else:
                    logging.debug(f"not veclen_guard: {_get_if_condition(op)}")
            elif isinstance(op, tvm.tir.Call) and op.op.same_as(tvm.ir.Op.get("tir.if_then_else")):
                if_expr_nest.append(op)
            elif (isinstance(op, tvm.tir.BufferStore) or isinstance(op, tvm.tir.BufferLoad)) and \
                 not if_expr_nest:
                level = def_levels[op.buffer.data]
                this_if_nest = [ (x[0], x[1]-level) for x in if_nest if x[1] > level ]
                cand = SECandidate(op, loop_nest[level:], this_if_nest, trip, veclen_guard)
                candidates.append(cand)
                loop = cand.outer_loop
                if loop:
                    loop_candidates[loop].append(cand)
                op_candidates[op] = cand

        def _find_candidates_post(op):
            nonlocal veclen_guard
            if isinstance(op, tvm.tir.For):
                loop_nest.pop()
            elif isinstance(op, tvm.tir.IfThenElse):
                if_nest.pop()
                veclen_guard = None
            elif isinstance(op, tvm.tir.Call) and op.op.same_as(tvm.ir.Op.get("tir.if_then_else")):
                if_expr_nest.pop()

        # body of find candidates pass
        for var in f.params:
            def_levels.update({var : 0})
        for (var,buf) in f.buffer_map.items():
            def_levels.update({buf.data : 0})
        tvm.tir.stmt_functor.ir_transform(
           f.body, _find_candidates_pre, _find_candidates_post)

    #--------------
    # Helpers for find_candidates pass
    def _get_if_condition(op):
        ''' Get condition from if, bypassing @tir.likely if present '''
        if not isinstance(op, tvm.tir.IfThenElse):
            return None
        condition = op.condition
        if isinstance(condition, tvm.tir.Call) and \
           condition.op.same_as(tvm.ir.Op.get("tir.likely")):
            condition = condition.args[0]
        return condition

    #--------------
    def _detect_veclen_guard(op):
        '''
        Detect the 'if' statement that guards the loads and stores in
        an inner loop with a dimension. The canonical form is:

        initial loop:
              for (i, 0, N)
                a[... + i]
        after splitting for vectorization:
              for (outer, 0, ceil(N/K))
                for (inner, 0, K)
                  if (outer*K + inner < N)   <--- "veclen guard"
                    a[... + outer*K + inner]  <--- inner coeffs must match
        '''
        condition = _get_if_condition(op)
        loop_vars = [op.loop_var for op in loop_nest]
        guard_coeffs = tvm.arith.detect_linear_equation(condition.a, loop_vars)
        if op.else_case or \
           len(loop_nest) < 2 or \
           not isinstance(condition, tvm.tir.LT) or \
           not isinstance(condition.b, tvm.tir.IntImm) or \
           not guard_coeffs:
               #logging.debug("detect guard false")
               return None
        # we expect the last 3 coefficients to be K, 1, 0
        guard_coeffs = list(guard_coeffs)[-3:]
        expected = [ loop_nest[-1].extent, 1, 0 ]
        if guard_coeffs == expected:
            return condition
        return None

    #---------------------------------------------------------------
    # deploy pass
    def _deploy_pre(op):
        '''
        Rewrite accesses and insert open/close calls.
        All mutation must happen on the pre-order walk. Mutation changes the
        object references, invalidating links beween IR objects and our
        local data structures, so make any changes on the way down.
        TODO: rewrite the whole SEPass in C++
        '''
        if isinstance(op, tvm.tir.BufferStore):
            cand = op_candidates.get(op)
            if cand and cand.qualified:
                logging.debug(f"streamify store, cand={cand}")
                # ir_transform requires a statement, not an expression, so 
                # we wrap the rhs in an 'Evaluate' statement
                rhs = tvm.tir.Evaluate(op.value)
                rhs = tvm.tir.stmt_functor.ir_transform(
                      rhs, _deploy_pre, None, ["tir.BufferLoad"])
                index = cand.access_call()
                return tvm.tir.BufferStore(op.buffer, rhs.value, [index])
        elif isinstance(op, tvm.tir.BufferLoad):
            cand = op_candidates.get(op)
            if cand and cand.qualified:
                logging.debug(f"streamify load, cand={cand}")
                index = cand.access_call()
                return tvm.tir.BufferLoad(op.buffer, [index])
        elif isinstance(op, tvm.tir.IfThenElse):
            # if we streamified all the accesses, remove the vector 
            # length guard
            if op in qualified_guards and _guard_nullified(op):
               then_case = tvm.tir.stmt_functor.ir_transform(
                      op.then_case, _deploy_pre, None)
               return then_case
        elif isinstance(op, tvm.tir.For):
            if not loop_ids.get(op):
                return op
            logging.debug(f"pre visit For id={loop_ids[op]}")
            body = tvm.tir.stmt_functor.ir_transform(op.body, _deploy_pre, None)
            candidates = [c for c in loop_candidates[op] if c.qualified]
            opens = [c.open_call() for c in candidates]
            closes = [c.close_call() for c in candidates]
            op = tvm.tir.For(op.loop_var, op.min, op.extent, op.kind,
                             body, op.thread_binding, op.annotations)
            return op if (not candidates) else tvm.tir.SeqStmt(opens + [op] + closes)

    #--------------
    def _guard_nullified(op):
        ''' 
        If this if statement function solely as a vector length guard, and
        all accesses within it are streamified, the guard is no longer
        needed (the streaming HW automatically predicates out-of-bounds
        accesses). 
        '''
        guard_condition = _get_if_condition(op)
        # A mini-pass to find all variables within a statement or expression.
        # We ignore variables used in the index of a streamified access,
        # since the indexing is performed directly by the HW.
        def _find_vars(vars_list, op):
            if isinstance(op, tvm.tir.Var):
                vars_list.add(op)
            elif isinstance(op, tvm.tir.BufferLoad):
                cand = op_candidates.get(op)
                if cand and cand.qualified and \
                   cand.guard_condition == guard_condition:
                    logging.debug("skip load index")
                    return op
            elif isinstance(op, tvm.tir.BufferStore):
                cand = op_candidates.get(op)
                if cand and cand.qualified and \
                   cand.guard_condition == guard_condition:
                    # turn RHS into statement, for recusive ir_transform
                    rhs = tvm.tir.Evaluate(op.value) 
                    tvm.tir.stmt_functor.ir_transform(
                        rhs, partial(_find_vars, vars_list), None)
                    logging.debug("skip store index")
                    return op
        # Find variables used in the condition
        guard_vars = set()
        stmt = tvm.tir.Evaluate(guard_condition)
        tvm.tir.stmt_functor.ir_transform(
                          stmt, partial(_find_vars, guard_vars), None)
        # Find variables used in the then clause
        then_vars = set()
        tvm.tir.stmt_functor.ir_transform(
                          op.then_case, partial(_find_vars, then_vars), None)
        # If none of the variables in the condition are referenced in the 
        # then clause (exluding streamified accesses), we assume the guard
        # is defunct.
        isect = guard_vars.intersection(then_vars)
        logging.debug(f"condition: {guard_condition} guard_vars={guard_vars} then_vars={then_vars} isect={isect}")
        return len(guard_vars) != 0 and len(isect) == 0

    #---------------------------------------------------------------
    # main body of SETransform

    # Find candidates
    _find_candidates(f)

    # Prioritize and qualify
    candidates.sort(reverse=True)
    idnum = 0
    for cand in candidates:
        if cand.qualify() and cand.allocate():
            cand.assign_config_var(idnum)
            cand.qualified = True
            idnum += 1

    logging.debug(f"qualified veclen_guards: {qualified_guards}")
    # For qualified candidates, rewrite accesses 
    stmt = tvm.tir.stmt_functor.ir_transform(f.body, _deploy_pre, None)

    # Collect stream config calls and emit at top of function
    stmts = []   # hoisted statements
    valid_candidates = [c for c in candidates if c.config_var is not None]
    for cand in valid_candidates:
        let_stmt = tvm.tir.LetStmt(cand.config_var, 
                                   cand.config.get_config_call(),
                                   tvm.tir.Evaluate(1))   # dummy body
        stmts.append(let_stmt)
    stmt = merge_block(stmts, stmt)
    return f.with_body(stmt)

#-----------
def C7xSEPass():
    ''' create the C7x Streaming pass '''
    return tvm.tir.transform.prim_func_pass(
        SETransform, opt_level=0, name="tir.c7x.C7xSEPass"
    )

# C7X DMA can only support dma transfers up to 4 dimensions
MAX_C7X_DMA_DIMS = 4

#----------------------------------------------------------------
# transferring a single block
def c7x_dma_add_inner_dims(axis, elem_bytes, loop_bounds, src_strides, dst_strides,
                           src_dma_icnts, src_dma_strides, dst_dma_icnts, dst_dma_strides):
    num_loops = len(loop_bounds)
    # from innermost loop to outermost loop
    for i in range(num_loops-1, -1, -1):
      #   if previous (inner) loop count * stride == current loop stride,
      #   then current loop can be merged into previous loop transfer,
      #   otherwise, we have to increase axis and use a separate stride.
      #   The first inner loop needs a new axis too (no previous axis).
      if (i == num_loops-1 or
          src_strides[i] != src_strides[i+1] * loop_bounds[i+1] or
          dst_strides[i] != dst_strides[i+1] * loop_bounds[i+1]):
        axis += 1
        if (axis >= MAX_C7X_DMA_DIMS):
            return axis
        # for fisrt dim/axis, elem_bytes is encoded in the ICNT[0] because STRIDE[0] is not used
        src_dma_icnts[axis]   = loop_bounds[i] * (1 if axis > 0 else elem_bytes)
        dst_dma_icnts[axis]   = loop_bounds[i] * (1 if axis > 0 else elem_bytes)
        src_dma_strides[axis] = src_strides[i] * (elem_bytes if axis > 0 else 1)
        dst_dma_strides[axis] = dst_strides[i] * (elem_bytes if axis > 0 else 1)
      else:
        src_dma_icnts[axis]   *= loop_bounds[i]
        dst_dma_icnts[axis]   *= loop_bounds[i]
    return axis

#----------------------------------------------------------------
# transferring blocks between double-bufferred on-chip buffer and off-chip buffer
def c7x_dma_add_outer_dims(axis, elem_bytes, num_blocks, loop_bounds, off_strides,
                           on_dma_icnts, on_dma_strides, off_dma_icnts, off_dma_strides):
    num_loops = len(loop_bounds)
    on_dma_icnts[axis+1]   = 2    # double buffering
    on_dma_strides[axis+1] = (on_dma_icnts[axis] * on_dma_strides[axis])
    # when num_blocks is odd, use [2, (num_blocks+1)/2] for the on_chip ICNTs, the off_chip
    # ICNTs are still configured to be num_blocks to ensure correct number of transfers.
    on_dma_icnts[axis+2]   = (num_blocks + 1) // 2

    # from innermost loop to outermost loop
    for i in range(num_loops-1, -1, -1):
        #   if previous (inner) loop count * stride == current loop stride,
        #   then current loop can be merged into previous loop transfer,
        #   otherwise, we have to increase axis and use a separate stride.
        #   The first outer loop needs a new axis too (should not merge with inner block).
        if (i != num_loops-1 and off_strides[i] == loop_bounds[i+1] * off_strides[i+1]):
            off_dma_icnts[axis] *= loop_bounds[i]
        else:
            axis += 1
            if (axis >= MAX_C7X_DMA_DIMS):
                return axis
            off_dma_icnts[axis]   = loop_bounds[i]
            off_dma_strides[axis] = off_strides[i] * elem_bytes
    return axis

#----------------------------------------------------------------
# "Injector" for dma intrinsic, called by inject_copy_intrin.cc
#
# Turning a loop nest copying between src and dst into a C7x dma intrinsic,
# passing information to downstream passes (C7xDMAPass, CodeGenC7x)
def c7x_dma_injector(src : tvm.tir.Buffer, 
                     dst : tvm.tir.Buffer, pad_before, pad_after, pad_value):
    '''
    This function is called by the InjectCopyIntrin pass to insert code 
    to replace the copy-in/copy-out code for a local buffer
    '''
    num_loops = len(dst.shape)
    # if non-constant shape or strides, cannot dma. Returning None will keep original copying code
    if any(not isinstance(x, tvm.tir.IntImm) for x in [*dst.shape, *src.strides, *dst.strides]):
      logging.debug(f"DMA disqualified for non-constant shape: {dst.shape} or strides: {src.strides}, {dst.strides}")
      return None

    assert('global' in src.scope() or 'global' in dst.scope())
    assert('local' in src.scope() or 'local' in dst.scope())
    copy_in = 'global' in src.scope() and 'local' in dst.scope()

    # compute the dma icnts and strides for the inner loops (this copying loop nest)
    elem_bytes = tvm.runtime.DataType(src.dtype).bits // 8
    loop_bounds = [ x.value for x in dst.shape ]
    src_strides = [ x.value for x in src.strides ]
    dst_strides = [ x.value for x in dst.strides ]

    src_dma_icnts   = [1, 1, 1, 1]
    src_dma_strides = [0, 0, 0, 0]   # 0th entry not used by C7x DMA
    dst_dma_icnts   = [1, 1, 1, 1]
    dst_dma_strides = [0, 0, 0, 0]   # 0th entry not used by C7x DMA

    sync_axis = c7x_dma_add_inner_dims(-1, elem_bytes, loop_bounds, src_strides, dst_strides,
                                        src_dma_icnts, src_dma_strides,
                                        dst_dma_icnts, dst_dma_strides)

    # C7x DMA can only support 4 dimensions, limit block sync to dim/axis 0 or 1,
    #   because dim/axis 2 and 3 will be used for double buffering [2, num_blocks/2]
    if (sync_axis > 1):
      logging.debug(f"DMA disqualified for sync_axis={sync_axis} (>1), cannot double buffer")
      return None

    expr = tvm.tir.call_extern("int32", "c7x_dma_copy", src.data, dst.data,
                               elem_bytes, sync_axis, src.elem_offset, dst.elem_offset,
                               *src_dma_icnts, *src_dma_strides,
                               *dst_dma_icnts, *dst_dma_strides, copy_in)
    stmt = tvm.tir.Evaluate(expr)
    return stmt

#----------------------------------------------------------------
def C7xDMAPass():
    """
    This is a C7x-specific pass that detects c7x_dma_copy calls and inserts 
    additional intrinsics to configure the DMA.
    """
    return tvm.tir.transform.prim_func_pass(
        C7xDMATransform, opt_level=0, name="tir.c7x.C7xDMAPass"
    )

def C7xDMATransform(f, mod, ctx):
    """
    Implementation of C7xDMAPass
    """
    # map from vars to allocation statements
    local_var_info = {}
    # list of vars involved in c7x_dma_calls
    dma_buffers = []
    # list of c7x_dma_calls
    dma_copy_calls = []
    # map from "c7x_dma_copy" op to list of outer loops
    outer_loops_info = {}
    # Stack of ForStmt visited
    loop_nest = []
    # map from "c7x_dma_copy" op to created dma handle
    dma_copy_handle_map = {}

    # pre-order walk: keep track of variables and operations involved 
    # in dma copies
    def _dma_pre(op):
        builtin_call_extern = tvm.ir.Op.get("tir.call_extern")
        if isinstance(op, tvm.tir.Allocate):
            local_var_info[op.buffer_var] = { 'alloc' : op }
        elif isinstance(op, tvm.tir.For):
            loop_nest.append(op)
        elif op.op.same_as(builtin_call_extern) and \
             op.args[0].value == "c7x_dma_copy":
            dma_copy_calls.append(op)
            dma_buffers.extend([op.args[1], op.args[2]])
            outer_loops_info[op] = loop_nest.copy()

    # post-order walk: remove allocation statements from inner loop;
    # they will be re-generated at outer loop level
    def _dma_post(op):
        if isinstance(op, tvm.tir.Allocate):
            if op.buffer_var in dma_buffers:
                return op.body
        elif isinstance(op, tvm.tir.For):
            loop_nest.pop()

    # helper function to determine dimensions of dma var
    def _get_dims(var):
        dims = None
        # if local buffer, use allocation statement
        if var in local_var_info:
            if 'alloc' in local_var_info[var]:
                alloc = local_var_info[var]['alloc']
                dims = [dim for dim in alloc.extents]
        # if function parameter, use buffer object from buffer map
        else:
            for (_,buf) in f.buffer_map.items():
                if buf.data == var:
                    dims = [dim for dim in buf.shape]
        assert dims
        # extend to 4 dims
        if len(dims) < 4:
            dims = ([1] * (4-len(dims))) + dims
        return dims

    # post-order walk: append dma handle to dma copy intrinsics in tir,
    #   so that dma handle does not get optimized away by later passes (e.g. tir.RemoveNoOp)
    def _dma_handle_post(op):
        if op in dma_copy_handle_map:
           new_args = op.args[1:] + [dma_copy_handle_map[op]]
           return tvm.tir.call_extern("int32", "c7x_dma_copy", *new_args, span=op.span)

    # Run the pre/post passes above
    stmt = tvm.tir.stmt_functor.ir_transform(
        f.body, _dma_pre, _dma_post,
        ["tir.Allocate", "tir.Call", "tir.For"])

    stmts = []   # hoisted statements
    for dma_call in dma_copy_calls:
        # get dims for src and dst variables
        src = dma_call.args[1]
        dst = dma_call.args[2]
        src_dims = _get_dims(src)
        dst_dims = _get_dims(dst)
        # create a variable for the dma object
        for var in (src,dst):
            if var in local_var_info:
                dma_name = var.name.split('.')[0] + ".dma"
        dma = tvm.tir.Var(dma_name, "handle")
        dma_copy_handle_map[dma_call] = dma
        # hoist the alloc statements for local buffers
        for var in (src,dst):
            if var in local_var_info:
                alloc = local_var_info[var]['alloc']
                stmts.append(alloc)

        # let dma_object = c7x_dma_setup(src, dim3, dim2, dim1, dim0,
        #                                dst, dim3, dim2, dim1, dim0,
        #                                num_blocks, sync_axis, src_offset, dst_offset,
        #                                src_icnts[4], src_strides[3], dst_icnts[4], dst_strides[3])
        elem_bytes = dma_call.args[3].value
        sync_axis  = dma_call.args[4].value
        src_inner_offset = dma_call.args[5]
        dst_inner_offset = dma_call.args[6]
        src_dma_icnts   = dma_call.args[7:11]
        src_dma_strides = dma_call.args[11:15]
        dst_dma_icnts   = dma_call.args[15:19]
        dst_dma_strides = dma_call.args[19:23]
        copy_in = dma_call.args[23]
        

        outer_loops = outer_loops_info[dma_call]
        outer_loops_vars = [x.loop_var for x in outer_loops]
        num_outer_loops = len(outer_loops)
        outer_loop_bounds = [(x.extent - x.min) for x in outer_loops]
        src_outer_strides = tvm.arith.detect_linear_equation(src_inner_offset, outer_loops_vars)
        dst_outer_strides = tvm.arith.detect_linear_equation(dst_inner_offset, outer_loops_vars)
        # Can only handle constant integer bounds and strides for outer loops, for now
        assert all(isinstance(x, tvm.tir.IntImm)
                   for x in [*outer_loop_bounds, *src_outer_strides, *dst_outer_strides])

        # args: num_blocks, OFFSET[2]
        num_blocks = 1
        for bound in outer_loop_bounds:
            num_blocks *= bound
        src_dma_offset = src_outer_strides[num_outer_loops] * elem_bytes
        dst_dma_offset = dst_outer_strides[num_outer_loops] * elem_bytes

        # Add outer loops, do double buffering
        if (num_outer_loops > 0):
            if copy_in: #copying in, dst is on-chip
                axis = c7x_dma_add_outer_dims(sync_axis, elem_bytes, num_blocks,
                                outer_loop_bounds, src_outer_strides,
                                dst_dma_icnts, dst_dma_strides, src_dma_icnts, src_dma_strides)
            else:                                            # copying out, src is on-chip
                axis = c7x_dma_add_outer_dims(sync_axis, elem_bytes, num_blocks,
                                outer_loop_bounds, dst_outer_strides,
                                src_dma_icnts, src_dma_strides, dst_dma_icnts, dst_dma_strides)
            assert (axis < MAX_C7X_DMA_DIMS)
        logging.debug(f"dma params: blocks={num_blocks} sync_axis={sync_axis} src_offset={src_dma_offset} dst_offset={dst_dma_offset}")
        logging.debug(f"            src_icnts={src_dma_icnts}, src_strides={src_dma_strides}, dst_icnts={dst_dma_icnts}, dst_strides={dst_dma_strides}")

        setup = tvm.tir.call_extern("handle", 
                      "c7x_dma_setup", src, *src_dims, dst, *dst_dims,
                      num_blocks, sync_axis, src_dma_offset, dst_dma_offset,
                      *src_dma_icnts, *src_dma_strides[1:], *dst_dma_icnts, *dst_dma_strides[1:])
        setup = tvm.tir.LetStmt(dma, setup, tvm.tir.Evaluate(1))   # dummy body
        stmts.append(setup)

    # Run the pre/post pass, append dma handle to dma copy intrinsics
    stmt = tvm.tir.stmt_functor.ir_transform(stmt, None, _dma_handle_post, ["tir.Call"])

    # hoist alloc, and dma setup to top of function body
    stmt = merge_block(stmts, stmt)
    return f.with_body(stmt)

