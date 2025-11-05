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
"""TI Offload backend compiler"""

import os
import sys
import subprocess
import shutil
import ctypes
import _ctypes
import re
import functools
import json
import numpy as np
import logging
import importlib
import tvm
from tvm import relay
import tvm.ir
from tvm.topi.utils import get_const_tuple
from tvm.relay.dataflow_pattern import is_op, is_constant, wildcard, is_tuple, FunctionPattern
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import Tuple, GlobalVar
from tvm.relay.function import Function
from tvm.contrib import graph_executor
from tvm.contrib.tidl.c7x import supported_platform, platform_map
#import tvm.relay.op.contrib.tidl as tidl_annotation
from .visualize import visualize_relay_graph
from .build_c7x_mod import enable_c7x_mod
from .prepare import prepare_graph_for_partitioning
from .prepare import prune_graph_for_ODPostProc_inputs

tidl_annotations_registered = False

# Macros
TIDL_DIM_MAX = 6

import tvm._ffi

from . import _ffi_tidl_api

def get_all_nodes(mod_func):
    # Traverse Relay IR graph and generate a dictionary of all nodes except tvm.ir.Op nodes)
    # mod_func is expected to be a function in Relay IR Module
    def traverse_expr(node, node_dict):
        if node in node_dict:
            return
        if isinstance(node, tvm.ir.Op):
            return
        node_dict[node] = len(node_dict)
    all_nodes_main = {}
    traverse_func = functools.partial(traverse_expr, node_dict=all_nodes_main)
    relay.analysis.post_order_visit(mod_func, traverse_func)
    return all_nodes_main


class SkipLocalFunctionsVisitor(ExprMutator):
    def __init__(self, target):
        super().__init__()
        self.tidl_target = target
        self.nodes = {}
        self.span_name_counts = {}  # Track count of each span name

    def visit_function(self, fn):
        if(hasattr(fn, "attrs") and "Composite" in fn.attrs and self.tidl_target in fn.attrs["Composite"]):
            return
        return super().visit_function(fn)

    def visit(self, expr):
        # Catch-all record
        if expr not in self.nodes:
            if isinstance(expr, relay.expr.Call) and hasattr(expr, 'span') and expr.span and hasattr(expr.span, 'source_name'):
                base_name = expr.span.source_name.name
                # Check if this span name already exists
                if base_name in self.span_name_counts:
                    # Increment counter and append to make unique
                    self.span_name_counts[base_name] += 1
                    unique_name = f"{base_name}_{self.span_name_counts[base_name]}"
                else:
                    # First occurrence, no suffix needed
                    self.span_name_counts[base_name] = 0
                    unique_name = base_name
                self.nodes[expr] = unique_name
            else:
                self.nodes[expr] = len(self.nodes)
        super().visit(expr)

def find_data_layout(mod):
    all_nodes = get_all_nodes(mod['main'])
    data_layout = "NCHW"
    for node in all_nodes:
        if isinstance(node, relay.expr.Call):
            if not node.attrs:
                continue
            if node.op.name == 'nn.conv2d' or node.op.name == 'qnn.conv2d':
                data_layout = node.attrs.data_layout
                break
            else:
                try:
                    data_layout = node.attrs.layout
                    break
                except:
                    pass
    return data_layout

def find_qnn_ops(mod):
    all_nodes = get_all_nodes(mod['main'])
    return any(isinstance(node, relay.expr.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name.startswith('qnn.')
               for node in all_nodes)

def get_tidl_subgraphs(mod, tidl_target):
    # Traverse Relay IR graph and generate a dictionary of all TIDL subgraphs
    all_nodes_main = get_all_nodes(mod['main'])
    tidl_subgraphs = []
    for node in all_nodes_main:
        if isinstance(node, relay.expr.GlobalVar):
            if tidl_target in node.name_hint:
                tidl_subgraphs.append(node.name_hint)
    return tidl_subgraphs

# borrowed from python/tvm/relay/op/contrib/tensorrt.py, modified to check all dimensions
def check_dynamism(args, op_name):
    """
    Check for dynamism inside any of the args in the op.

    Parameters
    ----------
    args : tvm.ir.container.Array
        Arguments of the op. Each of the argument shape is checked for presence of dynamic
        components.
    op_name: str
        Name of the op for debugging purposes only.
    Returns
    ----------
    ret : bool
        True if dynamism is present, False otherwise
    """
    for arg in args:
        if isinstance(arg, (relay.expr.Call, relay.expr.Var, relay.expr.Constant,
                            relay.expr.TupleGetItem)):
            for dim_shape in arg.checked_type.shape:
                if isinstance(dim_shape, tvm.tir.expr.Any):
                    return True
        elif isinstance(arg, Tuple):
            return check_dynamism(arg.fields, op_name)
        else:
            print(f"Arg not supported in TIDL for {op_name} with type {type(arg)}")
            return True
    return False

def find_dynamic_shape(mod):
    all_nodes = get_all_nodes(mod['main'])
    return any(isinstance(node, relay.expr.Call) and isinstance(node.op, tvm.ir.Op) and check_dynamism(node.args, node.op.name)
               for node in all_nodes)

def get_default_quantization():
    return np.array(0, dtype=np.int32), np.array(1.0, dtype=np.float32)

def convert_str_list_to_char_array(str_list):
    """ Convert list of strings to array of ctypes char * """
    char_array = (ctypes.c_char_p * len(str_list))()
    for i in range(len(str_list)):
        char_array[i] = bytes(str_list[i], 'utf-8')
    return char_array

def find_in_nodes(all_nodes, this_node, input_prefix):
    r""" Find the input nodes of a given relay.expr.Call node.

         Only find input nodes that are relay.expr.Call.
         If an input node is a relay.expr.TupleGetItem, then check this input
         node's input node.

    Parameters
    ----------
    all_nodes : dictionary
        Dictionary of all nodes of the graph: keys are nodes and values are node indices
    this_node : relay.expr.Call
        A relay.expr.Call node whose input nodes are to be found by this function
    input_prefix : string
        Prefix of input tensor name, e.g. "tidl" when target is "tidl"

    Returns
    -------
    input_nodes : list
        A list of all input nodes' names of the given node. For call node, the name is the node
        index in all_nodes dictionary. For input tensors, the name is the tensor's name.
    """
    def _get_result(node):
        """ Return the name or names of the tensor produced by a node as a flattened list.
            Tuples are "flattened" so that each result in the list represents a single tensor.
        """
        result = []
        # N is Call w/one output --> result is "N"
        # N is Call w/tuple output --> result is ["N", "N:1", ...]
        if isinstance(node, relay.expr.Call):
            node_name = str(all_nodes[node])
            result.append(node_name)
            if isinstance(node.checked_type, tvm.ir.TupleType):
               for i in range(1, len(node.checked_type.fields)):
                  result.append(node_name+f":{i}")
        # N is TuplegetItem(Tuple,i) --> result is i'th result of Tuple
        elif isinstance(node, relay.expr.TupleGetItem):
            tuple_val = _get_result(node.tuple_value)
            result.append(tuple_val[node.index])
        # N is Tuple(N1,N2,...)  --> result is [result(N1), result(N2), ...]
        elif isinstance(node, relay.expr.Tuple):
            for field in node.fields:
                result.extend(_get_result(field))
        # N is Var --> result is "Var"
        elif isinstance(node, relay.expr.Var): # input tensor is relay.expr.Var
            if input_prefix in node.name_hint and "_i" in node.name_hint:
                # this is an input tensor to the subgraph
                result.append(node.name_hint)
        #else: ignore all other types of nodes: const, etc.
        return result

    # Use the helper function to gather results of this node's inputs
    input_names = []
    if isinstance(this_node, relay.expr.Call):
        in_nodes = this_node.args
    elif isinstance(this_node, relay.expr.Tuple):
        in_nodes = this_node.fields
    for node in in_nodes:
        input_names.extend(_get_result(node))
    return input_names

def find_out_nodes(all_nodes, this_node, field_index=-1):
    r""" Find the output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary
        Dictionary of all nodes of the graph: keys are nodes and values are node indices
    this_node : relay.expr.Call
        A relay.expr.Call node whose output nodes are to be found by this function

    Returns
    -------
    output_nodes : list
        A list of all output node indices of the given node
    """

    output_nodes = []
    for node, node_idx in all_nodes.items():
        if isinstance(node, relay.expr.Call):
            # Count multiple times for uses: e.g. tflite_nasnet_mobile, %110 = add(%109, %109)
            for node_arg in node.args:
                if this_node == node_arg:
                    output_nodes.append(str(node_idx))
        elif isinstance(node, relay.expr.TupleGetItem):
            if this_node == node.tuple_value and (field_index == -1 or field_index == node.index):
                output_nodes = output_nodes + find_out_nodes(all_nodes, node)
        elif isinstance(node, relay.expr.Tuple):
            # Count multiple times for uses: e.g. onnx_yolov5s_opset11, %100 used 3 times by %102
            #   %100 = nn.max_pool2d(%99, pool_size=[3, 3], padding=[1, 1, 1, 1])
            #   %101 = (%99, %100, %100, %100);
            #   %102 = concatenate(%101, axis=1) /* ty=Tensor[(1, 1024, 20, 20), float32] */;
            for i, field_i in enumerate(node.fields):
                if this_node == field_i:
                    tuple_node_outs = find_out_nodes(all_nodes, node, i)
                    if len(tuple_node_outs) == 0:
                        # this is an output node
                        output_nodes.append(str(all_nodes[node]))
                    else:
                        # this is an input node to another node
                        output_nodes = output_nodes + tuple_node_outs

    return output_nodes

def add_prefix(nodes, prefix):
    r""" Add tidl_subgraph name prefix if the node name does not already has the prefix"""
    r""" e.g. 69 -> tidl_0_69,  tidl_0_i0 -> tidl_0_i0,  tidl_0_o0 -> tidl_0_o0"""
    return [ (node if node.startswith(prefix) else (prefix + '_' + node)) for node in nodes ]

def find_in_out_nodes(all_nodes, this_node, input_prefix, output_names):
    r""" Find the input and output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary
        Dictionary of all relay.expr.Call nodes of the graph
    this_node : relay.expr.Call
        A relay.expr.Call node whose input and output nodes are to be found
    input_prefix : string
        Prefix of input tensor name, e.g. "tidl" when target is "tidl"
    output_names : list
        List of output names of current subgraph

    Returns
    -------
    in_out_nodes : InOutNodes
        Structure that stores names (encoded indices) of input nodes and output nodes
        All names are prefixed with tidl_subgraph name, so that we can differentiate them from
        different subgraphs, e.g. when specified in output_feature_16bit_names_list in TIOffloadCompiler
    """

    in_out_nodes = InOutNodes()    # instantiate structure
    in_out_nodes.this_node = bytes(str(all_nodes[this_node]), 'utf-8')
    in_nodes = find_in_nodes(all_nodes, this_node, input_prefix) # node indices of input nodes
    if len(in_nodes) == 0:
        in_out_nodes.in_nodes = None
    else:
        # convert list to char * array in order to pass to C library
        in_nodes_char = convert_str_list_to_char_array(in_nodes)
        in_out_nodes.in_nodes = ctypes.cast(in_nodes_char, ctypes.c_void_p)

    in_out_nodes.num_in_nodes = len(in_nodes)

    out_nodes = find_out_nodes(all_nodes, this_node) # node indices of output nodes
    if len(out_nodes) == 0:
        # This is the last node, use the output tensor name as this node's name
        # When the last node is a call node, it can have only one output tensor.
        in_out_nodes.this_node = bytes(str(output_names[0]), 'utf-8')
        in_out_nodes.out_nodes = None # this is the last node
    else:
        # convert list to char * array in order to pass to C library
        out_nodes_array = convert_str_list_to_char_array(out_nodes)
        in_out_nodes.out_nodes = ctypes.cast(out_nodes_array, ctypes.c_void_p)

    in_out_nodes.num_out_nodes = len(out_nodes)

    return in_out_nodes

def obtain_subgraph_tensor(subgraph_tensors_list, tensor_name_prefix):
    r""" Obtain input/output tensor for a given subgraph"""

    tensors_list = []
    names_list = []
    for subgraph_tensors in subgraph_tensors_list:
        tensors = []
        names = []
        for key, value in subgraph_tensors.items():
            if key.find(tensor_name_prefix) != -1:
                tensors.append(value)
                names.append(key)
        tensors_list.append(tensors)
        names_list.append(names)

    return tensors_list, names_list

def obtain_tensor_quantization(names, relay_quantization):
    """ Obtain quantization for the named tensors, only per-tensor quantization is supported!
    """

    zp_list = []
    scale_inv_list = []
    for name in names:
        if name in relay_quantization:
            zp, scale = relay_quantization[name]
        else:
            zp, scale = get_default_quantization()
        assert zp.size == 1 and scale.size == 1, \
               f'Per-channel quantization on boundary input/output tensors not supported'
        zp_list.append(zp.item())
        scale_inv_list.append(1.0 / scale.item())  ### TIDL uses inverse of TVM/Relay scale

    return zp_list, scale_inv_list

def obtain_tensor_etype(names, relay_etypes):
    """ Obtain TIDL_ElementType for the named tensors
    """

    etype_list = []
    for name in names:
        assert name in relay_etypes, f"Unknow TIDL_ElementType for {name}"
        etype_list.append(relay_etypes[name])
    return etype_list

def obtain_inout_quant_dict(subgraph, subgraph_id, relay_quantization):
    r"""Populate input/output expr to quant dict with info from relay_quantization"""

    def find_insert_quant(name, expr, inout_quant_dict):
        if name in relay_quantization:
            inout_quant_dict[expr] = relay_quantization[name]

    inout_quant_dict = {}
    for i, param in enumerate(subgraph.params):
        find_insert_quant(f'tidl_{subgraph_id}_i{i}', param, inout_quant_dict)

    if isinstance(subgraph.body, relay.expr.Tuple):
        for i, expr in enumerate(subgraph.body.fields):
            find_insert_quant(f'tidl_{subgraph_id}_o{i}', expr, inout_quant_dict)
    else:
        find_insert_quant(f'tidl_{subgraph_id}_o0', subgraph.body, inout_quant_dict)

    return inout_quant_dict

def tensor_quant_flatten(input_tensors_list, data_layout, tensor_bits):
    r""" Convert float32 n-d array to int8/int16 or uint8/uint16 1-d array

    Parameters
    ----------
    input_tensor_list: list of float32 array, one calibration image/data per list element
    data_layout: "NCHW" or "NHWC"
    tensor_bits: 8 or 16
    Returns
    -------
    quant_tensors_list: each list element contains quantized tensors for one calibration image/data
    quant_scales: quant_scales for (multiple) subgraph inputs across all calibration images/data
    quant_signs: signs for (multiple) subgraph inputs across all calibration images/data
    """

    # find min, max for each subgraph input across all calibration images/data
    min_values = []
    max_values = []
    for i in range(len(input_tensors_list[0])):
        min_values_i = []
        max_values_i = []
        for input_tensors in input_tensors_list:
            # Handle tensors of different dimensions
            if len(input_tensors[i].shape) == 1:
                # For 1D tensors, use the entire tensor
                min_values_i.append(np.amin(input_tensors[i]))
                max_values_i.append(np.amax(input_tensors[i]))
            else:
                # For multi-dimensional tensors, use only the first batch
                min_values_i.append(np.amin(input_tensors[i][0, :]))
                max_values_i.append(np.amax(input_tensors[i][0, :]))

        min_values.append(min(min_values_i))
        max_values.append(max(max_values_i))

    # compute quant_scales, input_signs, quant_mins/quant_maxs for each subgraph input
    quant_scales = []
    input_signs = []
    quant_mins = []
    quant_maxs = []
    for i in range(len(input_tensors_list[0])):
        max_value = max(abs(min_values[i]), max_values[i])
        if max_value == 0:
            max_value = 1.0  # arbitrary number if input tensor is all 0's
        '''
        ideally a 16-bit scale should be 32768 / 65535, but that leads
        to bias clipping in certain 16-bit models which have layers with
        a very big difference between weightScale and biasScale

        We use a 2-bit reduced bit depth to prevent bias saturation on
        these models
        '''
        abs_signed_max   = 128.0 if (tensor_bits == 8) else 32768.0
        abs_unsigned_max = 255.0 if (tensor_bits == 8) else 65535.0

        scale_signed_max = 32.0 if tensor_bits == 8 else 8192.0
        scale_unsigned_max = 64.0 if tensor_bits == 8 else 16384.0

        if min_values[i] >= 0:
            # quantize to Uint8 or Uint16
            sign = 0
            scale = min(abs_unsigned_max/max_value, scale_unsigned_max)
            quant_min, quant_max = 0.0, abs_unsigned_max
        else:
            # quantize to Int8 or Int16
            sign = 1
            scale = min(abs_signed_max/max_value, scale_signed_max)
            quant_min, quant_max = (- abs_signed_max), (abs_signed_max - 1.0)

        if tensor_bits == 32:
            quant_scales.append(1.0)
        else:
            quant_scales.append(scale)
        input_signs.append(sign)
        quant_mins.append(quant_min)
        quant_maxs.append(quant_max)

    # quantize all calibration images/data
    quant_tensors_list = []
    for input_tensors in input_tensors_list:
        quant_tensors = []
        for input_tensor, scale, sign, quant_min, quant_max in zip(input_tensors, quant_scales,
                                                           input_signs, quant_mins, quant_maxs):
            # # only use 1 batch for calibration
            # input_tensor = input_tensor[0, :]
            # change layout to CxHxW to use numpy.flatten to change to 1-d array
            if data_layout == "NHWC" and len(input_tensor.shape) == 4:
                input_tensor = input_tensor.transpose(0, 3, 1, 2)

            # No more quant.  Keep TVM tensor as is, Use TIDL data convert layers
            #tensor_norm = np.multiply(input_tensor, scale)
            #tensor_quant = np.rint(tensor_norm)
            #tensor_quant = np.clip(tensor_quant, quant_min, quant_max)
            #output = tensor_quant.flatten()   # works only if tensor_quant is in "CxHxW" format
            output = input_tensor.flatten()

            quant_tensors.append(output)
        quant_tensors_list.append(quant_tensors)

    return quant_tensors_list, quant_scales, input_signs

class VarReplacer(ExprMutator):
    """
    Replaces vars in expr according to var_map.
    """
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

def unpack_composites(mod, target, global_vars_list):
    """Unpack all composite functions in the module by replacing composite call nodes with the
    ops inside the composite function."""

    class Unpacker(ExprMutator):
        """Unpacks composite functions."""
        def __init__(self, target):
            self.target = target
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, Function):
                if call.op.attrs and call.op.attrs['Composite'] != "" and self.target in call.op.attrs['Composite']:
                    # unpack the function back into new main function.
                    var_map = {}
                    for arg, param in zip(call.args, call.op.params):
                        var_map[param] = super().visit(arg)
                    return VarReplacer(var_map).visit(call.op.body)
            return super().visit_call(call)

    for var in global_vars_list:
        mod[var] = Unpacker(target).visit(mod[var])
    return mod

def unpack_specific_composites(mod, op_name, span_name):
    """Unpack a specific composite function in the module by replacing composite call node with the
    ops inside the composite function."""
    class Unpacker(ExprMutator):
        """Unpacks composite functions."""
        def __init__(self):
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, Function):
                if call.op.attrs and call.op.attrs['Composite'] == op_name and call.span.source_name.name == span_name:
                    # unpack the function back into new main function.
                    var_map = {}
                    for arg, param in zip(call.args, call.op.params):
                        var_map[param] = super().visit(arg)
                    super().visit_call(call)
                    return VarReplacer(var_map).visit(call.op.body)
            return super().visit_call(call)

    for func in mod.get_global_vars():
        mod[func.name_hint] = Unpacker().visit(mod[func.name_hint])
    return mod

def flatten_tuple_params(mod, compiler):
    """ TIDL can't handle passing Tuples as arguments to a subgraph. This pass
        flattens them into their constituent components.

        The declaration is rewritten as follows:
            def %tidl_0(%tidl_0_i0: (<typeA>, <typeB>),   /* tuple */
                        %tidl_0_i1: <typeC>) {            /* tensor */
               ... use %tidl_0_i0 ...
        ==>
            fn (%tidl_0_i0: <typeA>,                      /* tensor */
                %tidl_0_i1: <typeB>,                      /* tensor */
                %tidl_0_i2: <typeC>) {                    /* tensor */
               %newTuple = (%tidl_0_i0, %tidl_0_i1)
               ... use %newTuple ...

        The call is rewritten as follows:
            %z = @tidl_0(%t, %s)               /* tuple, tensor */
        ==>
            %z = @tidl_0(%t.0, %t.1, %s)       /* tuple, tensor, tensor */
    """
    def flatten_tuple_declaration(func_name):
        """ Rewrite a function with tuple parameters """
        new_params = []   # list of new param vars
        var_map = {}      # maps old param list to new
        func = mod[func_name]

        def _addparm(ptype):
            name = f'{func_name}_i{len(new_params)}'
            var = tvm.relay.var(name, type_annotation=ptype)
            new_params.append(var)
            return var

        for var in func.params:
            if isinstance(var.checked_type, tvm.ir.TupleType):
                # gather tuple subparams, and add func decl
                tuple_parms = []
                for t in var.checked_type.fields:
                    tuple_parms.append(_addparm(t))
                # create new tuple from tuple subparams, and enqueue for
                # rewriting (uses of old tuple param replaced with new tuple)
                new_tuple = relay.expr.Tuple(tuple_parms)
                var_map[var] = new_tuple
            else:
                new_parm = _addparm(var.checked_type)
                var_map[var] = new_parm

        # apply enqueued var replacements, and re-construct the function
        new_body = VarReplacer(var_map).visit(func.body)
        func = tvm.relay.Function(params=new_params,
                                  body=new_body,
                                  ret_type=func.ret_type,
                                  type_params=func.type_params,
                                  attrs=func.attrs)
        return func

    class Flatten_tuple_call(ExprMutator):
        """ Visit call sites and rewrite Tuple arguments """
        def __init__(self):
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar) and \
               call.op.name_hint in compiler_functions:
                new_args = []
                for i,arg in enumerate(call.args):
                    if isinstance(arg, relay.expr.Tuple):
                        new_args.extend(arg.fields)
                    else:
                        new_args.append(arg)
                call = relay.expr.Call(call.op, new_args, call.attrs, span=call.span)
            return super().visit_call(call)

    # Apply the first transformation to all the designated subgraphs
    compiler_functions = []
    for gv in mod.get_global_vars():
        func = mod[gv.name_hint]
        if isinstance(func, Function) and \
           func.attrs and "Compiler" in func.attrs and \
           func.attrs['Compiler'] == compiler:
            mod[gv.name_hint] = flatten_tuple_declaration(gv.name_hint)
            compiler_functions.append(gv.name_hint)

    # Apply the second transformation to all call sites
    mod['main'] = Flatten_tuple_call().visit(mod['main'])
    return mod

class CalibrationGraphMutator(ExprMutator):
    """This mutator should be called after partitioning to produce a module which
    can be executed purely using TVM and will produce additional outputs for
    subgraph inputs. name_map can be used to find the subgraph input name
    corresponding to the output of the same index.
    """
    def __init__(self, compiler):
        ExprMutator.__init__(self)
        self.num_original_outputs = 1
        self.additional_outputs = []
        self.compiler = compiler
        # Will map index in output to subgraph param name.
        self.name_map = {}

    def add_new_outputs(self, subgraph_name, expr, was_input=True):
        """
        Adds expr as an additional output to be generated by the module.
        If expr is a tuple, multiple outputs will be added.
        """
        if isinstance(expr, Tuple):
            for i, out in enumerate(expr.fields):
                if was_input:
                    name = subgraph_name + "_" + str(i)
                else:
                    name = subgraph_name + "_o" + str(i)
                self.name_map[self.num_original_outputs + len(self.additional_outputs)] = name
                self.additional_outputs.append(out)
        else:
            if was_input:
                name = subgraph_name
            else:
                name = subgraph_name + "_o0"
            self.name_map[self.num_original_outputs + len(self.additional_outputs)] = name
            self.additional_outputs.append(expr)

    def visit_call(self, call):
        if isinstance(call.op, Function) and "Compiler" in call.op.attrs \
           and call.op.attrs["Compiler"] == self.compiler:
            var_map = {}
            for arg, param in zip(call.args, call.op.params):
                subgraph_name = "_".join(param.name_hint.split("_")[:2])
                arg = super().visit(arg)
                var_map[param] = arg
                self.add_new_outputs(param.name_hint, arg, was_input=True)
            new_body = VarReplacer(var_map).visit(call.op.body)
            # Add subgraph outputs as well
            self.add_new_outputs(subgraph_name, new_body, was_input=False)
            return new_body
        return super().visit_call(call)

    def make_calibration_graph(self, expr):
        """Builds calibration graph for expr"""

        if isinstance(expr.body.checked_type, relay.TupleType):
            self.num_original_outputs = len(expr.body.checked_type.fields)
        visit_body = super().visit(expr.body)
        # Get original output(s)
        outputs = []
        if isinstance(visit_body, Tuple):
            for out in visit_body.fields:
                outputs.append(out)
        else:
            outputs.append(visit_body)
        # Create new function with added subgraph inputs + outputs
        return relay.Function(expr.params, relay.Tuple(outputs + self.additional_outputs))

class CalibrationPerLayerMutator(ExprMutator):
    """
    This mutator collects all per-tidl-layer outputs and add them as additional graph outputs.
    Rewrite let as original expression.
    """
    def __init__(self, compiler):
        ExprMutator.__init__(self)
        self.num_original_outputs = 1
        self.additional_outputs = []
        self.compiler = compiler
        # Will map index in output to tensor name
        self.name_map = {}

    def visit_let(self, let):
        var_name = let.var.name_hint
        if var_name.startswith(self.compiler):
            let_value = super().visit(let.value)
            self.name_map[self.num_original_outputs + len(self.additional_outputs)] = var_name
            self.additional_outputs.append(let_value)
            return let_value
        else:
            return super().visit_let(let)

    def make_calibration_graph(self, expr):
        """Builds calibration graph for expr"""

        if isinstance(expr.body.checked_type, relay.TupleType):
            self.num_original_outputs = len(expr.body.checked_type.fields)
        for i in range(self.num_original_outputs):
            self.name_map[i] = f"graph_output_{i}"
        visit_body = super().visit(expr.body)
        # Get original output(s)
        outputs = []
        if isinstance(visit_body, Tuple):
            for out in visit_body.fields:
                outputs.append(out)
        else:
            outputs.append(visit_body)
        # Create new function with added subgraph inputs + outputs
        return relay.Function(expr.params, relay.Tuple(outputs + self.additional_outputs))

def get_arg_quantization(expr, mod, all_nodes=None, inout_quant_dict={}, field_index=0):
    """ Get quantization (zp, scale) of the expr's output
        If expr is not a CallNode, we have to find the CallNode where expr is used,
        and find quantization of expr from the CallNode
    """
    if all_nodes == None:
        all_nodes = get_all_nodes(mod['main'])
    for node in all_nodes:
        if isinstance(node, relay.expr.Call):
            if expr in node.args:
                if node.op.name in ['qnn.conv2d', 'qnn.dense']:
                    if expr == node.args[0]:
                        return node.args[2].data.asnumpy(), node.args[4].data.asnumpy()
                    elif expr == node.args[1]:
                        return node.args[3].data.asnumpy(), node.args[5].data.asnumpy()
                if node.op.name in ['qnn.add', 'qnn.mul']:
                    if expr == node.args[0]:
                        return node.args[3].data.asnumpy(), node.args[2].data.asnumpy()
                    elif expr == node.args[1]:
                        return node.args[5].data.asnumpy(), node.args[4].data.asnumpy()
                if node.op.name == 'qnn.concatenate':
                    return node.args[2].fields[field_index].data.asnumpy(), \
                           node.args[1].fields[field_index].data.asnumpy()
                if node.op.name in ['qnn.dequantize', 'qnn.requantize']:
                    return node.args[2].data.asnumpy(), node.args[1].data.asnumpy()
                if node.op.name in ['cast', 'reshape']:
                    return get_quantization(node, mod, all_nodes, inout_quant_dict)
        elif isinstance(node, relay.expr.Tuple):
            indices = [ i for i,e in enumerate(node.fields) if e == expr ]
            if indices:
                return get_arg_quantization(node, mod, all_nodes, inout_quant_dict, indices[0])
    assert False, 'Do not know how to get arg quantization for expr'

def get_quantization(expr, mod, all_nodes=None, inout_quant_dict={}):
    """ Get quantization (zp, scale) of the expr's output from info embedded in Relay IR
        If expr is a CallNode, we can compute quantization directly
    Parameters
    ----------
    expr : tvm.relay.Expr
        Get quantization of the expr's output tensor
    mod : tvm.IRModule
        Module containing subgraphs using external codegen "compiler"
    all_nodes : dict { node : index }
        dictionary of nodes in the module to corresponding traversal indices
    inout_quant_dict: dict { node : quantization }
        dictionary of input/output nodes and their quantization, computed from unpartitioned graph

    Returns
    -------
    quantization : (numpy.int32 or numpy.ndarray for zp, numpy.float32 or numpy.ndarray for scale)
        Use numpy array so that (broadcasting) multiplication can be performed easily
        For per-tensor quantization, zp and scale are arrays containing single scalar value
        For per-axis/channel quantization, zp and/or scale are arrays containing vector of values
    """
    def get_known_quantization(expr):
        if isinstance(expr, relay.expr.Call):
            op_name = expr.op.name
            if op_name == 'qnn.requantize':
                return expr.args[4].data.asnumpy(), expr.args[3].data.asnumpy()
            elif op_name == 'qnn.quantize':
                return expr.args[2].data.asnumpy(), expr.args[1].data.asnumpy()
            elif op_name in ['qnn.conv2d', 'qnn.dense']:
                return np.array(0, dtype=np.int32), \
                       expr.args[4].data.asnumpy() * expr.args[5].data.asnumpy()
            elif op_name in ['qnn.add', 'qnn.mul']:
                return expr.args[7].data.asnumpy(), expr.args[6].data.asnumpy()
            elif op_name == 'qnn.concatenate':
                return expr.args[4].data.asnumpy(), expr.args[3].data.asnumpy()
        return None

    if isinstance(expr.checked_type, relay.ty.TensorType):
        if expr.checked_type.dtype == 'float32':
            return get_default_quantization()
        if expr in inout_quant_dict:
            return inout_quant_dict[expr]
        known_quantization = get_known_quantization(expr)
        if known_quantization != None:
            return known_quantization
        if isinstance(expr, relay.expr.Call):
            op_name = expr.op.name
            if op_name in ['nn.bias_add', 'image.resize2d', 'clip']:
                return get_quantization(expr.args[0], mod, all_nodes, inout_quant_dict)
            elif op_name == 'cast':
                if expr.checked_type.dtype == 'int32':
                    return get_quantization(expr.args[0], mod, all_nodes, inout_quant_dict)
                else:
                    return get_arg_quantization(expr, mod, all_nodes, inout_quant_dict)
            elif op_name in ['nn.avg_pool2d', 'nn.global_avg_pool2d', 'mean', 'strided_slice']:
                return get_arg_quantization(expr, mod, all_nodes, inout_quant_dict)
            elif op_name in ['nn.max_pool2d', 'reshape', 'squeeze', 'nn.batch_flatten', 'nn.pad',
                             'transpose', 'nn.upsampling']:
                # max_pool2d/reshape can get quantization either from its arg or its use
                # similarly, squeeze, nn.batch_flatten, nn.pad, transpose, nn.upsampling
                arg_quant = get_known_quantization(expr.args[0])
                if arg_quant != None:
                    return arg_quant
                return get_arg_quantization(expr, mod, all_nodes, inout_quant_dict)
            elif op_name == 'argmax':
                return get_default_quantization()
            else:
                assert False, f'Do not know how to get quantization for {op_name}'
        else:
            return get_arg_quantization(expr, mod, all_nodes, inout_quant_dict)
    elif isinstance(expr.checked_type, relay.ty.TupleType):
        if expr.checked_type.fields[0].dtype == 'float32':
            return get_default_quantization()
        else:
            assert False, f'Do not yet support getting quantization for Tuple'
    else:
        assert False, f'Do not know how to get quantization for expr'

def get_tidl_element_type(expr):
    """Get corresponding TIDL element type
    Parameters
    ----------
    expr : tvm.relay.Expr
        TIDL subgraph input parameter or output expr
    Returns
    -------
    etype: int
        TIDL element type
    """
    dtype = expr.checked_type.dtype
    if dtype == 'uint8':
        return 0  # TIDL_UnsignedChar
    if dtype == 'int8':
        return 1  # TIDL_SignedChar
    if dtype == 'uint32':
        return 4  # TIDL_UnsignedWord
    if dtype == 'int32':
        return 5  # TIDL_SignedWord
    if dtype == 'float32':
        return 6  # TIDL_SinglePrecFloat
    if dtype == 'uint64':
        return 7  # TIDL_UnsignedDoubleWord
    if dtype == 'int64':
        return 8  # TIDL_SignedDoubleWord
    assert False, f'Unsupported TVM dtype: {dtype} for TIDL subgraph input/output'

def dequantize_tensor(tensor, zp, scale, data_layout):
    """Dequantize tensor into floating point using zp and scale
    Parameters
    ----------
    tensor: numpy.ndarray
        input tensor (maybe quantized, if float, then zp will 0 and scale be 1.0)
    zp: numpy.ndarray
    scale: numpy.ndarray
        contatins single scalar for zero-point/scale if per-tensor quantization,
        contatins vector of  values for zero-points/scales if per-channel quantization,
    data_layout: string
        "NCHW" or "NHWC"
    Returns
    -------
    dequantized : numpy.ndarray
        dequantized tensor
    """
    dequantized = tensor.astype('float32')
    if zp.size == 1 and scale.size == 1:      # per-tensor quantization
        if zp.item() != 0 or scale.item() != 1.0:
            dequantized = (tensor.astype('float32') - zp.item()) * scale.item()
    elif len(tensor.shape) == 4:              # per-channel quantization on 4D tensor
        num_ch = max(len(zp.shape), len(scale.shape))
        if data_layout == "NHWC":
            assert num_ch == tensor.shape[3], \
                   f'Channels of tensor and quantization mismatch: {tensor.shape},{num_ch}'
            dequantized = (tensor.astype('float32') - zp) * scale
        else: # "NCHW"
            assert num_ch == tensor.shape[1], \
                   f'Channels of tensor and quantization mismatch: {tensor.shape},{num_ch}'
            for ch in range(num_ch):
                zp_ch = zp.item() if zp.size == 1 else zp[ch]
                scale_ch = scale.item() if scale.size == 1 else scale[ch]
                dequantized[:,ch,:,:] = (tensor[:,ch,:,:].astype('float32') - zp_ch) * scale_ch
    else:
        assert False, f'Cannot dequantize boundary tensor {tensor.shape} channel-wise'
    return dequantized

def generate_subgraph_tensors(tidl_target, mod, params, graph_input_list, temp_folder,
                              data_layout, has_qnn_ops=False, save_output=False):
    """Creates calibration graph from mod and executes on the cpu to generate boundary tensors.
    """

    #print("----------- Paritioned graph for generating subgraph boundary tensors -----------")
    #print(mod.astext(show_meta_data=False))
    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = unpack_composites(mod_tvm, tidl_target, mod_tvm.get_global_vars())
    mod_tvm = relay.transform.Inline()(mod_tvm)
    mod_tvm = relay.transform.InferType()(mod_tvm)
    calib_mutator = CalibrationGraphMutator(tidl_target)
    mod_tvm["main"] = calib_mutator.make_calibration_graph(mod_tvm["main"])
    mod_tvm = relay.transform.InferType()(mod_tvm)
    with open(os.path.join(temp_folder, "relay_graph.boundary.txt"), "w") as relay_txt:
        print(mod_tvm.astext(show_meta_data=False), file=relay_txt)

    relay_quantization = {}
    relay_etypes = {}
    outputs_expr = mod_tvm["main"].body
    for i, output_i_expr in enumerate(outputs_expr.fields):
        if i in calib_mutator.name_map:
            if has_qnn_ops:
                relay_quantization[calib_mutator.name_map[i]] = get_quantization(output_i_expr,
                                                                                 mod_tvm)
            else:
                relay_quantization[calib_mutator.name_map[i]] = get_default_quantization()
            relay_etypes[calib_mutator.name_map[i]] = get_tidl_element_type(output_i_expr)

    print("Building graph on host for tensor data collection...")
    # Build and execute calibration graph on host to get outputs
    # Use opt_level=0 to avoid optimizations which modify the module (could change original module)
    # Use opt_level=2 to support quantized models, which requires lowering at opt_level 2
    os.environ["TIDL_TVM_HOST_REF_ONLY_BUILD"] = "1"
    with tvm.transform.PassContext(opt_level=2):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    os.environ.pop("TIDL_TVM_HOST_REF_ONLY_BUILD")
    print("Running graph on host for tensor data collection...")
    mod = graph_executor.create(graph, lib, tvm.cpu(0))
    mod.set_input(**params)

    subgraph_tensors_list = []
    for graph_input in graph_input_list:
        mod.set_input(**graph_input)
        mod.run()

        results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]

        # We now have subgraph inputs
        # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
        subgraph_tensors = {}
        for i, res in enumerate(results):
            if i in calib_mutator.name_map:

                # TODO: if quantized output
                # 1) post-process quantized data with zp, scale to float data
                # Okay, do 1) first
                # 2) if data type is "uint8"/"int8", then assume maximum range utilization,
                #    compute tidl_scale from relay (zp, scale) if possible
                #    Note: this assumption of maximum range utilization may not be correct!!!
                #          in which case, need to determine maximum bits being used
                # background for quantization:
                # - TIDL activation tensor can only have zero point of 0 value,
                #   while relay activation tensor can have non-0 zero point
                # - TIDL can only have maximum 16-bit bias,
                #   while relay bias can have maximum 32-bit bias
                # - To avoid re-calibration in TIDL compilation/import flow, we need to provide
                #   (minTensorValue, maxTensorValue) for the activation of each TIDL layer,
                #   and set TIDL layer activation type to TIDL_Clip.  TIDL_updateScaleFactor()
                #   in TIDL_init() will use these information to compute roundBits
                #   and tensorScale for each layer.
                #   - Avoiding re-calibration is possible if every corresponding relay layer
                #     is of type "uint8"/"int8", and all bits are used
                #     - we then compute (minTensorValue, maxTensorValue) from relay layer,
                #       without any calibration data, set
                zp, scale = relay_quantization[calib_mutator.name_map[i]]
                # Keep TVM results as is, Use TIDL data convert layers to interface
                #res = dequantize_tensor(res, zp, scale, data_layout)
                subgraph_tensors[calib_mutator.name_map[i]] = res
                if save_output:
                    file_name = os.path.join(temp_folder, calib_mutator.name_map[i] + ".txt")
                    np.savetxt(file_name, res.flatten(), fmt='%10.5f')
        subgraph_tensors_list.append(subgraph_tensors)

    return subgraph_tensors_list, relay_quantization, relay_etypes

def generate_tidl_layer_tensors(tidl_target, mod, params, graph_input_list, temp_folder,
                                data_layout, has_qnn_ops=False):
    """Creates per-tidl-layer tensors to compare with tidl calibration per-layer output.
       If original model/graph data_laytout is "NHWC", transpose 4D tensors to "NCHW" before
       saving to file, so that comparing with TIDL calibration per-layer output in "NCHW"
       is easy.  1D, 2D and 3D tensors are left alone without any tranposing.
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for per-tidl-layer tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    mod_tvm["main"] = CalibrationGraphMutator(tidl_target).visit(mod_tvm["main"])
    mod_tvm = relay.transform.InferType()(mod_tvm)
    print("----------- after call rewriting -----------")
    print(mod_tvm.astext(show_meta_data=False))
    calib_perlayer_mutator = CalibrationPerLayerMutator(tidl_target)
    mod_tvm["main"] = calib_perlayer_mutator.make_calibration_graph(mod_tvm["main"])
    mod_tvm = relay.transform.InferType()(mod_tvm)
    #print("----------- after additional outputs -----------")
    #print(mod_tvm.astext(show_meta_data=False))

    relay_quantization = {}
    for i, output_i_expr in enumerate(mod_tvm["main"].body.fields):
        if i in calib_perlayer_mutator.name_map:
            output_i_name = calib_perlayer_mutator.name_map[i]
            if (not has_qnn_ops) or output_i_name.startswith('graph_output_'):
                relay_quantization[output_i_name] = get_default_quantization()
            else:
                relay_quantization[output_i_name] = get_quantization(output_i_expr, mod_tvm)

    # Build and execute calibration graph on host to get outputs
    # Use opt_level=0 to avoid optimizations which modify the module (could change original module)
    # Use opt_level=2 to support quantized models, which requires lowering at opt_level 2
    os.environ["TIDL_TVM_HOST_REF_ONLY_BUILD"] = "1"
    with tvm.transform.PassContext(opt_level=2):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    os.environ.pop("TIDL_TVM_HOST_REF_ONLY_BUILD")
    mod = graph_executor.create(graph, lib, tvm.cpu(0))
    mod.set_input(**params)

    graph_input = graph_input_list[-1]
    mod.set_input(**graph_input)
    mod.run()

    for i in range(mod.get_num_outputs()):
        tensor = mod.get_output(i).asnumpy()
        zp, scale = relay_quantization[calib_perlayer_mutator.name_map[i]]
        tensor = dequantize_tensor(tensor, zp, scale, data_layout)

        if data_layout == "NHWC" and len(tensor.shape) == 4:
            tensor = tensor.transpose(0, 3, 1, 2)
        file_name = os.path.join(temp_folder, calib_perlayer_mutator.name_map[i] + ".npy")
        np.save(file_name, tensor)

class VarRenamer(ExprMutator):
    """
    Renames vars to match the new subgraph name. Used when subgraphs are renamed starting from zero.
    If subgraph was originally "tidl_34", it would have inputs named like "tidl_34_i0".
    IF new_subgraph_name is "tidl_0", pass will rename that input to "tidl_0_i0".
    """
    def __init__(self, new_subgraph_name):
        ExprMutator.__init__(self)
        self.new_subgraph_name = new_subgraph_name

    def visit_var(self, var):
        # TODO: Make sure input isn't from a composite func.
        # TODO: Doesn't account for tuple inputs (not possible due to
        #       prune_subgraphs_with_multiple_inputs)
        if var.name_hint.startswith("tidl") and "_".join(var.name_hint.split('_')[:2]) \
                                                != self.new_subgraph_name:
            new_var_name = self.new_subgraph_name + "_" + var.name_hint.split('_')[2]
            return relay.Var(new_var_name, var.checked_type)
        return super().visit_var(var)

class SubgraphRemover(ExprMutator):
    """
    Removes subgraphs which are in the list subgraphs_to_remove and returns them back to regular
    TVM compilation in main function.
    """
    def __init__(self, subgraphs_to_remove, mod, new_mod, compiler="tidl",
                 rename_starting_from_0=True):
        ExprMutator.__init__(self)
        self.subgraphs_to_remove = subgraphs_to_remove
        self.mod = mod
        self.new_mod = new_mod
        self.compiler = compiler
        self.rename_starting_from_0 = rename_starting_from_0
        self.count = 0

    def visit_call(self, call):
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
            if name in self.subgraphs_to_remove:
                # "Inline" the subgraph back into new main function.
                func = self.mod[name]
                var_map = {}
                for arg, param in zip(call.args, func.params):
                    var_map[param] = super().visit(arg)
                new_body = VarReplacer(var_map).visit(func.body)
                return new_body
            if name != "main":
                # Copy the GlobalVar (subgraph function) to the new module and call.
                if self.rename_starting_from_0:
                    new_name = self.compiler + "_" + str(self.count)
                    self.count += 1
                else:
                    new_name = name
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                subgraph_gv = relay.GlobalVar(new_name)
                if self.rename_starting_from_0:
                    subgraph_func = VarRenamer(new_name).visit(self.mod[name])
                    subgraph_func = subgraph_func.with_attr("global_symbol", new_name)
                    self.new_mod[subgraph_gv] = subgraph_func
                else:
                    self.new_mod[subgraph_gv] = self.mod[name]
                return subgraph_gv(*args)
        return super().visit_call(call)

def prune_subgraphs_with_multiple_inputs(mod, compiler="tidl"):
    """Removes subgraphs which have more than one input from mod and returns them to the regular
    TVM compilation path.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing subgraphs using external codegen "compiler"
    compiler : str
        Only subgraphs from this external codegen compiler will be modified.

    Returns
    -------
    ret : tvm.IRModule
        New module with only single-input subgraphs left.
    """
    subgraph_names_to_remove = []
    # Remove subgraphs with more than 1 input or tuple inputs.
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or "Compiler" not in mod[name].attrs or \
                mod[name].attrs["Compiler"] != compiler:
            continue
        if len(mod[name].params) != 1 \
           or isinstance(mod[name].params[0].checked_type, relay.TupleType):
            subgraph_names_to_remove.append(name)
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod,
                                      compiler=compiler).visit(mod["main"])
    return new_mod

def prune_subgraphs_with_overlimit_inputs_outputs(mod, in_out_limit=16, compiler="tidl"):
    """Removes subgraphs which have more than 16 inputs or outputs.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing subgraphs using external codegen "compiler"
    in_out_limit : int
        Limit on number of inputs/outputs per subgraph
    compiler : str
        Only subgraphs from this external codegen compiler will be modified.

    Returns
    -------
    ret : tvm.IRModule
        New module with subgraphs only with inputs and outputs below limit.
    """
    subgraph_names_to_remove = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or "Compiler" not in mod[name].attrs or \
                mod[name].attrs["Compiler"] != compiler:
            continue
        # Remove subgraphs with inputs or outputs over limit.
        #   - mod[name].params has the input tensors
        #   - mod[name].body has the output tensors
        if len(mod[name].params) > in_out_limit \
           or (isinstance(mod[name].params[0].checked_type, relay.TupleType) \
              and len(mod[name].params[0].checked_type.fields) > in_out_limit) \
           or (isinstance(mod[name].body.checked_type, relay.TupleType) \
              and len(mod[name].body.checked_type.fields) > in_out_limit):
            subgraph_names_to_remove.append(name)
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod,
                                      compiler=compiler).visit(mod["main"])
    
    # unwind "tidl" marked Composites which got pruned from TIDL subgraphs into the main graph
    new_mod = unpack_composites(new_mod, "tidl", ["main"])
    new_mod = relay.transform.InferType()(new_mod)
    return new_mod

def prune_subgraphs(mod, compiler="tidl", num_subgraphs_to_keep=4, min_mac_threshold=None):
    """Removes subgraphs from mod and returns them to the regular TVM compilation path.
    The subgraphs with the highest number of multiply-accumulates are kept.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing subgraphs using external codegen "compiler"
    compiler : str
        Only subgraphs from this external codegen compiler will be modified.
    num_subgraphs_to_keep : int
        How many subgraphs to keep.
    min_mac_threshold : int (optional)
        If set, will also prune all subgraphs with # macs < the threshold.

    Returns
    -------
    ret : tvm.IRModule
        New module with only "num_subgraphs_to_keep" subgraphs left.
    """
    subgraph_with_macs = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or "Compiler" not in mod[name].attrs or \
                mod[name].attrs["Compiler"] != compiler:
            continue
        num_macs = relay.analysis.get_total_mac_number(mod[name])
        subgraph_with_macs.append([name, num_macs])
    subgraph_with_macs = sorted(subgraph_with_macs, key=lambda x: int(x[1]))
    # also support pruning all subgraphs
    num_subgraphs_to_keep = min(len(subgraph_with_macs), num_subgraphs_to_keep)
    subgraphs_to_prune = subgraph_with_macs[0 : len(subgraph_with_macs) - num_subgraphs_to_keep]
    if min_mac_threshold:
        # Also remove all subgraphs under the minimum threshold.
        subgraphs_to_prune += [[x[0], x[1]] for x in subgraph_with_macs if x[1] < min_mac_threshold]
    subgraph_names_to_remove = {x[0] for x in subgraphs_to_prune}
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod,
                                      compiler=compiler).visit(mod["main"])
    
    # unwind "tidl" marked Composites which got pruned from TIDL subgraphs into the main graph
    new_mod = unpack_composites(new_mod, "tidl", ["main"])
    new_mod = relay.transform.InferType()(new_mod)
    return new_mod

def subgraph_calibration(subgraph_id, input_quant_vec_list, input_etypes, temp_folder, platform):
    """ Run TIDL calibation for the imported subgraph.
    """
    # Save quantized input vector to a file for calib tool to read
    # Saving as 'int8' or 'uint8' is the same
    calib_raw_image = temp_folder + 'calib_raw_data'+str(subgraph_id)+'.bin'
    open(calib_raw_image, "wb").close() # delete old file contents
    fid = open(calib_raw_image, "ab")

    # Multiple calibration data are written to the same file, one after another
    for input_quant_vec in input_quant_vec_list:
        for i in range(len(input_quant_vec)):
            if (input_etypes[i] == 0):
                input_quant_vec[i].astype('uint8').tofile(fid)
            elif (input_etypes[i] == 1):
                input_quant_vec[i].astype('int8').tofile(fid)
            elif (input_etypes[i] == 6):
                input_quant_vec[i].astype('float32').tofile(fid)
            elif(input_etypes[i]==5):
                 input_quant_vec[i].astype('int32').tofile(fid)
            elif (input_etypes[i] == 8):
                input_quant_vec[i].astype('int64').tofile(fid)
            else:
                assert False, f'Unsupported TIDL calibration data type: {input_etypes[i]}'
    fid.close()

    if supported_platform(platform):
        import_lib_postprocess = tvm.get_global_func("TIDL_relayPostProcessNet")
        import_ret = import_lib_postprocess(len(input_quant_vec_list))
        return (import_ret == 0), 123  ## TODO: do we need dataQ for J7?

class InOutNodes(ctypes.Structure):
    """ Input/output nodes defined in ctypes for passing to TIDL C library """
    _fields_ = [('this_node', ctypes.c_char_p),
                ('num_in_nodes', ctypes.c_int), ('num_out_nodes', ctypes.c_int),
                ('in_nodes', ctypes.c_void_p), ('out_nodes', ctypes.c_void_p)]

class TensorDescriptor(ctypes.Structure):
    """ Input/output tensor descriptor for TIDL subgraphs """
    _fields_ = [('scale', ctypes.c_double),
                ('zp', ctypes.c_int),
                ('element_type', ctypes.c_int),
                ('n', ctypes.c_int),
                ('dim1', ctypes.c_int),
                ('dim2', ctypes.c_int),
                ('channel', ctypes.c_int),
                ('height', ctypes.c_int),
                ('width', ctypes.c_int),
                ('name', ctypes.c_char_p)]

class ODPostProcInfo(ctypes.Structure):
    """ Post Processing params defined in ctypes for passing to TIDL C library """
    _fields_ = [('in_node_names', ctypes.c_char * 512 * 64),
                ('num_in_nodes', ctypes.c_int), ('num_out_nodes', ctypes.c_int),
                ('out_nodes', TensorDescriptor * 64)]


class TIDLImport:
    """TIDL import module.
    Parameters
    ----------
    import_lib : ctypes.CDLL
        TIDL import library
    artifacts_folder : string
        Directory path to hold the artifacts
    tidl_target : string
        TIDL compilation target
    data_layout : string
        Data layout, "NCHW" or "NHWC"
    tensor_bits : int
        Number of bits for tidl tensors (and consequently params on J7)
    """
    def __init__(self, import_lib, tidl_tools_path, artifacts_folder,
                 tidl_target="tidl", tidl_platform="J7", data_layout="NCHW",
                 tensor_bits=8, tidl_od_meta_arch_type = -1, tidl_od_num_graph_outputs = 1,
                 tidl_od_meta_layers_names_list = "", tidl_od_postproc_inputs=[]):
        self.import_lib = import_lib
        self.tidl_tools_path = tidl_tools_path
        self.artifacts_folder = artifacts_folder
        self.tidl_target = tidl_target
        self.tidl_platform = tidl_platform
        self.data_layout = data_layout
        self.tensor_bits = tensor_bits
        self.tidl_od_meta_arch_type = tidl_od_meta_arch_type
        self.tidl_od_num_graph_outputs = tidl_od_num_graph_outputs
        self.tidl_od_meta_layers_names_list = tidl_od_meta_layers_names_list
        self.tidl_od_postproc_inputs = tidl_od_postproc_inputs
        self.info_dict = {}
        self.tidl_relay_import_debug = os.environ.get("TIDL_RELAY_IMPORT_DEBUG")
        self.temp_folder = os.path.join(artifacts_folder, 'tempDir/')

    def tidl_import_init(self, subgraph_id, input_zps, input_scale_invs, input_etypes,
                         input_tensors, input_names, output_zps, output_scale_invs, output_etypes):
        r""" Initializing TIDL import

        Parameters
        ----------
        subgraph_id: int
            Id of the subgraph to be imported to TIDL
        input_zps: list
            Zero-point of TVM input tensor
        input_scale_invs: list
            Inv scale of TVM input tensor
        input_etypes: list
            TIDL_ElementType of TVM input tensor
        input_tensors: list
            Input tensors to TIDL subgraph
        input_names: list
            Names of input tensors
        output_zps: list
            Zero-point of TVM output tensor
        output_scale_invs: list
            Inv scale of TVM output tensor
        output_etypes: list
            TIDL_ElementType of TVM output tensor
        Returns
        -------
        True if initialization succeeds or False if initialization fails
        """

        # Populate input shapes for communication with TIDL
        input_shapes = []
        for input_tensor in input_tensors:
            input_shape = input_tensor.shape
            if len(input_shape) > TIDL_DIM_MAX:
                print("Subgraph input_shape " + str(input_shape) + " is not supported")
                return False
            if self.data_layout in ["NCHW", "NCW"]:
                # Populate shapes consistent with TI-ONNXRT
                is_nchw = 1
                # input is a vector - expand (x,y,z) to (1,1,1,x,y,z) - and respectively for other dims < TIDL_DIM_MAX
                in_shape = (1,)*(TIDL_DIM_MAX-len(input_shape)) + input_shape
            elif self.data_layout in ["NHWC", "NWC"]:
                # Populate shapes consistent with TI-TfLiteRT
                is_nchw = 0
                if len(input_shape) == 2:
                    # input is a vector - expand (N,W) to (N,1,1,W)
                    in_shape = (input_shape[0], 1, 1, 1, 1, input_shape[1])
                elif len(input_shape) == 3:
                    # expand (N,H,W) to (N,1,H,W)
                    in_shape = (input_shape[0], 1, 1, 1, input_shape[1], input_shape[2])
                elif len(input_shape) == 4:
                    in_shape = input_shape
                    if self.data_layout == "NHWC":
                        in_shape = (in_shape[0], 1, 1, in_shape[3], in_shape[1], in_shape[2])
            else:
                print('data layout ' + self.data_layout + ' is not supported')
                return False

            input_shapes.append(in_shape)

        descr = (TensorDescriptor * (len(input_zps) + len(output_zps)))()
        for i in range(len(input_zps)):
            descr[i].scale = input_scale_invs[i]
            descr[i].zp = input_zps[i]
            descr[i].element_type = input_etypes[i]
            (descr[i].n, descr[i].dim1, descr[i].dim2, descr[i].channel, descr[i].height, descr[i].width) = input_shapes[i][0:TIDL_DIM_MAX]
            descr[i].name = bytes(input_names[i], 'utf-8')
        for i in range(len(output_zps)):
            descr[len(input_zps) + i].scale = output_scale_invs[i]
            descr[len(input_zps) + i].zp = output_zps[i]
            descr[len(input_zps) + i].element_type = output_etypes[i]
        inout_dscr_ptr = ctypes.cast(descr, ctypes.c_void_p)
        import_lib_init = tvm.get_global_func("TIDL_relayImportInit")
        if(import_lib_init(subgraph_id, len(input_zps), len(output_zps), inout_dscr_ptr, is_nchw,
                        self.tidl_tools_path, self.temp_folder) != 0):
            print('\n\nTIDL import initialization failed!!!\n\n')
            return False

        return True

    def tidl_import_node(self, all_nodes, this_node, output_names,
                         inout_quant_dict, has_qnn_ops=False):
        r""" Importing a given node (operator) to TIDL
            # https://docs.tvm.ai/langref/relay_op.html#relay-core-tensor-operators

        Parameters
        ----------
        all_nodes : dictionary
            Dictionary of all relay.expr.Call nodes of the graph
        this_node : relay.expr.Call
            A relay.expr.Call node which is to be imported
        output_names: names of the subgraph outputs
        inout_quant_dict: input/output expr to quantization dictionary

        Returns
        True if import succeeds or False if import fails
        """

        if supported_platform(self.tidl_platform):
            import_lib_node = tvm.get_global_func("TIDL_relayImportNode")
            if has_qnn_ops:
                zp, scale = get_quantization(this_node, None, all_nodes, inout_quant_dict)
            else:
                zp, scale = get_default_quantization()
            if not isinstance(zp, np.ndarray):
                zp = np.array(zp, dtype=np.int32)
            if not isinstance(scale, np.ndarray):
                scale = np.array(scale, dtype=np.float32)
            if import_lib_node(this_node, zp.size, zp.ctypes.data_as(ctypes.c_void_p),
                               scale.size, scale.ctypes.data_as(ctypes.c_void_p)) != 0:
                return False
            in_out_nodes = find_in_out_nodes(all_nodes, this_node, self.tidl_target, output_names)
            import_lib_linknode = tvm.get_global_func("TIDL_relayImportLinkNode")
            if import_lib_linknode(ctypes.cast(ctypes.byref(in_out_nodes), ctypes.c_void_p)) == 0:
                return True
            else:
                return False

    def tidl_import_out_tuple_node(self, all_nodes, node, out_tensor_names):
        """ Importing a Relay tuple node, e.g. (%232, %279, %283, %274).
            If this node is the last node, import it to TIDL output data layer.
            If this node is not the last node, do nothing.

        Parameters
        ----------
        all_nodes : dictionary
            Dictionary of all relay.expr.Call nodes of the graph
        node : relay.expr.Tuple
            A relay.expr.Tuple node that represents the multiple outputs of the subgraph
        out_tensor_names: names of the subgraph outputs

        Returns
        True if import succeeds or False if import fails
        """

        # make sure 16 matches TIDL_NUM_OUT_BUFS defined in itidl_ti.h
        max_num_outputs_per_data_layer = 16
        # this is the last node of the graph - import this to out data layer
        in_nodes = find_in_nodes(all_nodes, node, self.tidl_target)
        imported_nodes = 0
        new_node_ind = len(all_nodes) + 1
        status = True
        while imported_nodes < len(in_nodes):
            if len(in_nodes) - imported_nodes < max_num_outputs_per_data_layer:
                nodes_for_this_data_layer = len(in_nodes) - imported_nodes
                this_is_the_last_one = True
            else:
                nodes_for_this_data_layer = max_num_outputs_per_data_layer
                this_is_the_last_one = False

            import_lib_out_data = tvm.get_global_func(
                                        "TIDL_relayImportOutDataLayer")
            import_lib_out_data(nodes_for_this_data_layer)

            # prepare input/output nodes information for linking
            in_out_nodes = InOutNodes()    # instantiate structure
            # TODO: put a meaningful name to this_node, e.g. tidl_0_outnode_0, etc.
            in_out_nodes.this_node = bytes(str(new_node_ind), 'utf-8')
            in_out_nodes.num_in_nodes = nodes_for_this_data_layer
            in_nodes_this_layer = \
                in_nodes[imported_nodes:imported_nodes+nodes_for_this_data_layer]
            # convert list to char * array in order to pass to C library
            in_nodes_char = convert_str_list_to_char_array(in_nodes_this_layer)
            in_out_nodes.in_nodes = ctypes.cast(in_nodes_char, ctypes.c_void_p)
            out_tensor_names_this_layer = \
                out_tensor_names[imported_nodes:imported_nodes+nodes_for_this_data_layer]
            out_tensors_char = convert_str_list_to_char_array(out_tensor_names_this_layer)
            # output tensor names are stored in out_nodes[]
            in_out_nodes.out_nodes = ctypes.cast(out_tensors_char, ctypes.c_void_p)
            in_out_nodes.num_out_nodes = nodes_for_this_data_layer

            import_lib_linknode = tvm.get_global_func(
                                            "TIDL_relayImportLinkNode")
            if import_lib_linknode(ctypes.cast(ctypes.byref(
                                 in_out_nodes), ctypes.c_void_p)) != 0:
                status = False
                break

            imported_nodes = imported_nodes + nodes_for_this_data_layer
            new_node_ind = new_node_ind + 1
            if this_is_the_last_one:
                break

        return status

    def import_relay_ir(self, mod, params, subgraph_tensors_list, relay_quantization,
                        relay_etypes, has_qnn_ops=False):
        r""" Relay IR import to TIDL

        Parameters
        ----------
        mod : tvm.relay.Module
            Relay IR graph with subgraphs
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay
        subgraph_tensors_list: list of dict (list length equals number of calibration data)
            Input/output tensors of subgraphs obtained from TVM graph execution
        relay_quantization: { name: (zp, scale) } dictionary for input/output tensors
        relay_etypes: { name: TIDL_ElementType } dictionary for input/output tensors

        Returns
        -------
        len(tidl_subgraphs) : int
            >=0: number of imported TIDL subgraphs, if TIDL import succeeds
            -1: if TIDL import fails
        """

        # Generate svg for partitined graph
        visualize_relay_graph(module=mod, filename=self.temp_folder+'/relay.gv')

        # Define return values
        import_fail = -1

        # Put some information about the graph in the info file passed to the TIDL codegen
        self.info_dict['tvm'] = {
           'is_nchw'   : 1 if self.data_layout == "NCHW" else 0,
           'macs'      : relay.analysis.get_total_mac_number(mod['main']),
           'nodes'     : {},
        }
        self.info_dict['subgraphs'] = []

        tidl_subgraphs = get_tidl_subgraphs(mod, self.tidl_target)

        # Tally relay call nodes that are not calls to a TIDL subgraph (Nodes to be operated on by TVM, not TIDL)
        for node in get_all_nodes(mod['main']):
            if isinstance(node, relay.expr.Call) and isinstance(node.op, tvm.ir.op.Op): # Check for TVM specific ops
                self._tally_op(str(node.op), self.info_dict['tvm']['nodes'])

        # For each TIDL subgraph, import to TIDL and calibrate
        for tidl_subgraph in tidl_subgraphs:
            # Extract subgraph id and input/output tensor names from subgraph name
            subgraph_id = int(tidl_subgraph.replace(self.tidl_target+'_', ''))
            in_tensor_name = tidl_subgraph + '_i'
            out_tensor_name = tidl_subgraph + '_o'

            # Obtain input tensor from TVM graph execution
            input_fp_list, input_names_list = \
                  obtain_subgraph_tensor(subgraph_tensors_list, in_tensor_name)
            output_fp_list, output_names_list = \
                  obtain_subgraph_tensor(subgraph_tensors_list, out_tensor_name)
            input_names = input_names_list[0]
            output_names = output_names_list[0]
            input_zp_list, input_scale_inv_list = \
                  obtain_tensor_quantization(input_names, relay_quantization)
            output_zp_list, output_scale_inv_list = \
                  obtain_tensor_quantization(output_names, relay_quantization)
            input_etype_list = obtain_tensor_etype(input_names, relay_etypes)
            output_etype_list = obtain_tensor_etype(output_names, relay_etypes)
            if not input_fp_list:
                return import_fail

            # Quantize input tensors
            input_quant_vec_list, input_scale, input_signed = \
                    tensor_quant_flatten(input_fp_list, self.data_layout, self.tensor_bits)

            # Initialize TIDL import
            subgraph = mod[tidl_subgraph]
            if not self.tidl_import_init(subgraph_id, input_zp_list, input_scale_inv_list,
                                         input_etype_list, input_fp_list[0], input_names,
                                         output_zp_list, output_scale_inv_list, output_etype_list):
                return import_fail

            # Initialize subgraph info for nfo file
            subgraph_info_dict = {
               'name'    : tidl_subgraph,
               'is_nchw' : 1 if self.data_layout == "NCHW" else 0,
               'macs'    : relay.analysis.get_total_mac_number(subgraph),
               'ninputs' : len(input_names),
               'noutputs': len(output_names),
               'inouts_zp' : input_zp_list + output_zp_list,
               'inouts_scale_inv' : input_scale_inv_list + output_scale_inv_list,
               'nodes'   : {},
            }

            # Initialize subgraph input/output exprs to quantization mapping
            inout_quant_dict = obtain_inout_quant_dict(subgraph, subgraph_id, relay_quantization)

            # If subgraph contains "tidl_odpostproc" layer, only import up to the inputs
            #   of this layer, TIDL will add the postprocessing layers using MetaArch info
            def find_tidl_odpostproc(node, node_list):
                if isinstance(node, relay.expr.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tidl_odpostproc":
                    node_list.append(node)
            tidl_odpostproc_nodes = []
            traverse_func = functools.partial(find_tidl_odpostproc, node_list=tidl_odpostproc_nodes)
            relay.analysis.post_order_visit(subgraph, traverse_func)
            if len(tidl_odpostproc_nodes) == 0:
                subgraph_body = subgraph.body
            else:
                assert len(tidl_odpostproc_nodes) == 1, "Only one tidl_postproc is allowed"
                subgraph_body = relay.expr.Tuple(tidl_odpostproc_nodes[0].args)
                output_names = self.tidl_od_postproc_inputs
                import_lib_setup_odpostproc = tvm.get_global_func("TIDL_relaySetupODPostProc")
                if(import_lib_setup_odpostproc(self.tidl_od_meta_arch_type,
                        self.tidl_od_num_graph_outputs, self.tidl_od_meta_layers_names_list) != 0):
                    print("\n\nTIDL OD PostProc setup failed!!!\n\n")
                    return import_fail

            # Scan through all relay.expr.Call nodes and import each to TIDL
            all_nodes_tidl = {}
            # Skip traversing into function body if marked with Composite="tidl.<Op>"
            visitor = SkipLocalFunctionsVisitor(self.tidl_target)
            visitor.visit(subgraph_body)
            all_nodes_tidl = visitor.nodes
            for node in all_nodes_tidl:
                if isinstance(node, relay.expr.Call):
                    result = self.tidl_import_node(all_nodes_tidl, node, output_names,
                                                   inout_quant_dict, has_qnn_ops)
                    if not result:
                        if (node.span and node.span.source_name and hasattr(node.span.source_name, 'name') and 
                        node.span.source_name.name):
                            print(f'\n\nError importing node - {node.span.source_name.name}!!!\n\n')
                        else:
                            print('\n\nError importing node!!!\n\n')
                        return import_fail
                    self._tally_op(str(node.op), subgraph_info_dict['nodes'])

            # Import expr.Tuple node if it is the last node, after importing all expr.call nodes
            for node in all_nodes_tidl:
                if isinstance(node, relay.expr.Tuple) and \
                   len(find_out_nodes(all_nodes_tidl, node)) == 0:
                    #node.fields: array of expr.call nodes
                    result = self.tidl_import_out_tuple_node(all_nodes_tidl, node, output_names)
                    if not result:
                        print('\n\nError importing output tuple node!!!\n\n')
                        return import_fail

            # TIDL optimization
            import_lib_optimize = tvm.get_global_func("TIDL_relayOptimizeNet")
            if import_lib_optimize(subgraph_id) != 0:
                print('\n\nTIDL import optimization failed!!!\n\n')
                return import_fail

            # Calibrate TIDL for the imported subgraph
            status, out_data_q = subgraph_calibration(subgraph_id, input_quant_vec_list, input_etype_list, self.temp_folder,
                                     self.tidl_platform)
            self.info_dict['subgraphs'].append(subgraph_info_dict)
            if status:
                mod[tidl_subgraph] = self.mark_tidl_layers(subgraph, subgraph_id, all_nodes_tidl)
                continue  # import next subgraph
            else:
                print("\n\nSubgraph calibration failed!!!\n\n")
                return import_fail

        with open(os.path.join(self.temp_folder, "relay.nfo"), "w") as of:
            json.dump(self.info_dict, of, indent=4)

        return len(tidl_subgraphs)

    def _tally_op(self, op_name, node_dict):
        """ helper function to tally instance count of each operator in info dictionary """
        if op_name in node_dict:
            node_dict[op_name] += 1
        else:
            node_dict[op_name] = 1

    def mark_tidl_layers(self, subgraph, subgraph_id, all_nodes_tidl):
        """ mark Relay IR Call Node that corresponds to tidl layers by using a "let"
            expression with a "tidl_<subgraph_id>_layer<layer_index>" var name
        """

        class CallMarker(ExprMutator):
            """ mark Call Node with tidl layers info """
            def __init__(self, subgraph_id, all_nodes_tidl, layer_info):
                ExprMutator.__init__(self)
                self.subgraph_id = subgraph_id
                self.all_nodes_tidl = all_nodes_tidl
                self.layer_info = layer_info

            def get_tidl_layer_varname(self, node):
                node_name = str(self.all_nodes_tidl[node])
                for l in self.layer_info:
                    if l[2] == node_name or l[2] == f'tidl_{subgraph_id}_{node_name}':
                        return f"tidl_{self.subgraph_id}_layer{int(l[0]):04d}"
                return None

            def mark_with_let(self, var_name, value_expr):
                sb = relay.ScopeBuilder()
                layer_var = sb.let(var_name, value_expr)
                sb.ret(layer_var)
                return sb.get()

            def visit_call(self, call):
                visited_call = super().visit_call(call)
                if call in self.all_nodes_tidl and \
                   not isinstance(call.checked_type, tvm.ir.TupleType):
                    var_name = self.get_tidl_layer_varname(call)
                    if var_name != None:
                        return self.mark_with_let(var_name, visited_call)
                return visited_call

            def visit_tuple_getitem(self, getitem):
                """ BatchNorm in Relay has 3 outputs in a tuple.  When imported into TIDL,
                    it has only 1 output.  Need to mark batchnorm.%0 as TIDL layer output """
                visited_getitem = super().visit_tuple_getitem(getitem)
                if getitem.tuple_value in self.all_nodes_tidl and \
                   isinstance(getitem.tuple_value, relay.expr.Call) and \
                   getitem.tuple_value.op.name == 'nn.batch_norm' and \
                   getitem.index == 0:
                    var_name = self.get_tidl_layer_varname(getitem.tuple_value)
                    if var_name != None:
                        return self.mark_with_let(var_name, visited_getitem)
                return visited_getitem

        ### tempDir/subgraph<id>_net.bin_calib.layer_info.txt always available, even DEBUG level 0
        ### format of each line: layer_index data_id output_name
        layer_info_file = os.path.join(self.temp_folder,
                                       f"subgraph{subgraph_id}_net.bin_calib.layer_info.txt")
        layer_info = [ x.split(' ') for x in open(layer_info_file).readlines() ]
        # rewrite outputs as "tidl_<subgraph_id>_o<output_id>" to match up with TIDL names
        if isinstance(subgraph.body, Tuple):
            for i, out in enumerate(subgraph.body.fields):
                all_nodes_tidl[out] = f"tidl_{subgraph_id}_o{i}"
        else:
            all_nodes_tidl[subgraph.body] = f"tidl_{subgraph_id}_o0"
        return CallMarker(subgraph_id, all_nodes_tidl, layer_info).visit(subgraph)

class TIDLAnnotation:
    def __init__(self, platform, import_lib):
        self.tidl_platform = platform
        self.import_lib = import_lib

    def register_allowed_ops(self):
        """ TIDL operators registration """
        # Can't register annotations more than once.
        global tidl_annotations_registered
        if tidl_annotations_registered:
            return

        # Register common operators which are supported with different constraints
        self._register_constrained_op("nn.relu")
        self._register_constrained_op("argmax")
        self._register_constrained_op("nn.avg_pool2d")
        self._register_constrained_op("nn.batch_flatten")
        self._register_constrained_op("nn.batch_norm")
        self._register_constrained_op("nn.conv2d")
        #self._register_supported_op("nn.conv2d")    # use this for debugging
        # self._register_constrained_op("nn.dense")
        self._register_constrained_op("nn.conv2d_transpose")
        self._register_constrained_op("nn.global_avg_pool2d")
        self._register_constrained_op("nn.adaptive_avg_pool1d") # 1d global average pool
        self._register_constrained_op("nn.adaptive_avg_pool3d") # 3d global average pool
        self._register_constrained_op("nn.max_pool2d")
        self._register_constrained_op("nn.softmax")
        self._register_constrained_op("concatenate")
        self._register_constrained_op("mean")          # 'mean' mapped to avg_pooling layer
        self._register_constrained_op("nn.depth_to_space")
        self._register_constrained_op("nn.space_to_depth")
        # Register J7 specific operators, or those supported standalone by J7,
        # or those for which there are no allow functions.
        self._register_constrained_op("abs")
        self._register_constrained_op("acos")
        self._register_constrained_op("add")
        self._register_constrained_op("asin")
        self._register_constrained_op("asinh")
        self._register_constrained_op("atan")
        self._register_constrained_op("cos")
        self._register_constrained_op("cosh")
        self._register_constrained_op("nn.bias_add")
        self._register_constrained_op("reshape")
        self._register_constrained_op("subtract")
        self._register_constrained_op("maximum")
        self._register_constrained_op("minimum")
        self._register_constrained_op("transpose")
        self._register_constrained_op("multiply")
        self._register_constrained_op("divide")
        self._register_constrained_op("sin")
        self._register_constrained_op("split")
        self._register_constrained_op("strided_slice")
        self._register_constrained_op("image.resize2d")
        self._register_constrained_op("log")
        # "clip" is supported with constraints in J7
        self._register_constrained_op("clip")
        self._register_constrained_op("nn.leaky_relu")
        self._register_constrained_op("nn.prelu")
        self._register_constrained_op("sigmoid")
        # "tanh" is not supported on AM62A
        self._register_constrained_op("tanh")
        self._register_constrained_op("nn.upsampling")
        self._register_constrained_op("nn.upsampling3d")
        self._register_constrained_op("qnn.conv2d")
        self._register_constrained_op("qnn.requantize")
        self._register_constrained_op("qnn.add")
        self._register_constrained_op("cast")
        self._register_constrained_op("qnn.concatenate")
        self._register_constrained_op("qnn.dense")
        self._register_constrained_op("qnn.mul")
        self._register_constrained_op("nn.pad")
        self._register_constrained_op("negative")
        self._register_constrained_op("power")
        self._register_constrained_op("floor")
        self._register_constrained_op("nn.instance_norm")
        self._register_constrained_op("erf")
        self._register_constrained_op("exp")
        self._register_constrained_op("squeeze")
        self._register_constrained_op("sinh")
        self._register_constrained_op("sqrt")
        self._register_constrained_op("tan")
        self._register_constrained_op("max")
        self._register_constrained_op("min")
        self._register_constrained_op("sum")


        tidl_annotations_registered = True

    # Used in pattern 'checker' functions. The extract is a composite function, we need to traverse its body
    # Returns ops of given types (ops_to_match) to be checked for specific attributes
    def find_ops_in_composite(self, func_body, op_list, ops_to_match):
        """Recursively find all operations in the composite function body"""
        if isinstance(func_body, relay.expr.Call):
            if hasattr(func_body.op, 'name') and func_body.op.name in ops_to_match:
                op_list.append(func_body)
            # Recursively check arguments
            for arg in func_body.args:
                self.find_ops_in_composite(arg, op_list, ops_to_match)
        elif isinstance(func_body, relay.expr.Tuple):
            for field in func_body.fields:
                self.find_ops_in_composite(field, op_list, ops_to_match)
        elif isinstance(func_body, relay.expr.TupleGetItem):
            self.find_ops_in_composite(func_body.tuple_value, op_list, ops_to_match)
    
    def merge_sequential_ops(self, mod):
        """Fuse sequential ops for op registration."""

        #transpose has to be preceded and followed by reshape
        def _transpose_reshape_pattern():
            reshape_out1 = is_op('reshape')(wildcard())
            transpose_out = is_op('transpose')(reshape_out1)
            reshape_out2 = is_op('reshape')(transpose_out)
            return reshape_out2
        def _transpose_reshape_checker(extract):
            reshape_2 = extract
            transpose = extract.args[0]
            reshape_1 = extract.args[0].args[0]
            resh1 = reshape_1.attrs.newshape
            resh2 = reshape_2.attrs.newshape
            tran = transpose.attrs.axes
            #pattern (reshape, transpose, reshape) is supported with constraints
            is_shuffle = len(resh1) == 5 and len(tran) == 5 and len(resh2) == 4 and \
                         resh1[4] == resh2[3] and resh1[3] == resh2[2] and \
                         tran[4] == 4 and tran[3] == 3 and tran[2] == 1 and tran[1] == 2
            is_space = len(resh1) == 6 and len(tran) == 6 and len(resh2) == 4 and \
                       resh1[4]*resh1[2] == resh2[2] and resh1[3]*resh1[5] == resh2[3] and \
                       resh1[1] == resh2[1] and resh1[2] == resh1[3] and tran[5] == 3 and \
                       tran[4] == 5 and tran[3] == 2 and tran[2] == 4 and tran[1] == 1
            if is_shuffle or is_space:
                return True
            else:
                return False
        
        def _layernorm_pattern():
            """Create a pattern to match layer normalization decomposition.
            LayerNorm is typically decomposed into:
            (x - mean(x, axis=-1)) / sqrt(var(x, axis=-1) + epsilon)
            where var(x) = mean((x - mean(x))^2)
            """
            data = wildcard()
            epsilon = wildcard()

            # Pattern match
            mean1 = is_op('mean')(data)
            diff = is_op('subtract')(data, mean1)
            const_two = is_constant() | wildcard()
            squared = is_op('power')(diff, const_two) | is_op('multiply')(diff, diff)
            variance = is_op('mean')(squared)
            var_eps = (is_op('add')(variance, epsilon) | 
                      is_op('add')(epsilon, variance))
            sqrt_var = is_op('sqrt')(var_eps)
            rsqrt_var = is_op('rsqrt')(var_eps)
            normalized = is_op('divide')(diff, sqrt_var) | is_op('multiply')(diff, rsqrt_var)
            
            return normalized

        def _layernorm_checker():
            """Checker function for layer normalization pattern with specific validation"""
            def checker(extract):
                ops = []
                ops_to_match = ['mean', 'power']
                # Extract is a composite function, traverse its body
                if hasattr(extract, 'body'):
                    self.find_ops_in_composite(extract.body, ops, ops_to_match)
                
                # Check mean operations have axis=-1, if not then check is axis is along width (last dimension)
                mean_ops = [op for op in ops if hasattr(op.op, 'name') and op.op.name == 'mean']
                for mean_op in mean_ops:
                    if hasattr(mean_op, 'attrs') and hasattr(mean_op.attrs, 'axis'):
                        axis = mean_op.attrs.axis
                        
                        # First check if axis is -1
                        if isinstance(axis, (list, tuple)):
                            if -1 in axis:
                                continue  # Valid, axis contains -1
                        elif axis == -1:
                            continue  # Valid, axis is -1
                        
                        # If axis is not -1, then check is axis is along width
                        if len(mean_op.args) > 0 and hasattr(mean_op.args[0], 'checked_type'):
                            input_shape = mean_op.args[0].checked_type.shape
                            last_axis = len(input_shape) - 1
                            
                            if isinstance(axis, (list, tuple)):
                                if last_axis not in axis:
                                    return False
                            elif axis != last_axis:
                                return False
                        else:
                            return False
                    else:
                        return False
                
                # Check power operation has constant value 2
                power_ops = [op for op in ops if hasattr(op.op, 'name') and op.op.name == 'power']
                for power_op in power_ops:
                    if len(power_op.args) >= 2:
                        second_arg = power_op.args[1]
                        if hasattr(second_arg, 'data'):
                            # Check if the constant value is 2
                            import numpy as np
                            const_val = second_arg.data.asnumpy()
                            if not np.allclose(const_val, 2.0):
                                return False
                        else:
                            return False
                
                return True
            return checker

        def _gelu_pattern():
            """Create a pattern to match GELU activation function decomposition.
            GELU is typically decomposed into:
            0.5 * x * (1.0 + erf(x / sqrt(2.0)))
            """
            data = wildcard() # input
            sqrt2_inv = wildcard()
            div_sqrt2 = is_op('multiply')(data, sqrt2_inv) | is_op('divide')(data, sqrt2_inv)
            erf_result = is_op('erf')(div_sqrt2)
            one_const = wildcard()
            erf_plus_one = is_op('add')(erf_result, one_const) | is_op('add')(one_const, erf_result)
            gelu_erf = is_op('multiply')(data, erf_plus_one) | is_op('multiply')(erf_plus_one, data)
            half_const = wildcard()
            gelu_final = is_op('multiply')(gelu_erf, half_const) | is_op('multiply')(half_const, gelu_erf)
            
            return gelu_final

        def _gelu_checker():
            """Checker function for GELU pattern with precise constant validation at specific positions"""
            def checker(extract):
                import numpy as np
                # Since we know the exact pattern structure, validate constants at their specific positions
                # Pattern: 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
                # extract is the final multiply operation: multiply(gelu_erf, 0.5)
                
                # Check 3: Final multiply should have constant 0.5
                if not (hasattr(extract, 'args') and len(extract.args) >= 2):
                    return False
                
                half_found = False
                gelu_erf_expr = None
                for arg in extract.args:
                    if hasattr(arg, 'data'):
                        const_val = arg.data.asnumpy()
                        if np.allclose(const_val, 0.5, rtol=1e-5):
                            half_found = True
                    else:
                        gelu_erf_expr = arg  # This should be the gelu_erf expression
                
                if not half_found or gelu_erf_expr is None:
                    return False
                
                # gelu_erf should be: multiply(data, erf_plus_one)
                if not (isinstance(gelu_erf_expr, relay.expr.Call) and 
                        hasattr(gelu_erf_expr.op, 'name') and 
                        gelu_erf_expr.op.name == 'multiply' and
                        len(gelu_erf_expr.args) >= 2):
                    return False
                
                # Find the add operation (erf_plus_one): add(erf_result, 1.0)
                erf_plus_one_expr = None
                for arg in gelu_erf_expr.args:
                    if (isinstance(arg, relay.expr.Call) and 
                        hasattr(arg.op, 'name') and 
                        arg.op.name == 'add'):
                        erf_plus_one_expr = arg
                        break
                
                if erf_plus_one_expr is None:
                    return False
                
                # Check 2: Add operation should have constant 1.0
                one_found = False
                erf_result_expr = None
                for arg in erf_plus_one_expr.args:
                    if hasattr(arg, 'data'):
                        const_val = arg.data.asnumpy()
                        if np.allclose(const_val, 1.0, rtol=1e-5):
                            one_found = True
                    elif (isinstance(arg, relay.expr.Call) and 
                          hasattr(arg.op, 'name') and 
                          arg.op.name == 'erf'):
                        erf_result_expr = arg
                
                if not one_found or erf_result_expr is None:
                    return False
                
                # erf_result should be: erf(div_sqrt2)
                if not (len(erf_result_expr.args) >= 1):
                    return False
                
                div_sqrt2_expr = erf_result_expr.args[0]
                
                # Check 1: div_sqrt2 should be multiply(data, 1/sqrt(2)) or divide(data, sqrt(2))
                if not (isinstance(div_sqrt2_expr, relay.expr.Call) and 
                        hasattr(div_sqrt2_expr.op, 'name') and 
                        div_sqrt2_expr.op.name in ['multiply', 'divide'] and
                        len(div_sqrt2_expr.args) >= 2):
                    return False
                
                sqrt2_found = False
                for arg in div_sqrt2_expr.args:
                    if hasattr(arg, 'data'):
                        const_val = arg.data.asnumpy()
                        if div_sqrt2_expr.op.name == 'divide':
                            # Check for sqrt(2) ≈ 1.414
                            if np.allclose(const_val, np.sqrt(2.0), rtol=1e-5):
                                sqrt2_found = True
                                break
                        elif div_sqrt2_expr.op.name == 'multiply':
                            # Check for 1/sqrt(2) ≈ 0.707
                            if np.allclose(const_val, 1.0/np.sqrt(2.0), rtol=1e-5):
                                sqrt2_found = True
                                break
                
                return sqrt2_found
            return checker

        def _patch_merging_pattern():
            """Create a pattern to match patch merging in Vision Transformers.
            This creates a 2x2 patch merging pattern typical in Swin Transformer.
            """
            
            # Note : Below function seems to be the ideal way to create FunctionPattern object for Slice (in case of multiple Slices in pattern)
            # However, using it results in issue due to the pattern matching implementation using memoization to store 
            # relay expressions corresponding to matched patterns.
            # e.g. _memo_map[pattern1] = relay_expr_1
            # When same pattern object is called with different expression e.g. relay_expr_2, 
            # the pattern checker checks if relay_expr_2 == relay_expr_1 instead of relay_expr_2 == pattern1  
            # resulting in subsequent slice functions not matching with the pattern
            # So ensure to create new object of the FunctionPattern class to match each slice
            
            # def composite_call(name):
            #     func_pattern = FunctionPattern(None, wildcard()).has_attr({"Composite": name})
            #     return lambda *args: CallPattern(func_pattern, list(args) if args else None)
            
            data = wildcard()

            reshaped = is_op('reshape')(data)

            slice11 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(reshaped, wildcard(), wildcard(), wildcard(), wildcard())
            slice12 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(reshaped, wildcard(), wildcard(), wildcard(), wildcard())
            slice13 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(reshaped, wildcard(), wildcard(), wildcard(), wildcard())
            slice14 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(reshaped, wildcard(), wildcard(), wildcard(), wildcard())

            slice21 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(slice11, wildcard(), wildcard(), wildcard(), wildcard())
            slice22 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(slice12, wildcard(), wildcard(), wildcard(), wildcard())
            slice23 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(slice13, wildcard(), wildcard(), wildcard(), wildcard())
            slice24 = FunctionPattern(None, wildcard()).has_attr({"Composite": "tidl.slice"})(slice14, wildcard(), wildcard(), wildcard(), wildcard())

            slices_tuple = is_tuple([slice21, slice22, slice23, slice24]) 
            concat_result = is_op('concatenate')(slices_tuple)
            
            return concat_result

        def _patch_merging_checker():
            """Checker function for patch merging pattern with validation.
            Pattern already validates structure, so checker focuses on semantic constraints.
            1. Strides of all slice layers must be 2
            2. Level 1 and level 2 slices should interchangeably have axis as height/channel
            3. Start attribute across the 2 slice levels should cover all 4 combinations - ((0,0), (0,1), (1,0), (1,1))
            """
            def checker(extract):
                # Pattern already validates: tuple structure, 4 slices, tidl.slice composites, 5 args each
                # Checker focuses on semantic constraints: strides, axes, start positions
                
                slice_ops = extract.args[0].fields  # Pattern guarantees this is valid
                slice_level_1_info = []
                slice_level_2_info = []
                
                # Extract slice parameters (pattern guarantees structure is valid)
                for i, slice_op in enumerate(slice_ops):
                    begin_level_2 = slice_op.args[1].data.asnumpy().tolist()
                    axes_level_2 = slice_op.args[3].data.asnumpy().tolist()
                    strides_level_2 = slice_op.args[4].data.asnumpy().tolist()

                    begin_level_1 = slice_op.args[0].args[1].data.asnumpy().tolist()
                    axes_level_1 = slice_op.args[0].args[3].data.asnumpy().tolist()
                    strides_level_1 = slice_op.args[0].args[4].data.asnumpy().tolist()
                    
                    slice_level_2_info.append({
                        'begin': begin_level_2,
                        'axes': axes_level_2,
                        'strides': strides_level_2
                    })

                    slice_level_1_info.append({
                        'begin': begin_level_1,
                        'axes': axes_level_1,
                        'strides': strides_level_1
                    })
                
                # Semantic constraint 1: All strides must be 2
                for info in slice_level_2_info:
                    if not all(s == 2 for s in info['strides']):
                        return False
                    
                for info in slice_level_1_info:
                    if not all(s == 2 for s in info['strides']):
                        return False
                
                # Semantic constraint 2: Axis combinations (height vs channel)
                level1_axes = set()
                level2_axes = set()
                
                for info in slice_level_1_info:
                    level1_axes.update(info['axes'])
                
                for info in slice_level_2_info:
                    level2_axes.update(info['axes'])
                
                height_axes = {2}  # Height axis in NCHW
                channel_axes = {1}  # Channel axis in NCHW
                
                level1_has_height = bool(level1_axes & height_axes)
                level1_has_channel = bool(level1_axes & channel_axes)
                level2_has_height = bool(level2_axes & height_axes)
                level2_has_channel = bool(level2_axes & channel_axes)
                
                valid_axis_combination = (
                    (level1_has_height and level2_has_channel) or
                    (level1_has_channel and level2_has_height)
                )
                
                if not valid_axis_combination:
                    return False
                
                # Semantic constraint 3: Start positions create 2x2 grid
                level1_starts = [info['begin'][0] for info in slice_level_1_info]
                level2_starts = [info['begin'][0] for info in slice_level_2_info]
                
                # Create 1-1 mapping pairs and check they cover all 4 combinations
                start_pairs = list(zip(level1_starts, level2_starts))
                expected_combinations = {(0, 0), (0, 1), (1, 0), (1, 1)}
                actual_combinations = set(start_pairs)
                
                if actual_combinations != expected_combinations:
                    return False

                return True
            return checker

        # Common patterns - Create them with 'tidl.composite' prefix which is used to differentiate 
        # patterns created using MergeComposite pass with pattern checker vs. the 'tidl' composite functions
        # created in frontend
        # Note that these patterns are created only for annotating all the pattern nodes as supported.
        # Before actually passing to TIDL, these are unpacked, as TIDL is expected to internally map these to relevant TIDL backend layers
        pattern_table = [
            ('tidl.composite.layernorm', _layernorm_pattern(), _layernorm_checker()),
            ('tidl.composite.gelu', _gelu_pattern(), _gelu_checker()),
            ('tidl.composite.patch_merging', _patch_merging_pattern(), _patch_merging_checker()),
            # ('tidl.transpose_reshape', _transpose_reshape_pattern(), _transpose_reshape_checker)
        ]

        return relay.transform.MergeComposite(pattern_table)(mod)

    # Helper functions

    # This function is to be used only for debug purpose to force support a layer
    # Every operator should be constrained with TIDL determining whether supported or not
    def _register_supported_op(self, op_name):
        """ Helper function to register an op that is supported without any constraints """
        @tvm.ir.register_op_attr(op_name, "target.tidl")
        def _func_wrapper(expr):
            return True
        return _func_wrapper

    def _register_constrained_op(self, op_name):
        """ Helper function to register an op that is supported with some constraints """
        @tvm.ir.register_op_attr(op_name, "target.tidl")
        def _func_wrapper(expr):
            return self.allow_func(op_name, expr)
        return _func_wrapper

    def allow_func(self, op_name, expr):
        """ Allow function: constraint checking is delegated to the import library """

        ### TIDL does not support scalar as the first argument
        if isinstance(expr.args[0].checked_type, relay.TensorType) and \
                  len(expr.args[0].checked_type.shape) == 0:
            return False

        if self.import_lib is None:
            # For CI testing which doesn't have import library - still run TVM passes
            return True

        # Invoke TIDL import library call to check if this op can be supported
        #print(f"Invoking TIDL Relay Import allow function for {op_name}")
        allow_fn = tvm.get_global_func("TIDL_relayAllowNode")
        return allow_fn(expr)

    # Check if an image.resize2d case is optimized by TIDL
    # Will be moved to TIDL import library once we agree on what TIDL should allow
    def _check_tidl_optimized_resize(self, expr):
        attrs, args = expr.attrs, expr.args
        old_shape = args[0].checked_type.shape
        new_h, new_w = attrs.size
        old_h, old_w = old_shape[2:4] if attrs.layout == "NCHW" else old_shape[1:3]
        scale_h = float(new_h.value) / float(old_h.value)
        scale_w = float(new_w.value) / float(old_w.value)
        # TIDL has only optimized symmetric resize with power of 2 scaling factor
        #   all other cases are supported with natural C code (slow)
        if scale_h != scale_w or not scale_h.is_integer():
            return False
        scale = int(scale_h)
        return (scale > 1 and (scale & (scale -1)) == 0)


class TIOffloadCompiler:
    """TI offload compiler module.

    This module tries to compile a given Relay IR graph to deploy on C7x devices with TIDL.
    If compilation succeeds, artifacts for heterogeneous compute on C7x, optionally with TIDL
    offload, will be generated.

    Parameters
    ----------
    platform : string
        The platform to deploy the graph on.
    version : string
        The Processor-SDK version for the platform.
    **kwargs : keyword arguments to pass what's needed for Relay IR graph conversion
        max_num_tidl_subgraphs : int
            Max number of subgraphs to run on TIDL. Use 0 for no TIDL offload (only C7x generation)
            Offload up to \<num\> TIDL subgraphs, default is 16
        tidl_tools_path : string
            Folder to TIDL tools
        artifacts_folder : string
            Folder to hold TIDL artifacts
        tensor_bits : int
            Bits for import TIDL tensor and weights, default is 8
        debug_level : int
            0, 1, 2, 3, 4 for various debug info, default is 0
        accuracy_level: int
            0 for simple calibration, 1 for advanced bias calibration, 9 for user defined,
            default is 1
        c7x_codegen : int
            Generate C7x code for TIDL-unsupported layers.  0 for disable, 1 for enable, default is 0
        advanced_options: dict
            a dictionary to overwrite default calibration options, default is {}
            advanced_options keys / values:
            - 'calibration_iterations' : int
                  number of calibration iterations, default is 50
            - 'quantization_scale_type' : int
                  0 for non-power-of-2, 1 for power-of-2, default is 0
            - 'high_resolution_optimization' : int
                  0 for disable, 1 for enable, default is 0
            - 'pre_batchnorm_fold' : int
                  0 for disable, 1 for enable, default is 1
            - 'mixed_precision_factor' : float
                  -1.0 for disable, != -1.0 for auto mixed precision calibration
            The following keys / values can only be overwritten at accuracy level 9:
            - 'activation_clipping' : int
                  0 for disable, 1 for enable
            - 'weight_clipping' : int
                  0 for disable, 1 for enable
            - 'bias_calibration' : int
                  0 for disable, 1 for enable
            - 'channel_wise_quantization' : int
                  0 for disable, 1 for enable
        ti_internal_nc_flag: int
            Internal use only, default is 0x641
    """

    default_od_options = [
        'object_detection:meta_layers_names_list',
        'object_detection:meta_arch_type',
    ]

    def __init__(self, platform="J7", tidl_tools_path=None, enable_tidl_offload=True,
                 compile_for_device=1, reuse_tidl_artifacts=False, delegate_options={}):
        if supported_platform(platform):
            self.delegate_options = delegate_options
            # TODO: Ideally this entire code should move to TIDL or reuse existing TIDL code
            # TVM should only be pass through for options, TIDL should interpret and
            # parse the options, set default values, etc. as needed)
            self.tidl_platform = platform_map(platform)
            self.tidl_target = "tidl"
            self.tidl_tools_path = tidl_tools_path
            self.artifacts_folder = delegate_options.get("artifacts_folder", None)
            self.debug_level = delegate_options.get("debug_level", 0)
            self.tensor_bits = delegate_options.get("tensor_bits", 8)
            self.max_num_tidl_subgraphs = delegate_options.get("max_num_tidl_subgraphs", (16 if enable_tidl_offload else 0))
            self.c7x_codegen = delegate_options.get("advanced_options:c7x_codegen", 1)
            self.compile_for_device = compile_for_device
            self.od_options = {}
            object_detection_prefix = 'object_detection:'

            for k, v in delegate_options.items():
                if(k.startswith(object_detection_prefix)):
                    self.od_options[k] = v

            if enable_tidl_offload:
                self.tidl_import_lib = os.path.join(self.tidl_tools_path, "tidl_model_import_relay.so")
        
        else:
            sys.exit("Unsupported TIDL platform: " + platform)
        assert self.artifacts_folder, "artifacts_folder must be specified for TIDL compilation"
        self.temp_folder = os.path.join(self.artifacts_folder, 'tempDir/')
        # Set environment variable for C++ codegen to find temp folder
        os.environ["TIDL_ARTIFACTS_TEMP_FOLDER"] = self.temp_folder

        # Create and set up the TIDL context for C++ codegen
        CreateTIDLContext = tvm.get_global_func("tidl.CreateTIDLContext")
        self.tidl_context = CreateTIDLContext(self.artifacts_folder, self.tidl_platform,
                                              self.c7x_codegen, 0, self.compile_for_device)
        # Enter the context to make it active
        EnterTIDLContext = tvm.get_global_func("tidl.EnterTIDLContext")
        EnterTIDLContext(self.tidl_context)

        if self.debug_level:
            os.environ["TIDL_RELAY_IMPORT_DEBUG"] = str(self.debug_level)
        
        self.tidl_relay_import_debug = os.environ.get("TIDL_RELAY_IMPORT_DEBUG")
        self.reuse_tidl_artifacts = reuse_tidl_artifacts

    def enable(self, mod_orig, params, graph_input_list):
        """ Enable TIDL compilation

        This function tries to partition and compile the given Relay IR graph.
        If it succeeds, artifacts for heterogeneous compute with TIDL will be
        generated, and the partitioned graph will be returned. Otherwise, it will
        return None.

        Parameters
        ----------
        mod_orig : tvm.relay.Module
            Original Relay IR graph
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay
        graph_input_list: dictionary OR list of dictionaries (for multiple calibration data)
            A dictionary where the key is input name and the value is input tensor

        Returns
        -------
        mod : tvm.relay.Module
            Paritioned graph with subgraphs to run with TIDL
        status: int
            Status of TIDL compilation:
                1  - compilation success
                -1 - compilation failure
                0  - no compilation due to missing TIDL tools or user specified no TIDL offload
        """

        #
        # When self.max_num_tidl_subgraphs is 0, this means the caller wants *NO* TIDL offload
        #

        tidl_od_meta_arch_type = -1
        tidl_od_num_graph_outputs = 1
        tidl_od_meta_layers_names_list = ""
        tidl_od_postproc_inputs = []
        import_lib = None

        all_nodes_dict_orig = get_all_nodes(mod_orig['main'])
        total_nodes_original = 0
        for node in all_nodes_dict_orig:
            if isinstance(node, relay.expr.Call):
                total_nodes_original += 1
        print(f"Total Nodes - {total_nodes_original}")

        # TIDL-specific handling of object detection specifics. Skip if user doesn't want TIDL offload
        if (self.max_num_tidl_subgraphs > 0 and self.od_options and
            'object_detection:meta_layers_names_list' in self.od_options and
            'object_detection:meta_arch_type' in self.od_options):
            tidl_od_meta_layers_names_list = self.od_options['object_detection:meta_layers_names_list']
            tidl_od_meta_arch_type = self.od_options['object_detection:meta_arch_type']
            tidl_od_num_graph_outputs = len(mod_orig['main'].body.fields) \
                    if isinstance(mod_orig['main'].body, relay.expr.Tuple) else 1
            # call TIDL_relayGetODMetaArchInfo to get the following info
            # input node names, output shapes and output dtypes
            import_lib = ctypes.CDLL(self.tidl_import_lib, mode=ctypes.RTLD_GLOBAL)
            od_postproc_info = ODPostProcInfo()
            import_lib_get_od_info = tvm.get_global_func("TIDL_relayGetODMetaArchInfo")
            if(import_lib_get_od_info(tidl_od_meta_arch_type, tidl_od_num_graph_outputs, tidl_od_meta_layers_names_list,
                                    ctypes.cast(ctypes.byref(od_postproc_info), ctypes.c_void_p)) != 0):
                print("\n\nError fetching OD MetaArch Info!!!\n\n")
                return mod_orig, 0
            tidl_od_postproc_inputs = [name.value.decode() for name
                                        in od_postproc_info.in_node_names[:od_postproc_info.num_in_nodes]]
            tidl_od_output_shapes = [(node.n, node.channel, node.height, node.width) for node
                                        in od_postproc_info.out_nodes[:od_postproc_info.num_out_nodes]]
            element_type_map = {
                0: "uint8",
                1: "int8",
                6: "float32",
                8: "int64"
            }
            tidl_od_output_dtypes = [element_type_map[node.element_type] for node
                                     in od_postproc_info.out_nodes[:od_postproc_info.num_out_nodes]]

            # from meta data, get number of outputs and their shapes, use those to define operator,
            # tidl_odpostproc, that can be offloaded to TIDL, default impl just return zeros,
            # This is to help get the graph output type (tensors and shapes) correct early on
            mod_orig = prune_graph_for_ODPostProc_inputs(mod_orig, tidl_od_postproc_inputs,
                    tidl_od_output_shapes, tidl_od_output_dtypes)

        # Skip TIDL import and C7x code generation.  Proceed directly to
        # re-build the C7x deployable module and the Arm deployable module,
        # reusing the existing source in the tempDir from the previous compilation.
        if self.c7x_codegen > 0 and os.environ.get("TIDL_REBUILD_ONLY") != None:
            # bind_params will remove weights from arguments of main(), so that
            #     main() function API will be the same in TIDL_REBUILD_ONLY path.
            mod_orig['main'] = relay.build_module.bind_params_by_name(mod_orig['main'], params)
            mod_orig = relay.transform.DynamicToStatic()(mod_orig)
            return enable_c7x_mod(self, mod_orig, params, 0), 0

        # (Backward compatible) if single calibration image/data/dict, convert to list
        if not isinstance(graph_input_list, list):
            graph_input_list = [ graph_input_list ]
        # Ensure calibration image parameter names are same names as parameters in model
        mod_params_names = [ var.name_hint for var in mod_orig['main'].params ]
        for name_val_dict in graph_input_list:
            for name in name_val_dict.keys():
                if name not in mod_params_names:
                    raise Exception(f"Specified input name, {name}, is not found in the model.")

        #============= Find data layout of the original graph =============
        data_layout = find_data_layout(mod_orig)
        has_qnn_ops = find_qnn_ops(mod_orig)


        # Open TIDL import library. Skip if user doesn't want TIDL offload
        if self.max_num_tidl_subgraphs > 0:
            if os.path.exists(self.tidl_import_lib):
                if import_lib == None:
                    import_lib = ctypes.CDLL(self.tidl_import_lib, mode=ctypes.RTLD_GLOBAL)
                tidl_relay_init = tvm.get_global_func("TIDL_relayInit")
                is_nchw = data_layout == "NCHW"
                # Convert all delegate_options values to strings for C++ compatibility
                delegate_options_str = {k: str(v) for k, v in self.delegate_options.items()}
                if(tidl_relay_init(is_nchw, delegate_options_str) != 0):
                    print('\n\nTIDL initialization failed!!!\n\n')
                    return mod_orig, 0
            else:
                import_lib = None # Continue with graph annotation and partition for CI testing

            # Register TIDL annotation functions
            tidl_annotation = TIDLAnnotation(self.tidl_platform, import_lib)
            tidl_annotation.register_allowed_ops()

        with open(os.path.join(self.temp_folder, "relay_graph.orig.txt"), "w") as relay_txt:
            print(mod_orig.astext(show_meta_data=False), file=relay_txt)

        mod = prepare_graph_for_partitioning(mod_orig, has_qnn_ops, params)

        with open(os.path.join(self.temp_folder, "relay_graph.prepared.txt"), "w") as relay_txt:
            print(mod.astext(show_meta_data=False), file=relay_txt)
        mod_pre = mod

        #============= Reject dynamic shape/network for now ==============
        if find_dynamic_shape(mod):
            print("\n\nDynamic shape/network not supported by TVM+TIDL yet!!!\n\n")
            return mod_orig, 0

        #============= Graph annotation ==============
        # TIDL annotation and TIDL graph partitioning.
        # Skip when not performing TIDL offload (max_num_tidl_subgraphs == 0)
        if self.max_num_tidl_subgraphs > 0:
            mod = tidl_annotation.merge_sequential_ops(mod)

            # Invoking TIDL Relay Import allow function for Composite Functions
            allow_fn = tvm.get_global_func("TIDL_relayAllowNode")
            denylist_update_fn = tvm.get_global_func("TIDL_relayUpdateDenyList")
            all_nodes = get_all_nodes(mod['main'])
            for node in all_nodes:
                if isinstance(node, relay.expr.Call) and isinstance(node.op, relay.Function):
                    func = node.op
                    if hasattr(func, "attrs") and "Composite" in func.attrs and self.tidl_target in func.attrs["Composite"] and "composite" not in func.attrs["Composite"]:
                        result = allow_fn(node)
                        if(result == False):
                            # Unpack the composite function
                            mod = unpack_specific_composites(mod, func.attrs["Composite"], node.span.source_name.name)
                            # add span name to the deny list, to avoid TIDL offload of the relay decomposed operators
                            denylist_update_fn(node.span.source_name.name)

            mod = relay.transform.AnnotateTarget(self.tidl_target)(mod)
            with open(os.path.join(self.temp_folder, "relay_graph.annotated.txt"), "w") as relay_txt:
                print(mod.astext(show_meta_data=False), file=relay_txt)

            #============= Graph partition ==============
            mod = relay.transform.MergeCompilerRegions()(mod)
            mod = relay.transform.PartitionGraph()(mod)
            with open(os.path.join(self.temp_folder, "relay_graph.partitioned.txt"), "w") as relay_txt:
                print(mod.astext(show_meta_data=False), file=relay_txt)
            mod = prune_subgraphs_with_overlimit_inputs_outputs(mod, in_out_limit=32,
                                                                compiler=self.tidl_target)

            # If more than 16 TIDL subgraphs, pull functions back into main function
            # and out of TIDL offload
            mod = prune_subgraphs(mod, compiler=self.tidl_target,
                                  num_subgraphs_to_keep=self.max_num_tidl_subgraphs,
                                  min_mac_threshold=None)
            # After partitioning, unwind tidl.composite fused pattern combinations
            mod = unpack_composites(mod, "tidl.composite", mod.get_global_vars())
            mod = flatten_tuple_params(mod, self.tidl_target)
            mod = relay.transform.InferType()(mod)
        else:  # no TIDL offload, need to unpack tidl composites
            mod = unpack_composites(mod, self.tidl_target, mod.get_global_vars())
            mod = relay.transform.InferType()(mod)

        #============= Post-partition transformations  ==============
        # ConvertLayout pass does not yet work properly for graph with qnn ops
        # We could optionally lower the RelayIR with relay.qnn.transform.CanonicalizeOps() and
        # relay.transform.FoldConstant() before ConvertLayout pass.  However, with the presence
        # of TIDL subgraph, CanonicalizeOps() somehow caused int64 args/buffers (should be int32)
        # in MakePackedAPI()/ArgBinder::BindDLTensor(), which inserted asserts in generated code
        # and in turn caused assert fails at inference time:
        #     TVMError: Check failed: ret == 0 (-1 vs. 0) : Assert fail: (((tir.tvm_struct_get(arg1, 0, 5) == (uint8)0) && (tir.tvm_struct_get(arg1, 0, 6) == (uint8)64)) && (tir.tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be int64
        # Skip applying ConvertLayout pass to quantized models for now.
        # TODO: revisit this issue after finishing quantized model support for TIDL offload
        if not has_qnn_ops and data_layout != 'NCHW':
            with tvm.transform.PassContext(opt_level=3):
                convert_pass = [relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default']})]
                mod = tvm.transform.Sequential(convert_pass)(mod) # only affects non-TIDL subgraphs
                mod = relay.transform.InferType()(mod)

        with open(os.path.join(self.temp_folder, "relay_graph.import.txt"), "w") as relay_txt:
            print(mod.astext(show_meta_data=False), file=relay_txt)

        mod_final = mod
        status = 1
        num_imported_sgs = len(get_tidl_subgraphs(mod, self.tidl_target))
        print(f"TVM Relay detected {num_imported_sgs} subgraphs")

        # Check the number of Op nodes left in Module main function
        # tidl_subgraph_x is a GlobalVar, and is not counted as an Op node
        # Number of offloaded is (total op nodes - left op nodes)
        all_nodes_dict = get_all_nodes(mod['main'])
        num_nodes = 0
        for node in all_nodes_dict:
            if isinstance(node, relay.expr.Call):
                num_nodes += 1
        num_offloaded_nodes =  total_nodes_original - (num_nodes - num_imported_sgs)

        if not self.reuse_tidl_artifacts:
            # If reusing TIDL artifacts, skip creation of TIDLImport object and corresponding calls (these mainly create TIDL subgraph artifacts)
            # Any TIDL subgraph related artifacts will be reused from tempDir
            # Only update to IR Module as part of this code is to mark relay expressions corresponding to TIDL layers with let
            # This is a debug feature, and will not be available in case of artifacts re-use.
            # For any debug related runs, compile artifacts from scratch without re-use

            #================ Import the graph to TIDL, if caller specified =====================
            if self.max_num_tidl_subgraphs > 0 and self.tidl_tools_path is not None:
                print(f"Final number of subgraphs created are : {num_imported_sgs}, Offloaded Nodes - {num_offloaded_nodes}, Total Nodes - {total_nodes_original}")
                if (import_lib is not None):
                    tidl_import = TIDLImport(import_lib,
                                            self.tidl_tools_path, self.artifacts_folder,
                                            self.tidl_target, self.tidl_platform,
                                            data_layout, self.tensor_bits,
                                            tidl_od_meta_arch_type,
                                            tidl_od_num_graph_outputs,
                                            tidl_od_meta_layers_names_list,
                                            tidl_od_postproc_inputs)
                    print("Generating subgraph boundary tensors for calibration...")
                    subgraph_tensors_list, relay_quantization, relay_etypes = generate_subgraph_tensors(
                                    self.tidl_target, mod, params, graph_input_list, self.temp_folder,
                                    data_layout, has_qnn_ops)
                    print("Importing subgraph into TIDL...")
                    num_imported_sgs = tidl_import.import_relay_ir(mod, params, subgraph_tensors_list,
                                                        relay_quantization, relay_etypes, has_qnn_ops)
                    _ctypes.dlclose(import_lib._handle)
                    if num_imported_sgs >= 0:
                        print(f"TIDL import of {num_imported_sgs} Relay IR subgraphs succeeded.")
                        if num_imported_sgs > 0 and self.tidl_relay_import_debug == "4":
                            generate_tidl_layer_tensors(self.tidl_target, mod, params,
                                                        graph_input_list, self.temp_folder,
                                                        data_layout, has_qnn_ops)
                        print("TIDL artifacts are stored at " + self.artifacts_folder)
                        mod_final, status = mod, 1        # TIDL Compilation success
                    else:
                        print("TIDL import of Relay IR graph failed.")
                        mod_final, status = mod_pre, -1  # TIDL Compilation failure
                else:
                    print("TIDL import lib does not exist. TIDL import skipped.")
                    mod_final, status = mod_pre, 0       # No TIDL compilation
            else:
                if self.tidl_tools_path is None:
                    print("TIDL tools path is not set. TIDL import skipped.")
                if self.max_num_tidl_subgraphs == 0:
                    print("max_num_tidl_subgraphs is 0. TIDL import skipped.")
                mod_final, status = mod_pre, 0           # No TIDL compilation

        # Build the c7x deployable module that the C7x TVM C runtime can execute
        # This also will invoke C7x optimization passes
        if (self.c7x_codegen > 0):
            mod_final = enable_c7x_mod(self, mod_final, params, num_imported_sgs)

        return mod_final, status

@tvm._ffi.register_object("tidl.TIDLContext")
class TIDLContext(tvm.runtime.Object):
    def __init__(self,
                 artifacts_directory, platform):

        self.__init_handle_by_constructor__(_ffi_tidl_api.TIDLContext, artifacts_directory, platform)

    def __enter__(self):
        _ffi_tidl_api.EnterTIDLContext(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_tidl_api.ExitTIDLContext(self)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _ffi_tidl_api.GetCurrentTIDLContext()

class build_config():
    """ Configs TVM relay module build

    Parameters
    ----------
    ti_offload_compiler : TIOffloadCompiler
      artifacts_folder : string : where compilation artifacts are stored
      platform : string : TI SoC platform
      c7x_codegen : int: whether to generate C7x code for TIDL-unsupported layers
    gen_c7x_mod_enabled : int
      Internal option, building a C7x deployable module or an Arm deployable module

      c7x_codegen_enabled == 0: Disable C7x code generation, all TIDL-unsupported layers run on Arm
                           building an Arm deployable module
      c7x_codegen_enabled >  0: Enable  C7x code generation, all TIDL-unsupported layers run on C7x
        In the compilation flow, first we build a C7x deployable module (c7x_deploy_mod.out),
        then we embed C7x deployable module as a single node ("tidl_tvm_0") into Arm wrapper
        deployable module (deploy_graph.json, deploy_lib.so)
        - gen_c7x_mod_enabled = 1: building a C7x deployable module
        - gen_c7x_mod_enabled = 0: building an Arm wrapper deployable module

      TBD: we leave c7x_codegen_enabled==9 for now as a debug option to generate generic C code
        instead of optimized C7x code.  Need to disable vectorization when generating generic C
        code.  Will decide later if this debug option is useful.
    """
    def __init__(self, ti_offload_compiler=None, gen_c7x_mod_enabled=0):
        artifacts_folder = None
        platform = "J7"
        c7x_codegen_enabled = 0
        compile_for_device = 1
        if ti_offload_compiler != None:
            artifacts_folder    = ti_offload_compiler.artifacts_folder
            platform            = ti_offload_compiler.tidl_platform
            c7x_codegen_enabled = ti_offload_compiler.c7x_codegen
            compile_for_device  = ti_offload_compiler.compile_for_device
        assert artifacts_folder, "artifacts_folder must be specified for TVM+TIDL compilation"
        self.debug_c7x_codegen = False
        if (os.environ.get("TIDL_C7X_CODEGEN_DEBUG") != None) and (gen_c7x_mod_enabled != 0):
            self.debug_c7x_codegen = True
        self.temp_folder = os.path.join(artifacts_folder, "tempDir")
        if (c7x_codegen_enabled == 1 and gen_c7x_mod_enabled == 0):
            self.temp_folder = None
        CreateTIDLContext = tvm.get_global_func("tidl.CreateTIDLContext")
        self.tidl_context = CreateTIDLContext(artifacts_folder, platform, c7x_codegen_enabled,
                                              gen_c7x_mod_enabled, compile_for_device)
        self.tvm_context  = tvm.transform.PassContext(opt_level=3,
                                      config={'tir.disable_vectorize': (c7x_codegen_enabled == 9)})

    def __enter__(self):
        if self.debug_c7x_codegen:
            self.prev_logging_level = logging.getLogger().getEffectiveLevel()
            importlib.reload(logging)
            logging.basicConfig(level=logging.DEBUG)
            os.environ["TIDL_C7X_CODEGEN_DEBUG_BEGIN"] = "1"
        if self.temp_folder:
            os.environ["TIDL_ARTIFACTS_TEMP_FOLDER"] = self.temp_folder
        self.tidl_context.__enter__()
        self.tvm_context.__enter__()

    def __exit__(self, ctx_type, ctx_value, ctx_trace):
        self.tidl_context.__exit__(ctx_type, ctx_value, ctx_trace)
        self.tvm_context.__exit__(ctx_type, ctx_value, ctx_trace)
        if self.temp_folder:
            os.environ.pop("TIDL_ARTIFACTS_TEMP_FOLDER")
        if self.debug_c7x_codegen:
            os.environ.pop("TIDL_C7X_CODEGEN_DEBUG_BEGIN")
            importlib.reload(logging)
            logging.basicConfig(level=self.prev_logging_level)

def remove_tidl_params(params):
    """ Remove params used by TIDL subgraphs from deployable module params

    The params used by TIDL subgraph are already imported into TIDL subgraph
    network artifacts.  They will not used by the remaining non-TIDL parts
    of the graph.  Remove them from the deployable module params.

    Parameters
    ----------
    params : dict of str to tvm.NDArray
        At return, mutable dict object is updated, with "tidl_" params removed
    """
    tidl_params = [ key for key in params if key.find("tidl_") == 0 ]
    for tidl_param in tidl_params:
        del params[tidl_param]
