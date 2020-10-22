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
"""TIDL backend compiler"""

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
import tvm
from tvm import relay
import tvm.ir
from tvm.topi.util import get_const_tuple
from tvm.relay.dataflow_pattern import is_op, is_constant, wildcard, is_tuple_get_item
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import Tuple, GlobalVar
from tvm.relay.function import Function
from tvm.contrib import graph_runtime
#import tvm.relay.op.contrib.tidl as tidl_annotation
from .tidl_reduce_subgraph_size import reduce_subgraph_size
from .tidl_visualize import visualize_relay_graph

tidl_annotations_registered = False

import tvm._ffi

from . import _ffi_tidl_api

def traverse_expr(node, node_dict):
    if node in node_dict:
        return
    if isinstance(node, tvm.ir.Op):
        return
    node_dict[node] = len(node_dict)

def find_data_layout(mod):
    all_nodes = {}
    traverse_func = functools.partial(traverse_expr, node_dict=all_nodes)
    relay.analysis.post_order_visit(mod['main'], traverse_func)
    data_layout = None
    for node in all_nodes:
        if isinstance(node, relay.expr.Call) and node.op.name == 'nn.conv2d':
            data_layout = node.attrs.data_layout
            break
    return data_layout

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
        node_name = str(all_nodes[node])
        # N is Call w/one output --> result is "N"
        # N is Call w/tuple output --> result is ["N", "N:1", ...]
        if isinstance(node, relay.expr.Call):
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

def find_out_nodes(all_nodes, this_node):
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
            if this_node == node.tuple_value:
                output_nodes = output_nodes + find_out_nodes(all_nodes, node)
        elif isinstance(node, relay.expr.Tuple):
            if this_node in node.fields:
                tuple_node_outs = find_out_nodes(all_nodes, node)
                if len(tuple_node_outs) == 0:
                    # this is an output node
                    output_nodes.append(str(all_nodes[node]))
                else:
                    # this is an input node to another node
                    output_nodes = output_nodes + tuple_node_outs

    return output_nodes

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

    Returns
    -------
    in_out_nodes : InOutNodes
        Structure that stores indices of input nodes and output nodes
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
            # only use 1 batch per input for calibration
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
        abs_signed_max   = 128.0 if (tensor_bits == 8) else 32768.0
        abs_unsigned_max = 255.0 if (tensor_bits == 8) else 65535.0

        if min_values[i] >= 0:
            # quantize to Uint8 or Uint16
            sign = 0
            scale = abs_unsigned_max/max_value
            quant_min, quant_max = 0.0, abs_unsigned_max
        else:
            # quantize to Int8 or Int16
            sign = 1
            scale = abs_signed_max/max_value
            quant_min, quant_max = (- abs_signed_max), (abs_signed_max - 1.0)

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
            # only use 1 batch for calibration
            input_tensor = input_tensor[0, :]
            # change layout to CxHxW to use numpy.flatten to change to 1-d array
            if data_layout == "NHWC" and len(input_tensor.shape) == 3:
                input_tensor = input_tensor.transpose(2, 0, 1)

            tensor_norm = np.multiply(input_tensor, scale)
            tensor_quant = np.rint(tensor_norm)
            tensor_quant = np.clip(tensor_quant, quant_min, quant_max)
            output = tensor_quant.flatten()   # works only if tensor_quant is in "CxHxW" format

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

def unpack_composites(mod):
    """Unpack all composite functions in the module by replacing composite call nodes with the
    ops inside the composite function."""

    class Unpacker(ExprMutator):
        """Unpacks composite functions."""
        def __init__(self):
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, Function):
                if call.op.attrs and call.op.attrs['Composite'] != "":
                    # unpack the function back into new main function.
                    var_map = {}
                    for arg, param in zip(call.args, call.op.params):
                        var_map[param] = super().visit(arg)
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
                call.args = new_args
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

class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, call):
        if call.op.name == "multiply":
            if isinstance(call.args[1], tvm.relay.expr.Constant):
                data = call.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return call.args[0]
            if isinstance(call.args[0], tvm.relay.expr.Constant):
                data = call.args[0].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return call.args[1]
        return super().visit_call(call)

class RemoveTrainingOperators(ExprMutator):
    """
    Removes operators that apply to network training but not to inference.
    """
    # Dropout layer produces a tuple
    def visit_tuple_getitem(self, t):
        expr = t.tuple_value
        if t.index == 0  and \
           isinstance(expr, relay.expr.Call) and \
           expr.op.name in ["nn.dropout", "nn.dropout_raw"]:
            return expr.args[0]
        return super().visit_tuple_getitem(t)

def generate_subgraph_tensors(tidl_target, mod, params, graph_input_list, temp_folder, save_output=False):
    """Creates calibration graph from mod and executes on the cpu to generate boundary tensors.
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    calib_mutator = CalibrationGraphMutator(tidl_target)
    mod_tvm["main"] = calib_mutator.make_calibration_graph(mod_tvm["main"])

    # Build and execute calibration graph to get outputs
    # Use opt_level=0 to avoid optimizations which modify the module (could change original module)
    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
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
                subgraph_tensors[calib_mutator.name_map[i]] = res
                if save_output:
                    file_name = os.path.join(temp_folder, calib_mutator.name_map[i] + ".txt")
                    np.savetxt(file_name, res.flatten(), fmt='%10.5f')
        subgraph_tensors_list.append(subgraph_tensors)

    return subgraph_tensors_list

def generate_tidl_layer_tensors(tidl_target, mod, params, graph_input_list, temp_folder,
                                data_layout):
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
    print("----------- after call rewriting -----------")
    print(mod_tvm.astext(show_meta_data=False))
    calib_perlayer_mutator = CalibrationPerLayerMutator(tidl_target)
    mod_tvm["main"] = calib_perlayer_mutator.make_calibration_graph(mod_tvm["main"])
    #print("----------- after additional outputs -----------")
    #print(mod_tvm.astext(show_meta_data=False))

    # Build and execute calibration graph to get outputs
    # Use opt_level=0 to avoid optimizations which modify the module (could change original module)
    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(**params)

    graph_input = graph_input_list[-1]
    mod.set_input(**graph_input)
    mod.run()

    for i in range(mod.get_num_outputs()):
        tensor = mod.get_output(i).asnumpy()
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
    def __init__(self, subgraphs_to_remove, mod, new_mod, rename_starting_from_0=True):
        ExprMutator.__init__(self)
        self.subgraphs_to_remove = subgraphs_to_remove
        self.mod = mod
        self.new_mod = new_mod
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
                    new_name = name.split('_')[0] + "_" + str(self.count)
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
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        if len(mod[name].params) != 1 \
           or isinstance(mod[name].params[0].checked_type, relay.TupleType):
            subgraph_names_to_remove.append(name)
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
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
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        num_macs = relay.analysis.get_total_mac_number(mod[name])
        subgraph_with_macs.append([name, num_macs])
    subgraph_with_macs = sorted(subgraph_with_macs, key=lambda x: int(x[1]))
    subgraphs_to_prune = subgraph_with_macs[:-num_subgraphs_to_keep]
    if min_mac_threshold:
        # Also remove all subgraphs under the minimum threshold.
        subgraphs_to_prune += [[x[0], x[1]] for x in subgraph_with_macs if x[1] < min_mac_threshold]
    subgraph_names_to_remove = {x[0] for x in subgraphs_to_prune}
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod

def subgraph_cfg_gen(artifacts_folder, subgraph_id, data_layout,
                     input_scale, input_signed, output_scale, output_signed):
    r""" Generate subgraph configuration file to be used by TIDL runtime

    Parameters
    ----------
    input_scale : vector
        scaling factor to convert floating point TVM tensors to 8-bit TIDL inputs,
        where TIDL_input[i] = TVM_tensor[i] * input_scale[i]
    input_signed : vector
        indicating whether input tensor to TIDL is signed (1) or unsigned (0)
    output_scale : vector
        scaling factor to convert 8-bit TIDL outputs to floating point TVM tensors,
        where TVM_tensor[i] = TIDL_input[i] / output_scale[i]
    output_signed : vector
        indicating whether output tensor of TIDL is signed (1) or unsigned (0)

    Returns
    -------
    """

    def print_list(in_list):
        str0 = str(in_list)
        str1 = str0.replace("[", "")
        str2 = str1.replace("]", "")
        str3 = str2.replace(",", "", len(in_list)-1)
        return str3

    if data_layout == "NCHW":
        layout_is_nchw = 1
    else:
        layout_is_nchw = 0
    out_conv_type = [0 for i in range(len(output_scale))]
    out_is_nchw = [layout_is_nchw for i in range(len(output_scale))]

    sub_graph_cfg = os.path.join(artifacts_folder, "subgraph" + str(subgraph_id) + ".cfg")
    sub_graph_net_file = "./tidl_subgraph" + str(subgraph_id) + "_net.bin"
    sub_graph_params_file = "./tidl_subgraph" + str(subgraph_id) + "_params.bin"
    with open(sub_graph_cfg, 'w') as cfg_file:
        cfg_file.write("netBinFile    = {}\n".format(sub_graph_net_file))
        cfg_file.write("paramsBinFile = {}\n".format(sub_graph_params_file))
        cfg_file.write("inConvType    = 0\n")
        cfg_file.write("inIsSigned    = {}\n".format(input_signed))
        cfg_file.write("inScaleF2Q    = {}\n".format(round(input_scale, 2)))
        cfg_file.write("inIsNCHW      = {}\n".format(layout_is_nchw))
        cfg_file.write("outConvType   = {}\n".format(print_list(out_conv_type)))
        cfg_file.write("outIsSigned   = {}\n".format(print_list(output_signed)))
        cfg_file.write("outScaleF2Q   = {}\n".format(print_list(output_scale)))
        cfg_file.write("outIsNCHW     = {}\n".format(print_list(out_is_nchw)))

def subgraph_calibration(calib_tool, subgraph_id, input_quant_vec_list, input_signed,
                         temp_folder,
                         net_file, params_file, platform="AM57", tidl_tensor_bits=8):
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
            if tidl_tensor_bits == 8:
                if input_signed[i] == 1:
                    input_quant_vec[i].astype('int8').tofile(fid)
                else:
                    input_quant_vec[i].astype('uint8').tofile(fid)
            else:
                if input_signed[i] == 1:
                    input_quant_vec[i].astype('int16').tofile(fid)
                else:
                    input_quant_vec[i].astype('uint16').tofile(fid)
    fid.close()

    if platform == "J7":
        import_lib_postprocess = tvm.get_global_func("TIDL_relayPostProcessNet")
        import_ret = import_lib_postprocess(len(input_quant_vec_list))
        return (import_ret == 0), 123  ## TODO: do we need dataQ for J7?

    output_tmp_file = temp_folder + 'precalib_net.bin'
    shutil.copyfile(net_file, output_tmp_file)

    calib_config_file = temp_folder + 'configFilesList.txt'
    quant_config_file = temp_folder + 'quant_stats_config.txt'
    with open(calib_config_file, 'w') as config_file:
        config_file.write('1 ' + quant_config_file + '\n')
        config_file.write('0\n')

    with open(quant_config_file, 'w') as quant_file:
        quant_file.write('rawImage    = 1\n')
        quant_file.write('numFrames   = 1\n')
        quant_file.write('preProcType = 0\n')
        quant_file.write('inData      = {}\n'.format(calib_raw_image))
        quant_file.write('outData     = {}\n'.format(temp_folder + 'stats_tool_out.bin'))
        quant_file.write('traceDumpBaseName  = {}\n'.format(temp_folder + 'trace_dump_'))
        quant_file.write('updateNetWithStats = 1\n')
        quant_file.write('outputNetBinFile   = {}\n'.format(net_file))
        quant_file.write('paramsBinFile      = {}\n'.format(params_file))
        quant_file.write('netBinFile         = {}\n'.format(output_tmp_file))

    # Invoke TIDL emulation to calibrate
    try:
        proc = subprocess.Popen([calib_tool, calib_config_file], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        console_out = out.decode('ascii')
        error = err.decode('ascii')
        print(console_out)
    except: # pylint: disable=bare-except
        print("TIDL calibration crashed")
        return False, None

    # Find output dataQs from calibration console output
    if console_out.find('error') == -1 and console_out.find('ERROR') == -1 and error == '':
        output_data_token = "Number of output dataQ:"
        out_buf_ind = console_out.rfind(output_data_token)
        if out_buf_ind == -1:
            print("TIDL calibration failed - can't find number of output buffers.")
            status, out_data_q = False, None
        else:
            last_line = console_out.split(output_data_token, 1)[1]
            num_outputs = int(last_line.split(". Output dataQ:", 1)[0])
            out_quants = last_line.split(". Output dataQ:", 1)[1]
            quants = out_quants.split("End of output dataQ", 1)[0]
            outq_str = re.findall(r"\d+", quants)
            outq = list(map(int, outq_str))
            if num_outputs != len(outq):
                print("TIDL calibration failed - can't find all outputQ's")
                status, out_data_q = False, None
            else:
                status, out_data_q = True, outq
    else:
        print("TIDL calibration failed.")
        print(error)
        status, out_data_q = False, None

    return status, out_data_q

class TIDLconfigParams(ctypes.Structure):
    """ TIDL config parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('numParamBits', ctypes.c_int),
                ('quantRoundAdd', ctypes.c_int),
                ('inQuantFactor', ctypes.c_int),
                ('inElementType', ctypes.c_int),
                ('inNumChannels', ctypes.c_int),
                ('inHeight', ctypes.c_int),
                ('inWidth', ctypes.c_int)]

class Conv2dParams(ctypes.Structure):
    """ Conv2d parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('num_in_channels', ctypes.c_int),
                ('num_out_channels', ctypes.c_int),
                ('num_groups', ctypes.c_int),
                ('stride_h', ctypes.c_int), ('stride_w', ctypes.c_int),
                ('dilation_h', ctypes.c_int), ('dilation_w', ctypes.c_int),
                ('pad_t', ctypes.c_int), ('pad_l', ctypes.c_int),
                ('pad_b', ctypes.c_int), ('pad_r', ctypes.c_int),
                ('kernel_h', ctypes.c_int), ('kernel_w', ctypes.c_int),
                ('kernel_layout', ctypes.c_char_p),
                ('weights_array', ctypes.c_void_p),
                ('weights_type', ctypes.c_char_p)]

class BatchNormParams(ctypes.Structure):
    """ BatchNorm parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('num_params', ctypes.c_int),
                ('params_dtype', ctypes.c_char_p),
                ('gama', ctypes.c_void_p),
                ('beta', ctypes.c_void_p),
                ('mean', ctypes.c_void_p),
                ('var', ctypes.c_void_p),
                ('epsilon', ctypes.c_float),
                ('center_enable', ctypes.c_int),
                ('scale_enable', ctypes.c_int)]

class PoolingParams(ctypes.Structure):
    """ Pooling parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('kernel_h', ctypes.c_int),
                ('kernel_w', ctypes.c_int),
                ('stride_h', ctypes.c_int),
                ('stride_w', ctypes.c_int),
                ('pad_h', ctypes.c_int),
                ('pad_w', ctypes.c_int)]

class MulParams(ctypes.Structure):
    _fields_ = [('scale', ctypes.c_float)]

class InOutNodes(ctypes.Structure):
    """ Input/output nodes defined in ctypes for passing to TIDL C library """
    _fields_ = [('this_node', ctypes.c_char_p),
                ('num_in_nodes', ctypes.c_int), ('num_out_nodes', ctypes.c_int),
                ('in_nodes', ctypes.c_void_p), ('out_nodes', ctypes.c_void_p)]

class TensorDescriptor(ctypes.Structure):
    """ Input/output tensor descriptor for TIDL subgraphs """
    _fields_ = [('input_scale', ctypes.c_double),
                ('input_signed', ctypes.c_int),
                ('channel', ctypes.c_int),
                ('height', ctypes.c_int),
                ('width', ctypes.c_int),
                ('name', ctypes.c_char_p)]

class TIDLImport:
    """TIDL import module.
    Parameters
    ----------
    import_lib : ctypes.CDLL
        TIDL import library
    calib_tool : string
        TIDL calibration tool file
    artifacts_folder : string
        Directory path to hold the artifacts
    tidl_target : string
        TIDL compilation target
    data_layout : string
        Data layout, "NCHW" or "NHWC"
    tidl_tensor_bits : int
        Number of bits for tidl tensors (and consequently params on J7)
    """
    def __init__(self, import_lib, calib_tool, artifacts_folder,
                 tidl_target="tidl", tidl_platform="AM57", data_layout="NCHW",
                 tidl_tensor_bits=8):
        self.import_lib = import_lib
        self.calib_tool = calib_tool
        self.artifacts_folder = artifacts_folder
        self.tidl_target = tidl_target
        self.tidl_platform = tidl_platform
        self.data_layout = data_layout
        self.tidl_tensor_bits = tidl_tensor_bits
        self.info_dict = {}
        self.tidl_relay_import_debug = os.environ.get("TIDL_RELAY_IMPORT_DEBUG")

        # Prepare for import
        self.temp_folder = os.path.join(artifacts_folder, 'tempDir/')
        os.makedirs(self.temp_folder, exist_ok=True)
        for root, dirs, files in os.walk(self.temp_folder, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))

    def tidl_import_conv2d(self, this_node, params):
        r""" Import conv2d operator to TIDL
            There is an example how to get the attributes of conv2d in Relay:
            https://github.com/dmlc/tvm/blob/master/python/tvm/relay/op/nn/_nn.py#L144
            https://docs.tvm.ai/api/python/ndarray.html

        Parameters
        ----------
        all_nodes : dictionary
            Dictionary of all relay.expr.Call nodes of the graph
        this_node : relay.expr.Call
            A relay.expr.Call node which is a conv2d operator
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay

        Returns
        -------
        True if import succeeds or False if import fails
        """

        weight = this_node.args[1]
        weight_shape = get_const_tuple(weight.checked_type.shape)
        weight_type = weight.checked_type.dtype
        strides = get_const_tuple(this_node.attrs.strides)
        dilation = get_const_tuple(this_node.attrs.dilation)
        padding = get_const_tuple(this_node.attrs.padding)
        kernel_size = get_const_tuple(this_node.attrs.kernel_size)
        groups = this_node.attrs.groups
        kernel_layout = this_node.attrs.kernel_layout

        conv2d_params = Conv2dParams()
        (conv2d_params.stride_h, conv2d_params.stride_w) = strides
        (conv2d_params.dilation_h, conv2d_params.dilation_w) = dilation
        # top, left, bottom, right padding
        if len(padding) == 1:
            pad_t = pad_l = pad_b = pad_r = padding[0]
        elif len(padding) == 2:
            pad_t = pad_b = padding[0]
            pad_l = pad_r = padding[1]
        else:
            (pad_t, pad_l, pad_b, pad_r) = padding
        (conv2d_params.pad_t, conv2d_params.pad_l, conv2d_params.pad_b, conv2d_params.pad_r) \
          = (pad_t, pad_l, pad_b, pad_r)
        (conv2d_params.kernel_h, conv2d_params.kernel_w) = kernel_size
        conv2d_params.num_groups = groups

        # Obtain weights from Relay params
        if isinstance(weight, tvm.relay.expr.Constant):
            weights = weight.data
        else:
            weight_name = weight.name_hint
            weights = params[weight_name]
        # Convert to numpy array and then pass to C
        weights_np = weights.asnumpy()

        if kernel_layout == 'OIHW':
            # No need to reshape - TIDL natively uses 'OIHW'
            conv2d_params.kernel_layout = b'OIHW'
            conv2d_params.num_in_channels = weight_shape[1]
            conv2d_params.num_out_channels = weight_shape[0]
            weights_to_tidl = weights_np
        elif kernel_layout == 'HWIO':
            # Reshape numpy array from 'HWIO' to 'OIHW'
            weights_to_tidl = weights_np.transpose((3, 2, 0, 1))
            conv2d_params.num_in_channels = weight_shape[2]
            conv2d_params.num_out_channels = weight_shape[3]
        elif kernel_layout == 'HWOI':
            # Reshape numpy array from 'HWOI' to 'OIHW'
            weights_to_tidl = weights_np.transpose((2, 3, 0, 1))
            conv2d_params.num_in_channels = weight_shape[3]
            conv2d_params.num_out_channels = weight_shape[2]
        else:
            print('Kernel layout ' + kernel_layout + ' not supported')
            return False

        if weight_type == 'float32':
            conv2d_params.weights_type = b'float32'
        else:
            print('Weight type ' + weight_type + ' not supported')
            return False

        weights_flatten = weights_to_tidl.flatten()
        conv2d_params.weights_array = ctypes.c_void_p(weights_flatten.ctypes.data)

        # Invoke C lib functions to pass parameters to TIDL
        import_lib_conv2d = self.import_lib.tidlImportConv2d
        import_lib_conv2d.argtypes = (ctypes.POINTER(Conv2dParams), ctypes.c_void_p)
        import_lib_conv2d.restype = None
        import_lib_conv2d(conv2d_params, ctypes.POINTER(ctypes.c_int)())
        return True

    def tidl_import_pad(self, node):
        r""" Import pad operator to TIDL
            Get attributes pad_width, convert to array, and passs to C library.
            A typical pad_width looks like: [[0,0],[0,1],[0,1],[0,0]]

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a pad operator

        Returns
        -------
        """

        pad_width = []
        for width in node.attrs.pad_width:
            pad_width.append(get_const_tuple(width))
        pad_list = [x for xs in pad_width for x in xs]

        # convert list to numpy array in order to pass to C library
        pad_array = np.asarray(pad_list, dtype=np.int32)

        import_lib_pad = self.import_lib.tidlImportPad
        import_lib_pad.argtypes = (ctypes.c_int, ctypes.c_void_p)
        import_lib_pad.restype = None
        import_lib_pad(len(pad_array), ctypes.c_void_p(pad_array.ctypes.data))
        return True

    def tidl_import_add(self):
        r""" Import add operator to TIDL
            An "add" operator may be adding two nodes or adding one node with constant:
                - %3 = add(%2, %1)
                - %3 = add(%2, %MobilenetV2/Conv/Conv2D_bn_offset)
            This function imports "add" opertor with args being two nodes.
        """

        import_lib_add = self.import_lib.tidlImportAdd
        import_lib_add.argtypes = None
        import_lib_add.restype = None
        import_lib_add()
        return True

    def tidl_import_bias_add(self, node):
        r""" Import bias_add or add operator to TIDL
            An "add" operator may be adding two nodes or adding one node with constant:
                - %3 = add(%2, %1)
                - %3 = add(%2, %MobilenetV2/Conv/Conv2D_bn_offset)
            A "bias_add" operator always add one node with constant.
            This function imports a "bias_add" or "add" with args[1] being constant.

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a add operator

        Returns
        -------
        True if import succeeds or False if import fails
        """
        bias = node.args[1]
        if isinstance(bias, tvm.relay.expr.Constant):
            # bias is expr.Constant if bind_params_by_name is called
            bias_params = bias.data
        else:
            # bias is expr.Var if bind_params_by_name is not called
            print('bias_add op must have args[1] as expr.Constant')
            return False

        if bias.checked_type.dtype == 'float32':
            bias_params_dtype = b'float32'
        else:
            print('Unsupported data type of bias_add')
            return False

        bias_params_len = bias.checked_type.shape[0]
        bias_params_np = bias_params.asnumpy()

        import_lib_bias = self.import_lib.tidlImportBiasAdd
        import_lib_bias.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        import_lib_bias.restype = None
        import_lib_bias(bias_params_len, bias_params_dtype,
                        ctypes.c_void_p(bias_params_np.ctypes.data))
        return True

    def tidl_import_batch_norm(self, node, params):
        r""" Import batch_norm operator to TIDL
            https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.batch_norm
            https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1BatchNormAttrs.html

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a batch_norm operator
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay

        Returns
        -------
        True if import succeeds or False if import fails
        """

        bn_params = BatchNormParams()
        if node.args[1].checked_type.dtype == 'float32':
            bn_params.params_dtype = b'float32'
        else:
            print('Unsupported data type of batch norm')
            return False
        bn_params.num_params = node.args[1].checked_type.shape[0]

        # Obtain weights from Relay params
        if isinstance(node.args[1], tvm.relay.expr.Constant):
            gama = node.args[1].data.asnumpy()
            beta = node.args[2].data.asnumpy()
            mean = node.args[3].data.asnumpy()
            var = node.args[4].data.asnumpy()
        else:
            gama = params[node.args[1].name_hint].asnumpy()
            beta = params[node.args[2].name_hint].asnumpy()
            mean = params[node.args[3].name_hint].asnumpy()
            var = params[node.args[4].name_hint].asnumpy()
        bn_params.gama = gama.ctypes.data
        bn_params.beta = beta.ctypes.data
        bn_params.mean = mean.ctypes.data
        bn_params.var = var.ctypes.data
        bn_params.epsilon = node.attrs.epsilon
        center = node.attrs.center
        scale = node.attrs.scale
        bn_params.center_enable = int(center)
        bn_params.scale_enable = int(scale)

        import_lib_bn = self.import_lib.tidlImportBatchNorm
        import_lib_bn.argtypes = (ctypes.POINTER(BatchNormParams), ctypes.c_void_p)
        import_lib_bn.restype = None
        import_lib_bn(bn_params, ctypes.POINTER(ctypes.c_int)())
        return True

    def tidl_import_pooling(self, node):
        r""" Import pooling operator to TIDL
            https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.avg_pool2d
            https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1AvgPool2DAttrs.html
            https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.max_pool2d
            https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1MaxPool2DAttrs.html

        Parameters
        ----------
        node : relay.expr.Call
            A relay.expr.Call node which is a pooling operator

        Returns
        -------
        """

        pooling_params = PoolingParams()
        if node.op.name == "nn.global_avg_pool2d":
            pooling_params.kernel_h = pooling_params.kernel_w = 0
            pooling_params.pad_h = pooling_params.pad_w = 0
            pooling_params.stride_h = pooling_params.stride_w = 1
        else:
            (pooling_params.kernel_h, pooling_params.kernel_w) = node.attrs.pool_size
            (pooling_params.stride_h, pooling_params.stride_w) = node.attrs.strides
            if len(node.attrs.padding) == 4:
                (pooling_params.pad_h, pooling_params.pad_w) = node.attrs.padding[2:4]
            else:
                (pooling_params.pad_h, pooling_params.pad_w) = node.attrs.padding

        if node.op.name == "nn.avg_pool2d" or node.op.name == "nn.global_avg_pool2d":
            pooling_type = b'avg_pool2d'
        else:
            pooling_type = b'max_pool2d'

        import_lib_pooling = self.import_lib.tidlImportPooling
        import_lib_pooling.argtypes = (ctypes.POINTER(PoolingParams), ctypes.c_char_p)
        import_lib_pooling.restype = None
        import_lib_pooling(pooling_params, pooling_type)
        return True

    def tidl_import_concat(self, all_nodes, node):

        in_nodes = find_in_nodes(all_nodes, node, self.tidl_target) # node indices of input nodes
        import_lib_concat = self.import_lib.tidlImportConcat
        import_lib_concat.argtype = ctypes.c_int
        import_lib_concat.restype = None
        import_lib_concat(len(in_nodes))
        return True

    def tidl_import_dense(self, this_node):

        weights = this_node.args[1]
        (num_outnodes, num_innodes) = weights.data.shape
        weights_array = weights.data.asnumpy()
        import_lib_dense = self.import_lib.tidlImportDense
        import_lib_dense.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        import_lib_dense.restype = None
        import_lib_dense(num_innodes, num_outnodes, ctypes.c_void_p(weights_array.ctypes.data))
        return True

    def tidl_import_mul(self, this_node):

        mul_params = MulParams()
        scale = this_node.args[0].data.asnumpy()
        mul_params.scale = np.amax(scale)
        import_lib_mul = self.import_lib.tidlImportMul
        import_lib_mul.argtypes = (ctypes.POINTER(MulParams), ctypes.c_void_p)
        import_lib_mul.restype = None
        import_lib_mul(mul_params, ctypes.POINTER(ctypes.c_int)())
        return True

    def tidl_import_init(self, subgraph_id, input_scale, input_signed, input_tensors, input_names):
        r""" Initializing TIDL import

        Parameters
        ----------
        subgraph_id: int
            Id of the subgraph to be imported to TIDL
        input_scale: list
            Scaling factor to convert floating point input to 8-bit quantized input
        input_signed: list
            Signed (1) or unsigned (0) of input
        input_tensors: list
            Input tensors to TIDL subgraph
        input_names: list
            Names of input tensors
        Returns
        -------
        True if initialization succeeds or False if initialization fails
        """

        input_shapes = []
        for input_tensor in input_tensors:
            input_shape = input_tensor.shape
            if len(input_shape) == 2:
                # input is a vector - expand (N,W) to (N,1,1,W)
                in_shape = (input_shape[0], 1, 1, input_shape[1])
            elif len(input_shape) == 3:
                # expand (N,H,W) to (N,1,H,W)
                in_shape = (input_shape[0], 1, input_shape[1], input_shape[2])
            elif len(input_shape) == 4:
                in_shape = input_shape
                if self.data_layout == "NHWC":
                    in_shape = (in_shape[0], in_shape[3], in_shape[1], in_shape[2])
            else:
                print("Subgraph input_shape " + str(input_shape) + " is not supported")
                return False
            input_shapes.append(in_shape)

        if self.data_layout == "NCHW":
            layout = b'NCHW'
            is_nchw = 1
        elif self.data_layout == "NHWC":
            layout = b'NHWC'
            is_nchw = 0
        else:
            print('data layout ' + self.data_layout + ' is not supported')
            return False

        if self.tidl_platform == "J7":
            descr = (TensorDescriptor * len(input_tensors))()
            for i in range(len(input_tensors)):
                descr[i].input_scale = input_scale[i]
                descr[i].input_signed = input_signed[i]
                (descr[i].channel, descr[i].height, descr[i].width) = input_shapes[i][1:4]
                descr[i].name = bytes(input_names[i], 'utf-8')
            input_dscr_ptr = ctypes.cast(descr, ctypes.c_void_p)
            import_lib_init = tvm.get_global_func("TIDL_relayImportInit")
            import_lib_init(subgraph_id, len(input_tensors), input_dscr_ptr, is_nchw,
                            self.tidl_tensor_bits, self.temp_folder)
            return True

        (channel, height, width) = input_shapes[0][1:4]
        in_quant_factor = int(round(input_scale[0]*255))  # 255 is due to TIDL implementation
        config_params = TIDLconfigParams(12, 50, in_quant_factor, input_signed[0],
                                         channel, height, width)

        # Invoking C library call to initialize TIDL import
        import_lib_init = self.import_lib.tidlImportInit
        import_lib_init.argtypes = (ctypes.POINTER(TIDLconfigParams), ctypes.c_char_p)
        import_lib_init.restype = None
        import_lib_init(config_params, layout)

        return True

    def tidl_import_node(self, all_nodes, this_node, params, output_names):
        r""" Importing a given node (operator) to TIDL
            # https://docs.tvm.ai/langref/relay_op.html#relay-core-tensor-operators

        Parameters
        ----------
        all_nodes : dictionary
            Dictionary of all relay.expr.Call nodes of the graph
        this_node : relay.expr.Call
            A relay.expr.Call node which is to be imported
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay

        Returns
        True if import succeeds or False if import fails
        """

        if self.tidl_platform == "J7":
            import_lib_node = tvm.get_global_func("TIDL_relayImportNode")
            if import_lib_node(this_node) != 0:
                return False
            in_out_nodes = find_in_out_nodes(all_nodes, this_node, self.tidl_target, output_names)
            import_lib_linknode = tvm.get_global_func("TIDL_relayImportLinkNode")
            if import_lib_linknode(ctypes.cast(ctypes.byref(in_out_nodes), ctypes.c_void_p)) == 0:
                return True
            else:
                return False

        status = True
        if this_node.op.name == 'nn.conv2d':
            status = self.tidl_import_conv2d(this_node, params)
        elif this_node.op.name == 'nn.pad':
            status = self.tidl_import_pad(this_node)
        elif this_node.op.name == 'add':
            if isinstance(this_node.args[1], tvm.relay.expr.Constant):
                status = self.tidl_import_bias_add(this_node)
            else:
                status = self.tidl_import_add()
        elif this_node.op.name == 'nn.bias_add':
            status = self.tidl_import_bias_add(this_node)
        elif this_node.op.name == 'clip':
            import_lib_relu = self.import_lib.tidlImportRelu
            import_lib_relu.argtype = (ctypes.c_char_p)
            import_lib_relu.restype = None
            import_lib_relu(b'Relu6')
        elif this_node.op.name == 'nn.relu':
            import_lib_relu = self.import_lib.tidlImportRelu
            import_lib_relu.argtype = (ctypes.c_char_p)
            import_lib_relu.restype = None
            import_lib_relu(b'Relu')
        elif this_node.op.name == 'nn.batch_norm':
            status = self.tidl_import_batch_norm(this_node, params)
        elif this_node.op.name == 'nn.avg_pool2d':
            status = self.tidl_import_pooling(this_node)
        elif this_node.op.name == 'squeeze':
            import_lib_squeeze = self.import_lib.tidlImportSqueeze
            import_lib_squeeze.argtype = None
            import_lib_squeeze.restype = None
            import_lib_squeeze()
        elif this_node.op.name == 'reshape':
            import_lib_reshape = self.import_lib.tidlImportReshape
            import_lib_reshape.argtype = None
            import_lib_reshape.restype = None
            import_lib_reshape()
        elif this_node.op.name == 'nn.softmax':
            import_lib_softmax = self.import_lib.tidlImportSoftmax
            import_lib_softmax.argtype = None
            import_lib_softmax.restype = None
            import_lib_softmax()
        elif this_node.op.name == 'concatenate':
            status = self.tidl_import_concat(all_nodes, this_node)
        elif this_node.op.name == 'nn.max_pool2d':
            status = self.tidl_import_pooling(this_node)
        elif this_node.op.name == 'nn.dropout':
            import_lib_dropout = self.import_lib.tidlImportDropOut
            import_lib_dropout.argtype = None
            import_lib_dropout.restype = None
            import_lib_dropout()
        elif this_node.op.name == 'nn.global_avg_pool2d':
            status = self.tidl_import_pooling(this_node)
        elif this_node.op.name == 'nn.batch_flatten':
            import_lib_flatten = self.import_lib.tidlImportBatchFlatten
            import_lib_flatten.argtype = None
            import_lib_flatten.restype = None
            import_lib_flatten()
        elif this_node.op.name == 'multiply':
            status = self.tidl_import_mul(this_node)
        elif this_node.op.name == 'nn.dense':
            status = self.tidl_import_dense(this_node)
        else:
            print("Operator " + this_node.op.name + " is not supported by TIDL!")
            status = False

        if not status:
            return False

        # (AM57) Common for all nodes:
        # fill tensor names, update consumer counts, link input/output tensors
        in_out_nodes = find_in_out_nodes(all_nodes, this_node, self.tidl_target, output_names)

        import_lib_link_nodes = self.import_lib.tidlImportLinkNodes
        import_lib_link_nodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.c_void_p)
        import_lib_link_nodes.restype = ctypes.c_int
        if import_lib_link_nodes(in_out_nodes, ctypes.POINTER(ctypes.c_int)()) == 0:
            return False

        return True

    def tidl_import_out_tuple_node(self, all_nodes, node, out_tensor_names):
        """ Importing a Relay tuple node, e.g. (%232, %279, %283, %274).
            If this node is the last node, import it to TIDL output data layer.
            If this node is not the last node, do nothing.
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

            if self.tidl_platform == "AM57":
                import_lib_out_data = self.import_lib.tidlImportOutData
                import_lib_out_data.argtype = ctypes.c_int
                import_lib_out_data.restype = None
            else:
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

            if self.tidl_platform == "AM57":
                import_lib_link_nodes = self.import_lib.tidlImportLinkNodes
                import_lib_link_nodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.c_void_p)
                import_lib_link_nodes.restype = ctypes.c_int
                if import_lib_link_nodes(in_out_nodes, ctypes.POINTER(ctypes.c_int)()) == 0:
                    status = False
                    break
            else:
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

    def import_relay_ir(self, mod, params, subgraph_tensors_list):
        r""" Relay IR import to TIDL

        Parameters
        ----------
        mod : tvm.relay.Module
            Relay IR graph with subgraphs
        params : dict of str to tvm.NDArray
            The parameter dict to be used by relay
        subgraph_tensors_list: list of dict (list length equals number of calibration data)
            Input/output tensors of subgraphs obtained from TVM graph execution

        Returns
        -------
        1: if TIDL import succeeds
        -1: if TIDL import fails
        0: if there are no subgraphs for TIDL offload
        """
        if self.tidl_relay_import_debug != None:
            print("----- RelayIR Graph for importing to TIDL -----")
            print(mod.astext(show_meta_data=False))

        # Generate svg for partitined graph
        visualize_relay_graph(module=mod, filename=self.temp_folder+'/relay.gv')

        # Define return values
        import_succeed, import_fail, no_import = 1, -1, 0

        # Put some information about the graph in the info file passed to the TIDL codegen
        self.info_dict['tvm'] = { 
           'is_nchw'   : 1 if self.data_layout == "NCHW" else 0,
           'macs'      : relay.analysis.get_total_mac_number(mod['main']),
           'nodes'     : {},
        }
        self.info_dict['subgraphs'] = []

        # Traverse Relay IR graph and generate a dictionary of all TIDL subgraphs
        all_nodes_main = {}
        traverse_func = functools.partial(traverse_expr, node_dict=all_nodes_main)
        relay.analysis.post_order_visit(mod['main'], traverse_func)
        tidl_subgraphs = []
        relay_call_ops = 0
        for node in all_nodes_main:
            if isinstance(node, relay.expr.GlobalVar):
                if self.tidl_target in node.name_hint:
                    tidl_subgraphs.append(node.name_hint)
            # Tally relay call nodes that are not calls to a TIDL subgraph
            if isinstance(node, relay.expr.Call) and isinstance(node.op, tvm.ir.op.Op):
                self._tally_op(str(node.op), self.info_dict['tvm']['nodes'])

        # For each TIDL subgraph, import to TIDL and calibrate
        for tidl_subgraph in tidl_subgraphs:
            # Extract subgraph id and input/output tensor names from subgraph name
            subgraph_id = int(tidl_subgraph.replace(self.tidl_target+'_', ''))
            in_tensor_name = tidl_subgraph + '_i'
            out_tensor_name = tidl_subgraph + '_o'

            # Obtain input tensor from TVM graph execution
            input_fp_list, input_names_list = obtain_subgraph_tensor(subgraph_tensors_list,
                                                                     in_tensor_name)
            output_fp_list, output_names_list = obtain_subgraph_tensor(subgraph_tensors_list,
                                                                       out_tensor_name)
            input_names = input_names_list[0]
            output_names = output_names_list[0]
            if not input_fp_list:
                return import_fail
            if self.tidl_platform == "AM57" and len(input_fp_list[0]) > 1:
                print("Error - only 1 input tensor is supported for AM57x (J6)!")
                return import_fail

            # Quantize input tensors
            input_quant_vec_list, input_scale, input_signed = \
                    tensor_quant_flatten(input_fp_list, self.data_layout, self.tidl_tensor_bits)

            # Initialize TIDL import
            subgraph = mod[tidl_subgraph]
            if not self.tidl_import_init(subgraph_id, input_scale, input_signed, input_fp_list[0],
                                         input_names):
                return import_fail

            # Initialize subgraph info for nfo file
            subgraph_info_dict = { 
               'name'    : tidl_subgraph, 
               'is_nchw' : 1 if self.data_layout == "NCHW" else 0,
               'macs'    : relay.analysis.get_total_mac_number(subgraph),
               'ninputs' : len(input_names),
               'noutputs': len(output_names),
               'nodes'   : {},
            }

            # Scan through all relay.expr.Call nodes and import each to TIDL
            all_nodes_tidl = {}
            traverse_func = functools.partial(traverse_expr, node_dict=all_nodes_tidl)
            relay.analysis.post_order_visit(subgraph, traverse_func)
            for node in all_nodes_tidl:
                if isinstance(node, relay.expr.Call):
                    result = self.tidl_import_node(all_nodes_tidl, node, params, output_names)
                    if not result:
                        return import_fail
                    self._tally_op(str(node.op), subgraph_info_dict['nodes'])

            # Import expr.Tuple node if it is the last node, after importing all expr.call nodes
            for node in all_nodes_tidl:
                if isinstance(node, relay.expr.Tuple) and \
                   len(find_out_nodes(all_nodes_tidl, node)) == 0:
                    #node.fields: array of expr.call nodes
                    result = self.tidl_import_out_tuple_node(all_nodes_tidl, node, output_names)
                    if not result:
                        print('Error importing output tuple node')
                        return import_fail

            # Invoke TIDL optimization of the imported graph
            net_file = os.path.join(self.artifacts_folder,
                                    'tidl_subgraph'+str(subgraph_id)+'_net.bin')
            par_file = os.path.join(self.artifacts_folder,
                                    'tidl_subgraph'+str(subgraph_id)+'_params.bin')

            if self.tidl_platform == "AM57":
                import_lib_optimize = self.import_lib.tidlImportOptimize
                import_lib_optimize.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
                import_lib_optimize.restype = ctypes.c_int
                net_fname = net_file.encode('utf-8')
                par_fname = par_file.encode('utf-8')

                if import_lib_optimize(net_fname, par_fname, subgraph_id) == 0:
                    print('TIDL import optimization failed')
                    return import_fail
            else:  # == "J7"
                import_lib_optimize = tvm.get_global_func("TIDL_relayOptimizeNet")
                if import_lib_optimize() != 0:
                    print('TIDL import optimization failed')
                    return import_fail

            # Calibrate TIDL for the imported subgraph
            status, out_data_q = subgraph_calibration(self.calib_tool, subgraph_id,
                                     input_quant_vec_list, input_signed, self.temp_folder,
                                     net_file, par_file, self.tidl_platform,
                                     self.tidl_tensor_bits)

            self.info_dict['subgraphs'].append(subgraph_info_dict)
            if self.tidl_platform == "J7":
                if status:
                    mod[tidl_subgraph] = self.mark_tidl_layers(subgraph, subgraph_id,
                                                               all_nodes_tidl)
                    continue  # import next subgraph
                else:
                    return import_fail
            if not status:
                return import_fail

            # AM57x (J6) only: Calculate scaling factor to convert output tensor to floating point
            # Obtain output tensor from TVM graph execution
            output_fp = output_fp_list[0]

            # TODO: convert following lines into a function
            if output_fp is None:
                return import_fail
            if len(output_fp) != len(out_data_q):
                return import_fail
            output_signed = []
            output_scale = []
            for tensor in output_fp:
                # Find out if this output tensor is signed or unsigned
                output_signed.append(int(np.amin(tensor) < 0))
            for data_q in out_data_q:
                # Change data Q to scale - 255 is TIDL implementation specific
                output_scale.append(round(data_q/255.0, 5))

            # Generate subgraph configuration file
            subgraph_cfg_gen(self.artifacts_folder, subgraph_id, self.data_layout,
                             input_scale[0], input_signed[0], output_scale, output_signed)

        with open(os.path.join(self.temp_folder, "relay.nfo"), "w") as of:
            json.dump(self.info_dict, of, indent=4)

        return import_succeed if len(tidl_subgraphs) > 0 else no_import

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
                    if l[2] == node_name:
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

        ### tempDir/subgraph<id>_net.bin.layer_info.txt always available, even DEBUG level 0
        ### format of each line: layer_index data_id output_name
        layer_info_file = os.path.join(self.temp_folder,
                                       f"subgraph{subgraph_id}_net.bin.layer_info.txt")
        layer_info = [ x.split(' ') for x in open(layer_info_file).readlines() ]
        # rewrite outputs as "tidl_<subgraph_id>_o<output_id>" to match up with TIDL names
        if isinstance(subgraph.body, Tuple):
            for i, out in enumerate(subgraph.body.fields):
                all_nodes_tidl[out] = f"tidl_{subgraph_id}_o{i}"
        else:
            all_nodes_tidl[subgraph.body] = f"tidl_{subgraph_id}_o0"
        return CallMarker(subgraph_id, all_nodes_tidl, layer_info).visit(subgraph)

class TIDLAnnotation:
    def __init__(self, platform, version, import_lib, denylist=None):
        self.tidl_platform = platform
        self.version = version
        self.import_lib = import_lib
        self.denylist = denylist

    def register_whitelist_ops(self):
        """ TIDL operators registration """

        # Can't register annotations more than once.
        global tidl_annotations_registered
        if tidl_annotations_registered:
            return

        # Register J7/J6 common operators which are always supported
        self._register_supported_op("nn.relu")

        # Register J7/J6 common operators which are supported with different constraints
        self._register_constrained_op("argmax")
        self._register_constrained_op("nn.avg_pool2d")
        self._register_constrained_op("nn.batch_flatten")
        self._register_constrained_op("nn.batch_norm")
        self._register_constrained_op("nn.conv2d")
        #self._register_supported_op("nn.conv2d")    # use this for debugging
        self._register_constrained_op("nn.dense")
        self._register_constrained_op("nn.conv2d_transpose")
        self._register_constrained_op("nn.global_avg_pool2d")
        self._register_constrained_op("nn.max_pool2d")
        self._register_constrained_op("nn.softmax")
        self._register_constrained_op("concatenate")
        self._register_constrained_op("mean")          # 'mean' mapped to avg_pooling layer

        # Register J7 specific operators, or those supported standalone by J7 but not J6,
        # or those for which there are no whitelist functions.
        if self.tidl_platform == 'J7':
            self._register_constrained_op("add")
            self._register_constrained_op("nn.bias_add")
            self._register_constrained_op("maximum")
            self._register_constrained_op("minimum")
            self._register_constrained_op("multiply")
            self._register_constrained_op("divide")
            self._register_constrained_op("split")
            self._register_constrained_op("strided_slice")
            self._register_constrained_op("image.resize")
            # "clip" is supported with constraints in J7 but unsupported standalone in J6
            self._register_constrained_op("clip")
            self._register_supported_op("nn.leaky_relu")
            self._register_supported_op("nn.prelu")
            self._register_constrained_op("nn.upsampling")
            self._register_constrained_op("nn.upsampling3d")

        # Register operators that are J6 specific or have constraints only for J6
        if self.tidl_platform == 'AM57':  # J6 is known as 'AM57'
            self._register_supported_op("nn.dropout")
            self._register_constrained_op("max")           # 'max' mapped to max_pooling layer
            @tvm.ir.register_op_attr("add", "target.tidl")
            def add_whitelist_fn(attrs, args):
                if any([isinstance(arg, tvm.relay.expr.Constant) for arg in args]):
                    # This is the same as "bias_add" which is not supported standalone.
                    return False
                return True

        tidl_annotations_registered = True

    def merge_sequential_ops(self, mod):
        """Fuse sequential ops for op registration."""

        # Squeeze has to be followed by reshape.
        def _squeeze_reshape_pattern():
            squeeze_out = is_op('squeeze')(wildcard())
            reshape_out = is_op('reshape')(squeeze_out)
            return reshape_out
        def _squeeze_reshape_checker(extract):
            return not self._user_denied('squeeze', 'reshape')

        #transpose has to be preceded and followed by reshape
        def _transpose_reshape_pattern():
            reshape_out1 = is_op('reshape')(wildcard())
            transpose_out = is_op('transpose')(reshape_out1)
            reshape_out2 = is_op('reshape')(transpose_out)
            return reshape_out2
        def _transpose_reshape_checker(extract):
            if self._user_denied('reshape', 'transpose'):
                return False
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

        #reshape has to be preceded by avg_pool2d, global_avg_pool2d, dense, or mean
        def _reshape_avg_pool_pattern():
            avg_pool_out = is_op('nn.avg_pool2d')(wildcard())
            reshape_out = is_op('reshape')(avg_pool_out)
            return reshape_out
        def _reshape_avg_pool_checker(extract):
            if self._user_denied('nn.avg_pool2d', 'reshape'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.avg_pool2d', op.attrs, op.args)
        def _reshape_global_avg_pool_pattern():
            global_avg_pool_out = is_op('nn.global_avg_pool2d')(wildcard())
            reshape_out = is_op('reshape')(global_avg_pool_out)
            return reshape_out
        def _reshape_global_avg_pool_checker(extract):
            if self._user_denied('nn.global_avg_pool2d', 'reshape'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.global_avg_pool2d', op.attrs, op.args)
        def _reshape_dense_pattern():
            dense_out = is_op('nn.dense')(wildcard(), is_constant())
            reshape_out = is_op('reshape')(dense_out)
            return reshape_out
        def _reshape_dense_checker(extract):
            if self._user_denied('nn.dense', 'reshape'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.dense', op.attrs, op.args)
        def _reshape_mean_pattern():
            mean_out = is_op('mean')(wildcard())
            reshape_out = is_op('reshape')(mean_out)
            return reshape_out
        def _reshape_mean_checker(extract):
            if self._user_denied('mean', 'reshape'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('mean', op.attrs, op.args)

        #reshape has to be followed by softmax
        def _reshape_softmax_pattern():
            reshape_out = is_op('reshape')(wildcard())
            softmax_out = is_op('nn.softmax')(reshape_out)
            return softmax_out
        def _reshape_softmax_checker(extract):
            if self._user_denied('reshape', 'nn.softmax'):
                return False
            return self.whitelist_check_func('nn.softmax', extract.attrs, extract.args)

        #pad has to precede conv2d, (conv2d, bias_add), or (conv2d, add)
        def _pad_conv2d_pattern():
            pad_out = is_op('nn.pad')(wildcard())
            conv2d_out = is_op('nn.conv2d')(pad_out, is_constant())
            return conv2d_out
        def _pad_conv2d_checker(extract):
            if self._user_denied('nn.pad', 'nn.conv2d'):
                return False
            pad_op = extract.args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            conv2d_supported = self.whitelist_check_func('nn.conv2d', extract.attrs, extract.args)
            return conv2d_supported and pad_supported

        # common patterns required by J7 or J6
        pattern_table_common = [
            ('tidl.squeeze_reshape', _squeeze_reshape_pattern(), _squeeze_reshape_checker),
            ('tidl.reshape_avgpool', _reshape_avg_pool_pattern(), _reshape_avg_pool_checker),
            ('tidl.reshape_globalavgpool', _reshape_global_avg_pool_pattern(),
                                           _reshape_global_avg_pool_checker),
            ('tidl.reshape_dense', _reshape_dense_pattern(), _reshape_dense_checker),
            ('tidl.reshape_mean', _reshape_mean_pattern(), _reshape_mean_checker),
            ('tidl.reshape_softmax', _reshape_softmax_pattern(), _reshape_softmax_checker),
            ('tidl.pad_conv2d', _pad_conv2d_pattern(), _pad_conv2d_checker),
        ]

        # additional patterns required by J6
        #bias_add has be preceded by conv2d or (pad, conv2d)
        def _conv2d_bias_pattern():
            conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
            bias_out = is_op('nn.bias_add')(conv2d_out, is_constant())
            return bias_out
        def _conv2d_bias_checker(extract):
            if self._user_denied('nn.conv2d', 'nn.bias_add'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.conv2d', op.attrs, op.args)
        def _conv2d_add_pattern():
            conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
            add_out = is_op('add')(conv2d_out, is_constant())
            return add_out
        def _conv2d_add_checker(extract):
            if self._user_denied('nn.conv2d', 'add'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.conv2d', op.attrs, op.args)

        def _pad_conv2d_bias_pattern():
            pad_conv2d_out = _pad_conv2d_pattern()
            bias_out = is_op('nn.bias_add')(pad_conv2d_out, is_constant())
            return bias_out
        def _pad_conv2d_bias_checker(extract):
            if self._user_denied('nn.pad', 'nn.conv2d', 'nn.bias_add'):
                return False
            pad_op = extract.args[0].args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            conv2d_bias_supported = _conv2d_bias_checker(extract)
            return conv2d_bias_supported and pad_supported

        def _pad_conv2d_add_pattern():
            pad_conv2d_out = _pad_conv2d_pattern()
            add_out = is_op('add')(pad_conv2d_out, is_constant())
            return add_out
        def _pad_conv2d_add_checker(extract):
            if _user_denied('nn.pad', 'nn.conv2d', 'add'):
               return False
            pad_op = extract.args[0].args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            conv2d_add_supported = _conv2d_add_checker(extract)
            return conv2d_add_supported and pad_supported

        #bias_add has to be preceded by dense
        def _dense_bias_pattern():
            dense_out = is_op('nn.dense')(wildcard(), is_constant())
            bias_out = is_op('nn.bias_add')(dense_out, is_constant())
            return bias_out
        def _dense_bias_checker(extract):
            if self._user_denied('nn.dense', 'nn.bias_add'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.dense', op.attrs, op.args)

        def _dense_add_pattern():
            dense_out = is_op('nn.dense')(wildcard(), is_constant())
            add_out = is_op('add')(dense_out, is_constant())
            return add_out
        def _dense_add_checker(extract):
            if self._user_denied('nn.dense', 'add'):
                return False
            op = extract.args[0]
            return self.whitelist_check_func('nn.dense', op.attrs, op.args)

        #relu6 has to be preceded by conv2d or (conv2d, bias_add)
        def _relu6_check_fun(attrs): # clip(0, 6) is not supported standalone
            supported = (float(attrs.a_min) == 0.0) and (float(attrs.a_max) == 6.0)
            return supported

        def _conv2d_relu6_pattern():
            conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
            relu6_out = is_op('clip')(conv2d_out)
            return relu6_out
        def _conv2d_relu6_checker(extract):
            if self._user_denied('nn.conv2d', 'clip'):
                return False
            relu6_supported = _relu6_check_fun(extract.attrs)
            op = extract.args[0]
            return self.whitelist_check_func('nn.conv2d', op.attrs, op.args) and relu6_supported

        def _conv2d_bias_relu6_pattern():
            conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
            bias_out = is_op('nn.bias_add')(conv2d_out, is_constant())
            relu6_out = is_op('clip')(bias_out)
            return relu6_out
        def _conv2d_bias_relu6_checker(extract):
            if self._user_denied('nn.conv2d', 'nn.bias_add', 'clip'):
                return False
            relu6_supported = _relu6_check_fun(extract.attrs)
            op = extract.args[0].args[0]
            return self.whitelist_check_func('nn.conv2d', op.attrs, op.args) and relu6_supported

        def _conv2d_add_relu6_pattern():
            conv2d_out = is_op('nn.conv2d')(wildcard(), is_constant())
            # 'add' must be 'bias_add' in (conv2d, add, relu6) pattern
            bias_add_out = is_op('add')(conv2d_out, is_constant())
            relu6_out = is_op('clip')(bias_add_out)
            return relu6_out
        def _conv2d_add_relu6_checker(extract):
            if self._user_denied('nn.conv2', 'add', 'clip'):
                return False
            return _conv2d_bias_relu6_checker(extract)

        #relu6 has to be preceded by element-wise add, batch_norm, or dense
        def _add_relu6_pattern():
            # add must be element-wise add
            add_out = is_op('add')(wildcard(), wildcard())
            relu6_out = is_op('clip')(add_out)
            return relu6_out
        def _add_relu6_checker(extract):
            if self._user_denied('add', 'clip'):
                return False
            relu6_supported = _relu6_check_fun(extract.attrs)
            return relu6_supported

        def _bn_relu6_pattern():
            bn_out = is_op('nn.batch_norm')(wildcard(), wildcard(), wildcard(), wildcard(),
                                            wildcard())
            tuple_get_item_node = is_tuple_get_item(bn_out, 0)
            relu6_out = is_op('clip')(tuple_get_item_node)
            return relu6_out
        def _bn_relu6_checker(extract):
            if self._user_denied('nn.batch_norm', 'clip'):
                return False
            relu6_supported = _relu6_check_fun(extract.attrs)
            bn_op = extract.args[0].tuple_value
            bn_supported = self.whitelist_check_func('nn.batch_norm', bn_op.attrs, bn_op.args)
            return bn_supported and relu6_supported

        def _dense_relu6_pattern():
            dense_out = is_op('nn.dense')(wildcard(), is_constant())
            relu6_out = is_op('clip')(dense_out)
            return relu6_out
        def _dense_relu6_checker(extract):
            if self._user_denied('nn.dense', 'clip'):
                return False
            relu6_supported = _relu6_check_fun(extract.attrs)
            op = extract.args[0]
            return self.whitelist_check_func('nn.dense', op.attrs, op.args) and relu6_supported

        #relu6 can also be preceded by (dense, bias_add): 
        #  (dense, bias_add, relu6) -> (dense, relu6) -> dense
        def _dense_bias_relu6_pattern():
            dense_out = is_op('nn.dense')(wildcard(), is_constant())
            bias_out = is_op('nn.bias_add')(dense_out, is_constant())
            relu6_out = is_op('clip')(bias_out)
            return relu6_out
        def _dense_bias_relu6_checker(extract):
            if self._user_denied('nn.dense', 'nn.bias_add', 'clip'):
                return False
            dense_op = extract.args[0].args[0]
            relu6_supported = _relu6_check_fun(extract.attrs)
            dense_supported = self.whitelist_check_func('nn.dense', dense_op.attrs, dense_op.args)
            return relu6_supported and dense_supported

        def _dense_add_relu6_pattern():
            dense_out = is_op('nn.dense')(wildcard(), is_constant())
            bias_add_out = is_op('add')(dense_out, is_constant())
            relu6_out = is_op('clip')(bias_add_out)
            return relu6_out
        def _dense_add_relu6_checker(extract):
            if self._user_denied('nn.dense', 'add', 'clip'):
                return False
            return _dense_bias_relu6_checker(extract)

        def _pad_conv2d_relu6_pattern():
            _pad_conv2d_out = _pad_conv2d_pattern()
            relu6_out = is_op('clip')(_pad_conv2d_out)
            return relu6_out
        def _pad_conv2d_relu6_checker(extract):
            if self._user_denied('nn.pad', 'nn.conv2d', 'clip'):
                return False
            pad_op = extract.args[0].args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            return pad_supported and _conv2d_relu6_checker(extract)

        def _pad_conv2d_bias_relu6_pattern():
            _pad_conv2d_bias_out = _pad_conv2d_bias_pattern()
            relu6_out = is_op('clip')(_pad_conv2d_bias_out)
            return relu6_out
        def _pad_conv2d_bias_relu6_checker(extract):
            if self._user_denied('nn.pad', 'nn.conv2d', 'nn.bias_add', 'clip'):
                return False
            pad_op = extract.args[0].args[0].args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            return pad_supported and _conv2d_bias_relu6_checker(extract)

        def _pad_conv2d_add_relu6_pattern():
            _pad_conv2d_add_out = _pad_conv2d_add_pattern()
            relu6_out = is_op('clip')(_pad_conv2d_add_out)
            return relu6_out
        def _pad_conv2d_add_relu6_checker(extract):
            if _user_denied('nn.pad', 'nn.conv2d', 'add', 'clip'):
                return False
            pad_op = extract.args[0].args[0].args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            return pad_supported and _conv2d_add_relu6_checker(extract)

        # additional patterns required by J6
        pattern_table_j6 = [
            ('tidl.pad_conv2d_bias_relu6', _pad_conv2d_bias_relu6_pattern(), _pad_conv2d_bias_relu6_checker),
            ('tidl.pad_conv2d_add_relu6', _pad_conv2d_add_relu6_pattern(), _pad_conv2d_add_relu6_checker),
            ('tidl.pad_conv2d_relu6', _pad_conv2d_relu6_pattern(), _pad_conv2d_relu6_checker),
            ('tidl.conv2d_bias_relu6', _conv2d_bias_relu6_pattern(), _conv2d_bias_relu6_checker),
            ('tidl.conv2d_add_relu6', _conv2d_add_relu6_pattern(), _conv2d_add_relu6_checker),
            ('tidl.conv2d_relu6', _conv2d_relu6_pattern(), _conv2d_relu6_checker),
            ('tidl.dense_bias_relu6', _dense_bias_relu6_pattern(), _dense_bias_relu6_checker),
            ('tidl.dense_add_relu6', _dense_add_relu6_pattern(), _dense_add_relu6_checker),
            ('tidl.dense_relu6', _dense_relu6_pattern(), _dense_relu6_checker),
            ('tidl.add_relu6', _add_relu6_pattern(), _add_relu6_checker),
            ('tidl.bn_relu6', _bn_relu6_pattern(), _bn_relu6_checker),
            ('tidl.pad_conv2d_bias', _pad_conv2d_bias_pattern(), _pad_conv2d_bias_checker),
            ('tidl.pad_conv2d_add', _pad_conv2d_add_pattern(), _pad_conv2d_add_checker),
            ('tidl.conv2d_bias', _conv2d_bias_pattern(), _conv2d_bias_checker),
            ('tidl.conv2d_add', _conv2d_add_pattern(), _conv2d_add_checker),
            ('tidl.dense_bias', _dense_bias_pattern(), _dense_bias_checker),
            ('tidl.dense_add', _dense_add_pattern(), _dense_add_checker),
        ]

        # additional patterns required by J7
        #pad can also precede avg_pool2d
        def _pad_avg_pool_pattern():
            pad_out = is_op('nn.pad')(wildcard())
            avg_pool_out = is_op('nn.avg_pool2d')(pad_out)
            return avg_pool_out
        def _pad_avg_pool_checker(extract):
            if self._user_denied('nn.pad', 'nn.avg_pool2d'):
                return False
            pad_op = extract.args[0]
            pad_supported = self.whitelist_check_func('nn.pad', pad_op.attrs, pad_op.args)
            pool_supported = self.whitelist_check_func('nn.avg_pool2d', extract.attrs, extract.args)
            return pool_supported and pad_supported

        pattern_table_j7 = [
            ('tidl.transpose_reshape', _transpose_reshape_pattern(), _transpose_reshape_checker),
            ('tidl.pad_avgpool', _pad_avg_pool_pattern(), _pad_avg_pool_checker),
        ]

        if self.tidl_platform == 'AM57':  # J6 is known as 'AM57'
            # conv2d_bias_relu6/conv2d_add_relu6 must precede conv2d_bias/conv2d_add in the table
            # dense_bias_relu6/dense_add_relu6 must precede dense_bias/dense_add in the table
            pattern_table = pattern_table_j6 + pattern_table_common
        else:
            pattern_table = pattern_table_j7 + pattern_table_common
        return relay.transform.MergeComposite(pattern_table)(mod)

    # Helper functions
    def _user_denied(self, *args):
        """ The arguments are operator names. Return true if any of the operators
            are in the user-specified denylist (e.g. via the --deny option to the 
            unit test program).  """
        for op in args:
            if self.denylist and op in self.denylist:
                return True
        return False
       
    def _register_supported_op(self, op_name):
        """ Helper function to register an op that is supported without any constraints """
        @tvm.ir.register_op_attr(op_name, "target.tidl")
        def _func_wrapper(attrs, args):
            if self._user_denied(op_name):
                return False
            #TODO: add data type check
            #if any([arg.checked_type.dtype != "float32" for arg in args]):
            #    return False
            return True
        return _func_wrapper

    def _register_constrained_op(self, op_name):
        """ Helper function to register an op that is supported with some constraints """
        @tvm.ir.register_op_attr(op_name, "target.tidl")
        def _func_wrapper(attrs, args):
            if self._user_denied(op_name):
                return False
            return self.whitelist_check_func(op_name, attrs, args)
        return _func_wrapper

    def whitelist_check_func(self, op_name, attrs, args):
        """ Whitelist check function for operators with different constraints for J7 and J6 """
        if self.tidl_platform == "J7":
            return self.whitelist_fn_j7(op_name, attrs, args)
        else:
            return self.whitelist_fn_j6(op_name, attrs, args)

    def whitelist_fn_j7(self, op_name, attrs, args):
        """ Whitelisting function for J7: constraint checking is delegated to the import library """

        if self.import_lib is None:
            # For CI testing which doesn't have import library - still run TVM passes
            return True

        # Invoke TIDL import library call to check if this op can be supported
        op = tvm.ir.Op.get(op_name)
        callnode = tvm.relay.Call(op, args, attrs)
        #print(f"Invoking TIDL Relay Import whitelisting function for {op_name}")
        whitelist_fn = tvm.get_global_func("TIDL_relayWhitelistNode")
        return whitelist_fn(callnode)

    def whitelist_fn_j6(self, op_name, attrs, args):
        """ Whitelisting function for J6: checking operator attributes against constraints """

        def argmax_whitelist_fn(attrs, args):
            keepdims = attrs.keepdims
            exclude = attrs.exclude
            axis = attrs.axis
            data = args[0]
            data_shape = data.checked_type.shape
            supported = (int(data_shape[1]) <= 15 and keepdims == 1 and axis == 1 and exclude == 0)
            return supported

        def avg_pool_whitelist_fn(attrs, args):
            pool_size = get_const_tuple(attrs.pool_size)
            strides = get_const_tuple(attrs.strides)
            supported = (pool_size[0] <= 9 and pool_size[1] <= 9 \
                         and strides[0] <= 3 and strides[1] <= 2)
            return supported

        def batch_flatten_fn(attrs, args):
            data = args[0]
            data_shape = data.checked_type.shape
            if len(data_shape) == 4:
                supported = (int(data_shape[2]) <= 65535 and int(data_shape[3]) <= 65535)
            else:
                supported = True
            return supported

        def batch_norm_whitelist_fn(attrs, args):
            data1 = args[1]
            if data1.checked_type.dtype != 'float32':
                supported = False
            elif attrs.axis != 1 and attrs.axis != 3:
                supported = False
            else:
                supported = True
            return supported

        def get_conv2d_num_channels(kernel_layout, weight_shape):
            """ Get number of input and output channels of conv2d """
            if kernel_layout == 'OIHW':
                (num_in_channels, num_out_channels) = (weight_shape[1], weight_shape[0])
            elif kernel_layout == 'HWIO':
                (num_in_channels, num_out_channels) = (weight_shape[2], weight_shape[3])
            else: # 'HWOI'
                (num_in_channels, num_out_channels) = (weight_shape[3], weight_shape[2])
            return (num_in_channels, num_out_channels)

        def conv2d_whitelist_fn(attrs, args):
            weight = args[1]
            if weight.checked_type.dtype != 'float32':
                return False
            if attrs.kernel_layout not in ('OIHW', 'HWIO', 'HWOI'):
                return False

            weight_shape = weight.data.shape
            strides = get_const_tuple(attrs.strides)
            dilation = get_const_tuple(attrs.dilation)
            kernel_size = get_const_tuple(attrs.kernel_size)
            (dh, dw) = dilation
            (kh, kw) = kernel_size
            (num_in_chs, num_out_chs) = get_conv2d_num_channels(attrs.kernel_layout, weight_shape)
            channel_supported = (num_in_chs <= 2048 and num_out_chs <= 2048)
            stride_supported = (strides[0] <= 2 and strides[1] <= 2)
            dilation_supported = (dh in (1, 2, 4)) and (dw in (1, 2, 4))
            kernel_supported = (((kh-1)*dh+1) <= 9) and (((kw-1)*dw+1) <= 9)
            groups_supported = (attrs.groups <= 1024)
            supported = channel_supported and stride_supported and dilation_supported \
                        and kernel_supported and groups_supported
            return supported

        def conv2d_transpose_whitelist_fn(attrs, args):
            if attrs.kernel_layout not in ('OIHW', 'HWIO', 'HWOI'):
                return False
            weight = args[1]
            weight_shape = weight.data.shape
            strides = get_const_tuple(attrs.strides)
            (num_in_chs, num_out_chs) = get_conv2d_num_channels(attrs.kernel_layout, weight_shape)
            supported = (num_in_chs == num_out_chs) and (num_in_chs == attrs.groups) \
                        and (strides[1] == 2)
            return supported

        def dense_whitelist_fn(attrs, args):
            weight = args[1]
            weight_shape = weight.data.shape
            (w_in, w_out) = (weight_shape[1], weight_shape[0])
            supported = (w_in <= 65536) and (w_out <= 16384) and (w_in * w_out <= 67108864)
            return supported

        def global_avg_pool_whitelist_fn(attrs, args):
            shape = list(map(int, args[0].checked_type.shape))
            layout = attrs.layout
            if layout == "NCHW":
                (height, width) = (shape[2], shape[3])
            else: # "NHWC"
                (height, width) = (shape[1], shape[2])
            supported = height * width <= 4096
            return supported

        def max_pool_whitelist_fn(attrs, args):
            pool_size = get_const_tuple(attrs.pool_size)
            strides = get_const_tuple(attrs.strides)
            supported = (pool_size[0] <= 9) and (pool_size[1] <= 9) and (strides[0] <= 3) \
                        and (strides[1] <= 2)
            return supported

        def softmax_whitelist_fn(attrs, args):
            return (attrs.axis == -1)  # only support 1-D array softmax

        def concatenate_whitelist_fn(attrs, args):
            # Only support concatenate across channel
            return (attrs.axis == 1) or (attrs.axis == 3)

        def max_whitelist_fn(attrs, args):
            axis = attrs.axis
            supported = (attrs.exclude == False) and isinstance(axis, tvm.ir.container.Array) and \
                        (len(axis) == 2) and ((int(axis[0]) == 1 and int(axis[1]) == 2) or \
                                              (int(axis[0]) == 2 and int(axis[1]) == 3))
            return supported

        def mean_whitelist_fn(attrs, args):
            return max_whitelist_fn(attrs, args)  # same constraints as "max"

        def pad_whitelist_fn(attrs, args):
            return (float(attrs.pad_value) == 0.0 and attrs.pad_mode == 'constant')

        whitelist_funcs = {"argmax": argmax_whitelist_fn,
                           "max": max_whitelist_fn,
                           "mean": mean_whitelist_fn,
                           "nn.avg_pool2d": avg_pool_whitelist_fn,
                           "nn.batch_flatten": batch_flatten_fn,
                           "nn.batch_norm": batch_norm_whitelist_fn,
                           "nn.conv2d": conv2d_whitelist_fn,
                           "nn.conv2d_transpose": conv2d_transpose_whitelist_fn,
                           "nn.dense": dense_whitelist_fn,
                           "nn.global_avg_pool2d": global_avg_pool_whitelist_fn,
                           "nn.max_pool2d": max_pool_whitelist_fn,
                           "nn.softmax": softmax_whitelist_fn,
                           "concatenate": concatenate_whitelist_fn,
                           "nn.pad": pad_whitelist_fn,
                          }
        #print("Whitelisting " + op_name)
        return whitelist_funcs[op_name](attrs, args)

class TIDLCompiler:
    """TIDL compiler module.

    This module tries to compile a given Relay IR graph to deploy on devices with TIDL.
    If compilation for TIDL succeeds, artifacts for heterogeneous compute with TIDL
    will be generated.

    Parameters
    ----------
    platform : string
        The platform to deploy the graph on.
    version : tuple
        The Processor-SDK version for the platform.
    **kwargs : keyword arguments to pass what's needed for Relay IR graph conversion
        num_tidl_subgraphs : int
            Number of subgraphs to run on TIDL
        tidl_tools_path : string
            Folder to TIDL tools
        artifacts_folder : string
            Folder to hold TIDL artifacts
        tidl_denylist : list of strings
            Force-annotate Relay operators as unsupported
    """

    def __init__(self, platform, version, max_num_layers=225, max_total_memory_mb=448, **kwargs):
        self.tidl_platform = platform
        self.version = version
        if platform == "AM57" and version >= (6, 3):
            # Set default values for AM57 6.3
            self.tidl_target = "tidl"
            self.num_tidl_subgraphs = 1
            self.artifacts_folder = None
            self.tidl_tools_path = None
            self.tidl_tensor_bits = 8
            self.tidl_denylist = []
            # Read arguments provided through regular args
            self.max_num_layers = max_num_layers
            self.max_total_memory_mb = max_total_memory_mb
            # Read arguments provided through **kwargs
            for key in ('num_tidl_subgraphs', 'artifacts_folder', 'tidl_tools_path', 'tidl_denylist'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            self.tidl_calib_tool = os.path.join(self.tidl_tools_path,
                                                "eve_test_dl_algo_ref.out")
            self.tidl_import_lib = os.path.join(self.tidl_tools_path,
                                                "tidl_relayImport.so")
        elif platform == "J7" and version >= (7, 0):
            # Set default values for J7, PSDK 7.0 or newer
            self.tidl_target = "tidl"
            self.num_tidl_subgraphs = 1
            self.artifacts_folder = None
            self.tidl_tools_path = None
            self.tidl_tensor_bits = 16
            self.tidl_denylist = []
            # Read arguments provided through regular args
            self.max_num_layers = max_num_layers
            self.max_total_memory_mb = max_total_memory_mb
            # Read arguments provided through **kwargs
            for key in ('num_tidl_subgraphs', 'artifacts_folder',
                        'tidl_tools_path', 'tidl_tensor_bits', 'tidl_denylist'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            self.tidl_calib_tool = os.path.join(self.tidl_tools_path,
                                                "PC_dsp_test_dl_algo.out")
            self.tidl_import_lib = os.path.join(self.tidl_tools_path,
                                                "tidl_model_import_relay.so")
        else:
            sys.exit("Unsupported TIDL platform or version!")
        assert self.artifacts_folder, "artifacts_folder must be specified for TIDL compilation"
        self.temp_folder = os.path.join(self.artifacts_folder, 'tempDir/')
        self.tidl_relay_import_debug = os.environ.get("TIDL_RELAY_IMPORT_DEBUG")

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
                0  - no compilation due to missing TIDL tools
        """

        # (Backward compatible) if single calibration image/data/dict, convert to list
        if not isinstance(graph_input_list, list):
            graph_input_list = [ graph_input_list ]

        #============= Find data layout of the original graph =============
        data_layout = find_data_layout(mod_orig)

        # Open TIDL import library
        if os.path.exists(self.tidl_import_lib):
            import_lib = ctypes.CDLL(self.tidl_import_lib, mode=ctypes.RTLD_GLOBAL)
            if self.tidl_platform == "J7":
                tidl_relay_init = tvm.get_global_func("TIDL_relayInit")
                is_nchw = data_layout == "NCHW"
                tidl_relay_init(is_nchw, self.tidl_tensor_bits)
        else:
            import_lib = None # Continue with graph annotation and partition for CI testing

        # Register TIDL annotation functions
        tidl_annotation = TIDLAnnotation(self.tidl_platform, self.version, import_lib, 
                                         self.tidl_denylist)
        tidl_annotation.register_whitelist_ops()

        #============= Prepare graph for partitioning =============
        mod = relay.transform.RemoveUnusedFunctions()(mod_orig)
        # Bind params so that weights will appear as constants instead of variables
        mod['main'] = relay.build_module.bind_params_by_name(mod['main'], params)
        mod = relay.transform.FoldConstant()(mod)
        mod['main'] = RemoveMultiplyByOne().visit(mod['main'])
        mod['main'] = RemoveTrainingOperators().visit(mod['main'])
        # Removing redundant outputs
        mod = relay.transform.EliminateCommonSubexpr()(mod)
        #print("----------- original graph-----------")
        #print(mod.astext(show_meta_data=False))

        #============= Graph annotation ==============
        mod = tidl_annotation.merge_sequential_ops(mod)
        mod = relay.transform.AnnotateTarget(self.tidl_target)(mod)
        #print("----------- annotated graph-----------")
        #print(mod.astext(show_meta_data=False))

        #============= Graph partition ==============
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        #print("-----------initial partitioned graph-----------")
        #print(mod.astext(show_meta_data=False))
        if self.tidl_platform == "AM57":
            mod = prune_subgraphs_with_multiple_inputs(mod, compiler=self.tidl_target)
            mod = reduce_subgraph_size(mod, max_num_layers=self.max_num_layers,
                                       max_total_memory_mb=self.max_total_memory_mb)
        mod = unpack_composites(mod)
        mod = prune_subgraphs(mod, compiler=self.tidl_target,
                              num_subgraphs_to_keep=self.num_tidl_subgraphs,
                              min_mac_threshold=1)
        mod = flatten_tuple_params(mod, self.tidl_target)

        #============= Post-partition transformations  ==============
        with tvm.transform.PassContext(opt_level=3):
            convert_pass = [relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default']})]
            mod = tvm.transform.Sequential(convert_pass)(mod) # only affects non-TIDL subgraphs

        #print("-----------final partitioned graph-----------")
        #print(mod.astext(show_meta_data=False))

        #================ Import the graph to TIDL =====================
        if self.tidl_tools_path is not None:
            if (os.path.exists(self.tidl_calib_tool) and import_lib is not None):
                tidl_import = TIDLImport(import_lib, self.tidl_calib_tool,
                                         self.artifacts_folder,
                                         self.tidl_target, self.tidl_platform,
                                         data_layout, self.tidl_tensor_bits)
                subgraph_tensors_list = generate_subgraph_tensors(self.tidl_target, mod, params, graph_input_list, self.temp_folder)
                import_status = tidl_import.import_relay_ir(mod, params, subgraph_tensors_list)
                _ctypes.dlclose(import_lib._handle)
                if import_status == 1:
                    print("TIDL import of Relay IR graph succeeded.")
                    if self.tidl_relay_import_debug == "4":
                        generate_tidl_layer_tensors(self.tidl_target, mod, params,
                                                    graph_input_list, self.temp_folder, data_layout)
                    print("TIDL artifacts are stored at " + self.artifacts_folder)
                    mod_final, status = mod, 1        # TIDL Compilation success
                elif import_status == -1:
                    print("TIDL import of Relay IR graph failed.")
                    mod_final, status = mod_orig, -1  # TIDL Compilation failure
                else:
                    print("There are no subgraphs for TIDL offload.")
                    mod_final, status = mod_orig, 0   # No TIDL compilation
            else:
                print("TIDL import lib does not exist. TIDL import skipped.")
                mod_final, status = mod_orig, 0       # No TIDL compilation
        else:
            print("TIDL tools path is not set. TIDL import skipped.")
            mod_final, status = mod_orig, 0           # No TIDL compilation

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

def build_config(tidl_compiler=None, artifacts_folder=None, platform="AM57"):
    if tidl_compiler != None:
        artifacts_folder = tidl_compiler.artifacts_folder
        platform         = tidl_compiler.tidl_platform
    assert artifacts_folder, "artifacts_folder must be specified for TIDL codegen in compilation"
    CreateTIDLContext = tvm.get_global_func("tidl.CreateTIDLContext")
    return CreateTIDLContext(artifacts_folder, platform)

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

