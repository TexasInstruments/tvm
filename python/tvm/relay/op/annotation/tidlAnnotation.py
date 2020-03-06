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

######################################################################
# Overview for Supported Hardware Backend of TVM
# ----------------------------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/tvm_support_list.png
#      :align: center
#      :scale: 100%
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import Relay and TVM.
import onnx
import numpy as np
import inspect 

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime

import topi
from topi.util import get_const_tuple

from tvm.relay.expr_functor import ExprVisitor

class RelayGraphParams:
    def __init__(self):
        self.data_layout = 'UNDEFINED'

    def SetDataLayout(self, layout):
        self.data_layout = layout

    def GetDataLayout(self):
        return self.data_layout

    def DataLayoutIsSet(self):
        return(self.data_layout != 'UNDEFINED')

def traverse_expr(node, node_dict):
    if node in node_dict:
        return
    if isinstance(node, relay.op.op.Op):
        return 
    node_dict[node] = len(node_dict)


def find_in_call_nodes(node_dict, this_node):
    r""" Find the input nodes of a given relay.expr.Call node.
        If the input node is a relay.expr.TupleGetItem node, then go up one more level.

    Parameters
    ----------
       node_dict : dict
           Dictionary of all nodes of the graph 
       this_node :  relay.expr.Call
           Node (operator) whose input nodes are to be found
    Returns
    -------
       inpCallNodeDict : dict
           Dictionary of all input nodes of the given node
    """

    inpCallNodeDict = {}
    node_dict_key_list = list(node_dict.keys())
    node_dict_val_list = list(node_dict.values())
    args = [node_dict[arg] for arg in this_node.args]
    for idx in args:
        inpCallNode = node_dict_key_list[node_dict_val_list.index(idx)]
        if isinstance(inpCallNode, relay.expr.TupleGetItem):
            inpCallNode = node_dict_key_list[node_dict_val_list.index(idx-1)]
            inpCallNodeDict[len(inpCallNodeDict)] = inpCallNode
        elif isinstance(inpCallNode, relay.expr.Call):
            inpCallNodeDict[len(inpCallNodeDict)] = inpCallNode
             
    return inpCallNodeDict

def find_out_call_nodes(node_dict, this_node):
    r""" Find the output nodes of a given relay.expr.Call node.

    Parameters
    ----------
       node_dict : dict
           Dictionary of all nodes of the graph 
       this_node :  relay.expr.Call
           Node (operator) whose output nodes are to be found

    Returns
    -------
       outCallNodeDict : dict
           Dictionary of all output nodes of the given node
    """

    outCallNodeDict = {}
    node_dict_key_list = list(node_dict.keys())
    node_dict_val_list = list(node_dict.values())
    thisNodeIdx = node_dict[this_node]
    for node, nodeIdx in node_dict.items():
        if isinstance(node, relay.expr.Call):
            args = [node_dict[arg] for arg in node.args]
            if thisNodeIdx in args:
                outCallNodeDict[len(outCallNodeDict)] = node

        if isinstance(node, relay.expr.TupleGetItem):
            next_node = node_dict_key_list[node_dict_val_list.index(nodeIdx+1)]
            args = [node_dict[arg] for arg in next_node.args]
            if thisNodeIdx+1 in args:
                outCallNodeDict[len(outCallNodeDict)] = next_node

    return outCallNodeDict


def tidl_node_validation(node_dict, call_node):
    r""" Decide if a relay.expr.Call node can be supported by TIDL or not.
        Relay Operator documentation: https://docs.tvm.ai/langref/relay_op.html

    Parameters
    ----------
       node_dict : dictionary
           Dictionary of all nodes of the graph 
       call_node :  relay.expr.Call
           Node (operator) to be checked if it can be supported by TIDL
    Returns
    -------
       True  - if this node (operator) can be supported by TIDL
       False - if this node (operator) can not be supported by TIDL
    """

    #print("===== OP: " + call_node.op.name + " =====")
    data = call_node.args[0]  # call_node is tvm.relay.expr.Call

    if hasattr(call_node.attrs, 'data_layout') and (not graph_params.DataLayoutIsSet()):
        graph_params.data_layout = call_node.attrs.data_layout

    # Check the op to decide if it is supported by TIDL
    if call_node.op.name == "add":
        return True

    elif call_node.op.name == "nn.argmax":
        keepdims  = call_node.attrs.keepdims
        exclude   = call_node.attrs.exclude
        axis      = call_node.attrs.axis
        supported = (int(data.checked_type.shape[1]) <= 15 and keepdims == 1 and axis == 1 and exclude == 0)
        return (supported)

    elif call_node.op.name == "nn.avg_pool2d":
        pool_size = get_const_tuple(call_node.attrs.pool_size)
        strides   = get_const_tuple(call_node.attrs.strides)
        supported = (pool_size[0] <= 9 and pool_size[1] <= 9 and strides[0] <= 3 and strides[1] <=2)
        return (supported)

    elif call_node.op.name == "nn.batch_flatten":
        if(len(data.checked_type.shape) == 4):
            supported = (int(data.checked_type.shape[2]) <= 65535 and int(data.checked_type.shape[3]) <= 65535)
        else:
            supported = True
        return (supported)

    elif call_node.op.name == "nn.batch_norm":
        if call_node.args[1].checked_type.dtype != 'float32':
            supported = False
        elif graph_params.data_layout == 'NCHW' and call_node.attrs.axis != 1:
        #only axis along channel is supported
        #attributes include parameters that are optional and having default values in operator arguments
            supported = False
        elif graph_params.data_layout == 'NHWC' and call_node.attrs.axis != 3:
            supported = False
        else:
            supported = True
        return supported

    elif call_node.op.name == "nn.bias_add":
        return True

    elif call_node.op.name == "clip":
        a_min = call_node.attrs.a_min
        a_max = call_node.attrs.a_max
        supported = (a_min == 0 and a_max == 6)
        #print('nn.clip.a_min is ' + str(a_min) + ', ' + 'nn.clip.a_max is ' + str(a_max))
        return (supported)

    elif call_node.op.name == "nn.concatenate":
        return (call_node.attrs.axis == 1)

    elif call_node.op.name == "nn.conv2d":
        # There is an example how to get the attributes of conv2d in Relay:
        # https://github.com/dmlc/tvm/blob/master/python/tvm/relay/op/nn/_nn.py#L144
        weight = call_node.args[1]
        if weight.checked_type.dtype != 'float32':
            return False
        data_shape    = get_const_tuple(data.checked_type.shape)
        weight_shape  = get_const_tuple(weight.checked_type.shape)
        strides       = get_const_tuple(call_node.attrs.strides)
        dilation      = get_const_tuple(call_node.attrs.dilation)
        padding       = get_const_tuple(call_node.attrs.padding)
        kernel_size   = get_const_tuple(call_node.attrs.kernel_size)
        groups        = call_node.attrs.groups
        data_layout   = call_node.attrs.data_layout
        kernel_layout = call_node.attrs.kernel_layout
        out_layout    = call_node.attrs.out_layout
        out_dtype     = call_node.attrs.out_dtype
        
        (dh, dw) = dilation
        (kh, kw) = kernel_size
        channel_supported  = (weight_shape[0] <= 2048 and weight_shape[1] <= 2048)
        stride_supported   = (strides[0] <= 2 and strides[1] <= 2)
        dilation_supported = (dh == 1 or dh == 2 or dh == 4) and (dw == 1 or dw == 2 or dw == 4)
        kernel_supported = (((kh-1)*dh+1) <= 9) and (((kw-1)*dw+1) <= 9)
        supported = channel_supported and stride_supported and dilation_supported and kernel_supported
        return (supported)

    elif call_node.op.name == "nn.conv2d_transpose":
        weight = call_node.args[1]
        weight_shape  = get_const_tuple(weight.checked_type.shape)
        strides       = get_const_tuple(call_node.attrs.strides)
        groups        = call_node.attrs.groups

        supported = (weight_shape[0] == weight_shape[1]) and (weight_shape[0] == groups) and (strids[1] == 2)
        return (supported)

    elif call_node.op.name == "nn.dense":
        weight = call_node.args[1]
        weight_shape  = get_const_tuple(weight.checked_type.shape)
        w_in  = weight_shape[1]
        w_out = weight_shape[0]
        supported = (w_in <= 65536) and (w_out <= 16384) and (w_in * w_out <= 67108864)
        return (supported)

    elif call_node.op.name == "nn.dropout":
        return True

    elif call_node.op.name == "nn.global_avg_pool2d":
        data_shape  = get_const_tuple(data.checked_type.shape)
        layout = call_node.attrs.layout
        if layout == "NCHW":
            height = data_shape[2]
            width  = data_shape[3]
        else:
            height = data_shape[1]
            width  = data_shape[2]
        supported = (height * width <= 4096)
        return (supported)

    elif call_node.op.name == "nn.max_pool2d":
        pool_size = get_const_tuple(call_node.attrs.pool_size)
        strides   = get_const_tuple(call_node.attrs.strides)
        supported = (pool_size[0] <= 9) and (pool_size[1] <= 9) and (strides[0] <= 3) and (strides[1] <= 2)
        return (supported)

    elif call_node.op.name == "vision.multibox_prior":
        supported = 0
        outCallNodes = find_out_call_nodes(node_dict, call_node)
        for idx in outCallNodes:
            if outCallNodes[idx].op.name == "nn.concatenate" or \
               outCallNodes[idx].op.name == "vision.nms":
                supported = 1

        return (supported)

    elif call_node.op.name == "multiply":
        return True

    elif call_node.op.name == "nn.nms":
        return True

    elif call_node.op.name == "nn.pad":
        return (call_node.attrs.pad_value == 0.0 and call_node.attrs.pad_mode == 'constant')

    elif call_node.op.name == "nn.prelu":
        return True

    elif call_node.op.name == "nn.relu":
        return True

    elif call_node.op.name == "reshape":
        supported = False
        reshape_after_transpose = False
        transpose_after_reshape = False
        inpCallNodes = find_in_call_nodes(node_dict, call_node)
        for idx in inpCallNodes:
            if inpCallNodes[idx].op.name == "nn.avg_pool2d" or \
               inpCallNodes[idx].op.name == "nn.global_avg_pool2d" or \
               inpCallNodes[idx].op.name == "nn.dense" or \
               inpCallNodes[idx].op.name == "squeeze":
                supported = True
            elif inpCallNodes[idx].op.name == "transpose":
                reshape_after_transpose = True

        outCallNodes = find_out_call_nodes(node_dict, call_node)
        for idx in outCallNodes:
            if outCallNodes[idx].op.name == "nn.softmax":
                supported = True
            elif outCallNodes[idx].op.name == "transpose":
                transpose_after_reshape = True

        if reshape_after_transpose and transpose_after_reshape:
            supported = True

        # If this is the last node of the graph, and input and output shape are 
        # the same, this operator can be supported by TIDL
        if len(outCallNodes) ==0:
            node_is_identity = True
            for idx in range(len(data.checked_type.shape)):
                if int(data.checked_type.shape[idx]) != int(call_node.attrs.newshape[idx]):
                    node_is_identity = False
            if node_is_identity == True:
                supported = True

        return (supported)

    elif call_node.op.name == "slice_like":
        return (call_node.attrs.axis == 1)

    elif call_node.op.name == "nn.softmax":
        return (call_node.attrs.axis != 2)

    elif call_node.op.name == "split":
        return True

    elif call_node.op.name == "squeeze":
        supported = False
        outCallNodes = find_out_call_nodes(node_dict, call_node)
        for idx in outCallNodes:
            if outCallNodes[idx].op.name == "reshape":
                supported = True

        return (supported)

    elif call_node.op.name == "transpose":
        supported = False
        reshape_after_transpose = False
        transpose_after_reshape = False
        outCallNodes = find_out_call_nodes(node_dict, call_node)
        for idx in outCallNodes:
            if outCallNodes[idx].op.name == "nn.batch_flatten":
                supported = True
            elif outCallNodes[idx].op.name == "reshape":
                reshape_after_transpose = True

        inpCallNodes = find_in_call_nodes(node_dict, call_node)
        for idx in inpCallNodes:
            if inpCallNodes[idx].op.name == "reshape":
                transpose_after_reshape = True

        if reshape_after_transpose and transpose_after_reshape:
            supported = True
        return (supported)
    else:
        return False

def tidl_annotation(mod):
    r""" Annotate each operator (node) in a given graph as supported by TIDL or not

    Parameters
    ----------
       mod : tvm.relay.Module 
           Relay IR graph

    Returns
    -------
       op_annotations : dict 
           Dictionary of relay.expr.Call nodes with one of the following values:
           - True : if the node (operator) can be supported by TIDL
           - False: if the node (operator) can not be supported by TIDL
    """

    # Traverse the graph and generate a dictionary of all nodes
    node_dict = {}
    relay.analysis.post_order_visit(mod['main'], lambda node: traverse_expr(node, node_dict)) 

    op_annotations = {}
    for node in node_dict:
        # Only look at relay.expr.Call node
        if isinstance(node, relay.expr.Call):
            op_annotations[node] = tidl_node_validation(node_dict, node)

    return op_annotations

graph_params = RelayGraphParams()
