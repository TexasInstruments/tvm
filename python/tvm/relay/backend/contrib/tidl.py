
import subprocess

import numpy as np
import tvm
from tvm import relay
import topi
from topi.util import get_const_tuple
import ctypes
import os
import re

from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar
from tvm.relay.op import Op
from tvm.relay.function import Function
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime

#TODO: get tidl_import_lib from upper level test code - test_tidl.py
if os.getenv("TIDL_TOOLS_PATH") is None:
    sys.exit("Environment variable TIDL_TOOLS_PATH is not set!")
else:
    tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
tidl_import_lib = os.path.join(tidl_tools_path, "tidl_relayImport.so")
_tidl_mod = ctypes.CDLL(tidl_import_lib, mode=ctypes.RTLD_GLOBAL)


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
                ('var',  ctypes.c_void_p),
                ('epsilon', ctypes.c_float),
                ('center_enable', ctypes.c_int),
                ('scale_enable', ctypes.c_int)]

class PoolingParams(ctypes.Structure):
    """ Pooling parameters defined in ctypes for passing to TIDL C library """
    _fields_ = [('kernelH', ctypes.c_int), 
                ('kernelW', ctypes.c_int), 
                ('strideH', ctypes.c_int), 
                ('strideW', ctypes.c_int), 
                ('padH',    ctypes.c_int),
                ('padW',    ctypes.c_int)]

class MulParams(ctypes.Structure):
    _fields_ = [('scale', ctypes.c_float)]

class InOutNodes(ctypes.Structure):
    """ Input/output nodes defined in ctypes for passing to TIDL C library """
    _fields_ = [('this_node', ctypes.c_int),
                ('num_in_nodes', ctypes.c_int), ('num_out_nodes',ctypes.c_int),
                ('in_nodes', ctypes.c_void_p),  ('out_nodes',ctypes.c_void_p)]

def find_in_nodes(all_nodes, this_node):
    r""" Find the input nodes of a given relay.expr.Call node.
    
         Only find input nodes that are relay.expr.Call.
         If an input node is a relay.expr.TupleGetItem, then check this input
         node's input node.

    Parameters
    ----------
    all_nodes : dictionary 
        Dictionary of all nodes of the graph 
    this_node : relay.expr.Call
        A relay.expr.Call node whose input nodes are to be found by this function

    Returns
    -------
    input_nodes : list
        A list of all input node indices of the given node
    """

    input_nodes = []
    if isinstance(this_node, relay.expr.Call):
        in_nodes = this_node.args
    elif isinstance(this_node, relay.expr.Tuple):
        in_nodes = this_node.fields

    for node in in_nodes:
        if isinstance(node, relay.expr.Call):
            input_nodes.append(all_nodes[node])
        elif isinstance(node, relay.expr.TupleGetItem):
            input_nodes.append(all_nodes[node.tuple_value])
        elif isinstance(node, relay.expr.Tuple):
            input_nodes = input_nodes + find_in_nodes(all_nodes, node)
        elif isinstance(node, relay.expr.Var):
            if "tidl" in node.name_hint and "_i" in node.name_hint:
                # this is the input to the subgraph
                input_nodes.append(-1)
        #else: ignore all other types of nodes: var, const, etc.

    return input_nodes


def find_out_nodes(all_nodes, this_node):
    r""" Find the output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary 
        Dictionary of all relay.expr.Call nodes of the graph 
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
            if this_node in node.args:
                output_nodes.append(node_idx)
        elif isinstance(node, relay.expr.TupleGetItem):
            if this_node == node.tuple_value:
                output_nodes = output_nodes + find_out_nodes(all_nodes, node)
        elif isinstance(node, relay.expr.Tuple):
            for i in range(len(node.fields)):
                if this_node == node.fields[i]:
                    tuple_node_outs = find_out_nodes(all_nodes, node)
                    if len(tuple_node_outs) == 0:
                        # this is an output node
                        output_nodes.append(all_nodes[node])
                    else:
                        # this is an input node to another node
                        output_nodes = output_nodes + tuple_node_outs

    return output_nodes


def find_in_out_nodes(all_nodes, this_node):
    r""" Find the input and output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary 
        Dictionary of all relay.expr.Call nodes of the graph 
    this_node : relay.expr.Call
        A relay.expr.Call node whose input and output nodes are to be found

    Returns
    -------
    in_out_nodes : InOutNodes
        Structure that stores indices of input nodes and output nodes 
    """

    node_dict_key_list = list(all_nodes.keys())   # debugging
    node_dict_val_list = list(all_nodes.values()) # debugging

    in_out_nodes = InOutNodes()    # instantiate structure

    in_out_nodes.this_node = all_nodes[this_node]

    in_nodes = find_in_nodes(all_nodes, this_node) # node indices of input nodes
    #print('number of input nodes: ' + str(len(in_nodes)))
    if len(in_nodes) == 0:
        in_out_nodes.in_nodes = None  # this is the first node
    else:
        #for idx in range(len(in_nodes)):
        #    print('input node: ' + str(in_nodes[idx]) + ', ' + node_dict_key_list[in_nodes[idx]].op.name)
        # convert list to numpy arrary in order to pass to C library
        in_nodes_array = np.asarray(in_nodes, dtype=np.int32)
        in_out_nodes.in_nodes = ctypes.c_void_p(in_nodes_array.ctypes.data)

    in_out_nodes.num_in_nodes = len(in_nodes)

    out_nodes = find_out_nodes(all_nodes, this_node) # node indices of output nodes
    #print('number of output nodes: ' + str(len(out_nodes)))
    if len(out_nodes) == 0:
        in_out_nodes.out_nodes = None # this is the last node
    else:
        #for idx in range(len(out_nodes)):
        #    print('output node: ' + str(out_nodes[idx]) + ', ' + node_dict_key_list[out_nodes[idx]].op.name)
        # convert list to numpy arrary in order to pass to C library
        out_nodes_array = np.asarray(out_nodes, dtype=np.int32)
        in_out_nodes.out_nodes = ctypes.c_void_p(out_nodes_array.ctypes.data)

    in_out_nodes.num_out_nodes = len(out_nodes)

    return in_out_nodes


def tidl_import_conv2d(all_nodes, this_node, params):
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
    #data_shape    = get_const_tuple(data.checked_type.shape)
    weight_shape  = get_const_tuple(weight.checked_type.shape)
    weight_type   = weight.checked_type.dtype
    strides       = get_const_tuple(this_node.attrs.strides)
    dilation      = get_const_tuple(this_node.attrs.dilation)
    padding       = get_const_tuple(this_node.attrs.padding)
    kernel_size   = get_const_tuple(this_node.attrs.kernel_size)
    groups        = this_node.attrs.groups
    data_layout   = this_node.attrs.data_layout
    kernel_layout = this_node.attrs.kernel_layout
    out_layout    = this_node.attrs.out_layout
    out_dtype     = this_node.attrs.out_dtype

    conv2d_params = Conv2dParams()
    (conv2d_params.stride_h, conv2d_params.stride_w) = strides
    (conv2d_params.dilation_h, conv2d_params.dilation_w) = dilation
    # TODO: to pass all padding values to TIDL and use strideOffsetMethod
    # top, left, bottom, right
    if len(padding) == 1:
        pad_t = pad_l = pad_b = pad_r = padding[0]
    elif len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        (pad_t, pad_l, pad_b, pad_r) = padding
    (conv2d_params.pad_t, conv2d_params.pad_l, conv2d_params.pad_b, conv2d_params.pad_r) = (pad_t, pad_l, pad_b, pad_r)
    (conv2d_params.kernel_h, conv2d_params.kernel_w) = kernel_size
    conv2d_params.num_groups = groups

    # Obtain weights from Relay params
    if isinstance(weight,tvm.relay.expr.Constant):
        weights = weight.data
    else:
        #print(weight_name)
        weight_name = weight.name_hint
        weights = params[weight_name] 
    # Convert to numpy array and then pass to C
    weights_np = weights.asnumpy()

    if kernel_layout == 'OIHW':
        # No need to reshape - TIDL natively uses 'OIHW'
        conv2d_params.kernel_layout = b'OIHW'
        conv2d_params.num_in_channels  = weight_shape[1]
        conv2d_params.num_out_channels = weight_shape[0]
        weights_to_tidl = weights_np
    elif kernel_layout == 'HWIO':
        # Reshape numpy array from 'HWIO' to 'OIHW'
        weights_to_tidl = weights_np.transpose((3,2,0,1))
        conv2d_params.num_in_channels  = weight_shape[2]
        conv2d_params.num_out_channels = weight_shape[3]
    elif kernel_layout == 'HWOI':
        # Reshape numpy array from 'HWOI' to 'OIHW'
        weights_to_tidl = weights_np.transpose((2,3,0,1))
        conv2d_params.num_in_channels  = weight_shape[3]
        conv2d_params.num_out_channels = weight_shape[2]
    else:
        print('Kernel layout ' + kernel_layout + ' not supported')
        return False

    if weight_type == 'float32':
        conv2d_params.weights_type  = b'float32'
    #elif weight_type == 'int8':
    #    conv2d_params.weights_type  = b'int8'
    else:
        print('Weight type ' + weight_type + ' not supported')
        return False

    weights_flatten = weights_to_tidl.flatten()
    conv2d_params.weights_array = ctypes.c_void_p(weights_flatten.ctypes.data)

    # Invoke C lib functions to pass parameters to TIDL
    _tidlImportConv2d = _tidl_mod.tidlImportConv2d
    _tidlImportConv2d.argtypes = (ctypes.POINTER(Conv2dParams), ctypes.c_void_p) 
    _tidlImportConv2d.restype  = None
    _tidlImportConv2d(conv2d_params, ctypes.POINTER(ctypes.c_int)())

    return True

def tidl_import_pad(node):
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
    for i in range(len(node.attrs.pad_width)):
        pad_width.append(get_const_tuple(node.attrs.pad_width[i]))
    pad_list = [x for xs in pad_width for x in xs]

    # convert list to numpy array in order to pass to C library
    pad_array = np.asarray(pad_list, dtype=np.int32)

    _tidlImportPad = _tidl_mod.tidlImportPad
    _tidlImportPad.argtypes = (ctypes.c_int, ctypes.c_void_p)
    _tidlImportPad.restype  = None
    _tidlImportPad(len(pad_array), ctypes.c_void_p(pad_array.ctypes.data))

def tidl_import_add(node):
    r""" Import add operator to TIDL
        An "add" operator may be adding two nodes or adding one node with constant:
            - %3 = add(%2, %1) 
            - %3 = add(%2, %MobilenetV2/Conv/Conv2D_bn_offset) 
        This function imports "add" opertor with args being two nodes.
    """

    _tidlImportAdd = _tidl_mod.tidlImportAdd
    _tidlImportAdd.argtypes = None
    _tidlImportAdd.restype  = None
    _tidlImportAdd()

def tidl_import_bias_add(node, params):
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
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay

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
    #elif bias.checked_type.dtype == 'int8':
    #    bias_params_dtype = b'int8'
    else:
        print('Unsupported data type of bias_add')
        return False

    bias_params_len = bias.checked_type.shape[0]
    bias_params_np = bias_params.asnumpy()

    _tidlImportBiasAdd = _tidl_mod.tidlImportBiasAdd
    _tidlImportBiasAdd.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    _tidlImportBiasAdd.restype  = None
    _tidlImportBiasAdd(bias_params_len, bias_params_dtype,
                       ctypes.c_void_p(bias_params_np.ctypes.data))

def tidl_import_batch_norm(node, params):
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
    #elif node.args[1].checked_type.dtype == 'int8':
    #    bn_params.params_dtype = b'int8'
    else:
        print('Unsupported data type of batch norm')
        return False
    bn_params.num_params = node.args[1].checked_type.shape[0]

    # Obtain weights from Relay params
    if isinstance(node.args[1],tvm.relay.expr.Constant):
        gama = node.args[1].data.asnumpy()
        beta = node.args[2].data.asnumpy()
        mean = node.args[3].data.asnumpy()
        var  = node.args[4].data.asnumpy()
    else:
        gama = params[node.args[1].name_hint].asnumpy()
        beta = params[node.args[2].name_hint].asnumpy()
        mean = params[node.args[3].name_hint].asnumpy()
        var  = params[node.args[4].name_hint].asnumpy()
    #print('Batch norm parameters:')
    #print(gama)
    #print(beta)
    #print(mean)
    #print(var )
    bn_params.gama = gama.ctypes.data
    bn_params.beta = beta.ctypes.data
    bn_params.mean = mean.ctypes.data
    bn_params.var  = var.ctypes.data
    bn_params.epsilon = node.attrs.epsilon
    center = node.attrs.center
    scale  = node.attrs.scale
    bn_params.center_enable = int(center == True)
    bn_params.scale_enable  = int(scale  == True)

    _tidlImportBatchNorm = _tidl_mod.tidlImportBatchNorm
    _tidlImportBatchNorm.argtypes = (ctypes.POINTER(BatchNormParams), ctypes.c_void_p)
    _tidlImportBatchNorm.restype  = None
    _tidlImportBatchNorm(bn_params, ctypes.POINTER(ctypes.c_int)())

    return True

def tidl_import_pooling(node, type):
    r""" Import pooling operator to TIDL
        https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.avg_pool2d
        https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1AvgPool2DAttrs.html
        https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.max_pool2d
        https://docs.tvm.ai/doxygen/structtvm_1_1relay_1_1MaxPool2DAttrs.html

    Parameters
    ----------
    node : relay.expr.Call
        A relay.expr.Call node which is a pooling operator
    type : Bytes literals 
        A string indicating the type of the pooling operator

    Returns
    -------
    """

    pooling_params = PoolingParams()
    if node.op.name == "nn.global_avg_pool2d":
        pooling_params.kernelH = pooling_params.kernelW = 0
        pooling_params.padH    = pooling_params.padW    = 0
        pooling_params.strideH = pooling_params.strideW = 1
    else:
        (pooling_params.kernelH,pooling_params.kernelW) = node.attrs.pool_size
        (pooling_params.strideH,pooling_params.strideW) = node.attrs.strides
        if len(node.attrs.padding) == 4:
            (pooling_params.padH,pooling_params.padW) = node.attrs.padding[2:4]
        else:
            (pooling_params.padH,pooling_params.padW) = node.attrs.padding

    if node.op.name == "nn.avg_pool2d" or node.op.name == "nn.global_avg_pool2d":
        type = b'avg_pool2d'
    else:
        type = b'max_pool2d'

    _tidlImportPooling = _tidl_mod.tidlImportPooling
    _tidlImportPooling.argtypes = (ctypes.POINTER(PoolingParams), ctypes.c_char_p)
    _tidlImportPooling.restype  = None
    _tidlImportPooling(pooling_params, type)

    return

def tidl_import_concat(all_nodes, node):

    in_nodes = find_in_nodes(all_nodes, node) # node indices of input nodes
    #print("Importing concatenate layer, number of input nodes: " + str(len(in_nodes)))
    _tidlImportConcat = _tidl_mod.tidlImportConcat
    _tidlImportConcat.argtype = ctypes.c_int
    _tidlImportConcat.restype = None
    _tidlImportConcat(len(in_nodes))

def tidl_import_dense(this_node):

    weights = this_node.args[1]
    (num_outnodes, num_innodes) = weights.data.shape
    weights_array = weights.data.asnumpy()
    _tidlImportDense = _tidl_mod.tidlImportDense
    _tidlImportDense.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    _tidlImportDense.restype  = None
    _tidlImportDense(num_innodes, num_outnodes, ctypes.c_void_p(weights_array.ctypes.data))

def tidl_import_mul(this_node):
    mul_params = MulParams()
    scale = this_node.args[0].data.asnumpy()
    mul_params.scale = np.amax(scale)
    _tidlImportMul = _tidl_mod.tidlImportMul
    _tidlImportMul.argtypes = (ctypes.POINTER(MulParams), ctypes.c_void_p)
    _tidlImportMul.restype = None
    _tidlImportMul(mul_params, ctypes.POINTER(ctypes.c_int)())

def tidl_import_init(data_layout, input_scale, input_signed, input_shape):
    r""" Initializing TIDL import

    Parameters
    ----------
    data_layout: string
    input_scale: double
        Scaling factor to convert floating point input to 8-bit quantized input
    input_signed: int
        Signed (1) or unsigned (0) of input
    input_shape: tuple
        Input shape (N,C,H,W) or (N,H,W,C)
    Returns
    -------
    True if initialization succeeds or False if initialization fails
    """

    if len(input_shape) == 2:
        # input is a vector - expand (N,W) to (N,1,1,W) or (N,1,W,1)
        if data_layout == "NCHW":
            in_shape = (input_shape[0],1,1,input_shape[1])
        else:
            in_shape = (input_shape[0],1,input_shape[1],1)
    elif len(input_shape) == 4:
        in_shape = input_shape
    else:
        print("Subgraph input_shape " + str(input_shape) + " is not supported")
        return False

    if data_layout == "NCHW":
        layout = b'NCHW'
        (channel, height, width) = in_shape[1:4]
    elif data_layout == "NHWC":
        layout = b'NHWC'
        (channel, height, width) = (in_shape[3],in_shape[1],in_shape[2])
    else:
        print('data layout ' + node.attrs.data_layout + ' is not supported')
        return False

    inQuantFactor = int(round(input_scale*255))  # 255 is due to TIDL implementation
    config_params = TIDLconfigParams(12,50,inQuantFactor,input_signed,
                                     channel, height, width)

    # Invoking C library call to initialize TIDL import
    _tidlImportInit = _tidl_mod.tidlImportInit
    _tidlImportInit.argtypes = (ctypes.POINTER(TIDLconfigParams), ctypes.c_char_p)
    _tidlImportInit.restype = None
    _tidlImportInit(config_params, layout)

    return True

def tidl_import_node(all_nodes, this_node, params):
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

    #print('----- Node ' + str(all_nodes[this_node]) + ', ' + this_node.op.name + '-----')

    status = True
    if this_node.op.name == 'nn.conv2d':
        status = tidl_import_conv2d(all_nodes, this_node, params)
    elif this_node.op.name == 'nn.pad':
        status = tidl_import_pad(this_node)
    elif this_node.op.name == 'add':
        if isinstance(this_node.args[1], tvm.relay.expr.Constant):
            status = tidl_import_bias_add(this_node, params)
        else:
            status = tidl_import_add(this_node)
    elif this_node.op.name == 'nn.bias_add':
        status = tidl_import_bias_add(this_node, params)
    elif this_node.op.name == 'clip':
        _tidlImportRelu = _tidl_mod.tidlImportRelu
        _tidlImportRelu.argtype = (ctypes.c_char_p)
        _tidlImportRelu.restype  = None
        _tidlImportRelu(b'Relu6')
    elif this_node.op.name == 'nn.relu':
        _tidlImportRelu = _tidl_mod.tidlImportRelu
        _tidlImportRelu.argtype = (ctypes.c_char_p)
        _tidlImportRelu.restype  = None
        _tidlImportRelu(b'Relu')
    elif this_node.op.name == 'nn.batch_norm':
        status = tidl_import_batch_norm(this_node, params)
    elif this_node.op.name == 'nn.avg_pool2d':
        status = tidl_import_pooling(this_node, b'avg_pool2d')
    elif this_node.op.name == 'squeeze':
        _tidlImportSqueeze = _tidl_mod.tidlImportSqueeze
        _tidlImportSqueeze.argtype = None
        _tidlImportSqueeze.restype = None
        _tidlImportSqueeze()
    elif this_node.op.name == 'reshape':
        _tidlImportReshape = _tidl_mod.tidlImportReshape
        _tidlImportReshape.argtype = None
        _tidlImportReshape.restype = None
        _tidlImportReshape()
    elif this_node.op.name == 'nn.softmax':
        _tidlImportSoftmax = _tidl_mod.tidlImportSoftmax
        _tidlImportSoftmax.argtype = None
        _tidlImportSoftmax.restype = None
        _tidlImportSoftmax()
    elif this_node.op.name == 'concatenate':
        status = tidl_import_concat(all_nodes, this_node)
    elif this_node.op.name == 'nn.max_pool2d':
        status = tidl_import_pooling(this_node, b'max_pool2d')
    elif this_node.op.name == 'nn.dropout':
        _tidlImportDropOut = _tidl_mod.tidlImportDropOut
        _tidlImportDropOut.argtype = None
        _tidlImportDropOut.restype = None
        _tidlImportDropOut()
    elif this_node.op.name == 'nn.global_avg_pool2d':
        status = tidl_import_pooling(this_node, b'avg_pool2d')
    elif this_node.op.name == 'nn.batch_flatten':
        _tidlImportBatchFlatten = _tidl_mod.tidlImportBatchFlatten
        _tidlImportBatchFlatten.argtype = None
        _tidlImportBatchFlatten.restype = None
        _tidlImportBatchFlatten()
    elif this_node.op.name == 'multiply':
        #print("Importing multiply")
        tidl_import_mul(this_node)
    elif this_node.op.name == 'nn.dense':
        #print("Importing dense")
        tidl_import_dense(this_node)
    else:
        print("Operator " + this_node.op.name + " is not supported!")
        status = False

    if status == False:
        return False

    # Common for all nodes:
    # fill tensor names, update consumer counts, link input/output tensors
    in_out_nodes = find_in_out_nodes(all_nodes, this_node)

    _tidlImportLinkNodes = _tidl_mod.tidlImportLinkNodes
    _tidlImportLinkNodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.c_void_p)
    _tidlImportLinkNodes.restype = ctypes.c_int
    if _tidlImportLinkNodes(in_out_nodes, ctypes.POINTER(ctypes.c_int)()) == 0:
        return False

    return True

def obtain_subgraph_tensor(subgraph_tensors, tensor_name_prefix):
    r""" Obtain input/output tensor for a given subgraph

    Parameters
    ----------

    Returns
    -------
    """
    tensor = []
    for key, value in subgraph_tensors.items():
        if key.find(tensor_name_prefix) != -1:
            tensor.append(value)
    return tensor

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
        str1 = str0.replace("[","")
        str2 = str1.replace("]","")
        str3 = str2.replace(",","",len(in_list)-1)
        return str3

    if data_layout=="NCHW":
        isNCHW = 1
    else:
        isNCHW = 0
    out_conv_type = []
    out_is_nchw   = []
    for i in range(len(output_scale)):
        out_conv_type.append(0)
        out_is_nchw.append(isNCHW)

    sub_graph_cfg = os.path.join(artifacts_folder, "subgraph" + str(subgraph_id) + ".cfg")
    sub_graph_net_file = "./tidl_subgraph" + str(subgraph_id) + "_net.bin"
    sub_graph_params_file = "./tidl_subgraph" + str(subgraph_id) + "_params.bin"
    with open(sub_graph_cfg, 'w') as cfg_file:
        cfg_file.write("netBinFile    = {}\n".format(sub_graph_net_file))
        cfg_file.write("paramsBinFile = {}\n".format(sub_graph_params_file))
        cfg_file.write("inConvType    = 0\n")
        cfg_file.write("inIsSigned    = {}\n".format(input_signed))
        cfg_file.write("inScaleF2Q    = {}\n".format(round(input_scale,2)))
        cfg_file.write("inIsNCHW      = {}\n".format(isNCHW))
        cfg_file.write("outConvType   = {}\n".format(print_list(out_conv_type)))
        cfg_file.write("outIsSigned   = {}\n".format(print_list(output_signed)))
        cfg_file.write("outScaleF2Q   = {}\n".format(print_list(output_scale)))
        cfg_file.write("outIsNCHW     = {}\n".format(print_list(out_is_nchw)))

def tidl_import_tuple_node(all_nodes, node):
    """
    """

    MAX_NUM_OUTPUTS_PER_DATA_LAYER = 16
    out_nodes = find_out_nodes(all_nodes, node)
    if len(out_nodes) == 0:
        # this is the last node of the graph - import this to out data layer
        in_nodes = find_in_nodes(all_nodes, node)
        imported_nodes = 0
        new_node_ind = len(all_nodes)+1
        while (imported_nodes < len(in_nodes)):
            if len(in_nodes) - imported_nodes < MAX_NUM_OUTPUTS_PER_DATA_LAYER:
                nodes_for_this_data_layer = len(in_nodes) - imported_nodes
                this_is_the_last_one = True
            else:
                nodes_for_this_data_layer = MAX_NUM_OUTPUTS_PER_DATA_LAYER
                this_is_the_last_one = False
        
            print("Importing out data layer, number of input nodes: " + str(nodes_for_this_data_layer))
            _tidlImportOutData = _tidl_mod.tidlImportOutData
            _tidlImportOutData.argtype = ctypes.c_int
            _tidlImportOutData.restype = None
            _tidlImportOutData(nodes_for_this_data_layer)
    
            # prepare input/output nodes information for linking
            in_out_nodes = InOutNodes()    # instantiate structure
            in_out_nodes.this_node = new_node_ind
            in_out_nodes.num_in_nodes = nodes_for_this_data_layer
            in_nodes_this_layer = in_nodes[imported_nodes:imported_nodes+nodes_for_this_data_layer]
            in_nodes_array = np.asarray(in_nodes_this_layer, dtype=np.int32)
            in_out_nodes.in_nodes = ctypes.c_void_p(in_nodes_array.ctypes.data)
            in_out_nodes.out_nodes = None
            in_out_nodes.num_out_nodes = 0
    
            _tidlImportLinkNodes = _tidl_mod.tidlImportLinkNodes
            _tidlImportLinkNodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.c_void_p)
            _tidlImportLinkNodes.restype = ctypes.c_int
            if _tidlImportLinkNodes(in_out_nodes, ctypes.POINTER(ctypes.c_int)()) == 0:
                return False

            imported_nodes = imported_nodes + nodes_for_this_data_layer
            new_node_ind = new_node_ind + 1
            if this_is_the_last_one== True:
                break
        return True
    else:
        # this is not the last node of the graph - ignore it
        return True

def import_relay_ir(tidl_target, mod, params, subgraph_tensors, data_layout, tidl_calib_tool, artifacts_folder):
    r""" Relay IR import to TIDL 

    Parameters
    ----------
    mod : tvm.relay.Module 
        Relay IR graph with subgraphs
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    subgraph_tensors: dict
        Input/output tensors of subgraphs obtained from TVM graph execution

    Returns
    -------
    True:  if TIDL import succeeds or if there are no subgraphs for TIDL offload
    False: if TIDL import fails    
    """

    # Traverse Relay IR graph and generate a dictionary of all TIDL subgraphs
    all_nodes_main = {}
    relay.analysis.post_order_visit(mod['main'], lambda node: traverse_expr(node, all_nodes_main)) 
    tidl_subgraphs = []
    for node in all_nodes_main:
        if isinstance(node, relay.expr.GlobalVar):
            if tidl_target in node.name_hint:
                tidl_subgraphs.append(node.name_hint)

    if len(tidl_subgraphs) == 0:
        # There are no subgraphs for TIDL offload
        return False

    # For each TIDL subgraph, import to TIDL and calibrate 
    for tidl_subgraph in tidl_subgraphs:
        # Extract subgraph id and input/output tensor names from subgraph name
        subgraph_id = int(tidl_subgraph.replace(tidl_target+'_',''))
        in_tensor_name  = tidl_subgraph + '_i'
        out_tensor_name = tidl_subgraph + '_o'

        # Obtain input tensor from TVM graph execution
        input_fp = obtain_subgraph_tensor(subgraph_tensors, in_tensor_name)
        if input_fp is None:
            return False
        if len(input_fp) > 1:
            print("Error - only 1 input tensor is supported for now!")
            return False

        # Quantize input tensor into 8-bit integer (only support 1 input tensor)
        input_quant_vec, input_scale, input_signed = tensor_quant_flatten(input_fp[0], data_layout)

        # Initialize TIDL import
        if tidl_import_init(data_layout, input_scale, input_signed, input_fp[0].shape) == False:
            return False

        # Scan through all relay.expr.Call nodes and import each to TIDL
        # TODO: change this and _tidlImportOptimize to a function
        all_nodes_tidl = {}
        relay.analysis.post_order_visit(mod[tidl_subgraph], lambda node: traverse_expr(node, all_nodes_tidl)) 
        for node in all_nodes_tidl:
            if isinstance(node, relay.expr.Call):
                result = tidl_import_node(all_nodes_tidl, node, params)
                if result == False:
                    return False

        # Import expr.Tuple node after importing all expr.call nodes
        for node in all_nodes_tidl:
            if isinstance(node, relay.expr.Tuple):
                #node.fields: array of expr.call nodes
                result = tidl_import_tuple_node(all_nodes_tidl, node)
                if result == False:
                    print('Error importing output tuple node')
                    return False
    
        # Invoke TIDL optimization of the imported graph
        net_file = os.path.join(artifacts_folder, 'tidl_subgraph' + str(subgraph_id) + '_net.bin')
        par_file = os.path.join(artifacts_folder, 'tidl_subgraph' + str(subgraph_id) + '_params.bin')

        _tidlImportOptimize = _tidl_mod.tidlImportOptimize
        _tidlImportOptimize.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        _tidlImportOptimize.restype = ctypes.c_int
        net_fname = net_file.encode('utf-8')
        par_fname = par_file.encode('utf-8')
        
        if _tidlImportOptimize(net_fname, par_fname, subgraph_id) == 0:
            print('tidl import optimize failed')
            return False

        # Calibrate TIDL for the imported subgraph
        status, out_data_q = subgraph_calibration(tidl_calib_tool, input_quant_vec, 
                                                  input_signed, net_file, par_file)
        if status == False:
            return False
        
        # Calculate scaling factor to convert output tensor to floating point
        # Obtain output tensor from TVM graph execution
        output_fp = obtain_subgraph_tensor(subgraph_tensors, out_tensor_name)
        
        # TODO: convert following lines into a function
        if output_fp is None:
            return False
        if len(output_fp) != len(out_data_q):
            return False
        output_signed = []
        output_scale  = []
        for i in range(len(output_fp)):
            output_signed.append(int(np.amin(output_fp[i]) < 0))
        for i in range(len(out_data_q)):
            output_scale.append(round(out_data_q[i]/255.0,5))  # 255 is TIDL implementation specific
        print("Output conversion: " + str(out_data_q) + ", " + str(output_scale))
        
        # Generate subgraph configuration file
        subgraph_cfg_gen(artifacts_folder, subgraph_id, data_layout, 
                         input_scale, input_signed, output_scale, output_signed)
    return True

def tensor_quant_flatten(input_tensor, data_layout):
    r""" Convert float32 n-d array to int8 or uint8 1-d array
    Parameters
    ----------
    input: float32 array 
    data_layout: "NCHW" or "NHWC"
    output: 1 dimension int8 or uint8 array
    """

    # only use 1 batch for calibration
    input_tensor = input_tensor[0,:]
    # change layout to CxHxW to use numpy.flattern to change to 1-d array
    if data_layout == "NHWC" and len(input_tensor.shape) == 3:
        input_tensor = input_tensor.transpose(2,0,1)

    if np.amin(input_tensor) >= 0:
        # quantize to Uint8
        sign  = 0
        scale = 255.0/np.amax(input_tensor)
        quant_min = 0
        quant_max = 255
    else:
        # quantize to Int8
        sign  = 1
        scale = 128.0/max(abs(np.amin(input_tensor)), np.amax(input_tensor))
        quant_min = -128
        quant_max = 127
        
    y = np.multiply(input_tensor, scale)
    z = np.rint(y)
    z = np.clip(z, quant_min, quant_max)
    output = z.flatten()   # works only if z is in "CxHxW" format

    return output, scale, sign

def subgraph_calibration(calib_tool, input_quant_vec, input_signed, net_file, params_file):

    # Save quantized input vector to a file for calib tool to read
    # Saving as 'int8' or 'uint8' is the same
    calib_raw_image = './calib_raw_data.bin'
    if input_signed == 1:
        input_quant_vec.astype('int8').tofile(calib_raw_image);
    else:
        input_quant_vec.astype('uint8').tofile(calib_raw_image);

    # Prepare for calibration
    output_tmp_file = './tempDir/precalib_net.bin'
    proc = subprocess.Popen(['rm', '-rf', 'tempDir'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = subprocess.Popen(['mkdir', 'tempDir'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = subprocess.Popen(['cp', net_file, output_tmp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    calib_config_file = './tempDir/configFilesList.txt'
    with open(calib_config_file, 'w') as config_file:
        config_file.write('1 ./tempDir/quant_stats_config.txt\n')
        config_file.write('0\n')

    quant_config_file = './tempDir/quant_stats_config.txt'
    with open(quant_config_file, 'w') as quant_file:
        quant_file.write('rawImage    = 1\n')
        quant_file.write('numFrames   = 1\n')
        quant_file.write('preProcType = 0\n')
        quant_file.write('inData      = {}\n'.format(calib_raw_image))
        quant_file.write('outData     = {}\n'.format('./tempDir/stats_tool_out.bin'))
        quant_file.write('traceDumpBaseName  = {}\n'.format('./tempDir/trace_dump_'))
        quant_file.write('updateNetWithStats = 1\n')
        quant_file.write('outputNetBinFile   = {}\n'.format(net_file))
        quant_file.write('paramsBinFile      = {}\n'.format(params_file))
        quant_file.write('netBinFile         = {}\n'.format(output_tmp_file))

    # Invoke TIDL emulation to calibrate
    try:
        proc = subprocess.Popen([calib_tool, calib_config_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        o, e = proc.communicate()
        console_out = o.decode('ascii')
        error = e.decode('ascii')
        print(console_out)
    except:
        print("TIDL calibration crashed")
        return False, None

    # Find output dataQs
    if console_out.find('error')==-1 and console_out.find('ERROR')==-1 and error == '':
        output_dataQ_token = "Number of output dataQ:"
        out_buf_ind = console_out.rfind(output_dataQ_token)
        if out_buf_ind == -1:
            print("TIDL calibration failed - can't find number of output buffers.")
            return False, None
        else:
            last_line   = console_out.split(output_dataQ_token,1)[1]
            num_outputs = int(last_line.split(". Output dataQ:",1)[0])
            out_quants  = last_line.split(". Output dataQ:",1)[1]
            quants = out_quants.split("End of output dataQ",1)[0]
            qs = re.findall(r"\d+", quants)
            outq = list(map(int,qs))
            if num_outputs != len(outq):
                print("TIDL calibration failed - can't find all outputQ's")
                return False, None
            else:
                return True, outq
    else:
        print("TIDL calibration failed.")
        print(error)
        return False, None

class VarReplacer(ExprMutator):
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

def UnpackComposites(mod, compiler="tidlj6"):
    class Unpacker(ExprMutator):
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
        name = func.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        mod[name] = Unpacker().visit(mod[name])
    return mod

class CalibrationGraphMutator(ExprMutator):
    """This mutator should be called after partioning to produce a module which
    can be executed purely using TVM and will produce additional outputs for
    subgraph inputs. name_map can be used to find the subgraph input name
    corresponding to the output of the same index.
    """
    def __init__(self, compiler):
        ExprMutator.__init__(self)
        self.additional_outputs = []
        self.compiler = compiler
        # Will map index in output to subgraph param name.
        self.name_map = {}

    def add_new_outputs(self, subgraph_name, expr, was_input=True):
        """Adds expr as an additional output to be generated by the module. If expr is a tuple, multiple outputs will be added."""
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
        if isinstance(call.op, Function) and "Compiler" in call.op.attrs and call.op.attrs["Compiler"] == self.compiler:
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
        self.num_original_outputs = 1
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

#TODO: move enable_tidl to tidl.py
class RemoveMultiplyByOne(ExprMutator):
    """
    Removes multiply by 1.0f. This pass when followed by
    RemoveRedundantTranspose is intended to remove a pattern of
    Transpose([1, 0]) -> Scale(1.0f) -> Transpose([1, 0]) produced by
    PyTorch's addmm operator.
    """
    def visit_call(self, expr):
        if expr.op.name == "multiply":
            if isinstance(expr.args[1], tvm.relay.expr.Constant):
                data = expr.args[1].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return expr.args[0]
            if isinstance(expr.args[0], tvm.relay.expr.Constant):
                data = expr.args[0].data.asnumpy()
                if data.shape == () and data.item() == 1.0:
                    return expr.args[1]
        return super().visit_call(expr)

def generate_subgraph_tensors(tidl_target, mod, params, input_node, input_data):
    """
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    my_mutator = CalibrationGraphMutator(tidl_target)
    mod_tvm["main"] = my_mutator.make_calibration_graph(mod_tvm["main"])
    #print("Calibration module:", mod_tvm)
    print("Input map:", my_mutator.name_map)

    # Build and execute calibration graph to get outputs
    # Use opt_level=0 to avoid optimizations which modify the module (could change original module)
    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(input_node, input_data)
    mod.set_input(**params)
    mod.run()
    #mod.run(data=input_data, weight1=params_w1, weight2=params_w2)

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    np.savetxt('graph_output.txt', results[0].flatten(), fmt='%10.5f')

    # We now have subgraph inputs
    # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
    subgraph_tensors = {}
    for i in range(len(results)):
        if i in my_mutator.name_map:
            subgraph_tensors[my_mutator.name_map[i]]=results[i]
            #print("Subgraph input: ", my_mutator.name_map[i], " tensor: ", results[i])
            file_name = my_mutator.name_map[i] + ".txt"
            np.savetxt(file_name, results[i].flatten(), fmt='%10.5f')

    for key, value in subgraph_tensors.items():
        print("Subgraph tensor: ", key, value.shape)

    return subgraph_tensors

class VarReplacer(ExprMutator):
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

class VarRenamer(ExprMutator):
    def __init__(self, new_subgraph_name):
        ExprMutator.__init__(self)
        self.new_subgraph_name = new_subgraph_name

    def visit_var(self, var):
        if "_".join(var.name_hint.split('_')[:2]) != self.new_subgraph_name:
            new_var_name = self.new_subgraph_name + "_" + var.name_hint.split('_')[2]
            return relay.Var(new_var_name, var.checked_type)
        return super().visit_var(var)

class SubgraphRemover(ExprMutator):
    def __init__(self, subgraphs_to_remove, mod, new_mod, rename_starting_from_0=True):
        ExprVisitor.__init__(self)
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
            elif name != "main":
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

def PruneSubgraphsWithMoreThanOneInput(mod, compiler="tidlj6"):
    subgraph_names_to_remove = []
    # Remove subgraphs with more than 1 input or tuple inputs.
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        print("SUBGRAPH PARAMS", mod[name].params)
        if len(mod[name].params) != 1 or isinstance(mod[name].params[0].checked_type, relay.TupleType):
            subgraph_names_to_remove.append(name)
    print("Removing subgraphs due to having more than one input:", subgraph_names_to_remove)
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod

def PruneSubgraphs(mod, compiler="tidlj6", num_subgraphs_to_keep=4):
    subgraph_with_macs = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        num_macs = relay.analysis.get_total_mac_number(mod[name])
        subgraph_with_macs.append([name, num_macs])
    subgraph_with_macs = sorted(subgraph_with_macs, key=lambda x: int(x[1]))
    print("Subgraphs with computed # of MACS:", subgraph_with_macs)
    subgraphs_to_remove = subgraph_with_macs[:-num_subgraphs_to_keep]
    print("Will remove these subgraphs:", subgraphs_to_remove)
    subgraph_names_to_remove = set([x[0] for x in subgraphs_to_remove])
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod

def EnableTIDLJ6(mod, params, num_tidl_subgraphs, 
                 data_layout, input_node, input_data, 
                 artifacts_folder, calib_tool):

    import tvm.relay.op.contrib.tidl_j6 as tidl
    tidl_target = "tidlj6"

    mod = relay.transform.RemoveUnusedFunctions()(mod)
    # Bind params so that weights will appear as constants instead of variables 
    mod['main'] = bind_params_by_name(mod['main'], params)
    mod = relay.transform.FoldConstant()(mod)
    mod['main'] = RemoveMultiplyByOne().visit(mod['main'])
    print("---------- Original graph ----------")
    print(mod.astext(show_meta_data=False))

    #============= Annotate the graph ==============
    # Looks at annotated ops and marks them in the graph with compiler.begin 
    # and compiler.end.
    # Merges annotated regions together that use the same external target, 
    # and combines marked regions for each target
    #Merge sequence of ops into composite functions/ops
    print("---------- Merge Composite Functions ----------")
    mod = tidl._merge_sequential_ops(mod) 
    print("---------- Annotated Graph ----------")
    mod = transform.AnnotateTarget(tidl_target)(mod)
    #print(mod.astext(show_meta_data=False))
    print("---------- Merge Compiler Regions ----------")
    mod = transform.MergeCompilerRegions()(mod)
    #print(mod.astext(show_meta_data=False))
    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    #print(mod.astext(show_meta_data=False))
    print("---------- Unpack composite ops in the graph ----------")
    mod = UnpackComposites(mod, compiler=tidl_target)
    #print(mod.astext(show_meta_data=False))
    print('NUM TIDL SUBGRAPHS BEFORE PRUNING:', sum([1 for subgraph in mod.get_global_vars() if subgraph.name_hint.startswith(tidl_target)]))
    print("---------- Prune Graph ----------")
    mod = PruneSubgraphsWithMoreThanOneInput(mod, compiler=tidl_target)
    #print(mod.astext(show_meta_data=False))
    mod = PruneSubgraphs(mod, compiler=tidl_target, num_subgraphs_to_keep=num_tidl_subgraphs)
    print('NUM TIDL SUBGRAPHS AFTER PRUNING:', sum([1 for subgraph in mod.get_global_vars() if subgraph.name_hint.startswith(tidl_target)]))
    print(mod.astext(show_meta_data=False))

    #============= Generate subgraph boundary tensors ==============
    subgraph_tensors = generate_subgraph_tensors(tidl_target, mod, params, input_node, input_data)

    #======================== Import the graph to TIDL ========================
    if import_relay_ir(tidl_target, mod, params, subgraph_tensors, data_layout, calib_tool, artifacts_folder) == True:
        print("Graph execution with TIDL.")
        return mod
    else:
        print("Graph execution with general CPU.")
        return None
