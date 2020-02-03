
import numpy as np
from tvm import relay
from tvm.relay.op.annotation import tidlAnnotation
import topi
from topi.util import get_const_tuple
import ctypes



class TIDLconfigParams(ctypes.Structure):
    _fields_ = [('numParamBits', ctypes.c_int),  
                ('quantRoundAdd', ctypes.c_int), 
                ('inElementType', ctypes.c_int), 
                ('inNumChannels', ctypes.c_int), 
                ('inHeight', ctypes.c_int),      
                ('inWidth', ctypes.c_int)]


class Conv2dParams(ctypes.Structure):
    _fields_ = [('num_in_channels', ctypes.c_int), 
                ('num_out_channels', ctypes.c_int),
                ('num_groups', ctypes.c_int),
                ('stride_h', ctypes.c_int), ('stride_w', ctypes.c_int),     
                ('dilation_h', ctypes.c_int), ('dilation_w', ctypes.c_int), 
                ('pad_h', ctypes.c_int), ('pad_w', ctypes.c_int), 
                ('kernel_h', ctypes.c_int), ('kernel_w', ctypes.c_int),
                ('kernel_layout', ctypes.c_char_p),
                ('weights_array', ctypes.c_void_p),
                ('weights_type', ctypes.c_char_p)]

class InOutNodes(ctypes.Structure):
    _fields_ = [('this_node', ctypes.c_int),
                ('num_in_nodes', ctypes.c_int), ('num_out_nodes',ctypes.c_int),
                ('in_nodes', ctypes.c_void_p),  ('out_nodes',ctypes.c_void_p)]

def find_input_nodes(all_nodes, this_node):
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
    input_nodes : numpy array
        A numpy array of all input node indices of the given node
    """

    input_nodes = np.zeros(0, dtype=int)  
    node_dict_key_list = list(all_nodes.keys())
    node_dict_val_list = list(all_nodes.values())
    args = [all_nodes[arg] for arg in this_node.args]
    for idx in args:
        in_node = node_dict_key_list[node_dict_val_list.index(idx)]
        if isinstance(in_node, relay.expr.TupleGetItem):
            input_nodes = np.append(input_nodes, idx-1)
        elif isinstance(in_node, relay.expr.Call):
            input_nodes = np.append(input_nodes, idx)
             
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
    output_nodes : numpy array
        A numpy array of all output node indices of the given node
    """

    output_nodes = np.zeros(0, dtype=int)  
    node_dict_key_list = list(all_nodes.keys())
    node_dict_val_list = list(all_nodes.values())
    this_node_idx = all_nodes[this_node]
    for node, node_idx in all_nodes.items():
        if isinstance(node, relay.expr.Call):
            args = [all_nodes[arg] for arg in node.args]
            if this_node_idx in args:
                output_nodes = np.append(output_nodes, node_idx)

        if isinstance(node, relay.expr.TupleGetItem):
            next_node = node_dict_key_list[node_dict_val_list.index(node_idx+1)]
            args = [all_nodes[arg] for arg in next_node.args]
            if this_node_idx+1 in args:
                output_nodes = np.append(output_nodes, node_idx+1)
                print('Node after ' + this_node.op.name + ' is ' + next_node.op.name)

    return output_nodes


def find_in_out_nodes(all_nodes, this_node):
    r""" Find the input and output nodes of a given relay.expr.Call node.

    Parameters
    ----------
    all_nodes : dictionary 
        Dictionary of all relay.expr.Call nodes of the graph 
    this_node : relay.expr.Call
        A relay.expr.Call node whose output nodes are to be found by this function

    Returns
    -------
    in_out_nodes : InOutNodes
        Structure that stores indices of input nodes and output nodes 
    """

    in_out_nodes = InOutNodes()    # instantiate structure

    print('This node is ' + str(all_nodes[this_node]))
    in_out_nodes.this_node = all_nodes[this_node]

    in_nodes = find_input_nodes(all_nodes, this_node) # node indices of input nodes
    if len(in_nodes) == 0:
        in_out_nodes.in_nodes = None  # this is the first node
    else:
        for idx in range(len(in_nodes)):
            print(this_node.op.name + ' input node index is: ' + str(in_nodes[idx]))
        in_out_nodes.in_nodes = ctypes.c_void_p(in_nodes.ctypes.data)
    in_out_nodes.num_in_nodes = len(in_nodes)
    print('Number of input nodes is ' + str(in_out_nodes.num_in_nodes))

    out_nodes = find_out_nodes(all_nodes, this_node) # node indices of input nodes
    if len(out_nodes) == 0:
        in_out_nodes.out_nodes = None # this is the last node
    else:
        for idx in range(len(out_nodes)):
            print(this_node.op.name + ' output node index is: ' + str(out_nodes[idx]))
        in_out_nodes.out_nodes = ctypes.c_void_p(out_nodes.ctypes.data)
    in_out_nodes.num_out_nodes = len(out_nodes)

    print('Number of output nodes is ' + str(in_out_nodes.num_out_nodes))

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
    """

    weight = this_node.args[1]
    #data_shape    = get_const_tuple(data.checked_type.shape)
    weight_shape  = get_const_tuple(weight.checked_type.shape)
    weight_name   = weight.name_hint
    weight_type   = weight.checked_type.dtype
    #print(weight_name)
    #weights can be obtained by: weights=params[weight_name]
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
    (conv2d_params.pad_h, conv2d_params.pad_w) = padding
    (conv2d_params.kernel_h, conv2d_params.kernel_w) = kernel_size
    conv2d_params.num_groups = groups

    # Obtain weights from Relay params
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
    elif weight_type == 'int32':
        conv2d_params.weights_type  = b'int32'
    else:
        print('Weight type ' + weight_type + ' not supported')
        return False

    conv2d_params.weights_array = ctypes.c_void_p(weights_to_tidl.ctypes.data)
    # Invoke C lib functions to pass parameters to TIDL
    _tidlImportConv2d(conv2d_params, config_params)

    return True

def tidl_import_node(all_nodes, this_node, params):

    data = this_node.args[0]  # this_node is tvm.relay.expr.Call

    if this_node.op.name == "nn.conv2d":
        if tidl_import_conv2d(all_nodes, this_node, params) == False:
            return False

    # Common for all nodes:
    # prepare to fill tensor names, update consumer counts, link input/output tensors
    in_out_nodes = find_in_out_nodes(all_nodes, this_node)
    _tidlImportSetInOutNodes(in_out_nodes, config_params)

    return True


def relay_ir_import(mod, params):

    _tidlImportInit()

    all_nodes = {}
    relay.analysis.post_order_visit(mod['main'], lambda node: tidlAnnotation.traverse_expr(node, all_nodes)) 

    for node in all_nodes:
        if isinstance(node, relay.expr.Call):
            result = tidl_import_node(all_nodes, node, params)
            if result == False:
                return result

    return True

_file = './tidl_relayImport.so'
_tidl_mod = ctypes.CDLL(_file, mode=ctypes.RTLD_GLOBAL)

config_params = TIDLconfigParams(12,50,1,3,224,224)

_tidlImportInit = _tidl_mod.tidlImportInit
_tidlImportInit.argtypes = None
_tidlImportInit.restype  = None

_tidlImportConv2d = _tidl_mod.tidlImportConv2d
#_tidlImportConv2d = _tidl_mod.tidlSetConv2dParams
_tidlImportConv2d.argtypes = (ctypes.POINTER(Conv2dParams), ctypes.POINTER(TIDLconfigParams)) 
_tidlImportConv2d.restype = None

_tidlImportSetInOutNodes = _tidl_mod.tidlImportSetInOutNodes
_tidlImportSetInOutNodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.POINTER(TIDLconfigParams))
_tidlImportSetInOutNodes.restype = None

