
import numpy as np
from tvm import relay
from tvm.relay.op.annotation import tidlAnnotation
import topi
from topi.util import get_const_tuple
import ctypes



class TIDLconfigParams(ctypes.Structure):
    _fields_ = [('numParamBits', ctypes.c_int),  
                ('quantRoundAdd', ctypes.c_int), 
                ('inQuantFactor', ctypes.c_int), 
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
    input_nodes : list
        A list of all input node indices of the given node
    """

    input_nodes = []
    node_dict_key_list = list(all_nodes.keys())
    node_dict_val_list = list(all_nodes.values())
    args = [all_nodes[arg] for arg in this_node.args]
    for idx in args:
        in_node = node_dict_key_list[node_dict_val_list.index(idx)]
        if isinstance(in_node, relay.expr.TupleGetItem):
            input_nodes.append(idx-1)
        elif isinstance(in_node, relay.expr.Call):
            input_nodes.append(idx)
             
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
    node_dict_key_list = list(all_nodes.keys())
    node_dict_val_list = list(all_nodes.values())
    this_node_idx = all_nodes[this_node]
    for node, node_idx in all_nodes.items():
        if isinstance(node, relay.expr.Call):
            args = [all_nodes[arg] for arg in node.args]
            if this_node_idx in args:
                output_nodes.append(node_idx)

        if isinstance(node, relay.expr.TupleGetItem):
            next_node = node_dict_key_list[node_dict_val_list.index(node_idx+1)]
            args = [all_nodes[arg] for arg in next_node.args]
            if this_node_idx+1 in args:
                output_nodes.append(node_idx+1)
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

    node_dict_key_list = list(all_nodes.keys())   # debugging
    node_dict_val_list = list(all_nodes.values()) # debugging

    in_out_nodes = InOutNodes()    # instantiate structure

    in_out_nodes.this_node = all_nodes[this_node]

    in_nodes = find_input_nodes(all_nodes, this_node) # node indices of input nodes
    print('number of input nodes: ' + str(len(in_nodes)))
    if len(in_nodes) == 0:
        in_out_nodes.in_nodes = None  # this is the first node
    else:
        for idx in range(len(in_nodes)):
            print('input node: ' + str(in_nodes[idx]) + ', ' + node_dict_key_list[in_nodes[idx]].op.name)
        # convert list to numpy arrary in order to pass to C library
        in_nodes_array = np.asarray(in_nodes, dtype=np.int32)
        in_out_nodes.in_nodes = ctypes.c_void_p(in_nodes_array.ctypes.data)

    in_out_nodes.num_in_nodes = len(in_nodes)

    out_nodes = find_out_nodes(all_nodes, this_node) # node indices of input nodes
    print('number of output nodes: ' + str(len(out_nodes)))
    if len(out_nodes) == 0:
        in_out_nodes.out_nodes = None # this is the last node
    else:
        for idx in range(len(out_nodes)):
            print('output node: ' + str(out_nodes[idx]) + ', ' + node_dict_key_list[out_nodes[idx]].op.name)
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
    #elif weight_type == 'int8':
    #    conv2d_params.weights_type  = b'int8'
    else:
        print('Weight type ' + weight_type + ' not supported')
        return False

    conv2d_params.weights_array = ctypes.c_void_p(weights_to_tidl.ctypes.data)
    # Invoke C lib functions to pass parameters to TIDL
    _tidlImportConv2d(conv2d_params, ctypes.POINTER(ctypes.c_int)())
    #_tidlImportConv2d(conv2d_params)

    return True

def tidl_import_pad(node):
    r""" Import pad operator to TIDL
        Get attributes pad_width, convert to array, and passs to C library.
        A typical pad_width looks like: [[0,0],[0,1],[0,1],[0,0]

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
    _tidlImportPad(len(pad_array), ctypes.c_void_p(pad_array.ctypes.data))


def tidl_import_add(node, params):
    r""" Import add operator to TIDL
        An "add" operator may be adding two nodes or adding one node with const
            - %3 = add(%2, %1) 
            - %3 = add(%2, %MobilenetV2/Conv/Conv2D_bn_offset) 

        Need to distinguish these 2 cases and invoke corresponding TIDL mapping 
        functions:

    Parameters
    ----------
    node : relay.expr.Call
        A relay.expr.Call node which is a pad operator
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay

    Returns
    -------
    """

    if isinstance(node.args[1], relay.expr.Var):
        print('This is a bias_add operator')
        bias = node.args[1]
        bias_params_name = bias.name_hint
        if bias.checked_type.dtype == 'float32':
            bias_params_dtype = b'float32'
        #elif bias.checked_type.dtype == 'int8':
        #    bias_params_dtype = b'int8'
        else:
            printf('Unsupported data type of add')
            return False

        bias_params_len = bias.checked_type.shape[0]
        bias_params = params[bias_params_name]
        bias_params_np = bias_params.asnumpy()
        _tidlImportBiasAdd(bias_params_len, bias_params_dtype,
                           ctypes.c_void_p(bias_params_np.ctypes.data))
    elif isinstance(node.args[1], relay.expr.Call):
        print('This is an add operator')
        _tidlImportAdd()
    else:
        printf('Error in importing add operator')
        return False

    return True


def tidl_import_init(all_nodes):
    r""" Initializing TIDL import

    Parameters
    ----------
    all_nodes : dictionary 
        Dictionary of all relay.expr.Call nodes of the graph 

    Returns
    -------
    """

    # Find first node of the graph and get input tensor shape
    for node in all_nodes:
        if isinstance(node, relay.expr.Call): # node is tvm.relay.expr.Call
            # find input nodes of this node
            in_nodes = find_input_nodes(all_nodes, node) 
            if len(in_nodes) == 0:
                data = node.args[0]
                print('Found first node')
                break
    input_shape = get_const_tuple(data.checked_type.shape) 

    # Find first conv2d node to get data layout (first node may not have this infomation)
    for node in all_nodes:
        if isinstance(node, relay.expr.Call): # node is tvm.relay.expr.Call
            if node.op.name == "nn.conv2d":
                print('Found first conv2d node')
                break

    # Fill dimension parameters for TIDL based on input tensor shape and data layout
    if node.attrs.data_layout == "NCHW":
        config_params.inNumChannels = input_shape[1]
        config_params.inHeight      = input_shape[2]
        config_params.inWidth       = input_shape[3]
    elif node.attrs.data_layout == "NHWC":
        config_params.inNumChannels = input_shape[3]
        config_params.inHeight      = input_shape[1]
        config_params.inWidth       = input_shape[2]
    else:
        print('data layout ' + node.attrs.data_layout + ' is not supported')
        return False

    # pass a NULL pointer - Python to C has to have 2 arguments (to figure out why)
    _tidlImportInit(config_params, ctypes.POINTER(ctypes.c_int)())

    return True

def tidl_import_node(all_nodes, this_node, params):
    r""" Importing each supported node to TIDL
        # https://docs.tvm.ai/langref/relay_op.html#relay-core-tensor-operators
        #--- to add:
        #Operator add: True
        #Operator clip: True
        #Operator nn.batch_norm: True
        #Operator nn.avg_pool2d: True
        #Operator squeeze: True
        #Operator reshape: True
    """

    print('----- Node ' + str(all_nodes[this_node]) + ', ' + this_node.op.name + '-----')

    status = True
    if this_node.op.name == 'nn.conv2d':
        status = tidl_import_conv2d(all_nodes, this_node, params)
    elif this_node.op.name == "nn.pad":
        status = tidl_import_pad(this_node)
    elif this_node.op.name == "add":
        status = tidl_import_add(this_node, params)


    #else:
    #    status = False

    if status == False:
        return status

    # Common for all nodes:
    # fill tensor names, update consumer counts, link input/output tensors
    in_out_nodes = find_in_out_nodes(all_nodes, this_node)
    _tidlImportLinkNodes(in_out_nodes, config_params)

    return True

def relay_ir_import(mod, params):

    all_nodes = {}
    relay.analysis.post_order_visit(mod['main'], lambda node: tidlAnnotation.traverse_expr(node, all_nodes)) 

    if tidl_import_init(all_nodes) == False:
        return False

    for node in all_nodes:
        if isinstance(node, relay.expr.Call):
            result = tidl_import_node(all_nodes, node, params)
            if result == False:
                return result

    if _tidlImportOptimize() == -1:
        return False

    return True

_file = './tidl_relayImport.so'
_tidl_mod = ctypes.CDLL(_file, mode=ctypes.RTLD_GLOBAL)

config_params = TIDLconfigParams(12,50,255,1,3,224,224)

_tidlImportInit = _tidl_mod.tidlImportInit
_tidlImportInit.argtypes = (ctypes.POINTER(TIDLconfigParams), ctypes.c_void_p)
#_tidlImportInit.argtype = (ctypes.POINTER(TIDLconfigParams))
_tidlImportInit.restype = None

_tidlImportConv2d = _tidl_mod.tidlImportConv2d
_tidlImportConv2d.argtypes = (ctypes.POINTER(Conv2dParams), ctypes.c_void_p) 
_tidlImportConv2d.restype  = None

_tidlImportPad = _tidl_mod.tidlImportPad
_tidlImportPad.argtypes = (ctypes.c_int, ctypes.c_void_p)
_tidlImportPad.restype  = None

_tidlImportAdd = _tidl_mod.tidlImportAdd
_tidlImportAdd.argtypes = None
_tidlImportAdd.restype  = None

_tidlImportBiasAdd = _tidl_mod.tidlImportBiasAdd
_tidlImportBiasAdd.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_tidlImportBiasAdd.restype  = None

_tidlImportLinkNodes = _tidl_mod.tidlImportLinkNodes
_tidlImportLinkNodes.argtypes = (ctypes.POINTER(InOutNodes), ctypes.POINTER(TIDLconfigParams))
_tidlImportLinkNodes.restype = None

_tidlImportOptimize = _tidl_mod.tidlImportOptimize
_tidlImportOptimize.argtypes = None
_tidlImportOptimize.restype  = ctypes.c_int