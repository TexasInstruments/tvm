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
# pylint: disable=invalid-name, unused-argument
"""TIDL library supported operators.
"""
from ... import op as reg
from topi.util import get_const_tuple
from tvm import relay
from tvm.relay.frontend.common import infer_shape
from tvm.relay.frontend.common import infer_type

target = "target.tidl"

def _merge_sequential_ops(mod):
    """Fuse sequential ops for op registration. Ops: vision.multibox_prior, nn.reshape, squeeze, transpose
    """
    # Squeeze has to be followed by reshape.
    def _squeeze_reshape_pattern():
        x = relay.var('x')
        squeeze_out = relay.op.transform.squeeze(x)
        reshape_out = relay.op.transform.reshape(squeeze_out, [])
        return reshape_out

    #tranpose has to be followed by reshape (special pattern)
    #transpose has to be preceded by reshape (special pattern, reshape transpose reshape)
    def _transpose_reshape_pattern():
        x = relay.var('x')
        reshape_out1 = relay.op.transform.reshape(x, [])
        transpose_out = relay.op.transform.transpose(reshape_out1)
        reshape_out2 = relay.op.transform.reshape(transpose_out, [])
        return reshape_out2

    def _transpose_batch_reshape_pattern():
        x = relay.var('x')
        reshape_out1 = relay.op.transform.reshape(x, [])
        transpose_out = relay.op.transform.transpose(reshape_out1)
        batch_flatten_out = relay.op.nn.batch_flatten(transpose_out)
        reshape_out2 = relay.op.transform.reshape(batch_flatten_out, [])
        return reshape_out2

    #multibox_prior has to be followed by nn.concatenate or vision.nms
    def _multibox_prior_concat_pattern():
        x = relay.var('x')
        y = relay.var('y')
        multibox_prior_out = relay.op.vision.multibox.multibox_prior(x)
        concatenate_out = relay.op.tensor.concatenate((multibox_prior_out, y), 0)
        return concatenate_out

    def _multibox_prior_nms_pattern():
        x = relay.var('x')
        y = relay.var('y')
        multibox_prior_out = relay.op.vision.multibox.multibox_prior(x)
        nms_out = relay.op.vision.nms.non_max_suppression(multibox_prior_out, y)
        return nms_out

    #reshape has to be preced by nn.avg_pool2d, nn.global_avg_pool2d, nn.dense, squeeze, transpose (if transpose, special path)
    #reshape should be followed by softmax or transpose (special path, has to be transpose, reshape, transpose)
    def _reshape_avg_pool_pattern():
        x = relay.var('x')
        avg_pool_out = relay.op.nn.avg_pool2d(x)
        reshape_out = relay.op.transform.reshape(avg_pool_out, [])
        return reshape_out

    def _reshape_global_avg_pool_pattern():
        x = relay.var('x')
        global_avg_pool_out = relay.op.nn.global_avg_pool2d(x)
        reshape_out = relay.op.transform.reshape(global_avg_pool_out, [])
        return reshape_out

    def _reshape_dense_pattern():
        x = relay.var('x')
        y = relay.var('y')
        dense_out = relay.op.nn.dense(x, y)
        reshape_out = relay.op.transform.reshape(dense_out, [])
        return reshape_out

    def _reshape_softmax_pattern():
        x = relay.var('x')
        reshape_out = relay.op.transform.reshape(x, [])
        softmax_out = relay.op.nn.softmax(reshape_out)
        return softmax_out

    def _reshape_transpose_pattern():
        x = relay.var('x')
        transpose_out = relay.op.transform.transpose(x)
        reshape_out = relay.op.transform.reshape(transpose_out, [])
        transpose_out1 = relay.op.transform.transpose(reshape_out)
        return transpose_out1

    def _conv2d_relu_pattern():
        x = relay.var('x')
        w = relay.var('w')
        conv2d_out = relay.op.nn.conv2d(x, w)
        relu_out = relay.op.nn.relu(conv2d_out)
        return relu_out

    def _conv2d_bias_relu_pattern():
        x = relay.var('x')
        w = relay.var('w')
        b = relay.var('b')
        conv2d_out = relay.op.nn.conv2d(x, w)
        bias_out = relay.op.nn.bias_add(conv2d_out, b)
        relu_out = relay.op.nn.relu(bias_out)
        return relu_out

    def _bn_relu_pattern():
        x = relay.var('x')
        gamma = relay.var("gamma")
        beta = relay.var("beta")
        moving_mean = relay.var("moving_mean")
        moving_var = relay.var("moving_var")
        bn_out = relay.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
        relu_out = relay.op.nn.relu(bn_out[0])
        return relu_out

    def _add_relu_pattern():
        x = relay.var('x')
        y = relay.var('y')
        add_out = relay.op.add(x, y)
        relu_out = relay.op.nn.relu(add_out)
        return relu_out

    def _dense_relu_pattern():
        x = relay.var('x')
        w = relay.var('w')
        dense_out = relay.op.nn.dense(x, w)
        relu_out = relay.op.nn.relu(dense_out)
        return relu_out

    def _dense_bias_relu_pattern():
        x = relay.var('x')
        w = relay.var('w')
        b = relay.var('b')
        dense_out = relay.op.nn.dense(x, w)
        bias_out = relay.op.nn.bias_add(dense_out, b)
        relu_out = relay.op.nn.relu(bias_out)
        return relu_out

    def _conv2d_bias_pattern():
        x = relay.var('x')
        w = relay.var('w')
        b = relay.var('b')
        conv2d_out = relay.op.nn.conv2d(x, w)
        bias_out = relay.op.nn.bias_add(conv2d_out, b)
        return bias_out

    def _dense_bias_pattern():
        x = relay.var('x')
        w = relay.var('w')
        b = relay.var('b')
        dense_out = relay.op.nn.dense(x, w)
        bias_out = relay.op.nn.bias_add(dense_out, b)
        return bias_out

    def _conv2d_pad_pattern():
        x = relay.var('x')
        w = relay.var('w')
        p = list()
        conv2d_out = relay.op.nn.conv2d(x, w)
        pad_out = relay.op.nn.pad(conv2d_out,p)
        return pad_out

    pattern_table = [
        ('tidl.squeeze_reshape', _squeeze_reshape_pattern()),
        ('tidl.transpose_reshape', _transpose_reshape_pattern()),
        ('tidl.tanspose_batch_reshape', _transpose_batch_reshape_pattern()),
        ('tidl.multibox_prior_concat', _multibox_prior_concat_pattern()),
        ('tidl.mutlibox_prior_nms', _multibox_prior_nms_pattern()),
        ('tidl.reshape_avgpool', _reshape_avg_pool_pattern()),
        ('tidl.reshape_globalavgpool', _reshape_global_avg_pool_pattern()),
        ('tidl.reshape_dense', _reshape_dense_pattern()),
        ('tidl.reshape_softmax', _reshape_softmax_pattern()),
        ('tidl.reshape_transpose', _reshape_transpose_pattern()),
        ('tidl.conv2d_relu', _conv2d_relu_pattern()),
        ('tidl.conv2d_bias_relu', _conv2d_bias_relu_pattern()),
        ('tidl.bn_relu', _bn_relu_pattern()),
        ('tidl.add_relu',_add_relu_pattern()),
        ('tidl.dense_relu', _dense_relu_pattern()),
        ('tidl.dense_bias_relu', _dense_bias_relu_pattern()),
        ('tidl.conv2d_bias', _conv2d_bias_pattern()),
        ('tidl.dense_bias', _dense_bias_pattern()),
        ('tidl.conv2d_pad', _conv2d_pad_pattern()),
    ]

    return relay.transform.MergeComposite(pattern_table)(mod)

@reg.register("tidl.squeeze_reshape", "target.tidl")
def _tidl_squeeze_reshape_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.transpose_reshape", "target.tidl")
def _tidl_transpose_reshape_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.transpose_batch_reshape", "target.tidl")
def _tidl_transpose_batch_reshape_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.multibox_prior_concat", "target.tidl")
def _tidl_multibox_prior_concat_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.mutlibox_prior_nms", "target.tidl")
def _tidl_mutlibox_prior_nms_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.reshape_avgpool", "target.tidl")
def _tidl_reshape_avgpool_whitelist_fn(attrs, args):
    return _avg_pool_whitelist_fn(attrs, args)

@reg.register("tidl.reshape_globalavgpool", "target.tidl")
def _tidl_reshape_globalavgpool_whitelist_fn(attrs, args):
    return _global_avg_pool_whitelist_fn(attrs, args)

@reg.register("tidl.reshape_dense", "target.tidl")
def _tidl_reshape_dense_whitelist_fn(attrs, args):
    return _dense_whitelist_fn(attrs, args)

@reg.register("tidl.reshape_squeeze", "target.tidl")
def _tidl_reshape_squeeze_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.reshape_softmax", "target.tidl")
def _tidl_reshape_softmax_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.reshape_transpose", "target.tidl")
def _tidl_reshape_transpose_whitelist_fn(attrs, args):
    return True

@reg.register("tidl.conv2d_relu", "target.tidl")
def _conv2d_relu_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@reg.register("tidl.conv2d_bias_relu", "target.tidl")
def _conv2d_bias_relu_whitelist_fn(attrs, args):
    conv2d_op = args[0].args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@reg.register("tidl.bn_relu", "target.tidl")
def _bn_relu_whitelist_fn(attrs, args):
    bn_op = args[0]
    return _batch_norm_whitelist_fn(bn_op.attrs, bn_op.args)

@reg.register("tidl.add_relu", "target.tidl")
def _add_relu_whitelist_fn(attrs, args):
    add_op = args[0]
    return _add_whitelist_fn(add_op.attrs, add_op.args)

@reg.register("tidl.dense_relu", "target.tidl")
def _dense_relu_whitelist_fn(attrs, args):
    dense_op = args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@reg.register("tidl.dense_bias_relu", "target.tidl")
def _dense_bias_relu_whitelist_fn(attrs, args):
    dense_op = args[0].args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@reg.register("tidl.conv2d_bias", "target.tidl")
def _conv2d_bias_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    return _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)

@reg.register("tidl.dense_bias", "target.tidl")
def _dense_bias_relu_whitelist_fn(attrs, args):
    dense_op = args[0]
    return _dense_whitelist_fn(dense_op.attrs, dense_op.args)

@reg.register("tidl.conv2d_pad", "target.tidl")
def _conv2d_pad_whitelist_fn(attrs, args):
    conv2d_op = args[0]
    pad_supported = (float(attrs.pad_value) == 0.0 and attrs.pad_mode == 'constant')
    conv2d_supported = _conv2d_whitelist_fn(conv2d_op.attrs, conv2d_op.args)
    return (pad_supported and conv2d_supported)

@reg.register("add", "target.tidl")
def _add_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.argmax", "target.tidl")
def _argmax_whitelist_fn(attrs, args):
    keepdims = attrs.keepdims
    exclude = attrs.exclude
    axis = attrs.axis
    # is checked_type.shape always guaranteed to be there?
    data = args[0]
    supported = (int(infer_shape(data)[1]) <= 15 and keepdims == 1 and axis == 1 and exclude == 0)
    return supported

@reg.register("nn.avg_pool2d", "target.tidl")
def _avg_pool_whitelist_fn(attrs, args):
    pool_size = get_const_tuple(attrs.pool_size)
    strides = get_const_tuple(attrs.strides)
    supported = (pool_size[0] <= 9 and pool_size[1] <= 9 and strides[0] <= 3 and strides[1] <=2)
    return supported

@reg.register("nn.batch_flatten", "target.tidl")
def _batch_flatten_fn(attrs, args):
    data = args[0]
    if(len(infer_shape(data)) == 4):
        supported = (int(infer_shape(data)[2]) <= 65535 and int(infer_shape(data)[3]) <= 65535)
    else:
        supported = True
    return supported

@reg.register("nn.batch_norm", "target.tidl")
def _batch_norm_whitelist_fn(attrs, args):
    #These are the relay arguments... look up the operator to get the actual name...
    data1 = infer_type(args[1])
    supported = True

    if data1.checked_type.dtype != 'float32':
        supported = False
    elif attrs.axis != 1 and attrs.axis != 3:
        supported = False

    return supported

@reg.register("nn.bias_add", "target.tidl")
def _bias_add_whitelist_fn(attrs, args):
    # Standalone bias_add is not supported.
    return False

@reg.register("clip", "target.tidl")
def _clip_whitelist_fn(attrs, args):
    a_min = attrs.a_min
    a_max = attrs.a_max
    supported = (a_min == 0 and a_max == 6)
    return supported

@reg.register("concatenate", "target.tidl")
def _concatenate_whitelist_fn(attrs, args):
    supported = (attrs.axis == 1) or (attrs.axis == 3)
    return supported

@reg.register("nn.conv2d", "target.tidl")
def _conv2d_whitelist_fn(attrs, args):
    weight = infer_type(args[1])
    if weight.checked_type.dtype != 'float32':
        return False

    weight_shape  = get_const_tuple(infer_shape(weight))
    strides       = get_const_tuple(attrs.strides)
    dilation      = get_const_tuple(attrs.dilation)
    kernel_size   = get_const_tuple(attrs.kernel_size)
    groups        = attrs.groups

    (dh, dw) = dilation
    (kh, kw) = kernel_size
    channel_supported = (weight_shape[0] <= 2048 and weight_shape[1] <= 2048)
    stride_supported  = (strides[0] <= 2 and strides[1] <= 2)
    dilation_supported = (dh == 1 or dh == 2 or dh == 4) and (dw == 1 or dw == 2 or dw == 4)
    kernel_supported = (((kh-1)*dh+1) <= 9) and (((kw-1)*dw+1) <= 9)
    groups_supported = (groups <= 1024)
    supported = channel_supported and stride_supported and dilation_supported \
                and kernel_supported and groups_supported

    return supported

@reg.register("nn.conv2d_transpose", "target.tidl")
def _conv2d_transpose_whitelist_fn(attrs, args):
    weight = args[1]
    weight_shape  = get_const_tuple(infer_shape(weight))
    strides = get_const_tuple(attrs.strides)
    groups = attrs.groups
    supported = (weight_shape[0] == weight_shape[1]) and (weight_shape[0] == groups) and (strides[1] == 2)
    return supported

@reg.register("nn.dense", "target.tidl")
def _dense_whitelist_fn(attrs, args):
    weight = args[1]

    #weight_shape = get_const_tuple(weight.checked_type.shape)
    weight_shape = infer_shape(weight)
    w_in  = weight_shape[1]
    w_out = weight_shape[0]
    supported = (w_in <= 65536) and (w_out <= 16384) and (w_in * w_out <= 67108864)
    return supported

@reg.register("nn.dropout", "target.tidl")
def _dropout_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.global_avg_pool2d", "target.tidl")
def _global_avg_pool_whitelist_fn(attrs, args):
    shape = list(map(int, args[0].checked_type.shape))
    layout = attrs.layout
    if layout == "NCHW":
        height = shape[2]
        width  = shape[3]
    else:
        height = shape[1]
        width  = shape[2]
    supported = height * width <= 4096
    return supported

@reg.register("nn.max_pool2d", "target.tidl")
def _max_pool_whitelist_fn(attrs, args):
    pool_size = get_const_tuple(attrs.pool_size)
    strides   = get_const_tuple(attrs.strides)
    supported = (pool_size[0] <= 9) and (pool_size[1] <= 9) and (strides[0] <= 3) and (strides[1] <= 2)
    return supported

@reg.register("multiply", "target.tidl")
def _multiply_whitelist_fn(attrs, args):
    #supported = True
    supported = False
    return supported

@reg.register("nn.nms", "target.tidl")
def _nms_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.pad", "target.tidl")
def _pad_whitelist_fn(attrs, args):
    # Standalone pad is not supported.
    return False

@reg.register("nn.relu", "target.tidl")
def _relu_whitelist_fn(attrs, args):
    # Standalone relu is not supported.
    return False

@reg.register("nn.slice_like", "target.tidl")
def _slice_like_whitelist_fn(attrs, args):
    #supported = (attrs.axis == 1)
    supported = False
    return supported

@reg.register("nn.softmax", "target.tidl")
def _softmax_whitelist_fn(attrs, args):
    supported = (attrs.axis != 2)
    return supported
