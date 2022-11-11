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
"""TI C7x TVM operator input resize compute."""
"""Copied and adapted from image/resize.py.  Simpilified index computation
       for better performance on C7x.
   - main scope is in resize2d (resize1d, resize3d removed)
   - get rid of floorf() in index computation

   Entry function, compute_resize2d, is at the end of this file.
"""
#from __future__ import absolute_import
import tvm
from tvm import te
from tvm.topi.utils import nchw_pack_layout, nchw_xc_layout
from .. import tag


def get_2d_indices(indices, layout="NCHW"):
    """Get 2d indices"""
    (cc, inum, ic) = (0, 0, 0)
    if layout == "NHWC":
        n, y, x, c = indices
        cc = None
    elif layout == "NCHW":
        n, c, y, x = indices
        cc = None
    elif nchw_pack_layout(layout):
        n, c, y, x, inum, ic = indices
    else:
        # else must be NCHWxc
        assert nchw_xc_layout(layout)
        n, c, y, x, cc = indices

    return n, c, y, x, cc, inum, ic


def get_2d_pixel(data, layout, image_height, image_width, n, c, y, x, cc, ib, ic):
    """Get 2d pixel"""
    y = tvm.te.max(tvm.te.min(y, image_height - 1), 0)
    x = tvm.te.max(tvm.te.min(x, image_width - 1), 0)
    if layout == "NHWC":
        return data(n, y, x, c).astype("float")
    if layout == "NCHW":
        return data(n, c, y, x).astype("float")
    if nchw_pack_layout(layout):
        return data(n, c, y, x, ib, ic).astype("float")

    # else must be NCHWxc
    assert nchw_xc_layout(layout)
    return data(n, c, y, x, cc).astype("float")


def get_inx(x, image_width, target_width, coordinate_transformation_mode, start_x=0, end_x=-1):
    """Infer input x from output x with various coordinate transformation methods"""
    scale_x = te.div(image_width.astype("float"), target_width.astype("float"))
    if coordinate_transformation_mode == "half_pixel":
        #BeginTI: simplify index computation ended up in C7x code, e.g.
        #   placeholder[((((i1 * 51200) + (max(min(((int)floorf((float)((((float)i2) + 5.000000e-01f) - 5.000000e-01f))), 63), 0) * 800)) + max(min(((int)floorf((float)(((((float)i3) + 5.000000e-01f) * 5.000000e-01f) - 5.000000e-01f))), 799), 0)))]
        #     can be simplified to
        #   placeholder[((((i1 * 51200) + (i2 * 800)) + max(min(((int)floorf((float)((((float)i3) * 5.000000e-01f) + -2.500000e-01f))), 799), 0)))]
        #in_x = (x + 0.5) * scale_x - 0.5
        if image_width == target_width:
            in_x = x
        else:
            in_x = x * scale_x + (0.5 * scale_x - 0.5)
        #EndTI
    elif coordinate_transformation_mode == "align_corners":
        in_x = (image_width - 1).astype("float") / (target_width - 1) * x
    elif coordinate_transformation_mode == "asymmetric":
        in_x = scale_x * x
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        in_x = te.if_then_else(target_width > 1, (x + 0.5) * scale_x - 0.5, 0.0)
    elif coordinate_transformation_mode == "tf_half_pixel_for_nn":
        in_x = (x + 0.5) * scale_x
    elif coordinate_transformation_mode == "tf_crop_and_resize":
        in_x = te.if_then_else(
            target_width > 1,
            start_x * (image_width - 1)
            + x * (end_x - start_x) * (image_width - 1).astype("float") / (target_width - 1),
            0.5 * (start_x + end_x) * (image_width - 1),
        )
    else:
        raise ValueError(
            "Unsupported coordinate_transformation_mode: {}".format(coordinate_transformation_mode)
        )
    return in_x


def get_closest_index(in_x, rounding_method, boxes):
    """get the closest index to a value based on a certain rounding method"""
    if rounding_method == "round" or boxes is not None:
        closest_x_index = te.round(in_x).astype("int32")
    elif rounding_method == "round_prefer_floor":
        closest_x_index = te.ceil(in_x - 0.5).astype("int32")
    elif rounding_method == "round_prefer_ceil":
        closest_x_index = te.floor(in_x + 0.5).astype("int32")
    elif rounding_method == "floor":
        #BeginTI: no need for c7x
        ## Add epsilon to floor to prevent gpu rounding errors.
        #epsilon = 1e-5
        #closest_x_index = te.floor(in_x + epsilon).astype("int32")
        closest_x_index = te.floor(in_x).astype("int32")
        #EndTI
    elif rounding_method == "ceil":
        #BeginTI: no need for c7x
        ## Subract epsilon from ceil to prevent gpu rounding errors.
        #epsilon = 1e-5
        #closest_x_index = te.ceil(in_x - epsilon).astype("int32")
        closest_x_index = te.ceil(in_x).astype("int32")
        #EndTI
    else:
        raise ValueError("Uknown rounding method: {}".format(rounding_method))
    return closest_x_index


def _lerp(A, B, t):
    """Perform Linear interpolation in 1D"""
    return A * (1.0 - t) + B * t


def _cubic_spline_weights(t, alpha):
    """create cubic spline weights in 1D"""
    t2 = t * t
    t3 = t * t * t
    w1 = alpha * (t3 - 2 * t2 + t)
    w2 = (alpha + 2) * t3 - (3 + alpha) * t2 + 1
    w3 = -(alpha + 2) * t3 + (3 + 2 * alpha) * t2 - alpha * t
    w4 = -alpha * t3 + alpha * t2
    return [w1, w2, w3, w4]


def _cubic_kernel(inputs, w):
    """perform cubic interpolation in 1D"""
    return sum([a_i * w_i for a_i, w_i in zip(inputs, w)])


def _resize_2d(
    indices,
    data,
    roi,
    image_height,
    image_width,
    target_height,
    target_width,
    boxes=None,
    box_indices=None,
    method=None,
    extrapolation_value=0.0,
    layout="NCHW",
    coordinate_transformation_mode="align_corners",
    rounding_method="",
    alpha=-0.5,
    exclude_outside=0,
    out_dtype=None,
):

    """Perform resize operation on the data with selected method and options.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    image_height : integer
        Input image height

    image_width : integer
        Input image width

    target_height : integer
        The target resized image height

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    alpha: float, optional
        Bicubic spline coefficient

    exclude_outside: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    n, c, y, x, cc, inum, ic = get_2d_indices(indices, layout)
    box_idx = box_indices(n) if box_indices is not None else n
    if boxes is not None:
        y1, x1 = boxes(n, 0), boxes(n, 1)
        y2, x2 = boxes(n, 2), boxes(n, 3)

        in_h = (image_height - 1) * (y2 - y1)
        in_w = (image_width - 1) * (x2 - x1)
        h_scale = in_h.astype("float") / (target_height - 1)
        w_scale = in_w.astype("float") / (target_width - 1)

        in_y = y1 * (image_height - 1) + h_scale * y
        in_x = x1 * (image_width - 1) + w_scale * x
    else:
        in_x = get_inx(x, image_width, target_width, coordinate_transformation_mode, roi[1], roi[3])
        in_y = get_inx(
            y, image_height, target_height, coordinate_transformation_mode, roi[0], roi[2]
        )

    if method == "nearest_neighbor":
        if rounding_method == "":
            if coordinate_transformation_mode == "align_corners":
                rounding_method = "round"
            else:
                rounding_method = "floor"

        closest_x_index = get_closest_index(in_x, rounding_method, boxes)
        closest_y_index = get_closest_index(in_y, rounding_method, boxes)

        value = get_2d_pixel(
            data,
            layout,
            image_height,
            image_width,
            box_idx,
            c,
            closest_y_index,
            closest_x_index,
            cc,
            inum,
            ic,
        )
    elif method == "linear":
        #BeginTI: Remove floorf() call that prevents C7x software pipelining.
        #         Because the final index is non-negative (>=0),
        #         (int)floor(x) can be simplified to (int)(x).  E.g.
        #  placeholder[((((i1 * 51200) + (i2 * 800)) + max(min(((int)floorf((float)((((float)i3) * 5.000000e-01f) + -2.500000e-01f))), 799), 0)))]
        #    can be simplified to
        #  placeholder[((((i1 * 51200) + (i2 * 800)) + max(min(((int)((((float)i3) * 5.000000e-01f) + -2.500000e-01f)), 799), 0)))]
        #y_int = te.floor(in_y).astype("int32")
        #x_int = te.floor(in_x).astype("int32")
        #y_lerp = in_y - y_int
        #x_lerp = in_x - x_int
        y_ints = [in_y.astype("int32"), (in_y+1).astype("int32")]
        x_ints = [in_x.astype("int32"), (in_x+1).astype("int32")]
        y_lerp = in_y - y_ints[0]
        x_lerp = in_x - x_ints[0]
        #EndTI


        p = [[0 for i in range(2)] for j in range(2)]
        for j in range(2):
            for i in range(2):
                p[j][i] = get_2d_pixel(
                    data,
                    layout,
                    image_height,
                    image_width,
                    box_idx,
                    c,
                    #BeginTI
                    #y_int + j,
                    #x_int + i,
                    y_ints[j],
                    x_ints[i],
                    #EndTI
                    cc,
                    inum,
                    ic,
                )

        top = _lerp(*p[0], x_lerp)
        bottom = _lerp(*p[1], x_lerp)
        value = _lerp(top, bottom, y_lerp)

    elif method == "cubic":
        xint = te.floor(in_x).astype("int32")
        xfract = in_x - te.floor(in_x)

        yint = te.floor(in_y).astype("int32")
        yfract = in_y - te.floor(in_y)

        # Get the surrounding values
        p = [[0 for i in range(4)] for j in range(4)]
        for j in range(4):
            for i in range(4):
                p[j][i] = get_2d_pixel(
                    data,
                    layout,
                    image_height,
                    image_width,
                    box_idx,
                    c,
                    yint + j - 1,
                    xint + i - 1,
                    cc,
                    inum,
                    ic,
                )

        wx = _cubic_spline_weights(xfract, alpha)
        wy = _cubic_spline_weights(yfract, alpha)
        if exclude_outside:
            for i in range(4):
                wx[i] = te.if_then_else(
                    te.any(xint - 1 + i < 0, xint + i > image_width), 0.0, wx[i]
                )
                wy[i] = te.if_then_else(
                    te.any(yint - 1 + i < 0, yint + i > image_height), 0.0, wy[i]
                )
            sum_wx = sum(wx)
            sum_wy = sum(wy)
            wx = [w / sum_wx for w in wx]
            wy = [w / sum_wy for w in wy]
        col0 = _cubic_kernel(p[0], wx)
        col1 = _cubic_kernel(p[1], wx)
        col2 = _cubic_kernel(p[2], wx)
        col3 = _cubic_kernel(p[3], wx)
        value = _cubic_kernel([col0, col1, col2, col3], wy)

    else:
        raise ValueError("Unknown resize method:", method)

    if coordinate_transformation_mode == "tf_crop_and_resize":
        out = tvm.tir.if_then_else(
            in_y < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_y > image_height - 1, extrapolation_value, value),
        )
        # use extrapolation_value if in_x is out of boundary
        value = tvm.tir.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm.tir.if_then_else(in_x > image_width - 1, extrapolation_value, out),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize2d(
    data,
    roi,
    size,
    layout="NCHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    bicubic_alpha=-0.5,
    bicubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
    output_shape=None,
):
    """Perform resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method:
        Method for rounding coordinate locations

    bicubic_alpha: float, optional
        Bicubic spline coefficient

    bicubic_exclude: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    output_shape: tvm.tir.container.Array, optional
        Shape to return. If left None will be inferred
        (If shape is determined dynamically, pass out_dtype.shape as output_shape)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
        or 5-D with shape [batch, channel-major, in_height*scale, in_width*scale, channel-minor]
    """
    method = method.lower()
    if layout == "NHWC":
        in_n, in_h, in_w, in_c = data.shape
        if output_shape is None:
            output_shape = [in_n, size[0], size[1], in_c]
    elif layout == "NCHW":
        in_n, in_c, in_h, in_w = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1]]
    elif nchw_pack_layout(layout):  # for NCHWinic
        in_n, in_c, in_h, in_w, in_inum, in_ic = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1], in_inum, in_ic]
    elif nchw_xc_layout(layout):  # for NCHWxc
        in_n, in_c, in_h, in_w, in_cc = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1], in_cc]
    else:
        raise ValueError("%s layout is not supported." % layout)

    if isinstance(size, tuple):
        size = list(size)

    for i in range(2):
        if isinstance(size[i], int):
            size[i] = tvm.tir.IntImm("int32", size[i])

    def compute_func(*indices):
        return _resize_2d(
            indices,
            data,
            roi,
            in_h,
            in_w,
            size[0],
            size[1],
            method=method,
            layout=layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            alpha=bicubic_alpha,
            exclude_outside=bicubic_exclude,
            extrapolation_value=extrapolation_value,
            out_dtype=out_dtype,
        )

    return te.compute(output_shape, compute_func, name="resize", tag=tag.INJECTIVE)


# Adapted from tvm.relay.op.image._image.compute_resize2d
#   to call resize2d defined in this file instead of topi.image.resize2d
def compute_resize2d(attrs, inputs, out_type):
    """compute definition for resize2d op"""
    size = attrs.size
    roi = attrs.roi
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    extrapolation_value = attrs.extrapolation_value
    out_dtype = attrs.out_dtype
    return [
        #topi.image.resize2d(
        resize2d(
            inputs[0],
            roi,
            size,
            layout,
            method,
            coord_trans,
            rounding_method,
            cubic_alpha,
            cubic_exclude,
            extrapolation_value,
            out_dtype,
        )
    ]

