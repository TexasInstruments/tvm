=============
Extending TVM
=============

In this section, we talk how to extend TVM to help improve the inference performance of
a model.  These extensions should not require a rebuild of TVM.


Transforming Relay IR
=====================

After the model is converted into TVM Relay IR, it is possible to rewrite Relay IR
into another equivalent Relay IR for various reasons: arithmetic simplification,
removing identity ops, and maximizing TIDL offload.
``python/tvm/relay/backend/contrib/tidl/prepare.py`` has the transformations that
we run before TIDL partitioning and c7x code generation.  They are python based,
straightforward to understand and easy to write.

Here we use one example to illustrate how transforming Relay IR can help maximize TIDL offload.
The example is available in ``tests/python/relay/ti_tests/unit_tests/conv2d_1x2_stride.py``.
We have a convolution layer with 3x3 kernel size and 1x2 stride, as shown in the below Relay IR.

.. code:: text

  %0 = nn.conv2d(%i0, %w1, strides=[1, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3]);

However, TIDL does not support such a kernel size and stride combination.  TIDL does support
the combination of 3x3 kernel size and 1x1 stride.  Using ``ConvertConvStride`` transformation,
we rewrite the original convolution as convolution with 3x3 kernel size and 1x1 stride,
followed by maxpooling with 1x1 pool size and 1x2 stride, as shown in the below Relay IR.

.. code:: text

  %0 = nn.conv2d(%i0, meta[relay.Constant][0] /* ty=Tensor[(16, 8, 3, 3), float32] */, Tensor[(1, 8, 224, 224), float32], Tensor[(16, 8, 3, 3), float32], padding=[1, 1, 1, 1], kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 224, 224), float32] */;
  %1 = nn.max_pool2d(%0, Tensor[(1, 16, 224, 224), float32], pool_size=[1, 1], strides=[1, 2], padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 16, 224, 112), float32] */;

After the transformation, the convolution layer can now be offloaded to TIDL and benefit
from performance boost offered by TIDL on C7x/MMA.  The max pooling layer with 1x1 pool size and
1x2 stride is still not supported by TIDL, but the computation is much simpler and we can use
TVM C7x code generation to generate code for this layer.  Although this transformation doubles
the amount of computation than originally required, the overall performance still beat running
the original conv2d on Arm or C7x.


Customizing compute and schedule
================================

TVM uses strategies to turn each operator into code.  An operator can have many strategies
associated with it.  A strategy consists of two parts, ``compute`` and ``schedule``.
``Compute`` specifies the what this operator does, e.g. defined as a math formula.
``Schedule`` specifies how to realize the computation via loop nests and data movement.
It is possible to customize the compute or the schedule or both for an operator in order
to generate more performant code on C7x.  The following are some examples.

Customizing compute
-------------------

In ``python/tvm/topi/c7x/resize.py``, we make a copy of the default
``tvm.relay.op.image._image.compute_resize2d``, and simplify the index computation so that
the generated code can be software pipelined on C7x.

In ``python/tvm/relay/op/strategy/c7x.py``, we override the default strategy for resize2d
on c7x with the updated compute and schedule.

Customizing schedule
--------------------

In ``python/tvm/topi/c7x/injective.py``, we develop customized schedule for c7x injective
operators.  We transform loop nests, apply DMA with double buffering, and vectorize the
innermost loop for C7x.

Customizing compute and schedule only for a special case
--------------------------------------------------------

In ``python/tvm/relay/op/strategy/c7x.py``, we customize the strategy for ``max_pool2d``
only when the pool size is 1x1, data layout is "NCHW", dilation is 1x1 and no padding.
For this special case, we can have much simplified computation defined in
``python/tvm/topi/c7x/pooling.py, compute_max_pool2d_1x1_pool_size``,
and we can treat the operator as injective operator to use the c7x injective schedule.
For all other cases, we follow the default strategy.

Customizing compute and schedule in user compilation script
-----------------------------------------------------------

It is also possible to customize strategy for a relay operator in user compilation script.
In ``tests/python/relay/ti_tests/unit_tests/resize_nchw_1x2.py``, we show how to overwrite
the default resize2d strategy for C7x in user script, with ``add_c7x_resize_strategy``.


Customizing a relay op to call into external library
====================================================

In the same example ``tests/python/relay/ti_tests/unit_tests/resize_nchw_1x2.py``,
when resize2d op has 1x2 upscaling factor on NCHW float data, we customize the compute
to be calling an external function, ``resize_nchw_1x2``, in an external library.

``resize_nchw_1x2.cpp`` has the function definition.  Tensors are passed in ``DLTensor``
data structure from TVM runtime to the C/C++ function.  Definition of ``DLTensor`` can be
found in
`3rdparty/dlpack/include/dlpack/dlpack.h <https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h>`_.
We show a naive implementation of resize2d with 1x2 upscaling factor in ``V1`` version
of the code.  Then we optimize the implementation with streaming engine (SE) feature on C7x
in ``V2`` version of the code.  The cpp file is compiled into a library and linked into
the C7x deployable module for the model.
As shown in the example, ``unit_utils.py, build_and_set_ext_lib``,
environment variable ``CGT7X_EXT_LIBS`` is used to specify the additional libraries that
can be linked into the C7x deployable module.  E.g.

.. code:: text

  CGT7X_EXT_LIBS="-l /path/to/lib1 -l /path/to/lib2"

.. note::

    ``resize_nchw_1x2.cpp`` is not a generic implementation for all resize2d cases with upscaling
    factor 1x2.  E.g. it only handles float data in "NCHW" layout.
