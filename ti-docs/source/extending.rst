.. _ti-tvm-extending:

=============
Extending TVM
=============

This section explains how to extend TVM to help improve the inference performance of
a model.  These extensions should not require a rebuild of TVM.


Transforming Relay IR
=====================

After a model is converted into TVM's Relay IR format, it is possible to rewrite Relay IR
into another equivalent Relay IR for various reasons such as: simplifying arithmetic,
removing identity ops, or maximizing TIDL offload.
The ``python/tvm/relay/backend/contrib/tidl/prepare.py`` script contains the transformations 
run before TIDL partitioning and C7x code generation.  These transformations are Python based,
straightforward to understand, and easy to write.

The following example shows how transforming Relay IR can help maximize TIDL offload.
The example is available in the ``tests/python/relay/ti_tests/unit_tests/conv2d_1x2_stride.py`` script.
The input for this example has a convolution layer with 3x3 kernel size and a 1x2 stride, as shown in the below Relay IR.

.. code:: text

  %0 = nn.conv2d(%i0, %w1, strides=[1, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3]);

However, TIDL does not support such a kernel size and stride combination.  TIDL does support
the combination of 3x3 kernel size and 1x1 stride.  Using the ``ConvertConvStride`` transformation,
you can rewrite the original convolution as a convolution with 3x3 kernel size and 1x1 stride,
followed by maxpooling with 1x1 pool size and 1x2 stride, as shown in the below Relay IR.

.. code:: text

  %0 = nn.conv2d(%i0, meta[relay.Constant][0] /* ty=Tensor[(16, 8, 3, 3), float32] */, 
       Tensor[(1, 8, 224, 224), float32], Tensor[(16, 8, 3, 3), float32], 
       padding=[1, 1, 1, 1], kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 224, 224), float32] */;
  %1 = nn.max_pool2d(%0, Tensor[(1, 16, 224, 224), float32], pool_size=[1, 1], strides=[1, 2],
       padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 16, 224, 112), float32] */;

After the transformation, the convolution layer can be offloaded to TIDL and can benefit
from the performance boost offered by TIDL on C7x/MMA.  The max pooling layer with a 1x1 pool size and
1x2 stride is still not supported by TIDL, but the computation is much simpler and we can use
TVM C7x code generation to generate code for this layer.  Although this transformation doubles
the amount of computation from what was originally required, the overall performance still is better than running
the original convolution layer on Arm or C7x.


Customizing Compute and Schedule
================================

TVM uses strategies to turn each operator into code.  An operator can have many strategies
associated with it.  A strategy consists of two parts, the ``compute`` and the ``schedule``.

* The :term:`compute` specifies what this operator does and is defined as a math formula.
* The :term:`schedule` specifies how to realize the computation via loop nests and data movement. 

For each operator, you can customize the compute, the schedule, or both in order to generate better performing code on C7x.  The following subsection provide some examples.

Customizing compute
-------------------

The ``python/tvm/topi/c7x/resize.py`` script makes a copy of the default
``tvm.relay.op.image._image.compute_resize2d`` and simplifies the index computation so that
the generated code can be software pipelined on C7x devices.

The ``python/tvm/relay/op/strategy/c7x.py`` script overrides the default strategy for resize2d
on C7x with the updated compute and schedule.

Customizing schedule
--------------------

The ``python/tvm/topi/c7x/injective.py`` script uses a customized schedule for C7x injective
operators.  It transforms loop nests, applies DMA with double buffering, and vectorizes the
innermost loop for C7x.

Customizing compute and schedule only for a special case
--------------------------------------------------------

Significantly simplified computation is possible if *all* of the following are true:

* Pool size is 1x1.
* Data layout is "NCHW".
* Dilation is 1x1.
* No padding is used.

The ``python/tvm/relay/op/strategy/c7x.py`` script customizes the strategy for ``max_pool2d``
only in this special case. The simplified computation is defined in the ``compute_max_pool2d_1x1_pool_size`` function in the ``python/tvm/topi/c7x/pooling.py``, which treats the operator as an injective operator to use the C7x injective schedule.

For all other cases, the default strategy is used.

Customizing compute and schedule in compilation script
-----------------------------------------------------------

It is also possible to customize the strategy for a relay operator in your compilation script.
The ``tests/python/relay/ti_tests/unit_tests/resize_nchw_1x2.py`` script shows how to overwrite
the default resize2d strategy for C7x in the ``add_c7x_resize_strategy`` function.


Customizing a relay operator to call into an external library
-------------------------------------------------------------

If the resize2d operator has a 1x2 upscaling factor on NCHW float data, the same script used in the previous section, ``tests/python/relay/ti_tests/unit_tests/resize_nchw_1x2.py`` customizes the compute
to call an external function, ``resize_nchw_1x2``, in an external library.

The ``resize_nchw_1x2.cpp`` C++ file contains the function definition.  Tensors are passed in a ``DLTensor``
data structure from TVM runtime to the C/C++ function. The definition of ``DLTensor`` can be
found in the
`3rdparty/dlpack/include/dlpack/dlpack.h <https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h>`_ header file.

The ``V1`` version of the code shows a naive implementation of resize2d with a 1x2 upscaling factor. The ``V2`` version of the code optimizes the implementation using the C7x streaming engine (SE) feature.  

The C++ file is compiled into a library and linked into
the C7x deployable module for the model.
As shown in the example, ``build_and_set_ext_lib`` function in the ``unit_utils.py`` script,
the ``CGT7X_EXT_LIBS`` environment variable specifies additional libraries that
can be linked into the C7x deployable module. For example:

.. code:: text

  CGT7X_EXT_LIBS="-l /path/to/lib1 -l /path/to/lib2"

.. note::

    ``resize_nchw_1x2.cpp`` is not a generic implementation for all resize2d cases with an upscaling
    factor 1x2. This file handles only float data in "NCHW" layout.
