.. _ti-tvm-rlsn:

#############
Release Notes
#############

TIDL_PSDK_8.6
-------------
* New devices

  * Support AM62A - To compile model for AM62A, set ``platform`` argument to "AM62A", see :ref:`ti-tvm-gs-compilation` for details.

* Bug fixes in C7x code generation

  * Fix DMAPass src on/off chip check
  * Fix scatter_nd code generation

TIDL_PSDK_8.5
-------------
* Bug fixes in C7x code generation

  * Support non-contiguous access pattern in DMA
  * Support strided access pattern in SE
  * Support odd number of DMA blocks and iterations
  * Simplify resize2d index computation
  * Optimize maxpool2d with 1x1 pool size

* TIDL offload

  * Rewrite broadcast 1D "add" as "bias_add" and offload to TIDL
  * Do not offload resize2d with asymmetric or non-power-of-2 scaling factor
  * Enabled TIDL batch processing
  * Rewrite conv2d with 1x2 strides as conv2d with 1x1 strides followed by maxpool2d with 1x2 strides
  * Offload sigmoid to TIDL

* TVM extension

  * Support calling external function with DLTensor args
  * Demonstrate overwriting default TVM strategy for an operator
  * Demonstrate overwriting default TVM strategy for an operator for a special case

TIDL_PSDK_8.4
-------------
* Bug fixes in C7x code generation

  * Fix streaming engine pass for loops containing multiple accesses to same tensor
  * Add nop() function to support Reshape op in TVM C runtime
  * Avoid overriding generic op strategy in "hls.py" (back-ported from upstream TVM)
  * Add C7x strategy for concatenate instead of using generic strategy
  * Do not vectorize inner loop with iterations less than vector length
  * Do not vectorize if loop body contains call
  * Fix vectorization factor to use largest data type in computation
  * Add 64-bit integer support in streaming engine config
  * Fix TVM C runtime to support more than 255 functions/layers
  * Return tvm runtime create failure to OpenVX node
  * Apply tiling and dma schedule only to broadcast ops in injective.py

* J721S2 support
* Merge with upstream neo-ai tvm 1.11.2
* Tensor debug support for TVM Arm runtime and TVM C7x runtime

TIDL_PSDK_8.2
-------------
* Initial release of C7x code generation support
* Merge with neo-ai-tvm 1.10.0
