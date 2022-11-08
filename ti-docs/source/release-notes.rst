#############
Release Notes
#############

TIDL_PSDK_8.5
-------------
* Bug fixes in C7x code generation

  * Support non-contiguous access pattern in DMA (CODEGEN-10530)
  * Support strided access pattern in SE (CODEGEN-10531)
  * Support odd number of DMA blocks and iterations (CODEGEN-10555)
  * Simplify resize2d index computation (CODEGEN-10561)
  * Optimize maxpool2d with 1x1 pool size (CODEGEN-10665)

* TIDL offload

  * Rewrite broadcast 1D "add" as "bias_add" and offload to TIDL (CODEGEN-10325)
  * Do not offload resize2d with asymmetric or non-power-of-2 scaling factor (CODEGEN-10561)
  * Enabled TIDL batch processing (CODEGEN-10375)
  * Rewrite conv2d with 1x2 strides as conv2d with 1x1 strides followed by maxpool2d with 1x2 strides (CODEGEN-10662)
  * Offload sigmoid to TIDL (CODEGEN-10568)

* TVM extension

  * Support calling external function with DLTensor args (CODEGEN-10520)
  * Demonstrate overwriting default TVM strategy for an operator (CODEGEN-10520)
  * Demonstrate overwriting default TVM strategy for an operator for a special case (CODEGEN-10520)

TIDL_PSDK_8.4
-------------
* Bug fixes in C7x code generation

  * Fix streaming engine pass for loops containing multiple accesses to same tensor (CODEGEN-9810)
  * Add nop() function to support Reshape op in TVM C runtime (CODEGEN-9795)
  * Avoid overriding generic op strategy in "hls.py" (back-ported from upstream TVM)
  * Add C7x strategy for concatenate instead of using generic strategy (CODEGEN-9794)
  * Do not vectorize inner loop with iterations less than vector length (CODEGEN-9841)
  * Do not vectorize if loop body contains call (CODEGEN-9424)
  * Fix vectorization factor to use largest data type in computation (CODEGEN-9848)
  * Add 64-bit integer support in streaming engine config (CODEGEN-9909)
  * Fix TVM C runtime to support more than 255 functions/layers (CODEGEN-9981)
  * Return tvm runtime create failure to OpenVX node (CODEGEN-8941)
  * Apply tiling and dma schedule only to broadcast ops in injective.py (CODEGEN-10004)

* J721S2 support
* Merge with upstream neo-ai tvm 1.11.2
* Tensor debug support for TVM Arm runtime and TVM C7x runtime

TIDL_PSDK_8.2
-------------
* Initial release of C7x code generation support
* Merge with neo-ai-tvm 1.10.0
