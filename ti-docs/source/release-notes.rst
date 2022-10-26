#############
Release Notes
#############


TIDL_PSDK_8.4
---------------
- Including bug fixes in TI.8.2.2
- J721S2 support
- Merge with upstream neo-ai tvm 1.11.2
- Tensor debug support for TVM Arm runtime and TVM C7x runtime

TI.8.2.2
--------
Bug fixes in C7x code generation

- Fix streaming engine pass for loops containing multiple accesses to same tensor (CODEGEN-9810)
- Add nop() function to support Reshape op in TVM C runtime (CODEGEN-9795)
- Avoid overriding generic op strategy in "hls.py" (back-ported from upstream TVM)
- Add C7x strategy for concatenate instead of using generic strategy (CODEGEN-9794)
- Do not vectorize inner loop with iterations less than vector length (CODEGEN-9841)
- Do not vectorize if loop body contains call (CODEGEN-9424)
- Fix vectorization factor to use largest data type in computation (CODEGEN-9848)
- Add 64-bit integer support in streaming engine config (CODEGEN-9909)
- Fix TVM C runtime to support more than 255 functions/layers (CODEGEN-9981)
- Return tvm runtime create failure to OpenVX node (CODEGEN-8941)
- Apply tiling and dma schedule only to broadcast ops in injective.py (CODEGEN-10004)

TI.8.2.0, TI.8.2.1, TIDL_PSDK_8.2
-----------------------------------
- Initial release of C7x code generation support
- Merge with neo-ai-tvm 1.10.0
