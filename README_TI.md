<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://software-dl.ti.com/codegen/docs/tvm/index.html) |
[Release Notes](https://software-dl.ti.com/codegen/docs/tvm/release-notes.html)

Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

Neo-AI/TVM is a downstream branch of TVM that includes vendor- and product-specific features on top of the upstream codebase.

TexasInstruments/tvm is a fork of Neo-AI/TVM.


Branches
--------
  * tidl-j7 
  
 This release branch supports the Texas Instruments (TI) TDA4 family of processors. These processors use C7x DSP and Matrix Multiplication Accelerator (MMA) to accelerate the inference of machine learning models. 
 
 The TI Deep Learning library (TIDL) contains highly optimized implementations of common layers on C7x/MMA. To improve model coverage, TVM is used to run layers unsupported by TIDL on Arm (Cortex-A) cores. Using TVM's unique code generation capability, unsupported layers are also run on the C7x. The TVM BYOC infrastructure is used to partition & offload subgraphs to TIDL. 
 Custom schedules are used to generate performant C/C++ code for C7x. The generated code utilizes hardware features such as DMA, Streaming Engine, vector execution and C7x intrinsics. The TVM C runtime is ported to run on C7x and handles both TIDL subgraphs and TIDL-unsupported layers.  An Arm wrapper enables users to use unmodified TVM APIs to launch inference from Arm.

TI's TVM releases are synchronized with TI's [Processor SDK RTOS](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E) releases. The following table lists the PSDK RTOS release and the corresponding TVM tag that is compatible with it.

| PSDK release | TVM release tag      | Key features                                              |
|--------------|----------------------|-----------------------------------------------------------|
| 8.4          | TIDL\_PSDK\_8.4      | Bug fixes, J721S2 support, debug support                  |
| 8.2          | TIDL\_PSDK\_8.2, TI.8.2.{0, 1, 2} | C7x code generation support                  |
| 8.1          | TIDL\_PSDK\_8.1      |                                                           |
| 8.0          | TIDL\_PSDK\_8.0      |                                                           |
| 7.3          | TIDL\_PSDK\_7.3      | TIDL offload, unsupported layers run on Arm               |

  Suffix "RC" stands for release candidates, suffix "UPDATE" stands for updates
that are still compatible with certain releases.

License
-------
TVM is licensed under the [Apache-2.0](LICENSE) license.

Getting Started
---------------
Refer to [Building from Source](https://software-dl.ti.com/codegen/docs/tvm/building.html) in the TI TVM User's Guide.

Release Notes
-------------
Refer to [Release Notes](https://software-dl.ti.com/codegen/docs/tvm/release-notes.html) in the TI TVM User's Guide.