.. _ti-tvm-home:

#############################
TI TVM User's Guide - PSDK8.4
#############################

Texas Instrument's fork of TVM enables support for the TDA4 family of processors. These processors use C7x DSP
and Matrix Multiplication Accelerator (MMA) to accelerate the inference of machine learning models.

The TI Deep Learning library (TIDL) contains highly optimized implementations of common layers on C7x/MMA.
To improve model coverage, TVM is used to run layers unsupported by TIDL on Arm (Cortex-A) cores. Using TVM's
unique code generation capability, unsupported layers are also run on the C7x. The TVM BYOC infrastructure is
used to partition & offload subgraphs to TIDL. Custom schedules are used to generate performant C/C++ code for C7x.
The generated code utilizes hardware features such as DMA, Streaming Engine, vector execution and C7x intrinsics.
The TVM C runtime is ported to run on C7x and handles both TIDL subgraphs and TIDL-unsupported layers.  An Arm
wrapper enables users to use unmodified TVM APIs to launch inference from Arm.

This user's guide documents the TI TVM Machine Learning Compiler and its usage.

.. toctree::
    :maxdepth: 1
    :hidden:

    getting-started
    extending
    building
    release-notes
    support
    Important Notice <notice>


.. raw:: html

    For offline use, a PDF version of the guide is available here: <a href="https://software-dl.ti.com/codegen/docs/tiarmclang/compiler_tools_user_guide/latex/TI_Arm_Clang_Compiler_Tools_User_Guide.pdf">TI TVM User's Guide</a><br>


.. |(R)| unicode:: U+00AE
    :ltrim:
