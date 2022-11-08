.. _ti-tvm-home:

###########################
TI TVM User's Guide - 8.5.x
###########################

Texas Instrument's fork of Tensor Virtual Machine (TVM) enables support for the TDA4 family of processors. These processors use C7x DSP
and Matrix Multiplication Accelerator (MMA) to accelerate the inference of deep learning models. For additional information on TDA4x processors,
and TI's Edge AI ecosystem, refer to the `Edge AI page on ti.com <https://www.ti.com/technologies/edge-ai.html>`_. 

.. figure:: images/TDA4.png
  :scale: 30
  :align: center

The TI Deep Learning library (TIDL) contains highly optimized implementations of common layers on C7x/MMA. 
If a model contains layers that are not implemented by TIDL, TVM can be used to run these layers on Arm (Cortex-A) cores 
or the C7x DSP core.

This user's guide documents the TI TVM Compiler and its usage.

.. toctree::
    :maxdepth: 1
    :hidden:

    getting-started/index
    compiling
    infering
    extending
    developing
    building
    release-notes
    support
    Important Notice <notice>

.. extending

.. raw:: html

    For offline use, a PDF version of the guide is available here: <a href="https://software-dl.ti.com/codegen/docs/tvm/users_guide/latex/TI_TVM_User_Guide.pdf">TI TVM User's Guide</a><br>


.. |(R)| unicode:: U+00AE
    :ltrim:
