.. _ti-tvm-home:

###########################
TI TVM User's Guide - 11.2
###########################

Texas Instrument's fork of the Apache Tensor Virtual Machine (:term:`TVM`) enables support for the Jacinto/Sitara family of processors. 
These processors use C7x DSPs and Matrix Multiplication Accelerators (:term:`MMA`) to accelerate :term:`inference`-making by machine learning models. 
For additional information about TDA4x processors and TI's Edge AI ecosystem, refer to the `Edge AI page on ti.com <https://www.ti.com/technologies/edge-ai.html>`_. 

TVM is a complete compilation and inference runtime solution for Deep Neural Networks deployment on embedded devices. 
TI's fork of Apache TVM integrates TI's highly optimized TI Deep Learning library (:term:`TIDL`) as an accelerator to the TVM framework to accelerate commonly used layers on c7x/MMA.
It enables complete model inference on c7x DSP core even in the case of TIDL unsupported operators by leveraging code generation capability of TVM to utilize relevant TI platform hardware features.

This user's guide documents the TI TVM Compiler and its usage.

.. toctree::
    :maxdepth: 1
    :hidden:
    :numbered:

    getting-started/index
    compiling
    infering
    developing
    building
    additional-docs
    glossary
    release-notes
    support
    Important Notice <notice>
