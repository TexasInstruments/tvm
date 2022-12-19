.. _ti-tvm-gs-inference:

#################
Running Inference
#################

TVM inference can be run using a Python script or a C/C++ application. Examples
are provided in the `TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples>`_
and `TI TVM fork <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_
repositories on Github. You can use these examples as templates to modify for your own use cases.

The default runtime setup used by edgeai-tidl is `DLR (Deep Learning Runtime from Amazon AWS) <https://github.com/TexasInstruments/neo-ai-dlr/tree/tidl-j7>`_.
For the purpose of running TVM compiled models with the DLR runtime, DLR is simply a wrapper
around the TVM runtime.

See :ref:`ti-tvm-infering` for further details about TVM inference.

The following Python functions and C++ code are used in the examples provided in the TI TVM fork.

.. _ti-tvm-gs-inference-python:

Python
------
The ``run_model`` function shows how to run inference with the DLR or the TVM Runtime.

.. autofunction:: relay.ti_tests.infer_model.run_model

.. _ti-tvm-gs-inference-cpp:

C++
---

:numref:`cpp-inference` shows how to use the C++ and TVM Runtime APIs to:

* Load compilation artifacts (shared library, parameter file, and JSON representation of the network graph)
* Create a TVM Graph Executor
* Set up inputs to the Graph Executor
* Run inference on the Graph Executor
* Extract outputs from the Graph Executor

.. literalinclude:: ../../../tests/python/relay/ti_tests/relay_mul/relay_mul.cc
  :language: c++
  :lines: 37-103
  :caption: Running inference using C++ and the TVM Runtime
  :name: cpp-inference
  :linenos:

.. todo::
    Add references to Edge AI SDK docs on additional examples and GStreamer integration.
