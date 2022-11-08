#################
Running Inference
#################

TVM inference can be done using a python script or a C/C++ application.  We have examples
in `TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples>`_
and `TI TVM fork <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_
on Github.  Users can take these examples as template and modify for their own use case scenarios.

The default runtime set up by edgeai-tidl is `DLR (Deep Learning Runtime from Amazon AWS) <https://github.com/TexasInstruments/neo-ai-dlr/tree/tidl-j7>`_.
For the purpose of running TVM compiled models with the DLR runtime, DLR is simply a wrapper
around the TVM runtime.

The following python functions and C++ code are used in the examples we have in the TI TVM fork.

Python
------
The ``run_model`` function illustrates running inference with the DLR or the TVM Runtime.

.. autofunction:: relay.ti_tests.infer_model.run_model

C++
---

:numref:`cpp-inference` illustrates using C++ and TVM Runtime APIs to:

* Load compilation artifacts (shared library, parameter file and JSON representation of the network graph)
* Create a TVM Graph Executor
* Setup inputs to the Graph Executor
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
