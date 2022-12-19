.. _ti-tvm-gs-compilation:

################
Compiling Models
################

TI TVM supports two options for compiling models with TIDL offload. The distinction is based on where layers that are not supported
by TIDL are executed during inference:

#. Executing unsupported layers on Arm (See flow in Figure 1).
#. Executing unsupported layers on C7x (See flow in Figure 2). This option frees up the Arm device to run other aspects of the application. It can also improve overall inference performance by minimizing communication across the Arm and C7x.

The following figures show components added by TIDL and the TI TVM fork outlined in red.

.. table:: Model compilation with unsupported layers

        +----------------+-------------------------------------------+
        | \              | \                                         |
        +================+===========================================+
        | \              | \                                         |
        +----------------+-------------------------------------------+
        | \              | \                                         |
        +----------------+-------------------------------------------+
        | \              | \                                         |
        +----------------+-------------------------------------------+
        | Figure 1:      | .. figure:: ../images/TVM_Compile_Arm.png |
        | Mapped to Arm  |    :scale: 30 %                           |
        |                |    :align: center                         |
        +----------------+-------------------------------------------+
        | Figure 2:      | .. figure:: ../images/TVM_Compile_C7x.png |
        | Mapped to C7x  |    :scale: 30 %                           |
        |                |    :align: center                         |
        +----------------+-------------------------------------------+


See :ref:`ti-tvm-compiling` for further details about TI TVM compilation.


.. _ti-tvm-gs-tidl-offload:

Compiling for TIDL Offload
==========================

TVM compilation is typically performed using a Python compilation script.  Example scripts
are provided in the `TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_python/tvm_dlr>`_
and `TI TVM fork <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_
Git repositories. You can use these examples as templates to modify for your own use cases.

An overview of TI Open Source Runtime and compilation options are provided in the
`TI edgeai-tidl-tools overview <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_python>`_.

The following Python functions are used in the examples provided by the TI TVM fork.

.. _ti-tvm-gs-compile-model:

compile_model
-------------

The ``compile_model`` function encapsulates the steps required to compile a model with TIDL offload. 

.. autofunction:: relay.ti_tests.compile_model.compile_model

:numref:`tidl-offload-overview` shows the key functions called from `compile_model`.

.. literalinclude:: ../../../tests/python/relay/ti_tests/compile_model.py
  :language: python
  :lines: 57-71
  :caption: Compiling a model with TIDL offload
  :name: tidl-offload-overview
  :linenos:

After a successful compile, the artifacts required to deploy the model are stored in the `artifacts_folder`.

.. _ti-tvm-gs-compile-relay:

compile_relay
-------------

The ``compile_relay`` function uses the TIDLCompiler class to partition the relay graph for offload subgraphs to TIDL. 

.. autofunction:: tvm.contrib.tidl.compile.compile_relay

