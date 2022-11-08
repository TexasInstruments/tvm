=====================
Compilation Explained
=====================

In this section, we will explain in more details about TVM compilation.


Environment Setup
=================
If not already set up by the edgeai, the following three environment variables are required
before running the compilation script.

- ``TIDL_TOOLS_PATH``: set to installed /path/to/processor_sdk_rtos/tidl_release/tidl_tools
- ``ARM64_GCC_PATH``: set to installed /path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
- ``CGT7X_ROOT``: set to installed /path/to/`TI C7x C/C++ compiler 2.1.1 LTS <https://www.ti.com/tool/download/C7000-CGT/2.1.1.LTS>`_

.. note::
  Processor SDK RTOS currently ships TI C7x compiler 3.0.0 STS.  There is a known bug in the
  compiler impacting TVM C7x code generation.  It has been fixed and will be released in the next
  Processor SDK.  For now, please use 2.1.1 LTS for TVM C7x code generation.

Frontends
=========

TVM can accept machine learning models in many formats, including Tensorflow/TFLite, Keras,
Core ML, MXNet, ONNX, PyTorch.  As the first step of compilation, these formats are all
imported into TVM's internal common representation, Relay IR, using different frontends in TVM.
In addition to various examples in Apache TVM documentation, we also have examples in
``tests/python/relay/ti_tests/models.py`` to show how to import from different network
formats into Relay IR.


Calibration Data
================

After we partition layers into subgraphs that can be offloaded to TIDL, these subgraphs need
to be imported to TIDL.  Because TIDL runs inference with quantized fixed-point values,
TIDL import process requires calibration data so that each layer's dynamic range can be
estimated and the scaling factor for converting between floating point and fixed point
can be computed.  

User will only need to provide the calibration data for the whole model.  TVM+TIDL compilation
flow will automatically obtain the corresponding tensor values at the TIDL subgraph boundaries
and feed those values into the TIDL import process for calibration.  User-provided calibration
data should represent the typical input for the model.


Artifacts
=========

Deployable module
-----------------

After successful TVM+TIDL compilation, a deployable module consists of 3 files are saved in
the ``<artifacts_folder>``.

- .json: Json file describing the compiled graph with information about nodes, allocation, etc.
- .so: The shared lib containing code to run nodes in the compiled graph.  This is a fat binary.
  Imported TIDL subgraph artifacts, generated C7x code are all embedded in this fat binary.
- .params: The weights associated with the nodes in the compiled graph.

At inference time, DLR/TVM runtime read these 3 files and create an runtime instance to run
inference.

.. hint::
  During the development, you may export the x86_64 Linux filesystem where you run compilation,
  and mount the filesystem on your EVM so that you do not need to copy the deployable module.

For deploying onto EVM, the deployable module is all that is needed.  During compilation, we
also save intermediate results in the ``<artifacts_folder>/tempDir`` directory.  They may
help understand and debug the compilation.  The following are some of the intermediate artifacts.

Relay graphs
------------

- relay_graph.orig.txt: the original relay graph from the TVM frontend
- relay_graph.prepared.txt: relay graph after transformations that prepare for TIDL offload
- relay_graph.annotated.txt: relay graph annotated for TIDL offload
- relay_graph.partitioned.txt: relay graph partitioned for TIDL offload
- relay_graph.import.txt: relay graph used to import into TIDL
- relay_graph.boundary.txt: relay graph used to obtain calibration data at TIDL subgraph boundaries
- relay_graph.optimized.txt: optimized relay graph for code generation
- relay_graph.wrapper.txt: wrapper relay graph on Arm side for dispatching the graph to C7x

For example, user may look into ``relay_graph.import.txt`` to see how many TIDL subgraphs are
created and which layers are not offloaded to TIDL.


Imported TIDL artifacts
-----------------------

TIDL subgraphs are imported into TIDL artifacts in TIDL specific formats.  They are embedded into
the ``.so`` fat binary in the deployable module.  The DLR/TVM runtime will retrieve TIDL artifacts
and invoke the TIDL runtime at inference time.

- relay.gv.svg: graphical view of the whole network and where the TIDL subgraphs are
- subgraph<n>_net.bin.svg: graphical view of TIDL subgraphs


Generated C7x code
------------------

When ``c7x_codegen`` is set to 1 in the compilation script, TVM will generated C7x code for layers
not offloaded to TIDL.  These C7x code are compiled and embedded into the ``.so`` fat binary in
the deployable module.  The DLR/TVM runtime will retrieve the C7x code and dispatch to C7x
for execution.

- model_<n>.c: generated code either to run a TIDL subgraph or non-TIDL layers


Debugging Compilation
=====================

We use an environment variable to help debug the TVM+TIDL compilation flow.

TIDL_RELAY_IMPORT_DEBUG=1
-------------------------

When set, the verbose output at the terminal gives more information about the TIDL import, e.g.
whether a node is supported by TIDL, relay node to TIDL node conversion, imported TIDL subgraph,
optimized TIDL subgraph, calibration process, etc.

.. code:: bash

  RelayImportDebug: In TIDL_relayAllowNode: 
  RelayImportDebug:   name: nn.conv2d
  RelayImportDebug: In TIDL_relayAllowNode: 
  RelayImportDebug:   name: nn.batch_norm

TIDL_RELAY_IMPORT_DEBUG=2, 3
----------------------------

More verbose information about importing TIDL subgraphs.

TIDL_RELAY_IMPORT_DEBUG=4
-------------------------

When set to 4, TIDL import will generate the output for each TIDL layer in the imported TIDL
subgraph, using calibration inputs.  They are stored in the files
``tempDir/tidl_import_subgraph<subgraph_id>.txt<layer_id><dimensions>_float.bin``.
The compilation will also generate corresponding output from running the original model on
x86_64 host using TVM code generation for x86_64.  They are stored in the files
``tempDir/tidl_<subgraph_id>_layer<layer_id>.npy``.
A script, ``python/tvm/contrib/tidl/compare_tensors.py`` is provided to compare the two
results with graphical view.

.. code:: bash

  # in tests/python/relay/ti_tests/
  TIDL_RELAY_IMPORT_DEBUG=4 python3 ./compile_model.py mv1_tf --target --tidl --c7x
  # compare_tensors.py <artifacts_folder> <subgraph_id> <layer_id>
  python3 $TVM_HOME/python/tvm/contrib/tidl/compare_tensors.py artifacts/mv1_tf_J7_target_tidl_c7x 0 2

.. figure:: images/compare_tensors_example.png
  :align: center

