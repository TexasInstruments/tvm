.. _ti-tvm-developing:

============================
Recommended Development Flow
============================

This section describes the recommended development flow for using TVM to compile and
infer a model.


Step 1: Model Selection
=======================

You may have already developed and trained a model.  But if you are using a model
downloaded from public domain, we recommend you look at
`TI EdgeAI ModelZoo <https://github.com/TexasInstruments/edgeai-modelzoo>`_ first.
TI EdgeAI ModelZoo contains models that have been tweaked and optimized for inference speed
on TI SoCs.


Step 2: Compile with c7x_codegen=0
==================================

Adapt the example compilation scripts in
`TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_python/tvm_dlr>`_
for use with your model.  Additional examples are provided in the
`TVM Git repository <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_.

See the :ref:`ti-tvm-compiling` section for more about the compilation process and compiled artifacts. 

When troubleshooting and optimizing, check the following:

- Have all layers been offloaded to TIDL?
- If not, which layers are not offloaded?
- How many TIDL subgraphs are there?


Step 3: Inference and Performance Profiling
===========================================

Adapt the example inference scripts in 
`TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_python/tvm_dlr>`_
for your model.  Additional examples are provided in the
`TVM Git repository <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_.

First, get the compiled model artifacts (TVM deployable module) to run on the EVM. Then 
check to make sure the inference results match the expected outputs for the given inputs.

After the model is running correctly on the EVM, use the performance profiling method described
in the :ref:`ti-tvm-infering` section to see if performance matches expectations.


Step 4: Compile with c7x_codegen=1
==================================

If there are TIDL unsupported layers in the model, you may also try running them on the C7x.
Running layers on the C7x can help save the overhead between C7x TIDL subgraphs and layers on Arm.  This can
also lead to better performance with either TVM auto-generated C7x code or user-written C7x code
for the TIDL unsupported layers.


Step 5: Inference and Performance Profiling
===========================================

Once the model is compiled successfully with c7x_codegen=1, run it on the EVM and check to make sure the
inference results still match the expected outputs for the given inputs.

After the model is running correctly on the EVM, use the performance profiling method described
in the :ref:`ti-tvm-infering` section to see if performance matches expectations.

Step 6: Performance Tuning
==========================

If the performance of TIDL unsupported layers does not match expectations, try the following:

* Work around issues by rewriting Relay IR code.
* Optimize the C7x code (either TVM-generated or user-written)

See the :ref:`ti-tvm-extending` section for examples.  Feedback on TI E2E forum is welcome
(see the :ref:`ti-tvm-support` section).
