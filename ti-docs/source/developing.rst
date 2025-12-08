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
`TI EdgeAI ModelZoo <https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo>`_ first.
TI EdgeAI ModelZoo contains models that have been tweaked and optimized for inference speed
on TI SoCs.


Step 2: Compile with c7x_codegen=0
==================================

Adapt the example compilation scripts in
`TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools>`_
for use with your model by updating the 'config.yaml' file.

See the :ref:`ti-tvm-compiling` section for more about the compilation process and compiled artifacts. 

When troubleshooting and optimizing, check the following:

- Have all layers been offloaded to TIDL?
- If not, which layers are not offloaded?
- How many TIDL subgraphs are there?


Step 3: Inference and Performance Profiling
===========================================

Adapt the example inference scripts in 
`TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools>`_
for your model.

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

