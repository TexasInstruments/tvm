============================
Recommended Development Flow
============================

In this section, we talk about recommended development flow of using TVM to compile and
infer a model.


Step 1: Model selection
=======================

You may already have a model developed and trained by yourself.  But if you are using a model
downloaded from public domain, we recommend you take a look at
`TI EdgeAI ModelZoo <https://github.com/TexasInstruments/edgeai-modelzoo>`_ first.
TI EdgeAI ModelZoo contains models that have been tweaked and optimized for inference speed
on TI SoCs.


Step 2: Compile with c7x_codegen=0
==================================

Follow example compilation scripts in
`TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_python/tvm_dlr>`_
and adapt for your model.  We have additional examples in the
`TVM github repo <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_.

If you want to understand more about the compilation process and compiled artifacts, please
see the section "Compilation Explained".  Things to pay attention include:

- Has all layers been offloaded to TIDL
- If not, which layers are not offloaded
- The number of TIDL subgraphs


Step 3: Inference and performance profiling
===========================================

Follow example inference scripts in 
`TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_python/tvm_dlr>`_
and adapt for your model.  We have additional examples in the
`TVM github repo <https://github.com/TexasInstruments/tvm/tree/tidl-j7/tests/python/relay/ti_tests>`_.

The first thing is to get the compiled model artifacts (TVM deployable module) to run on EVM
and check if the inference results match expected outputs for the given inputs.

After the model is running correctly on the EVM, use the performance profiling method described
in section "Inference Explained" to see if performance match expectation.


Step 4: Compile with c7x_codegen=1
==================================

When there are TIDL unsupported layers in the model, you may also try running them on the C7x.
This could help save the overhead between C7x TIDL subgraphs and layers on Arm.  This could
also lead to better performance with either TVM auto-generated C7x code or user written C7x code
for the TIDL unsupported layers.


Step 5: Inference and performance profiling
===========================================

Once the model is compiled successfully with c7x_codegen=1, run it on EVM and check if the
inference results match expected outputs for the given inputs.

After the model is running correctly on the EVM, use the performance profiling method described
in section "Inference Explained" to see if performance match expectation.

Step 6: Performance tuning
==========================

If the performance of TIDL unsupported layers does not match expectation, we can try work
around them via Relay rewriting or try to optimize them with C7x code, either TVM generated
or user written, see section "Extending TVM" for examples.  Feedback on TI E2E forum is welcome
(please see "Support" section).
