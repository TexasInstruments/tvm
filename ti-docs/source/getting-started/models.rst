.. _ti-tvm-gs-models:

############
Model Basics
############

Machine learning models usually take input data in tensor (multi-dimensional array) format
and produce output data in tensor format.  Before compiling and inferencing a model using
TVM, it is recommended that you understand the basic requirements for your model:

- model semantics, including the meanings of input and output
- input name, shape
- output shape
- sample of input data
- expected output data corresponding to the sample input data

Experimenting with pre-processing input data, running inference, and post-processing input data
can be accomplished even without a TVM setup and TI devices. For example, if you have an ONNX model, you can
write a Python script like the following to import the ``onnxruntime`` package, pre-process input data, run inferencing, post-process
output data, and verify the results.

.. code-block:: python

  import onnxruntime
  model_file = ...
  input_name = ...
  input_data = ...
  sess = onnxruntime.InferenceSession(model_file)
  output = sess.run([], {input_name : input_data})

In the topics that follow, we assume you understand the basics of your model, so we focus on TVM-related aspects.  Although we show data pre-processing, model input names and shapes, and data
post-processing in our examples, the details about these steps are likely to be different for your model, so it is your
responsibility to set them correctly.
