############
Model Basics
############

Machine learning models usually take input data in tensor (multi-dimensional array) format
and produce output data in tensor format.  Before compiling and inferencing a model using
TVM, it is recommended to understand the basics of a model.

- model semantics, meanings of input and output
- input name, shape
- output shape
- a sample input data
- expected output data corresponding to the sample input data

Experimenting with pre-processing input data, running inference and post-processing input data
can be done even without TVM setup and TI devices.  E.g. if you have an ONNX model, you can
write a python script, import onnxruntime package, pre-process data, run inference, post-process
data, and verify the results.

.. code-block:: python

  import onnxruntime
  model_file = ...
  input_name = ...
  input_data = ...
  sess = onnxruntime.InferenceSession(model_file)
  output = sess.run([], {input_name : input_data})

In what follows, we assume that users understand the model basics and we will focus on TVM
related aspects.  Although we have data pre-processing, model input names and shapes, data
post-processing in our examples, but they could be different for users models and it is users'
responsibility to set them correctly.
