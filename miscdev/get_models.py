"""
Tensorflow testcases
====================
This article is a test script to test tensorflow operator with Relay.
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from distutils.version import LooseVersion
import tvm
from tvm import relay
import tvm.relay.testing.tf as tf_testing


with tf.Graph().as_default():
  graph_def = tf_testing.get_workload(
                'InceptionV3/inception_v3_2016_08_28_frozen-with_shapes.pb')
  # Call the utility to import the graph definition into default graph.
  graph_def = tf_testing.ProcessGraphDefParam(graph_def)
  data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')

