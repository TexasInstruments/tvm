# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Helper functions to convert TensorFlow, TFLite, PyTorch, ONNX and MXNet models to Relay"""


import os


inputs_224 = {
  'name': "input",
  'shape': (1, 3, 224, 224),
  'is_nchw': True,
  'dtype': "float32",
  'resize_wh': [256, 256],
  'crop_wh': [224, 224],
  'mean': [128, 128, 128],
  'scale': [0.0078125, 0.0078125, 0.0078125],
}

inputs_npy = {
  'name': "input",
  'is_nchw': True,
  'dtype': "float32",
  'resize_wh': None,
  'crop_wh': None,
  'mean': None,
  'scale': None,
}

test_top5 = {
  'test_data' : ['airshow', 'cat'],
  'in_top5' : [[895, 403], [282, 281]],
  #'prob5': [],
}


test_seg = {
  'test_data' : ['street_small'],
  'save_seg' : True,
}


test_od = {
  'test_data' : ['street_small'],
  'save_od' : True,
}

test_od_no_scores = {
  'test_data' : ['street_small'],
  'save_od2' : True,
}

test_od_yolo = {
  'test_data' : ['street_small'],
  'save_od_yolo' : True,
}

test_save_npy = {
  'save_npy' : True,
}

tvm_models_dir = os.path.join(os.environ["HOME"], ".tvm_test_data/models")


models = {
  #'mv1_tf' : {
  #  'file': ["url", "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz", "mobilenet_v1_1.0_224.tgz", "mobilenet_v1_1.0_224_frozen.pb"],
  #  'output_name': "MobilenetV1/Predictions/Softmax",
  #  'input_info': {**inputs_224, 'shape':(1,224,224,3), 'is_nchw':False},
  #  'calib_data': ['airshow', 'cat', 'cat2'],
  #  'test': {**test_top5, 'in_top5':[[896, 404], [283, 282]]},
  #  'batch_size': [0, 2],
  #},

  'mv2_quant_tfl' : {
    'file': ["url", "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz", "mobilenet_v2_1.0_224_quant.tgz", "mobilenet_v2_1.0_224_quant.tflite"],
    'input_info': {**inputs_224, 'shape':(1,224,224,3), 'is_nchw':False, 'dtype':"uint8"},
    'calib_data': ['cat'],
    'test': {**test_top5, 'in_top5':[[896, 404], [283, 282]]},
  },

  'mv2_onnx' : {
    'file': ["url", "https://github.com/onnx/models/raw/cbda9ebd037241c6c6a0826971741d5532af8fa4/vision/classification/mobilenet/model/mobilenetv2-7.onnx", "mobilenetv2-7.onnx", "mobilenetv2-7.onnx"],
    'input_info': {**inputs_224, 'name':"data"},
    'calib_data': ['cat2', 'airshow'],
    'test': {**test_top5, 'in_top5':[[895, 403], [282]]},
    'batch_size': [0, 2],
    'advanced_options': {
      'calibration_iterations': 10
    },
  },

  'mv2_pth' : {
    'file': ["torchvision", "mobilenet_v2"],
    'input_info': {**inputs_224, 'name':"data", 'mean':[123.675,116.28,103.53], 'scale':[0.017125,0.017507,0.017429]},
    'calib_data': ['airshow', 'cat2', 'cat'],
    'test': test_top5,
    'tidl_bits' : 16,
    'batch_size': [0, 2],
  },

  'deeplabv3_tfl' : {
    'file': ["url", "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite?raw=true", "deeplabv3_257_mv_gpu.tflite", "deeplabv3_257_mv_gpu.tflite"],
    'input_info': {**inputs_224, 'name':"sub_7", 'shape':(1,257,257,3), 'is_nchw':False,
                   'resize_wh':[257,257], 'crop_wh':[257,257]},
    'calib_data': ['street_small'],
    'test': test_seg,
  },

  'swin_tiny_timm' : {
    'file': ["timm", "swin_tiny_patch4_window7_224"],
    'input_info': {**inputs_224, 'name':"input.1"},
    'calib_data': ['cat', 'cat2'],
    'test': test_top5,
  },

  'mv3_large_timm' : {
    'file': ["timm", "mobilenetv3_large_100"],
    'input_info': {**inputs_224, 'name':"input.1"},
    'tidl_bits' : 16,
    'calib_data': ['cat', 'cat2', 'airshow'],
    'test': test_top5,
  },

  'mv2_od_onnx' : {
    'file': ["file", "/cgnas/edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_model.onnx", "ssd_mobilenetv2_fpn_lite_512x512_20201110_model", "ssd_mobilenetv2_fpn_lite_512x512_20201110_model"],
    'input_info': {**inputs_224, 'name':"input", 'shape':(1,3,512,512), 'is_nchw':True,
                   'resize_wh':[512,512], 'crop_wh':[512,512]},
    'calib_data': ['cat', 'cat2'],
    'test': test_od_no_scores,
    'advanced_options': {
      'object_detection:meta_layers_names_list': "/cgnas/edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_model.prototxt",
      'object_detection:meta_arch_type': 3
    },
  },

  'yolov5_od_onnx' : {
    'file': ["file", "/cgnas/edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx", "yolov5s6_pose_640_ti_lite_54p9_82p2", "yolov5s6_pose_640_ti_lite_54p9_82p2"],
    'input_info': {**inputs_224, 'name':"images", 'shape':(1,3,640,640), 'is_nchw':True,
                   'resize_wh':[640,640], 'crop_wh':[640,640]},
    'calib_data': ['street_small'],
    'test': test_od_yolo,
    'tidl_bits' : 16,
    'advanced_options': {
      'object_detection:meta_layers_names_list': "/cgnas/edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_metaarch.prototxt",
      'object_detection:meta_arch_type': 6
    },
  },

  'lidar_od_onnx' : {
    'file': ["file", "/cgnas/edgeai-modelzoo/models/vision/detection_3d/kitti/mmdet3d/lidar_point_pillars_10k_496x432_3class_qat-p2.onnx", "lidar_point_pillars_10k_496x432_3class_qat-p2", "lidar_point_pillars_10k_496x432_3class_qat-p2"],
    'input_info': [ {**inputs_npy, 'name':"x.3", 'shape':(1,10,32,10000), 'calib_input':'/cgnas/tvm/deps/testdata/point_pillars/pp.npy'},
                    {**inputs_npy, 'name':"data.1", 'shape':(1,64,214272), 'calib_input':'/cgnas/tvm/deps/testdata/point_pillars/data.npy'},
                    {**inputs_npy, 'name':"coors", 'shape':(1,64,10000), 'calib_input':'/cgnas/tvm/deps/testdata/point_pillars/coors.npy'}, ],
    'test': {**test_save_npy, 'inputs': None},
    'tidl_bits' : 8,
    'advanced_options': {
      'object_detection:meta_layers_names_list': "/cgnas/edgeai-modelzoo/models/vision/detection_3d/kitti/mmdet3d/lidar_point_pillars_10k_496x432_3class.prototxt",
      'object_detection:meta_arch_type': 7,
      'calibration_iterations': 1
    },
  },
}


def get_tidl_bits(model_name):
  """ Return the number of tensor bits used for TIDL import """
  if 'tidl_bits' in models[model_name]:
    return models[model_name]['tidl_bits']
  return 8


def get_model_file(model_name):
  """ Return the model file, given the model name """
  if models[model_name]['file'][0] == "url":
    from tvm.contrib.download import download_testdata
    from tvm.contrib.tar import untar
    _, url, save_name, real_name = models[model_name]['file']
    download_file = download_testdata(url, save_name, module="models")
    if save_name != real_name:
      # model file is inside a downloaded tarball
      model_dir = os.path.dirname(download_file)
      model_file = os.path.join(model_dir, real_name)
      if not os.path.exists(model_file):
        untar(download_file, model_dir)
    else:
      model_file = download_file
    return model_file
  elif models[model_name]['file'][0] == "mxnet":
    from gluoncv import model_zoo
    _, mxnet_model_name = models[model_name]['file']
    return model_zoo.get_model(mxnet_model_name, pretrained=True)
  elif models[model_name]['file'][0] == "torchvision":
    import torchvision.models as tv_models
    _, tv_model_name = models[model_name]['file']
    model = getattr(tv_models, tv_model_name)(pretrained=True)
    return model.eval()
  elif models[model_name]['file'][0] == "timm":
    _, timm_model_name = models[model_name]['file']
    input_node = models[model_name]['input_info']['name']
    model_file = os.path.join(tvm_models_dir, timm_model_name + ".onnx")
    if not os.path.exists(model_file):
      import torch
      import timm
      from prepostproc import get_calib_inputs
      model = timm.create_model(timm_model_name, pretrained=True).eval()
      calib_inputs = get_calib_inputs(model_name)
      data = torch.from_numpy(calib_inputs[0].get(input_node))
      torch.onnx.export(model, data, model_file,
                        export_params=True, opset_version=11, do_constant_folding=True)
    return model_file
  elif models[model_name]['file'][0] == "file":
    _, model_file, _, _ = models[model_name]['file']
    return model_file

def get_relay_model(model_name : str, batch_size:int=0):
  """Obtain model and convert to Relay"""

  from tvm import relay
  from utils import disable_outputs, restore_outputs

  def from_tf(model_file, model_name):
    import tensorflow as tf
    from tvm.relay.testing import tf as tf_testing

    layout = "NCHW" if models[model_name]['input_info']['is_nchw'] else "NHWC"
    output_node = models[model_name]['output_name']
    shape_dict = {input_node : input_shape}
    print("Input node shape dict:" + str(shape_dict))

    fds = disable_outputs()
    with tf.gfile.GFile(model_file, 'rb') as f:
      # Import tensorflow graph definition to relay frontend.
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      graph = tf.import_graph_def(graph_def, name='')
      graph_def = tf_testing.ProcessGraphDefParam(graph_def)

      # Add shapes to the graph.
      with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, output_node)
      mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict,
                                                   outputs=None)
    restore_outputs(fds)
    print(f"Tensorflow model {model_name} imported to Relay IR.")
    return mod, params

  def from_tfl(model_file, model_name):
    tflite_model_buf = open(model_file, "rb").read()
    try:
      import tflite
      tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
      import tflite.Model
      tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input_dtype = models[model_name]['input_info']['dtype']
    mod, params = relay.frontend.from_tflite(tflite_model, shape_dict={input_node : input_shape},
                                             dtype_dict={input_node : input_dtype})
    print(f"TFLite model {model_name} imported to Relay IR.")
    return mod, params

  def from_onnx(model_file, model_name):
    import onnx

    #mod, params = relay.frontend.from_onnx(onnx.load(model_file), shape={input_node : input_shape})
    mod, params = relay.frontend.from_onnx(onnx.load(model_file), shape=inputs_shape_dict)
    print(f"ONNX model {model_name} imported to Relay IR.")
    return mod, params

  def from_mxnet(model_file, model_name):
    mod, params = relay.frontend.from_mxnet(model_file, {input_node : input_shape})
    print(f"MxNet model {model_name} imported to Relay IR.")
    return mod, params

  def from_pytorch(model_file, model_name):
    import torch
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model_file, input_data).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [(input_node, input_shape)])
    print(f"Pytorch model {model_name} imported to Relay IR.")
    return mod, params


  model_file = get_model_file(model_name)

  # Convert the model to Relay
  inputs = models[model_name]['input_info']
  if not isinstance(inputs, list):
    inputs = [inputs]
  inputs_shape_dict = {}
  inputs_dtype_dict = {}
  for inp in inputs:
    input_node = inp['name']
    input_shape = inp['shape']
    input_dtype = inp['dtype']
    if (batch_size != 0):
      input_shape = (batch_size, *input_shape[1:])
    inputs_shape_dict[input_node] = input_shape
    inputs_dtype_dict[input_node] = input_dtype

  advanced_options = models[model_name].get('advanced_options')
  mod, params = None, None

  if model_name.endswith("_tf"):
    mod, params = from_tf(model_file, model_name)
  elif model_name.endswith("_tfl"):
    mod, params = from_tfl(model_file, model_name)
  elif model_name.endswith("_onnx") or model_name.endswith("_timm"):
    mod, params = from_onnx(model_file, model_name)
  elif model_name.endswith("_mxnet"):
    mod, params = from_mxnet(model_file, model_name)
  elif model_name.endswith("_pth"):
    mod, params = from_pytorch(model_file, model_name)

  return mod, params, advanced_options

