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
"""Pre/Post processing for inputs/outputs to models."""


import os
import numpy as np
from models import models

testdata = {
  'airshow' : {
    'file': ["url", "https://git.ti.com/cgit/tidl/tidl-utils/plain/test/testvecs/input/airshow.jpg", "airshow.jpg"]
  },
  'cat' : {
    'file': ["url", "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true", "cat.png"]
  },
  'cat2' : {
    'file': ["url", "https://git.ti.com/cgit/tidl/tidl-api/plain/examples/test/testvecs/input/objects/cat-pet-animal-domestic-104827.jpeg", "cat2.jpeg"]
  },
  'street_small' : {
    'file': ["url", "https://github.com/dmlc/web-data/blob/master/gluoncv/detection/street_small.jpg?raw=true", "street_small.jpg"]
  }
}

colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128),
           (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255),
         ]


def get_calibdata_file(testdata_name):
  """ Return the testdata file, given the model name """
  # workaround SSL: CERTIFICATE_VERIFY_FAILED
  import ssl
  ssl._create_default_https_context = ssl._create_unverified_context

  if testdata[testdata_name]['file'][0] == "url":
    from tvm.contrib.download import download_testdata
    _, url, save_name = testdata[testdata_name]['file']
    download_file = download_testdata(url, save_name, module="data")
    return download_file


def get_testdata_file(testdata_name):
  """ Return the local testdata file, given the model name """
  if testdata[testdata_name]['file'][0] == "url":
    _, url, save_name = testdata[testdata_name]['file']
    return f"testdata/{save_name}"


def load_image(img_file, resize_wh, crop_wh, mean, scale, needs_nchw, needs_quant):
  from PIL import Image
  orig_img = Image.open(img_file)  # HWC
  resized_img = orig_img.resize((resize_wh[0], resize_wh[1]))
  assert(resize_wh[0] >= crop_wh[0] and resize_wh[1] >= crop_wh[1]), \
        "resize size needs to be bigger than crop size"
  if resize_wh[0] > crop_wh[0] or resize_wh[1] > crop_wh[1]:
    wh_start = [ (x - y) / 2 for x, y in zip(resize_wh, crop_wh)]
    wh_end   = [ x + y for x, y in zip(wh_start, crop_wh)]
    cropped_img = resized_img.crop((wh_start[0], wh_start[1], wh_end[0], wh_end[1]))
  else:
    cropped_img = resized_img

  if needs_quant:
    norm_img = np.asarray(cropped_img).astype("uint8")
  else:
    # Normalize input data to (-1, 1)
    norm_img = np.asarray(cropped_img).astype("float32")
    norm_img[:, :, 0] = (norm_img[:, :, 0] - mean[0]) * scale[0]
    norm_img[:, :, 1] = (norm_img[:, :, 1] - mean[1]) * scale[1]
    norm_img[:, :, 2] = (norm_img[:, :, 2] - mean[2]) * scale[2]

  # Set batch_size dimension of input data: NHWC
  input_data = norm_img[np.newaxis, :, :, :]
  if needs_nchw:
    input_data = input_data.transpose(0, 3, 1, 2)  # NCHW
  return input_data


def get_input_from_files(model_name, files, batch_size=0):
  input_name = models[model_name]['input_info']['name']
  batch      = models[model_name]['input_info']['shape'][0] if batch_size == 0 else batch_size
  resize_wh  = models[model_name]['input_info']['resize_wh']
  crop_wh    = models[model_name]['input_info']['crop_wh']
  mean       = models[model_name]['input_info']['mean']
  scale      = models[model_name]['input_info']['scale']
  is_nchw    = models[model_name]['input_info']['is_nchw']
  is_quant   = models[model_name]['input_info']['dtype'] == "uint8"

  num_files = len(files)
  num_batches = (num_files // batch) + (0 if num_files % batch == 0 else 1)
  inputs = []
  for j in range(num_batches):
    batch_images = []
    for i in range(batch):
      f = files[ (j * batch  + i) % num_files]
      batch_images.append(load_image(f, resize_wh, crop_wh, mean, scale, is_nchw, is_quant))
    inputs.append({input_name : np.concatenate(batch_images)})
  return inputs


def get_calib_inputs(model_name, batch_size=0):
  """ Return pre-processed calibration data"""
  calib_files = [get_calibdata_file(x) for x in models[model_name]['calib_data']]
  return get_input_from_files(model_name, calib_files, batch_size)


def get_test_inputs(model_name, batch_size=0):
  """ Return pre-processed test data"""
  test_files = [get_testdata_file(x) for x in models[model_name]['test']['test_data']]
  return get_input_from_files(model_name, test_files, batch_size)


def get_top5(res):
  """ Return top5 values and corresponding indicies """
  bs_top5 = []
  bs_values = []
  for n in range(res.shape[0]):
    top5 = []
    values = []
    for i in range(5):
      top = np.argmax(res[n, :])
      top5.append(top)
      values.append(res[n, top])
      res[n, top] = 0
    print(f"{n} Inference results (top5):\n  {top5}\n  {values}")
    bs_top5.append(top5)
    bs_values.append(values)
  return bs_top5, bs_values


def save_seg(model_name, res, artifacts_folder):
  """ Save blended image using semantic segmentation results as masks """
  import cv2
  img_file = get_testdata_file(models[model_name]['test']['test_data'][0])
  w, h = models[model_name]['input_info']['crop_wh']
  orig_img = cv2.imread(img_file)
  resized_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_CUBIC)
  output = np.squeeze(res[0], axis=0)
  class_IDs = np.argmax(output, axis=2).astype(int)
  mask_img = resized_img.copy()
  for i in range(w):
    for j in range(h):
      if class_IDs[i][j] != 0:
        mask_img[i][j] = list(colors[class_IDs[i][j]%len(colors)])
  output_img = cv2.addWeighted(resized_img, 0.7, mask_img, 0.3, 0.0)
  output_img = cv2.resize(output_img, (orig_img.shape[0], orig_img.shape[1]),
                          interpolation=cv2.INTER_CUBIC)
  out_file = os.path.join(artifacts_folder, "output.png")
  cv2.imwrite(out_file, output_img)
  print(f"Semantic segmentation results in {out_file}")


def save_od(model_name, res, artifacts_folder):
  """ Save image with boudning boxes from object detection results """
  import cv2
  img_file = get_testdata_file(models[model_name]['test']['test_data'][0])
  w, h = models[model_name]['input_info']['crop_wh']
  orig_img = cv2.imread(img_file)
  resized_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_CUBIC)
  class_IDs, scores, bounding_boxes = res
  for i, score in enumerate(np.squeeze(scores)):
    if score > 0.2:
      cv2.rectangle(resized_img, (int(bounding_boxes[0][i][0]), int(bounding_boxes[0][i][1])),
                                 (int(bounding_boxes[0][i][2]), int(bounding_boxes[0][i][3])),
                                 colors[int(class_IDs[0][i][0])%len(colors)], 2)
  output_img = cv2.resize(resized_img, (orig_img.shape[0], orig_img.shape[1]),
                          interpolation=cv2.INTER_CUBIC)
  out_file = os.path.join(artifacts_folder, "output.png")
  cv2.imwrite(out_file, output_img)
  print(f"Object detection results in {out_file}")


def save_od2(model_name, res, artifacts_folder):
  """ Save image with boudning boxes from object detection results """
  import cv2
  img_file = get_testdata_file(models[model_name]['test']['test_data'][0])
  w, h = models[model_name]['input_info']['crop_wh']
  orig_img = cv2.imread(img_file)
  resized_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_CUBIC)
  bounding_boxes, class_IDs = res
  for box, label in zip(np.squeeze(bounding_boxes), np.squeeze(class_IDs)):
    if box[4] > 0.45:
      cv2.rectangle(resized_img, (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])),
                                 colors[int(label)%len(colors)], 2)
  output_img = cv2.resize(resized_img, (orig_img.shape[0], orig_img.shape[1]),
                          interpolation=cv2.INTER_CUBIC)
  out_file = os.path.join(artifacts_folder, "output.png")
  cv2.imwrite(out_file, output_img)
  print(f"Object detection results in {out_file}")

def save_od_yolo(model_name, res, artifacts_folder):
  """ Save image with boudning boxes from object detection results """
  import cv2
  img_file = get_testdata_file(models[model_name]['test']['test_data'][0])
  w, h = models[model_name]['input_info']['crop_wh']
  orig_img = cv2.imread(img_file)
  resized_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_CUBIC)
  output = res[0]

  for box in np.squeeze(output):
    label = box[5]
    if box[4] > 0.34:
      cv2.rectangle(resized_img, (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])),
                                 colors[int(label)%len(colors)], 2)
      keypoints = box[6:]
      for i in range(0, len(keypoints), 3):
        if keypoints[i+2] > 0.5:
          cv2.circle(resized_img, (int(keypoints[i]), int(keypoints[i+1])), radius=1,
                    color=colors[int(label)%len(colors)], thickness=-1)
  output_img = cv2.resize(resized_img, (orig_img.shape[0], orig_img.shape[1]),
                          interpolation=cv2.INTER_CUBIC)
  out_file = os.path.join(artifacts_folder, "output.png")
  cv2.imwrite(out_file, output_img)
  print(f"Object detection results in {out_file}")

def check_test_results(model_name, res, artifacts_folder):
  """ Check inference results """
  if 'in_top5' in models[model_name]['test']:
    bs_top5, bs_values = get_top5(res[0])
    e_top5 = models[model_name]['test']['in_top5']
    for j in range(len(bs_top5)):
      for i in e_top5[j % len(e_top5)]:
        if i not in bs_top5[j]:
          print(f"Expected index {i} not in top5 results")
          return False
  elif 'save_seg' in models[model_name]['test']:
    if models[model_name]['test']['save_seg']:
      save_seg(model_name, res, artifacts_folder)
  elif 'save_od' in models[model_name]['test']:
    if models[model_name]['test']['save_od']:
      save_od(model_name, res, artifacts_folder)
  elif 'save_od2' in models[model_name]['test']:
    if models[model_name]['test']['save_od2']:
      save_od2(model_name, res, artifacts_folder)
  elif 'save_od_yolo' in models[model_name]['test']:
    if models[model_name]['test']['save_od_yolo']:
      save_od_yolo(model_name, res, artifacts_folder)

  return True

