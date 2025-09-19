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

"""
Model preprocessing configuration for calibration data generation.

This module provides model-specific preprocessing parameters (input_mean and input_scale)
extracted from EdgeAI model configurations to ensure consistency between calibration
data generation and model inference.
"""

import os
from typing import Dict, List, Optional, Tuple


def get_model_preprocessing_config(model_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Get preprocessing configuration for a model based on its filename.

    Args:
        model_path: Path to the model file

    Returns:
        Tuple of (input_mean, input_scale) or (None, None) if no specific config found
    """
    # Model-specific preprocessing configurations extracted from EdgeAI model_configs.py
    model_configs = {
        # ONNX Classification models - ImageNet preprocessing
        "resnet18_opset9": {
            "input_mean": [123.675, 116.28, 103.53],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },
        "regnetx-200mf": {
            "input_mean": [123.675, 116.28, 103.53],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },
        "deeplabv3lite_mobilenetv2": {
            "input_mean": [123.675, 116.28, 103.53],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },

        # ONNX Detection models - Detection preprocessing
        "ssd-lite_mobilenetv2_fpn": {
            "input_mean": [0, 0, 0],
            "input_scale": [0.003921568627, 0.003921568627, 0.003921568627]
        },
        "ssd_mobilenetv2_lite_512x512_20201214_model": {
            "input_mean": [0, 0, 0],
            "input_scale": [0.003921568627, 0.003921568627, 0.003921568627]
        },
        "yolox_s_lite_640x640_20220307_model": {
            "input_mean": [0, 0, 0],
            "input_scale": [0.003921568627, 0.003921568627, 0.003921568627]
        },
        "yolox_nano_lite_416x416_20220214_model": {
            "input_mean": [0, 0, 0],
            "input_scale": [1, 1, 1]
        },
        "yolox_s_lite_640x640_20220221_model": {
            "input_mean": [0, 0, 0],
            "input_scale": [1, 1, 1]
        },

        # TFLite models - MobileNet preprocessing
        "mobilenet_v1_1.0_224": {
            "input_mean": [127.5, 127.5, 127.5],
            "input_scale": [1/127.5, 1/127.5, 1/127.5]
        },
        "mobilenetv2_4batch": {
            "input_mean": [127.5, 127.5, 127.5],
            "input_scale": [1/127.5, 1/127.5, 1/127.5]
        },
        "ssd_mobilenet_v2_300_float": {
            "input_mean": [127.5, 127.5, 127.5],
            "input_scale": [1/127.5, 1/127.5, 1/127.5]
        },
        "ssdlite_mobiledet_dsp_320x320_coco_20200519": {
            "input_mean": [127.5, 127.5, 127.5],
            "input_scale": [1/127.5, 1/127.5, 1/127.5]
        },
        "deeplabv3_mnv2_ade20k_float": {
            "input_mean": [127.5, 127.5, 127.5],
            "input_scale": [1/127.5, 1/127.5, 1/127.5]
        },

        # Caffe-based models (converted to ONNX)
        "caffe_mobilenet_v1": {
            "input_mean": [103.94, 116.78, 123.68],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },
        "caffe_mobilenet_v2": {
            "input_mean": [103.94, 116.78, 123.68],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },
        "caffe_squeezenet_v1_1": {
            "input_mean": [103.94, 116.78, 123.68],
            "input_scale": [1, 1, 1]
        },
        "caffe_resnet10": {
            "input_mean": [0, 0, 0],
            "input_scale": [1, 1, 1]
        },
        "caffe_mobilenetv1_ssd": {
            "input_mean": [103.94, 116.78, 123.68],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },
        "caffe_pelee_ssd": {
            "input_mean": [103.94, 116.78, 123.68],
            "input_scale": [0.017125, 0.017507, 0.017429]
        },
        "caffe_erfnet": {
            "input_mean": [0, 0, 0],
            "input_scale": [1, 1, 1]
        }
    }

    model_name = os.path.basename(model_path)
    model_name_no_ext = os.path.splitext(model_name)[0]

    # Try exact match first
    if model_name_no_ext in model_configs:
        config = model_configs[model_name_no_ext]
        return config["input_mean"], config["input_scale"]

    # Try pattern matching for common model types
    model_lower = model_name_no_ext.lower()

    # ResNet models -> ImageNet preprocessing
    if "resnet18" in model_lower and model_path.endswith('.onnx'):
        return [123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429]

    # MobileNet models - depends on format
    if "mobilenet" in model_lower:
        if model_path.endswith('.tflite'):
            return [127.5, 127.5, 127.5], [1/127.5, 1/127.5, 1/127.5]
        elif model_path.endswith('.onnx'):
            return [123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429]

    # SSD models
    if "ssd" in model_lower:
        if model_path.endswith('.tflite'):
            return [127.5, 127.5, 127.5], [1/127.5, 1/127.5, 1/127.5]
        else:
            return [0, 0, 0], [0.003921568627, 0.003921568627, 0.003921568627]

    # YOLO models -> Detection preprocessing
    if "yolo" in model_lower:
        return [0, 0, 0], [0.003921568627, 0.003921568627, 0.003921568627]

    # DeepLab models
    if "deeplabv3" in model_lower:
        if model_path.endswith('.tflite'):
            return [127.5, 127.5, 127.5], [1/127.5, 1/127.5, 1/127.5]
        else:
            return [123.675, 116.28, 103.53], [0.017125, 0.017507, 0.017429]

    # Caffe models -> Caffe preprocessing
    if "caffe" in model_lower:
        return [103.94, 116.78, 123.68], [0.017125, 0.017507, 0.017429]

    # No specific configuration found
    return None, None