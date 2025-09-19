..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Deploy Models to TI TIDL
=========================

Introduction
------------

Texas Instruments Deep Learning (TIDL) is an accelerated inference library for TI's EdgeAI SoCs
with C7x DSP cores and Matrix Multiplication Accelerator (MMA). This integration uses TVM's Bring
Your Own Codegen (BYOC) infrastructure to partition models and offload supported subgraphs to TIDL
for acceleration, while running unsupported operations on ARM Cortex-A cores or generating optimized
C7x code.

The TIDL backend enables:

* Automatic model partitioning between TIDL and TVM
* Quantization support (8-bit, 16-bit, 32-bit)
* C7x DSP code generation for unsupported layers
* Integration with TI's processor SDK ecosystem

Supported Platforms
-------------------

The TIDL target supports the following TI EdgeAI platforms:

* **AM68PA** - Automotive processors
* **AM68A** - High-performance automotive SoCs
* **AM69A** - Edge AI inference processors
* **AM67A** - Vision processing SoCs
* **AM62A** - General purpose edge AI processors

Prerequisites
-------------

Before using the TIDL backend, ensure you have:

* TI Processor SDK RTOS installation.
* edgeai-tidl-tools repo built from source.
* Target hardware/EVM setup.
* Model in supported format (ONNX or TFLite).

Basic Usage with TVMC
---------------------

The simplest way to compile models for TIDL is using TVM's command-line interface (TVMC):

.. code-block:: bash

    # Basic TIDL compilation
    tvmc compile model.onnx \
        --target tidl \
        --target-tidl-platform am69a \
        --target-tidl-enable-offload

    # With calibration images for quantization
    tvmc compile model.onnx \
        --target tidl \
        --target-tidl-platform am69a \
        --target-tidl-calibration-images ./calibration_images \
        --target-tidl-enable-offload \
        --target-tidl-tensor-bits 8

TVMC TIDL Options
-----------------

Required Options
~~~~~~~~~~~~~~~~

* ``--target-tidl-platform`` - Target TI platform (am68pa, am68a, am69a, am67a, am62a)

Acceleration Options
~~~~~~~~~~~~~~~~~~~

* ``--target-tidl-enable-offload`` - Enable TIDL acceleration for supported layers
* ``--target-tidl-enable-c7x-codegen`` - Generate C7x code for unsupported layers (instead of ARM)

Quantization Options
~~~~~~~~~~~~~~~~~~~

* ``--target-tidl-calibration-images`` - Directory containing calibration images for quantization
* ``--target-tidl-calibration-frames`` - Number of calibration frames to generate from images (default: 10)
* ``--target-tidl-input-mean`` - Input mean values for RGB channels [R, G, B] (auto-detected if not specified)
* ``--target-tidl-input-scale`` - Input scale values for RGB channels [R, G, B] (auto-detected if not specified)
* ``--target-tidl-tensor-bits`` - Quantization precision: 8, 16, or 32 bits (default: 8)

Output and Build Options
~~~~~~~~~~~~~~~~~~~~~~~~

* ``--target-tidl-artifacts-folder`` - Output directory for compilation artifacts (default: ./tidl_artifacts)
* ``--target-tidl-compile-for-device`` - Compile for target device (aarch64) vs host (x86)

Advanced Options
~~~~~~~~~~~~~~~

* ``--target-tidl-deny-list`` - Comma-separated list of operations to exclude from TIDL offloading

Object Detection Options
~~~~~~~~~~~~~~~~~~~~~~~~

* ``--target-tidl-od-meta-arch-type`` - Object detection meta architecture type (e.g., 3 for SSD, required for OD models)
* ``--target-tidl-od-meta-layers-names-list`` - Path to prototxt file containing object detection layer metadata (required for OD models)

Preparing Calibration Data
---------------------------

TVM assumes certain calibration data, but for more accurage
performance, provide a set of calibration images in a directory.

Image Directory Setup
~~~~~~~~~~~~~~~~~~~~~~

The TVM TIDL integration automatically generates calibration data from image directories during compilation:

.. code-block:: bash

    # Create a directory with representative images
    mkdir calibration_images
    # Copy images to be used for calibration (jpg)
    cp dataset/image1.jpg calibration_images/
    cp dataset/image2.png calibration_images/
    # ... add more images

    # TVMC will automatically process these images during compilation
    tvmc compile model.onnx \
        --target tidl \
        --target-tidl-platform am69a \
        --target-tidl-calibration-images ./calibration_images \
        --target-tidl-calibration-frames 20

Complete Examples
-----------------

Here's a complete example compiling a ResNet-50 model for AM69A with 8-bit quantization:

.. code-block:: bash

    # Compile ResNet-50 with TIDL
    tvmc compile path/to/resnet50.onnx \
        --target tidl \
        --target-tidl-platform am69a \
        --target-tidl-calibration-images ./calibration_images \
        --target-tidl-artifacts-folder ./resnet50_artifacts \
        --target-tidl-tensor-bits 8 \
        --target-tidl-enable-offload \
        --target-tidl-enable-c7x-codegen \
        --target-tidl-compile-for-device

    # Compilation artifacts will be generated in ./resnet50_artifacts/

Object Detection Model Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For object detection models that require prototxt metadata, use the additional OD options:

.. code-block:: bash

    # Compile SSD MobileNet object detection model
    tvmc compile path/to/ssd_mobilenet.onnx \
        --target tidl \
        --target-tidl-platform am69a \
        --target-tidl-calibration-images ./od_calibration_images \
        --target-tidl-artifacts-folder ./ssd_artifacts \
        --target-tidl-tensor-bits 8 \
        --target-tidl-enable-offload \
        --target-tidl-enable-c7x-codegen \
        --target-tidl-compile-for-device \
        --target-tidl-od-meta-arch-type 3 \
        --target-tidl-od-meta-layers-names-list path/to/model_metadata.prototxt

    # The meta architecture type depends on your OD model:
    # TODO: document possible meta-arch-type values

    # The prototxt file contains layer metadata specific to the OD model
    # Refer to TI documentation for the exact format and requirements

Programming Interface
--------------------

For programmatic access (see examples/osrt_python/tvm in the
edgeai-tidl-tools for an example), use the Python API directly:

.. code-block:: python

    import tvm
    from tvm.contrib.tidl import compile as tidl_compile
    from tvm.driver import tvmc
    import numpy as np

    # Load model
    model = tvmc.load('model.onnx')

    # Prepare calibration data
    calibration_list = [
        {'input': tvm.nd.array(sample1)},
        {'input': tvm.nd.array(sample2)},
        # ... add more samples
    ]

    # Configure TIDL options
    delegate_options = {
        'artifacts_folder': './artifacts',
        'tensor_bits': 8,
        'deny_list': '',
        # For object detection models, add:
        # 'od_meta_arch_type': 3,  # SSD architecture
        # 'od_meta_layers_names_list': './model_metadata.prototxt',
    }

    # Compile with TIDL
    success = tidl_compile.compile_model(
        platform='am69a',
        compile_for_device=True,
        enable_tidl_offload=True,
        enable_c7x_codegen=True,
        delegate_options=delegate_options,
        calibration_input_list=calibration_list,
        mod=model.mod,
        params=model.params
    )

    if success:
        print("TIDL compilation completed successfully!")

