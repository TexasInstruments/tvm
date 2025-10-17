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

    # Basic TIDL compilation with required preprocessing parameters
    tvmc compile model.onnx \
        --target tidl \
        --target-tidl-platform am68a \
        --target-tidl-input-mean 123.675 116.28 103.53 \
        --target-tidl-input-scale 0.017125 0.017507 0.017429 \
        --target-tidl-enable-offload \
        --target-tidl-tensor-bits 8

TVMC TIDL Options
-----------------

Required Options
~~~~~~~~~~~~~~~~

* ``--target-tidl-platform`` - Target TI platform (am68pa, am68a, am69a, am67a, am62a)
* ``--target-tidl-input-mean`` - Input mean values for RGB channels [R, G, B] (required)
* ``--target-tidl-input-scale`` - Input scale values for RGB channels [R, G, B] (required)

Acceleration Options
~~~~~~~~~~~~~~~~~~~

* ``--target-tidl-enable-offload`` - Enable TIDL acceleration for supported layers
* ``--target-tidl-enable-c7x-codegen`` - Generate C7x code for unsupported layers (instead of ARM)

Quantization Options
~~~~~~~~~~~~~~~~~~~

* ``--target-tidl-tensor-bits`` - Quantization precision: 8, 16, or 32 bits (default: 8)
* ``--target-tidl-calibration-data`` - Path to pickle file containing calibration data (graph_input_list)

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

Preprocessing Parameters
------------------------

TIDL requires input preprocessing parameters (mean and scale values) for proper model compilation.
These values depend on how your model was trained.

Common Preprocessing Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ImageNet models** (ResNet, MobileNet, etc.):

* Mean: ``123.675 116.28 103.53``
* Scale: ``0.017125 0.017507 0.017429``

**Detection models** (SSD, YOLO, etc.):

* Mean: ``0 0 0``
* Scale: ``0.003921568627 0.003921568627 0.003921568627``

**TFLite MobileNet models**:

* Mean: ``127.5 127.5 127.5``
* Scale: ``0.007874 0.007874 0.007874`` (1/127.5)

.. note::
   Calibration data can be provided via ``--target-tidl-calibration-data`` option by
   passing a path to a pickle file containing the calibration samples. The pickle file
   should contain a list of dictionaries where each dictionary maps input names to
   numpy arrays (graph_input_list format).

   Alternatively, calibration data can be provided through the Python API.

Complete Examples
-----------------

Here's a complete example compiling a ResNet-50 model for AM68A with 8-bit quantization:

.. code-block:: bash

    # Compile ResNet-50 with TIDL
    tvmc compile path/to/resnet50.onnx \
        --target tidl \
        --target-tidl-platform am68a \
        --target-tidl-input-mean 123.675 116.28 103.53 \
        --target-tidl-input-scale 0.017125 0.017507 0.017429 \
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
        --target-tidl-platform am68a \
        --target-tidl-input-mean 0 0 0 \
        --target-tidl-input-scale 0.003921568627 0.003921568627 0.003921568627 \
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

Using Calibration Data with TVMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have pre-generated calibration data, you can pass it via the ``--target-tidl-calibration-data`` option:

.. code-block:: bash

    # Compile model with pre-generated calibration data
    tvmc compile path/to/model.onnx \
        --target tidl \
        --target-tidl-platform am68a \
        --target-tidl-input-mean 123.675 116.28 103.53 \
        --target-tidl-input-scale 0.017125 0.017507 0.017429 \
        --target-tidl-artifacts-folder ./artifacts \
        --target-tidl-tensor-bits 8 \
        --target-tidl-enable-offload \
        --target-tidl-calibration-data ./calibration_data.pkl

The calibration data pickle file should be prepared using Python:

.. code-block:: python

    import pickle
    import numpy as np

    # Prepare calibration samples (e.g., 10-50 representative images)
    graph_input_list = []
    for sample in calibration_samples:
        # Each sample is a dictionary mapping input names to numpy arrays
        graph_input_list.append({
            'input': sample  # Replace 'input' with your model's input name
        })

    # Save to pickle file
    with open('calibration_data.pkl', 'wb') as f:
        pickle.dump(graph_input_list, f)

Programming Interface
--------------------

For programmatic access, use the Python API directly:

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
        platform='am68a',
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

