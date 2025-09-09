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

* TI Processor SDK RTOS installation
* TIDL tools (edgeai-tidl-tools) properly configured
* Target hardware/EVM setup
* Model in supported format (ONNX, TensorFlow Lite, etc.)

For detailed installation instructions, refer to the `TI TVM User's Guide <https://software-dl.ti.com/codegen/docs/tvm/tvm_tidl_users_guide/building.html>`_.

Basic Usage with TVMC
---------------------

The simplest way to compile models for TIDL is using TVM's command-line interface (TVMC):

.. code-block:: bash

    # Basic TIDL compilation
    tvmc compile model.onnx \
        --target tidl \
        --tidl-platform am69a \
        --tidl-enable-offload

    # With calibration data for quantization
    tvmc compile model.onnx \
        --target tidl \
        --tidl-platform am69a \
        --tidl-calibration-data calibration.npz \
        --tidl-enable-offload \
        --tidl-tensor-bits 8

TVMC TIDL Options
-----------------

Required Options
~~~~~~~~~~~~~~~~

* ``--tidl-platform`` - Target TI platform (am68pa, am68a, am69a, am67a, am62a)

Acceleration Options
~~~~~~~~~~~~~~~~~~~

* ``--tidl-enable-offload`` - Enable TIDL acceleration for supported layers
* ``--tidl-enable-c7x-codegen`` - Generate C7x code for unsupported layers (instead of ARM)

Quantization Options
~~~~~~~~~~~~~~~~~~~

* ``--tidl-calibration-data`` - Path to .npz calibration file for quantization
* ``--tidl-tensor-bits`` - Quantization precision: 8, 16, or 32 bits (default: 8)

Output and Build Options
~~~~~~~~~~~~~~~~~~~~~~~~

* ``--tidl-artifacts-folder`` - Output directory for compilation artifacts (default: ./tidl_artifacts)
* ``--tidl-compile-for-device`` - Compile for target device (aarch64) vs host (x86)

Advanced Options
~~~~~~~~~~~~~~~

* ``--tidl-deny-list`` - Comma-separated list of operations to exclude from TIDL offloading

Preparing Calibration Data
---------------------------

For optimal quantization performance, provide representative calibration data as a NumPy .npz archive:

.. code-block:: python

    import numpy as np

    # Example: Create calibration data with multiple samples
    # Keys should match your model's input tensor names
    calibration_data = {
        'input_1': np.random.randn(10, 3, 224, 224).astype(np.float32),  # 10 samples
        'input_2': np.random.randn(10, 100).astype(np.float32)           # if multi-input
    }

    # Save as NPZ file
    np.savez('calibration_data.npz', **calibration_data)

The calibration data should contain multiple representative samples (typically 10-100) that cover
the expected input distribution for your model.

Complete Example
---------------

Here's a complete example compiling a ResNet-50 model for AM69A with 8-bit quantization:

.. code-block:: bash

    # Compile ResNet-50 with TIDL
    tvmc compile resnet50.onnx \
        --target tidl \
        --tidl-platform am69a \
        --tidl-calibration-data calibration.npz \
        --tidl-artifacts-folder ./resnet50_artifacts \
        --tidl-tensor-bits 8 \
        --tidl-enable-offload \
        --tidl-enable-c7x-codegen \
        --tidl-compile-for-device

    # Compilation artifacts will be generated in ./resnet50_artifacts/

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

Performance Considerations
-------------------------

* **Quantization**: 8-bit typically provides the best performance vs accuracy trade-off
* **Layer Support**: Check TIDL documentation for supported operations list
* **Memory Layout**: TIDL prefers NCHW layout for optimal performance
* **Batch Size**: Single batch inference is typically optimal for edge deployment
* **Calibration Quality**: Use diverse, representative calibration data for best quantization results

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~~

**"Platform not supported" error**
    Ensure ``--tidl-platform`` matches your target hardware exactly

**"Calibration file not found" error**
    Verify the path to your .npz file and ensure it exists and is readable

**"No layers offloaded to TIDL" warning**
    Your model may not contain TIDL-supported operations. Check the model architecture
    against TIDL's supported operator list in the TI documentation.

**Compilation fails with "missing tools" error**
    Verify your TI Processor SDK installation and environment setup are correct

Debug Strategies
~~~~~~~~~~~~~~~

* Use ``--tidl-deny-list`` to exclude specific problematic operations and narrow down issues
* Check the generated artifacts in the output folder for detailed compilation logs
* Verify calibration data shapes match your model's expected input dimensions
* Start with ``--tidl-tensor-bits 32`` to isolate quantization-related issues

For comprehensive documentation and troubleshooting, refer to the official
`TI TVM User's Guide <https://software-dl.ti.com/codegen/docs/tvm/tvm_tidl_users_guide/index.html>`_.
