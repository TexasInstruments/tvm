.. _ti-tvm-gs-compilation:

################
Compiling Models
################

TI TVM supports two options for compiling models with TIDL offload. The distinction is based on where layers that are not supported
by TIDL are executed during inference:

#. Executing unsupported layers on Arm (See flow in Figure 1).
#. Executing unsupported layers on C7x (See flow in Figure 2). This option frees up the Arm device to run other aspects of the application. It can also improve overall inference performance by minimizing communication across the Arm and C7x.

The following figures show components added by TIDL and the TI TVM fork outlined in red.

+----------------+----------------------------------------------------------------------+
+ Model compilation with TIDL unsupported layers                                        +
+===========================================+===========================================+
+ Mapped to Arm                             + Mapped to C7x                             +
+-------------------------------------------+-------------------------------------------+
| .. figure:: ../images/TVM_Compile_Arm.png | .. figure:: ../images/TVM_Compile_C7x.png |
|    :scale: 30 %                           |    :scale: 30 %                           |
|    :align: center                         |    :align: center                         |
+-------------------------------------------+-------------------------------------------+


See :ref:`ti-tvm-compiling` for further details about TI TVM compilation.


.. _ti-tvm-gs-tidl-offload:

Compiling for TIDL Offload
==========================

TVM compilation is typically performed using a Python compilation script.  Example scripts
are provided in the `TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools>`_
Git repository. You can use these examples as templates to modify for your own use cases.
An overview of TI Open Source Runtime and compilation options are provided in the same repository.

The following Python functions are used in the examples provided by TI.

.. _ti-tvm-gs-compile-model:

Compile using python interface - compile_model
-----------------------------------------------

The ``compile_model`` function encapsulates the steps required to compile a model with TIDL offload. 

.. autofunction:: tvm.contrib.tidl.compile.compile_model

After a successful compile, the artifacts required to deploy the model are stored in the `artifacts_folder`.

.. _ti-tvm-gs-tvmc:

Compile using command line interface - TVMC
--------------------------------------------

TVMC can be used to compile a model using command line as follows:

.. code-block:: bash

    python -m tvm.driver.tvmc compile model.onnx \
      --target=tidl \
      --tidl-config config.yaml \
      --tidl-calibration-input calibration.npz \
      --enable-tidl-offload 1 \
      --compile-for-device 1 \
      --c7x-codegen 0 \
      --output ./artifacts/  

Note that this interface needs user to specify following files as input to the command:

* Configuration file (--tidl-config) - The configuration file is an optional YAML file which contains compile options for TIDL. YAML file should contain a single "compile_options" section with TIDL-specific options. Values specified in this file override the default values of respective options.

Example YAML configuration file:

.. code-block:: yaml
  
    compile_options:
      "debug_level": 1 \
      "tensor_bits": 8 \
      "advanced_options:calibration_frames": 2 \
      "advanced_options:calibration_iterations": 5 \
      "advanced_options:c7x_codegen": 1

* Calibration data (--tidl-calibration-input) - Refer :ref:`ti-tvm-compiling-calib` for details on calibration. TVMC expects an `npz` file containing data frames for calibration, so user is expected to convert image files into an `npz` file to be specified as input to TVMC command.

Example script to pack images into `npz` file for TVMC consumption.

.. code-block:: python
  
  # calib_dict - Dictionary of form - {'input_1': numpy array of N frames, 'input_2': numpy array of N frames} assuming N calibration frames \
  # npz_path - Path to 'npz' file to be created \
  import numpy as np \
  np.savez_compressed(npz_path, **calib_dict)

After a successful compile, the artifacts required to deploy the model are stored in the `artifacts_folder` specified via `--output` option.
