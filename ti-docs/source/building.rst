.. _ti-tvm-building:

#################
Building Packages
#################

Building TVM
==================

.. note::

   These steps described here are required only if you intend to modify the TI TVM package. Refer to :ref:`Getting-started` for instructions on using the prebuilt packages from TI.

The TVM `Install from Source <https://tvm.apache.org/docs/install/from_source.html>`_ page provides instructions on installing
the dependencies required for building TVM from source.

The sections below specify additional dependencies required to build TI's tidl-j7 branch.

.. note::

    The TI TVM package builds on Linux only. MacOS and Windows builds are not currently supported.

TI's TVM releases are synchronized with TI's Processor SDK releases.
Each Processor SDK release / Firmware Builder Release has a corresponding TVM release version tag compatible with it.

Prerequisites
-------------

edgeai-tidl-tools
++++++++++++++++++

* Setup edgeai-tidl-tools : Clone the `TI edgeai-tidl-tools <https://github.com/TexasInstruments/edgeai-tidl-tools>`_ repository and run the scripts/setup/setup.sh script.
* Run the scripts/setup/setup_env.sh script to set required environment variables.

PSDK RTOS
+++++++++

* Download and install the `Processor SDK RTOS <https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721S2>`_ release corresponding to the SoC and the TVM release tag you plan to use. This example points to J721S2 for demonstration.
* Set the PSDKR_PATH environment variable to point to the installation. For example: 

  .. code-block:: bash

     export PSDKR_PATH=/path/to/ti-processor-sdk-rtos-j721s2-evm-aa_bb_zz_ww

TVM compiler and runtime for x86_64
-----------------------------------

The steps below outline building the TVM compiler and creating the Python package for x86_64.
Note that building x86_64 wheel requires python version 3.10 (Users are adviced to create a python environment with this version for creation of wheel)

.. code-block:: bash

    git clone https://github.com/TexasInstruments/tvm.git; cd tvm
    git checkout <corresponding_tag>
    git submodule update --init --recursive

    mkdir build_x86; cd build_x86
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/c7x-mma-tidl/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DUSE_CGT7X_ROOT=${CGT7X_ROOT} ..
    make clean; make

    # Build python package in $TVM_HOME/python/dist
    cd ..; rm -fr build; ln -s build_x86 build
    cd python; python3 ./setup.py bdist_wheel; ls dist

TVM runtime for AArch64
-----------------------

The steps below outline building the TVM runtime for AArch64.
Note that building AArch64 wheel requires python version 3.12 (Users are adviced to create a python environment with this version for creation of wheel)

.. code-block:: bash

    mkdir build_aarch64; cd build_aarch64
    cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/c7x-mma-tidl/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake -DUSE_CGT7X_ROOT=${CGT7X_ROOT} -DUSE_LIBBACKTRACE=OFF -DUSE_ALTERNATIVE_LINKER=OFF ..
    make clean; make

    # Build python package in $TVM_HOME/python/dist
    cd ..; rm -fr build; ln -s build_aarch64 build
    cd python; python3 ./setup.py bdist_wheel --plat-name linux_aarch64; ls dist

