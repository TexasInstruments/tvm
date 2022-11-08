############
Building TVM
############

.. note::

   These steps are required only if a user intends to modify the TI TVM package. Refer to :ref:`Getting-started` for instructions on using the prebuilt packages from TI.

The TVM `Install from Source page <https://tvm.apache.org/docs/install/from_source.html>`_ provides instructions on installing
the dependencies required for building TVM from source.

The sections below specify additional dependencies required to build TI's tidl-j7 branch.

.. note::

    The TI TVM package builds on Linux only. MacOS and Windows builds are not currently supported.

TI's TVM releases are synchronized with TI's `Processor SDK RTOS <https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E>`_ releases.
The following table lists the PSDK RTOS release and the corresponding TVM tag that is compatible with it.

+--------------+--------------------+-----------------------------------------------------------+
| PSDK release | TVM release tag    | Key features                                              |
+--------------+--------------------+-----------------------------------------------------------+
| 8.4          | TIDL_PSDK_8.4      | Bug fixes, J721S2 support, debug support                  |
+--------------+--------------------+-----------------------------------------------------------+
| 8.2          | TIDL_PSDK_8.2      | C7x code generation support                               |
+--------------+--------------------+-----------------------------------------------------------+
| 8.1          | TIDL_PSDK_8.1      |                                                           |
+--------------+--------------------+-----------------------------------------------------------+
| 8.0          | TIDL_PSDK_8.0      |                                                           |
+--------------+--------------------+-----------------------------------------------------------+
| 7.3          | TIDL_PSDK_7.3      | TIDL offload, unsupported layers run on Arm               |
+--------------+--------------------+-----------------------------------------------------------+

Prerequisites
-------------

PSDK RTOS
+++++++++

Download and install the `Processor SDK RTOS <https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E>`_ release corresponding to the TVM release tag and set PSDKR_PATH to point to the installation.
E.g.,

.. code-block:: bash

    export PSDKR_PATH=/path/to/ti-processor-sdk-rtos-j721e-evm-08_04_00_06

Clang/LLVM
++++++++++

Download and install clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04 from `LLVM github <https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz>`_.

Arm GCC
+++++++

If building the aarch64 TVM runtime and DLR packages, download and install the x86_64 Linux hosted cross compiler for AArch64 GNU/Linux from the `Arm GNU Toolchain download <https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/9-2-2019-12>`_ page.
Set ARM64_GCC_PATH to point to the installation directory. E.g.,

.. code-block:: bash

    export ARM64_GCC_PATH=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

TVM Compiler and Runtime for x86_64
-----------------------------------

The steps below outline building the TVM compiler and creating the Python package for x86_64.

.. code-block:: bash

    git clone https://github.com/TexasInstruments/tvm.git; cd tvm
    git checkout <corresponding_tag>
    git submodule update --init --recursive

    mkdir build_x86; cd build_x86
    cmake -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="/path/to/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
    make clean; make

    # Build python package in $TVM_HOME/python/dist
    cd ..; rm -fr build; ln -s build_x86 build
    cd python; python3 ./setup.py bdist_wheel; ls dist


.. note::

    Building the TVM compiler for AArch64 is not supported.


TVM Runtime for AArch64
-----------------------

The TVM Runtime is an alternative to using the DLR for running inference. It provides C and Python APIs to load and run
models compiled by TVM. The steps below outline building just the TVM runtime for AArch64.

.. code-block:: bash

    mkdir build_aarch64; cd build_aarch64
    cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
    make clean; make runtime

############
Building DLR
############

The Neo-AI-DLR (Deep Learning Runtime) is used for inference i.e. load and run models compiled by TVM.
DLR can be built for x86_64 to enable host emulation i.e. run a model with TIDL offload on a x86_64 PC. DLR can also be built
AArch64 and used for inference on the device.

x86_64 Package
--------------
.. code-block:: bash

    git clone https://github.com/TexasInstruments/neo-ai-dlr.git; cd neo-ai-dlr
    git checkout <corresponding_tag>
    git submodule update --init --recursive
    
    mkdir build_x86; cd build_x86
    cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DDLR_BUILD_TESTS=OFF ..
    make clean; make
    
    # Build python package in $DLR_HOME/python/dist
    cd ..; rm -f build; ln -s build_x86 build
    cd python; python3 ./setup.py bdist_wheel; ls dist


AArch64 Package
---------------
.. code-block:: bash

    git clone <this_repo>; cd neo-ai-dlr
    git checkout <corresponding_tag>
    git submodule update --init --recursive
    
    mkdir build_aarch64; cd build_aarch64
    cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DDLR_BUILD_TESTS=OFF -DCMAKE_TOOLCHAIN_FILE=../cmake/ti-aarch64-linux-gcc-toolchain.cmake ..
    make clean; make -j$(nproc)
    
    # build python package in $DLR_HOME/python/dist
    cd ..; rm -f build; ln -s build_aarch64 build
    cd python; python3 ./setup.py bdist_wheel; ls dist
