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

TI's TVM releases are synchronized with TI's `Processor SDK RTOS <https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E>`_ releases.
The following table lists the PSDK RTOS releases and the corresponding TVM tag that is compatible with each one.

+--------------+--------------------------------------------------------------------------------+
| PSDK release | TVM release tag                                                                |
+==============+================================================================================+
| 8.6          | TIDL_PSDK_8.6.0                                                                |
+--------------+--------------------------------------------------------------------------------+
| 8.5          | TIDL_PSDK_8.5.0, TIDL_PSDK_8.5.1                                               |
+--------------+--------------------------------------------------------------------------------+
| 8.4          | TIDL_PSDK_8.4                                                                  |
+--------------+--------------------------------------------------------------------------------+
| 8.2          | TIDL_PSDK_8.2                                                                  |
+--------------+--------------------------------------------------------------------------------+
| 8.1          | TIDL_PSDK_8.1                                                                  |
+--------------+--------------------------------------------------------------------------------+
| 8.0          | TIDL_PSDK_8.0                                                                  |
+--------------+--------------------------------------------------------------------------------+
| 7.3          | TIDL_PSDK_7.3                                                                  |
+--------------+--------------------------------------------------------------------------------+

Prerequisites
-------------

PSDK RTOS
+++++++++

* Download and install the `Processor SDK RTOS <https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E>`_ release corresponding to the TVM release tag you plan to use.
* Set the PSDKR_PATH environment variable to point to the installation. For example: 

  .. code-block:: bash

     export PSDKR_PATH=/path/to/ti-processor-sdk-rtos-j721e-evm-08_06_zz_ww

Clang/LLVM
++++++++++

* Download and install `clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04 <https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz>`_ from the LLVM Git repository.

Arm GCC
+++++++

* If you are building the aarch64 TVM runtime and DLR packages, download and install the x86_64 Linux hosted cross compiler for AArch64 GNU/Linux from the `Arm GNU Toolchain download <https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/9-2-2019-12>`_ page.
* Set the ARM64_GCC_PATH environment variable to point to the installation directory. For example:

  .. code-block:: bash

     export ARM64_GCC_PATH=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

TVM compiler and runtime for x86_64
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


TVM runtime for AArch64
-----------------------

The TVM Runtime is an alternative to using the DLR for running inference. It provides C and Python APIs to load and run
models compiled by TVM. The steps below outline building just the TVM runtime for AArch64.

.. code-block:: bash

    mkdir build_aarch64; cd build_aarch64
    cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
    make clean; make runtime

Building DLR
==================

The Neo-AI Deep Learning Runtime (:term:`DLR`) is used for inference, that is, to load and run models compiled by TVM.
DLR can be built for x86_64 to enable host emulation, that is, to run a model with TIDL offload on a x86_64 PC. DLR can also be built
AArch64 and used for inference on the device.

x86_64 package
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


AArch64 package
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


Rebuilding C7x Firmware
========================

In some cases, you may want to rebuild the C7x firmware when using the TVM+TIDL flow.
For example to increase the number of subgraphs or the heap size as described below.
Please refer to the
`official user guide on rebuilding firmware <https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-j721s2/08_06_00_11/exports/docs/psdk_rtos/docs/user_guide/firmware_builder.html>`_
for more details.

Increase allowed number of TIDL subgraphs
-----------------------------------------

When using C7x to run the TIDL unsupported layers, the maximum number of TIDL subgraphs is 16
by default.  If you have a model that has more than 16 TIDL subgraphs, this limit can be increased
by rebuilding the TIDL library and the C7x firmware.  Information on the number of TIDL subgraphs
are available in the output messages shown during compilation.  For example,
"TIDL import of 41 Relay IR subgraphs succeeded."

First, set up the PSDK RTOS build.

.. code-block:: bash

    $ export PSDK_INSTALL_PATH=/path/to/ti-processor-sdk-rtos-<SOC>-evm-<xx>_<yy>_<zz>_<ww>
    $ cd ${PSDK_INSTALL_PATH}
    $ ./psdk_rtos/scripts/setup_psdk_rtos.sh

Next, edit `${PSDK_INSTALL_PATH}/tidl_<SOC>_<xx>_<yy>_<zz>_<ww>/ti_dl/inc/itidl_ti.h` header file.
Increase `TIDL_MAX_OBJECTS_PER_LEVEL` to match the number of TIDL subgraphs in your model.  Then
rebuild the `tidl_algo.lib` library.

.. code-block:: bash

    $ cd ${PSDK_INSTALL_PATH}/tidl_<SOC>_<xx>_<yy>_<zz>_<ww>
    $ vi ti_dl/inc/itidl_ti.h
    $ make tidl_algo

Finally, rebuild the C7x firmware and copy it to your EVM.

.. code-block:: bash

    $ cd ${PSDK_INSTALL_PATH}/vision_apps
    $ make firmware
    $ scp out/<SOC>/C7<XYZ>/FREERTOS/release/vx_app_rtos_linux_c7x_1.out root@evm:/lib/firmware/vision_apps_evm (or vision_apps_eaik)


.. note::

    Rebuilding C7x firmware alone does not work if layers unsupported by TIDL are executed on the Arm, because the maximum number of TIDL subgraphs is also subject to the OpenVX environment setup.


Increase allocated C7x DDR Heap Size
------------------------------------

The default C7x firmware comes with a default allocation for C7x local heap, which is used by TVM
runtime to run TIDL subgraphs and unsupported layers.  If your model requires more memory, you can
increase the allocated C7x local heap size and rebuild the C7x firmware.  Please refer to the
`official developer notes on updating SDK memory map <https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-j721s2/08_06_00_11/exports/docs/psdk_rtos/docs/user_guide/developer_notes_memory_map.html>`_
for details.

