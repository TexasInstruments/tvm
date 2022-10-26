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


Building the x86_64 TVM package
--------------------------------

.. code-block:: bash

    # download and install corresponding PSDK_RTOS to <PSDKR_PATH>
    # download and install clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04
    # https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

    git clone https://github.com/TexasInstruments/tvm.git; cd tvm
    git checkout <corresponding_tag>
    git submodule update --init --recursive

    mkdir build_x86; cd build_x86
    cmake -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="/path/to/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
    make clean; make -j$(nproc)

    # build python package in $TVM_HOME/python/dist
    cd ..; rm -fr build; ln -s build_x86 build
    cd python; python3 ./setup.py bdist_wheel; ls dist


Building the TVM Graph Executor (Runtime) for aarch64
-----------------------------------------------------
.. code-block:: bash

    export ARM64_GCC_PATH=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

    mkdir build_aarch64; cd build_aarch64
    cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
    make clean; make -j$(nproc) runtime
