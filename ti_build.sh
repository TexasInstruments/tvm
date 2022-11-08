#!/usr/bin/env bash

set -x
set -e

if [ -z "${PSDKR_PATH}" ]; then
    echo "You must define PSDKR_PATH before calling this script"
    exit 1
fi

if [ -z "${TVM_DEPS_PATH}" ]; then
    echo "You must define TVM_DEPS_PATH before calling this script"
    exit 1
fi

if [ -z "${WORKSPACE}" ]; then
    echo "You must define WORKSPACE before calling this script"
    exit 1
fi

if [ -z "${EVM_IP}" ]; then
    echo "You must define EVM_IP before calling this script"
    exit 1
fi


function build_tvm {

    local BUILD_DIR=${WORKSPACE}/build
    local CLANG_VERSION=clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04

    # Build TVM
    rm -rf $BUILD_DIR
    mkdir $BUILD_DIR
    cd $BUILD_DIR
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="${TVM_DEPS_PATH}/$CLANG_VERSION/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
    make -j$(nproc)

    # Create the python package
    cd ../python
    python3 ./setup.py bdist_wheel
    cd -
}

# Build the Graph Executor for aarch64
function build_aarch64_ge {
    local BUILD_DIR_AARCH64=${WORKSPACE}/build_aarch64
    local GCC_VERSION=gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

    export ARM64_GCC_PATH=${TVM_DEPS_PATH}/$GCC_VERSION

    rm -rf $BUILD_DIR_AARCH64
    mkdir $BUILD_DIR_AARCH64
    cd $BUILD_DIR_AARCH64

    cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
    make -j$(nproc) runtime

    # Initialize the ssh connection to EVM, save EVM into .known_hosts
    ssh -o "StrictHostKeyChecking no" root@${EVM_IP} 'uname -a'
    scp libtvm_runtime.so root@${EVM_IP}:
    cd -
}

function test_tvm {
    # Use TVM to compile a unit test
    local TEST_DIR=${WORKSPACE}/tests/python/relay/ti_tests/relay_mul
    cd $TEST_DIR

    # Create a Python virtual environment to install the TVM wheel file and build a test case
    unset PYTHONPATH
    python3 -m venv build_env && source ./build_env/bin/activate
    export https_proxy=http://wwwgate.ti.com:80
    export http_proxy=http://wwwgate.ti.com:80
    export no_proxy=ti.com

    # Update pip3 first
    python3 -m pip install -U pip

    # Install TVM in the venv
    pip3 install ${WORKSPACE}/python/dist/tvm-*-cp36-cp36m-linux_x86_64.whl

    # Install packages required to compile models with TVM
    pip3 install graphviz
    pip3 install tflite==2.4.0 onnx==1.9.0 mxnet==1.7.0.post2 gluoncv==0.8.0 torch==1.10.2 tensorflow==1.14.0 timm==0.5.4
    pip3 install --no-deps torchvision==0.11.2

    # Required for building docs
    pip3 install sphinx

    # Initialize the ssh connection to EVM, save EVM into .known_hosts
    ssh -o "StrictHostKeyChecking no" root@${EVM_IP} 'uname -a'

    # Compile model on host, generate artifacts directory and copy to EVM
    ./relay_mul.py --compile --copy_to_evm ${EVM_IP}

    # Run model on EVM
    ssh root@${EVM_IP} './relay_mul.py --inference'
    local retval=$?

    if [ $retval -eq 0 ]
    then
        echo "Unit test PASSED"
    else
        echo "Unit test FAILED"
        return $retval
    fi

    # Test C++ Graph Executor
    make clean; make
    retval=$?
    if [ $retval -ne 0 ]
    then
        return $retval
    fi
    scp -rq artifacts_relay_mul_c7x_target relay_mul root@${EVM_IP}:
    ssh root@${EVM_IP} 'LD_LIBRARY_PATH=. ./relay_mul'
    if [ $retval -ne 0 ]
    then
        return $retval
    fi

    # Build docs
    local DOCS_DIR=${WORKSPACE}/ti-docs
    cd $DOCS_DIR
    make clean; make
    if [ $retval -ne 0 ]
    then
        return $retval
    fi

    return $retval
}

function setup_tvm_tidl_tests {
    # Set up environment variables for TVM+TIDL compilation
    # ARM64_GCC_PATH already set in build_aarch64_ge
    export TIDL_TOOLS_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/tidl_tools)
    # In PSDK 8.4, TI C7x compiler 3.0.0.STS contains a known bug that generates an illegal
    # VSUBSP instruction on scalar A side registers.  It is fixed in a future release.
    # For now, use 2.1.1.LTS in the previous PSDK 8.2.
    export CGT7X_ROOT=$(ls -d ${PSDKR_PATH}/../ti-processor-sdk-rtos-j721e-evm-08_02_00_05/ti-cgt-c7000_*)
    pip3 install pytest opencv-python

    # Export workspace dir and mount it on EVM
    ssh root@${EVM_IP} 'mkdir -p /home/sdomcbld; mount -t nfs sdomc-build4.dhcp.ti.com:/home/sdomcbld /home/sdomcbld'
}

function run_tvm_tidl_unit_tests {
    # Use TVM to compile a unit test
    local TEST_DIR=${WORKSPACE}/tests/python/relay/ti_tests/unit_tests
    cd $TEST_DIR

    # Run compilation tests on host
    python3 ./run_unit_tests.py
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run inference tests on EVM (export workspace dir and mount it on EVM)
    ssh root@${EVM_IP} 'cd /home/sdomcbld/workspace/build-tvm-tidl/bem/neo-tvm/tests/python/relay/ti_tests/unit_tests; python3 ./run_unit_tests.py'
    retval=$?

    return $retval
}

function run_tvm_tidl_tests {
    # Use TVM to compile a unit test
    local TEST_DIR=${WORKSPACE}/tests/python/relay/ti_tests
    cd $TEST_DIR

    # Run compilation tests
    python3 ./test_compile.py
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run inference tests on host
    mkdir -p testdata
    cp ~/.tvm_test_data/data/* testdata
    python3 ./test_infer.py --tvm
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run inference tests on EVM (export workspace dir and mount it on EVM)
    ssh root@${EVM_IP} 'cd /home/sdomcbld/workspace/build-tvm-tidl/bem/neo-tvm/tests/python/relay/ti_tests; python3 ./test_infer.py'
    retval=$?

    return $retval
}


build_tvm
build_aarch64_ge

# Run test_tvm and exit with return value from test_tvm
test_tvm
retval=$?

# If VALIDATE is set to 0, skip running tests
if [ "$VALIDATE" -eq "0" ]; then
    exit $retval
fi

# If test_tvm succeeds, run TVM+TIDL tests in tests/python/relay/ti_tests
if [ $retval -eq 0 ]; then
    setup_tvm_tidl_tests
    run_tvm_tidl_unit_tests
    retval=$?
    if [ $retval -eq 0 ]; then
        run_tvm_tidl_tests
        retval=$?
    fi
fi

exit $retval
