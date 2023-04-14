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

export PATH=${TVM_DEPS_PATH}/cmake-latest/bin:$PATH


function build_tvm {

    local BUILD_DIR=${WORKSPACE}/build
    local CLANG_VERSION=clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04

    # Build TVM
    rm -rf $BUILD_DIR
    mkdir $BUILD_DIR
    cd $BUILD_DIR
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="${TVM_DEPS_PATH}/$CLANG_VERSION/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
    make -j$(nproc)

    # Create the python package
    cd ../python
    python3 ./setup.py bdist_wheel
    cd -
}

# Build the Graph Executor for aarch64
function build_aarch64_ge {
    # ARM64_GCC_PATH, CGT7X_ROOT already set in the environment (e.g. Jenkins)
    local BUILD_DIR_AARCH64=${WORKSPACE}/build_aarch64

    rm -rf $BUILD_DIR_AARCH64
    mkdir $BUILD_DIR_AARCH64
    cd $BUILD_DIR_AARCH64

    cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/arm-tidl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
    make -j$(nproc) runtime

    # Initialize the ssh connection to EVM, save EVM into .known_hosts
    ssh-keygen -f "/home/sdomcbld/.ssh/known_hosts" -R "${EVM_IP}"
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
    make clean; make TVM_HOME=${WORKSPACE}
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

function setup_tvm_tidl_tests_common {
    # ARM64_GCC_PATH, CGT7X_ROOT already set in the environment (e.g. Jenkins)

    pip3 install pytest opencv-python
    mkdir -p ${WORKSPACE}/ti_tests_logs
    mkdir -p ${WORKSPACE}/tests/python/relay/ti_tests/testdata
    cp ~/.tvm_test_data/data/* ${WORKSPACE}/tests/python/relay/ti_tests/testdata
}

function setup_tvm_tidl_tests {
    platform=$1
    # Set up environment variables for TVM+TIDL compilation
    export TIDL_TOOLS_PATH=${TVM_DEPS_PATH}/tidl_tools/latest/${platform}/tidl_tools
    export EVM_IP=sdtocg-${platform,,}-0.hou.asp.ti.com  # ${platform,,} to lower case
    export LD_LIBRARY_PATH=${TIDL_TOOLS_PATH}

    # Initialize the ssh connection to EVM, save EVM into .known_hosts
    ssh-keygen -f "/home/sdomcbld/.ssh/known_hosts" -R "${EVM_IP}"
    OTHER_IP=`host ${EVM_IP} | cut -d' ' -f4`
    ssh-keygen -f "/home/sdomcbld/.ssh/known_hosts" -R "${OTHER_IP}"
    ssh -o "StrictHostKeyChecking no" root@${EVM_IP} 'uname -a'

    # Export workspace dir and mount it on EVM
    ssh root@${EVM_IP} 'mkdir -p /home/sdomcbld; mount -t nfs sdomc-build4.dhcp.ti.com:/home/sdomcbld /home/sdomcbld'
}

function run_tvm_tidl_unit_tests {
    platform=$1
    # Use TVM to compile a unit test
    local TEST_DIR=${WORKSPACE}/tests/python/relay/ti_tests/unit_tests
    cd $TEST_DIR

    # Run compilation tests on host
    python3 ./run_unit_tests.py --platform ${platform}
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run inference tests on EVM (export workspace dir and mount it on EVM)
    ssh root@${EVM_IP} "cd /home/sdomcbld/workspace/build-tvm-tidl/bem/neo-tvm/tests/python/relay/ti_tests/unit_tests; python3 ./run_unit_tests.py --platform ${platform}"
    retval=$?

    return $retval
}

function run_tvm_tidl_tests {
    platform=$1
    # Use TVM to compile a unit test
    local TEST_DIR=${WORKSPACE}/tests/python/relay/ti_tests
    cd $TEST_DIR

    # Run compilation tests
    python3 ./test_compile.py --platform ${platform}
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run inference tests on host
    python3 ./test_infer.py --tvm --platform ${platform}
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run inference tests on EVM (export workspace dir and mount it on EVM)
    ssh root@${EVM_IP} "cd /home/sdomcbld/workspace/build-tvm-tidl/bem/neo-tvm/tests/python/relay/ti_tests; python3 ./test_infer.py --platform ${platform}"
    retval=$?
    if [ $retval -ne 0 ]; then
        return $retval
    fi

    # Run dlr cpp tests on EVM
    ssh root@${EVM_IP} "cd /home/sdomcbld/workspace/build-tvm-tidl/bem/neo-tvm/tests/python/relay/ti_tests/test_dlr_cpp; python3 ./test_dlr_cpp.py ${platform}"
    retval=$?

    return $retval
}

function test_tvm_tidl {
    platform=$1
    run_tvm_tidl_unit_tests ${platform}
    retval=$?
    if [ $retval -eq 0 ]; then
        run_tvm_tidl_tests ${platform}
        retval=$?
    fi
    return $retval
}

echo "PLATFORMS=${PLATFORMS}"
echo "ARM64_GCC_PATH=${ARM64_GCC_PATH}"
echo "CGT7X_ROOT=${CGT7X_ROOT}"

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
    setup_tvm_tidl_tests_common

    # test each platform in a separate process in background, they can have different env vars
    for platform in ${PLATFORMS}; do
        setup_tvm_tidl_tests ${platform}
        test_tvm_tidl ${platform} > ${WORKSPACE}/ti_tests_logs/${platform}.log 2>&1 &
        declare pid_${platform}=$!
    done

    # wait for platform processes to finish
    for platform in ${PLATFORMS}; do
        pid="pid_${platform}"
        wait ${!pid}
        retval_tmp=$?
        if [ ${retval_tmp} -ne 0 ]; then
            echo "${platform} tests failed.  See ${platform}.log in artifacts/ti_tests_logs"
            retval=${retval_tmp}
        else
            echo "${platform} tests passed.  See ${platform}.log in artifacts/ti_tests_logs"
        fi
    done
fi

exit $retval
