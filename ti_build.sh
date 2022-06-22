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

CLANG_VERSION=clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04

BUILD_DIR=${WORKSPACE}/build

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

# Use TVM to compile a unit test
TEST_DIR=${WORKSPACE}/tests/python/relay/ti_tests/relay_mul
cd $TEST_DIR

# Create a Python virtual environment to install the TVM wheel file and build a test case
unset PYTHONPATH
python3 -m venv build_env && source ./build_env/bin/activate
export https_proxy=http://wwwgate.ti.com:80
export http_proxy=http://wwwgate.ti.com:80

# Update pip3 first
python3 -m pip install -U pip

# Install TVM in the venv
pip3 install ${WORKSPACE}/python/dist/tvm-*-cp36-cp36m-linux_x86_64.whl

# Install packages required to compile models with TVM
pip3 install graphviz
pip3 install tflite==2.4.0 onnx==1.9.0 mxnet==1.7.0.post2 gluoncv==0.8.0 torch==1.10.2 tensorflow==1.14.0 timm==0.5.4
pip3 install --no-deps torchvision==0.11.2

# Compile model on host, generate artifacts directory and copy to EVM
./relay_mul.py --compile --copy_to_evm ${EVM_IP}

# Run model on EVM
ssh root@${EVM_IP} './relay_mul.py --inference'
retval=$?
if [ $retval -eq 0 ]
then
    echo "Unit test PASSED"
else
    echo "Unit test FAILED"
fi
exit $retval

#ansible j7-evm -i ansible_evm.yml -u root -a "./relay_mul.py --inference"
