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

CLANG_VERSION=clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04

BUILD_DIR=${WORKSPACE}/build

# Build TVM
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM=${TVM_DEPS_PATH}/$CLANG_VERSION/bin/llvm-config -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
make -j$(nproc)

# Create the python package
cd ../python
python3 ./setup.py bdist_wheel
