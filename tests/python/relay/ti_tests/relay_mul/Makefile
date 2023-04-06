# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Makefile to build Graph Executor version of relay_mul

CXX=${ARM64_GCC_PATH}/bin/aarch64-none-linux-gnu-g++

DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++17 -g -fPIC\
	-I${TVM_HOME}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_HOME}/3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>


LIB_PATH=${TVM_HOME}/build_aarch64
PKG_LDFLAGS = -L ${LIB_PATH} -ldl -pthread

.PHONY: clean all

all: relay_mul

# Create the relay_mul shared object
artifacts_relay_mul_c7x_target/deploy_lib.so: relay_mul.py
	@mkdir -p $(@D)
	python3 relay_mul.py --compile

# Deploy using libtvm_runtime.so
relay_mul: relay_mul.cc artifacts_relay_mul_c7x_target/deploy_lib.so
	$(CXX) $(PKG_CFLAGS)  -o $@  $^ -l tvm_runtime $(PKG_LDFLAGS)

clean:
	rm -rf relay_mul artifacts_relay_mul_c7x_target
