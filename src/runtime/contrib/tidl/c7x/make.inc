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

SILICON_VERSION ?= 7100

QUIET    ?= @
TVM_ROOT ?= ../../../../..
ifeq ($(CGT7X_ROOT), )
    $(error CGT7X_ROOT is not set.  Please set to ti-cgt-c7000 compiler installation path)
endif

MKDIR_P = mkdir -p
RMDIR   = rm -fr


ifeq ($(TARGET_PLATFORM), PC)
### Begin host emulation setup

    CC=gcc-11
    CXX=g++-11
    AR=gcc-ar-11
    STRIP=strip
    OPTION_OUTFILE = -o
    BUILD_DIR_NAME = build_hostemu

    CFLAGS  = -O3 -std=c11 -D_XOPEN_SOURCE=500 -DHOST_EMULATION -D_FILE_OFFSET_BITS=64
    CFLAGS += -fPIC -fno-strict-aliasing
    CXXFLAGS = -std=c++14

    ifeq (7120_j784s4, $(SILICON_VERSION))
      CFLAGS += -I $(CGT7X_ROOT)/host_emulation/include/C7120/
      CFLAGS += -DSOC_J784S4
    else ifeq (7120_j721s2, $(SILICON_VERSION))
      CFLAGS += -I $(CGT7X_ROOT)/host_emulation/include/C7120/
      CFLAGS += -DSOC_J721S2
    else ifeq (7524, $(SILICON_VERSION))
      CFLAGS += -I $(CGT7X_ROOT)/host_emulation/include/C7524-MMA2_256/
      CFLAGS += -DSOC_J722S
    else ifeq (7504, $(SILICON_VERSION))
      CFLAGS += -I $(CGT7X_ROOT)/host_emulation/include/C7504/
      CFLAGS += -DSOC_AM62A
    else
      CFLAGS += -I $(CGT7X_ROOT)/host_emulation/include/C7100/
      CFLAGS += -DSOC_J721E
    endif

### End host emulation setup
else
### Begin c7x cross compilation setup

    CC=$(CGT7X_ROOT)/bin/cl7x
    CXX=$(CGT7X_ROOT)/bin/cl7x
    AR=$(CGT7X_ROOT)/bin/ar7x
    export C7X_C_DIR=$(CGT7X_ROOT)/include;$(CGT7X_ROOT)/lib
    STRIP=$(CGT7X_ROOT)/bin/strip7x
    OPTION_OUTFILE = --output_file
    BUILD_DIR_NAME = build

    PKG_CFLAGS  = -o3 --abi=eabi --gcc --c11
    CFLAGS += $(PKG_CFLAGS)
    CFLAGS += --auto_stream=no_saving --assume_addresses_ok_for_stream
    CXXFLAGS = --c++14

    # -mv* flag is to indicate which c7x tool chain to be used for compilation based on SoC
    # TVM build (for tidl_api) depends on c7x-mma-tidl/arm-tidl/rt/inc/itidl_ti.h
    #     which has structures with dependency on SoC hence need to correctly specify SoC flag

    ifeq (7120_j784s4, $(SILICON_VERSION))
      CFLAGS += -mv7120
      CFLAGS += -DSOC_J784S4
    else ifeq (7120_j721s2, $(SILICON_VERSION))
      CFLAGS += -mv7120
      CFLAGS += -DSOC_J721S2
    else ifeq (7524, $(SILICON_VERSION))
      CFLAGS += -mv7524
      CFLAGS += -DSOC_J722S
    else ifeq (7504, $(SILICON_VERSION))
      CFLAGS += -mv7504
      CFLAGS += -DSOC_AM62A
    else
      CFLAGS += -mv$(SILICON_VERSION)
      CFLAGS += -DSOC_J721E
    endif

### End c7x cross compilation setup
endif
