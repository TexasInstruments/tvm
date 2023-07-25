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

TVM_ROOT    ?= ../../../../..

ifeq (, $(shell which cl7x))
    CC=$(CGT7X_ROOT)/bin/cl7x
    CXX=$(CGT7X_ROOT)/bin/cl7x
    AR=$(CGT7X_ROOT)/bin/ar7x
    export C7X_C_DIR=$(CGT7X_ROOT)/include;$(CGT7X_ROOT)/lib
    STRIP=$(CGT7X_ROOT)/bin/strip7x
else
    CC=cl7x
    CXX=cl7x
    AR=ar7x
    STRIP=strip7x
endif

MKDIR_P = mkdir -p
RMDIR   = rm -fr
QUIET  ?= @

SILICON_VERSION ?= 7100
PKG_CFLAGS  = -o3 --abi=eabi --gcc --c11
CFLAGS += $(PKG_CFLAGS)

ifeq (7100_j784s4, $(SILICON_VERSION))
  CFLAGS += -mv7100
  CFLAGS += -DSOC_J784S4
else
  CFLAGS += -mv$(SILICON_VERSION)
  CFLAGS += -DSOC_J721E
endif
