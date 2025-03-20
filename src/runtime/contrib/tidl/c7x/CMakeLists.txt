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

# Build TIDL C7x libs: TVM CRT lib, TIDL API lib, and C7x firmware symbols

if (USE_TIDL_PSDKR_PATH STREQUAL "none")
  message(FATAL_ERROR "Please set USE_TIDL_PSKDR_PATH to PSDK RTOS installation")
endif()

file(GLOB cgt7x_root_tmp ${USE_TIDL_PSDKR_PATH}/ti-cgt-c7000_*)
file(GLOB ivision_path_tmp ${USE_TIDL_PSDKR_PATH}/ivision)
file(GLOB pdk_path_tmp ${USE_TIDL_PSDKR_PATH}/pdk*)
file(GLOB mcu_plus_sdk_path_tmp ${USE_TIDL_PSDKR_PATH}/mcu_plus_sdk*)

list(LENGTH cgt7x_root_tmp cgt7x_root_len)
if (${cgt7x_root_len} EQUAL 0)
  file(GLOB cgt7x_root_tmp ${USE_CGT7X_ROOT})
endif()

list(LENGTH cgt7x_root_tmp cgt7x_root_len)
list(LENGTH ivision_path_tmp ivision_path_len)
list(LENGTH pdk_path_tmp pdk_path_len)
list(LENGTH mcu_plus_sdk_path_tmp mcu_plus_sdk_path_len)

if (${cgt7x_root_len} EQUAL 0)
  message(FATAL_ERROR "Cannot find C7000 code generation tools in ${USE_TIDL_PSDKR_PATH} or ${USE_CGT7X_ROOT}")
endif()

if ((${pdk_path_len} EQUAL 0) AND (${mcu_plus_sdk_path_len} EQUAL 0))
  message(FATAL_ERROR "Cannot find ivision, pdk/mcu_plus_sdk in ${USE_TIDL_PSDKR_PATH}")
endif()

list(GET cgt7x_root_tmp 0 cgt7x_root)
list(GET ivision_path_tmp 0 ivision_path)

# Jacinto and Sitara devices have a different structure for PDK tools
# Jacinto - pdk/packages   Sitara - mcu_plus_sdk/source
# We need to build libs for all SoCs independent of which PSDKR_PATH we use during build
# Hence, search in SDK to check if it has pdk or mcu_plus_sdk and set the pdk_path accordingly
# We only need one dmautils header for TVM build - so include the correct dmautils header path based on SoC
# We still need to maintain pdk_path since dmautils has some stray csl dependencies which are relative to the pdk path
if (NOT ${pdk_path_len} EQUAL 0)
  list(GET pdk_path_tmp 0 pdk_path_base)
  string(APPEND pdk_path "${pdk_path_base}/packages/")
  string(APPEND dmautils_path "${pdk_path_base}/packages/ti/drv/udma/dmautils/include/")
else()
  list(GET mcu_plus_sdk_path_tmp 0 pdk_path_base)
  string(APPEND pdk_path "${pdk_path_base}/source/")
  string(APPEND dmautils_path "${pdk_path_base}/source/drivers/dmautils/include/")
endif()

list(APPEND build_env_paths "USE_TIDL_RT_PATH=${USE_TIDL_RT_PATH}")
list(APPEND build_env_paths "CGT7X_ROOT=${cgt7x_root}")
list(APPEND build_env_paths "IVISION_PATH=${ivision_path}")
list(APPEND build_env_paths "PDK_PATH=${pdk_path}")
list(APPEND build_env_paths "DMAUTILS_PATH=${dmautils_path}")

message("-- Build with contrib.tidl.c7x libs, path= ${build_env_paths}")

# tvm_crt and dsp_syms are SoC agnostic, however the compiler tool chain used to build them has
# to be compatible with the one to be used by Makefile.c7x_mod as part of the model compilation
# Hence, they still need to be added as part of separate targets with different Si versions

add_custom_target(tidl_c7x_libs_j721e
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100 -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100 -f Makefile.tidl_api 
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_j721s2
                  COMMAND make ${build_env_paths} SILICON_VERSION=7120_j721s2 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7120_j721s2 -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7120_j721s2 -f Makefile.tidl_api
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_j784s4
                  COMMAND make ${build_env_paths} SILICON_VERSION=7120_j784s4 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7120_j784s4 -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7120_j784s4 -f Makefile.tidl_api
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_am62a
                  COMMAND make ${build_env_paths} SILICON_VERSION=7504 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7504 -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7504 -f Makefile.tidl_api
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_j722s
                  COMMAND make ${build_env_paths} SILICON_VERSION=7524 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7524 -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7524 -f Makefile.tidl_api
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

# works with tvm top-level "make clean"
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j721s2/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j721s2/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j721s2/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j721s2/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j721s2/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j721s2/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j784s4/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j784s4/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j784s4/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j784s4/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j784s4/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7120_j784s4/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524/)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clean_files})

