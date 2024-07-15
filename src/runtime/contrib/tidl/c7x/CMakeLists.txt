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

list(LENGTH cgt7x_root_tmp cgt7x_root_len)
if (${cgt7x_root_len} EQUAL 0)
  file(GLOB cgt7x_root_tmp ${USE_CGT7X_ROOT})
endif()

list(LENGTH cgt7x_root_tmp cgt7x_root_len)
list(LENGTH ivision_path_tmp ivision_path_len)
list(LENGTH pdk_path_tmp pdk_path_len)

if (${cgt7x_root_len} EQUAL 0)
  message(FATAL_ERROR "Cannot find C7000 code generation tools in ${USE_TIDL_PSDKR_PATH} or ${USE_CGT7X_ROOT}")
endif()

if ((${pdk_path_len} EQUAL 0) OR (${pdk_path_len} EQUAL 0))
  message(FATAL_ERROR "Cannot find ivision, pdk* in ${USE_TIDL_PSDKR_PATH}")
endif()

list(GET cgt7x_root_tmp 0 cgt7x_root)
list(GET ivision_path_tmp 0 ivision_path)
list(GET pdk_path_tmp 0 pdk_path)

list(APPEND build_env_paths "USE_TIDL_RT_PATH=${USE_TIDL_RT_PATH}")
list(APPEND build_env_paths "CGT7X_ROOT=${cgt7x_root}")
list(APPEND build_env_paths "IVISION_PATH=${ivision_path}")
list(APPEND build_env_paths "PDK_PATH=${pdk_path}")

message("-- Build with contrib.tidl.c7x libs, path= ${build_env_paths}")

add_custom_target(tidl_c7x_libs
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100 -f Makefile.dsp_syms 
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100 -f Makefile.tidl_api 
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_j784s4
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100_j784s4 -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100_j784s4 -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7100_j784s4 -f Makefile.tidl_api
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_j722s
                  COMMAND make ${build_env_paths} SILICON_VERSION=7524_j722s -f Makefile.tvm_crt
                  COMMAND make ${build_env_paths} SILICON_VERSION=7524_j722s -f Makefile.dsp_syms
                  COMMAND make ${build_env_paths} SILICON_VERSION=7524_j722s -f Makefile.tidl_api
                  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                 )

add_custom_target(tidl_c7x_libs_mv7504
                 COMMAND make ${build_env_paths} SILICON_VERSION=7504 -f Makefile.tvm_crt 
                 COMMAND make ${build_env_paths} SILICON_VERSION=7504 -f Makefile.dsp_syms 
                 COMMAND make ${build_env_paths} SILICON_VERSION=7504 -f Makefile.tidl_api 
                 WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                )


# works with tvm top-level "make clean"
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100_j784s4/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100_j784s4/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100_j784s4/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100_j784s4/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100_j784s4/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7100_j784s4/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524_j722s/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524_j722s/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524_j722s/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524_j722s/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524_j722s/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7524_j722s/)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/build)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/dsp_syms.out)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/dsp_syms.out.obj)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/tvm_crt.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/tidl_api.lib)
list(APPEND clean_files ${CMAKE_CURRENT_LIST_DIR}/7504/)


set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${clean_files})

