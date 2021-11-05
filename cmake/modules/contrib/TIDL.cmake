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

if(USE_TIDL STREQUAL "ON")
  message(STATUS "Build with contrib.tidl, path=" ${USE_TIDL_RT_PATH})
  if (USE_TIDL_RT_PATH STREQUAL "none")
    message(FATAL_ERROR "USE_TIDL_RT_PATH not set, J7 runtime support will not be available")
  else()
    include_directories(${USE_TIDL_RT_PATH}/inc)
  endif()

  file(GLOB TIDL_RELAY_CONTRIB_SRC src/relay/backend/contrib/tidl/*.cc)
  list(APPEND COMPILER_SRCS ${TIDL_RELAY_CONTRIB_SRC})

  file(GLOB TIDL_CONTRIB_SRC src/runtime/contrib/tidl/*.cc)
  list(APPEND RUNTIME_SRCS ${TIDL_CONTRIB_SRC})

  # Only build c7x in tvm project (compilation), not in dlr project (runtime)
  # c7x is used in TVM compilation only.  Do not include it in neo-ai-dlr build.
  # neo-ai-dlr CMakeLists.txt simply include top-level TVM CMakeLists.txt,
  # so we do a check here on the project name.  This is to simplify neo-ai-dlr
  # build, so that it does not require USE_TIDL_PSDKR_PATH.
  if (CMAKE_PROJECT_NAME STREQUAL "tvm")
    add_subdirectory("src/runtime/contrib/tidl/c7x")
  endif()
endif()

