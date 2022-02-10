/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_APPS_BUNDLE_DEPLOY_BUNDLE_H_
#define TVM_APPS_BUNDLE_DEPLOY_BUNDLE_H_

#include <tvm/runtime/c_runtime_api.h>

TVM_DLL void* tvm_runtime_create(const char* json_data, const char* params_data,
                                 const uint64_t params_size);

TVM_DLL int tvm_runtime_destroy(void* runtime);

TVM_DLL int tvm_runtime_set_input(void* runtime, const char* name, DLTensor* tensor);

TVM_DLL int tvm_runtime_set_input_raw(void* runtime, const char* name, void* tensor_raw);

TVM_DLL int tvm_runtime_run(void* runtime);

TVM_DLL int tvm_runtime_get_output(void* runtime, int32_t index, DLTensor* tensor);

TVM_DLL int tvm_runtime_get_output_raw(void* runtime, int32_t index, void* tensor_raw);

#endif /* TVM_APPS_BUNDLE_DEPLOY_BUNDLE_H_ */
