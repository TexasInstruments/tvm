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

/*!
 * \file tvm/runtime/crt_runtime_api.h
 * \brief TVM CRT (C Runtime) API functions.
 *
 * This header provides CRT-specific versions of TVM runtime API functions
 * to avoid naming conflicts with the C++ runtime.
 */
#ifndef TVM_RUNTIME_CRT_RUNTIME_API_H_
#define TVM_RUNTIME_CRT_RUNTIME_API_H_

#include <tvm/runtime/c_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Get the last error message from CRT runtime.
 *
 * \return The error message, or null if no error.
 */
TVM_DLL const char* CRT_TVMGetLastError(void);

/*!
 * \brief Register a global function in CRT runtime.
 *
 * \param name The name of the function.
 * \param f The function handle.
 * \param override Whether to override existing function.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int CRT_TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override);

/*!
 * \brief Get a global function from CRT runtime.
 *
 * \param name The name of the function.
 * \param out The result function handle.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int CRT_TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out);

/*!
 * \brief Call a packed function in CRT runtime.
 *
 * \param func_handle The function handle.
 * \param arg_values The argument values.
 * \param type_codes The type codes of the arguments.
 * \param num_args The number of arguments.
 * \param ret_val The return value.
 * \param ret_type_code The return type code.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int CRT_TVMFuncCall(TVMFunctionHandle func_handle, TVMValue* arg_values, int* type_codes,
                            int num_args, TVMValue* ret_val, int* ret_type_code);

/*!
 * \brief Get a function from module in CRT runtime.
 *
 * \param mod The module handle.
 * \param func_name The function name.
 * \param query_imports Whether to query imports.
 * \param out The result function handle.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int CRT_TVMModGetFunction(TVMModuleHandle mod, const char* func_name, int query_imports,
                                  TVMFunctionHandle* out);

/*!
 * \brief Allocate workspace in CRT runtime.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param nbytes Number of bytes to allocate.
 * \param dtype_code_hint Type code hint.
 * \param dtype_bits_hint Type bits hint.
 * \return Pointer to allocated workspace, or nullptr on failure.
 */
TVM_DLL void* CRT_TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                           int dtype_code_hint, int dtype_bits_hint);

/*!
 * \brief Free workspace in CRT runtime.
 *
 * \param device_type The device type.
 * \param device_id The device id.
 * \param ptr Pointer to workspace to free.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int CRT_TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_RUNTIME_API_H_