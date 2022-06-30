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
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>


#include <string>
#include <fstream>
#include <cstdio>
#include <iostream>


void DeployGraphExecutor() {
  const std::string artifacts_folder("artifacts_relay_mul_c7x_target/");

  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module loaded_lib = tvm::runtime::Module::LoadFromFile(artifacts_folder + "deploy_lib.so");

    // Load JSON
  std::ifstream loaded_json(artifacts_folder + "deploy_graph.json");
  std::string json_data((std::istreambuf_iterator<char>(loaded_json)), std::istreambuf_iterator<char>());
  loaded_json.close();

  // Load params from file
  std::ifstream loaded_params(artifacts_folder + "deploy_param.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(loaded_params)), std::istreambuf_iterator<char>());
  loaded_params.close();
  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  LOG(INFO) << "Creating graph executor...";
  // Create the graph executor module
  int device_type = dev.device_type; // Need an int, the DLDeviceType enum 
                                     // results in an ambiguity for TVMArgsSetter.

  tvm::runtime::Module mod = 
    (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data,
                                                                loaded_lib,
                                                                device_type,
                                                                dev.device_id);

  // Load params into Graph Executor
  LOG(INFO) << "Loading params ...";
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  tvm::runtime::PackedFunc set_input           = mod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output          = mod.GetFunction("get_output");
  tvm::runtime::PackedFunc run                 = mod.GetFunction("run");

  LOG(INFO) << "Initializing inputs ...";
  auto f32 = tvm::runtime::DataType::Float(32);
  tvm::runtime::NDArray a = tvm::runtime::NDArray::Empty({672, 14, 14}, f32, dev);
  tvm::runtime::NDArray b = tvm::runtime::NDArray::Empty({672, 1, 1},   f32, dev);
  tvm::runtime::NDArray c = tvm::runtime::NDArray::Empty({672, 14, 14}, f32, dev);

  for (int i = 0; i < 672; ++i)
    static_cast<float*>(b->data)[i] = 4;

  for (int i = 0; i < 672*14*14; ++i)
    static_cast<float*>(a->data)[i] = i;

  set_input("a", a);
  set_input("b", b);

  // run the code
  LOG(INFO) << "Running ...";
  run();

  // get the output
  get_output(0, c);

  for (int i = 0; i < 672*14*14; ++i)
    ICHECK_EQ(static_cast<float*>(c->data)[i], i * 4);

  LOG(INFO) << "Pass";
}

int main(void) {
  DeployGraphExecutor();
  return 0;
}
