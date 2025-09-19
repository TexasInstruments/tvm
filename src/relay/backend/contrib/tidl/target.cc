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

#include <tvm/target/target.h>

namespace tvm {

TVM_REGISTER_TARGET_KIND("tidl", kDLCPU)
    .add_attr_option<String>("platform")
    .add_attr_option<String>("calibration_images")
    .add_attr_option<Integer>("calibration_frames")
    .add_attr_option<Array<FloatImm>>("input_mean")
    .add_attr_option<Array<FloatImm>>("input_scale")
    .add_attr_option<String>("artifacts_folder")
    .add_attr_option<Integer>("tensor_bits")
    .add_attr_option<Bool>("enable_offload")
    .add_attr_option<Bool>("enable_c7x_codegen")
    .add_attr_option<Bool>("compile_for_device")
    .add_attr_option<String>("deny_list")
    .add_attr_option<Integer>("od_meta_arch_type")
    .add_attr_option<String>("od_meta_layers_names_list");

}  // namespace tvm