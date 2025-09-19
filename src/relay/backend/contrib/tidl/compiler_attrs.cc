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
#include "compiler_attrs.h"

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
namespace relay {
namespace contrib {
namespace tidl {

TVM_REGISTER_NODE_TYPE(TIDLCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.tidl.options", TIDLCompilerConfig);

Target CreateTarget(const tvm::transform::PassContext& ctx) {
  auto cfg = ctx->GetConfig<TIDLCompilerConfig>("relay.ext.tidl.options");
  if (!cfg.defined()) {
    return Target("tidl");
  }

  String platform = cfg.value()->platform;
  String calibration_images = cfg.value()->calibration_images;
  Integer calibration_frames = cfg.value()->calibration_frames;
  Array<FloatImm> input_mean = cfg.value()->input_mean;
  Array<FloatImm> input_scale = cfg.value()->input_scale;
  String artifacts_folder = cfg.value()->artifacts_folder;
  Integer tensor_bits = cfg.value()->tensor_bits;
  runtime::Bool enable_offload = cfg.value()->enable_offload->value;
  runtime::Bool enable_c7x_codegen = cfg.value()->enable_c7x_codegen->value;
  runtime::Bool compile_for_device = cfg.value()->compile_for_device->value;
  String deny_list = cfg.value()->deny_list;
  Integer od_meta_arch_type = cfg.value()->od_meta_arch_type;
  String od_meta_layers_names_list = cfg.value()->od_meta_layers_names_list;

  Target tidl_target(TargetJSON{
      {"kind", String("tidl")},
      {"platform", platform},
      {"calibration_images", calibration_images},
      {"calibration_frames", calibration_frames},
      {"input_mean", input_mean},
      {"input_scale", input_scale},
      {"artifacts_folder", artifacts_folder},
      {"tensor_bits", tensor_bits},
      {"enable_offload", enable_offload},
      {"enable_c7x_codegen", enable_c7x_codegen},
      {"compile_for_device", compile_for_device},
      {"deny_list", deny_list},
      {"od_meta_arch_type", od_meta_arch_type},
      {"od_meta_layers_names_list", od_meta_layers_names_list},
  });

  return tidl_target;
}

}  // namespace tidl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm