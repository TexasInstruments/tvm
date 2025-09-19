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
 * \file src/relay/backend/contrib/tidl/compiler_attrs.h
 * \brief TIDL Compiler Attribute functionality
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_TIDL_COMPILER_ATTRS_H_
#define TVM_RELAY_BACKEND_CONTRIB_TIDL_COMPILER_ATTRS_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace tidl {

/*! \brief Attributes to store the compiler options for TIDL. */
struct TIDLCompilerConfigNode : public tvm::AttrsNode<TIDLCompilerConfigNode> {
  String platform;
  String calibration_images;
  Integer calibration_frames;
  Array<FloatImm> input_mean;
  Array<FloatImm> input_scale;
  String artifacts_folder;
  Integer tensor_bits;
  Bool enable_offload{Bool(false)};
  Bool enable_c7x_codegen{Bool(false)};
  Bool compile_for_device{Bool(false)};
  String deny_list;
  Integer od_meta_arch_type;
  String od_meta_layers_names_list;

  TVM_DECLARE_ATTRS(TIDLCompilerConfigNode, "ext.attrs.TIDLCompilerConfigNode") {
    TVM_ATTR_FIELD(platform)
        .describe("TI platform for TIDL compilation (am68pa, am68a, am69a, am67a, am62a)")
        .set_default("");
    TVM_ATTR_FIELD(calibration_images)
        .describe("directory containing calibration images for quantization")
        .set_default("");
    TVM_ATTR_FIELD(calibration_frames)
        .describe("number of calibration frames to generate from images (default: 10)")
        .set_default(10);
    TVM_ATTR_FIELD(input_mean)
        .describe("input mean values for RGB channels [R, G, B]")
        .set_default(Array<FloatImm>());
    TVM_ATTR_FIELD(input_scale)
        .describe("input scale values for RGB channels [R, G, B]")
        .set_default(Array<FloatImm>());
    TVM_ATTR_FIELD(artifacts_folder)
        .describe("output directory for TIDL compilation artifacts")
        .set_default("./tidl_artifacts");
    TVM_ATTR_FIELD(tensor_bits)
        .describe("number of bits for TIDL tensor quantization (8, 16, 32)")
        .set_default(8);
    TVM_ATTR_FIELD(enable_offload)
        .describe("enable TIDL acceleration offloading")
        .set_default(Bool(false));
    TVM_ATTR_FIELD(enable_c7x_codegen)
        .describe("enable C7x code generation for layers not offloaded to TIDL")
        .set_default(Bool(false));
    TVM_ATTR_FIELD(compile_for_device)
        .describe("compile for target device (aarch64) instead of host (x86)")
        .set_default(Bool(false));
    TVM_ATTR_FIELD(deny_list)
        .describe("comma-separated list of operations to exclude from TIDL offloading")
        .set_default("");
    TVM_ATTR_FIELD(od_meta_arch_type)
        .describe("object detection meta architecture type (e.g., 3 for SSD)")
        .set_default(-1);
    TVM_ATTR_FIELD(od_meta_layers_names_list)
        .describe("path to prototxt file containing object detection metadata")
        .set_default("");
  }
};

class TIDLCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TIDLCompilerConfig, Attrs,
                                            TIDLCompilerConfigNode);
};

/*! \brief Convert External Code Generator options to TVM Target. */
Target CreateTarget(const tvm::transform::PassContext& ctx);

}  // namespace tidl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_TIDL_COMPILER_ATTRS_H_