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
 * \file codegen_c7x.cc
 *
 * This is an adaptation derived from CodeGenC (the generic C backend) and
 * CodeGenCHost (the C backend for a host CPU).
 *
 * For development I duplicated all the base methods, even if they are not
 * changed. Comments indicate which methods are copied verbatim and which are
 * adapted.
 */
#include "codegen_c7x.h"

#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/ir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/runtime/logging.h>

#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>

#include "../../arith/pattern_match.h"
#include "../../support/str_escape.h"
#include "../build_common.h"
#include "../func_registry_generator.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

// We use a registered op (rather than a call_extern) for the stream
// access intrinsic, because we can indicate that it can be vectorized.
TVM_REGISTER_OP("tir.c7x.stream_access")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kUpdateState))
    .set_attr<TVectorizable>("TVectorizable", true);

// Helper functions to detect specific c7x intrinsic calls
static bool is_call_extern(const CallNode *call, const String& fname) {
  return call->op.same_as(builtin::call_extern()) &&
	 Downcast<StringImm>(call->args[0])->value == fname;
}
static bool is_call_extern(PrimExpr op, const String& fname) {
  const CallNode *call = op.as<CallNode>();
  return call && is_call_extern(call, fname);
}
static bool is_call_builtin(const CallNode *call, const String& fname) {
  return call->op.same_as(Op::Get(fname));
}
static bool is_call_builtin(PrimExpr op, const String& fname) {
  const CallNode *call = op.as<CallNode>();
  return call && is_call_builtin(call, fname);
}

// A simple pre-pass to find all the variables that are the src and dst
// of DMA calls.
class ScanDMA : public StmtExprVisitor {
  std::set<const VarNode *>& dma_set_;
  int& count_;
public:
  ScanDMA(std::set<const VarNode *>& vs, int& count) : dma_set_(vs), count_(count) {}
  void VisitExpr_(const CallNode* op) override {
    if (is_call_extern(op, "c7x_dma_setup"))
    {
      dma_set_.insert(Downcast<Var>(op->args[1]).get());
      dma_set_.insert(Downcast<Var>(op->args[6]).get());
      count_ += 1;
      ICHECK(count_ <= TVM_TARGET_C7X_MAX_DMA_CHANNELS)
          << "Number of c7x_dma_setup()s exceeded TVM_TARGET_C7X_MAX_DMA_CHANNELS";
    }
  }
};

// A pre-pass to find "packed calls" which is how the host kernel
// calls the device kernel. This is so we can emit declarations.
// @tir.tvm_call_packed("fused_multiply_58_kernel0", A, B, C)
// Skip tvm.contrib.sort.argsort_nms since it will be rewritten to
//   calling tvm_tidl_argsort_nms from c7x_tvm_runtime.h
class ScanPackedCalls : public StmtExprVisitor {
public:
  std::set<const CallNode *>& calls_;
public:
  ScanPackedCalls(std::set<const CallNode*>& cs) : calls_(cs) {}
  void VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::tvm_call_packed())) {
      std::string fname = Downcast<StringImm>(op->args[0])->value;
      if (fname != "tvm.contrib.sort.argsort_nms")
        calls_.insert(op);
    }
  }
};

// A pre-pass to find and collect stream (SE/SA) configurations
class ScanStreamAccess : public StmtExprVisitor {
public:
  StreamInfo& info_;
  ScanStreamAccess(StreamInfo& info) : info_(info) {}
  // Look for Let config_var = c7x_stream_config(...) and capture setup info
  void VisitStmt_(const LetStmtNode* op) override {
    if (is_call_extern(op->value, "c7x_stream_config"))
      info_.AddConfig(op->var.get(), Downcast<Call>(op->value).get());
    VisitStmt(op->body);
  }
  // Look for @tir.c7x.stream_access(...) and update vector lengths in setup
  void VisitExpr_(const CallNode* op) override {
    if (is_call_builtin(op, "tir.c7x.stream_access"))
      info_.UpdateVecLen(op);

    // Recurse to handle the case where there is more than one
    // stream_access in an expression
    StmtExprVisitor::VisitExpr_(op);
  }
};

// A pre-pass to find and collect vector stream (SE) used in condition of SelectNode
class ScanVSEinSelCond : public StmtExprVisitor {
  std::vector<DataType>& vse_;
  bool in_vector_cond;
public:
  ScanVSEinSelCond(std::vector<DataType>& vse) : vse_(vse), in_vector_cond(false) {}
  // Look for vector SelectNode and capture vector SEs
  void VisitExpr_(const SelectNode* op) override {
    in_vector_cond = (op->condition->dtype.lanes() > 1);
    StmtExprVisitor::VisitExpr(op->condition);
    in_vector_cond = false;
    StmtExprVisitor::VisitExpr(op->true_value);
    StmtExprVisitor::VisitExpr(op->false_value);
  }
  // Look for @tir.c7x.stream_access(...) and update vse_ if used in vector condition
  void VisitExpr_(const LoadNode* op) override {
    if (in_vector_cond && is_call_builtin(op->index, "tir.c7x.stream_access"))
      vse_.push_back(op->dtype);

    // Recurse to handle the case where there is more than one
    // stream_access in an expression
    StmtExprVisitor::VisitExpr_(op);
  }
};

class ScanMemory : public StmtExprVisitor {
  public:
  struct MemoryStats
  {
    size_t local_alloc_size;
    size_t global_alloc_size;
  };

  MemoryStats stats_ = { 0, 0};

  ScanMemory(const std::set<const VarNode *>& dma_buffers,
             bool& local_allocations_present,
             bool& global_allocations_present,
             size_t& max_global_alloc_in_bytes) : dma_buffers_(dma_buffers),
                                                  local_allocations_present_(local_allocations_present),
                                                  global_allocations_present_(global_allocations_present),
                                                  max_global_alloc_sz_in_bytes_(max_global_alloc_in_bytes) {
    stats_.local_alloc_size = stats_.global_alloc_size = 0;
  }

  ~ScanMemory() {
    if (stats_.global_alloc_size > max_global_alloc_sz_in_bytes_)
      max_global_alloc_sz_in_bytes_ = stats_.global_alloc_size;
    if (stats_.local_alloc_size > 0)
      local_allocations_present_ = true;
    if (stats_.global_alloc_size > 0)
      global_allocations_present_ = true;
  }

  void VisitExpr_(const CallNode* op) override {
    if (!is_call_extern(op, "c7x_dma_setup"))
    {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    // Calculate the size of the local buffer. Returns 0 the buffer is not a local buffer.
    auto calc_buffer_size =
      [&](const VarNode* buffer_ptr, int n) -> size_t {

        if (!IsLocal(buffer_ptr))
          return 0;

        const PointerTypeNode *ptr_type;
        const PrimTypeNode *prim_type = nullptr;
        if ((ptr_type = buffer_ptr->type_annotation.as<PointerTypeNode>()))
          prim_type = ptr_type->element_type.as<PrimTypeNode>();
        ICHECK(prim_type);

        // Compute size using the appropriate arguments to c7x_dma_setup
        size_t size = prim_type->dtype.bytes();
        for (int i = 1; i <= 4; ++i)
          size *= Downcast<IntImm>(op->args[n+i]).get()->value;

        // Account for double buffering
        size *= 2;

        // Account for alignment
        return aligned_size(size);
    };

    // Signature is:
    // @tir.call_extern("c7x_dma_setup", src_var, dim3, dim2, dim1, dim0,
    //                                   dst_var, dim3, dim2, dim1, dim0)
    const VarNode* src = Downcast<Var>(op->args[1]).get();
    const VarNode* dst = Downcast<Var>(op->args[6]).get();
    size_t src_size = calc_buffer_size(src, 1);
    size_t dst_size = calc_buffer_size(dst, 6);

    DLOG(INFO) << "ScanMemory:c7x_dma_setup: " << src->name_hint << " : " << src_size << std::endl;
    DLOG(INFO) << "ScanMemory:c7x_dma_setup: " << dst->name_hint << " : " << dst_size << std::endl;

    stats_.local_alloc_size += src_size;
    stats_.local_alloc_size += dst_size;

    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocateNode* op) override {
    ICHECK(!is_zero(op->condition));

    auto scope = GetPtrStorageScope(op->buffer_var);
    alloc_storage_scope_[op->buffer_var.get()] = scope;

    // Skip allocation calls for DMA buffers; they are allocated as part of DMA setup
    // Note: All DMA allocations via c7x_dma_setup have been hoisted to the top of the function.
    //       See c7x.py, C7xDMATransform
    if (IsLocal(op->buffer_var.get()) && IsDMA(op->buffer_var.get())) {
      VisitStmt(op->body);
      return;
    }

    int32_t alloc_size = op->constant_allocation_size();
    ICHECK_GT(alloc_size, 0) << "Can only handle constant size stack allocation for now";

    size_t alloc_size_in_bytes = aligned_size(alloc_size * op->dtype.bytes());
 
    DLOG(INFO) << "ScanMemory:AllocateNode: " << op->buffer_var->name_hint << " : "
               << alloc_size_in_bytes << std::endl;

    if (!IsLocal(op->buffer_var.get())) {
      stats_.global_alloc_size += alloc_size_in_bytes;
    } else {
      stats_.local_alloc_size += alloc_size_in_bytes;
    }

    VisitStmt(op->body);
  }

  friend std::ostream& operator<<(std::ostream& os, const ScanMemory& m) {
    os << "L2 alloc: " << m.stats_.local_alloc_size;
    os << " Global: " << m.stats_.global_alloc_size << std::endl;
    return os;
  }

  private:
  /*! the storage scope of allocation */
  std::unordered_map<const VarNode*, std::string> alloc_storage_scope_;
  const std::set<const VarNode *>& dma_buffers_;

  bool& local_allocations_present_;
  bool& global_allocations_present_;

  /* \brief Maximum global allocation across functions */
  size_t& max_global_alloc_sz_in_bytes_;

  /* \brief Account for alignment when computing allocation sizes */
  const size_t MEM_ALIGN = 8;
  size_t aligned_size(size_t size) {
    return (((size + MEM_ALIGN-1)/MEM_ALIGN) * MEM_ALIGN);
  }

  // Is variable a locally allocated buffer
  bool IsLocal(const VarNode* var) {
    auto it = alloc_storage_scope_.find(var);
    return it != alloc_storage_scope_.end() && it->second.compare(0, 5, "local") == 0;
  }

  // Is variable used in DMA copy-in/copy-out
  bool IsDMA(const VarNode* var) {
    return dma_buffers_.find(var) != dma_buffers_.end();
  }
};

CodeGenC7x::CodeGenC7x() { module_name_ = GetUniqueName("__tvm_module_ctx"); }

// adapted from CodeGenC
void CodeGenC7x::Init(bool output_ssa, bool emit_asserts, std::string target_str) {
  emit_asserts_ = emit_asserts;
  declared_globals_.clear();

  max_global_alloc_sz_in_bytes_ = 0;

  decl_stream << "// custom backend for C7x" << "\n";
  decl_stream << "// tvm target: " << target_str << "\n";
  // We don't need packed func API
  decl_stream << "//#include \"tvm/runtime/c_runtime_api.h\"\n";
  decl_stream << "#include \"tvm/runtime/c_backend_api.h\"\n";
  decl_stream << "//#include <math.h>\n";
  // C7x-specific runtime support (DMA, SE, etc)
  decl_stream << "#include \"c7x_tvm_runtime.h\"\n\n";
  //decl_stream << "void* " << module_name_ << " = NULL;\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenC7x::PrintTrailer() {
  if (max_global_alloc_sz_in_bytes_ > 0)
    this->stream << "extern \"C\" size_t get_ddr_scratch_mem_size() { return " << max_global_alloc_sz_in_bytes_ << "; }\n";
}

void CodeGenC7x::AddFunction(const PrimFunc& f) {
  // stream << "/* AddFunction */\n";
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC7x: Expect PrimFunc to have the global_symbol attribute";
  function_names_.push_back(global_symbol.value());

  // below verbatim from CodeGenC::AddFunction(f);
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  // declare functions called via "packed calls"
  DeclarePackedCalls(f);

  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  this->PrintFuncPrefix();
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      PrintType(GetType(v), stream);
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && restrict_keyword_.length() != 0) {
        stream << ' ' << restrict_keyword_;
      }
    } else {
      PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PreFunctionBody(f);
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

// adapted
void CodeGenC7x::InitFuncState(const PrimFunc& f) {
  // Run the pre-pass to find all the variables used in DMA copy
  // intrinsics
  dma_buffers_.clear();
  num_dma_intrinsics_ = 0;
  ScanDMA DMAScanner(dma_buffers_, num_dma_intrinsics_);
  DMAScanner(f->body);

  // Run the pre-pass to gather SE/SA info
  stream_info_.Clear();
  ScanStreamAccess StreamScanner(stream_info_);
  StreamScanner(f->body);

  // Run the pre-pass to gather SE used in vector select conditions (work around cl7x/opt7x abort)
  // cl7x/opt7x aborts on: pout[i] = __SE1ADV(float16) != (float16)0.0f ? p2[i] : (float16)0.0f;
  // work around: pout[i] = (t = __SE1ADV(float16), t) != (float16)0.0f ? p2[i] : (float16)0.0f;
  // Step 1: collect the SE types into a vector in pre-pass
  // Step 2: print out all declarations for temporaries, e.g. float16 vse_t0;
  // Step 3: Rewrite SE load as the comma expression using the workaround
  sel_cond_vse_dtypes_.clear();
  ScanVSEinSelCond VSEScanner(sel_cond_vse_dtypes_);
  VSEScanner(f->body);
  in_vector_cond = false;
  se_in_vec_cond_count = 0;

  local_allocations_present_ = false;
  global_allocations_present_ = false;
  ScanMemory MemoryScanner(dma_buffers_, local_allocations_present_, global_allocations_present_, 
                          max_global_alloc_sz_in_bytes_);
  MemoryScanner(f->body);
  DLOG(INFO) << MemoryScanner;

  CodeGenC::InitFuncState(f);
}

// When a kernel is split between host and device (via tir.SplitHostDevice),
// the host kernel uses the packed_func protocol to call the device
// kernel. We disable packed call lowering (by disabling tir.LowerTVMBuiltin)
// so that packed calls appear as normal calls. But this requires declarations
// for the callee.
void CodeGenC7x::DeclarePackedCalls(const PrimFunc& f) {
  // Find all the packed calls
  std::set<const CallNode*> packed_calls;
  ScanPackedCalls CallScanner(packed_calls);
  CallScanner(f->body);

  // declare the callees
  for(const CallNode *call : packed_calls) {
    std::string fname = Downcast<StringImm>(call->args[0])->value;
    PrintIndent();
    stream << "extern \"C\" ";
    PrintType(call->dtype, stream);
    stream << " " << fname << "(";
    for (size_t i = 1; i < call->args.size(); ++i) {
      PrintType(call->args[i].dtype(), stream);
      if (i < call->args.size() - 1)
        stream << ", ";
    }
    stream << ");\n";
  }
  if (packed_calls.size() > 0)
    stream << "\n";
}

// adapted
void CodeGenC7x::PreFunctionBody(const PrimFunc& f) {
  // Only print these in device function, not in host function
  // TODO: Host function's kTarget attr is set to nullptr (undefined) in SplitHostDevice Pass.
  // However, when the flow reaches here, it becomes defined again.  Use name to check for now.
  // auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  // if (! target.defined())  return;  // does not work here for host function
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  if (static_cast<std::string>(global_symbol.value()).find("_kernel") == std::string::npos)
    return;

  // Protect generated C7x kernel (with l2,dma,etc) with critical section
  this->PrintIndent();
  stream << "CriticalSectionContext csContext;\n";

  // Initialize L2Context first, DMAContext needs it for L2 memory allocation
  if (local_allocations_present_) {
    this->PrintIndent();
    stream << "AllocL2Context L2Context;\n";
  }

  if (global_allocations_present_) {
    this->PrintIndent();
    stream << "AllocDDRContext DDRContext;\n";
  }

  // Do not define a DMAContext if there are no DMA calls
  if (num_dma_intrinsics_ > 0)
  {
    this->PrintIndent();
    stream << "DMAContext DMAContext("<< num_dma_intrinsics_ << ");\n\n";
  }

  // Declare vector SE temps that will be used in vector condition of SelectNode
  int se_count = 0;
  for (const DataType &vdtype : sel_cond_vse_dtypes_)
  {
    this->PrintIndent();
    PrintType(vdtype, stream);
    stream << " vse_t" << se_count++ << ";\n";
  }
}

#if 0
// verbatim from CodegenCHost
void CodeGenC7x::LinkParameters(Map<String, LinkedParam> params) {
  PrintFuncPrefix();
  stream << " " << tvm::runtime::symbol::tvm_lookup_linked_param
         << "(void* args, int* arg_type_ids, int num_args, void* out_ret_value, "
         << "int* out_ret_tcode, void* resource_handle) {\n";
  ICHECK_EQ(GetUniqueName(tvm::runtime::symbol::tvm_lookup_linked_param),
            tvm::runtime::symbol::tvm_lookup_linked_param)
      << "builtin PackedFunc name already taken: " << tvm::runtime::symbol::tvm_lookup_linked_param;
  stream << "    switch (((int64_t*) args)[0]) {\n"
         << "    default:\n"
         << "        out_ret_tcode[0] = " << kTVMNullptr << ";\n"
         << "        return 0;\n";

  function_names_.push_back(tvm::runtime::symbol::tvm_lookup_linked_param);
  for (auto kv : params) {
    decl_stream << "\n"
                << "#ifdef __cplusplus\n"
                << "extern \"C\" {\n"
                << "#endif\n"
                << "static const ";
    int64_t num_elements = 1;
    for (int64_t dim : kv.second->param.Shape()) {
      num_elements *= dim;
    }
    PrintType(kv.second->param.DataType(), decl_stream);
    decl_stream << " " << ::tvm::runtime::symbol::tvm_param_prefix << kv.first << "["
                << num_elements << "] = {\n";
    NDArrayDataToC(kv.second->param, 4, decl_stream);
    decl_stream << "};\n"
                << "#ifdef __cplusplus\n"
                << "}  // extern \"C\"\n"
                << "#endif\n";
    stream << "    case " << kv.second->id << ":\n"
           << "        ((uint64_t*)out_ret_value)[0] = (uint64_t) (uintptr_t) "
           << ::tvm::runtime::symbol::tvm_param_prefix << kv.first << ";\n"
           << "        out_ret_tcode[0] = " << kTVMOpaqueHandle << ";\n"
           << "        return 0;\n";
  }
  stream << "    }\n"
         << "}\n";
}
#endif

// verbatim from CodegenCHost
void CodeGenC7x::PrintFuncPrefix() {  // NOLINT(*)
  #if 0
  stream << "#ifdef __cplusplus\n"
         << "extern \"C\"\n"
         << "#endif\n"
         << "TVM_DLL int32_t";
  #endif
  stream << "extern \"C\"\n";
  stream << "int32_t";
}

// verbatim from CodegenCHost
void CodeGenC7x::PrintFinalReturn() {  // NOLINT(*)
  this->PrintIndent();
  stream << "return 0;\n";
}

// verbatim from CodegenCHost
void CodeGenC7x::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64: // On C7x, long is 64 bits
        os << "long";
        break;
      case 1:
        os << "int";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

// verbatim from CodegenC
void CodeGenC7x::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

// verbatim from CodegenC
void CodeGenC7x::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << "/* PrintVecElemLoad */";
  os << vec << ".s" << std::hex << i << std::dec;
}

// verbatim from CodegenC
void CodeGenC7x::PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }

  if (i == 0) {
    os << "((";
    PrintType(t, os);
    os << ")(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "))";
  }
  return;
}

// verbatim from CodegenC
void CodeGenC7x::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i << " = " << value << ";\n" << std::dec;
}

// verbatim from CodegenC
std::string CodeGenC7x::GetVecLoad(DataType t, const VarNode* buffer, PrimExpr base) {
  return GetBufferRef(t, buffer, base);
}

// verbatim from CodegenC
void CodeGenC7x::PrintVecStore(const VarNode* buffer, DataType t, PrimExpr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

// verbatim from CodegenC
std::string CodeGenC7x::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;

  // C7x compiler requires convert_ intrinsics for vector casts
  if (target.lanes() == 1)
  {
    os << "((";
    this->PrintType(target, os);
    os << ")";
  } else {
    os << "convert_";
    this->PrintType(target, os);
    os << "(";
  }
  os << value << ")";
  return os.str();
}

// verbatim from CodegenC
inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  if (op->dtype == DataType::Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->dtype, os);
    os << ")" << op->value;
  }
}

// verbatim from CodegenC
inline void PrintUIntConst(DataType dtype, uint64_t val, std::ostream& os,
                           CodeGenC* p) {  // NOLINT(*)
  if (dtype == DataType::UInt(32)) {
    std::ostringstream temp;
    temp << val << "U";
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(dtype, os);
    os << ")" << val;
  }
}

// verbatim from CodegenC
inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->dtype, os);
      os << ')' << std::scientific << op->value << 'f';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

// adapted
void CodeGenC7x::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << "\"" << op->value << "\"";
}

// verbatim from CodegenC
template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

// verbatim from CodegenC
inline void PrintBinaryIntrinsic(const CallNode* op, const char* opstr,
                                 std::ostream& os,  // NOLINT(*)
                                 CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    ICHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->args[0], op->args[1], os);
  }
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const AddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const SubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const MulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const DivNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenC7x::VisitExpr_(const ModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
#if 0
// verbatim from CodegenCHost
void CodeGenC7x::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

// verbatim from CodegenCHost
void CodeGenC7x::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}
#endif
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const EQNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const NENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const LTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const LENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const GTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const GENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const AndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const OrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const NotNode* op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

// adapted from CodegenC
void CodeGenC7x::PrintCallExtern(Type ret_type, String global_symbol,
                                 const Array<PrimExpr>& args,
                                 bool skip_first_arg, std::ostream& os) {  // NOLINT(*)

  // dma copy call handled by EvaluateNode, so ignore here
  if (global_symbol == "c7x_dma_copy")
    ;
  else if (global_symbol == "tvm.contrib.sort.argsort_nms") {
    const CallNode *compute  = args[1].as<CallNode>();
    const CallNode *sort_num = args[2].as<CallNode>();
    const CallNode *output   = args[3].as<CallNode>();
    const IntImmNode *axis   = args[4].as<IntImmNode>();
    const IntImmNode *ascend = args[5].as<IntImmNode>();
    ICHECK(compute != nullptr && sort_num != nullptr && output != nullptr);
    ICHECK( compute->op.same_as(builtin::tvm_stack_make_array()) &&
           sort_num->op.same_as(builtin::tvm_stack_make_array()) &&
             output->op.same_as(builtin::tvm_stack_make_array()) );
    ICHECK(axis != nullptr && axis->value == 1);
    ICHECK(ascend != nullptr && ascend->value == 0);

    os << "tvm_tidl_argsort_nms(";
    this->PrintExpr(compute->args[0], os);   os << ", ";
    this->PrintExpr(sort_num->args[0], os);  os << ", ";
    this->PrintExpr(output->args[0], os);
    os << ")";
  }
  else {
    os << global_symbol << "(";
    for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
      this->PrintExpr(args[i], os);
      if (i < args.size() - 1) {
	os << ", ";
      }
    }
    os << ")";
  }
}

// verbatim from CodegenC
#if 0
void CodeGenC7x::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr_op = op->op.as<OpNode>()) {
    auto call_op = GetRef<Op>(ptr_op);

    if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      ICHECK_GE(op->args.size(), 1U);
      auto func = Downcast<StringImm>(op->args[0]);
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), func->value, op->args, true, os);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), op_attr_global_symbol_[call_op],
                            op->args, false, os);
    } else if (op->op.same_as(builtin::bitwise_and())) {
      PrintBinaryIntrinsic(op, " & ", os, this);
    } else if (op->op.same_as(builtin::large_uint_imm())) {
      ICHECK_EQ(op->args.size(), 2U);
      uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
      uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
      uint64_t val = (high << 32U) | low;
      PrintUIntConst(op->dtype, val, os, this);
    } else if (op->op.same_as(builtin::bitwise_xor())) {
      PrintBinaryIntrinsic(op, " ^ ", os, this);
    } else if (op->op.same_as(builtin::bitwise_or())) {
      PrintBinaryIntrinsic(op, " | ", os, this);
    } else if (op->op.same_as(builtin::bitwise_not())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(~";
      this->PrintExpr(op->args[0], os);
      os << ')';
    } else if (op->op.same_as(builtin::shift_left())) {
      PrintBinaryIntrinsic(op, " << ", os, this);
    } else if (op->op.same_as(builtin::shift_right())) {
      PrintBinaryIntrinsic(op, " >> ", os, this);
    } else if (op->op.same_as(builtin::if_then_else())) {
      os << "(";
      PrintExpr(op->args[0], os);
      os << " ? ";
      PrintExpr(op->args[1], os);
      os << " : ";
      PrintExpr(op->args[2], os);
      os << ")";
    } else if (op->op.same_as(builtin::address_of())) {
      const LoadNode* l = op->args[0].as<LoadNode>();
      ICHECK(op->args.size() == 1 && l);
      os << "((";
      this->PrintType(l->dtype.element_of(), os);
      os << " *)" << this->GetVarID(l->buffer_var.get()) << " + "
         << "(";
      this->PrintExpr(l->index, os);
      if (l->dtype.bits() == 4 || (l->dtype.bits() == 1 && l->dtype.is_int())) {
        os << " / " << (32 / l->dtype.bits());
      }
      os << "))";
    } else if (op->op.same_as(builtin::tvm_struct_get())) {
      ICHECK_EQ(op->args.size(), 3U);
      os << GetStructRef(op->dtype, op->args[0], op->args[1], op->args[2].as<IntImmNode>()->value);
    } else if (op->op.same_as(builtin::isnullptr())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " == NULL)";
    } else if (op->op.same_as(builtin::reinterpret())) {
      int ssa_scope = BeginScope();
      std::string rhs = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
      os << "(*(";
      this->PrintType(op->dtype, os);
      os << " *)(&(" << rhs << ")))";
      EndScope(ssa_scope);
    } else if (op->op.same_as(builtin::isnan())) {
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " != ";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else {
      LOG(FATAL) << "Unresolved call " << op->op;
    }
  } else {
    ICHECK(op->op.as<GlobalVarNode>());
    LOG(FATAL) << "Do not yet support cross function call";
  }
}
#endif

#if 1
// from CodeGenCHost
void CodeGenC7x::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::tvm_stack_alloca())) {
    std::string stack_name = GetUniqueName("stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    ICHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMValue " << stack_name << "[" << size << "];\n";
    os << stack_name;
  } else if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    const StringImmNode* s = op->args[0].as<StringImmNode>();
    ICHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
    int64_t begin = op->args[3].as<IntImmNode>()->value;
    int64_t end = op->args[4].as<IntImmNode>()->value;
    int64_t num_args = end - begin;
    ICHECK_GE(num_args, 0);
    std::string func_name = s->value;
    // NOTE: cannot rely on GetUnique for global decl_stream declarations
    // because it is reset between AddFunction().
    std::string packed_func_name = func_name + "_packed";
    if (declared_globals_.insert(packed_func_name).second) {
      // Still reserve the name among unique names.
      ICHECK(GetUniqueName(packed_func_name) == packed_func_name)
          << "Expected name " << packed_func_name << " to not be taken";
      decl_stream << "static void* " << packed_func_name << " = NULL;\n";
    }
    this->PrintGetFuncFromBackend(func_name, packed_func_name);
    this->PrintFuncCall(packed_func_name, num_args);
  } else if (op->op.same_as(builtin::tvm_call_packed())) {
    const StringImmNode* s = op->args[0].as<StringImmNode>();
    std::string func_name = s->value;
    this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), s->value, op->args, true, os);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}
#endif

// verbatim from CodegenC
void CodeGenC7x::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os << "(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

// adapted from CodegenC
void CodeGenC7x::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  if (is_call_builtin(op->index, "tir.c7x.stream_access")) {
    StreamAccess access(op->index.as<CallNode>());
    // cl7x/opt7x aborts on: pout[i] = __SE1ADV(float16) != (float16)0.0f ? p2[i] : (float16)0.0f;
    // work around: pout[i] = (t = __SE1ADV(float16), t) != (float16)0.0f ? p2[i] : (float16)0.0f;
    if (in_vector_cond)  os << "(vse_t" << se_in_vec_cond_count << " = ";
    // example: __SE0ADV(float16)
    os << "__" << access.engine;
    if (access.adv) os << "ADV";
    os << "(";
    PrintType(op->dtype, os);
    os << ")";
    if (in_vector_cond)  os << ", vse_t" << se_in_vec_cond_count++ << ")";
    return;
  }
  int lanes = op->dtype.lanes();
  // delcare type.
  if (op->dtype.lanes() == 1) {
    std::string ref = GetBufferRef(op->dtype, op->buffer_var.get(), op->index);
    HandleVolatileLoads(ref, op, os);
  } else {
    ICHECK(is_one(op->predicate)) << "predicated load is not supported";

    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(op->index)) {
      std::string ref = GetVecLoad(op->dtype, op->buffer_var.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string vid = GetVarID(op->buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

// adapted from CodegenC
void CodeGenC7x::VisitStmt_(const StoreNode* op) {
  //stream << "// VisitStmt<StoreNode>\n";
  DataType t = op->value.dtype();
  if (is_call_builtin(op->index, "tir.c7x.stream_access")) {
    StreamAccess access(op->index.as<CallNode>());
    std::string vid = GetVarID(Downcast<Var>(op->buffer_var).get());
    std::string rhs_value = this->PrintExpr(op->value);
    // example: __SA0ADV(float16, ptr)
    std::ostringstream sa_os;
    sa_os << "__" << access.engine;
    if (access.adv) sa_os << "ADV";
    sa_os << "(";
    PrintType(t, sa_os);
    sa_os << ", " << vid << ")";

    // If access requires predication, generate a predicate and a predicated
    // store. Example:
    //   float16 value = <rhs expression>
    //   __vpred pred = __SA0_VPRED(float16);
    //   __vstore_pred(pred, __SA0ADV(float16, ptr), value);
    // Note: For scalar access __vstore_pred cannot be used. Convert the
    // __vpred to a scalar using __create_scalar and use assignment.
    if (access.pred) {
      auto rhs_var = Var("value", t);
      auto pred_var = Var("pred", DataType::Handle());

      new_variables_.push_back(rhs_var);
      new_variables_.push_back(pred_var);

      this->PrintIndent();
      PrintType(t, stream);
      stream << " " << AllocVarID(rhs_var.get()) << " = " << rhs_value << ";\n";

      this->PrintIndent();
      stream << "__vpred " << AllocVarID(pred_var.get()) << " = "
             << "__" << access.engine << "_VPRED(";
      PrintType(t, stream);
      stream << ");\n";

      this->PrintIndent();
      if (t.lanes() == 1) // scalar variable, use scalar assignment
      {
        stream << "if (__create_scalar(" << GetVarID(pred_var.get()) << ")) { \n";
        int if_scope = this->BeginScope();
        this->PrintIndent();
        stream << "*" << sa_os.str() << " = " << GetVarID(rhs_var.get()) << ";\n";
        this->EndScope(if_scope);
        this->PrintIndent();
        stream << "}\n";
      }
      else
      {
          stream << "__vstore_pred(" << GetVarID(pred_var.get()) << ", "
                                    << sa_os.str() << ", "
				    << GetVarID(rhs_var.get()) << ");\n";
      }
    }
    // no predication: just generate *__SA0ADV(float16, ptr) = rhs;
    else {
      this->PrintIndent();
      stream << "*" << sa_os.str() << " = " << rhs_value << ";\n";
    }
  }
  else if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    ICHECK(is_one(op->predicate)) << "Predicated store is not supported";
    arith::PVar<PrimExpr> base;

    if (arith::ramp(base, 1, t.lanes()).Match(op->index)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base.Eval(), value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        DataType elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, op->index.dtype(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.dtype(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  /*
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  */
  std::string value = PrintExpr(op->value);
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  ICHECK_EQ(op->base.dtype(), DataType::Int(32));
  os << "((int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "))";
}

// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  LOG(FATAL) << "Shuffle: not supported ";
}

// adapted from CodegenCHost
void CodeGenC7x::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  // C7x ("OpenCL-style") broadcast syntax is simply e.g. ((float16)(x))
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  os << v;
  os << "))";
  #if 0
  os << "/* broadcast */ ((";
  PrintType(op->dtype, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
  #endif
}

// verbatim from CodegenC
void CodeGenC7x::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  os << "/* select */ (";
  in_vector_cond = (op->condition->dtype.lanes() > 1);
  PrintExpr(op->condition, os);
  in_vector_cond = false;
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

// Emit setup code for DMA operations.
// A DMA operation involves two variables: a pointer to the src buffer and
// a pointer to the dst buffer. These are used to initialize 'Buffer' (or
// 'DoubleBuffer') objects, parameterized by their layouts, and a DMA object
// which manages the transfers.
// For example:
//
//  Buffer<float>       A_buffer(1*672*14*14, A);
//  DoubleBuffer<float> A_local_buffer(1*8*14*14, L2Context);
//  auto A_dma = create_DMA(DMAContext, A_buffer, A_local_buffer);

void CodeGenC7x::PrintDMASetup(const VarNode* dma_var, const CallNode* call) {
  // Helper function to declare each buffer
  auto declare_buffer =
    [&](const VarNode* buffer_ptr, int n) -> const VarNode* {
      bool is_local = IsLocal(buffer_ptr);
      dma_map_[buffer_ptr] = dma_var;
      std::string buffer_type = is_local ? "DoubleBuffer" : "Buffer";
      auto buffer_var = Var(buffer_ptr->name_hint + "_buffer", DataType::Handle());
      new_variables_.push_back(buffer_var);
      std::string buffer_name = AllocVarID(buffer_var.get());
      const PointerTypeNode *ptr_type;
      const PrimTypeNode *prim_type = nullptr;
      if ((ptr_type = buffer_ptr->type_annotation.as<PointerTypeNode>()))
	    prim_type = ptr_type->element_type.as<PrimTypeNode>();
      ICHECK(prim_type);

      // example output: Buffer<float> A_buffer(1*672*14*14, A);
      this->PrintIndent();
      stream << buffer_type << "<";
      PrintType(prim_type->dtype, stream);
      stream << "> " << buffer_name << "(";
      // If we do "stream << call->args[n+i], we are using tir's representation printer
      //     (ReprPrint), which will print "(int64)1" for 64-bit IntImm of value 1, which
      //     will cause cl7x compilation error.  So, we print out IntImm value for C7x code,
      //     which will simply be "1" for the above case.
      for (int i = 1; i <= 4; ++i)
      {
	      stream << Downcast<IntImm>(call->args[n+i]).get()->value;
        if (i != 4) stream << "*";
      }
      // If this is a local buffer, call the allocator to initialize it
      if (is_local)
	      stream << ", L2Context";
      else
	      stream << ", " << GetVarID(buffer_ptr);
      stream << ");\n";

      return buffer_var.get();
  };
  // Signature is:
  // @tir.call_extern("c7x_dma_setup", src_var, dim3, dim2, dim1, dim0,
  //                                   dst_var, dim3, dim2, dim1, dim0,
  //                                   num_blocks, sync_axis, soffset, doffset,
  //                                   sicnt0, sicnt1,   sicnt2,   sicnt3,
  //                                           sstride1, sstride2, sstride3,
  //                                   dicnt0, dicnt1,   dicnt2,   dicnt3,
  //                                           dstride1, dstride2, dstride3)
  const VarNode* src = Downcast<Var>(call->args[1]).get();
  const VarNode* dst = Downcast<Var>(call->args[6]).get();
  const VarNode* src_buf = declare_buffer(src, 1);
  const VarNode* dst_buf = declare_buffer(dst, 6);

  // auto A_dma = create_DMA(DMAContext, A_buffer, A_local_buffer,
  //                                   num_blocks, sync_axis, soffset, doffset,
  //                                   sicnt0, sicnt1,   sicnt2,   sicnt3,
  //                                           sstride1, sstride2, sstride3,
  //                                   dicnt0, dicnt1,   dicnt2,   dicnt3,
  //                                           dstride1, dstride2, dstride3);
  this->PrintIndent();
  std::string dma_var_name = AllocVarID(dma_var);
  stream << "auto " << dma_var_name
         << " = create_DMA(DMAContext, "
         << GetVarID(src_buf) << ", "
         << GetVarID(dst_buf);
  for (int i = 11; i <= 28; i++)
    stream << ", " << call->args[i].as<IntImmNode>()->value;
  stream << ");\n";

  // For local (allocated) buffers, initialize the TVM variable using the
  // accessor of the DMA object:
  //   void* __restrict__ A_local = A_dma.dst_ptr();
  auto define_ptr = [&](const VarNode* ptr_var, bool is_src) -> void {
    if (!IsLocal(ptr_var))
      return;
    this->PrintIndent();
    PrintType(ptr_var->dtype, stream);
    stream << ' ' << restrict_keyword_;
    stream << ' ' << AllocVarID(ptr_var) << " = ";
    std::string method = is_src ? "src_ptr" : "dst_ptr";
    stream << GetVarID(dma_var) << "." << method << "();\n";
  };
  define_ptr(src, true);
  define_ptr(dst, false);
}

// Emit setup code for SE/SA config
void CodeGenC7x::PrintStreamConfig(const VarNode* config_var) {
  const std::string vid = AllocVarID(config_var);
  const StreamDesc& desc = stream_info_.GetDesc(config_var);

  // SEConfig<type, veclen> vid(icnt0, icnt1, ..., icnt5, dim1, ..., dim5);
  this->PrintIndent();
  stream << desc.kind << "Config<";
  PrintType(desc.dtype, stream);
  stream << ", " << desc.veclen;
  stream << "> " << vid << "(";
  for (int i = 0; i < 6; ++i) { // generate six params since ICNT5 is max for SE/SA
    this->PrintExpr(desc.icnts[i], stream);
    stream << ", ";
  }
  for (int i = 0; i < 5; ++i) { // generate five params since DIM0=1 by default and DIM5 is max for SE/SA
    if (i != 0)  stream << ", ";
    this->PrintExpr(desc.dims[i], stream);
  }
  stream << ");\n";
}

// adapted from CodegenC
void CodeGenC7x::VisitStmt_(const LetStmtNode* op) {
  //stream << "// VisitStmt<LetStmtNode>\n";
  if (is_call_extern(op->value, "c7x_dma_setup")) {
    PrintDMASetup(op->var.get(), Downcast<Call>(op->value).get());
    PrintStmt(op->body);
    return;
  }
  else if (is_call_extern(op->value, "c7x_stream_config")) {
    PrintStreamConfig(op->var.get());
    PrintStmt(op->body);
    return;
  }
  PrintIndent();
  std::string value = PrintExpr(op->value);
  if (op->var.dtype() == DataType::Handle() && handle_data_type_.count(op->var.get())) {
    PrintType(handle_data_type_.at(op->var.get()), stream);
    stream << "* " << AllocVarID(op->var.get()) << " = (";
    PrintType(handle_data_type_.at(op->var.get()), stream);
    stream << "*)" << value << ";\n";
  } else if (op->var.dtype() == DataType::Handle()) {
    PrintType(op->var.dtype(), stream);
    stream << ' ' << restrict_keyword_;
    stream << ' ' << AllocVarID(op->var.get()) << " = static_cast<";
    PrintType(op->var.dtype(), stream);
    stream << ">(" << value << ");\n";
  } else {
    PrintType(op->var.dtype(), stream);
    stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
  }

  PrintStmt(op->body);
}

// adapted from CodegenC
void CodeGenC7x::VisitStmt_(const AllocateNode* op) {
  //stream << "// VisitStmt<AllocateNode>\n";
  ICHECK(!is_zero(op->condition));
  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;

  // Skip allocation calls for DMA buffers; they are allocated as part
  // of DMA setup
  if (IsLocal(op->buffer_var.get()) && IsDMA(op->buffer_var.get())) {
    this->PrintStmt(op->body);
    return;
  }

  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
  PrintStorageScope(scope, stream);
  //PrintType(op->dtype, stream);
  // stream << ' ' << vid << '[' << constant_size << "];\n";
  auto ptype = PointerType(PrimType(op->dtype));
  PrintType(ptype, stream);
  // always put restrict on local buffers
  stream << ' ' << restrict_keyword_;
  stream << ' ' << vid << " = static_cast<";
  PrintType(ptype, stream);
  if (IsLocal(op->buffer_var.get()))
    stream << ">(L2Context.allocate(" << constant_size * op->dtype.bytes() << "));\n";
  else
    stream << ">(DDRContext.allocate(" << constant_size * op->dtype.bytes() << "));\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

// verbatim from CodegenC
void CodeGenC7x::VisitStmt_(const AttrStmtNode* op) {
  //stream << "// VisitStmt<AttrStmtNode>\n";
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
    }
  /*
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    volatile_buf_.insert(v);
    */
  } else if (op->attr_key == tir::attr::pragma_import_c) {
    const StringImmNode* value = op->value.as<StringImmNode>();
    ICHECK(value != nullptr);
    decl_stream << value->value;
  }
  this->PrintStmt(op->body);
}

// verbatim from CodegenC
void CodeGenC7x::VisitStmt_(const AssertStmtNode* op) {
  #if 0
  //stream << "// VisitStmt<AssertStmtNode>\n";
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (const auto* str = op->message.as<StringImmNode>()) {
    // GLOG style check
    stream << "ICHECK(" << cond << ") << \"" << str->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
  #endif
  this->PrintStmt(op->body);
}

#if 0
// from CodegenCHost
void CodeGenC7x::VisitStmt_(const AssertStmtNode* op) {  // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    PrintIndent();
    stream << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value << "\");\n";
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
  this->PrintStmt(op->body);
}
#endif

// verbatim from CodegenC
void CodeGenC7x::VisitStmt_(const ForNode* op) {
  //stream << "// VisitStmt<ForStmtNode>\n";
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

// verbatim from CodegenC
void CodeGenC7x::VisitStmt_(const IfThenElseNode* op) {
  //stream << "// VisitStmt<IfThenElseNode>\n";
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
    stream << "if " << cond << " {\n";
  } else {
    stream << "if (" << cond << ") {\n";
  }
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

// verbatim from CodegenC
void CodeGenC7x::VisitStmt_(const SeqStmtNode* op) {
  //stream << "// VisitStmt<SeqStmtNode>\n";
  for (Stmt stmt : op->seq) {
    PrintStmt(stmt);
  }
}

// adapted from CodegenC
void CodeGenC7x::VisitStmt_(const EvaluateNode* op) {
  //stream << "// VisitStmt<EvaluateNode>\n";
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call) {
    if (call->op.same_as(builtin::tvm_storage_sync())) {
      this->PrintStorageSync(call);
      return;
    } else if (call->op.same_as(builtin::tvm_struct_set())) {
      ICHECK_EQ(call->args.size(), 4);
      std::string value = PrintExpr(call->args[3]);
      std::string ref = GetStructRef(call->args[3].dtype(), call->args[0], call->args[1],
                                     call->args[2].as<IntImmNode>()->value);
      this->PrintIndent();
      this->stream << ref << " = " << value << ";\n";
      return;
    } else if (call->op.same_as(builtin::call_extern())) {
      String func = Downcast<StringImm>(call->args[0])->value;
      // Calls to c7x_dma_copy turn into X_dma.copy().
      // The src and dst pointers are captured in the DMA object.
      if (func == "c7x_dma_copy") {
        const VarNode* src = Downcast<Var>(call->args[1]).get();
        const VarNode* dst = Downcast<Var>(call->args[2]).get();
        const VarNode* dma_var = dma_map_.at(src);
        this->PrintIndent();
	    stream << GetVarID(dma_var) << ".copy();\n";
	    auto reset_ptr = [&](const VarNode* ptr_var, bool is_src) -> void {
	        if (!IsLocal(ptr_var))
	        return;
	        this->PrintIndent();
	        stream << GetVarID(ptr_var) << " = ";
	        std::string method = is_src ? "src_ptr" : "dst_ptr";
	        stream << GetVarID(dma_var) << "." << method << "();\n";
	    };
	    reset_ptr(src, true);
	    reset_ptr(dst, false);
        return;
      }
      else if (func == "c7x_stream_open") {
	    const VarNode* config_var = Downcast<Var>(call->args[1]).get();
	    const std::string& engine = Downcast<StringImm>(call->args[2]).get()->value;
	    PrimExpr buf_ptr = call->args[3];
        const std::string vid = GetVarID(config_var);

        this->PrintIndent();
        stream << "__" << engine << "_OPEN(";
        if (engine.compare(0, 2, "SE") == 0) {
          stream << "(void *)(";
              PrintExpr(buf_ptr, stream);
          stream  << "), ";
        }
	    stream << vid << ".params());\n";
        return;
      }
      else if (func == "c7x_stream_close") {
	    const std::string& engine = Downcast<StringImm>(call->args[2]).get()->value;
        this->PrintIndent();
        stream << "__" << engine << "_CLOSE();\n";
        return;
      }
    }
  }
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << "(void)" << vid << ";\n";
  }
}


void CodeGenC7x::PrintGetFuncFromBackend(const std::string& func_name,
                                           const std::string& packed_func_name) {
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "if (TVMBackendGetFuncFromEnv(" << module_name_ << ", \"" << func_name << "\""
               << ", &" << packed_func_name << ") != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenC7x::PrintFuncCall(const std::string& packed_func_name, int num_args) {
  this->PrintIndent();
  std::string ret_val = GetUniqueName("ret_val");
  std::string ret_type_code = GetUniqueName("ret_type_code");
  this->stream << "TVMValue " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();
  this->stream << "if (TVMFuncCall(" << packed_func_name << ", "
               << "(TVMValue*) stack_value"
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

// adapted
void CodeGenC7x::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  //ICHECK_EQ(scope, "global");
}


template <typename T>
inline void CodeGenC7x::PrintTernaryCondExpr(const T* op, const char* compare,
                                               std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.dtype());

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

runtime::Module BuildC7x(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenC7x cg;
  cg.Init(output_ssa, emit_asserts, target->str());

  // debug
  if (getenv("TIDL_C7X_CODEGEN_DEBUG_BEGIN")) {
    LOG_INFO << "BuildC7x";
    LOG_INFO << PrettyPrint(mod);
  }

  Map<String, LinkedParam> linked_params;
  // bool found_linked_params = false;
  bool could_have_linked_params = target->GetAttr<Bool>("link-params").value_or(Bool(false));
  for (auto kv : mod->functions) {
    if (could_have_linked_params &&
        kv.first->name_hint == ::tvm::runtime::symbol::tvm_lookup_linked_param) {
      Map<String, ObjectRef> attrs_dict = Downcast<Map<String, ObjectRef>>(kv.second->attrs->dict);
      CHECK(attrs_dict.find(::tvm::tir::attr::kLinkedParams) != attrs_dict.end())
          << "no " << ::tvm::tir::attr::kLinkedParams << " attribute found!";
      linked_params =
          Downcast<Map<String, LinkedParam>>(attrs_dict[::tvm::tir::attr::kLinkedParams]);
      //found_linked_params = true;
      continue;
    }

    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodegenC7x: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }

  cg.PrintTrailer();

  #if 0
  if (could_have_linked_params) {
    ICHECK(found_linked_params) << "-link-params given but none found";
    cg.LinkParameters(linked_params);
  }
  #endif

  if (target->GetAttr<Bool>("system-lib").value_or(Bool(false))) {
    ICHECK_EQ(target->GetAttr<String>("runtime").value_or(""), "c")
        << "c target only supports generating C runtime SystemLibs";
  }

  std::string code = cg.Finish();
  if (getenv("TIDL_C7X_CODEGEN_DEBUG_BEGIN"))
    LOG_INFO << "Code:\n" << code;
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_REGISTER_GLOBAL("target.build.c7x").set_body_typed(BuildC7x);
}  // namespace codegen
}  // namespace tvm
