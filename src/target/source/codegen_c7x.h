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
 * \file codegen_c7x.h
 * \brief Generate C code for C7x.
 *
 * This is an adaptation derived from CodeGenC (the generic C backend) and
 * CodeGenCHost (the C backend for a host CPU).
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_C7X_H_
#define TVM_TARGET_SOURCE_CODEGEN_C7X_H_

#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "codegen_c.h"
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"

/* TIDL set max number of channels to 8 for generic flow, we do not expect TVM to go beyond */
#define TVM_TARGET_C7X_MAX_DMA_CHANNELS (8)

namespace tvm {
namespace codegen {

using namespace tir;

//---------------------------------------------------------------------------
// Record config parameters for a single SE/SA instance
class StreamDesc {
public:
  // Parse a c7x_stream_config call and capture the parameters
  // @tir.call_extern("c7x_stream_config", "SE", "float32",
  //                  196, 8, 1, 1, 196, 0, 0)
  StreamDesc(const VarNode* CV, const CallNode* CC) :
    config_var(CV),
    kind(Downcast<StringImm>(CC->args[1])->value),
    dtype(runtime::String2DLDataType(Downcast<StringImm>(CC->args[2])->value)),
    veclen(0) {
    int argnum = 3;
    for (int i = 0; i < 6; ++i) // generate six params since ICNT5 is max for SE/SA
      icnts[i] = CC->args[argnum++];
    for (int i = 0; i < 5; ++i)
      dims[i] = CC->args[argnum++]; // generate five params since DIM0=1 by default and DIM5 is max for SE/SA
  }
public:
  const VarNode* config_var;
  std::string kind;      // SE or SA
  DataType dtype;
  int veclen;
  PrimExpr icnts[6]; // support for ICNT0 to ICNT5
  PrimExpr dims[5];  // support for DIM1 to DIM5
  // debug
  void dump() const {
    printf("StreamDesc: config=%s kind=%s dtype=... "
           "veclen=%d icnts=... dims=...\n",
    config_var->name_hint.c_str(),
    kind.c_str(),
    veclen);
  }
};

// Parse stream access call
class StreamAccess {
public:
  // Helper function to bypass broadcast node. (The vectorizer turns all
  // the arguments into broadcast expressions)
  static PrimExpr scalar(PrimExpr e) {
    const BroadcastNode *bcst = e.as<BroadcastNode>();
    if (bcst)
      return bcst->value;
    else
      return e;
  }
  // Parse a c7x_stream_config call and capture the parameters
  // @tir.c7x_stream_access(config_var, "SE0", "pred", "adv", index_expr)
  StreamAccess(const CallNode* CC) :
    config_var(Downcast<Var>(scalar(CC->args[0])).get()),
    engine(Downcast<StringImm>(scalar(CC->args[1])).get()->value),
    pred(Downcast<StringImm>(scalar(CC->args[2])).get()->value == "pred"),
    adv(Downcast<StringImm>(scalar(CC->args[3])).get()->value == "adv"),
    index(CC->args[4].get()) {
  }

public:
  const VarNode *config_var;
  const String& engine;
  bool pred;
  bool adv;
  const PrimExprNode *index;
};

//---------------------------------------------------------------------------
// Database of stream setups for the current function
class StreamInfo {
  // Map from config varaiable to config info
  std::map<const VarNode*, StreamDesc> config_map_;
public:
  // Called during pre-scan for Let config_var = c7x_stream_config(...)
  void AddConfig(const VarNode* CV, const CallNode* CC) {
    config_map_.emplace(CV, StreamDesc(CV, CC));
    // GetDesc(CV).dump();
  }
  // Given config var, lookup config info
  StreamDesc& GetDesc(const VarNode* CV) {
    auto it = config_map_.find(CV);
    ICHECK(it != config_map_.end());
    return it->second;
  }
  // Update the vector length for a given config. The vector length is not
  // passed in the TIR config call, to avoid having to update it during
  // vectorization. Instead we run a pre-pass in the codegen to detect it.
  void UpdateVecLen(const CallNode* call) {
//    const VarNode* config_var = Downcast<Var>(access->args[1]).get();
    StreamAccess access(call);
    int lanes = call->dtype.lanes();
    GetDesc(access.config_var).veclen = lanes;
  }
  void Clear() {
    config_map_.clear();
  }
};

//---------------------------------------------------------------------------
// Customized Code Generator for C7x
// This codegen is adapted from CodeGenC.
// Partial list of customizations:
//   - generates C++ instead of C
//   - handles C7x DMA instrinsics
//   - handles C7x SE/SA instrinsics
class CodeGenC7x final : public CodeGenC {
 public:
  CodeGenC7x();
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl, std::string target_str,
            const std::unordered_set<std::string>& devices);

  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) override;
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f, bool emit_fwd_func_decl);
  void InitFuncState(const PrimFunc& f) override;
  void PreFunctionBody(const PrimFunc& f) override;
  void DeclarePackedCalls(const PrimFunc& f);

  /*! \brief Add linked parameters, if they are present. */
  //void LinkParameters(Map<String, LinkedParam> params);

  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintType(const Type& type, std::ostream& os);  // NOLINT(*)
  void PrintFuncPrefix(std::ostream& os) final;        // NOLINT(*)
  void PrintTrailer();
  void PrintRestrict(const Var& v, std::ostream& os) final;

  // expression visitors
  void VisitExpr_(const VarNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const AddNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const SubNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const MulNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const ModNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const EQNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const NENode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const LTNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const LENode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const GTNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const GENode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) override;         // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const NotNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) override;    // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) override;     // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) override;  // NOLINT(*)
  // statment vistors
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;

  Array<String> GetFunctionNames() { return function_names_; }

  // Binary vector op.
  void PrintVecBinaryOp(const std::string& op, DataType op_type, PrimExpr lhs, PrimExpr rhs,
                                std::ostream& os) override;  // NOLINT(*)
  // print vector load
  std::string GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) override;
  // print vector store
  void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                             const std::string& value) override;  // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os) override;  // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value) override;
  // Get a cast type from to
  std::string CastFromTo(std::string value, DataType from, DataType target) override;
  // Get load of single element with expression
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) override;

  /*!
   * \brief Print external function call.
   * \param ret_type The return type.
   * \param global_symbol The symbolc of the target function.
   * \param args The arguments to the function.
   * \param skip_first_arg Whether to skip the first arguments.
   * \param os The output stream.
   */
  void PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                               bool skip_first_arg, std::ostream& os) override; // NOLINT(*)

 private:
  /* \brief Internal structure to store information about function calls */
  struct FunctionInfo {
    /* \brief function name */
    std::string func_name;
    /* number of arguments required by the function */
    int64_t num_args;
    /* \brief name of resource_handle to pass */
    std::string resource_handle_name;
  };
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;
  /*! \brief whether to emit forwared function declarations in the resulting C code */
  bool emit_fwd_func_decl_;

  /* \brief names of variables that are used as src or dst in dma intrinsics */
  std::set<const VarNode*> dma_buffers_;
  /* \brief number of c7x_dma_setup() intrinsics, not necessarily (dma_buffers_.size() / 2) */
  int num_dma_intrinsics_;
  /* \brief map of dma buffer variables to dma manager objects */
  std::map<const VarNode*, const VarNode*> dma_map_;
  /* \brief SE/SA config information */
  StreamInfo stream_info_;
  /* \brief vector SE dtypes used in the condition of SelectNode */
  std::vector<DataType> sel_cond_vse_dtypes_;
  /* \brief if currently visiting vector condition of SelectNode */
  bool in_vector_cond;
  /* \brief current count of vector SE in vector condition of SelectNode */
  int  se_in_vec_cond_count;
  /* \brief Maximum global allocation across functions. Size is assigned to a global in generated code */
  size_t max_global_alloc_sz_in_bytes_;
  /* \brief Per function variable, set to true if there are any local allocations in the function */
  bool local_allocations_present_;
  /* \brief Per function variable, set to true if there are any global allocations in the function */
  bool global_allocations_present_;


  /* \brief Save variables created during codegen - this prevents them from being deallocated.
   *        This is required because these variables are not part of the IR itself.
   */
  Array<Var> new_variables_;

  // Is variable used in DMA copy-in/copy-out
  bool IsDMA(const VarNode* var) {
    return dma_buffers_.find(var) != dma_buffers_.end();
  }

  // Is variable a locally allocated buffer
  bool IsLocal(const VarNode* var) {
    auto it = alloc_storage_scope_.find(var);
    return it != alloc_storage_scope_.end() && it->second.compare(0, 5, "local") == 0;
  }

  FunctionInfo GetFunctionInfo(const CallNode* op, bool has_resource_handle);
  std::string GetPackedName(const CallNode* op);
  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void PrintFuncCall(const std::string& packed_func_name, int num_args);
  void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*)
  void PrintDMASetup(const VarNode* dma_var, const CallNode* call);
  void PrintStreamConfig(const VarNode* config_var);
  std::string PrintDLTensor(const CallNode* make_array, std::ostream& os);
  std::string MangleExternCallFuncName(String func_name, const CallNode* call=nullptr);

  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param op binary operator being expressed
   * \param compare string representation of comparison operator
   * \param os stream reference to print into
   */
  template <typename T>
  inline void PrintTernaryCondExpr(const T* op, const char* compare,
                                   std::ostream& os);  // NOLINT(*)
  /*! \brief restrict keyword */
  std::string restrict_keyword_{"__restrict__"};
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_C7X_H_
