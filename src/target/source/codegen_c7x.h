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

namespace tvm {
namespace codegen {

using namespace tir;

class CodeGenC7x final : public CodeGenC {
 public:
  CodeGenC7x();
  void Init(bool output_ssa, bool emit_asserts, std::string target_str);

  void AddFunction(const PrimFunc& f);
  void InitFuncState(const PrimFunc& f) override;
  void PreFunctionBody(const PrimFunc& f) override;

  /*! \brief Add linked parameters, if they are present. */
  //void LinkParameters(Map<String, LinkedParam> params);

  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintType(const Type& type, std::ostream& os);  // NOLINT(*)
  void PrintFuncPrefix() final;                        // NOLINT(*)
  void PrintFinalReturn() final;                       // NOLINT(*)

  // expression visitors
  void VisitExpr_(const VarNode* op, std::ostream& os) override;        // NOLINT(*)
  void VisitExpr_(const LoadNode* op, std::ostream& os) override;       // NOLINT(*)
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
  void VisitStmt_(const StoreNode* op) override;
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
  std::string GetVecLoad(DataType t, const VarNode* buffer, PrimExpr base) override;
  // print vector store
  void PrintVecStore(const VarNode* buffer, DataType t, PrimExpr base,
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
  std::string module_name_;
  /* \brief tracks declared global variables which live despite GetUniqueName */
  std::set<std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;

  /* \brief names of variables that are used as src or dst in dma intrinsics */
  std::set<const VarNode*> dma_buffers_;
  /* \brief map of dma buffer variables to dma manager objects */
  std::map<const VarNode*, Var> dma_map_;

  bool IsDMA(Var var) {
    return dma_buffers_.find(var.get()) != dma_buffers_.end();
  }

  bool IsLocal(Var var) { 
    auto it = alloc_storage_scope_.find(var.get());
    return it != alloc_storage_scope_.end() && it->second == "local";
  }

  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void PrintFuncCall(const std::string& packed_func_name, int num_args);
  void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*)
  void PrintDMASetup(Var dma_var, Call call);

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
