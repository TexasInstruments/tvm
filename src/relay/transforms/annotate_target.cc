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
 * \file src/relay/transforms/annotate_target.cc
 * \brief Wraps an expr with compiler_begin and compiler_end to indicate that
 * this expr should be handled by the external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include "pass_utils.h"

namespace tvm {
namespace relay {
namespace annotate_target {

static const PackedFunc* make_begin_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
static const PackedFunc* make_end_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_end");
static const char default_target[] = "default";
// A helper class to insert annotation boundaries for all the ops of a program
// region that will be handled by a specific compiler.
class AnnotateTargetRewriter : public ExprRewriter {
 public:
  explicit AnnotateTargetRewriter(Array<runtime::String> targets) : targets_(std::move(targets)) {}

 protected:
  /*! \brief The target backends for annotation. */
  Array<runtime::String> targets_;
  /*! \brief Maintain the decision of the target for each op expr. */
  std::unordered_map<Expr, std::string, ObjectPtrHash, ObjectPtrEqual> op_expr_to_target_;

  /*!
   * \brief This function annotates a compiler end and a compiler begin to all arguments.
   *
   *  The compiler end is based on the arg target while the compiler begin is based on the given
   *  target. If target is not given and all arguments are going to the same target, then we will
   *  use that target; otherwise we use default for this op. Note that all arg exprs must be
   *  available in op_expr_to_target before calling this function.
   *
   * \param args An array of arguments of the given node.
   * \param target The target of the current node.
   * \return A pair of target and annotated argument expressions.
   */
  std::pair<std::string, Array<Expr>> AnnotateArgs(const Array<Expr>& args,
                                                   const std::string& target = "") {
    std::string ref_target = "";
    Array<Expr> compiler_begins;
    Array<Expr> compiler_ends;
    // Begin TI
    int args_present = 0; /* Begin annotation is added only if at least one arg is present */
    // End TI
    for (auto arg : args) {
      // Begin TI
      args_present = 1;
      // End TI
      std::string arg_target = default_target;
      const CallNode* call = arg.as<CallNode>();

      if (call && call->op == CompilerBeginOp()) {
        // Argument is already compiler begin node meaning that this is not the first time
        // running this pass, so we simply remove it and will add a new one later.
        ICHECK_EQ(call->args.size(), 1U);
        // Do not alter existing annotation if not default
        if (default_target != call->attrs.as<CompilerAttrs>()->compiler) {
          compiler_begins.push_back(arg);
        } else {
          // Remove default
          compiler_ends.push_back(call->args[0]);
        }
        const CallNode* end = call->args[0].as<CallNode>();
        if (end && end->op == CompilerEndOp()) {
          arg_target = end->attrs.as<CompilerAttrs>()->compiler;
        }
      } else if (op_expr_to_target_.find(arg) != op_expr_to_target_.end()) {
        arg_target = op_expr_to_target_[arg];
        // If an argument is a call node and has no argument, then it should be tensor ops such as
        // zeros, so we treat it as input vars.
        if (call && call->args.size() == 0) {
          compiler_ends.push_back(arg);
        } else {
          // Begin TI - CODEGEN-14572
          const TupleNode* tuple = arg.as<TupleNode>();
          if (tuple) {
            std::string tuple_target = "";
            // If current CallNode target is default, the input tuple should also be assigned default target.
            if (target == "default")
            {
              tuple_target = target;
            }
            // Recursive call to tuples' args
            auto tuple_target_n_args = AnnotateArgs(tuple->fields, tuple_target); // Ensure tuple target is passed in case it is updated above, else pass "" default
            auto annotated_tuple = WithFields(Downcast<Tuple>(arg), std::get<1>(tuple_target_n_args));
            // Update tuple target returned by AnnotateArgs (based on tuples' args targets)
            tuple_target = std::get<0>(tuple_target_n_args);
            op_expr_to_target_[annotated_tuple] = tuple_target;
            compiler_ends.push_back(InsertAnnotation(annotated_tuple, tuple_target, make_end_op)); // Add end annotation for tuple
          } 
          // End TI
          else {
            compiler_ends.push_back(InsertAnnotation(arg, arg_target, make_end_op));
          }
        }
      } else {
        // Input vars.
        
        // Begin TI - CODEGEN-14572
        /** Input vars are assigned default_target ("default"). 
         * 2 possible cases arise in this function :
         * 1. "target" != "" in function arguments - 
         *    Current expression's target passed as function argument is assigned to all input vars (refer op_target below)
         * 2. "target" == "" in function arguments (typically occurs in case of Tuple)
         *    In this case, op_target (current expression's target) needs to be determined. It is determined as follows:
         *    Each args target is set as "ref_target", in case of mismatch between args targets, ref_target = "default"
         *    So effectively, input vars (this else condition) are always assigned "default" target which is in turn used to determine op_target
         *    ==> Tuple with input vars will always get "default" target (irrespective of other inputs) --> incorrect
         * Solution to case 2: 
         *    Remove input vars from target determination process altogether by setting their target as "", exclude "" from relevant targets for 
         *    op_target determination
         * 
         * Example annotation before update (not the "default" compiler):
         *  %204 = annotation.compiler_begin(%203, compiler="tidl");
            %205 = transpose(%204, axes=[0, 2, 1]);
            %206 = annotation.compiler_end(%205, compiler="tidl");
            %207 = annotation.compiler_begin(meta[relay.Constant][98] , compiler="default");
            %208 = annotation.compiler_begin(%206, compiler="default");
            %209 = (%207, %208);
            %210 = annotation.compiler_end(%209, compiler="default");
            %211 = annotation.compiler_begin(%210, compiler="tidl");
            %212 = concatenate(%211, axis=1);
            %213 = annotation.compiler_end(%212, compiler="tidl");
         * 
           Example annotation after update:
         *  %204 = annotation.compiler_begin(%203, compiler="tidl");
            %205 = transpose(%204, axes=[0, 2, 1]) ;
            %206 = annotation.compiler_end(%205, compiler="tidl") ;
            %207 = annotation.compiler_begin(meta[relay.Constant][98] , compiler="tidl");
            %208 = annotation.compiler_begin(%206, compiler="tidl");
            %209 = (%207, %208) ;
            %210 = annotation.compiler_end(%209, compiler="tidl") ;
            %211 = annotation.compiler_begin(%210, compiler="tidl") ;
            %212 = concatenate(%211, axis=1);
            %213 = annotation.compiler_end(%212, compiler="tidl");
        */
        arg_target = "";
        
        // End TI
        compiler_ends.push_back(arg);
      }

      // Maintain reference target in case the target of the current node is unassigned.
      if (ref_target == "") {
        ref_target = arg_target;
      } else if ((ref_target != arg_target) /* Begin TI */ && (arg_target != "") /* End TI */) {
        ref_target = default_target;
      }
    }

    // Determine compiler begin target.
    std::string op_target = (target == "") ? ref_target : target;

    // Begin TI
    if (op_target == "") // Neither of the input args has target populated (can happen in concat unit test kind of scenario )
    {
      op_target = default_target; // We don't want op_target to be "" in annotation
    }
    // End TI

    #if 0
    if (ref_target != "") {
    // Begin TI
    #else
    if (args_present) {
    #endif
    // End TI
      for (const auto& end : compiler_ends) {
        compiler_begins.push_back(InsertAnnotation(end, op_target, make_begin_op));
      }
    } else {
      return {op_target, args};
    }
    return {op_target, compiler_begins};
  }

  Expr InsertAnnotation(const Expr& expr, const std::string& target, const PackedFunc* ann_op) {
    Expr new_op = (*ann_op)(expr, target);
    new_op->checked_type_ = expr->checked_type_;
    return new_op;
  }

  Expr InsertCompilerEndAndPropogateTarget(const Expr& expr) {
    /*!
     * \brief This function inserts compiler end to expr and maps the corresponding target to the
     * new expression.
     *
     *  This function checks for expr existence within the map and inserts the annotation.
     *  If the expression has a free variable (e.g: relay.zeros, relay.ones) we do not insert
     *  compiler end, since there are no compiler begins for it.
     *  Further, it propagates the target to the new expression and returns it
     *
     * \param expr A relay expression
     * \return An annotated and target-propagated relay expression.
     */
    Expr new_expr = expr;
    const CallNode* call = expr.as<CallNode>();
    const TupleNode* tup = expr.as<TupleNode>();
    if (op_expr_to_target_.find(expr) != op_expr_to_target_.end()) {
      // Check whether expr has args, if not - do not insert compiler_end.
      if (expr->IsInstance<RefWriteNode>() || expr->IsInstance<RefCreateNode>() ||
          expr->IsInstance<RefReadNode>() || expr->IsInstance<TupleGetItemNode>() ||
          (call && !call->args.empty()) || (tup && !tup->fields.empty())) {
        
        // Begin TI - Handle final tuples that have no consumer CallNode
        if (tup && !tup->fields.empty()) {
          // Process tuple fields using AnnotateArgs to ensure they get proper annotations
          // This handles the case where tuple is the final expression with no consumer CallNode
          std::string tuple_target = op_expr_to_target_[expr];
          auto target_n_args = AnnotateArgs(tup->fields, tuple_target);
          auto annotated_tuple = WithFields(Downcast<Tuple>(expr), std::get<1>(target_n_args));
          op_expr_to_target_[annotated_tuple] = tuple_target;
          new_expr = InsertAnnotation(annotated_tuple, tuple_target, make_end_op);
          op_expr_to_target_[new_expr] = tuple_target;
        } else {
          // End TI
          std::string target = op_expr_to_target_[new_expr];
          new_expr = InsertAnnotation(new_expr, target, make_end_op);
          op_expr_to_target_[new_expr] = target;
          // Begin TI
        }
        // End TI
      }
    } else if (call && call->op == CompilerEndOp()) {
      if (default_target == call->attrs.as<CompilerAttrs>()->compiler) {
        ICHECK_EQ(call->args.size(), 1U);
        new_expr = call->args[0];
        std::string target = op_expr_to_target_[new_expr];
        new_expr = InsertAnnotation(new_expr, target, make_end_op);
        op_expr_to_target_[new_expr] = target;
      }
    }

    return std::move(new_expr);
  }

 public:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    // Supported targets for this node. The order implies the priority.
    std::vector<std::string> supported_targets;

    auto op_node = pre->op.as<OpNode>();

    // This graph has annotations, meaning that this is not the first time running this pass.
    if (op_node && pre->op == CompilerBeginOp()) {
      // Bypass compiler begin due to lack of target information. It will be processed
      // when the following op handling arguments.
      ICHECK_EQ(pre->args.size(), 1U);
      // Preserve annotations
      return post;
    } else if (op_node && pre->op == CompilerEndOp()) {
      // Override compiler end with the new target.
      ICHECK_EQ(pre->args.size(), 1U);
      auto input_expr = post.as<CallNode>()->args[0];
      // Already annotated. Recover target
      if (op_expr_to_target_.find(input_expr) == op_expr_to_target_.end()) {
        op_expr_to_target_[input_expr] = post.as<CallNode>()->attrs.as<CompilerAttrs>()->compiler;
      }
      ICHECK(op_expr_to_target_.find(input_expr) != op_expr_to_target_.end());
      // Preserve annotated nodes
      return post;
    }
    // Check prior to peeking first argument
    if (pre->args.size()) {
      // Peek the first argument. If it is compiler begin then this node had annotated by
      // another target before, so we also consider that target as a supported target.
      const CallNode* first_arg_call = pre->args[0].as<CallNode>();
      if (first_arg_call && first_arg_call->op == CompilerBeginOp()) {
        std::string arg_target = first_arg_call->attrs.as<CompilerAttrs>()->compiler;
        if (arg_target != default_target) {
          // annotated already
          return post;
        }
      }
    }

    // Check which targets this op can be offloaded.
    if (op_node) {
      // TVM operators: Check target specific op checking function and add to supported_targets
      // if it is supported.
      Op op = Downcast<Op>(pre->op);
      ICHECK(op.defined());
      for (const auto& target : this->targets_) {
        if (!Op::HasAttrMap("target." + std::string(target))) {
          continue;
        }
        auto fannotate = Op::GetAttrMap<FTVMAnnotateTarget>("target." + std::string(target));
        const Expr& ex = GetRef<Expr>(pre);
        if (fannotate.count(op) && fannotate[op](ex)) {
          supported_targets.push_back(target);
        }
      }
    } else if (pre->op->IsInstance<FunctionNode>()) {
      // Composite function: Add the target of a composite function to supported_targets
      // if it is in the target list.
      Function func = Downcast<Function>(pre->op);
      ICHECK(func.defined());
      if (auto comp_name = func->GetAttr<String>(attr::kComposite)) {
        std::string comp_name_str = comp_name.value();
        size_t i = comp_name_str.find('.');
        if (i != std::string::npos) {
          std::string comp_target = comp_name_str.substr(0, i);
          for (const auto& target : this->targets_) {
            if (std::string(target) == comp_target) {
              supported_targets.push_back(comp_target);
              break;
            }
          }
        }
      }
    }
    supported_targets.push_back(default_target);  // Make default as the last option.
    // Visit and mutate arguments after the target of this op has been determined.
    Call post_call = Downcast<Call>(post);
    if (pre->op->IsInstance<VarNode>()) {
      auto new_call = RewriteVarCall(post_call);
      if (nullptr != new_call) return GetRef<Expr>(new_call->get());
    }
    // TODO(@comaniac, @zhiics): Now we simply assign this node to the target with
    // the highest priority, but we should preserve all supported targets so that
    // we can make a better decision.
    std::string target = supported_targets[0];

    // Add annotations to each arg.
    auto target_n_args = AnnotateArgs(post_call->args, target);
    Array<Expr> compiler_begins = std::get<1>(target_n_args);
    Call new_call = Call(post_call->op, compiler_begins, post_call->attrs);
    new_call->checked_type_ = pre->checked_type_;
    new_call->span = pre->span;

    // Update the target map.
    op_expr_to_target_[new_call] = target;
    return std::move(new_call);
  }

  virtual std::unique_ptr<Call> RewriteVarCall(const Call& post_call) { return nullptr; }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) override {
    auto tuple = Downcast<Tuple>(post);
    #if 0
    auto target_n_args = AnnotateArgs(tuple->fields);
    auto new_expr = WithFields(tuple, std::get<1>(target_n_args));
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
    
    #else // Begin TI - CODEGEN-14572
    /** Tuple node target determination process:
     * 1. Look at inputs. If all have same target, assign it, in case of target mismatch tuple gets "default" target
     *    Constants are excluded from this determination process.
     * 2. Just inputs are not sufficient. e.g. case when output is default target, having tuple as TIDL target does not make sense.
     *    It also impacts constant inputs to tuples. Consider following problematic case
     *    (Tuple output - Default, Tuple - TIDL, Tuple input call - TIDL, Constant tuple input - TIDL) 
     *    Due to FlattenTupleOutputs pass in partitioner, tuple's compiler end annotation gets passed to each of its inputs
     *    In this case, the Constant input is encased by TIDL compiler_begin/compiler_end annotations resulting it to be pulled in a function
     *    In such a case, it is logical to force both tuple and the constant inputs to default target
     * 
     * Solution : Instead of processing a tuple node and inputs here, process tuple node and its inputs as part of
     * tuple node's consumer CallNode, where the consumer's target would be known and can be used to force "default" target
     * to tuple and its constant input if required. Here, just save tuple's target based on field consensus and return tuple to 
     * keep the graph parsing undisturbed. The actual annotation will be applied when this tuple is used as argument in CallNode.
     * This target is a safety net - it is recalculated by InsertCompilerEndAndPropogateTarget if tuple is final expression
     */
    std::string ref_target = "";
    for (auto field : tuple->fields) {
      if (op_expr_to_target_.find(field) != op_expr_to_target_.end()) {
        std::string field_target = op_expr_to_target_[field];
        if (ref_target == "") {
          ref_target = field_target;
        } else if (ref_target != field_target && field_target != "") {
          ref_target = default_target;
        }
      }
    }
    std::string tuple_target = (ref_target == "") ? default_target : ref_target;
    op_expr_to_target_[tuple] = tuple_target;
    
    return tuple;
    #endif
    // End TI
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) override {
    auto expr = Downcast<TupleGetItem>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->tuple}));
    auto new_expr = TupleGetItem(std::get<1>(target_n_args)[0], expr->index);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const FunctionNode* fn, const Expr& post) override {
    Function func;
    Expr new_body;
    // don't step into composite functions
    if (fn->GetAttr<String>(attr::kComposite).defined()) {
      func = GetRef<Function>(fn);
      new_body = func->body;
    } else {
      func = Downcast<Function>(post);
      new_body = InsertCompilerEndAndPropogateTarget(func->body);
    }
    return WithFields(func, func->params, new_body);
  }

  Expr Rewrite_(const LetNode* op, const Expr& post) override {
    auto let = Downcast<Let>(post);

    Expr new_expr;
    std::pair<std::string, Array<Expr>> target_n_args;
    Expr new_body = InsertCompilerEndAndPropogateTarget(let->body);
    // Do not annotate function literal with let binding.
    if (let->value->IsInstance<FunctionNode>()) {
      new_expr = Let(let->var, let->value, new_body);
    } else {
      target_n_args = AnnotateArgs({let->value});
      new_expr = Let(let->var, std::get<1>(target_n_args)[0], new_body);
    }

    return std::move(new_expr);
  }

  Expr Rewrite_(const IfNode* op, const Expr& post) override {
    auto expr = Downcast<If>(post);
    Expr new_cond = InsertCompilerEndAndPropogateTarget(expr->cond);
    Expr new_true_branch = InsertCompilerEndAndPropogateTarget(expr->true_branch);
    Expr new_false_branch = InsertCompilerEndAndPropogateTarget(expr->false_branch);

    auto new_expr = If(new_cond, new_true_branch, new_false_branch);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefCreateNode* op, const Expr& post) override {
    auto expr = Downcast<RefCreate>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->value}));
    auto new_expr = RefCreate(std::get<1>(target_n_args)[0]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefReadNode* op, const Expr& post) override {
    auto expr = Downcast<RefRead>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->ref}));
    auto new_expr = RefRead(std::get<1>(target_n_args)[0]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefWriteNode* op, const Expr& post) override {
    auto expr = Downcast<RefWrite>(post);

    auto target_n_args = AnnotateArgs(Array<Expr>({expr->ref, expr->value}));
    auto new_expr = RefWrite(std::get<1>(target_n_args)[0], std::get<1>(target_n_args)[1]);
    op_expr_to_target_[new_expr] = std::get<0>(target_n_args);
    return std::move(new_expr);
  }
};

// A helper class to insert annotation boundaries for call ops and function nodes
// in a program region that will be handled by a specific compiler.
class CallOpsTargetRewriter : public AnnotateTargetRewriter {
 public:
  explicit CallOpsTargetRewriter(Array<runtime::String> targets)
      : AnnotateTargetRewriter(std::move(targets)) {}

  std::unique_ptr<Call> RewriteVarCall(const Call& post_call) override {
    Array<Expr> ends;
    for (auto arg : post_call->args) {
      ends.push_back(InsertCompilerEndAndPropogateTarget(arg));
    }
    auto new_call = std::make_unique<Call>(post_call->op, ends, post_call->attrs);
    (*new_call)->checked_type_ = post_call->checked_type_;
    return new_call;
  }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) override {
    auto tuple = Downcast<Tuple>(post);
    Array<Expr> new_fields;
    new_fields.reserve(tuple->fields.size());

    for (auto f : tuple->fields) {
      new_fields.push_back(InsertCompilerEndAndPropogateTarget(f));
    }
    return WithFields(tuple, new_fields);
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) override {
    auto expr = Downcast<TupleGetItem>(post);
    return std::move(TupleGetItem(InsertCompilerEndAndPropogateTarget(expr->tuple), expr->index));
  }

  Expr Rewrite_(const IfNode* op, const Expr& post) override {
    auto expr = Downcast<If>(post);
    Expr new_cond = InsertCompilerEndAndPropogateTarget(expr->cond);
    Expr new_true_branch = InsertCompilerEndAndPropogateTarget(expr->true_branch);
    Expr new_false_branch = InsertCompilerEndAndPropogateTarget(expr->false_branch);

    auto new_expr = If(new_cond, new_true_branch, new_false_branch);
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefCreateNode* op, const Expr& post) override {
    auto expr = Downcast<RefCreate>(post);
    auto new_expr = RefCreate(InsertCompilerEndAndPropogateTarget(expr->value));
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefReadNode* op, const Expr& post) override {
    auto expr = Downcast<RefRead>(post);
    auto new_expr = RefRead(InsertCompilerEndAndPropogateTarget(expr->ref));
    return std::move(new_expr);
  }

  Expr Rewrite_(const RefWriteNode* op, const Expr& post) override {
    auto expr = Downcast<RefWrite>(post);
    auto new_expr = RefWrite(InsertCompilerEndAndPropogateTarget(expr->ref),
                             InsertCompilerEndAndPropogateTarget(expr->value));
    return std::move(new_expr);
  }
};

Expr AnnotateTarget(const Expr& expr, const Array<runtime::String>& targets,
                    bool include_non_call_ops) {
  auto r = include_non_call_ops ? std::make_unique<AnnotateTargetRewriter>(targets)
                                : std::make_unique<CallOpsTargetRewriter>(targets);
  return PostOrderRewrite(expr, r.get());
}

}  // namespace annotate_target

namespace transform {

Pass AnnotateTarget(const Array<runtime::String>& targets, bool include_non_call_ops) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::annotate_target::AnnotateTarget(f, targets, include_non_call_ops));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateTargetFunc", {"InferType"});
  return transform::Sequential({func_pass, InferType()}, "AnnotateTarget");
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateTarget").set_body_typed(AnnotateTarget);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
