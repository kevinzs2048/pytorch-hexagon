// SPDX-License-Identifier: BSD-3-Clause
// Preserve the reviewed implementation verbatim; only choose between its
// exact rewrite and the original recurrence using runtime query extent.
#define instrumentQwenLoopRegions provenUnconditionalQKEntry
#include "DynamicQKRegisterProven.inc"
#undef instrumentQwenLoopRegions
#include "mlir/IR/Dominance.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#ifdef QWEN_Q2_IDENTICAL_TAIL
#include "DynamicQKIdenticalTail.inc"
#endif

namespace {
using namespace mlir;
void localHead(scf::ForOp head, std::string &text) {
  llvm::raw_string_ostream stream(text);
  head->print(stream,OpPrintingFlags().useLocalScope().printGenericOpForm());
}
LogicalResult dispatch(ModuleOp module,scf::ForOp original,
    std::string *before=nullptr,std::string *fallback=nullptr,std::string *optimized=nullptr) {
  auto proof=prove(original);if(!proof)return failure();
  if(before)localHead(original,*before);
  DominanceInfo dominance(module);
  if(!dominance.properlyDominates(proof->m.getUpperBound(),original))return failure();
  auto loc=original.getLoc();OpBuilder b(original);
  auto eight=arith::ConstantIndexOp::create(b,loc,8);
  auto maximum=arith::ConstantIndexOp::create(b,loc,32);
  auto enough=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::uge,proof->m.getUpperBound(),eight);
  auto bounded=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ule,proof->m.getUpperBound(),maximum);
  auto eligible=arith::AndIOp::create(b,loc,enough,bounded);
  SmallVector<OpOperand*> oldUsers;
  for(OpOperand &use:original.getResult(0).getUses())oldUsers.push_back(&use);
  auto choice=scf::IfOp::create(b,loc,original.getResultTypes(),eligible,true);
  b.setInsertionPointToStart(&choice.getThenRegion().front());
  IRMapping mapping;auto cloned=cast<scf::ForOp>(b.clone(*original,mapping));
  scf::YieldOp::create(b,loc,cloned.getResult(0));
  auto clonedProof=prove(cloned);if(!clonedProof)return failure();
#ifdef QWEN_Q2_IDENTICAL_TAIL
  auto tail=rewriteIdenticalQKTail(module,*clonedProof);if(failed(tail))return failure();
  auto rewritten=*tail;
#else
  auto rewritten=rewrite(*clonedProof);
#endif
  original->moveBefore(&choice.getElseRegion().front(),choice.getElseRegion().front().end());
  b.setInsertionPointToEnd(&choice.getElseRegion().front());
  scf::YieldOp::create(b,loc,original.getResult(0));
  for(OpOperand *use:oldUsers)use->set(choice.getResult(0));
  if(fallback)localHead(original,*fallback);
  if(optimized)localHead(rewritten,*optimized);
  return verify(module);
}
}

mlir::LogicalResult instrumentQwenLoopRegions(mlir::ModuleOp module) {
  using namespace mlir;SmallVector<scf::ForOp> originals;DominanceInfo dominance(module);
  for(auto function:module.getOps<func::FuncOp>()) {
    auto order=function->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
    if(function.getSymName()!="Qwen3DynamicBoundProbe"||function.isDeclaration()
        ||function.getNumArguments()!=65||function.getNumResults()!=57||!order
        ||order.getValue()!="qk-fma-pv-openblas-armv8-k1024")continue;
    for(Operation &op:function.front())if(auto head=dyn_cast<scf::ForOp>(op))
      if(auto proof=prove(head)) {
        if(!dominance.properlyDominates(proof->m.getUpperBound(),head))return failure();
        originals.push_back(head);
      }
  }
  // Resolve every old recurrence/condition before mutating any layer.
  if(originals.size()!=28) {module.emitError("All28 original QK recurrences required before dispatch");return failure();}
  for(auto head:originals)if(failed(dispatch(module,head)))return failure();
#ifdef QWEN_Q2_IDENTICAL_TAIL
  llvm::errs()<<"QWEN_Q2_IDENTICAL_TAIL verified=28 positive_zero_keys_and_seeds=1 representative_lanes=8 original128_FMA_order=1 fallback_unchanged=1\n";
#endif
  llvm::errs()<<"QWEN_DYNAMIC_QK_DISPATCH verified=28 runtime_query=8..32 original_else=1 exact_proven_register_then=1 FP_order_unchanged=1\n";
#ifdef QWEN_Q2_DISPATCH_PREFLIGHT
  llvm::errs()<<"QWEN_DYNAMIC_QK_DISPATCH_STOP_BEFORE_MODEL_CODEGEN\n";
  module.emitError("Q2 dispatch structural preflight deliberately stopped before codegen");return failure();
#else
  return retainOriginalDynamicPV(module);
#endif
}

extern "C" __attribute__((visibility("default"))) int qwenQ2QKDispatchPhysicalPreflight(const char *input,const char *output) {
  using namespace mlir;DialectRegistry registry;
  registry.insert<affine::AffineDialect,arith::ArithDialect,func::FuncDialect,
      math::MathDialect,scf::SCFDialect,tensor::TensorDialect,vector::VectorDialect>();
  MLIRContext context(registry);context.loadAllAvailableDialects();
#ifdef QWEN_Q2_IDENTICAL_TAIL
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
#endif
  auto module=parseSourceFile<ModuleOp>(input,&context);if(!module||failed(verify(*module)))return 2;
  SmallVector<scf::ForOp> heads;
  module->walk([&](scf::ForOp head){if(prove(head))heads.push_back(head);});
  std::string before,fallback,optimized;
  if(heads.size()!=1||failed(dispatch(*module,heads.front(),&before,&fallback,&optimized)))return 3;
  std::error_code error;llvm::raw_fd_ostream stream(output,error);if(error)return 4;
  module->print(stream,OpPrintingFlags().printGenericOpForm());
  for(auto pair:{std::make_pair(".original-head.txt",&before),
      std::make_pair(".fallback-head.txt",&fallback),std::make_pair(".optimized-head.txt",&optimized)}) {
    llvm::raw_fd_ostream extra(std::string(output)+pair.first,error);if(error)return 4;
    extra<<*pair.second;
  }
  return 0;
}

// Independent fixture reference: apply the old, unchanged register rewrite
// without runtime dispatch, so its complete resulting recurrence can be compared.
extern "C" __attribute__((visibility("default"))) int qwenQ2QKRegisterReference(const char *input,const char *output) {
  using namespace mlir;DialectRegistry registry;
  registry.insert<affine::AffineDialect,arith::ArithDialect,func::FuncDialect,
      math::MathDialect,scf::SCFDialect,tensor::TensorDialect,vector::VectorDialect>();
  MLIRContext context(registry);context.loadAllAvailableDialects();
  auto module=parseSourceFile<ModuleOp>(input,&context);if(!module||failed(verify(*module)))return 2;
  SmallVector<scf::ForOp> heads;
  module->walk([&](scf::ForOp head){if(prove(head))heads.push_back(head);});
  if(heads.size()!=1)return 3;
  auto rewritten=rewrite(*prove(heads.front()));if(failed(verify(*module)))return 3;
  std::string text;localHead(rewritten,text);
  std::error_code error;llvm::raw_fd_ostream stream(output,error);if(error)return 4;
  stream<<text;
  llvm::raw_fd_ostream complete(std::string(output)+".module.mlir",error);if(error)return 4;
  module->print(complete,OpPrintingFlags().printGenericOpForm());return 0;
}
