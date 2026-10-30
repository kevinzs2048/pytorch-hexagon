// SPDX-License-Identifier: BSD-3-Clause
// Linker wrapper: keep the original q1 compiler/PV guard, add only the exact
// target-tested QK transform at its already observed inspection boundary.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

mlir::LogicalResult guardQwenPhysicalQK(mlir::ModuleOp,mlir::scf::ForOp);
mlir::LogicalResult originalQwenPVInspection(mlir::ModuleOp)
    __asm__("__real__Z25inspectQwenOrderedPVLoopsN4mlir8ModuleOpE");
mlir::LogicalResult guardedQwenQKInspection(mlir::ModuleOp)
    __asm__("__wrap__Z25inspectQwenOrderedPVLoopsN4mlir8ModuleOpE");

mlir::LogicalResult guardedQwenQKInspection(mlir::ModuleOp module) {
  using namespace mlir;SmallVector<scf::ForOp> heads;unsigned functions=0;
  for(auto function:module.getOps<func::FuncOp>()) {
    if(function.isDeclaration()||function.getSymName()!="Qwen3DynamicBoundProbe")continue;
    if(function.getNumArguments()!=62||function.getNumResults()!=57)return failure();
    auto tokens=dyn_cast<RankedTensorType>(function.getArgument(0).getType());
    auto order=function->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
    auto ready=function->getAttrOfType<BoolAttr>("qwen.gqa_identity_view_complete");
    if(!tokens||tokens.getShape()!=ArrayRef<int64_t>({1,1})||!tokens.getElementType().isInteger(64)
        ||!order||order.getValue()!="qk-fma-pv-openblas-armv8-k1024"||!ready||!ready.getValue())
      return failure();
    ++functions;
    for(Operation &op:function.front()) {
      auto head=dyn_cast<scf::ForOp>(op);if(!head||head.getNumResults()!=1)continue;
      auto type=dyn_cast<RankedTensorType>(head.getResult(0).getType());
      if(!type||!type.getElementType().isF32()
          ||type.getShape()!=ArrayRef<int64_t>({16,ShapedType::kDynamic}))continue;
      unsigned fmas=0;bool fast=false;
      head.walk([&](math::FmaOp f){++fmas;fast|=f.getFastmath()!=arith::FastMathFlags::none;});
      if(fmas==1&&!fast)heads.push_back(head);
    }
  }
  if(functions!=1||heads.size()!=28) {
    module.emitError("Exactly the observed 28 original q1 QK recurrences required");return failure();
  }
  for(auto head:heads)if(failed(guardQwenPhysicalQK(module,head)))return failure();
  if(failed(verify(module)))return failure();
  llvm::errs()<<"QWEN_Q1_QK_IDENTICAL_TAIL verified=28 query_FP_cloned_unchanged=1 original_FMA_per_component=1 read_only_guard=1 model_ABI_unchanged=1\n";
  llvm::errs()<<"QWEN_Q1_QK_GUARDED_FIRST_BEGIN\n";
  heads.front()->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs()<<"\nQWEN_Q1_QK_GUARDED_FIRST_END\n";
#ifdef QWEN_QK_MODEL_PREFLIGHT
  llvm::errs()<<"QWEN_Q1_QK_PREFLIGHT_STOP_BEFORE_MODEL_CODEGEN\n";
  module.emitError("QK model structural preflight deliberately stops before codegen");
  return failure();
#else
  // The existing inspection and subsequent PV transform remain in their
  // original order, with every original object and runtime helper unchanged.
  return originalQwenPVInspection(module);
#endif
}
