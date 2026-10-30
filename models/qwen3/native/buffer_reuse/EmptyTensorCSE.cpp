// SPDX-License-Identifier: BSD-3-Clause
// Isolated replay of the accepted SDK: retain its original empty-tensor pass
// and insert CSE before One-Shot Bufferize. Link --wrap redirects only the
// factory reference; no original SDK object or numerical pass is rebuilt.
#include "mlir/IR/BuiltinOps.h"
#ifdef QWEN_GQA_VALUE_VIEW
mlir::LogicalResult eliminateQwenValueLayoutCopy(mlir::ModuleOp module);
#endif
#ifdef QWEN_DIAGNOSTIC_LOOP_PROFILE
mlir::LogicalResult instrumentQwenLoopRegions(mlir::ModuleOp module);
#endif
#ifdef QWEN_LATE_CAPACITY_REFINEMENT
#include "mlir/Dialect/Func/IR/FuncOps.h"
void refineQwenGuardedCapacityLate(mlir::func::FuncOp op);
#endif
#ifdef QWEN_DYNAMIC_QK_COLUMN_PREFLIGHT
mlir::LogicalResult inspectQwenDynamicQKColumns(mlir::ModuleOp module);
#endif
#ifdef QWEN_DYNAMIC_QK_INVARIANT_VECTOR
mlir::LogicalResult hoistAndVectorizeQwenDynamicQK(mlir::ModuleOp module);
#endif
#ifdef QWEN_ORDERED_PV_LOOP_PREFLIGHT
mlir::LogicalResult inspectQwenOrderedPVLoops(mlir::ModuleOp module);
#endif
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

std::unique_ptr<mlir::Pass> originalEmptyTensorFactory()
    asm("__real__ZN4mlir13bufferization32createEmptyTensorEliminationPassEv");
std::unique_ptr<mlir::Pass> wrappedEmptyTensorFactory()
    asm("__wrap__ZN4mlir13bufferization32createEmptyTensorEliminationPassEv");

namespace {
struct EmptyTensorCSEPass
    : mlir::PassWrapper<EmptyTensorCSEPass,
                       mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmptyTensorCSEPass)
  llvm::StringRef getArgument() const final { return "qwen-empty-tensor-cse"; }
  llvm::StringRef getDescription() const final {
    return "Original empty-tensor elimination followed by CSE";
  }
  void runOnOperation() final {
    mlir::OpPassManager pipeline(mlir::ModuleOp::getOperationName());
    pipeline.addPass(originalEmptyTensorFactory());
    pipeline.addPass(mlir::createCSEPass());
    if (failed(runPipeline(pipeline, getOperation()))) {
      signalPassFailure();
      return;
    }
#ifdef QWEN_DYNAMIC_QK_INVARIANT_VECTOR
    if (mlir::failed(hoistAndVectorizeQwenDynamicQK(getOperation()))) {
      signalPassFailure(); return;
    }
#endif
#ifdef QWEN_DYNAMIC_QK_COLUMN_PREFLIGHT
    if (mlir::failed(inspectQwenDynamicQKColumns(getOperation()))) {
      signalPassFailure(); return;
    }
#endif
#ifdef QWEN_LATE_CAPACITY_REFINEMENT
    getOperation().walk([](mlir::func::FuncOp op) {
      refineQwenGuardedCapacityLate(op);
    });
#endif
#ifdef QWEN_DIAGNOSTIC_LOOP_PROFILE
    if (mlir::failed(instrumentQwenLoopRegions(getOperation())))
      signalPassFailure();
#endif
#ifdef QWEN_GQA_VALUE_VIEW
    if (mlir::failed(eliminateQwenValueLayoutCopy(getOperation())))
      signalPassFailure();
#endif
#ifdef QWEN_GQA_VALUE_FORWARD
    // The real model is admitted only after all28 identity views are proved.
    // Earlier wrapped CSE calls retain their original behavior.
    mlir::OpPassManager forwarding(mlir::ModuleOp::getOperationName());
    forwarding.addPass(mlir::createCSEPass());
    if (mlir::failed(runPipeline(forwarding,getOperation()))) signalPassFailure();
    auto stop=getOperation()->getAttrOfType<mlir::BoolAttr>("qwen.diagnostic_stop_after_gqa_copies");
    if(stop && stop.getValue()) {
      llvm::errs()<<"QWEN_GQA_COPY_PREFLIGHT_STOP\n";
      signalPassFailure();
    }
#endif
#ifdef QWEN_ORDERED_PV_LOOP_PREFLIGHT
    // Inspect after the complete original GQA view/forward/dead-repeat stages.
    // Always fail deliberately before bufferization; never produce model code.
    if (mlir::failed(inspectQwenOrderedPVLoops(getOperation())))
      signalPassFailure();
#endif
  }
};
} // namespace

std::unique_ptr<mlir::Pass> wrappedEmptyTensorFactory() {
  return std::make_unique<EmptyTensorCSEPass>();
}
