// SPDX-License-Identifier: BSD-3-Clause
// Read-only q1 PV inspection after the complete original N32/GQA stages.
// Deliberately stops before bufferization; emits no model object.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#ifdef QWEN_ORDERED_PV_REGISTER_PREFLIGHT
mlir::LogicalResult diagnoseQwenOrderedPVRegister(mlir::ModuleOp module,
    llvm::ArrayRef<mlir::scf::ForOp> loops);
#endif
namespace {
using namespace mlir;
bool admitted(func::FuncOp op) {
  if (op.isDeclaration() || op.getSymName() != "Qwen3DynamicBoundProbe" ||
      op.getNumArguments() != 62 || op.getNumResults() != 57) return false;
  auto tokens = dyn_cast<RankedTensorType>(op.getArgument(0).getType());
  auto order = op->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
  auto ready = op->getAttrOfType<BoolAttr>("qwen.gqa_identity_view_complete");
  return tokens && tokens.getShape() == ArrayRef<int64_t>({1, 1}) &&
      tokens.getElementType().isInteger(64) && order &&
      order.getValue() == "qk-fma-pv-openblas-armv8-k1024" && ready && ready.getValue();
}
}
mlir::LogicalResult inspectQwenOrderedPVLoops(mlir::ModuleOp module) {
  using namespace mlir;
  SmallVector<scf::ForOp> loops;
  unsigned admittedFunctions = 0;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!admitted(function)) continue;
    ++admittedFunctions;
    for (auto &operation : function.front()) {
      auto loop = dyn_cast<scf::ForOp>(operation);
      if (!loop || loop.getNumResults() != 1) continue;
      auto result = dyn_cast<RankedTensorType>(loop.getResult(0).getType());
      // Canonicalization folds the singleton query axis before this stage.
      // Both representations describe the same physical q1 PV destination.
      if (!result || (result.getShape() != ArrayRef<int64_t>({16, 1, 128}) &&
                      result.getShape() != ArrayRef<int64_t>({16, 128})) ||
          !result.getElementType().isF32()) continue;
      unsigned fma = 0, mul = 0, add = 0;
      bool fast = false;
      loop.walk([&](Operation *op) {
        if (auto x = dyn_cast<math::FmaOp>(op)) {
          ++fma;
          fast |= x.getFastmath() != arith::FastMathFlags::none;
        }
        mul += isa<arith::MulFOp>(op);
        add += isa<arith::AddFOp>(op);
      });
      if (fma == 1 && !mul && !add && !fast) loops.push_back(loop);
    }
  }
  if (!admittedFunctions) return success();
  llvm::errs() << "QWEN_Q1_PV_LOOP_PREFLIGHT matched=" << loops.size()
               << " admitted_functions=" << admittedFunctions << "\n";
  if (admittedFunctions != 1 || loops.size() != 28) {
    module.emitError("Complete original all28 ordered q1 PV loops required");
    return failure();
  }
  unsigned preorder = 0;
  loops.front().walk([&](scf::ForOp loop) {
    llvm::errs() << "QWEN_Q1_PV_ACTUAL_LOOP preorder=" << preorder++ << " bounds=";
    for (Value value : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
      if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
        llvm::errs() << constant.value();
      else {
        llvm::errs() << "dynamic[";
        if (auto defining = value.getDefiningOp())
          defining->print(llvm::errs(), OpPrintingFlags().useLocalScope());
        llvm::errs() << "]";
      }
      llvm::errs() << ";";
    }
    llvm::errs() << "\n";
  });
  llvm::errs() << "QWEN_Q1_PV_FIRST_BEGIN\n";
  loops.front()->print(llvm::errs(), OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs() << "\nQWEN_Q1_PV_FIRST_END\n";
#ifndef QWEN_ORDERED_PV_REGISTER_MODEL
  llvm::errs() << "QWEN_Q1_PV_PREFLIGHT_STOP_NO_MODEL_CODE\n";
#endif
  #ifdef QWEN_ORDERED_PV_REGISTER_PREFLIGHT
  if (failed(diagnoseQwenOrderedPVRegister(module, loops))) return failure();
  #endif
#ifdef QWEN_ORDERED_PV_REGISTER_MODEL
  llvm::errs()<<"QWEN_Q1_PV_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE\n";
  return success();
#else
  return failure();
#endif
}
