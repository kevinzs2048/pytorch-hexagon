// SPDX-License-Identifier: BSD-3-Clause
#ifndef QWEN_POLICY_TRANSFORMS_H
#define QWEN_POLICY_TRANSFORMS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
namespace mlir::hexagon {
#define GEN_PASS_DECL
#include "hexagon/Transforms/Passes.h.inc"
std::unique_ptr<OperationPass<func::FuncOp>> createStabilizeFP32SumPass();
}
#endif
