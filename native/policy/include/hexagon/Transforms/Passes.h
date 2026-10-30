// SPDX-License-Identifier: BSD-3-Clause
#ifndef QWEN_POLICY_PASSES_H
#define QWEN_POLICY_PASSES_H
#include "hexagon/Transforms/Transforms.h"
namespace mlir::hexagon {
#define GEN_PASS_REGISTRATION
#include "hexagon/Transforms/Passes.h.inc"
}
#endif
