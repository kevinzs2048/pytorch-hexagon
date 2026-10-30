// RUN: not qwen-policy-sums-opt %s --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' 2>&1 | FileCheck %s

// CHECK: hexagon.fp32_dynamic_rmsnorm_max_q requires 1..32 : i64
func.func @invalid_limit()
    attributes {hexagon.fp32_dynamic_rmsnorm_max_q = 33 : i64} {
  return
}
