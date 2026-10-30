// RUN: qwen-policy-sums-opt %s --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' | FileCheck %s

// Native ARM SiLU divides x by its denominator directly; retaining a rounded
// reciprocal before the multiply can change later per-token quantized codes.
#id = affine_map<(a,b,c)->(a,b,c)>
func.func private @__hexagon_silu_exp_compat(f32) -> f32

// CHECK-LABEL: func.func @dynamic
// CHECK: hexagon.fp32_arm_silu_denominator
// CHECK: arith.addf
// CHECK-NOT: arith.divf
// CHECK: linalg.yield
// CHECK: hexagon.fp32_arm_silu_division
// CHECK: ^bb0(%[[DEN0:[a-zA-Z0-9_]+]]: f32, %[[X0:[a-zA-Z0-9_]+]]: f32,
// CHECK: arith.divf %[[X0]], %[[DEN0]] : f32
// CHECK: return
func.func @dynamic(%x: tensor<1x?x3072xf32>, %up: tensor<1x?x3072xf32>) -> (tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %one = arith.constant 1.0 : f32
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x?x3072xf32>
 %empty = tensor.empty(%q) : tensor<1x?x3072xf32>
 %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%v: f32, %unused: f32):
  %neg = arith.negf %v : f32
  %exp = func.call @__hexagon_silu_exp_compat(%neg) {hexagon.fp32_silu_exp_compat} : (f32) -> f32
  %den = arith.addf %exp, %one : f32
  %inv = arith.divf %one, %den : f32
  linalg.yield %inv : f32
 } -> tensor<1x?x3072xf32>
 %silu = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%x : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 %gated = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%silu,%up : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 return %silu,%gated : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>
}

// CHECK-LABEL: func.func @singleton
// CHECK: hexagon.fp32_arm_silu_denominator
// CHECK: arith.addf
// CHECK-NOT: arith.divf
// CHECK: linalg.yield
// CHECK: hexagon.fp32_arm_silu_division
// CHECK: ^bb0(%[[DEN1:[a-zA-Z0-9_]+]]: f32, %[[X1:[a-zA-Z0-9_]+]]: f32,
// CHECK: arith.divf %[[X1]], %[[DEN1]] : f32
// CHECK: return
func.func @singleton(%x: tensor<1x1x3072xf32>, %up: tensor<1x1x3072xf32>) -> (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %one = arith.constant 1.0 : f32
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x1x3072xf32>
 %empty = tensor.empty() : tensor<1x1x3072xf32>
 %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x1x3072xf32>) outs(%empty : tensor<1x1x3072xf32>) {
 ^bb0(%v: f32, %unused: f32):
  %neg = arith.negf %v : f32
  %exp = func.call @__hexagon_silu_exp_compat(%neg) {hexagon.fp32_silu_exp_compat} : (f32) -> f32
  %den = arith.addf %exp, %one : f32
  %inv = arith.divf %one, %den : f32
  linalg.yield %inv : f32
 } -> tensor<1x1x3072xf32>
 %silu = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%x : tensor<1x1x3072xf32>,tensor<1x1x3072xf32>) outs(%empty : tensor<1x1x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x1x3072xf32>
 %gated = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%silu,%up : tensor<1x1x3072xf32>,tensor<1x1x3072xf32>) outs(%empty : tensor<1x1x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x1x3072xf32>
 return %silu,%gated : tensor<1x1x3072xf32>,tensor<1x1x3072xf32>
}

// CHECK-LABEL: func.func @unannotated
// CHECK-NOT: hexagon.fp32_arm_silu_
// CHECK: return
func.func @unannotated(%x: tensor<1x?x3072xf32>, %up: tensor<1x?x3072xf32>) -> (tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) attributes {llvm.emit_c_interface} {
 %one = arith.constant 1.0 : f32
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x?x3072xf32>
 %empty = tensor.empty(%q) : tensor<1x?x3072xf32>
 %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%v: f32, %unused: f32):
  %neg = arith.negf %v : f32
  %exp = func.call @__hexagon_silu_exp_compat(%neg) {hexagon.fp32_silu_exp_compat} : (f32) -> f32
  %den = arith.addf %exp, %one : f32
  %inv = arith.divf %one, %den : f32
  linalg.yield %inv : f32
 } -> tensor<1x?x3072xf32>
 %silu = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%x : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 %gated = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%silu,%up : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 return %silu,%gated : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>
}

// CHECK-LABEL: func.func @other_multiplier
// CHECK-NOT: hexagon.fp32_arm_silu_
// CHECK: return
func.func @other_multiplier(%x: tensor<1x?x3072xf32>, %up: tensor<1x?x3072xf32>) -> (tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %one = arith.constant 1.0 : f32
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x?x3072xf32>
 %empty = tensor.empty(%q) : tensor<1x?x3072xf32>
 %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%v: f32, %unused: f32):
  %neg = arith.negf %v : f32
  %exp = func.call @__hexagon_silu_exp_compat(%neg) {hexagon.fp32_silu_exp_compat} : (f32) -> f32
  %den = arith.addf %exp, %one : f32
  %inv = arith.divf %one, %den : f32
  linalg.yield %inv : f32
 } -> tensor<1x?x3072xf32>
 %silu = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%up : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 %gated = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%silu,%up : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 return %silu,%gated : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>
}

// CHECK-LABEL: func.func @shared_sigmoid
// CHECK-NOT: hexagon.fp32_arm_silu_
// CHECK: return
func.func @shared_sigmoid(%x: tensor<1x?x3072xf32>, %up: tensor<1x?x3072xf32>) -> (tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %one = arith.constant 1.0 : f32
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x?x3072xf32>
 %empty = tensor.empty(%q) : tensor<1x?x3072xf32>
 %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%v: f32, %unused: f32):
  %neg = arith.negf %v : f32
  %exp = func.call @__hexagon_silu_exp_compat(%neg) {hexagon.fp32_silu_exp_compat} : (f32) -> f32
  %den = arith.addf %exp, %one : f32
  %inv = arith.divf %one, %den : f32
  linalg.yield %inv : f32
 } -> tensor<1x?x3072xf32>
 %silu = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%x : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 %gated = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%up : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
 ^bb0(%a: f32, %b: f32, %unused: f32):
  %r = arith.mulf %a,%b : f32
  linalg.yield %r : f32
 } -> tensor<1x?x3072xf32>
 return %silu,%gated : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>
}
