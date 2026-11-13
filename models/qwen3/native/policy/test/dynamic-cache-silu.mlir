// RUN: qwen-policy-first-opt %s --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum{fp32-silu-exp-compat=true}))' | qwen-policy-sums-opt --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' | FileCheck %s

// The native SiLU policy must not require a fixed attention capacity.
#id = affine_map<(a,b,c)->(a,b,c)>
func.func private @__hexagon_silu_exp_compat(f32) -> f32

// CHECK-LABEL: func.func @dynamic
// CHECK: call @__hexagon_silu_exp_compat
// CHECK-NOT: math.exp
// CHECK: hexagon.fp32_arm_silu_division
// CHECK: return
func.func @dynamic(%x: tensor<1x?x3072xf32>) -> tensor<1x?x3072xf32>
    attributes {hexagon.fp32_native_silu = true, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
  %one = arith.constant 1.0 : f32
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x3072xf32>
  %empty = tensor.empty(%q) : tensor<1x?x3072xf32>
  %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %neg = arith.negf %v : f32
    %exp = math.exp %neg : f32
    %den = arith.addf %exp, %one : f32
    %inv = arith.divf %one, %den : f32
    linalg.yield %inv : f32
  } -> tensor<1x?x3072xf32>
  %silu = linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%sigmoid,%x : tensor<1x?x3072xf32>,tensor<1x?x3072xf32>) outs(%empty : tensor<1x?x3072xf32>) {
  ^bb0(%a: f32, %b: f32, %unused: f32):
    %v = arith.mulf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<1x?x3072xf32>
  return %silu : tensor<1x?x3072xf32>
}

// An ordinary sigmoid must retain its original exponent and division.
// CHECK-LABEL: func.func @sigmoid
// CHECK-NOT: call @__hexagon_silu_exp_compat
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: return
func.func @sigmoid(%x: tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
    attributes {hexagon.fp32_native_silu = true} {
  %one = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<1x1x3072xf32>
  %sigmoid = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x1x3072xf32>) outs(%empty : tensor<1x1x3072xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %neg = arith.negf %v : f32
    %exp = math.exp %neg : f32
    %den = arith.addf %exp, %one : f32
    %inv = arith.divf %one, %den : f32
    linalg.yield %inv : f32
  } -> tensor<1x1x3072xf32>
  return %sigmoid : tensor<1x1x3072xf32>
}
