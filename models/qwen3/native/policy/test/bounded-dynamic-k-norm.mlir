// RUN: qwen-policy-sums-opt %s --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' | FileCheck %s

// Only the opted-in 8-head, 128-wide square sum receives the checked order.
// CHECK-LABEL: func.func @k_norm
// CHECK: cf.assert
// CHECK: scf.for
// ARM uses 16 interleaved accumulators followed by a four-lane fold.
// CHECK: tensor<8x1x8x16xf32>
// CHECK: hexagon.fp32_sum_cascade_partial
// CHECK: hexagon.fp32_arm_k_norm_sum
// CHECK: return
func.func @k_norm(%x : tensor<1x?x8x128xf32>) -> tensor<1x?x8x1xf32>
    attributes {hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x8x128xf32>
  %square_empty = tensor.empty(%q) : tensor<1x?x8x128xf32>
  %two = arith.constant 2 : i64
  %square = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x?x8x128xf32>)
      outs(%square_empty : tensor<1x?x8x128xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %s = math.fpowi %value, %two : f32, i64
      linalg.yield %s : f32
  } -> tensor<1x?x8x128xf32>
  %out_empty = tensor.empty(%q) : tensor<1x?x8x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<1x?x8x1xf32>) -> tensor<1x?x8x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%square : tensor<1x?x8x128xf32>)
      outs(%init : tensor<1x?x8x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1x?x8x1xf32>
  return %sum : tensor<1x?x8x1xf32>
}

// CHECK-LABEL: func.func @unannotated_k_norm
// CHECK-NOT: scf.for
// CHECK: return
func.func @unannotated_k_norm(%x : tensor<1x?x8x128xf32>) -> tensor<1x?x8x1xf32> {
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x8x128xf32>
  %square_empty = tensor.empty(%q) : tensor<1x?x8x128xf32>
  %two = arith.constant 2 : i64
  %square = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x?x8x128xf32>)
      outs(%square_empty : tensor<1x?x8x128xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %s = math.fpowi %value, %two : f32, i64
      linalg.yield %s : f32
  } -> tensor<1x?x8x128xf32>
  %out_empty = tensor.empty(%q) : tensor<1x?x8x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<1x?x8x1xf32>) -> tensor<1x?x8x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%square : tensor<1x?x8x128xf32>)
      outs(%init : tensor<1x?x8x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1x?x8x1xf32>
  return %sum : tensor<1x?x8x1xf32>
}

// A non-Q/K head count must not take the admitted normalization routes.
// CHECK-LABEL: func.func @wrong_heads
// CHECK-NOT: scf.for
// CHECK: return
func.func @wrong_heads(%x : tensor<1x?x12x128xf32>) -> tensor<1x?x12x1xf32>
    attributes {hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x12x128xf32>
  %square_empty = tensor.empty(%q) : tensor<1x?x12x128xf32>
  %two = arith.constant 2 : i64
  %square = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x?x12x128xf32>)
      outs(%square_empty : tensor<1x?x12x128xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %s = math.fpowi %value, %two : f32, i64
      linalg.yield %s : f32
  } -> tensor<1x?x12x128xf32>
  %out_empty = tensor.empty(%q) : tensor<1x?x12x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<1x?x12x1xf32>) -> tensor<1x?x12x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%square : tensor<1x?x12x128xf32>)
      outs(%init : tensor<1x?x12x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1x?x12x1xf32>
  return %sum : tensor<1x?x12x1xf32>
}
