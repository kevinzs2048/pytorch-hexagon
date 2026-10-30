// RUN: qwen-policy-sums-opt %s --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' | FileCheck %s

// The symbolic query dimension drives an outer loop, not a q17 tensor.cast.
// The fixed 1024-wide inner reduction keeps the previously checked order.
// CHECK-LABEL: func.func @bounded
// CHECK: cf.assert
// CHECK: scf.for
// CHECK: hexagon.fp32_sum_cascade_partial
// CHECK: hexagon.fp32_arm_rmsnorm_sum
func.func @bounded(%x : tensor<1x?x1024xf32>) -> tensor<1x?x1xf32>
    attributes {hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x1024xf32>
  %square_empty = tensor.empty(%q) : tensor<1x?x1024xf32>
  %two = arith.constant 2 : i64
  %square = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1,d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x?x1024xf32>)
      outs(%square_empty : tensor<1x?x1024xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %s = math.fpowi %value, %two : f32, i64
      linalg.yield %s : f32
  } -> tensor<1x?x1024xf32>
  %out_empty = tensor.empty(%q) : tensor<1x?x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1,0)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%square : tensor<1x?x1024xf32>)
      outs(%init : tensor<1x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1x?x1xf32>
  return %sum : tensor<1x?x1xf32>
}

// The same graph without explicit opt-in must remain untouched.
// CHECK-LABEL: func.func @unannotated
// CHECK-NOT: scf.for
// CHECK: linalg.generic
func.func @unannotated(%x : tensor<1x?x1024xf32>) -> tensor<1x?x1xf32> {
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x1024xf32>
  %square_empty = tensor.empty(%q) : tensor<1x?x1024xf32>
  %two = arith.constant 2 : i64
  %square = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1,d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x?x1024xf32>)
      outs(%square_empty : tensor<1x?x1024xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %s = math.fpowi %value, %two : f32, i64
      linalg.yield %s : f32
  } -> tensor<1x?x1024xf32>
  %out_empty = tensor.empty(%q) : tensor<1x?x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1,0)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%square : tensor<1x?x1024xf32>)
      outs(%init : tensor<1x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1x?x1xf32>
  return %sum : tensor<1x?x1xf32>
}

// Even with opt-in, a different reduction width is not a Qwen RMSNorm sum.
// CHECK-LABEL: func.func @wrong_width
// CHECK-NOT: scf.for
// CHECK: return
func.func @wrong_width(%x : tensor<1x?x128xf32>) -> tensor<1x?x1xf32>
    attributes {hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
  %c1 = arith.constant 1 : index
  %q = tensor.dim %x, %c1 : tensor<1x?x128xf32>
  %square_empty = tensor.empty(%q) : tensor<1x?x128xf32>
  %two = arith.constant 2 : i64
  %square = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1,d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x?x128xf32>)
      outs(%square_empty : tensor<1x?x128xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %s = math.fpowi %value, %two : f32, i64
      linalg.yield %s : f32
  } -> tensor<1x?x128xf32>
  %out_empty = tensor.empty(%q) : tensor<1x?x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%out_empty : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1,0)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%square : tensor<1x?x128xf32>)
      outs(%init : tensor<1x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %s = arith.addf %value, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1x?x1xf32>
  return %sum : tensor<1x?x1xf32>
}
