// RUN: qwen-policy-sums-opt %s --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' | FileCheck %s

// The reduction order is fixed within one 1024-slot row, while q stays runtime.
// CHECK-LABEL: func.func @bounded_softmax
// CHECK: cf.assert
// CHECK: cf.assert
// CHECK: scf.for
// CHECK: tensor<16x256x4xf32>
// CHECK: hexagon.fp32_arm_softmax_partial
// CHECK: hexagon.fp32_arm_softmax_sum
// CHECK: hexagon.fp32_arm_softmax_reciprocal
// CHECK: arith.divf
// CHECK: hexagon.fp32_arm_softmax_normalization
// CHECK: arith.mulf
func.func @bounded_softmax(%scores: tensor<1x16x?x?xf32>) -> tensor<1x16x?x?xf32>
    attributes {hexagon.fp32_dynamic_softmax_max_q = 32 : i64} {
  %c2 = arith.constant 2 : index
  %q = tensor.dim %scores, %c2 : tensor<1x16x?x?xf32>
  %c3 = arith.constant 3 : index
  %k = tensor.dim %scores, %c3 : tensor<1x16x?x?xf32>
  %empty = tensor.empty(%q, %k) : tensor<1x16x?x?xf32>
  %row_empty = tensor.empty(%q) : tensor<1x16x?x1xf32>
  %negative_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %max_init = linalg.fill ins(%negative_inf : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %maximum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%scores : tensor<1x16x?x?xf32>)
      outs(%max_init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.maximumf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  %shift = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%scores, %maximum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %max : f32, %unused : f32):
      %v = arith.subf %value, %max : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  %exp = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%shift : tensor<1x16x?x?xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %v = math.exp %value : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  %sum_init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%exp : tensor<1x16x?x?xf32>)
      outs(%sum_init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.addf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  %normalized = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%exp, %sum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %denominator : f32, %unused : f32):
      %v = arith.divf %value, %denominator : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  return %normalized : tensor<1x16x?x?xf32>
}

// Even with opt-in, a same-shaped ordinary reduction is not Softmax.
// CHECK-LABEL: func.func @wrong_chain
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK: return
func.func @wrong_chain(%scores: tensor<1x16x?x?xf32>) -> tensor<1x16x?x1xf32>
    attributes {hexagon.fp32_dynamic_softmax_max_q = 32 : i64} {
  %c2 = arith.constant 2 : index
  %q = tensor.dim %scores, %c2 : tensor<1x16x?x?xf32>
  %row_empty = tensor.empty(%q) : tensor<1x16x?x1xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%scores : tensor<1x16x?x?xf32>)
      outs(%init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.addf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  return %sum : tensor<1x16x?x1xf32>
}

// CHECK-LABEL: func.func @singleton
// CHECK: hexagon.fp32_arm_softmax_normalization
// CHECK: arith.mulf

func.func @singleton(%scores: tensor<1x16x1x?xf32>) -> tensor<1x16x1x?xf32>
    attributes {hexagon.fp32_dynamic_softmax_max_q = 32 : i64} {
  %c2 = arith.constant 2 : index
  %q = tensor.dim %scores, %c2 : tensor<1x16x1x?xf32>
  %c3 = arith.constant 3 : index
  %k = tensor.dim %scores, %c3 : tensor<1x16x1x?xf32>
  %empty = tensor.empty(%k) : tensor<1x16x1x?xf32>
  %row_empty = tensor.empty() : tensor<1x16x1x1xf32>
  %negative_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %max_init = linalg.fill ins(%negative_inf : f32)
      outs(%row_empty : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %maximum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%scores : tensor<1x16x1x?xf32>)
      outs(%max_init : tensor<1x16x1x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.maximumf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x1x1xf32>
  %shift = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%scores, %maximum : tensor<1x16x1x?xf32>, tensor<1x16x1x1xf32>)
      outs(%empty : tensor<1x16x1x?xf32>) {
    ^bb0(%value : f32, %max : f32, %unused : f32):
      %v = arith.subf %value, %max : f32
      linalg.yield %v : f32
  } -> tensor<1x16x1x?xf32>
  %exp = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%shift : tensor<1x16x1x?xf32>)
      outs(%empty : tensor<1x16x1x?xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %v = math.exp %value : f32
      linalg.yield %v : f32
  } -> tensor<1x16x1x?xf32>
  %sum_init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%exp : tensor<1x16x1x?xf32>)
      outs(%sum_init : tensor<1x16x1x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.addf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x1x1xf32>
  %normalized = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%exp, %sum : tensor<1x16x1x?xf32>, tensor<1x16x1x1xf32>)
      outs(%empty : tensor<1x16x1x?xf32>) {
    ^bb0(%value : f32, %denominator : f32, %unused : f32):
      %v = arith.divf %value, %denominator : f32
      linalg.yield %v : f32
  } -> tensor<1x16x1x?xf32>
  return %normalized : tensor<1x16x1x?xf32>
}


// CHECK-LABEL: func.func @multi_use_sum
// CHECK-NOT: hexagon.fp32_arm_softmax
// CHECK: return

func.func @multi_use_sum(%scores: tensor<1x16x?x?xf32>) -> (tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
    attributes {hexagon.fp32_dynamic_softmax_max_q = 32 : i64} {
  %c2 = arith.constant 2 : index
  %q = tensor.dim %scores, %c2 : tensor<1x16x?x?xf32>
  %c3 = arith.constant 3 : index
  %k = tensor.dim %scores, %c3 : tensor<1x16x?x?xf32>
  %empty = tensor.empty(%q, %k) : tensor<1x16x?x?xf32>
  %row_empty = tensor.empty(%q) : tensor<1x16x?x1xf32>
  %negative_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %max_init = linalg.fill ins(%negative_inf : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %maximum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%scores : tensor<1x16x?x?xf32>)
      outs(%max_init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.maximumf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  %shift = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%scores, %maximum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %max : f32, %unused : f32):
      %v = arith.subf %value, %max : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  %exp = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%shift : tensor<1x16x?x?xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %v = math.exp %value : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  %sum_init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%exp : tensor<1x16x?x?xf32>)
      outs(%sum_init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.addf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  %normalized = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%exp, %sum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %denominator : f32, %unused : f32):
      %v = arith.divf %value, %denominator : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  return %normalized, %sum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>
}


// CHECK-LABEL: func.func @not_opted_in
// CHECK-NOT: hexagon.fp32_arm_softmax
// CHECK: return

func.func @not_opted_in(%scores: tensor<1x16x?x?xf32>) -> tensor<1x16x?x?xf32>
    attributes {test.unrelated = 32 : i64} {
  %c2 = arith.constant 2 : index
  %q = tensor.dim %scores, %c2 : tensor<1x16x?x?xf32>
  %c3 = arith.constant 3 : index
  %k = tensor.dim %scores, %c3 : tensor<1x16x?x?xf32>
  %empty = tensor.empty(%q, %k) : tensor<1x16x?x?xf32>
  %row_empty = tensor.empty(%q) : tensor<1x16x?x1xf32>
  %negative_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %max_init = linalg.fill ins(%negative_inf : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %maximum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%scores : tensor<1x16x?x?xf32>)
      outs(%max_init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.maximumf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  %shift = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%scores, %maximum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %max : f32, %unused : f32):
      %v = arith.subf %value, %max : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  %exp = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%shift : tensor<1x16x?x?xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %unused : f32):
      %v = math.exp %value : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  %sum_init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<1x16x?x1xf32>) -> tensor<1x16x?x1xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%exp : tensor<1x16x?x?xf32>)
      outs(%sum_init : tensor<1x16x?x1xf32>) {
    ^bb0(%value : f32, %acc : f32):
      %v = arith.addf %value, %acc : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x1xf32>
  %normalized = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,0)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%exp, %sum : tensor<1x16x?x?xf32>, tensor<1x16x?x1xf32>)
      outs(%empty : tensor<1x16x?x?xf32>) {
    ^bb0(%value : f32, %denominator : f32, %unused : f32):
      %v = arith.divf %value, %denominator : f32
      linalg.yield %v : f32
  } -> tensor<1x16x?x?xf32>
  return %normalized : tensor<1x16x?x?xf32>
}
