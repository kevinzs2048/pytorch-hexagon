// RUN: qwen-policy-sums-opt %s --verify-each --pass-pipeline="builtin.module(func.func(stabilize-fp32-sum))" | FileCheck %s
// CHECK-LABEL: func.func @trig(
// CHECK: attrs = {{.*}}hexagon.fp32_arm_rope_trig4
// CHECK: func.call @__hexagon_arm_sin4_lane
// CHECK: attrs = {{.*}}hexagon.fp32_arm_rope_trig4
// CHECK: func.call @__hexagon_arm_cos4_lane
module {
  func.func private @__hexagon_arm_sin4_lane(f32,f32,f32,f32,i32) -> f32
  func.func private @__hexagon_arm_cos4_lane(f32,f32,f32,f32,i32) -> f32
 func.func @trig(%x:tensor<1x1x128xf32>) -> (tensor<1x1x128xf32>,tensor<1x1x128xf32>) attributes {llvm.emit_c_interface,hexagon.fp32_arm_rope_trig4=true} {
  %empty = tensor.empty() : tensor<1x1x128xf32>
  %sin = linalg.generic {indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel","parallel","parallel"]} ins(%x : tensor<1x1x128xf32>) outs(%empty : tensor<1x1x128xf32>) {
  ^bb0(%in:f32,%out:f32):
    %v = math.sin %in : f32
    linalg.yield %v : f32
  } -> tensor<1x1x128xf32>
  %cos = linalg.generic {indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel","parallel","parallel"]} ins(%x : tensor<1x1x128xf32>) outs(%empty : tensor<1x1x128xf32>) {
  ^bb0(%in:f32,%out:f32):
    %v = math.cos %in : f32
    linalg.yield %v : f32
  } -> tensor<1x1x128xf32>
  return %sin,%cos : tensor<1x1x128xf32>,tensor<1x1x128xf32>
 }
}
