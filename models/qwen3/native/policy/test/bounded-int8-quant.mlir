// RUN: qwen-packed-i8-opt %s --qwen-vectorize-activation=expected-count=1 --canonicalize --cse --verify-each | FileCheck %s
// CHECK-LABEL: func.func @Qwen3DynamicPrefill
// CHECK: cf.assert
// CHECK: arith.divf
// CHECK: arith.minui
// CHECK: arith.trunci
// CHECK-NOT: arith.fptosi

#id = affine_map<(b,r,c)->(b,r,c)>
#scale = affine_map<(b,r,c)->(b,r,0)>
#scalar = affine_map<(b,r,c)->()>
module {
func.func @Qwen3DynamicPrefill(%source: memref<1x32x1024xf32>, %scales: memref<1x32x1xf32>, %output: memref<1x32x1024xi8>) attributes {llvm.emit_c_interface} {
 %input_tensor = bufferization.to_tensor %source restrict : memref<1x32x1024xf32> to tensor<1x32x1024xf32>
 %scale_tensor = bufferization.to_tensor %scales restrict : memref<1x32x1xf32> to tensor<1x32x1xf32>
 %zero = arith.constant 0.0 : f32
 %one = arith.constant 1.0 : f32
 %lo = arith.constant -127.0 : f32
 %hi = arith.constant 127.0 : f32
 %zero_tensor = arith.constant dense<0.0> : tensor<f32>
 %si = tensor.empty() : tensor<1x32x1xf32>
 %init = tensor.empty() : tensor<1x32x1024xf32>
 %oi = tensor.empty() : tensor<1x32x1024xi8>
 %inverse = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel","parallel"]} ins(%scale_tensor:tensor<1x32x1xf32>) outs(%si:tensor<1x32x1xf32>) {
 ^bb0(%x:f32,%unused:f32):
  %ok = arith.cmpf one, %x,%zero:f32
  cf.assert %ok,"scale nonzero"
  %v = arith.divf %one,%x:f32
  linalg.yield %v:f32
 } -> tensor<1x32x1xf32>
 %scaled = linalg.generic {indexing_maps=[#id,#scale,#id], iterator_types=["parallel","parallel","parallel"]} ins(%input_tensor,%inverse:tensor<1x32x1024xf32>,tensor<1x32x1xf32>) outs(%init:tensor<1x32x1024xf32>) {
 ^bb0(%x:f32,%s:f32,%unused:f32):
  %v = arith.mulf %x,%s:f32
  linalg.yield %v:f32
 } -> tensor<1x32x1024xf32>
 %rounded = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel","parallel"]} ins(%scaled:tensor<1x32x1024xf32>) outs(%init:tensor<1x32x1024xf32>) {
 ^bb0(%x:f32,%unused:f32):
  %v = math.roundeven %x:f32
  linalg.yield %v:f32
 } -> tensor<1x32x1024xf32>
 %offset = linalg.generic {indexing_maps=[#id,#scalar,#id], iterator_types=["parallel","parallel","parallel"]} ins(%rounded,%zero_tensor:tensor<1x32x1024xf32>,tensor<f32>) outs(%init:tensor<1x32x1024xf32>) {
 ^bb0(%x:f32,%z:f32,%unused:f32):
  %v = arith.addf %x,%z:f32
  linalg.yield %v:f32
 } -> tensor<1x32x1024xf32>
 %clamped = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel","parallel"]} ins(%offset:tensor<1x32x1024xf32>) outs(%init:tensor<1x32x1024xf32>) {
 ^bb0(%x:f32,%unused:f32):
  %c0 = arith.cmpf ult,%x,%lo:f32
  %v0 = arith.select %c0,%lo,%x:f32
  %c1 = arith.cmpf ugt,%v0,%hi:f32
  %v1 = arith.select %c1,%hi,%v0:f32
  linalg.yield %v1:f32
 } -> tensor<1x32x1024xf32>
 %casted = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel","parallel"]} ins(%clamped:tensor<1x32x1024xf32>) outs(%oi:tensor<1x32x1024xi8>) {
 ^bb0(%x:f32,%unused:i8):
  %v = arith.fptosi %x:f32 to i8
  linalg.yield %v:i8
 } -> tensor<1x32x1024xi8>
 bufferization.materialize_in_destination %casted in writable %output : (tensor<1x32x1024xi8>,memref<1x32x1024xi8>) -> ()
 return
}
}
