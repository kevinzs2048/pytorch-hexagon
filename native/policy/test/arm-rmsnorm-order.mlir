// RUN: qwen-policy-sums-opt %s --verify-each --pass-pipeline='builtin.module(func.func(stabilize-fp32-sum))' | FileCheck %s

// CHECK-LABEL: func.func @static_hidden
// CHECK: hexagon.fp32_arm_static_rmsnorm_sum
// CHECK: return

#static_hidden_id = affine_map<(a,b,c)->(a,b,c)>
#static_hidden_row = affine_map<(a,b,c)->(a,b,0)>
#static_hidden_w = affine_map<(a,b,c)->(c)>
func.func @static_hidden(%x: tensor<1x1x1024xf32>, %w: tensor<1024xf32>) -> tensor<1x1x1024xf32> attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %two = arith.constant 2 : i64
 %zero = arith.constant 0.0 : f32
 %width = arith.constant 1024.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty() : tensor<1x1x1024xf32>
 %rowempty = tensor.empty() : tensor<1x1x1xf32>
 %sq = linalg.generic {indexing_maps=[#static_hidden_id,#static_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x1x1024xf32>) outs(%empty : tensor<1x1x1024xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x1x1024xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
 %sum = linalg.generic {indexing_maps=[#static_hidden_id,#static_hidden_row], iterator_types=["parallel","parallel","reduction"]} ins(%sq : tensor<1x1x1024xf32>) outs(%init : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %mean = linalg.generic {indexing_maps=[#static_hidden_id,#static_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%sum : tensor<1x1x1xf32>) outs(%rowempty : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %variance = linalg.generic {indexing_maps=[#static_hidden_id,#static_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%mean : tensor<1x1x1xf32>) outs(%rowempty : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %factor = linalg.generic {indexing_maps=[#static_hidden_id,#static_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%variance : tensor<1x1x1xf32>) outs(%rowempty : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %unit = linalg.generic {indexing_maps=[#static_hidden_id,#static_hidden_row,#static_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x1x1024xf32>,tensor<1x1x1xf32>) outs(%empty : tensor<1x1x1024xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1024xf32>
 %out = linalg.generic {indexing_maps=[#static_hidden_w,#static_hidden_id,#static_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%w,%unit : tensor<1024xf32>,tensor<1x1x1024xf32>) outs(%empty : tensor<1x1x1024xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1024xf32>
 return %out : tensor<1x1x1024xf32>
}

// CHECK-LABEL: func.func @dynamic_Q
// CHECK: hexagon.fp32_arm_rmsnorm_sum
// CHECK: return

#dynamic_Q_id = affine_map<(a,b,c,d)->(a,b,c,d)>
#dynamic_Q_row = affine_map<(a,b,c,d)->(a,b,c,0)>
#dynamic_Q_w = affine_map<(a,b,c,d)->(d)>
func.func @dynamic_Q(%x: tensor<1x?x16x128xf32>, %w: tensor<128xf32>) -> tensor<1x?x16x128xf32> attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x?x16x128xf32>
 %two = arith.constant 2 : i64
 %zero = arith.constant 0.0 : f32
 %width = arith.constant 128.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty(%q) : tensor<1x?x16x128xf32>
 %rowempty = tensor.empty(%q) : tensor<1x?x16x1xf32>
 %sq = linalg.generic {indexing_maps=[#dynamic_Q_id,#dynamic_Q_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%x : tensor<1x?x16x128xf32>) outs(%empty : tensor<1x?x16x128xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x?x16x128xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x?x16x1xf32>) -> tensor<1x?x16x1xf32>
 %sum = linalg.generic {indexing_maps=[#dynamic_Q_id,#dynamic_Q_row], iterator_types=["parallel","parallel","parallel","reduction"]} ins(%sq : tensor<1x?x16x128xf32>) outs(%init : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %mean = linalg.generic {indexing_maps=[#dynamic_Q_id,#dynamic_Q_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%sum : tensor<1x?x16x1xf32>) outs(%rowempty : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %variance = linalg.generic {indexing_maps=[#dynamic_Q_id,#dynamic_Q_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%mean : tensor<1x?x16x1xf32>) outs(%rowempty : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %factor = linalg.generic {indexing_maps=[#dynamic_Q_id,#dynamic_Q_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%variance : tensor<1x?x16x1xf32>) outs(%rowempty : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %unit = linalg.generic {indexing_maps=[#dynamic_Q_id,#dynamic_Q_row,#dynamic_Q_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x?x16x128xf32>,tensor<1x?x16x1xf32>) outs(%empty : tensor<1x?x16x128xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x128xf32>
 %out = linalg.generic {indexing_maps=[#dynamic_Q_w,#dynamic_Q_id,#dynamic_Q_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%w,%unit : tensor<128xf32>,tensor<1x?x16x128xf32>) outs(%empty : tensor<1x?x16x128xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x128xf32>
 return %out : tensor<1x?x16x128xf32>
}

// CHECK-LABEL: func.func @nonzero_static
// CHECK-NOT: hexagon.fp32_arm_
// CHECK: return

#nonzero_static_id = affine_map<(a,b,c)->(a,b,c)>
#nonzero_static_row = affine_map<(a,b,c)->(a,b,0)>
#nonzero_static_w = affine_map<(a,b,c)->(c)>
func.func @nonzero_static(%x: tensor<1x1x1024xf32>, %w: tensor<1024xf32>) -> tensor<1x1x1024xf32> attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %two = arith.constant 2 : i64
 %zero = arith.constant 1.0 : f32
 %width = arith.constant 1024.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty() : tensor<1x1x1024xf32>
 %rowempty = tensor.empty() : tensor<1x1x1xf32>
 %sq = linalg.generic {indexing_maps=[#nonzero_static_id,#nonzero_static_id], iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x1x1024xf32>) outs(%empty : tensor<1x1x1024xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x1x1024xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
 %sum = linalg.generic {indexing_maps=[#nonzero_static_id,#nonzero_static_row], iterator_types=["parallel","parallel","reduction"]} ins(%sq : tensor<1x1x1024xf32>) outs(%init : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %mean = linalg.generic {indexing_maps=[#nonzero_static_id,#nonzero_static_id], iterator_types=["parallel","parallel","parallel"]} ins(%sum : tensor<1x1x1xf32>) outs(%rowempty : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %variance = linalg.generic {indexing_maps=[#nonzero_static_id,#nonzero_static_id], iterator_types=["parallel","parallel","parallel"]} ins(%mean : tensor<1x1x1xf32>) outs(%rowempty : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %factor = linalg.generic {indexing_maps=[#nonzero_static_id,#nonzero_static_id], iterator_types=["parallel","parallel","parallel"]} ins(%variance : tensor<1x1x1xf32>) outs(%rowempty : tensor<1x1x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1xf32>
 %unit = linalg.generic {indexing_maps=[#nonzero_static_id,#nonzero_static_row,#nonzero_static_id], iterator_types=["parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x1x1024xf32>,tensor<1x1x1xf32>) outs(%empty : tensor<1x1x1024xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1024xf32>
 %out = linalg.generic {indexing_maps=[#nonzero_static_w,#nonzero_static_id,#nonzero_static_id], iterator_types=["parallel","parallel","parallel"]} ins(%w,%unit : tensor<1024xf32>,tensor<1x1x1024xf32>) outs(%empty : tensor<1x1x1024xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x1x1024xf32>
 return %out : tensor<1x1x1024xf32>
}

// CHECK-LABEL: func.func @nonzero_dynamic
// CHECK-NOT: hexagon.fp32_arm_
// CHECK: return

#nonzero_dynamic_id = affine_map<(a,b,c,d)->(a,b,c,d)>
#nonzero_dynamic_row = affine_map<(a,b,c,d)->(a,b,c,0)>
#nonzero_dynamic_w = affine_map<(a,b,c,d)->(d)>
func.func @nonzero_dynamic(%x: tensor<1x?x16x128xf32>, %w: tensor<128xf32>) -> tensor<1x?x16x128xf32> attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %c1 = arith.constant 1 : index
 %q = tensor.dim %x, %c1 : tensor<1x?x16x128xf32>
 %two = arith.constant 2 : i64
 %zero = arith.constant 1.0 : f32
 %width = arith.constant 128.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty(%q) : tensor<1x?x16x128xf32>
 %rowempty = tensor.empty(%q) : tensor<1x?x16x1xf32>
 %sq = linalg.generic {indexing_maps=[#nonzero_dynamic_id,#nonzero_dynamic_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%x : tensor<1x?x16x128xf32>) outs(%empty : tensor<1x?x16x128xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x?x16x128xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x?x16x1xf32>) -> tensor<1x?x16x1xf32>
 %sum = linalg.generic {indexing_maps=[#nonzero_dynamic_id,#nonzero_dynamic_row], iterator_types=["parallel","parallel","parallel","reduction"]} ins(%sq : tensor<1x?x16x128xf32>) outs(%init : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %mean = linalg.generic {indexing_maps=[#nonzero_dynamic_id,#nonzero_dynamic_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%sum : tensor<1x?x16x1xf32>) outs(%rowempty : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %variance = linalg.generic {indexing_maps=[#nonzero_dynamic_id,#nonzero_dynamic_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%mean : tensor<1x?x16x1xf32>) outs(%rowempty : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %factor = linalg.generic {indexing_maps=[#nonzero_dynamic_id,#nonzero_dynamic_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%variance : tensor<1x?x16x1xf32>) outs(%rowempty : tensor<1x?x16x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x1xf32>
 %unit = linalg.generic {indexing_maps=[#nonzero_dynamic_id,#nonzero_dynamic_row,#nonzero_dynamic_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x?x16x128xf32>,tensor<1x?x16x1xf32>) outs(%empty : tensor<1x?x16x128xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x128xf32>
 %out = linalg.generic {indexing_maps=[#nonzero_dynamic_w,#nonzero_dynamic_id,#nonzero_dynamic_id], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%w,%unit : tensor<128xf32>,tensor<1x?x16x128xf32>) outs(%empty : tensor<1x?x16x128xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x?x16x128xf32>
 return %out : tensor<1x?x16x128xf32>
}

// CHECK-LABEL: func.func @static_multiple_hidden
// CHECK: tensor<8x4x16x16xf32>
// CHECK: hexagon.fp32_arm_static_rmsnorm_sum
// CHECK: return

#static_multiple_hidden_id = affine_map<(a,b,c)->(a,b,c)>
#static_multiple_hidden_row = affine_map<(a,b,c)->(a,b,0)>
#static_multiple_hidden_w = affine_map<(a,b,c)->(c)>
func.func @static_multiple_hidden(%x: tensor<1x8x1024xf32>, %w: tensor<1024xf32>) -> tensor<1x8x1024xf32> attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %two = arith.constant 2 : i64
 %zero = arith.constant 0.0 : f32
 %width = arith.constant 1024.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty() : tensor<1x8x1024xf32>
 %rowempty = tensor.empty() : tensor<1x8x1xf32>
 %sq = linalg.generic {indexing_maps=[#static_multiple_hidden_id,#static_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x8x1024xf32>) outs(%empty : tensor<1x8x1024xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x8x1024xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
 %sum = linalg.generic {indexing_maps=[#static_multiple_hidden_id,#static_multiple_hidden_row], iterator_types=["parallel","parallel","reduction"]} ins(%sq : tensor<1x8x1024xf32>) outs(%init : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %mean = linalg.generic {indexing_maps=[#static_multiple_hidden_id,#static_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%sum : tensor<1x8x1xf32>) outs(%rowempty : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %variance = linalg.generic {indexing_maps=[#static_multiple_hidden_id,#static_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%mean : tensor<1x8x1xf32>) outs(%rowempty : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %factor = linalg.generic {indexing_maps=[#static_multiple_hidden_id,#static_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%variance : tensor<1x8x1xf32>) outs(%rowempty : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %unit = linalg.generic {indexing_maps=[#static_multiple_hidden_id,#static_multiple_hidden_row,#static_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x8x1024xf32>,tensor<1x8x1xf32>) outs(%empty : tensor<1x8x1024xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1024xf32>
 %out = linalg.generic {indexing_maps=[#static_multiple_hidden_w,#static_multiple_hidden_id,#static_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%w,%unit : tensor<1024xf32>,tensor<1x8x1024xf32>) outs(%empty : tensor<1x8x1024xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1024xf32>
 return %out : tensor<1x8x1024xf32>
}


// CHECK-LABEL: func.func @unannotated_multiple_hidden
// CHECK-NOT: hexagon.fp32_arm_
// CHECK: return

#unannotated_multiple_hidden_id = affine_map<(a,b,c)->(a,b,c)>
#unannotated_multiple_hidden_row = affine_map<(a,b,c)->(a,b,0)>
#unannotated_multiple_hidden_w = affine_map<(a,b,c)->(c)>
func.func @unannotated_multiple_hidden(%x: tensor<1x8x1024xf32>, %w: tensor<1024xf32>) -> tensor<1x8x1024xf32> attributes {llvm.emit_c_interface} {
 %two = arith.constant 2 : i64
 %zero = arith.constant 0.0 : f32
 %width = arith.constant 1024.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty() : tensor<1x8x1024xf32>
 %rowempty = tensor.empty() : tensor<1x8x1xf32>
 %sq = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x8x1024xf32>) outs(%empty : tensor<1x8x1024xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x8x1024xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
 %sum = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_row], iterator_types=["parallel","parallel","reduction"]} ins(%sq : tensor<1x8x1024xf32>) outs(%init : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %mean = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%sum : tensor<1x8x1xf32>) outs(%rowempty : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %variance = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%mean : tensor<1x8x1xf32>) outs(%rowempty : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %factor = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%variance : tensor<1x8x1xf32>) outs(%rowempty : tensor<1x8x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1xf32>
 %unit = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_row,#unannotated_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x8x1024xf32>,tensor<1x8x1xf32>) outs(%empty : tensor<1x8x1024xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1024xf32>
 %out = linalg.generic {indexing_maps=[#unannotated_multiple_hidden_w,#unannotated_multiple_hidden_id,#unannotated_multiple_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%w,%unit : tensor<1024xf32>,tensor<1x8x1024xf32>) outs(%empty : tensor<1x8x1024xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x8x1024xf32>
 return %out : tensor<1x8x1024xf32>
}


// CHECK-LABEL: func.func @out_of_budget_hidden
// CHECK-NOT: hexagon.fp32_arm_
// CHECK: return

#out_of_budget_hidden_id = affine_map<(a,b,c)->(a,b,c)>
#out_of_budget_hidden_row = affine_map<(a,b,c)->(a,b,0)>
#out_of_budget_hidden_w = affine_map<(a,b,c)->(c)>
func.func @out_of_budget_hidden(%x: tensor<1x33x1024xf32>, %w: tensor<1024xf32>) -> tensor<1x33x1024xf32> attributes {llvm.emit_c_interface, hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64} {
 %two = arith.constant 2 : i64
 %zero = arith.constant 0.0 : f32
 %width = arith.constant 1024.0 : f32
 %eps = arith.constant 1.0e-6 : f32
 %empty = tensor.empty() : tensor<1x33x1024xf32>
 %rowempty = tensor.empty() : tensor<1x33x1xf32>
 %sq = linalg.generic {indexing_maps=[#out_of_budget_hidden_id,#out_of_budget_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x : tensor<1x33x1024xf32>) outs(%empty : tensor<1x33x1024xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.fpowi %v, %two : f32, i64
  linalg.yield %r : f32
 } -> tensor<1x33x1024xf32>
 %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<1x33x1xf32>) -> tensor<1x33x1xf32>
 %sum = linalg.generic {indexing_maps=[#out_of_budget_hidden_id,#out_of_budget_hidden_row], iterator_types=["parallel","parallel","reduction"]} ins(%sq : tensor<1x33x1024xf32>) outs(%init : tensor<1x33x1xf32>) {
 ^bb0(%v: f32, %a: f32):
  %r = arith.addf %v, %a : f32
  linalg.yield %r : f32
 } -> tensor<1x33x1xf32>
 %mean = linalg.generic {indexing_maps=[#out_of_budget_hidden_id,#out_of_budget_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%sum : tensor<1x33x1xf32>) outs(%rowempty : tensor<1x33x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.divf %v, %width : f32
  linalg.yield %r : f32
 } -> tensor<1x33x1xf32>
 %variance = linalg.generic {indexing_maps=[#out_of_budget_hidden_id,#out_of_budget_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%mean : tensor<1x33x1xf32>) outs(%rowempty : tensor<1x33x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = arith.addf %v, %eps : f32
  linalg.yield %r : f32
 } -> tensor<1x33x1xf32>
 %factor = linalg.generic {indexing_maps=[#out_of_budget_hidden_id,#out_of_budget_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%variance : tensor<1x33x1xf32>) outs(%rowempty : tensor<1x33x1xf32>) {
 ^bb0(%v: f32, %o: f32):
  %r = math.rsqrt %v : f32
  linalg.yield %r : f32
 } -> tensor<1x33x1xf32>
 %unit = linalg.generic {indexing_maps=[#out_of_budget_hidden_id,#out_of_budget_hidden_row,#out_of_budget_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%x,%factor : tensor<1x33x1024xf32>,tensor<1x33x1xf32>) outs(%empty : tensor<1x33x1024xf32>) {
 ^bb0(%v: f32, %a: f32, %o: f32):
  %r = arith.mulf %v,%a : f32
  linalg.yield %r : f32
 } -> tensor<1x33x1024xf32>
 %out = linalg.generic {indexing_maps=[#out_of_budget_hidden_w,#out_of_budget_hidden_id,#out_of_budget_hidden_id], iterator_types=["parallel","parallel","parallel"]} ins(%w,%unit : tensor<1024xf32>,tensor<1x33x1024xf32>) outs(%empty : tensor<1x33x1024xf32>) {
 ^bb0(%a: f32, %v: f32, %o: f32):
  %r = arith.mulf %a,%v : f32
  linalg.yield %r : f32
 } -> tensor<1x33x1024xf32>
 return %out : tensor<1x33x1024xf32>
}
