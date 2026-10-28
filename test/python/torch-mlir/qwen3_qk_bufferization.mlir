// Exact late-QK structure from the accepted query1 graph. Empty tensor
// elimination independently creates identical input/output slices. The
// accumulator must retain the same head, component, column and FMA order.
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
#tail = affine_map<(d0)[s0] -> (128, -d0 + s0)>
#column = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @qwen_qk_copy(%q: tensor<16x128xf32>,
                         %keys: tensor<1x16x128x?xf32>,
                         %scores: tensor<16x?xf32>) -> tensor<16x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %capacity = tensor.dim %scores, %c1 : tensor<16x?xf32>
    %one = arith.cmpi eq, %capacity, %c1 : index
    %flag = tensor.from_elements %one : tensor<i1>
    %result = scf.for %head = %c0 to %c16 step %c1 iter_args(%a = %scores) -> tensor<16x?xf32> {
      %component_result = scf.for %component = %c0 to %c128 step %c1 iter_args(%b = %a) -> tensor<16x?xf32> {
        %query = tensor.extract %q[%head, %component] : tensor<16x128xf32>
        %query_scalar = tensor.from_elements %query : tensor<f32>
        %column_result = scf.for %tile = %c0 to %capacity step %c128 iter_args(%c = %b) -> tensor<16x?xf32> {
          %size = affine.min #tail(%tile)[%capacity]
          %read = tensor.extract_slice %c[%head, %tile] [1, %size] [1, 1] : tensor<16x?xf32> to tensor<?xf32>
          %write = tensor.extract_slice %c[%head, %tile] [1, %size] [1, 1] : tensor<16x?xf32> to tensor<?xf32>
          %updated = linalg.generic {indexing_maps = [#id, #scalar, #scalar, #id], iterator_types = ["parallel"]}
              ins(%read, %flag, %query_scalar : tensor<?xf32>, tensor<i1>, tensor<f32>) outs(%write : tensor<?xf32>) {
          ^bb0(%acc: f32, %broadcast: i1, %query_value: f32, %unused: f32):
            %index = linalg.index 0 : index
            %column = affine.apply #column(%tile)[%index]
            %key_column = arith.select %broadcast, %c0, %column : index
            %key = tensor.extract %keys[%c0, %head, %component, %key_column] : tensor<1x16x128x?xf32>
            %sum = math.fma %query_value, %key, %acc : f32
            linalg.yield %sum : f32
          } -> tensor<?xf32>
          %inserted = tensor.insert_slice %updated into %c[%head, %tile] [1, %size] [1, 1] : tensor<?xf32> into tensor<16x?xf32>
          scf.yield %inserted : tensor<16x?xf32>
        }
        scf.yield %column_result : tensor<16x?xf32>
      }
      scf.yield %component_result : tensor<16x?xf32>
    }
    return %result : tensor<16x?xf32>
  }
}
