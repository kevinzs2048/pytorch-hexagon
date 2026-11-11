// Two graph temporaries and one buffer returned by a private call, all released
// together before the function returns. The call result has an origin that the
// buffer-origin analysis cannot resolve.
#id = affine_map<(d0) -> (d0)>
module {
  func.func private @fresh(%a: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %a, %c0 : tensor<?xf32>
    %e = tensor.empty(%n) : tensor<?xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%a : tensor<?xf32>) outs(%e : tensor<?xf32>) {
    ^bb0(%x: f32, %o: f32):
      %y = arith.negf %x : f32
      linalg.yield %y : f32
    } -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  func.func @graph(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %a, %c0 : tensor<?xf32>
    %f = call @fresh(%a) : (tensor<?xf32>) -> tensor<?xf32>
    %e1 = tensor.empty(%n) : tensor<?xf32>
    %t1 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
        ins(%f, %b : tensor<?xf32>, tensor<?xf32>) outs(%e1 : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    %e2 = tensor.empty(%n) : tensor<?xf32>
    %t2 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
        ins(%t1, %f : tensor<?xf32>, tensor<?xf32>) outs(%e2 : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.mulf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    %e3 = tensor.empty(%n) : tensor<?xf32>
    %t3 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
        ins(%t2, %t1 : tensor<?xf32>, tensor<?xf32>) outs(%e3 : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.subf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    return %t3 : tensor<?xf32>
  }
}
