// SPDX-License-Identifier: BSD-3-Clause
include "mlir/Pass/PassBase.td"

def StabilizeFP32Sum :
    Pass<"stabilize-fp32-sum", "mlir::func::FuncOp"> {
  let summary = "Explicit bounded FP32 reduction and stable-softmax numerical policies";
  let constructor = "hexagon::createStabilizeFP32SumPass()";
  let dependentDialects = ["arith::ArithDialect", "cf::ControlFlowDialect",
                          "func::FuncDialect", "linalg::LinalgDialect",
                          "math::MathDialect", "scf::SCFDialect", "tensor::TensorDialect"];
  let options = [
    Option<"shortExpSumTree", "short-exp-sum-tree", "bool", /*default=*/"false",
           "Explicit left fold for zero-initialized FP32 exponent sums of length 4/5/6/7, eight-lane tree with masked tail for 8..12/16/24/32/64; bounded numerical policy, not generic torch.sum.">,
    Option<"shortSoftmaxReciprocal", "short-softmax-reciprocal", "bool", /*default=*/"false",
           "With short-exp-sum-tree, normalize matching exp(x-max(x))/sum divisions by one strict reciprocal and multiplication; explicit CPU-order policy, not arbitrary exponent normalization.">,
    Option<"shortSoftmaxExpCompat", "short-softmax-exp-compat", "bool", /*default=*/"false",
           "With the short softmax sum/reciprocal policies, call an explicitly declared __hexagon_softmax_exp_compat(f32)->f32 only for recognized stable softmax exponent producers. Does not replace SiLU or other exp operations; caller must link the diagnostic runtime.">,
    Option<"fp32SiluExpCompat", "fp32-silu-exp-compat", "bool", /*default=*/"false",
           "Explicit compatibility policy for strict FP32 x/(1+exp(-x)) at bounded [B,1|2,3072] shapes, B=1..28. Replace only the exclusively used SiLU exp with declared __hexagon_silu_exp_compat(f32)->f32; no Softmax/global expf replacement, no reassociation. Runtime linking and device accuracy remain caller obligations.">,
    Option<"fp32AttentionDotOrder", "fp32-attention-dot-order", "bool", /*default=*/"false",
           "Explicit tested FP32 four-token attention policy: K=128 QK uses four noncontracted partials and opposite-half addition; K=4 context uses sequential FMA. Not a generic GEMM reassociation permission.">,
    Option<"fp32DecodeQKOrder", "fp32-decode-qk-order", "bool", /*default=*/"false",
           "Explicit bounded FP32 M=1 K=128 N=5..12, 24, 32 or 64 cached QK order: four-term prefix, four/eight interleaved partials, opposite-half tree and scalar tail. Default off; not a generic GEMV policy.">,
    Option<"fp32DecodePVOrder", "fp32-decode-pv-order", "bool", /*default=*/"false",
           "Explicit bounded FP32 M=1 K=64 N=128 cached PV order: sequential unfused accumulation with independent output columns innermost. Default off; not generic GEMM or reassociation permission.">,
    Option<"fp32PairQKOrder", "fp32-pair-qk-order", "bool", /*default=*/"false",
           "Explicit FP32 M=2 K=128 N=64 QK numerical policy: four interleaved unfused partials, opposite-half tree, no decode prefix. Independently checked two-query CPU order; default off, not a generic GEMM permission.">,
    Option<"fp32PairPVOrder", "fp32-pair-pv-order", "bool", /*default=*/"false",
           "Explicit FP32 M=2 K=64 N=128 PV numerical policy: sequential unfused accumulation with independent output columns. Default off; does not broaden the single-query policy.">
  ];
}
