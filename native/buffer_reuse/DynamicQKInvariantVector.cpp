// SPDX-License-Identifier: BSD-3-Clause
// Preserve each output's ordered FMA recurrence; hoist only an invariant
// immutable query tensor read and vectorize independent output columns.
#define QWEN_DYNAMIC_QK_INVARIANT_VECTOR 1
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
namespace { using namespace mlir;
bool admitted(func::FuncOp op) {
  if (op.isDeclaration()) return false;
  if (op.getSymName() == "qwen_dynamic_qk_invariant" &&
      op->hasAttr("qwen.diagnostic_dynamic_qk_invariant")) return true;
  // The test fixture requires an explicit marker; never instrument arbitrary
  // model functions merely because they contain loops.
  if (op.getSymName() == "qwen_qk_copy" &&
      op->hasAttr("qwen.diagnostic_loop_profile")) return true;
  auto order = op->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
#ifdef QWEN_DYNAMIC_QK_INVARIANT_VECTOR
  // Read from the actual original prepared query2..32 ABI, not the q1 ABI.
  // This mode instruments neither q1 nor an unmarked unrelated function.
  auto integerAttr = [&](StringRef name, int64_t expected) {
    auto attr = op->getAttrOfType<IntegerAttr>(name);
    return attr && attr.getInt() == expected;
  };
  auto tensor = [&](Type type, ArrayRef<int64_t> shape, bool integer = false) {
    auto ranked = dyn_cast<RankedTensorType>(type);
    return ranked && ranked.getShape() == shape &&
        (integer ? ranked.getElementType().isInteger(64)
                 : ranked.getElementType().isF32());
  };
  if (op.getSymName() != "Qwen3DynamicBoundProbe" ||
      op.getNumArguments() != 65 || op.getNumResults() != 57 ||
      !order || order.getValue() != "qk-fma-pv-openblas-armv8-k1024" ||
      !integerAttr("hexagon.fp32_prefill_query", 17) ||
      !integerAttr("hexagon.fp32_prefill_capacity", 1024) ||
      !integerAttr("hexagon.fp32_dynamic_rmsnorm_max_q", 32) ||
      !integerAttr("hexagon.fp32_dynamic_softmax_max_q", 32)) return false;
  for (unsigned index : {0u, 2u, 4u, 6u, 64u})
    if (!op.getArgument(index).getType().isInteger(64)) return false;
  for (unsigned index : {1u, 5u})
    if (!tensor(op.getArgument(index).getType(), {1, ShapedType::kDynamic}, true))
      return false;
  if (!tensor(op.getArgument(3).getType(), {ShapedType::kDynamic}, true) ||
      !tensor(op.getArgument(9).getType(),
              {1, 1, ShapedType::kDynamic, ShapedType::kDynamic}) ||
      !tensor(op.getResultTypes()[56], {1, 1, 151936})) return false;
  for (unsigned index = 7; index < 64; ++index) {
    if (index == 9) continue;
    if (!tensor(op.getArgument(index).getType(),
                {1, 8, ShapedType::kDynamic, 128})) return false;
  }
  for (unsigned index = 0; index < 56; ++index)
    if (!tensor(op.getResultTypes()[index],
                {1, 8, ShapedType::kDynamic, 128})) return false;
  return true;
#else
  auto tokens = op.getNumArguments() ?
      dyn_cast<RankedTensorType>(op.getArgument(0).getType()) : RankedTensorType();
  return op.getSymName() == "Qwen3DynamicBoundProbe" &&
      op.getNumArguments() == 62 && op.getNumResults() == 57 && tokens &&
      tokens.getShape() == ArrayRef<int64_t>({1, 1}) &&
      tokens.getElementType().isInteger(64) && order &&
      order.getValue() == "qk-fma-pv-openblas-armv8-k1024";
#endif
}

struct Candidate {
  linalg::GenericOp generic;
  arith::SelectOp select;
  tensor::ExtractOp query;
  math::FmaOp fma;
};
std::optional<Candidate> prove(linalg::GenericOp op, bool fixture) {
  if (op.getNumLoops()!=1 || op.getNumParallelLoops()!=1 ||
      op.getNumDpsInputs()!=3 || op.getNumDpsInits()!=1 || op.getNumResults()!=1) return {};
  auto output=dyn_cast<RankedTensorType>(op.getResult(0).getType());
  auto flag=dyn_cast<RankedTensorType>(op.getDpsInputOperand(2)->get().getType());
  if (!output || output.getRank()!=1 || !output.getElementType().isF32() ||
      !flag || flag.getRank()!=0 || !flag.getElementType().isInteger(1)) return {};
  auto maps=op.getIndexingMapsArray();
  if (maps.size()!=4 || !maps[0].isIdentity() || maps[1]!=maps[0] ||
      maps[3]!=maps[0] || maps[2].getNumDims()!=1 || maps[2].getNumResults()!=0) return {};
  Block &body=op.getRegion().front();
  if (body.getNumArguments()!=4 || !body.getArgument(3).use_empty() ||
      body.getOperations().size()!=4) return {};
  auto iterator=body.begin();
  auto select=dyn_cast<arith::SelectOp>(*iterator++);
  auto query=dyn_cast<tensor::ExtractOp>(*iterator++);
  auto fma=dyn_cast<math::FmaOp>(*iterator++);
  auto yield=dyn_cast<linalg::YieldOp>(*iterator++);
  if (!select || !query || !fma || !yield ||
      select.getCondition()!=body.getArgument(2) ||
      fma->getOperand(0)!=query.getResult() || fma->getOperand(1)!=body.getArgument(0) ||
      fma->getOperand(2)!=body.getArgument(1) ||
      fma.getFastmath()!=arith::FastMathFlags::none ||
      yield.getNumOperands()!=1 || yield.getOperand(0)!=fma.getResult()) return {};
  auto qtype=dyn_cast<RankedTensorType>(query.getTensor().getType());
  if (!qtype || qtype.getShape()!=ArrayRef<int64_t>({1,16,ShapedType::kDynamic,128}) ||
      !qtype.getElementType().isF32() || query.getIndices().size()!=4 ||
      query.getIndices()[2]!=select.getResult()) return {};
  DominanceInfo dominance(op->getParentOp());
  if (!dominance.properlyDominates(query.getTensor(),op) ||
      !dominance.properlyDominates(select.getTrueValue(),op) ||
      !dominance.properlyDominates(select.getFalseValue(),op)) return {};
  for (unsigned i : {0u,1u,3u})
    if (!dominance.properlyDominates(query.getIndices()[i],op)) return {};
  // The output extent is exactly the original min(remaining-columns,128).
  auto slice=op.getDpsInputOperand(0)->get().getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice || (slice.getMixedSizes().size()!=3 &&
      !(fixture && slice.getMixedSizes().size()==1))) return {};
  auto size=dyn_cast<Value>(slice.getMixedSizes().back());
  auto bound=size ? size.getDefiningOp<affine::AffineMinOp>() : nullptr;
  if (!bound || !llvm::is_contained(bound.getAffineMap().getResults(),
      getAffineConstantExpr(128,op.getContext()))) return {};
  return Candidate{op,select,query,fma};
}
}
mlir::LogicalResult hoistAndVectorizeQwenDynamicQK(mlir::ModuleOp module) {
  using namespace mlir;
  unsigned total=0;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!admitted(function)) continue;
    bool fixture=function.getSymName()=="qwen_dynamic_qk_invariant";
    SmallVector<Candidate> candidates;
    function.walk([&](linalg::GenericOp op) {
      if (auto candidate=prove(op,fixture)) candidates.push_back(*candidate);
    });
    if (candidates.empty()) continue;
    if (candidates.size()!=(fixture ? 1u : 28u)) {
      module.emitError("Complete original all28 invariant QK scalars required");return failure();
    }
    for (auto candidate : candidates) {
      IRRewriter rewriter(module.getContext());
      auto op=candidate.generic;rewriter.setInsertionPoint(op);
      auto flag=tensor::ExtractOp::create(rewriter,op.getLoc(),
          op.getDpsInputOperand(2)->get(),ValueRange{});
      auto index=arith::SelectOp::create(rewriter,op.getLoc(),flag,
          candidate.select.getTrueValue(),candidate.select.getFalseValue());
      SmallVector<Value> indices(candidate.query.getIndices());indices[2]=index;
      auto scalar=tensor::ExtractOp::create(rewriter,op.getLoc(),candidate.query.getTensor(),indices);
      op.getDpsInputOperand(2)->set(scalar);
      Block &body=op.getRegion().front();body.getArgument(2).setType(rewriter.getF32Type());
      candidate.fma->setOperand(0,body.getArgument(2));
      rewriter.eraseOp(candidate.query);rewriter.eraseOp(candidate.select);
      if (failed(linalg::vectorizeOpPrecondition(op,{128},{false}))) {
        module.emitError("Invariant QK must retain original bounded column vectorization");return failure();
      }
      rewriter.setInsertionPoint(op);
      auto vectorized=linalg::vectorize(rewriter,op,{128},{false});
      if (failed(vectorized)) return failure();
      rewriter.replaceOp(op,vectorized->replacements);++total;
    }
  }
  if (!total) return success();
  RewritePatternSet patterns(module.getContext());
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
  if (failed(applyPatternsGreedily(module,std::move(patterns)))) return failure();
  llvm::errs()<<"QWEN_DYNAMIC_QK_INVARIANT_VECTOR matched="<<total
      <<" per_output_FMA_order=original component_loop_unchanged=1 columns=128\n";
  return success();
}
