//===- StabilizeFP32SumPass.cpp - Explicit bounded sum order ---------------===//
// SPDX-License-Identifier: BSD-3-Clause.
//
// Opt-in only; not part of the default pipeline. This is a numerical policy,
// not permission to reassociate other arithmetic or to drop BF16 rounding.
// The order is independently checked against PyTorch's CPU AVX2 SumKernel at
// commit 7661cd9c6b841b62b7f411aa52ec51f05457263b.
#include "hexagon/Transforms/Transforms.h"
#include "hexagon/Transforms/FP32AttentionDotOrder.h"
#include "hexagon/Transforms/FP32DecodeQKOrder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_STABILIZEFP32SUM
#include "hexagon/Transforms/Passes.h.inc"

namespace {
// Recognize the existing unfused SiLU division; do not infer SiLU merely from
// a negated exponent. Exclusively-used intermediates prevent affecting another
// consumer. This opt-in compatibility boundary changes only exp evaluation,
// not the surrounding negate/add/divide, and is deliberately shape-bounded.
bool strictSiluElementwise(linalg::GenericOp op, RankedTensorType type,
                           unsigned inputs) {
  if (!op || !op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
      op.getNumDpsInputs() != inputs || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 || op.getResult(0).getType() != type ||
      op.getDpsInits()[0].getType() != type ||
      !llvm::all_of(op.getDpsInputs(), [&](Value v) { return v.getType() == type; }) ||
      !llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(op.getIteratorTypesArray(), [](auto it) { return it == utils::IteratorType::parallel; }))
    return false;
  Block *body = op.getBody();
  if (body->getOperations().size() != 2 || body->getNumArguments() != inputs + 1)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         body->front().getNumResults() == 1 &&
         yield.getOperand(0) == body->front().getResult(0);
}

math::ExpOp siluExponent(linalg::GenericOp division) {
  if (!division || division.getNumResults() != 1) return {};
  auto type = dyn_cast<RankedTensorType>(division.getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      !type.getElementType().isF32() || type.getRank() != 3 ||
      type.getDimSize(0) < 1 || type.getDimSize(0) > 28 ||
      (type.getDimSize(1) != 1 && type.getDimSize(1) != 2) ||
      type.getDimSize(2) != 3072 || !strictSiluElementwise(division, type, 2))
    return {};
  auto div = dyn_cast<arith::DivFOp>(division.getBody()->front());
  if (!div || div.getFastmath() != arith::FastMathFlags::none ||
      div.getLhs() != division.getBody()->getArgument(0) ||
      div.getRhs() != division.getBody()->getArgument(1)) return {};
  auto addition = division.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
  if (!strictSiluElementwise(addition, type, 1) || !addition.getResult(0).hasOneUse()) return {};
  auto add = dyn_cast<arith::AddFOp>(addition.getBody()->front());
  if (!add || add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != addition.getBody()->getArgument(0)) return {};
  auto constant = add.getRhs().getDefiningOp<arith::ConstantOp>();
  auto one = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  if (!one || !one.getType().isF32() || !one.getValue().isExactlyValue(1.0)) return {};
  auto exponent = addition.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!strictSiluElementwise(exponent, type, 1) || !exponent.getResult(0).hasOneUse()) return {};
  auto exp = dyn_cast<math::ExpOp>(exponent.getBody()->front());
  if (!exp || exp.getFastmath() != arith::FastMathFlags::none ||
      exp.getOperand() != exponent.getBody()->getArgument(0)) return {};
  auto negative = exponent.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!strictSiluElementwise(negative, type, 1) || !negative.getResult(0).hasOneUse() ||
      negative.getDpsInputs()[0] != division.getDpsInputs()[0]) return {};
  auto neg = dyn_cast<arith::NegFOp>(negative.getBody()->front());
  if (!neg || neg.getFastmath() != arith::FastMathFlags::none ||
      neg.getOperand() != negative.getBody()->getArgument(0)) return {};
  return exp;
}


// Torch decomposition can materialize sigmoid(x) * x. ARM's native SiLU
// rounds x / (1 + exp(-x)) once instead. Only rewrite the already opted-in,
// checked compat-exp chain when its sigmoid has this single SiLU consumer.
bool rewriteARMSilu(IRRewriter &rw, linalg::GenericOp product) {
  if (!product || product.getNumResults() != 1) return false;
  auto type = dyn_cast<RankedTensorType>(product.getResult(0).getType());
  if (!type || type.getEncoding() || !type.getElementType().isF32() ||
      type.getRank() != 3 || type.getDimSize(0) != 1 ||
      type.getDimSize(2) != 3072 ||
      (!type.isDynamicDim(1) && (type.getDimSize(1) < 1 || type.getDimSize(1) > 32)) ||
      !strictSiluElementwise(product, type, 2)) return false;
  auto mul = dyn_cast<arith::MulFOp>(product.getBody()->front());
  if (!mul || mul.getFastmath() != arith::FastMathFlags::none ||
      mul.getLhs() != product.getBody()->getArgument(0) ||
      mul.getRhs() != product.getBody()->getArgument(1)) return false;
  auto sigmoid = product.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!sigmoid || !sigmoid.hasPureTensorSemantics() || sigmoid->hasAttr("library_call") ||
      sigmoid.getNumDpsInputs() != 1 || sigmoid.getNumDpsInits() != 1 ||
      sigmoid.getNumResults() != 1 || !sigmoid.getResult(0).hasOneUse() ||
      sigmoid.getResult(0).getType() != type || sigmoid.getDpsInits()[0].getType() != type ||
      sigmoid.getDpsInputs()[0] != product.getDpsInputs()[1] ||
      sigmoid.getDpsInputs()[0].getType() != type ||
      !llvm::all_of(sigmoid.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }) ||
      !llvm::all_of(sigmoid.getIteratorTypesArray(), [](auto it) { return it == utils::IteratorType::parallel; }))
    return false;
  Block *body = sigmoid.getBody();
  if (body->getNumArguments() != 2 || body->getOperations().size() != 5) return false;
  auto it = body->begin();
  auto neg = dyn_cast<arith::NegFOp>(*it++);
  auto exp = dyn_cast<func::CallOp>(*it++);
  auto add = dyn_cast<arith::AddFOp>(*it++);
  auto div = dyn_cast<arith::DivFOp>(*it++);
  auto yield = dyn_cast<linalg::YieldOp>(*it);
  if (!neg || !exp || !add || !div || !yield ||
      neg.getFastmath() != arith::FastMathFlags::none || neg.getOperand() != body->getArgument(0) ||
      exp.getCallee() != "__hexagon_silu_exp_compat" || !exp->hasAttr("hexagon.fp32_silu_exp_compat") ||
      exp.getNumOperands() != 1 || exp.getNumResults() != 1 || exp.getOperand(0) != neg.getResult() ||
      add.getFastmath() != arith::FastMathFlags::none || add.getLhs() != exp.getResult(0) ||
      div.getFastmath() != arith::FastMathFlags::none || div.getRhs() != add.getResult() ||
      div.getLhs() != add.getRhs() || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != div.getResult()) return false;
  auto constant = add.getRhs().getDefiningOp<arith::ConstantOp>();
  auto one = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  if (!one || !one.getType().isF32() || !one.getValue().isExactlyValue(1.0)) return false;
  rw.replaceOp(div, add.getResult());
  sigmoid->setAttr("hexagon.fp32_arm_silu_denominator", rw.getUnitAttr());
  rw.setInsertionPoint(mul);
  Value direct = arith::DivFOp::create(rw, mul.getLoc(), mul.getRhs(), mul.getLhs());
  rw.replaceOp(mul, direct);
  product->setAttr("hexagon.fp32_arm_silu_division", rw.getUnitAttr());
  return true;
}

bool isShortExponentSum(linalg::GenericOp op) {
  auto producer = op.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!producer || !fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()) ||
      !producer.hasPureTensorSemantics() || producer.getNumDpsInputs() != 1 ||
      producer.getNumDpsInits() != 1 || producer->hasAttr("library_call") ||
      !llvm::all_of(producer.getIndexingMapsArray(), [](AffineMap map) {
        return map.isIdentity();
      }) || !llvm::all_of(producer.getIteratorTypesArray(), [](auto it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  Block *body = producer.getBody();
  if (body->getOperations().size() != 2 || body->getNumArguments() != 2)
    return false;
  auto exp = dyn_cast<math::ExpOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return exp && exp.getFastmath() == arith::FastMathFlags::none &&
         exp.getOperand() == body->getArgument(0) && yield &&
         yield.getNumOperands() == 1 &&
         yield.getOperand(0) == exp.getResult();
}

bool candidate(linalg::GenericOp op, bool shortExponentTree, bool capacity256 = false) {
  if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 1 ||
      op.getNumDpsInits() != 1 || op.getNumResults() != 1 ||
      op->hasAttr("library_call") || op->hasAttr("hexagon.fp32_sum_cascade"))
    return false;
  auto input = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto output = cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!input.hasStaticShape() || !output.hasStaticShape() ||
      input.getRank() < 1 || output.getRank() < 1 ||
      input.getEncoding() || output.getEncoding() ||
      !input.getElementType().isF32() || !output.getElementType().isF32())
    return false;
  int64_t rank = input.getRank(), k = input.getShape().back(), rows = 1;
  bool shortSum = shortExponentTree &&
                  ((k >= 4 && k <= 12) || k == 16 || k == 24 || k == 32 ||
                   k == 64 || k == 128 || (capacity256 && k == 256)) &&
                  isShortExponentSum(op);
  if (!shortSum && k != 128 && k != 1024)
    return false;
  for (int64_t extent : input.getShape().drop_back()) {
    if (extent <= 0 || rows > 1048576 / extent)
      return false;
    rows *= extent;
  }
  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != static_cast<size_t>(rank) ||
      iterators.back() != utils::IteratorType::reduction ||
      !llvm::all_of(ArrayRef(iterators).drop_back(), [](auto it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  auto maps = op.getIndexingMapsArray();
  if (!maps[0].isIdentity() ||
      (output.getRank() != rank - 1 && output.getRank() != rank))
    return false;
  for (int64_t i = 0; i < rank - 1; ++i)
    if (maps[1].getResult(i) != getAffineDimExpr(i, op.getContext()) ||
        output.getDimSize(i) != input.getDimSize(i))
      return false;
  if (output.getRank() == rank &&
      (output.getShape().back() != 1 ||
       maps[1].getResult(rank - 1) != getAffineConstantExpr(0, op.getContext())))
    return false;
  Block *body = op.getBody();
  if (body->getOperations().size() != 2 || body->getNumArguments() != 2)
    return false;
  auto add = dyn_cast<arith::AddFOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return add && yield && add.getLhs() == body->getArgument(0) &&
         add.getRhs() == body->getArgument(1) &&
         add.getFastmath() == arith::FastMathFlags::none &&
         yield.getNumOperands() == 1 && yield.getOperand(0) == add.getResult();
}

Value zeroTensor(IRRewriter &rw, Location loc, ArrayRef<int64_t> shape) {
  Value empty = tensor::EmptyOp::create(rw, loc, shape, rw.getF32Type());
  Value zero = arith::ConstantOp::create(rw, loc, rw.getF32FloatAttr(0.0));
  return linalg::FillOp::create(rw, loc, zero, empty).getResult(0);
}

Value addTensor(IRRewriter &rw, Location loc, Value a, Value b) {
  auto type = cast<RankedTensorType>(a.getType());
  assert(type == b.getType());
  Value empty = tensor::EmptyOp::create(rw, loc, type.getShape(), rw.getF32Type());
  auto map = rw.getMultiDimIdentityMap(type.getRank());
  SmallVector<AffineMap> maps(3, map);
  SmallVector<utils::IteratorType> iters(type.getRank(), utils::IteratorType::parallel);
  auto op = linalg::GenericOp::create(
      rw, loc, TypeRange{type}, ValueRange{a,b}, ValueRange{empty}, maps, iters,
      [](OpBuilder &builder, Location loc, ValueRange args) {
        Value sum = arith::AddFOp::create(builder, loc, args[0], args[1]);
        linalg::YieldOp::create(builder, loc, sum);
      });
  return op.getResult(0);
}

Value slice(IRRewriter &rw, Location loc, Value input,
            ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
            ArrayRef<int64_t> resultShape) {
  SmallVector<OpFoldResult> o, s, strides;
  for (int64_t n : offsets) o.push_back(rw.getIndexAttr(n));
  for (int64_t n : sizes) { s.push_back(rw.getIndexAttr(n)); strides.push_back(rw.getIndexAttr(1)); }
  return tensor::ExtractSliceOp::create(
      rw, loc, RankedTensorType::get(resultShape, rw.getF32Type()), input, o, s, strides);
}

// A reciprocal may overflow for an arbitrary sum of tiny exponentials. Only
// recognize exp(x - max(x)) here: finite rows include exp(0), so 1 <= sum <= K.
// This remains an opt-in CPU-order policy, not general division reassociation.
bool hasStableSoftmaxShift(linalg::GenericOp sum, bool dynamicShape = false) {
  auto exponent = sum.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!exponent) return false;
  auto shift = exponent.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!shift || !shift.hasPureTensorSemantics() ||
      shift.getNumDpsInputs() != 2 || shift.getNumDpsInits() != 1 ||
      shift->hasAttr("library_call")) return false;
  auto type = dyn_cast<RankedTensorType>(shift.getDpsInputs()[0].getType());
  if (!type || (!dynamicShape && !type.hasStaticShape()) ||
      !type.getElementType().isF32() ||
      type.getRank() < 2 || shift.getResult(0).getType() != type) return false;
  int64_t rank = type.getRank();
  auto identity = AffineMap::getMultiDimIdentityMap(rank, sum.getContext());
  auto reducedMap = [&](RankedTensorType reduced) -> AffineMap {
    if (reduced.getRank() != rank - 1 && reduced.getRank() != rank)
      return {};
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank - 1; ++i) {
      if (reduced.getDimSize(i) != type.getDimSize(i)) return {};
      exprs.push_back(getAffineDimExpr(i, sum.getContext()));
    }
    if (reduced.getRank() == rank) {
      if (reduced.getDimSize(rank - 1) != 1) return {};
      exprs.push_back(getAffineConstantExpr(0, sum.getContext()));
    }
    return AffineMap::get(rank, 0, exprs, sum.getContext());
  };
  Value maximumValue = shift.getDpsInputs()[1];
  auto broadcastType = dyn_cast<RankedTensorType>(maximumValue.getType());
  if (!broadcastType || (!dynamicShape && !broadcastType.hasStaticShape()))
    return false;
  auto broadcastMap = reducedMap(broadcastType);
  auto shiftMaps = shift.getIndexingMapsArray();
  if (!broadcastMap || shiftMaps[0] != identity ||
      shiftMaps[1] != broadcastMap || shiftMaps[2] != identity ||
      !llvm::all_of(shift.getIteratorTypesArray(), [](auto it) {
        return it == utils::IteratorType::parallel;
      })) return false;
  Block *body = shift.getBody();
  if (body->getOperations().size() != 2) return false;
  auto sub = dyn_cast<arith::SubFOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!sub || sub.getLhs() != body->getArgument(0) ||
      sub.getRhs() != body->getArgument(1) ||
      sub.getFastmath() != arith::FastMathFlags::none || !yield ||
      yield.getOperand(0) != sub.getResult()) return false;
  if (auto expand = maximumValue.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto srcType = expand.getSrcType();
    if (srcType.getRank() != rank - 1 || broadcastType.getRank() != rank ||
        !reducedMap(srcType)) return false;
    auto groups = expand.getReassociationIndices();
    for (int64_t i = 0; i < rank - 2; ++i)
      if (groups[i] != ReassociationIndices{i}) return false;
    if (groups.back() != ReassociationIndices{rank - 2, rank - 1}) return false;
    maximumValue = expand.getSrc();
  }
  auto maximum = maximumValue.getDefiningOp<linalg::GenericOp>();
  if (!maximum || !maximum.hasPureTensorSemantics() ||
      maximum.getNumDpsInputs() != 1 || maximum.getNumDpsInits() < 1 ||
      maximumValue != maximum.getResult(0) || maximum->hasAttr("library_call") ||
      maximum.getDpsInputs()[0] != shift.getDpsInputs()[0]) return false;
  auto maxType = cast<RankedTensorType>(maximumValue.getType());
  auto maxMaps = maximum.getIndexingMapsArray();
  auto iterators = maximum.getIteratorTypesArray();
  if (maxMaps[0] != identity || maxMaps[1] != reducedMap(maxType) ||
      iterators.size() != static_cast<size_t>(rank) ||
      iterators.back() != utils::IteratorType::reduction ||
      !llvm::all_of(ArrayRef(iterators).drop_back(), [](auto it) {
        return it == utils::IteratorType::parallel;
      })) return false;
  auto fill = maximum.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill) return false;
  auto initial = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto scalar = initial ? dyn_cast<FloatAttr>(initial.getValue()) : FloatAttr();
  if (!scalar || !scalar.getValue().isInfinity() ||
      !scalar.getValue().isNegative()) return false;
  Block *maxBody = maximum.getBody();
  auto maxYield = dyn_cast<linalg::YieldOp>(maxBody->getTerminator());
  if (!maxYield || maxYield.getNumOperands() < 1) return false;
  auto max = maxYield.getOperand(0).getDefiningOp<arith::MaximumFOp>();
  return max && max.getLhs() == maxBody->getArgument(0) &&
         max.getRhs() == maxBody->getArgument(1) &&
         max.getFastmath() == arith::FastMathFlags::none;
}

// Shape alone cannot distinguish QK from PV when capacity == head_dim == 128.
// Recognize their connected native attention expression BEFORE any rewrites.
// Unconnected square matmuls remain untouched, even under both decode flags.
Value skipAttentionReshapes(Value value) {
  for (unsigned i = 0; i < 8; ++i) {
    if (auto op = value.getDefiningOp<tensor::CollapseShapeOp>()) value = op.getSrc();
    else if (auto op = value.getDefiningOp<tensor::ExpandShapeOp>()) value = op.getSrc();
    else return value;
  }
  return value;
}

bool attentionPointwise(linalg::GenericOp op, unsigned inputs) {
  if (!op || !op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
      op.getNumDpsInputs() != inputs || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 || op.getBody()->getNumArguments() != inputs + 1)
    return false;
  auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getEncoding() || type.getRank() < 2 || type.getShape().back() != 128 ||
      op.getDpsInputs()[0].getType() != type || op.getDpsInits()[0].getType() != type)
    return false;
  auto maps = op.getIndexingMapsArray();
  return maps.front().isIdentity() && maps.back().isIdentity() &&
      llvm::all_of(op.getIteratorTypesArray(), [](auto it) {
        return it == utils::IteratorType::parallel;
      });
}

linalg::BatchMatmulOp squareAttentionQK(linalg::BatchMatmulOp pv) {
  if (!isFP32AttentionDotCandidate(pv,false,true,false,false,true) ||
      cast<RankedTensorType>(pv.getDpsInputs()[0].getType()).getDimSize(2) != 128)
    return {};
  auto norm = skipAttentionReshapes(pv.getDpsInputs()[0]).getDefiningOp<linalg::GenericOp>();
  if (!attentionPointwise(norm,2) || norm.getBody()->getOperations().size() != 2) return {};
  auto div = dyn_cast<arith::DivFOp>(norm.getBody()->front());
  auto yield = dyn_cast<linalg::YieldOp>(norm.getBody()->getTerminator());
  if (!div || div.getFastmath() != arith::FastMathFlags::none ||
      div.getLhs() != norm.getBody()->getArgument(0) ||
      div.getRhs() != norm.getBody()->getArgument(1) || !yield ||
      yield.getOperand(0) != div.getResult()) return {};
  auto sum = norm.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
  if (!sum || !candidate(sum,true) || !isShortExponentSum(sum) ||
      !hasStableSoftmaxShift(sum) || norm.getDpsInputs()[0] != sum.getDpsInputs()[0] ||
      norm.getIndexingMapsArray()[1] != sum.getIndexingMapsArray()[1]) return {};
  auto exponent = sum.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  auto shift = exponent.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  auto mask = shift.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!attentionPointwise(mask,2) || mask.getBody()->getOperations().size() != 2) return {};
  auto add = dyn_cast<arith::AddFOp>(mask.getBody()->front());
  if (!add || add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != mask.getBody()->getArgument(0) ||
      add.getRhs() != mask.getBody()->getArgument(1) ||
      cast<linalg::YieldOp>(mask.getBody()->getTerminator()).getOperand(0) != add.getResult())
    return {};
  auto scale = mask.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!attentionPointwise(scale,1)) return {};
  auto scaleYield = dyn_cast<linalg::YieldOp>(scale.getBody()->getTerminator());
  auto mul = scaleYield ? scaleYield.getOperand(0).getDefiningOp<arith::MulFOp>() : arith::MulFOp();
  if (!mul || mul.getFastmath() != arith::FastMathFlags::none ||
      mul.getLhs() != scale.getBody()->getArgument(0)) return {};
  Value factor = mul.getRhs();
  if (auto trunc = factor.getDefiningOp<arith::TruncFOp>()) factor = trunc.getIn();
  FloatAttr constant;
  if (!matchPattern(factor,m_Constant(&constant))) return {};
  auto qk = skipAttentionReshapes(scale.getDpsInputs()[0]).getDefiningOp<linalg::BatchMatmulOp>();
  if (!qk || qk == pv || !isFP32AttentionDotCandidate(qk,true,false,false,false,true) ||
      qk.getResult(0).getType() != pv.getResult(0).getType()) return {};
  auto transpose = skipAttentionReshapes(qk.getDpsInputs()[1]).getDefiningOp<linalg::TransposeOp>();
  if (!transpose || !transpose.hasPureTensorSemantics() || transpose->hasAttr("library_call")) return {};
  auto permutation = transpose.getPermutation();
  if (permutation.size() < 2) return {};
  for (unsigned i = 0; i < permutation.size(); ++i) {
    unsigned expected = i + 2 == permutation.size() ? i + 1 :
                        i + 1 == permutation.size() ? i - 1 : i;
    if (permutation[i] != expected) return {};
  }
  return qk;
}

void rewriteShortSoftmaxDivision(IRRewriter &rw, linalg::GenericOp sum,
                                Value denominator) {
  if (!hasStableSoftmaxShift(sum)) return;
  Value numerator = sum.getDpsInputs()[0];
  auto type = cast<RankedTensorType>(numerator.getType());
  auto identity = rw.getMultiDimIdentityMap(type.getRank());
  SmallVector<linalg::GenericOp> divisions;
  for (Operation *user : sum.getResult(0).getUsers()) {
    auto div = dyn_cast<linalg::GenericOp>(user);
    if (!div || !div.hasPureTensorSemantics() || div.getNumDpsInputs() != 2 ||
        div.getNumDpsInits() != 1 || div->hasAttr("library_call") ||
        div.getDpsInputs()[0] != numerator ||
        div.getDpsInputs()[1] != sum.getResult(0) ||
        div.getDpsInits()[0].getType() != type ||
        !llvm::all_of(div.getIteratorTypesArray(), [](auto it) {
          return it == utils::IteratorType::parallel;
        }))
      continue;
    auto maps = div.getIndexingMapsArray();
    if (maps[0] != identity || maps[1] != sum.getIndexingMapsArray()[1] ||
        maps[2] != identity)
      continue;
    Block *body = div.getBody();
    if (body->getOperations().size() != 2 || body->getNumArguments() != 3)
      continue;
    auto quotient = dyn_cast<arith::DivFOp>(body->front());
    auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!quotient || quotient.getLhs() != body->getArgument(0) ||
        quotient.getRhs() != body->getArgument(1) ||
        quotient.getFastmath() != arith::FastMathFlags::none || !yield ||
        yield.getNumOperands() != 1 ||
        yield.getOperand(0) != quotient.getResult())
      continue;
    divisions.push_back(div);
  }
  if (divisions.empty())
    return;
  auto sumType = cast<RankedTensorType>(denominator.getType());
  Value empty = tensor::EmptyOp::create(
      rw,sum.getLoc(),sumType.getShape(),rw.getF32Type());
  auto map = rw.getMultiDimIdentityMap(sumType.getRank());
  SmallVector<AffineMap> maps(2,map);
  SmallVector<utils::IteratorType> iters(sumType.getRank(),
                                       utils::IteratorType::parallel);
  auto reciprocal = linalg::GenericOp::create(
      rw,sum.getLoc(),TypeRange{sumType},ValueRange{denominator},
      ValueRange{empty},maps,iters,
      [](OpBuilder &builder,Location loc,ValueRange args) {
        Value one = arith::ConstantOp::create(builder,loc,
                                              builder.getF32FloatAttr(1.0));
        Value inverse = arith::DivFOp::create(builder,loc,one,args[0]);
        linalg::YieldOp::create(builder,loc,inverse);
      });
  reciprocal->setAttr("hexagon.fp32_short_softmax_reciprocal",rw.getUnitAttr());
  for (auto div : divisions) {
    OpBuilder::InsertionGuard guard(rw);
    div.getDpsInputOperand(1)->set(reciprocal.getResult(0));
    Operation *quotient = &div.getBody()->front();
    rw.setInsertionPoint(quotient);
    Value product = arith::MulFOp::create(
        rw,quotient->getLoc(),div.getBody()->getArgument(0),
        div.getBody()->getArgument(1));
    rw.replaceOp(quotient,product);
    div->setAttr("hexagon.fp32_short_softmax_normalization",rw.getUnitAttr());
  }
}

Value stabilize(IRRewriter &rw, linalg::GenericOp op, bool reciprocalNormalization,
                bool shortExponentTree, bool capacity256, int64_t cascadeLanes = 32) {
  rw.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value input = op.getDpsInputs()[0], init = op.getDpsInits()[0];
  auto inputType = cast<RankedTensorType>(input.getType());
  auto outputType = cast<RankedTensorType>(init.getType());
  int64_t rank = inputType.getRank(), k = inputType.getShape().back();
  int64_t rows = inputType.getNumElements() / k;
  auto matrixType = RankedTensorType::get({rows,k}, rw.getF32Type());
  if (rank == 1) {
    input = tensor::ExpandShapeOp::create(
        rw, loc, matrixType, input, ArrayRef<ReassociationIndices>{{0,1}});
  } else if (rank > 2) {
    ReassociationIndices outer;
    for (int64_t i = 0; i < rank - 1; ++i) outer.push_back(i);
    SmallVector<ReassociationIndices> groups{outer, {rank - 1}};
    input = tensor::CollapseShapeOp::create(rw, loc, matrixType, input, groups);
  }
  // A 128-wide exponent sum is a Softmax specialization only under explicit
  // opt-in. Ordinary 128-wide sums (including RMSNorm) retain their existing
  // 32-lane cascade. Do not select this solely from the reduction length.
  if (k <= 64 || ((k == 128 || (capacity256 && k == 256)) && shortExponentTree && isShortExponentSum(op))) {
    // Pinned AVX2 softmax uses a left fold below one vector, and eight
    // interleaved partials plus an opposite-half tree for the larger admitted
    // lengths. Keep these policies separate; padding a short row to eight
    // would change the order. This is NOT a generic torch.sum rewrite.
    Value lanes;
    if (k < 8) {
      lanes = slice(rw,loc,input,{0,0},{rows,1},{rows,1});
      for (int64_t column = 1; column < k; ++column)
        lanes = addTensor(rw,loc,lanes,
            slice(rw,loc,input,{0,column},{rows,1},{rows,1}));
    } else {
      lanes = slice(rw,loc,input,{0,0},{rows,8},{rows,8});
      int64_t offset = 8;
      for (; offset + 8 <= k; offset += 8)
        lanes = addTensor(rw,loc,lanes,
                          slice(rw,loc,input,{0,offset},{rows,8},{rows,8}));
      if (offset < k) {
        const int64_t tail = k - offset;
        Value updated = addTensor(rw,loc,
            slice(rw,loc,lanes,{0,0},{rows,tail},{rows,tail}),
            slice(rw,loc,input,{0,offset},{rows,tail},{rows,tail}));
        SmallVector<OpFoldResult> positions{rw.getIndexAttr(0),rw.getIndexAttr(0)},
            sizes{rw.getIndexAttr(rows),rw.getIndexAttr(tail)},
            strides{rw.getIndexAttr(1),rw.getIndexAttr(1)};
        lanes = tensor::InsertSliceOp::create(rw,loc,updated,lanes,positions,sizes,strides);
      }
      for (int64_t half : {4,2,1})
        lanes = addTensor(rw,loc,
            slice(rw,loc,lanes,{0,0},{rows,half},{rows,half}),
            slice(rw,loc,lanes,{0,half},{rows,half},{rows,half}));
    }
    auto flatType = RankedTensorType::get({rows},rw.getF32Type());
    Value result = tensor::CollapseShapeOp::create(
        rw,loc,flatType,lanes,ArrayRef<ReassociationIndices>{{0,1}});
    if (outputType.getRank() > 1) {
      ReassociationIndices group;
      for (int64_t i=0;i<outputType.getRank();++i) group.push_back(i);
      init = tensor::CollapseShapeOp::create(
          rw,loc,flatType,init,ArrayRef<ReassociationIndices>{group});
    }
    result = addTensor(rw,loc,result,init);
    result.getDefiningOp()->setAttr("hexagon.fp32_sum_cascade",rw.getUnitAttr());
    result.getDefiningOp()->setAttr("hexagon.fp32_short_exp_sum_tree",
                                    rw.getUnitAttr());
    result.getDefiningOp()->setAttr("hexagon.fp32_short_exp_sum_order",
        rw.getStringAttr(k < 8 ? "left" : "eight-lane-opposite-half"));
    if (outputType.getRank() > 1) {
      ReassociationIndices group;
      for (int64_t i=0;i<outputType.getRank();++i) group.push_back(i);
      result = tensor::ExpandShapeOp::create(
          rw,loc,outputType,result,ArrayRef<ReassociationIndices>{group});
    }
    if (reciprocalNormalization)
      rewriteShortSoftmaxDivision(rw,op,result);
    rw.replaceOp(op,result);
    return result;
  }
  int64_t chunks = k == 1024 ? (cascadeLanes == 16 ? 4 : 2) : 1;
  int64_t steps = k / (chunks * cascadeLanes);
  int64_t vectorLanes = cascadeLanes / 4;
  auto expandedType = RankedTensorType::get({rows,chunks,steps,cascadeLanes}, rw.getF32Type());
  Value expanded = tensor::ExpandShapeOp::create(
      rw, loc, expandedType, input, ArrayRef<ReassociationIndices>{{0},{1,2,3}});
  auto partialType = RankedTensorType::get({rows,chunks,cascadeLanes}, rw.getF32Type());
  Value partialInit = zeroTensor(rw,loc,{rows,chunks,cascadeLanes});
  // Reduction precedes the 32 parallel lanes. The ordinary tiler vectorizes
  // lanes but cannot apply its innermost-reduction splitting to this loop.
  SmallVector<AffineMap> maps{
      rw.getMultiDimIdentityMap(4),
      AffineMap::get(4,0,{rw.getAffineDimExpr(0),rw.getAffineDimExpr(1),rw.getAffineDimExpr(3)},rw.getContext())};
  SmallVector<utils::IteratorType> iters{
      utils::IteratorType::parallel,utils::IteratorType::parallel,
      utils::IteratorType::reduction,utils::IteratorType::parallel};
  auto partial = linalg::GenericOp::create(
      rw,loc,TypeRange{partialType},ValueRange{expanded},ValueRange{partialInit},maps,iters,
      [](OpBuilder &builder,Location loc,ValueRange args) {
        Value sum=arith::AddFOp::create(builder,loc,args[0],args[1]);
        linalg::YieldOp::create(builder,loc,sum);
      });
  partial->setAttr("hexagon.fp32_sum_cascade_partial",rw.getUnitAttr());
  Value lanes=slice(rw,loc,partial.getResult(0),{0,0,0},{rows,1,cascadeLanes},{rows,cascadeLanes});
  for (int64_t chunk = 1; chunk < chunks; ++chunk)
    lanes=addTensor(rw,loc,lanes,slice(rw,loc,partial.getResult(0),{0,chunk,0},{rows,1,cascadeLanes},{rows,cascadeLanes}));
  Value eight=slice(rw,loc,lanes,{0,0},{rows,vectorLanes},{rows,vectorLanes});
  for (int64_t offset = vectorLanes; offset < cascadeLanes; offset += vectorLanes)
    eight=addTensor(rw,loc,eight,slice(rw,loc,lanes,{0,offset},{rows,vectorLanes},{rows,vectorLanes}));
  Value result=zeroTensor(rw,loc,{rows,1});
  for (int64_t lane = 0; lane < vectorLanes; ++lane)
    result=addTensor(rw,loc,result,slice(rw,loc,eight,{0,lane},{rows,1},{rows,1}));
  auto flatType=RankedTensorType::get({rows},rw.getF32Type());
  result=tensor::CollapseShapeOp::create(
      rw,loc,flatType,result,ArrayRef<ReassociationIndices>{{0,1}});
  if (outputType.getRank() > 1) {
    ReassociationIndices group;
    for (int64_t i=0;i<outputType.getRank();++i) group.push_back(i);
    init=tensor::CollapseShapeOp::create(
        rw,loc,flatType,init,ArrayRef<ReassociationIndices>{group});
  }
  // The caller's initial accumulator participates exactly once.
  result=addTensor(rw,loc,result,init);
  result.getDefiningOp()->setAttr("hexagon.fp32_sum_cascade",rw.getUnitAttr());
  if (outputType.getRank() > 1) {
    ReassociationIndices group;
    for (int64_t i=0;i<outputType.getRank();++i) group.push_back(i);
    result=tensor::ExpandShapeOp::create(
        rw,loc,outputType,result,ArrayRef<ReassociationIndices>{group});
  }
  rw.replaceOp(op,result);
  return result;
}

// Correctness-first opt-in for a symbolic outer query dimension. Apply the
// already checked 1024-wide RMSNorm or 128-wide KNorm order to one query row
// at a time. The runtime dimension controls only the outer scf.for; no query
// count is cast to a captured static example size. This is intentionally
// narrower than generic torch.sum, and its performance must be measured
// before enabling by default.
bool armNormSquaredSum(linalg::GenericOp op) {
  if (!op || !op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
      op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1) return false;
  auto input = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto output = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!input || !output || input.getEncoding() || output.getEncoding() ||
      !input.getElementType().isF32() || !output.getElementType().isF32() ||
      op.getDpsInits()[0].getType() != output) return false;
  // Query rows do not change the last-axis reduction algorithm. Static and
  // symbolic captures use one semantic matcher; bounds are checked separately.
  int64_t rank = input.getRank();
  if ((rank != 3 && rank != 4) || output.getRank() != rank ||
      input.getDimSize(0) != 1 || output.getDimSize(0) != 1 ||
      input.getDimSize(1) != output.getDimSize(1) ||
      (!input.isDynamicDim(1) && input.getDimSize(1) < 1) ||
      output.getShape().back() != 1) return false;
  if (rank == 3) {
    if (input.getDimSize(2) != 1024) return false;
  } else if ((input.getDimSize(2) != 8 && input.getDimSize(2) != 16) ||
             input.getDimSize(2) != output.getDimSize(2) ||
             input.getDimSize(3) != 128) return false;
  auto maps = op.getIndexingMapsArray();
  auto ctx = op.getContext();
  SmallVector<AffineExpr> reducedExprs;
  for (int64_t i = 0; i < rank - 1; ++i)
    reducedExprs.push_back(getAffineDimExpr(i, ctx));
  reducedExprs.push_back(getAffineConstantExpr(0, ctx));
  auto reduced = AffineMap::get(rank, 0, reducedExprs, ctx);
  auto iters = op.getIteratorTypesArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || maps[1] != reduced ||
      iters.size() != static_cast<size_t>(rank) ||
      iters.back() != utils::IteratorType::reduction ||
      !llvm::all_of(ArrayRef(iters).drop_back(), [](auto iterator) {
        return iterator == utils::IteratorType::parallel;
      })) return false;
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat())) return false;
  Block *body = op.getBody();
  if (body->getOperations().size() != 2 || body->getNumArguments() != 2) return false;
  auto add = dyn_cast<arith::AddFOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!add || !yield || add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != body->getArgument(0) ||
      add.getRhs() != body->getArgument(1) ||
      yield.getOperand(0) != add.getResult()) return false;
  auto square = op.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!square || !square.hasPureTensorSemantics() ||
      square->hasAttr("library_call") ||
      square.getNumDpsInputs() != 1 || square.getNumDpsInits() != 1 ||
      square.getResult(0).getType() != input ||
      square.getDpsInits()[0].getType() != input ||
      !square.getResult(0).hasOneUse() ||
      square.getBody()->getOperations().size() != 2 ||
      square.getBody()->getNumArguments() != 2 ||
      !llvm::all_of(square.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(square.getIteratorTypesArray(), [](auto iterator) {
        return iterator == utils::IteratorType::parallel;
      }))
    return false;
  auto power = dyn_cast<math::FPowIOp>(square.getBody()->front());
  auto squareYield = dyn_cast<linalg::YieldOp>(square.getBody()->getTerminator());
  if (!power || !squareYield ||
      power.getOperand(0) != square.getBody()->getArgument(0) ||
      squareYield.getOperand(0) != power.getResult()) return false;
  auto exponent = power.getOperand(1).getDefiningOp<arith::ConstantOp>();
  auto integer = exponent ? dyn_cast<IntegerAttr>(exponent.getValue()) : IntegerAttr();
  return integer && integer.getInt() == 2;
}

bool staticARMNormSquaredSum(linalg::GenericOp op, int64_t maxQuery) {
  if (!armNormSquaredSum(op)) return false;
  auto type = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  return type.hasStaticShape() && type.getDimSize(1) <= maxQuery;
}

void stabilizeBoundedDynamicRMSNormSum(IRRewriter &rw, linalg::GenericOp op,
                                       int64_t maxQuery) {
  rw.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value input = op.getDpsInputs()[0], init = op.getDpsInits()[0];
  auto inputType = cast<RankedTensorType>(input.getType());
  Value query = tensor::DimOp::create(rw, loc, input, 1);
  Value zero = arith::ConstantIndexOp::create(rw, loc, 0);
  Value one = arith::ConstantIndexOp::create(rw, loc, 1);
  Value maximum = arith::ConstantIndexOp::create(rw, loc, maxQuery);
  Value nonempty = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::sge,
                                       query, one);
  Value bounded = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::sle,
                                      query, maximum);
  Value valid = arith::AndIOp::create(rw, loc, nonempty, bounded);
  cf::AssertOp::create(rw, loc, valid,
                       "bounded dynamic RMSNorm query outside 1..maxQuery");

  auto loop = scf::ForOp::create(rw, loc, zero, query, one, ValueRange{init});
  rw.setInsertionPointToStart(loop.getBody());
  Value row = loop.getInductionVar();
  Value carried = loop.getRegionIterArgs().front();
  SmallVector<OpFoldResult> offsets{rw.getIndexAttr(0), row, rw.getIndexAttr(0)};
  SmallVector<OpFoldResult> inputSizes{rw.getIndexAttr(1), rw.getIndexAttr(1)};
  SmallVector<OpFoldResult> outputSizes{rw.getIndexAttr(1), rw.getIndexAttr(1)};
  SmallVector<int64_t> rowInputShape{1, 1}, rowOutputShape{1, 1};
  if (inputType.getRank() == 4) {
    offsets.push_back(rw.getIndexAttr(0));
    int64_t heads = inputType.getDimSize(2);
    inputSizes.push_back(rw.getIndexAttr(heads));
    outputSizes.push_back(rw.getIndexAttr(heads));
    rowInputShape.push_back(heads);
    rowOutputShape.push_back(heads);
  }
  inputSizes.push_back(rw.getIndexAttr(inputType.getShape().back()));
  outputSizes.push_back(rw.getIndexAttr(1));
  rowInputShape.push_back(inputType.getShape().back());
  rowOutputShape.push_back(1);
  SmallVector<OpFoldResult> strides(inputType.getRank(), rw.getIndexAttr(1));
  auto rowInputType = RankedTensorType::get(rowInputShape, rw.getF32Type());
  auto rowOutputType = RankedTensorType::get(rowOutputShape, rw.getF32Type());
  Value rowInput = tensor::ExtractSliceOp::create(
      rw, loc, rowInputType, input, offsets, inputSizes, strides);
  Value rowInit = tensor::ExtractSliceOp::create(
      rw, loc, rowOutputType, carried, offsets, outputSizes, strides);
  auto rowSum = linalg::GenericOp::create(
      rw, loc, TypeRange{rowOutputType}, ValueRange{rowInput},
      ValueRange{rowInit}, op.getIndexingMapsArray(), op.getIteratorTypesArray(),
      [](OpBuilder &builder, Location location, ValueRange args) {
        Value sum = arith::AddFOp::create(builder, location, args[0], args[1]);
        linalg::YieldOp::create(builder, location, sum);
      });
  // The opted-in ARM reference uses the same four-lane vector cascade
  // for each hidden, Q-normalization, and K-normalization row.
  Value ordered = stabilize(rw, rowSum, false, false, false,
                            16);
  if (inputType.getRank() == 4 && inputType.getDimSize(2) == 8)
    ordered.getDefiningOp()->setAttr("hexagon.fp32_arm_k_norm_sum", rw.getUnitAttr());
  else
    ordered.getDefiningOp()->setAttr("hexagon.fp32_arm_rmsnorm_sum", rw.getUnitAttr());
  rw.setInsertionPointToEnd(loop.getBody());
  Value updated = tensor::InsertSliceOp::create(
      rw, loc, ordered, carried, offsets, outputSizes, strides);
  scf::YieldOp::create(rw, loc, updated);
  rw.replaceOp(op, loop.getResults());
}

// Only the model's max-shift -> named exponential -> sum -> divide chain.
// The old q17 path statically cast the entire query axis; this candidate
// retains that axis as a runtime loop and specializes only the KV reduction
// width after an explicit 1024-slot assertion.
bool boundedDynamicSoftmaxSum(linalg::GenericOp sum, int64_t maxQuery) {
  if (!sum || !sum.hasPureTensorSemantics() || sum->hasAttr("library_call") ||
      sum.getNumDpsInputs() != 1 || sum.getNumDpsInits() != 1 ||
      sum.getNumResults() != 1 || !sum.getResult(0).hasOneUse()) return false;
  auto input = dyn_cast<RankedTensorType>(sum.getDpsInputs()[0].getType());
  auto output = dyn_cast<RankedTensorType>(sum.getResult(0).getType());
  // Static and symbolic KV/query axes use the same ARM Softmax row algorithm.
  bool boundedQueryShape = input && output && input.getRank() == 4 &&
      output.getRank() == 4 && input.getDimSize(0) == 1 &&
      input.getDimSize(1) == 16 &&
      (input.isDynamicDim(3) || input.getDimSize(3) == 1024) &&
      (input.isDynamicDim(2) ||
       (input.getDimSize(2) >= 1 && input.getDimSize(2) <= maxQuery)) &&
      output.getShape() == ArrayRef<int64_t>({1, 16, input.getDimSize(2), 1});
  if (!input || !output || input.getEncoding() || output.getEncoding() ||
      !input.getElementType().isF32() || !output.getElementType().isF32() ||
      !boundedQueryShape ||
      sum.getDpsInits()[0].getType() != output) return false;
  auto maps = sum.getIndexingMapsArray();
  auto context = sum.getContext();
  auto reduced = AffineMap::get(4, 0,
      {getAffineDimExpr(0, context), getAffineDimExpr(1, context),
       getAffineDimExpr(2, context), getAffineConstantExpr(0, context)},
      context);
  auto iters = sum.getIteratorTypesArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || maps[1] != reduced ||
      iters.size() != 4 || iters[0] != utils::IteratorType::parallel ||
      iters[1] != utils::IteratorType::parallel ||
      iters[2] != utils::IteratorType::parallel ||
      iters[3] != utils::IteratorType::reduction) return false;
  auto fill = sum.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
    return false;
  Block *body = sum.getBody();
  if (body->getOperations().size() != 2 || body->getNumArguments() != 2)
    return false;
  auto add = dyn_cast<arith::AddFOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!add || !yield || add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != body->getArgument(0) ||
      add.getRhs() != body->getArgument(1) ||
      yield.getOperand(0) != add.getResult()) return false;
  auto exponent = sum.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!exponent || !exponent.hasPureTensorSemantics() ||
      exponent->hasAttr("library_call") ||
      exponent.getNumDpsInputs() != 1 || exponent.getNumDpsInits() != 1 ||
      exponent.getResult(0).getType() != input ||
      exponent.getDpsInits()[0].getType() != input ||
      exponent.getBody()->getOperations().size() != 2 ||
      exponent.getBody()->getNumArguments() != 2 ||
      !llvm::all_of(exponent.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(exponent.getIteratorTypesArray(), [](auto iterator) {
        return iterator == utils::IteratorType::parallel;
      })) return false;
  Operation &expOperation = exponent.getBody()->front();
  auto expYield = dyn_cast<linalg::YieldOp>(exponent.getBody()->getTerminator());
  if (!expYield || expOperation.getNumResults() != 1 ||
      expYield.getOperand(0) != expOperation.getResult(0)) return false;
  if (auto call = dyn_cast<func::CallOp>(expOperation)) {
    if (call.getCallee() != "__hexagon_softmax_exp_compat" ||
        !call->hasAttr("hexagon.fp32_softmax_exp_compat") ||
        call.getNumOperands() != 1 ||
        call.getOperand(0) != exponent.getBody()->getArgument(0)) return false;
  } else if (auto exp = dyn_cast<math::ExpOp>(expOperation)) {
    if (exp.getFastmath() != arith::FastMathFlags::none ||
        exp.getOperand() != exponent.getBody()->getArgument(0)) return false;
  } else return false;
  if (!hasStableSoftmaxShift(sum, true)) return false;
  return llvm::any_of(sum.getResult(0).getUsers(), [&](Operation *user) {
    auto norm = dyn_cast<linalg::GenericOp>(user);
    if (!norm || !norm.hasPureTensorSemantics() ||
        norm->hasAttr("library_call") || norm.getNumDpsInputs() != 2 ||
        norm.getNumDpsInits() != 1 ||
        norm.getDpsInputs()[0] != exponent.getResult(0) ||
        norm.getDpsInputs()[1] != sum.getResult(0) ||
        norm.getResult(0).getType() != input ||
        norm.getDpsInits()[0].getType() != input ||
        norm.getIndexingMapsArray().size() != 3 ||
        !norm.getIndexingMapsArray()[0].isIdentity() ||
        norm.getIndexingMapsArray()[1] != reduced ||
        !norm.getIndexingMapsArray()[2].isIdentity() ||
        !llvm::all_of(norm.getIteratorTypesArray(), [](auto iterator) {
          return iterator == utils::IteratorType::parallel;
        }) || norm.getBody()->getOperations().size() != 2) return false;
    auto division = dyn_cast<arith::DivFOp>(norm.getBody()->front());
    auto normYield = dyn_cast<linalg::YieldOp>(norm.getBody()->getTerminator());
    return division && normYield &&
        division.getFastmath() == arith::FastMathFlags::none &&
        division.getLhs() == norm.getBody()->getArgument(0) &&
        division.getRhs() == norm.getBody()->getArgument(1) &&
        normYield.getOperand(0) == division.getResult();
  });
}

// ARM's native last-dimension Softmax accumulates four vector lanes, then
// combines opposite lanes before multiplying by one reciprocal per row.
Value stabilizeARMSoftmaxRow(IRRewriter &rw, linalg::GenericOp op) {
  rw.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value input = op.getDpsInputs()[0];
  auto matrixType = RankedTensorType::get({16,1024}, rw.getF32Type());
  Value matrix = tensor::CollapseShapeOp::create(rw,loc,matrixType,input,
      ArrayRef<ReassociationIndices>{{0,1,2},{3}});
  auto expandedType = RankedTensorType::get({16,256,4}, rw.getF32Type());
  Value expanded = tensor::ExpandShapeOp::create(rw,loc,expandedType,matrix,
      ArrayRef<ReassociationIndices>{{0},{1,2}});
  auto partialType = RankedTensorType::get({16,4},rw.getF32Type());
  Value init = zeroTensor(rw,loc,{16,4});
  SmallVector<AffineMap> maps{rw.getMultiDimIdentityMap(3),
      AffineMap::get(3,0,{rw.getAffineDimExpr(0),rw.getAffineDimExpr(2)},rw.getContext())};
  SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
      utils::IteratorType::reduction,utils::IteratorType::parallel};
  auto partial = linalg::GenericOp::create(rw,loc,TypeRange{partialType},
      ValueRange{expanded},ValueRange{init},maps,iters,
      [](OpBuilder &b,Location l,ValueRange args) {
        Value sum = arith::AddFOp::create(b,l,args[0],args[1]);
        linalg::YieldOp::create(b,l,sum);
      });
  partial->setAttr("hexagon.fp32_arm_softmax_partial",rw.getUnitAttr());
  Value pair = addTensor(rw,loc,
      slice(rw,loc,partial.getResult(0),{0,0},{16,2},{16,2}),
      slice(rw,loc,partial.getResult(0),{0,2},{16,2},{16,2}));
  Value total = addTensor(rw,loc,
      slice(rw,loc,pair,{0,0},{16,1},{16,1}),
      slice(rw,loc,pair,{0,1},{16,1},{16,1}));
  auto resultType = RankedTensorType::get({1,16,1,1},rw.getF32Type());
  Value result = tensor::ExpandShapeOp::create(rw,loc,resultType,total,
      ArrayRef<ReassociationIndices>{{0,1,2},{3}});
  result.getDefiningOp()->setAttr("hexagon.fp32_arm_softmax_sum",rw.getUnitAttr());
  rw.replaceOp(op,result);
  return result;
}

void stabilizeBoundedDynamicSoftmaxSum(IRRewriter &rw, linalg::GenericOp sum,
                                       int64_t maxQuery) {
  auto normalization = cast<linalg::GenericOp>(*sum.getResult(0).getUsers().begin());
  rw.setInsertionPoint(sum);
  Location loc = sum.getLoc();
  Value input = sum.getDpsInputs()[0], init = sum.getDpsInits()[0];
  Value query = tensor::DimOp::create(rw, loc, input, 2);
  Value capacity = tensor::DimOp::create(rw, loc, input, 3);
  Value zero = arith::ConstantIndexOp::create(rw, loc, 0);
  Value one = arith::ConstantIndexOp::create(rw, loc, 1);
  Value maximum = arith::ConstantIndexOp::create(rw, loc, maxQuery);
  Value expectedCapacity = arith::ConstantIndexOp::create(rw, loc, 1024);
  Value positive = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::sge,
                                        query, one);
  Value bounded = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::sle,
                                       query, maximum);
  Value valid = arith::AndIOp::create(rw, loc, positive, bounded);
  cf::AssertOp::create(rw, loc, valid,
                       "bounded dynamic Softmax query outside 1..maxQuery");
  Value fullCapacity = arith::CmpIOp::create(rw, loc,
      arith::CmpIPredicate::eq, capacity, expectedCapacity);
  cf::AssertOp::create(rw, loc, fullCapacity,
                       "bounded dynamic Softmax requires 1024 KV slots");
  auto loop = scf::ForOp::create(rw, loc, zero, query, one, ValueRange{init});
  rw.setInsertionPointToStart(loop.getBody());
  Value row = loop.getInductionVar();
  Value carried = loop.getRegionIterArgs().front();
  SmallVector<OpFoldResult> offsets{rw.getIndexAttr(0), rw.getIndexAttr(0),
                                    row, rw.getIndexAttr(0)};
  SmallVector<OpFoldResult> strides(4, rw.getIndexAttr(1));
  SmallVector<OpFoldResult> inputSizes{rw.getIndexAttr(1), rw.getIndexAttr(16),
                                      rw.getIndexAttr(1), rw.getIndexAttr(1024)};
  SmallVector<OpFoldResult> outputSizes{rw.getIndexAttr(1), rw.getIndexAttr(16),
                                       rw.getIndexAttr(1), rw.getIndexAttr(1)};
  auto rowInputType = RankedTensorType::get({1, 16, 1, 1024}, rw.getF32Type());
  auto rowOutputType = RankedTensorType::get({1, 16, 1, 1}, rw.getF32Type());
  Value rowInput = tensor::ExtractSliceOp::create(
      rw, loc, rowInputType, input, offsets, inputSizes, strides);
  Value rowInit = tensor::ExtractSliceOp::create(
      rw, loc, rowOutputType, carried, offsets, outputSizes, strides);
  auto rowSum = linalg::GenericOp::create(
      rw, loc, TypeRange{rowOutputType}, ValueRange{rowInput},
      ValueRange{rowInit}, sum.getIndexingMapsArray(),
      sum.getIteratorTypesArray(),
      [](OpBuilder &builder, Location location, ValueRange args) {
        Value add = arith::AddFOp::create(builder, location, args[0], args[1]);
        linalg::YieldOp::create(builder, location, add);
      });
  Value ordered = stabilizeARMSoftmaxRow(rw, rowSum);
  rw.setInsertionPointToEnd(loop.getBody());
  Value updated = tensor::InsertSliceOp::create(
      rw, loc, ordered, carried, offsets, outputSizes, strides);
  scf::YieldOp::create(rw, loc, updated);
  rw.setInsertionPoint(normalization);
  auto sumType = cast<RankedTensorType>(sum.getResult(0).getType());
  SmallVector<AffineMap> reciprocalMaps(2,rw.getMultiDimIdentityMap(4));
  SmallVector<utils::IteratorType> parallel(4,utils::IteratorType::parallel);
  auto reciprocal = linalg::GenericOp::create(rw,loc,TypeRange{sumType},
      ValueRange{loop.getResult(0)},ValueRange{init},reciprocalMaps,parallel,
      [](OpBuilder &b,Location l,ValueRange args) {
        Value one = arith::ConstantOp::create(b,l,b.getF32FloatAttr(1.0));
        Value inverse = arith::DivFOp::create(b,l,one,args[0]);
        linalg::YieldOp::create(b,l,inverse);
      });
  reciprocal->setAttr("hexagon.fp32_arm_softmax_reciprocal",rw.getUnitAttr());
  auto normalized = linalg::GenericOp::create(rw,loc,normalization.getResultTypes(),
      ValueRange{input,reciprocal.getResult(0)},normalization.getDpsInits(),
      normalization.getIndexingMapsArray(),normalization.getIteratorTypesArray(),
      [](OpBuilder &b,Location l,ValueRange args) {
        Value value = arith::MulFOp::create(b,l,args[0],args[1]);
        linalg::YieldOp::create(b,l,value);
      });
  normalized->setAttr("hexagon.fp32_arm_softmax_normalization",rw.getUnitAttr());
  rw.replaceOp(normalization,normalized.getResults());
  rw.replaceOp(sum,loop.getResults());
}


// Preserve ARM's four-lane range-reduction decision, including mixed groups.
static bool armRoPETrig(linalg::GenericOp op) {
  if (op.getNumDpsInputs()!=1 || op.getNumDpsInits()!=1 ||
      op.getNumResults()!=1 || op.getNumLoops()!=3 ||
      op.getNumReductionLoops()!=0 || op.getBody()->getOperations().size()!=2)
    return false;
  auto t=dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  if (!t || !t.getElementType().isF32() || t.getRank()!=3 ||
      t.getDimSize(0)!=1 || t.getDimSize(2)!=128 ||
      (!t.isDynamicDim(1) && (t.getDimSize(1)<1 || t.getDimSize(1)>32)) ||
      op.getResult(0).getType()!=t) return false;
  for (auto map:op.getIndexingMapsArray()) if (!map.isIdentity()) return false;
  Operation &trig=op.getBody()->front();
  return isa<math::SinOp,math::CosOp>(trig) &&
         trig.getOperand(0)==op.getBody()->getArgument(0) &&
         op.getBody()->getTerminator()->getOperand(0)==trig.getResult(0);
}
static void stabilizeARMTrig(IRRewriter &rw,linalg::GenericOp op) {
  rw.setInsertionPoint(op);Location loc=op.getLoc();
  SmallVector<AffineMap> maps;
  auto d0=rw.getAffineDimExpr(0),d1=rw.getAffineDimExpr(1),d2=rw.getAffineDimExpr(2);
  for (int lane=0;lane<4;++lane)
    maps.push_back(AffineMap::get(3,0,{d0,d1,d2.floorDiv(4)*4+lane},rw.getContext()));
  maps.push_back(rw.getMultiDimIdentityMap(3));
  StringRef callee=isa<math::SinOp>(op.getBody()->front())?
      "__hexagon_arm_sin4_lane":"__hexagon_arm_cos4_lane";
  Value x=op.getDpsInputs()[0];
  auto result=linalg::GenericOp::create(rw,loc,op.getResultTypes(),
      ValueRange{x,x,x,x},op.getDpsInits(),maps,op.getIteratorTypesArray(),
      [&](OpBuilder &b,Location l,ValueRange a) {
        Value index=linalg::IndexOp::create(b,l,2);
        Value four=arith::ConstantIndexOp::create(b,l,4);
        Value lane=arith::RemUIOp::create(b,l,index,four);
        lane=arith::IndexCastOp::create(b,l,b.getI32Type(),lane);
        auto call=func::CallOp::create(b,l,callee,TypeRange{b.getF32Type()},
            ValueRange{a[0],a[1],a[2],a[3],lane});
        linalg::YieldOp::create(b,l,call.getResults());
      });
  result->setAttr("hexagon.fp32_arm_rope_trig4",rw.getUnitAttr());
  rw.replaceOp(op,result.getResults());
}

struct StabilizeFP32SumPass : ::impl::StabilizeFP32SumBase<StabilizeFP32SumPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,cf::ControlFlowDialect,scf::SCFDialect,
                    linalg::LinalgDialect,tensor::TensorDialect,math::MathDialect,
                    func::FuncDialect>();
  }
  void runOnOperation() override {
    int64_t dynamicRMSNormMaxQuery = 0;
    int64_t dynamicSoftmaxMaxQuery = 0;
    if (auto attr = getOperation()->getAttr("hexagon.fp32_dynamic_rmsnorm_max_q")) {
      auto maximum = dyn_cast<IntegerAttr>(attr);
      if (!maximum || !maximum.getType().isInteger(64) ||
          maximum.getInt() < 1 || maximum.getInt() > 32) {
        getOperation()->emitError(
            "hexagon.fp32_dynamic_rmsnorm_max_q requires 1..32 : i64");
        signalPassFailure();
        return;
      }
      dynamicRMSNormMaxQuery = maximum.getInt();
    }
    if (auto attr = getOperation()->getAttr("hexagon.fp32_dynamic_softmax_max_q")) {
      auto maximum = dyn_cast<IntegerAttr>(attr);
      if (!maximum || !maximum.getType().isInteger(64) ||
          maximum.getInt() < 1 || maximum.getInt() > 32) {
        getOperation()->emitError(
            "hexagon.fp32_dynamic_softmax_max_q requires 1..32 : i64");
        signalPassFailure();
        return;
      }
      dynamicSoftmaxMaxQuery = maximum.getInt();
    }
    // Explicit shape-scoped candidate annotation. Absence preserves all prior
    // policies; this is not admission of a capacity256 model or runtime.
    bool capacity256 = false;
    if (auto attr = getOperation()->getAttr("hexagon.fp32_decode_capacity")) {
      auto capacity = dyn_cast<IntegerAttr>(attr);
      if (!capacity || !capacity.getType().isInteger(64) || capacity.getInt() != 256) {
        getOperation()->emitError("hexagon.fp32_decode_capacity requires exactly 256 : i64");
        signalPassFailure();
        return;
      }
      capacity256 = true;
    }
    if (shortSoftmaxReciprocal && !shortExpSumTree) {
      getOperation()->emitError("short-softmax-reciprocal requires short-exp-sum-tree");
      signalPassFailure();
      return;
    }
    // Declaration ownership stays with the enclosing module / runtime linker.
    // A function pass must not mutate sibling declarations while running in
    // parallel. The explicit declaration is also a fail-closed ABI contract.
    constexpr StringLiteral expCallee = "__hexagon_softmax_exp_compat";
    constexpr StringLiteral siluCallee = "__hexagon_silu_exp_compat";
    if (fp32SiluExpCompat) {
      auto module = getOperation()->getParentOfType<ModuleOp>();
      auto callee = module ? module.lookupSymbol<func::FuncOp>(siluCallee) : func::FuncOp();
      auto expected = FunctionType::get(&getContext(), {Float32Type::get(&getContext())},
                                       {Float32Type::get(&getContext())});
      if (!callee || !callee.isExternal() || callee.getFunctionType() != expected) {
        getOperation()->emitError("fp32-silu-exp-compat requires an external __hexagon_silu_exp_compat(f32) -> f32 declaration");
        signalPassFailure();
        return;
      }
    }
    if (shortSoftmaxExpCompat) {
      auto module = getOperation()->getParentOfType<ModuleOp>();
      auto callee = module ? module.lookupSymbol<func::FuncOp>(expCallee) : func::FuncOp();
      auto expected = FunctionType::get(&getContext(), {Float32Type::get(&getContext())},
                                       {Float32Type::get(&getContext())});
      if (!shortExpSumTree || !shortSoftmaxReciprocal || !callee ||
          !callee.isExternal() || callee.getFunctionType() != expected) {
        getOperation()->emitError("short-softmax-exp-compat requires short sum/reciprocal policies and an external __hexagon_softmax_exp_compat(f32) -> f32 declaration");
        signalPassFailure();
        return;
      }
    }
    SmallVector<linalg::BatchMatmulOp> squareQKs, squarePVs;
    if (fp32DecodeQKOrder || fp32DecodePVOrder) {
      getOperation().walk([&](linalg::BatchMatmulOp pv) {
        if (auto qk = squareAttentionQK(pv)) {
          if (!llvm::is_contained(squareQKs,qk)) squareQKs.push_back(qk);
          squarePVs.push_back(pv);
        }
      });
    }
    SmallVector<linalg::GenericOp> candidates;
    SmallVector<linalg::GenericOp> dynamicRMSNormSums;
    SmallVector<linalg::GenericOp> dynamicSoftmaxSums;
    SmallVector<math::ExpOp> softmaxExponents;
    getOperation().walk([&](linalg::GenericOp op) {
      if (dynamicRMSNormMaxQuery && armNormSquaredSum(op) &&
          cast<RankedTensorType>(op.getDpsInputs()[0].getType()).isDynamicDim(1))
        dynamicRMSNormSums.push_back(op);
      if (dynamicSoftmaxMaxQuery && boundedDynamicSoftmaxSum(op, dynamicSoftmaxMaxQuery)) {
        dynamicSoftmaxSums.push_back(op);
        return; // The generic static sum path must not rewrite the same op.
      }
      if (candidate(op, shortExpSumTree, capacity256)) {
        candidates.push_back(op);
        if (shortSoftmaxExpCompat && isShortExponentSum(op) && hasStableSoftmaxShift(op)) {
          auto producer = op.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
          auto exponent = cast<math::ExpOp>(producer.getBody()->front());
          if (!llvm::is_contained(softmaxExponents, exponent)) softmaxExponents.push_back(exponent);
        }
      }
    });
    IRRewriter rw(&getContext());
    if (getOperation()->hasAttr("hexagon.fp32_arm_rope_trig4")) {
      auto enabled=dyn_cast<BoolAttr>(getOperation()->getAttr("hexagon.fp32_arm_rope_trig4"));
      if (!enabled || !enabled.getValue()) {
        getOperation()->emitError("ARM RoPE trig requires true boolean opt-in");
        signalPassFailure(); return;
      }
      auto module=getOperation()->getParentOfType<ModuleOp>();
      auto expected=FunctionType::get(&getContext(),
          {rw.getF32Type(),rw.getF32Type(),rw.getF32Type(),rw.getF32Type(),rw.getI32Type()},
          {rw.getF32Type()});
      for (StringRef name:{"__hexagon_arm_sin4_lane","__hexagon_arm_cos4_lane"}) {
        auto callee=module?module.lookupSymbol<func::FuncOp>(name):func::FuncOp();
        if (!callee || !callee.isExternal() || callee.getFunctionType()!=expected) {
          getOperation()->emitError("ARM RoPE trig requires explicit five-argument external ABI");
          signalPassFailure(); return;
        }
      }
      SmallVector<linalg::GenericOp> trig;
      getOperation().walk([&](linalg::GenericOp op){if (armRoPETrig(op)) trig.push_back(op);});
      for (auto op:trig) stabilizeARMTrig(rw,op);
    }

    if (fp32SiluExpCompat) {
      SmallVector<math::ExpOp> siluExponents;
      getOperation().walk([&](linalg::GenericOp op) {
        if (auto exponent = siluExponent(op)) siluExponents.push_back(exponent);
      });
      for (auto exponent : siluExponents) {
        rw.setInsertionPoint(exponent);
        auto call = func::CallOp::create(rw, exponent.getLoc(), siluCallee,
            TypeRange{rw.getF32Type()}, ValueRange{exponent.getOperand()});
        call->setAttr("hexagon.fp32_silu_exp_compat", rw.getUnitAttr());
        rw.replaceOp(exponent, call.getResults());
      }
    }
    for (auto op : candidates) {
      bool armNorm = dynamicRMSNormMaxQuery > 0 && staticARMNormSquaredSum(op, dynamicRMSNormMaxQuery);
      Value ordered = stabilize(rw,op,shortSoftmaxReciprocal,shortExpSumTree,capacity256,
                                armNorm ? 16 : 32);
      if (armNorm)
        ordered.getDefiningOp()->setAttr("hexagon.fp32_arm_static_rmsnorm_sum", rw.getUnitAttr());
    }
    for (auto op : dynamicRMSNormSums)
      stabilizeBoundedDynamicRMSNormSum(rw, op, dynamicRMSNormMaxQuery);
    for (auto op : dynamicSoftmaxSums)
      stabilizeBoundedDynamicSoftmaxSum(rw, op, dynamicSoftmaxMaxQuery);
    if (dynamicRMSNormMaxQuery > 0) {
      SmallVector<linalg::GenericOp> products;
      getOperation().walk([&](linalg::GenericOp op) { products.push_back(op); });
      for (auto op : products) rewriteARMSilu(rw, op);
    }
    for (auto exponent : softmaxExponents) {
      auto producer = exponent->getParentOfType<linalg::GenericOp>();
      // A stable exponent sum alone is not Softmax. The checked normalization
      // matcher must have rewritten an actual consumer of this numerator.
      if (!llvm::any_of(producer.getResult(0).getUsers(), [&](Operation *user) {
            auto normalization = dyn_cast<linalg::GenericOp>(user);
            return normalization && normalization->hasAttr("hexagon.fp32_short_softmax_normalization") &&
                   normalization.getDpsInputs()[0] == producer.getResult(0);
          })) continue;
      rw.setInsertionPoint(exponent);
      auto call = func::CallOp::create(rw, exponent.getLoc(), expCallee,
                                      TypeRange{rw.getF32Type()}, ValueRange{exponent.getOperand()});
      call->setAttr("hexagon.fp32_softmax_exp_compat", rw.getUnitAttr());
      rw.replaceOp(exponent, call.getResults());
    }
    if (fp32AttentionDotOrder) {
      SmallVector<linalg::BatchMatmulOp> dots;
      getOperation().walk([&](linalg::BatchMatmulOp op) {
        if (isFP32AttentionDotCandidate(op)) dots.push_back(op);
      });
      for (auto op:dots)
        if (failed(applyFP32AttentionDotOrder(rw,op))) {
          getOperation()->emitError("failed to apply explicit FP32 attention dot order");
          signalPassFailure();
          return;
        }
    }
    if (fp32PairQKOrder || fp32PairPVOrder) {
      SmallVector<linalg::BatchMatmulOp> qkDots, pvDots;
      getOperation().walk([&](linalg::BatchMatmulOp op) {
        if (fp32PairQKOrder && isFP32AttentionDotCandidate(op,false,false,true)) qkDots.push_back(op);
        if (fp32PairPVOrder && isFP32AttentionDotCandidate(op,false,false,false,true)) pvDots.push_back(op);
      });
      for (auto op:qkDots)
        if (failed(applyFP32AttentionDotOrder(rw,op,true))) {
          getOperation()->emitError("failed explicit two-query QK numerical policy");
          signalPassFailure(); return;
        }
      for (auto op:pvDots)
        if (failed(applyFP32DecodePVOrder(rw,op,true))) {
          getOperation()->emitError("failed explicit two-query PV numerical policy");
          signalPassFailure(); return;
        }
    }
    if (fp32DecodeQKOrder) {
      SmallVector<linalg::BatchMatmulOp> dots;
      getOperation().walk([&](linalg::BatchMatmulOp op) {
        if (isFP32AttentionDotCandidate(op,true,false,false,false,false,capacity256) || llvm::is_contained(squareQKs,op)) dots.push_back(op);
      });
      for (auto op:dots)
        if (failed(applyFP32DecodeQKOrder(rw,op))) {
          getOperation()->emitError("failed to apply explicit cached FP32 QK order");
          signalPassFailure();
          return;
        }
    }
    if (fp32DecodePVOrder) {
      SmallVector<linalg::BatchMatmulOp> dots;
      getOperation().walk([&](linalg::BatchMatmulOp op) {
        if (isFP32AttentionDotCandidate(op,false,true,false,false,false,capacity256) || llvm::is_contained(squarePVs,op)) dots.push_back(op);
      });
      for (auto op:dots)
        if (failed(applyFP32DecodePVOrder(rw,op))) {
          getOperation()->emitError("failed to apply explicit cached FP32 PV order");
          signalPassFailure();
          return;
        }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createStabilizeFP32SumPass() {
  return std::make_unique<StabilizeFP32SumPass>();
}
