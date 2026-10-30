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

math::ExpOp siluExponent(linalg::GenericOp division, int64_t variableQuery = 0) {
  if (!division || division.getNumResults() != 1) return {};
  auto type = dyn_cast<RankedTensorType>(division.getResult(0).getType());
  // Only the model-owned, explicitly annotated symbolic Qwen SiLU.
  // The complete elementwise x/(1+exp(-x)) chain must still match.
  bool dynamicSilu = type && variableQuery >= 1 && variableQuery < 1024 &&
      (type.getRank() == 3 && type.getDimSize(0) == 1 && type.getDimSize(2) == 3072 &&
       (type.isDynamicDim(1) || (type.getDimSize(1) >= 1 && type.getDimSize(1) <= 32)));
  if (!type || (!dynamicSilu && !type.hasStaticShape()) || type.getEncoding() ||
      !type.getElementType().isF32() || type.getRank() != 3 ||
      type.getDimSize(0) < 1 || type.getDimSize(0) > 28 ||
      (variableQuery ? (!dynamicSilu && type.getDimSize(1) != variableQuery) : (type.getDimSize(1) != 1 && type.getDimSize(1) != 2 &&
       type.getDimSize(1) != 3 && type.getDimSize(1) != 4 && type.getDimSize(1) != 8)) ||
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


// The lifted dynamic Qwen graph fuses sigmoid into one elementwise region,
// whereas the already-tested static graph kept neg/exp/add/div separate.
// Match the complete sigmoid and its single x*sigmoid(x) consumer before
// replacing only exp. No global expf substitution or fast-math permission.
math::ExpOp siluExponentInline(linalg::GenericOp sigmoid, int64_t variableQuery) {
  if (!sigmoid || variableQuery < 1 || variableQuery >= 1024 ||
      !sigmoid.hasPureTensorSemantics() || sigmoid->hasAttr("library_call") ||
      sigmoid.getNumDpsInputs() != 1 || sigmoid.getNumDpsInits() != 1 ||
      sigmoid.getNumResults() != 1 || !sigmoid.getResult(0).hasOneUse()) return {};
  auto type = dyn_cast<RankedTensorType>(sigmoid.getResult(0).getType());
  if (!type || (type.getRank() != 3 || type.getDimSize(0) != 1 || type.getDimSize(2) != 3072 ||
                (!type.isDynamicDim(1) && (type.getDimSize(1) < 1 || type.getDimSize(1) > 32))) ||
      !type.getElementType().isF32() || type.getEncoding() ||
      sigmoid.getDpsInputs()[0].getType() != type ||
      sigmoid.getDpsInits()[0].getType() != type ||
      !llvm::all_of(sigmoid.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }) ||
      !llvm::all_of(sigmoid.getIteratorTypesArray(), [](auto it) {
        return it == utils::IteratorType::parallel;
      })) return {};
  Block *body = sigmoid.getBody();
  if (body->getNumArguments() != 2 || body->getOperations().size() != 5) return {};
  auto it = body->begin();
  auto neg = dyn_cast<arith::NegFOp>(*it++);
  auto exp = dyn_cast<math::ExpOp>(*it++);
  auto add = dyn_cast<arith::AddFOp>(*it++);
  auto div = dyn_cast<arith::DivFOp>(*it++);
  auto yield = dyn_cast<linalg::YieldOp>(*it++);
  FloatAttr one;
  if (!neg || !exp || !add || !div || !yield ||
      neg.getOperand() != body->getArgument(0) ||
      exp.getOperand() != neg.getResult() ||
      add.getLhs() != exp.getResult() ||
      !matchPattern(add.getRhs(), m_Constant(&one)) ||
      !one.getType().isF32() || !one.getValue().isExactlyValue(1.0) ||
      div.getLhs() != add.getRhs() || div.getRhs() != add.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != div.getResult() ||
      neg.getFastmath() != arith::FastMathFlags::none ||
      exp.getFastmath() != arith::FastMathFlags::none ||
      add.getFastmath() != arith::FastMathFlags::none ||
      div.getFastmath() != arith::FastMathFlags::none) return {};
  auto user = dyn_cast<linalg::GenericOp>(*sigmoid.getResult(0).getUsers().begin());
  if (!strictSiluElementwise(user, type, 2) ||
      user.getDpsInputs()[0] != sigmoid.getResult(0) ||
      user.getDpsInputs()[1] != sigmoid.getDpsInputs()[0]) return {};
  auto mul = dyn_cast<arith::MulFOp>(user.getBody()->front());
  auto mulYield = dyn_cast<linalg::YieldOp>(user.getBody()->getTerminator());
  if (!mul || !mulYield || mul.getFastmath() != arith::FastMathFlags::none ||
      mul.getLhs() != user.getBody()->getArgument(0) ||
      mul.getRhs() != user.getBody()->getArgument(1) ||
      mulYield.getOperand(0) != mul.getResult()) return {};
  return exp;
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

bool candidate(linalg::GenericOp op, bool shortExponentTree, int64_t decodeCapacity = 0) {
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
                   k == 64 || k == 128 || (decodeCapacity && k == decodeCapacity)) &&
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
bool hasStableSoftmaxShift(linalg::GenericOp sum, bool dynamicQ17 = false) {
  auto exponent = sum.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!exponent) return false;
  auto shift = exponent.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!shift || !shift.hasPureTensorSemantics() ||
      shift.getNumDpsInputs() != 2 || shift.getNumDpsInits() != 1 ||
      shift->hasAttr("library_call")) return false;
  auto type = dyn_cast<RankedTensorType>(shift.getDpsInputs()[0].getType());
  if (!type || (!dynamicQ17 && !type.hasStaticShape()) || !type.getElementType().isF32() ||
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
  if (!broadcastType || (!dynamicQ17 && !broadcastType.hasStaticShape())) return false;
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


// Diagnostic only: the lifted q17 graph has a symbolic Q/K length. Select
// exactly the native max-shift -> exp -> sum -> divide chain, without changing
// reduction order or division. This is not full CPU softmax compatibility.
math::ExpOp dynamicSoftmaxExponent(linalg::GenericOp sum, int64_t query) {
  if (query != 17 || !sum || !sum.hasPureTensorSemantics() ||
      sum->hasAttr("library_call") || sum.getNumDpsInputs() != 1 ||
      sum.getNumDpsInits() != 1 || sum.getNumResults() != 1 ||
      !isShortExponentSum(sum)) return {};
  auto input = dyn_cast<RankedTensorType>(sum.getDpsInputs()[0].getType());
  auto output = dyn_cast<RankedTensorType>(sum.getResult(0).getType());
  bool queryShape = input && output && input.getRank() == 4 && output.getRank() == 4 &&
      input.getDimSize(0) == 1 && input.getDimSize(1) == 16 &&
      (input.isDynamicDim(3) || input.getDimSize(3) == 1024) &&
      (input.isDynamicDim(2) || (input.getDimSize(2) >= 1 && input.getDimSize(2) <= 32)) &&
      output.getShape() == ArrayRef<int64_t>({1,16,input.getDimSize(2),1});
  if (!input || !output || !queryShape ||
      !input.getElementType().isF32() || !output.getElementType().isF32() ||
      input.getEncoding() || output.getEncoding()) return {};
  auto maps = sum.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[0].isIdentity() ||
      maps[1] != AffineMap::get(4,0,{getAffineDimExpr(0,sum.getContext()),
          getAffineDimExpr(1,sum.getContext()),getAffineDimExpr(2,sum.getContext()),
          getAffineConstantExpr(0,sum.getContext())},sum.getContext()) ||
      sum.getIteratorTypesArray().size() != 4 ||
      sum.getIteratorTypesArray()[0] != utils::IteratorType::parallel ||
      sum.getIteratorTypesArray()[1] != utils::IteratorType::parallel ||
      sum.getIteratorTypesArray()[2] != utils::IteratorType::parallel ||
      sum.getIteratorTypesArray()[3] != utils::IteratorType::reduction) return {};
  auto fill = sum.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0],m_AnyZeroFloat())) return {};
  auto body = sum.getBody();
  if (body->getOperations().size() != 2) return {};
  auto add = dyn_cast<arith::AddFOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!add || !yield || add.getFastmath() != arith::FastMathFlags::none ||
      add.getLhs() != body->getArgument(0) || add.getRhs() != body->getArgument(1) ||
      yield.getOperand(0) != add.getResult()) return {};
  auto exponent = sum.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!exponent || !hasStableSoftmaxShift(sum,true)) return {};
  bool normalized = llvm::any_of(sum.getResult(0).getUsers(),[&](Operation *user) {
    auto norm = dyn_cast<linalg::GenericOp>(user);
    if (!norm || !norm.hasPureTensorSemantics() || norm->hasAttr("library_call") ||
        norm.getNumDpsInputs() != 2 || norm.getNumDpsInits() != 1 ||
        norm.getDpsInputs()[0] != exponent.getResult(0) ||
        norm.getDpsInputs()[1] != sum.getResult(0) ||
        norm.getResult(0).getType() != input ||
        norm.getIndexingMapsArray().size() != 3 ||
        !norm.getIndexingMapsArray()[0].isIdentity() ||
        norm.getIndexingMapsArray()[1] != maps[1] ||
        !norm.getIndexingMapsArray()[2].isIdentity() ||
        !llvm::all_of(norm.getIteratorTypesArray(),[](auto it) {
          return it == utils::IteratorType::parallel; })) return false;
    if (norm.getBody()->getOperations().size() != 2) return false;
    auto div = dyn_cast<arith::DivFOp>(norm.getBody()->front());
    auto normYield = dyn_cast<linalg::YieldOp>(norm.getBody()->getTerminator());
    return div && normYield &&
        div.getFastmath() == arith::FastMathFlags::none &&
        div.getLhs() == norm.getBody()->getArgument(0) &&
        div.getRhs() == norm.getBody()->getArgument(1) &&
        normYield.getOperand(0) == div.getResult();
  });
  return normalized ? cast<math::ExpOp>(exponent.getBody()->front()) : math::ExpOp();
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

bool staticARMNormSquaredSum(linalg::GenericOp op, int64_t maxQuery = 32) {
  if (!armNormSquaredSum(op)) return false;
  auto type = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  return type.hasStaticShape() && type.getDimSize(1) <= maxQuery;
}


void stabilize(IRRewriter &rw, linalg::GenericOp op, bool reciprocalNormalization,
               bool shortExponentTree, int64_t decodeCapacity, int64_t cascadeLanes = 32) {
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
  if (k <= 64 || ((k == 128 || (decodeCapacity && k == decodeCapacity)) && shortExponentTree && isShortExponentSum(op))) {
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
    return;
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
  if (cascadeLanes == 16)
    result.getDefiningOp()->setAttr("hexagon.fp32_arm_static_rmsnorm_sum",rw.getUnitAttr());
  if (outputType.getRank() > 1) {
    ReassociationIndices group;
    for (int64_t i=0;i<outputType.getRank();++i) group.push_back(i);
    result=tensor::ExpandShapeOp::create(
        rw,loc,outputType,result,ArrayRef<ReassociationIndices>{group});
  }
  rw.replaceOp(op,result);
}

struct StabilizeFP32SumPass : ::impl::StabilizeFP32SumBase<StabilizeFP32SumPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,cf::ControlFlowDialect,scf::SCFDialect,
                    linalg::LinalgDialect,tensor::TensorDialect,math::MathDialect,
                    func::FuncDialect>();
  }
  void runOnOperation() override {
    // Explicit shape-scoped candidate annotation. Absence preserves all prior
    // policies; this is not admission of a decodeCapacity model or runtime.
    int64_t decodeCapacity = 0;
    if (auto attr = getOperation()->getAttr("hexagon.fp32_decode_capacity")) {
      auto capacity = dyn_cast<IntegerAttr>(attr);
      if (!capacity || !capacity.getType().isInteger(64) || (capacity.getInt() != 256 && capacity.getInt() != 512 && capacity.getInt() != 1024)) {
        getOperation()->emitError("hexagon.fp32_decode_capacity requires exactly 256, 512 or 1024 : i64");
        signalPassFailure();
        return;
      }
      decodeCapacity = capacity.getInt();
    }
    int64_t pairCapacity = 0;
    if (auto attr = getOperation()->getAttr("hexagon.fp32_pair_capacity")) {
      auto capacity = dyn_cast<IntegerAttr>(attr);
      if (!capacity || !capacity.getType().isInteger(64) || (capacity.getInt() != 256 && capacity.getInt() != 512 && capacity.getInt() != 1024) || decodeCapacity) {
        getOperation()->emitError("hexagon.fp32_pair_capacity requires exactly 256, 512 or 1024 : i64 without decode annotation");
        signalPassFailure();
        return;
      }
      pairCapacity = capacity.getInt();
    }
    int64_t prefillQuery = 0;
    auto prefillQueryAttr = getOperation()->getAttr("hexagon.fp32_prefill_query");
    auto prefillCapacityAttr = getOperation()->getAttr("hexagon.fp32_prefill_capacity");
    bool variablePrefill = false;
    if (auto attr = getOperation()->getAttr("hexagon.fp32_prefill_order")) {
      auto order = dyn_cast<StringAttr>(attr);
      if (!order || order.getValue() != "qk-fma-pv-openblas-armv8-k1024" || !prefillQueryAttr || !prefillCapacityAttr) {
        getOperation()->emitError("explicit prefill order requires qk-fma-pv-openblas-armv8-k1024 plus query/capacity annotations");
        signalPassFailure(); return;
      }
      variablePrefill = true;
    }
    if (prefillQueryAttr || prefillCapacityAttr) {
      auto query = dyn_cast_or_null<IntegerAttr>(prefillQueryAttr);
      auto capacity = dyn_cast_or_null<IntegerAttr>(prefillCapacityAttr);
      if (!query || !capacity || !query.getType().isInteger(64) ||
          !capacity.getType().isInteger(64) || (variablePrefill ? (query.getInt()<1 || query.getInt()>=1024) : (query.getInt()!=3 && query.getInt()!=4 && query.getInt()!=8)) ||
          capacity.getInt()!=1024 || decodeCapacity || pairCapacity) {
        getOperation()->emitError("prefill attention requires capacity1024 and legacy query3/4/8, or explicit order with 1<=query<1024; no decode/pair annotations");
        signalPassFailure(); return;
      }
      prefillQuery = query.getInt();
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
      if ((!variablePrefill && !getOperation().isExternal() && (!shortExpSumTree || !shortSoftmaxReciprocal)) || !callee ||
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
    SmallVector<math::ExpOp> softmaxExponents;
    SmallVector<math::ExpOp> dynamicSoftmaxExponents;
    getOperation().walk([&](linalg::GenericOp op) {
      if (shortSoftmaxExpCompat && variablePrefill)
        if (auto exponent = dynamicSoftmaxExponent(op,prefillQuery)) {
          dynamicSoftmaxExponents.push_back(exponent);
          // Defer this sum to the ARM four-lane policy, including static KV.
          if (getOperation()->hasAttr("hexagon.fp32_dynamic_softmax_max_q")) return;
        }
      if (candidate(op, shortExpSumTree, (prefillQuery ? 1024 : (decodeCapacity ? decodeCapacity : pairCapacity)))) {
        candidates.push_back(op);
        if (shortSoftmaxExpCompat && isShortExponentSum(op) && hasStableSoftmaxShift(op)) {
          auto producer = op.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
          auto exponent = cast<math::ExpOp>(producer.getBody()->front());
          if (!llvm::is_contained(softmaxExponents, exponent)) softmaxExponents.push_back(exponent);
        }
      }
    });
    IRRewriter rw(&getContext());
    if (fp32SiluExpCompat) {
      SmallVector<math::ExpOp> siluExponents;
      getOperation().walk([&](linalg::GenericOp op) {
        if (auto exponent = siluExponent(op, variablePrefill ? prefillQuery : 0)) siluExponents.push_back(exponent);
        if (auto exponent = siluExponentInline(op, variablePrefill ? prefillQuery : 0)) siluExponents.push_back(exponent);
      });
      for (auto exponent : siluExponents) {
        rw.setInsertionPoint(exponent);
        auto call = func::CallOp::create(rw, exponent.getLoc(), siluCallee,
            TypeRange{rw.getF32Type()}, ValueRange{exponent.getOperand()});
        call->setAttr("hexagon.fp32_silu_exp_compat", rw.getUnitAttr());
        rw.replaceOp(exponent, call.getResults());
      }
    }
    for (auto op : candidates)
      stabilize(rw,op,shortSoftmaxReciprocal,shortExpSumTree,(prefillQuery ? 1024 : (decodeCapacity ? decodeCapacity : pairCapacity)),
                getOperation()->hasAttr("hexagon.fp32_dynamic_rmsnorm_max_q") && staticARMNormSquaredSum(op) ? 16 : 32);
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
    for (auto exponent : dynamicSoftmaxExponents) {
      rw.setInsertionPoint(exponent);
      auto call = func::CallOp::create(rw,exponent.getLoc(),expCallee,
          TypeRange{rw.getF32Type()},ValueRange{exponent.getOperand()});
      call->setAttr("hexagon.fp32_softmax_exp_compat",rw.getUnitAttr());
      rw.replaceOp(exponent,call.getResults());
    }
    if (fp32AttentionDotOrder && prefillQuery) {
      SmallVector<linalg::BatchMatmulOp> prefillDots;
      getOperation().walk([&](linalg::BatchMatmulOp op) {
        if (isFP32AttentionDotCandidate(op,false,false,false,false,false,0,0,prefillQuery,variablePrefill))
          prefillDots.push_back(op);
      });
      for (auto op : prefillDots) {
        if (prefillQuery == 3 && !variablePrefill) {
          auto type=cast<RankedTensorType>(op.getDpsInputs()[0].getType());
          if (type.getDimSize(2) == 128) {
            if (failed(applyFP32AttentionDotOrder(rw,op,true))) {
              signalPassFailure(); return;
            }
          } else if (failed(applyFP32DecodePVOrder(rw,op,true))) {
            signalPassFailure(); return;
          }
        } else applyFP32PrefillAttentionOrder(rw,op);
      }
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
        if (fp32PairQKOrder && isFP32AttentionDotCandidate(op,false,false,true,false,false,false,pairCapacity)) qkDots.push_back(op);
        if (fp32PairPVOrder && isFP32AttentionDotCandidate(op,false,false,false,true,false,false,pairCapacity)) pvDots.push_back(op);
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
        if (isFP32AttentionDotCandidate(op,true,false,false,false,false,decodeCapacity) || llvm::is_contained(squareQKs,op)) dots.push_back(op);
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
        if (isFP32AttentionDotCandidate(op,false,true,false,false,false,decodeCapacity) || llvm::is_contained(squarePVs,op)) dots.push_back(op);
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
