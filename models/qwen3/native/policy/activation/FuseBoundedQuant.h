// SPDX-License-Identifier: BSD-3-Clause
// Fuse only round-even + zero point 0 + [-127,127] clamp + signed i8 cast.
// Integer saturation boundaries commute with round-even. After capping |x| at
// 127, bitcast(|x| + 2**23) - bitcast(2**23) is the exact rounded magnitude.
// Restore the sign and map NaN to -127, as the original unordered lower clamp
// does. This avoids converting the rounded value back to FP32 and then to i8.
// No reassociation, approximate math, or floating-point status observation.
namespace {
bool zeroActivationTensor(Value value, unsigned depth = 0) {
  if (depth > 5) return false;
  DenseElementsAttr dense;
  if (matchPattern(value, m_Constant(&dense)))
    return dense.isSplat() && dense.getElementType().isF32() &&
           dense.getSplatValue<llvm::APFloat>().isZero();
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return fill.hasPureTensorSemantics() && !fill->hasAttr("library_call") &&
           activationScalar(fill.getInputs()[0], 0.0);
  if (auto view = value.getDefiningOp<tensor::ExpandShapeOp>())
    return zeroActivationTensor(view.getSrc(), depth + 1);
  if (auto view = value.getDefiningOp<tensor::CollapseShapeOp>())
    return zeroActivationTensor(view.getSrc(), depth + 1);
  if (auto op = value.getDefiningOp<linalg::GenericOp>()) {
    auto &body = op.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (op.hasPureTensorSemantics() && !op->hasAttr("library_call") &&
        op.getNumDpsInputs() == 1 && op.getNumDpsInits() == 1 &&
        body.getOperations().size() == 1 && yield &&
        yield.getOperand(0) == body.getArgument(0))
      return zeroActivationTensor(op.getDpsInputs()[0], depth + 1);
  }
  return false;
}
bool plainQuantGeneric(linalg::GenericOp op, unsigned inputs, unsigned operations) {
  return op && op.hasPureTensorSemantics() && !op->hasAttr("library_call") &&
      op.getNumDpsInputs() == inputs && op.getNumDpsInits() == 1 &&
      op.getNumResults() == 1 &&
      op.getBody()->getOperations().size() == operations &&
      op.getBody()->getArgument(inputs).use_empty() &&
      llvm::all_of(op.getIteratorTypesArray(), [](auto kind) {
        return kind == utils::IteratorType::parallel;
      });
}
linalg::GenericOp nextQuantGeneric(linalg::GenericOp op) {
  if (!op || !op.getResult(0).hasOneUse()) return {};
  return dyn_cast<linalg::GenericOp>(*op.getResult(0).getUsers().begin());
}
bool fuseBoundedQuant(math::RoundEvenOp round, IRRewriter &b) {
  auto source = round->getParentOfType<linalg::GenericOp>();
  if (!plainQuantGeneric(source, 1, 2) ||
      round.getOperand() != source.getBody()->getArgument(0) ||
      cast<linalg::YieldOp>(source.getBody()->getTerminator()).getOperand(0) != round.getResult()) return false;
  auto add = nextQuantGeneric(source);
  if (!plainQuantGeneric(add, 2, 2) || add.getDpsInputs()[0] != source.getResult(0) ||
      !zeroActivationTensor(add.getDpsInputs()[1])) return false;
  auto sum = dyn_cast<arith::AddFOp>(add.getBody()->front());
  if (!sum || sum.getLhs() != add.getBody()->getArgument(0) ||
      sum.getRhs() != add.getBody()->getArgument(1) ||
      sum.getFastmath() != arith::FastMathFlags::none ||
      cast<linalg::YieldOp>(add.getBody()->getTerminator()).getOperand(0) != sum.getResult()) return false;
  auto clamp = nextQuantGeneric(add);
  if (!plainQuantGeneric(clamp, 1, 5) || clamp.getDpsInputs()[0] != add.getResult(0)) return false;
  auto it = clamp.getBody()->begin();
  auto lo = dyn_cast<arith::CmpFOp>(&*it++);
  auto low = dyn_cast<arith::SelectOp>(&*it++);
  auto hi = dyn_cast<arith::CmpFOp>(&*it++);
  auto high = dyn_cast<arith::SelectOp>(&*it++);
  if (!lo || !low || !hi || !high ||
      lo.getPredicate() != arith::CmpFPredicate::ULT ||
      hi.getPredicate() != arith::CmpFPredicate::UGT ||
      lo.getFastmath() != arith::FastMathFlags::none ||
      hi.getFastmath() != arith::FastMathFlags::none ||
      lo.getLhs() != clamp.getBody()->getArgument(0) || !activationScalar(lo.getRhs(), -127.0) ||
      low.getCondition() != lo.getResult() || low.getTrueValue() != lo.getRhs() || low.getFalseValue() != lo.getLhs() ||
      hi.getLhs() != low.getResult() || !activationScalar(hi.getRhs(), 127.0) ||
      high.getCondition() != hi.getResult() || high.getTrueValue() != hi.getRhs() || high.getFalseValue() != hi.getLhs() ||
      cast<linalg::YieldOp>(clamp.getBody()->getTerminator()).getOperand(0) != high.getResult()) return false;
  auto castOp = nextQuantGeneric(clamp);
  if (!plainQuantGeneric(castOp, 1, 2) || castOp.getDpsInputs()[0] != clamp.getResult(0)) return false;
  auto cast = dyn_cast<arith::FPToSIOp>(castOp.getBody()->front());
  if (!cast || !cast.getType().isSignlessInteger(8) ||
      cast.getIn() != castOp.getBody()->getArgument(0) ||
      mlir::cast<linalg::YieldOp>(castOp.getBody()->getTerminator()).getOperand(0) != cast.getResult()) return false;
  for (auto op : {source, clamp, castOp})
    if (llvm::any_of(op.getIndexingMapsArray(), [](AffineMap m) { return !m.isIdentity(); })) return false;
  if (!add.getIndexingMapsArray()[0].isIdentity() || !add.getIndexingMapsArray()[2].isIdentity() ||
      source.getDpsInputs()[0].getType() != castOp.getDpsInputs()[0].getType()) return false;
  b.setInsertionPoint(cast); auto loc = cast.getLoc();
  auto c = [&](uint32_t v) -> Value {return b.create<arith::ConstantIntOp>(loc, v, 32);};
  Value bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), cast.getIn());
  Value mag = b.create<arith::AndIOp>(loc, bits, c(0x7fffffff));
  Value capped = b.create<arith::MinUIOp>(loc, mag, c(0x42fe0000));
  Value value = b.create<arith::BitcastOp>(loc, b.getF32Type(), capped);
  Value magic = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(8388608.0));
  Value shifted = b.create<arith::AddFOp>(loc, value, magic);
  Value encoded = b.create<arith::BitcastOp>(loc, b.getI32Type(), shifted);
  Value integer = b.create<arith::SubIOp>(loc, encoded, c(0x4b000000));
  Value negative = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, bits, c(0));
  Value negated = b.create<arith::SubIOp>(loc, c(0), integer);
  Value signedValue = b.create<arith::SelectOp>(loc, negative, negated, integer);
  Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, mag, c(0x7f800000));
  Value result = b.create<arith::SelectOp>(loc, nan, c(uint32_t(-127)), signedValue);
  b.replaceOp(cast, b.create<arith::TruncIOp>(loc, b.getI8Type(), result).getResult());
  castOp.getDpsInputOperand(0)->set(source.getDpsInputs()[0]);
  return true;
}
} // namespace
