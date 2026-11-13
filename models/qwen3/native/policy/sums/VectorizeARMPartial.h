// SPDX-License-Identifier: BSD-3-Clause
// Preserve each ARM cascade lane's sequential additions. Pair independent
// groups in a full HVX register, without reassociation or FMA contraction.
static Value fuseARMSquareInput(IRRewriter &rw, Value input) {
  SmallVector<Operation *> views;
  Value value = input;
  while (auto *def = value.getDefiningOp()) {
    if (!isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp>(def))
      break;
    if (views.size() == 4)
      return {};
    views.push_back(def);
    value = def->getOperand(0);
  }
  auto square = value.getDefiningOp<linalg::GenericOp>();
  if (!square || !square.hasPureTensorSemantics() ||
      square.getNumDpsInputs() != 1 || square.getNumDpsInits() != 1 ||
      square.getNumResults() != 1 ||
      square.getDpsInputs()[0].getType() != value.getType())
    return {};
  for (auto map : square.getIndexingMapsArray())
    if (!map.isIdentity())
      return {};
  for (auto iterator : square.getIteratorTypesArray())
    if (iterator != utils::IteratorType::parallel)
      return {};
  auto *body = square.getBody();
  if (body->getOperations().size() != 2)
    return {};
  auto power = dyn_cast<math::FPowIOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!power || !yield || power->getOperand(0) != body->getArgument(0) ||
      yield.getOperand(0) != power.getResult())
    return {};
  auto exponent = power->getOperand(1).getDefiningOp<arith::ConstantIntOp>();
  if (!exponent || exponent.value() != 2)
    return {};
  // Rebuild only the tensor views on the unsquared input. The original square
  // remains valid for other consumers; canonicalization removes dead producers.
  value = square.getDpsInputs()[0];
  for (auto *view : llvm::reverse(views)) {
    auto *copy = rw.clone(*view);
    copy->setOperand(0, value);
    value = copy->getResult(0);
  }
  return value;
}

static Value vectorizeARMPartial(IRRewriter &rw, linalg::GenericOp partial) {
  auto inputType = cast<RankedTensorType>(partial.getDpsInputs()[0].getType());
  int64_t groups = inputType.getDimSize(0) * inputType.getDimSize(1);
  if (inputType.getDimSize(3) != 16 || inputType.getDimSize(0) > 16 ||
      groups % 2 != 0)
    return partial.getResult(0);
  OpBuilder::InsertionGuard guard(rw);
  rw.setInsertionPoint(partial);
  auto loc = partial.getLoc();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(rw, loc, n);
  };
  Value zero = index(0), one = index(1), two = index(2);
  Value chunks = index(inputType.getDimSize(1));
  Value steps = index(inputType.getDimSize(2));
  Value input = partial.getDpsInputs()[0];
  Value unsquared = fuseARMSquareInput(rw, input);
  if (unsquared)
    input = unsquared;
  Value pad = arith::ConstantOp::create(rw, loc, rw.getF32FloatAttr(0.0));
  auto type = VectorType::get({32}, rw.getF32Type());
  auto half = VectorType::get({16}, rw.getF32Type());
  SmallVector<int64_t> mask;
  for (int64_t lane = 0; lane < 32; ++lane)
    mask.push_back(lane);
  auto group = scf::ForOp::create(rw, loc, zero, index(groups), two,
                                partial.getDpsInits());
  rw.setInsertionPointToStart(group.getBody());
  Value g = group.getInductionVar();
  Value g1 = arith::AddIOp::create(rw, loc, g, one);
  Value i = arith::DivUIOp::create(rw, loc, g, chunks);
  Value j = arith::RemUIOp::create(rw, loc, g, chunks);
  Value i1 = arith::DivUIOp::create(rw, loc, g1, chunks);
  Value j1 = arith::RemUIOp::create(rw, loc, g1, chunks);
  auto read = [&](Value source, ValueRange offsets) -> Value {
    auto load = vector::TransferReadOp::create(rw, loc, half, source, offsets,
                                             pad, std::nullopt);
    load.setInBoundsAttr(rw.getBoolArrayAttr({true}));
    return load;
  };
  Value l0 = read(group.getRegionIterArgs()[0], ValueRange{i, j, zero});
  Value r0 = read(group.getRegionIterArgs()[0], ValueRange{i1, j1, zero});
  Value initial = vector::ShuffleOp::create(rw, loc, l0, r0, mask);
  auto step = scf::ForOp::create(rw, loc, zero, steps, one, ValueRange{initial});
  rw.setInsertionPointToStart(step.getBody());
  Value left = read(input, ValueRange{i, j, step.getInductionVar(), zero});
  Value right = read(input, ValueRange{i1, j1, step.getInductionVar(), zero});
  Value value = vector::ShuffleOp::create(rw, loc, left, right, mask);
  if (unsquared)
    value = arith::MulFOp::create(rw, loc, value, value);
  Value next = arith::AddFOp::create(rw, loc, value, step.getRegionIterArgs()[0]);
  scf::YieldOp::create(rw, loc, next);
  rw.setInsertionPointAfter(step);
  // Scalar V73 sfadd canonicalizes NaN to 0xffffffff, HVX to 0x7fffffff.
  // Once NaN occurs it stays NaN; restore the scalar bit contract at the end.
  auto ints = VectorType::get({32}, rw.getI32Type());
  auto constant = [&](int32_t x) -> Value {
    return arith::ConstantOp::create(rw, loc,
        DenseElementsAttr::get(ints, rw.getI32IntegerAttr(x)));
  };
  Value bits = arith::BitcastOp::create(rw, loc, ints, step.getResult(0));
  Value abs = arith::AndIOp::create(rw, loc, bits, constant(0x7fffffff));
  Value nan = arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::ugt,
                                 abs, constant(0x7f800000));
  Value canonical = arith::SelectOp::create(rw, loc, nan, constant(-1), bits);
  Value result = arith::BitcastOp::create(rw, loc, type, canonical);
  Value output = group.getRegionIterArgs()[0];
  for (int part = 0; part < 2; ++part) {
    Value slice = vector::ExtractStridedSliceOp::create(rw, loc, result,
        ArrayRef<int64_t>{part * 16}, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
    auto write = vector::TransferWriteOp::create(rw, loc, slice, output,
        ValueRange{part ? i1 : i, part ? j1 : j, zero}, std::nullopt);
    write.setInBoundsAttr(rw.getBoolArrayAttr({true}));
    output = write.getResult();
  }
  scf::YieldOp::create(rw, loc, output);
  group->setAttr("hexagon.fp32_sum_cascade_partial", rw.getUnitAttr());
  rw.eraseOp(partial);
  return group.getResult(0);
}
