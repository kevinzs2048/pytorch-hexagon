// SPDX-License-Identifier: BSD-3-Clause
// Fuse the graph dequantization of a prepacked INT8 projection, and an
// optional residual addition that consumes it, into the runtime output stage.
//
// Replaced chain (all intermediates single-use, no fast-math flags):
//   alloc(i32) + fill 0 -> __hexagon_i8_prepacked_v1_matmul -> to_tensor
//   -> sitofp -> mulf(expand-broadcast row scale S) -> mulf(column scale C[col])
//   [-> expand_shape -> addf(R) | mulf(M)]
// The fused stage computes R + d or M * d with d = (float(acc) * S[X == 1 ? 0 :
// row]) * C[col], per element in that order; X is the broadcast source row count.
// Projections wider than 3072 columns (the LM head) are fused only for a
// single static row (decode), up to 151936 columns.
struct FuseInt8DequantPass
    : PassWrapper<FuseInt8DequantPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseInt8DequantPass)
  FuseInt8DequantPass() = default;
  FuseInt8DequantPass(const FuseInt8DequantPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  Option<unsigned> expectedResidual{*this, "expected-residual", llvm::cl::init(0)};
  Option<unsigned> expectedProduct{*this, "expected-product", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-fuse-i8-dequant"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
  StringRef getDescription() const final {
    return "Fuse INT8 projection dequantization (and residual) into the runtime output stage";
  }

  static Operation *single(linalg::GenericOp g) {
    Block &b = *g.getBody();
    if (b.getOperations().size() != 2) return nullptr;
    auto y = cast<linalg::YieldOp>(b.getTerminator());
    Operation *op = &b.front();
    if (y.getNumOperands() != 1 || op->getNumResults() != 1 || y.getOperand(0) != op->getResult(0))
      return nullptr;
    return op;
  }
  static bool parallel(linalg::GenericOp g, unsigned inputs) {
    return g && g.hasPureTensorSemantics() && g.getNumDpsInputs() == inputs &&
           g.getNumDpsInits() == 1 && g->getNumResults() == 1 &&
           llvm::all_of(g.getIteratorTypesArray(),
                        [](utils::IteratorType t) { return t == utils::IteratorType::parallel; });
  }
  static bool identity(linalg::GenericOp g) {
    return llvm::all_of(g.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); });
  }
  static linalg::GenericOp onlyUser(Value v) {
    return v.hasOneUse() ? dyn_cast<linalg::GenericOp>(*v.getUsers().begin()) : linalg::GenericOp();
  }
  // Binary FP op of the block arguments (either order) without FMF.
  template <typename Op> static bool binary(linalg::GenericOp g) {
    auto op = dyn_cast_or_null<Op>(single(g));
    if (!op || op.getFastmath() != arith::FastMathFlags::none) return false;
    Value a = g.getBody()->getArgument(0), b = g.getBody()->getArgument(1);
    return (op.getLhs() == a && op.getRhs() == b) || (op.getLhs() == b && op.getRhs() == a);
  }
  // Returns S for B[i, j] = S[i] (rank-1 row scale broadcast along columns,
  // as in the decode graph); the caller reshapes it to [rows, 1].
  static Value rowBroadcastSource(Value b) {
    auto g = b.getDefiningOp<linalg::GenericOp>();
    if (!parallel(g, 1) || g.getBody()->getOperations().size() != 1) return {};
    auto y = cast<linalg::YieldOp>(g.getBody()->getTerminator());
    MLIRContext *ctx = g.getContext();
    auto maps = g.getIndexingMapsArray();
    if (y.getOperand(0) != g.getBody()->getArgument(0) ||
        maps[0] != AffineMap::get(2, 0, {getAffineDimExpr(0, ctx)}, ctx) || !maps[1].isIdentity())
      return {};
    Value source = g.getDpsInputs()[0];
    auto type = dyn_cast<RankedTensorType>(source.getType());
    auto out = cast<RankedTensorType>(b.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32() ||
        type.getDimSize(0) != out.getDimSize(0))
      return {};
    return source;
  }
  // Returns S for B[i, j] = S[X == 1 ? 0 : i, 0], S = expand_shape [X, 1].
  static Value broadcastSource(Value b) {
    auto g = b.getDefiningOp<linalg::GenericOp>();
    if (!parallel(g, 0) || !identity(g) || g.getIndexingMapsArray()[0].getNumDims() != 2) return {};
    Block &body = *g.getBody();
    if (body.getOperations().size() != 5) return {};
    auto it = body.begin();
    auto index = dyn_cast<linalg::IndexOp>(&*it++);
    auto compare = dyn_cast<arith::CmpIOp>(&*it++);
    auto choose = dyn_cast<arith::SelectOp>(&*it++);
    auto extract = dyn_cast<tensor::ExtractOp>(&*it++);
    auto yield = dyn_cast<linalg::YieldOp>(&*it);
    if (!index || index.getDim() != 0 || !compare || !choose || !extract || !yield ||
        yield.getOperand(0) != extract.getResult() ||
        compare.getPredicate() != arith::CmpIPredicate::eq ||
        !matchPattern(compare.getRhs(), m_One()) ||
        choose.getCondition() != compare.getResult() ||
        !matchPattern(choose.getTrueValue(), m_Zero()) ||
        choose.getFalseValue() != index.getResult() ||
        extract.getIndices().size() != 2 || extract.getIndices()[0] != choose.getResult() ||
        !matchPattern(extract.getIndices()[1], m_Zero()))
      return {};
    Value source = extract.getTensor();
    auto type = dyn_cast<RankedTensorType>(source.getType());
    auto expand = source.getDefiningOp<tensor::ExpandShapeOp>();
    if (!type || type.getRank() != 2 || type.getDimSize(1) != 1 || !type.getElementType().isF32() ||
        !expand || source.getParentBlock() == &body)
      return {};
    auto shape = expand.getMixedOutputShape();
    auto rows = dyn_cast<Value>(shape[0]);
    if (!rows || rows != compare.getLhs()) return {};
    return source;
  }

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter rw(ctx);
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp c) {
      if (c.getCallee() == "__hexagon_i8_prepacked_v1_matmul") calls.push_back(c);
    });
    auto strided = [&](ArrayRef<int64_t> shape, Type element) {
      SmallVector<int64_t> dyn(shape.size(), ShapedType::kDynamic);
      return MemRefType::get(shape, element, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dyn));
    };
    Type f32 = rw.getF32Type();
    MemRefType matrix = strided({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
    MemRefType vector = strided({ShapedType::kDynamic}, f32);
    unsigned plain = 0, residual = 0, product = 0;
    for (auto call : calls) {
      if (call.getNumOperands() != 3) continue;
      auto output = call.getOperand(2).getDefiningOp<memref::CastOp>();
      auto alloc = output ? output.getSource().getDefiningOp<memref::AllocOp>() : memref::AllocOp();
      if (!alloc || !output->hasOneUse()) continue;
      linalg::FillOp fill;
      bufferization::ToTensorOp result;
      bool other = false;
      for (Operation *user : alloc->getUsers()) {
        if (auto f = dyn_cast<linalg::FillOp>(user)) fill = f;
        else if (auto t = dyn_cast<bufferization::ToTensorOp>(user)) result = t;
        else if (user != output.getOperation()) other = true;
      }
      if (other || !fill || !result || !matchPattern(fill.getDpsInputs()[0], m_Zero()) ||
          std::distance(alloc->getUsers().begin(), alloc->getUsers().end()) != 3 ||
          !fill->isBeforeInBlock(call) || !call->isBeforeInBlock(result))
        continue;
      auto type = alloc.getType();
      if (type.getRank() != 2 || !type.getElementType().isSignlessInteger(32) ||
          type.isDynamicDim(1) || type.getDimSize(1) % 32 ||
          // Wider outputs (the LM head) only for a single static row.
          (type.getDimSize(1) > 3072 && (type.getDimSize(0) != 1 || type.getDimSize(1) > 151936)))
        continue;
      auto convert = onlyUser(result.getResult());
      if (!parallel(convert, 1) || !identity(convert)) continue;
      auto sitofp = dyn_cast_or_null<arith::SIToFPOp>(single(convert));
      if (!sitofp || sitofp.getIn() != convert.getBody()->getArgument(0)) continue;
      auto scaled = onlyUser(convert->getResult(0));
      if (!parallel(scaled, 2) || !identity(scaled) || !binary<arith::MulFOp>(scaled)) continue;
      unsigned position = scaled.getDpsInputs()[0] == convert->getResult(0) ? 1 : 0;
      if (scaled.getDpsInputs()[1 - position] != convert->getResult(0)) continue;
      Value broadcast = scaled.getDpsInputs()[position];
      Value source = broadcastSource(broadcast);
      Value rowSource = source ? Value() : rowBroadcastSource(broadcast);
      if (!source && !rowSource) continue;
      auto weighted = onlyUser(scaled->getResult(0));
      if (!parallel(weighted, 2) || !binary<arith::MulFOp>(weighted) ||
          weighted.getDpsInputs()[0] != scaled->getResult(0))
        continue;
      auto maps = weighted.getIndexingMapsArray();
      auto d1 = AffineMap::get(2, 0, {getAffineDimExpr(1, ctx)}, ctx);
      Value column = weighted.getDpsInputs()[1];
      auto columnType = dyn_cast<RankedTensorType>(column.getType());
      if (!maps[0].isIdentity() || maps[1] != d1 || !maps[2].isIdentity() || !columnType ||
          columnType.getRank() != 1 || columnType.getDimSize(0) != type.getDimSize(1) ||
          !columnType.getElementType().isF32())
        continue;
      // Optional residual addition consuming the dequantized projection.
      Operation *replaced = weighted;
      tensor::ExpandShapeOp expand;
      Value residualValue;
      bool multiply = false;
      if (weighted->getResult(0).hasOneUse())
        if (auto e = dyn_cast<tensor::ExpandShapeOp>(*weighted->getResult(0).getUsers().begin()))
          if (auto add = onlyUser(e.getResult());
              add && parallel(add, 2) && identity(add) &&
              (binary<arith::AddFOp>(add) || binary<arith::MulFOp>(add))) {
            auto et = e.getResultType();
            Value r = add.getDpsInputs()[0] == e.getResult() ? add.getDpsInputs()[1] : add.getDpsInputs()[0];
            if (et.getRank() == 3 && et.getDimSize(0) == 1 && r != e.getResult() &&
                (add.getDpsInputs()[0] == e.getResult() || add.getDpsInputs()[1] == e.getResult()) &&
                (e.getReassociationIndices() == SmallVector<ReassociationIndices>{{0, 1}, {2}} ||
                 (e.getReassociationIndices() == SmallVector<ReassociationIndices>{{0}, {1, 2}} &&
                  et.getDimSize(1) == 1)) &&
                r.getType() == e.getResult().getType()) {
              expand = e;
              residualValue = r;
              multiply = binary<arith::MulFOp>(add);
              replaced = add;
            }
          }
      rw.setInsertionPoint(replaced);
      Location loc = call.getLoc();
      auto outType = MemRefType::get(type.getShape(), f32);
      Value out = rw.create<memref::AllocOp>(loc, outType, alloc.getDynamicSizes());
      if (rowSource) {
        auto rt = cast<RankedTensorType>(rowSource.getType());
        SmallVector<OpFoldResult> shape{tensor::getMixedSize(rw, loc, rowSource, 0), rw.getIndexAttr(1)};
        source = rw.create<tensor::ExpandShapeOp>(loc,
            RankedTensorType::get({rt.getDimSize(0), 1}, rt.getElementType()), rowSource,
            SmallVector<ReassociationIndices>{{0, 1}}, shape);
      }
      auto sourceType = cast<RankedTensorType>(source.getType());
      Value scaleBuffer = rw.create<bufferization::ToBufferOp>(loc,
          MemRefType::get(sourceType.getShape(), f32), source, true);
      Value columnBuffer = rw.create<bufferization::ToBufferOp>(loc,
          MemRefType::get(columnType.getShape(), f32), column, true);
      SmallVector<Value> operands{call.getOperand(0), call.getOperand(1),
          rw.create<memref::CastOp>(loc, matrix, scaleBuffer),
          rw.create<memref::CastOp>(loc, vector, columnBuffer)};
      if (residualValue) {
        Value collapsed = rw.create<tensor::CollapseShapeOp>(loc, residualValue,
            SmallVector<ReassociationIndices>{{0, 1}, {2}});
        auto ct = cast<RankedTensorType>(collapsed.getType());
        Value rb = rw.create<bufferization::ToBufferOp>(loc, MemRefType::get(ct.getShape(), f32), collapsed, true);
        operands.push_back(rw.create<memref::CastOp>(loc, matrix, rb));
      }
      operands.push_back(rw.create<memref::CastOp>(loc, matrix, out));
      StringRef symbol = !residualValue ? "__hexagon_i8_prepacked_v1_matmul_dequant"
                         : multiply ? "__hexagon_i8_prepacked_v1_matmul_dequant_product"
                                    : "__hexagon_i8_prepacked_v1_matmul_dequant_residual";
      if (!module.lookupSymbol(symbol)) {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToStart(module.getBody());
        SmallVector<Type> inputs;
        for (Value v : operands) inputs.push_back(v.getType());
        auto fn = rw.create<func::FuncOp>(module.getLoc(), symbol, rw.getFunctionType(inputs, {}));
        fn.setPrivate();
        fn->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      rw.create<func::CallOp>(loc, symbol, TypeRange{}, operands);
      Value fused = rw.create<bufferization::ToTensorOp>(loc,
          RankedTensorType::get(type.getShape(), f32), out, true, true);
      if (expand) {
        fused = rw.create<tensor::ExpandShapeOp>(loc, expand.getResultType(), fused,
            expand.getReassociationIndices(), expand.getMixedOutputShape());
      }
      rw.replaceOp(replaced, fused);
      if (expand) {
        rw.eraseOp(expand);
        rw.eraseOp(weighted);
      }
      rw.eraseOp(scaled);
      if (auto b = broadcast.getDefiningOp(); b && b->use_empty()) rw.eraseOp(b);
      rw.eraseOp(convert);
      rw.eraseOp(result);
      rw.eraseOp(call);
      rw.eraseOp(output);
      rw.eraseOp(fill);
      rw.eraseOp(alloc);
      ++(!residualValue ? plain : multiply ? product : residual);
    }
    if ((expectedCount && plain + residual + product != expectedCount) ||
        (expectedResidual && residual != expectedResidual) ||
        (expectedProduct && product != expectedProduct)) {
      module.emitError() << "fused INT8 dequant coverage: expected " << expectedCount.getValue()
                         << " (" << expectedResidual.getValue() << " residual, "
                         << expectedProduct.getValue() << " product), got "
                         << plain + residual + product << " (" << residual << " residual, "
                         << product << " product)";
      return signalPassFailure();
    }
    llvm::errs() << "fused INT8 dequant: " << plain << " plain, " << residual << " residual, "
                 << product << " product\n";
  }
};
