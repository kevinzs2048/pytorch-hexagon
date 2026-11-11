// SPDX-License-Identifier: BSD-3-Clause
// Replace the graph rotary embedding of a head-major transpose,
//   X = transpose(Y, [0, 2, 1, 3]);  X * C + concat(-X[..., 64:], X[..., :64]) * S
// (C, S broadcast over heads), with one call to the row-parallel Triton kernel
// qwen_rope_heads, which reads Y directly and evaluates each element in the
// same order. Every replaced intermediate must be used only inside the chain.
struct FuseParallelRoPEPass
    : PassWrapper<FuseParallelRoPEPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseParallelRoPEPass)
  FuseParallelRoPEPass() = default;
  FuseParallelRoPEPass(const FuseParallelRoPEPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-fuse-parallel-rope"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
  StringRef getDescription() const final {
    return "Fuse head-major transpose and rotary embedding into a row-parallel kernel call";
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
  template <typename Op> static bool binary(linalg::GenericOp g) {
    auto op = dyn_cast_or_null<Op>(single(g));
    if (!op || op.getFastmath() != arith::FastMathFlags::none) return false;
    Value a = g.getBody()->getArgument(0), b = g.getBody()->getArgument(1);
    return (op.getLhs() == a && op.getRhs() == b) || (op.getLhs() == b && op.getRhs() == a);
  }
  // g = mulf(first, table) with maps [identity, (d0, 0, d2, d3), identity].
  static Value broadcastMul(linalg::GenericOp g, Value first) {
    if (!parallel(g, 2) || !binary<arith::MulFOp>(g) || g.getDpsInputs()[0] != first) return {};
    MLIRContext *ctx = g.getContext();
    auto e = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
    auto heads = AffineMap::get(4, 0, {e(0), getAffineConstantExpr(0, ctx), e(2), e(3)}, ctx);
    auto maps = g.getIndexingMapsArray();
    if (!maps[0].isIdentity() || maps[1] != heads || !maps[2].isIdentity()) return {};
    Value table = g.getDpsInputs()[1];
    auto t = dyn_cast<RankedTensorType>(table.getType());
    if (!t || t.getRank() != 4 || t.getDimSize(0) != 1 || t.getDimSize(1) != 1 ||
        t.getDimSize(3) != 128 || !t.getElementType().isF32())
      return {};
    return table;
  }
  static bool slice(tensor::ExtractSliceOp s, Value source, int64_t offset) {
    if (!s || s.getSource() != source || !s.hasUnitStride()) return false;
    auto offsets = s.getStaticOffsets(), sizes = s.getStaticSizes();
    auto type = cast<RankedTensorType>(source.getType());
    return offsets.size() == 4 && offsets[0] == 0 && offsets[1] == 0 && offsets[2] == 0 &&
           offsets[3] == offset && sizes[0] == 1 && sizes[1] == type.getDimSize(1) &&
           sizes[3] == 64;
  }

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter rw(ctx);
    SmallVector<tensor::ConcatOp> concats;
    module.walk([&](tensor::ConcatOp c) { concats.push_back(c); });
    SmallVector<int64_t> dyn(4, ShapedType::kDynamic);
    auto any4 = MemRefType::get(dyn, rw.getF32Type(), StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dyn));
    unsigned changed = 0;
    for (auto concat : concats) {
      if (concat.getDim() != 3 || concat.getInputs().size() != 2 || !concat->hasOneUse()) continue;
      auto negate = concat.getInputs()[0].getDefiningOp<linalg::GenericOp>();
      if (!parallel(negate, 1) || !negate->hasOneUse() ||
          !llvm::all_of(negate.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
        continue;
      auto neg = dyn_cast_or_null<arith::NegFOp>(single(negate));
      if (!neg || neg.getOperand() != negate.getBody()->getArgument(0)) continue;
      auto high = negate.getDpsInputs()[0].getDefiningOp<tensor::ExtractSliceOp>();
      auto low = concat.getInputs()[1].getDefiningOp<tensor::ExtractSliceOp>();
      if (!high || !low) continue;
      Value x = high.getSource();
      auto transpose = x.getDefiningOp<linalg::TransposeOp>();
      auto xt = dyn_cast<RankedTensorType>(x.getType());
      if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({0, 2, 1, 3}) || !xt ||
          xt.getRank() != 4 || xt.getDimSize(0) != 1 || xt.isDynamicDim(1) ||
          xt.getDimSize(3) != 128 || !xt.getElementType().isF32() ||
          !slice(high, x, 64) || !slice(low, x, 0) || !high->hasOneUse() || !low->hasOneUse())
        continue;
      auto rotate = dyn_cast<linalg::GenericOp>(*concat->getUsers().begin());
      Value sinTable = broadcastMul(rotate, concat.getResult());
      if (!sinTable || !rotate->hasOneUse()) continue;
      auto add = dyn_cast<linalg::GenericOp>(*rotate->getUsers().begin());
      if (!parallel(add, 2) || !binary<arith::AddFOp>(add) ||
          !llvm::all_of(add.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
        continue;
      Value other = add.getDpsInputs()[0] == rotate->getResult(0) ? add.getDpsInputs()[1]
                                                                  : add.getDpsInputs()[0];
      if (add.getDpsInputs()[0] != rotate->getResult(0) && add.getDpsInputs()[1] != rotate->getResult(0))
        continue;
      auto scaled = other.getDefiningOp<linalg::GenericOp>();
      Value cosTable = broadcastMul(scaled, x);
      if (!cosTable || !scaled->hasOneUse()) continue;
      // X may only feed this chain.
      bool exclusive = llvm::all_of(x.getUsers(), [&](Operation *u) {
        return u == high.getOperation() || u == low.getOperation() || u == scaled.getOperation();
      });
      if (!exclusive) continue;
      Value y = transpose.getInput();
      auto yt = cast<RankedTensorType>(y.getType());
      Location loc = add.getLoc();
      rw.setInsertionPoint(add);
      SmallVector<Value> dims;
      if (xt.isDynamicDim(2)) dims.push_back(rw.create<tensor::DimOp>(loc, y, 1));
      Value out = rw.create<memref::AllocOp>(loc, MemRefType::get(xt.getShape(), xt.getElementType()), dims);
      auto buffer = [&](Value t) -> Value {
        auto tt = cast<RankedTensorType>(t.getType());
        Value b = rw.create<bufferization::ToBufferOp>(loc, MemRefType::get(tt.getShape(), tt.getElementType()), t, true);
        return rw.create<memref::CastOp>(loc, any4, b);
      };
      SmallVector<Value> operands{buffer(y), buffer(cosTable), buffer(sinTable),
                                  rw.create<memref::CastOp>(loc, any4, out)};
      StringRef symbol = "qwen_rope_heads_v1";
      if (!module.lookupSymbol(symbol)) {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToStart(module.getBody());
        SmallVector<Type> types(4, any4);
        auto fn = rw.create<func::FuncOp>(module.getLoc(), symbol, rw.getFunctionType(types, {}));
        fn.setPrivate();
        fn->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      rw.create<func::CallOp>(loc, symbol, TypeRange{}, operands);
      Value result = rw.create<bufferization::ToTensorOp>(loc, xt, out, true, true);
      rw.replaceOp(add, result);
      rw.eraseOp(scaled);
      rw.eraseOp(rotate);
      rw.eraseOp(concat);
      rw.eraseOp(negate);
      rw.eraseOp(high);
      rw.eraseOp(low);
      if (transpose->use_empty()) rw.eraseOp(transpose);
      (void)yt;
      ++changed;
    }
    if (expectedCount && changed != expectedCount) {
      module.emitError() << "fused RoPE coverage: expected " << expectedCount.getValue() << ", got " << changed;
      return signalPassFailure();
    }
    llvm::errs() << "fused parallel RoPE: " << changed << "\n";
  }
};
