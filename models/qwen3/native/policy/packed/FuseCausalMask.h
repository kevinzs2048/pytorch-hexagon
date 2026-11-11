// SPDX-License-Identifier: BSD-3-Clause
// Replace the exported attention-mask construction
//   valid = Valid != 0; causal = iota(kv) <= Pos; m = (true & causal) & valid[0, j]
//   mask = select(broadcast(m), 0.0, -FLT_MAX) : tensor<1x1x?x?xf32>
// with one call to the row-parallel runtime qwen_causal_mask_v1. The chain is
// checked operation by operation; the dead producers are left to canonicalize.
struct FuseCausalMaskPass : PassWrapper<FuseCausalMaskPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseCausalMaskPass)
  FuseCausalMaskPass() = default;
  FuseCausalMaskPass(const FuseCausalMaskPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-fuse-causal-mask"; }
  StringRef getDescription() const final {
    return "Build the additive causal/padding attention mask with a row-parallel kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
  using G = linalg::GenericOp;
  // Single non-yield body operation of type T applied to the block arguments in order.
  template <typename T> static T bodyOp(G g) {
    Block &b = g.getRegion().front();
    if (b.getOperations().size() != 2) return nullptr;
    auto op = dyn_cast<T>(b.front());
    if (!op || cast<linalg::YieldOp>(b.back()).getOperand(0) != op->getResult(0)) return nullptr;
    return op;
  }
  static bool args(Operation *op, G g) {
    Block &b = g.getRegion().front();
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
      if (i >= b.getNumArguments() || op->getOperand(i) != b.getArgument(i)) return false;
    return true;
  }
  static bool splat(Value v, std::function<bool(Attribute)> check) {
    DenseElementsAttr a;
    return matchPattern(v, m_Constant(&a)) && a.isSplat() && check(a.getSplatValue<Attribute>());
  }
  static bool isMap(G g, unsigned i, ArrayRef<int64_t> pattern, MLIRContext *ctx) {
    // pattern entry: k >= 0 -> dim k, -1 -> constant 0; empty -> rank-0 map.
    SmallVector<AffineExpr> e;
    for (int64_t p : pattern) e.push_back(p >= 0 ? getAffineDimExpr(p, ctx) : getAffineConstantExpr(0, ctx));
    return g.getIndexingMapsArray()[i] == AffineMap::get(4, 0, e, ctx);
  }
  // iota: generic, no inputs, body index_cast(linalg.index 0) : tensor<?xi64>.
  static bool iota(Value v) {
    auto g = v.getDefiningOp<G>();
    if (!g || g.getNumDpsInputs() != 0) return false;
    Block &body = g.getRegion().front();
    if (body.getOperations().size() != 3) return false;
    auto widen = dyn_cast_or_null<arith::IndexCastOp>(
        cast<linalg::YieldOp>(body.back()).getOperand(0).getDefiningOp());
    auto index = widen ? widen.getIn().getDefiningOp<linalg::IndexOp>() : nullptr;
    return index && index.getDim() == 0 && widen.getType().isInteger(64);
  }
  static Value unexpand(Value v) {
    auto e = v.getDefiningOp<tensor::ExpandShapeOp>();
    return e ? e.getSrc() : Value();
  }

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter rw(ctx);
    SmallVector<G> candidates;
    module.walk([&](G g) { candidates.push_back(g); });
    unsigned changed = 0;
    for (G g23 : candidates) {
      auto type = dyn_cast<RankedTensorType>(g23.getResult(0).getType());
      auto sel = g23.getNumResults() == 1 && g23.getNumDpsInputs() == 3 ? bodyOp<arith::SelectOp>(g23) : nullptr;
      if (!type || !sel || !args(sel, g23) || type.getRank() != 4 || !type.getElementType().isF32() ||
          type.getDimSize(0) != 1 || type.getDimSize(1) != 1 ||
          !isMap(g23, 0, {0, 1, 2, 3}, ctx) || !isMap(g23, 1, {}, ctx) || !isMap(g23, 2, {}, ctx) ||
          !splat(g23.getDpsInputOperand(1)->get(), [](Attribute a) {
            auto f = dyn_cast<FloatAttr>(a); return f && f.getValue().isPosZero(); }) ||
          !splat(g23.getDpsInputOperand(2)->get(), [](Attribute a) {
            auto f = dyn_cast<FloatAttr>(a);
            return f && f.getValue().bitwiseIsEqual(APFloat::getLargest(APFloat::IEEEsingle(), true)); }))
        continue;
      // %21: broadcast gather of %16.
      auto g21 = g23.getDpsInputOperand(0)->get().getDefiningOp<G>();
      auto gather = g21 && g21.getNumDpsInputs() == 0 ? bodyOp<tensor::ExtractOp>(g21) : nullptr;
      if (!gather) {
        // The body also holds the broadcast index selects; find the single extract.
        if (!g21 || g21.getNumDpsInputs() != 0) continue;
        unsigned extracts = 0;
        g21.getRegion().walk([&](tensor::ExtractOp e) { gather = e; ++extracts; });
        if (extracts != 1 || cast<linalg::YieldOp>(g21.getRegion().front().back()).getOperand(0) != gather.getResult())
          continue;
      }
      auto g16 = gather.getTensor().getDefiningOp<G>();
      auto and16 = g16 && g16.getNumDpsInputs() == 2 ? bodyOp<arith::AndIOp>(g16) : nullptr;
      if (!and16 || !args(and16, g16) || !isMap(g16, 0, {0, 1, 2, 3}, ctx) || !isMap(g16, 1, {0, 1, -1, 3}, ctx))
        continue;
      auto g12 = g16.getDpsInputOperand(0)->get().getDefiningOp<G>();
      auto and12 = g12 && g12.getNumDpsInputs() == 2 ? bodyOp<arith::AndIOp>(g12) : nullptr;
      if (!and12 || !args(and12, g12) || !isMap(g12, 0, {}, ctx) || !isMap(g12, 1, {0, 1, 2, 3}, ctx) ||
          !splat(g12.getDpsInputOperand(0)->get(), [](Attribute a) {
            auto b = dyn_cast<BoolAttr>(a); if (b) return b.getValue();
            auto i = dyn_cast<IntegerAttr>(a); return i && i.getValue().isOne(); }))
        continue;
      auto g11 = g12.getDpsInputOperand(1)->get().getDefiningOp<G>();
      auto cmp11 = g11 && g11.getNumDpsInputs() == 2 ? bodyOp<arith::CmpIOp>(g11) : nullptr;
      if (!cmp11 || !args(cmp11, g11) || cmp11.getPredicate() != arith::CmpIPredicate::sle ||
          !isMap(g11, 0, {0, 1, -1, 3}, ctx) || !isMap(g11, 1, {0, 1, 2, -1}, ctx))
        continue;
      Value iotaExpanded = g11.getDpsInputOperand(0)->get();
      Value pos = unexpand(g11.getDpsInputOperand(1)->get());
      auto posType = pos ? dyn_cast<RankedTensorType>(pos.getType()) : nullptr;
      if (!iota(unexpand(iotaExpanded)) || !posType || posType.getRank() != 1 || !posType.getElementType().isInteger(64))
        continue;
      // %14: valid[0, j] gathered from %3 = Valid != 0.
      auto g14 = g16.getDpsInputOperand(1)->get().getDefiningOp<G>();
      if (!g14 || g14.getNumDpsInputs() != 2 || g14.getDpsInputOperand(1)->get() != iotaExpanded ||
          !isMap(g14, 0, {-1, -1, -1, -1}, ctx) || !isMap(g14, 1, {-1, -1, -1, 3}, ctx))
        continue;
      tensor::ExtractOp pick;
      unsigned picks = 0;
      g14.getRegion().walk([&](tensor::ExtractOp e) { pick = e; ++picks; });
      if (picks != 1) continue;
      // Row index of the padding gather must come from an all-zero i64 tensor.
      auto zero = unexpand(g14.getDpsInputOperand(0)->get());
      auto zeroGen = zero ? zero.getDefiningOp<G>() : nullptr;
      if (!zeroGen || zeroGen.getNumDpsInputs() != 0) continue;
      auto zeroYield = cast<linalg::YieldOp>(zeroGen.getRegion().front().back());
      if (!matchPattern(zeroYield.getOperand(0), m_Zero())) continue;
      auto g3 = pick.getTensor().getDefiningOp<G>();
      auto cmp3 = g3 && g3.getNumDpsInputs() == 1 ? bodyOp<arith::CmpIOp>(g3) : nullptr;
      if (!cmp3 || cmp3.getPredicate() != arith::CmpIPredicate::ne ||
          cmp3.getLhs() != g3.getRegion().front().getArgument(0) || !matchPattern(cmp3.getRhs(), m_Zero()))
        continue;
      Value valid = g3.getDpsInputOperand(0)->get();
      auto validType = dyn_cast<RankedTensorType>(valid.getType());
      if (!validType || validType.getRank() != 2 || !validType.getElementType().isInteger(64)) continue;

      Location loc = g23.getLoc();
      rw.setInsertionPoint(g23);
      SmallVector<Value> dims;
      for (int64_t i = 0; i < 4; ++i)
        if (type.isDynamicDim(i)) dims.push_back(rw.create<tensor::DimOp>(loc, g23.getDpsInitOperand(0)->get(), i));
      Value out = rw.create<memref::AllocOp>(loc, MemRefType::get(type.getShape(), type.getElementType()), dims);
      auto anyType = [&](Type element, int64_t rank) {
        SmallVector<int64_t> dyn(rank, ShapedType::kDynamic);
        return MemRefType::get(dyn, element, StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dyn));
      };
      auto buffer = [&](Value t, MemRefType to) -> Value {
        auto tt = cast<RankedTensorType>(t.getType());
        Value b = rw.create<bufferization::ToBufferOp>(loc, MemRefType::get(tt.getShape(), tt.getElementType()), t, true);
        return rw.create<memref::CastOp>(loc, to, b);
      };
      auto p1 = anyType(rw.getI64Type(), 1), v2 = anyType(rw.getI64Type(), 2), o4 = anyType(rw.getF32Type(), 4);
      SmallVector<Value> operands{buffer(pos, p1), buffer(valid, v2), rw.create<memref::CastOp>(loc, o4, out)};
      StringRef symbol = "qwen_causal_mask_v1";
      if (!module.lookupSymbol(symbol)) {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToStart(module.getBody());
        auto fn = rw.create<func::FuncOp>(module.getLoc(), symbol, rw.getFunctionType({p1, v2, o4}, {}));
        fn.setPrivate();
        fn->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      rw.create<func::CallOp>(loc, symbol, TypeRange{}, operands);
      Value result = rw.create<bufferization::ToTensorOp>(loc, type, out, true, true);
      rw.replaceOp(g23, result);
      ++changed;
    }
    if (expectedCount && changed != expectedCount) {
      module.emitError() << "fused causal mask coverage: expected " << expectedCount.getValue() << ", got " << changed;
      return signalPassFailure();
    }
    llvm::errs() << "fused causal mask: " << changed << "\n";
  }
};
