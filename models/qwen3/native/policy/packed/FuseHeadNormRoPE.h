// SPDX-License-Identifier: BSD-3-Clause
// Fold the per-head RMSNorm (Qwen3 q_norm / k_norm) that feeds a fused RoPE call
// into the kernel:
//   qwen_rope_heads_v1(W * (X * rsqrt(S / 128 + eps)), C, S, O)
//   -> qwen_rope_heads_norm_v1(X, W, C, S, O, eps)
// S must be the ARM-ordered sum-of-squares loop over X that the earlier policies
// mark (hexagon.fp32_arm_rmsnorm_sum / hexagon.fp32_arm_k_norm_sum); the kernel
// evaluates the same order. Run after --qwen-fuse-parallel-rope. The now-dead
// norm loops are removed by the compile pipeline.
struct FuseHeadNormRoPEPass : PassWrapper<FuseHeadNormRoPEPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseHeadNormRoPEPass)
  FuseHeadNormRoPEPass() = default;
  FuseHeadNormRoPEPass(const FuseHeadNormRoPEPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-fuse-head-norm-rope"; }
  StringRef getDescription() const final { return "Fold per-head RMSNorm into the fused RoPE call"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  // The body is one op whose result is yielded; all loops are parallel.
  static Operation *single(linalg::GenericOp g) {
    if (!g || !g.hasPureTensorSemantics() || g.getNumDpsInits() != 1 || g->getNumResults() != 1 ||
        !llvm::all_of(g.getIteratorTypesArray(),
                      [](utils::IteratorType t) { return t == utils::IteratorType::parallel; }))
      return nullptr;
    Block &b = *g.getBody();
    if (b.getOperations().size() != 2) return nullptr;
    Operation *op = &b.front();
    auto y = cast<linalg::YieldOp>(b.getTerminator());
    if (op->getNumResults() != 1 || y.getNumOperands() != 1 || y.getOperand(0) != op->getResult(0)) return nullptr;
    if (auto fm = dyn_cast<arith::ArithFastMathInterface>(op))
      if (fm.getFastMathFlagsAttr() && fm.getFastMathFlagsAttr().getValue() != arith::FastMathFlags::none)
        return nullptr;
    return op;
  }
  // Op of the given kind over the input block arguments, in order (unary: arg 0).
  template <typename Op> static bool unaryOrBinary(linalg::GenericOp g, unsigned inputs) {
    if (!g || g.getNumDpsInputs() != inputs) return false;
    auto op = dyn_cast_or_null<Op>(single(g));
    if (!op) return false;
    Block &b = *g.getBody();
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
      if (i < inputs && op->getOperand(i) != b.getArgument(i)) return false;
    return true;
  }
  static bool maps(linalg::GenericOp g, ArrayRef<AffineMap> expected) {
    return g.getIndexingMapsArray() == SmallVector<AffineMap>(expected.begin(), expected.end());
  }
  static bool splat(Value v, double expect, float *out) {
    DenseElementsAttr a;
    if (!matchPattern(v, m_Constant(&a)) || !a.isSplat() || !a.getElementType().isF32()) return false;
    float f = a.getSplatValue<APFloat>().convertToFloat();
    if (out) *out = f; else if (f != float(expect)) return false;
    return true;
  }
  void runOnOperation() final {
    ModuleOp module = getOperation(); MLIRContext *ctx = module.getContext(); IRRewriter b(ctx);
    SmallVector<int64_t> dyn(4, ShapedType::kDynamic);
    auto any4 = MemRefType::get(dyn, b.getF32Type(), StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dyn));
    auto any1 = MemRefType::get({ShapedType::kDynamic}, b.getF32Type(),
                                StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {ShapedType::kDynamic}));
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp c) { if (c.getCallee() == "qwen_rope_heads_v1") calls.push_back(c); });
    unsigned changed = 0;
    for (auto call : calls) {
      auto cast = call.getOperand(0).getDefiningOp<memref::CastOp>();
      auto tb = cast ? cast.getSource().getDefiningOp<bufferization::ToBufferOp>() : nullptr;
      auto scaled = tb ? tb.getTensor().getDefiningOp<linalg::GenericOp>() : nullptr;
      if (!unaryOrBinary<arith::MulFOp>(scaled, 2)) continue;
      Value weight = scaled.getDpsInputs()[0];
      auto wt = dyn_cast<RankedTensorType>(weight.getType());
      auto normed = scaled.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
      if (!wt || wt.getRank() != 1 || wt.getDimSize(0) != 128 || !unaryOrBinary<arith::MulFOp>(normed, 2)) continue;
      Value x = normed.getDpsInputs()[0];
      auto xt = dyn_cast<RankedTensorType>(x.getType());
      auto rs = normed.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
      if (!xt || xt.getRank() != 4 || xt.getDimSize(3) != 128 || !unaryOrBinary<math::RsqrtOp>(rs, 1)) continue;
      auto plus = rs.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
      float eps;
      if (!unaryOrBinary<arith::AddFOp>(plus, 2) || !splat(plus.getDpsInputs()[1], 0, &eps)) continue;
      auto mean = plus.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
      if (!unaryOrBinary<arith::DivFOp>(mean, 1)) continue;
      auto div = cast_or_null<arith::DivFOp>(single(mean));
      FloatAttr k;
      if (!matchPattern(div.getRhs(), m_Constant(&k)) || k.getValueAsDouble() != 128.0) continue;
      auto loop = mean.getDpsInputs()[0].getDefiningOp<scf::ForOp>();
      if (!loop || loop->getNumResults() != 1 || loop.getInitArgs().size() != 1) continue;
      // Indexing maps: W broadcast over (d3), the rsqrt factor over (d0, d1, d2, 0),
      // eps scalar, everything else identity; the loop starts from zero.
      {
        auto e = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
        AffineMap id = AffineMap::getMultiDimIdentityMap(4, ctx);
        AffineMap lane = AffineMap::get(4, 0, {e(3)}, ctx);
        AffineMap head = AffineMap::get(4, 0, {e(0), e(1), e(2), getAffineConstantExpr(0, ctx)}, ctx);
        AffineMap scalar = AffineMap::get(4, 0, {}, ctx);
        auto fill = loop.getInitArgs()[0].getDefiningOp<linalg::FillOp>();
        FloatAttr zero;
        if (!maps(scaled, {lane, id, id}) || !maps(normed, {id, head, id}) || !maps(rs, {id, id}) ||
            !maps(plus, {id, scalar, id}) || !maps(mean, {id, id}) || !fill ||
            !matchPattern(fill.getInputs()[0], m_Constant(&zero)) || !zero.getValue().isPosZero())
          continue;
      }
      // The normalized value must feed only this call; tensor.dim uses (the RoPE
      // fusion reads the row count from it) move to X, which has the same type.
      if (x.getType() != scaled.getResult(0).getType() || !tb->hasOneUse() || !cast->hasOneUse() ||
          !llvm::all_of(scaled->getUsers(), [&](Operation *u) {
            return u == tb.getOperation() || isa<tensor::DimOp>(u);
          }))
        continue;
      bool marker = false, reads = false;
      loop.walk([&](Operation *op) {
        if (op->hasAttr("hexagon.fp32_arm_rmsnorm_sum") || op->hasAttr("hexagon.fp32_arm_k_norm_sum")) marker = true;
        if (auto s = dyn_cast<tensor::ExtractSliceOp>(op)) if (s.getSource() == x) reads = true;
      });
      if (!marker || !reads) continue;
      Location loc = call.getLoc(); b.setInsertionPoint(call);
      Value xb = bufferization::ToBufferOp::create(b, loc, MemRefType::get(xt.getShape(), b.getF32Type()), x, true);
      Value wb = bufferization::ToBufferOp::create(b, loc, MemRefType::get({128}, b.getF32Type()), weight, true);
      Value e = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(eps));
      SmallVector<Value> ops{memref::CastOp::create(b, loc, any4, xb), memref::CastOp::create(b, loc, any1, wb),
                             call.getOperand(1), call.getOperand(2), call.getOperand(3), e};
      StringRef symbol = "qwen_rope_heads_norm_v1";
      auto type = b.getFunctionType({any4, any1, any4, any4, any4, b.getF32Type()}, {});
      if (auto existing = module.lookupSymbol<func::FuncOp>(symbol)) {
        if (existing.getFunctionType() != type) {
          existing.emitError("head norm RoPE: inconsistent qwen_rope_heads_norm_v1 declaration");
          return signalPassFailure();
        }
      } else {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(module.getBody());
        auto fn = func::FuncOp::create(b, module.getLoc(), symbol, type);
        fn.setPrivate(); fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
      }
      func::CallOp::create(b, loc, symbol, TypeRange{}, ops);
      call.erase();
      for (Operation *u : llvm::make_early_inc_range(scaled->getUsers()))
        if (auto dim = dyn_cast<tensor::DimOp>(u)) dim.getSourceMutable().assign(x);
      if (cast->use_empty()) cast.erase();
      if (tb->use_empty()) tb.erase();
      ++changed;
    }
    if (expectedCount && changed != expectedCount) {
      module.emitError() << "head norm RoPE: expected " << expectedCount.getValue() << ", got " << changed;
      return signalPassFailure();
    }
    llvm::errs() << "fused head norm RoPE: " << changed << "\n";
  }
};
