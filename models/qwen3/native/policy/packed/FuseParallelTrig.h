// SPDX-License-Identifier: BSD-3-Clause
// Replace the batched ARM RoPE trig loop (hexagon.fp32_arm_rope_trig4_batch):
//   for r: for j in [0, 64) step 4:
//     f4(t, S[0,r,j], S[0,r,j+1], S[0,r,j+2], S[0,r,j+3]);
//     out[0,r,j+k] = out[0,r,64+j+k] = t[k]
// with one call to the row-parallel runtime qwen_arm_trig_rows_v1, which calls
// the same four-lane entry with the same groups (bitwise identical).
struct FuseParallelTrigPass
    : PassWrapper<FuseParallelTrigPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseParallelTrigPass)
  FuseParallelTrigPass() = default;
  FuseParallelTrigPass(const FuseParallelTrigPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-fuse-parallel-trig"; }
  StringRef getDescription() const final {
    return "Run the batched ARM RoPE trig loop through a row-parallel runtime call";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
  static bool constantIndex(Value v, int64_t value) {
    APInt a;
    return matchPattern(v, m_ConstantInt(&a)) && a.getSExtValue() == value;
  }
  // value == base + offset (offset 0 means base itself).
  static bool plus(Value value, Value base, int64_t offset) {
    if (offset == 0 && value == base) return true;
    auto add = value.getDefiningOp<arith::AddIOp>();
    return add && ((add.getLhs() == base && constantIndex(add.getRhs(), offset)) ||
                   (add.getRhs() == base && constantIndex(add.getLhs(), offset)));
  }

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter rw(ctx);
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp f) {
      if (f->hasAttr("hexagon.fp32_arm_rope_trig4_batch")) loops.push_back(f);
    });
    unsigned changed = 0;
    for (auto outer : loops) {
      if (outer.getNumResults() != 1 || !constantIndex(outer.getLowerBound(), 0) ||
          !constantIndex(outer.getStep(), 1) ||
          !outer.getInitArgs()[0].getDefiningOp<tensor::EmptyOp>())
        continue;
      auto type = dyn_cast<RankedTensorType>(outer.getResult(0).getType());
      if (!type || type.getRank() != 3 || type.getDimSize(0) != 1 || type.getDimSize(2) != 128 ||
          !type.getElementType().isF32())
        continue;
      auto inners = outer.getBody()->getOps<scf::ForOp>();
      if (std::distance(inners.begin(), inners.end()) != 1) continue;
      scf::ForOp inner = *inners.begin();
      if (!constantIndex(inner.getLowerBound(), 0) || !constantIndex(inner.getUpperBound(), 64) ||
          !constantIndex(inner.getStep(), 4) || inner.getNumResults() != 1)
        continue;
      auto calls = inner.getBody()->getOps<func::CallOp>();
      if (std::distance(calls.begin(), calls.end()) != 1) continue;
      func::CallOp call = *calls.begin();
      int which = call.getCallee() == "__hexagon_arm_cos4" ? 0 : call.getCallee() == "__hexagon_arm_sin4" ? 1 : -1;
      if (which < 0 || call.getNumOperands() != 5) continue;
      Value source;
      bool ok = true;
      for (int k = 0; k < 4; ++k) {
        auto ex = call.getOperand(k + 1).getDefiningOp<tensor::ExtractOp>();
        if (!ex || ex.getIndices().size() != 3 || !constantIndex(ex.getIndices()[0], 0) ||
            ex.getIndices()[1] != outer.getInductionVar() ||
            !plus(ex.getIndices()[2], inner.getInductionVar(), k)) { ok = false; break; }
        if (!source) source = ex.getTensor();
        else if (source != ex.getTensor()) { ok = false; break; }
      }
      // The table source must be defined outside the loop nest (not an iter_arg or loop value).
      if (!ok || !source || outer->isAncestor(source.getParentBlock()->getParentOp()))
        continue;
      if (source.getType() != type) continue;
      // Every write of the inner body must be the duplicated insert pattern.
      unsigned inserts = 0;
      for (Operation &op : inner.getBody()->without_terminator())
        if (isa<tensor::InsertOp>(op)) ++inserts;
      if (inserts != 8) continue;
      rw.setInsertionPoint(outer);
      Location loc = outer.getLoc();
      SmallVector<Value> dims;
      if (type.isDynamicDim(1)) dims.push_back(rw.create<tensor::DimOp>(loc, source, 1));
      Value out = rw.create<memref::AllocOp>(loc, MemRefType::get(type.getShape(), type.getElementType()), dims);
      SmallVector<int64_t> dyn(3, ShapedType::kDynamic);
      auto any3 = MemRefType::get(dyn, rw.getF32Type(), StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dyn));
      Value in = rw.create<bufferization::ToBufferOp>(loc, MemRefType::get(type.getShape(), type.getElementType()), source, true);
      SmallVector<Value> operands{rw.create<memref::CastOp>(loc, any3, in), rw.create<memref::CastOp>(loc, any3, out),
                                  rw.create<arith::ConstantIntOp>(loc, which, 32)};
      StringRef symbol = "qwen_arm_trig_rows_v1";
      if (!module.lookupSymbol(symbol)) {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToStart(module.getBody());
        auto fn = rw.create<func::FuncOp>(module.getLoc(), symbol,
            rw.getFunctionType({any3, any3, rw.getI32Type()}, {}));
        fn.setPrivate();
        fn->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      rw.create<func::CallOp>(loc, symbol, TypeRange{}, operands);
      Value result = rw.create<bufferization::ToTensorOp>(loc, type, out, true, true);
      rw.replaceOp(outer, result);
      ++changed;
    }
    if (expectedCount && changed != expectedCount) {
      module.emitError() << "fused trig coverage: expected " << expectedCount.getValue() << ", got " << changed;
      return signalPassFailure();
    }
    llvm::errs() << "fused parallel trig: " << changed << "\n";
  }
};
