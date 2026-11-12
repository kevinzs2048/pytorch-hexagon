// SPDX-License-Identifier: BSD-3-Clause
// Replace the token embedding gather
//   out[0, t, j] = table[ids[0, t], j]   (with the two bounds asserts per element)
// by one call qwen_embedding_rows_v1(ids, table, out) that checks each id once
// and copies whole rows. Values are copied unchanged.
struct EmbeddingRowsPass : PassWrapper<EmbeddingRowsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmbeddingRowsPass)
  EmbeddingRowsPass() = default;
  EmbeddingRowsPass(const EmbeddingRowsPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-embedding-rows"; }
  StringRef getDescription() const final { return "Copy embedding rows instead of a per-element gather"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  // Body: index_cast(id); linalg.index 2; cmpi slt idx, V; assert; cmpi sge id, 0; assert;
  // tensor.extract table[idx, index2]; yield.
  static Value gatherTable(linalg::GenericOp g) {
    if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || g->getNumResults() != 1 ||
        !g.hasPureTensorSemantics() ||
        !llvm::all_of(g.getIteratorTypesArray(),
                      [](utils::IteratorType t) { return t == utils::IteratorType::parallel; }))
      return {};
    auto ids = dyn_cast<RankedTensorType>(g.getDpsInputs()[0].getType());
    auto out = dyn_cast<RankedTensorType>(g.getResult(0).getType());
    if (!ids || ids.getRank() != 2 || !ids.getElementType().isInteger(64) || !out || out.getRank() != 3 ||
        !out.getElementType().isF32() || out.isDynamicDim(2) || out.getDimSize(0) != 1)
      return {};
    MLIRContext *ctx = g.getContext();
    auto e = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
    auto maps = g.getIndexingMapsArray();
    if (maps[0] != AffineMap::get(3, 0, {e(0), e(1)}, ctx) || !maps[1].isIdentity()) return {};
    Block &b = *g.getBody();
    auto yield = cast<linalg::YieldOp>(b.getTerminator());
    auto extract = yield.getOperand(0).getDefiningOp<tensor::ExtractOp>();
    if (!extract || extract->getBlock() != &b || extract.getIndices().size() != 2) return {};
    auto cast = extract.getIndices()[0].getDefiningOp<arith::IndexCastOp>();
    auto column = extract.getIndices()[1].getDefiningOp<linalg::IndexOp>();
    if (!cast || cast.getIn() != b.getArgument(0) || !column || column.getDim() != 2) return {};
    auto table = dyn_cast<RankedTensorType>(extract.getTensor().getType());
    if (!table || table.getRank() != 2 || table.getDimSize(1) != out.getDimSize(2) || !table.getElementType().isF32())
      return {};
    // The table is defined outside the body (it becomes a call operand).
    if (g->isAncestor(extract.getTensor().getParentRegion()->getParentOp())) return {};
    // Exactly the two bounds asserts: row < rows (as index) and id >= 0.
    bool upper = false, lower = false;
    for (Operation &op : b.without_terminator()) {
      if (isa<arith::IndexCastOp, linalg::IndexOp, arith::CmpIOp, tensor::ExtractOp>(op)) continue;
      auto check = dyn_cast<cf::AssertOp>(op);
      if (!check) return {};
      auto cmp = check.getArg().getDefiningOp<arith::CmpIOp>();
      APInt bound;
      if (cmp && cmp.getPredicate() == arith::CmpIPredicate::slt && cmp.getLhs() == cast.getResult() &&
          matchPattern(cmp.getRhs(), m_ConstantInt(&bound)) && bound.getSExtValue() == table.getDimSize(0))
        upper = true;
      else if (cmp && cmp.getPredicate() == arith::CmpIPredicate::sge && cmp.getLhs() == b.getArgument(0) &&
               matchPattern(cmp.getRhs(), m_Zero()))
        lower = true;
      else
        return {};
    }
    if (!upper || !lower) return {};
    return extract.getTensor();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation(); MLIRContext *ctx = module.getContext(); IRRewriter b(ctx);
    SmallVector<linalg::GenericOp> hits;
    module.walk([&](linalg::GenericOp g) { if (gatherTable(g)) hits.push_back(g); });
    int64_t d = ShapedType::kDynamic;
    auto ids2 = MemRefType::get({d, d}, b.getI64Type(), StridedLayoutAttr::get(ctx, d, {d, d}));
    auto table2 = MemRefType::get({d, d}, b.getF32Type(), StridedLayoutAttr::get(ctx, d, {d, d}));
    auto out3 = MemRefType::get({d, d, d}, b.getF32Type(), StridedLayoutAttr::get(ctx, d, {d, d, d}));
    StringRef symbol = "qwen_embedding_rows_v1";
    for (auto g : hits) {
      Value table = gatherTable(g), ids = g.getDpsInputs()[0];
      auto it = cast<RankedTensorType>(ids.getType()), tt = cast<RankedTensorType>(table.getType());
      auto ot = cast<RankedTensorType>(g.getResult(0).getType());
      Location loc = g.getLoc(); b.setInsertionPoint(g);
      if (!module.lookupSymbol(symbol)) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(module.getBody());
        auto fn = func::FuncOp::create(b, module.getLoc(), symbol, b.getFunctionType({ids2, table2, out3}, {}));
        fn.setPrivate(); fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
      }
      SmallVector<Value> dims;
      for (int64_t i = 0; i < ot.getRank(); ++i)
        if (ot.isDynamicDim(i)) dims.push_back(tensor::DimOp::create(b, loc, g.getDpsInits()[0], i));
      Value out = memref::AllocOp::create(b, loc, MemRefType::get(ot.getShape(), ot.getElementType()), dims);
      Value ib = bufferization::ToBufferOp::create(b, loc, MemRefType::get(it.getShape(), it.getElementType()), ids, true);
      Value tb = bufferization::ToBufferOp::create(b, loc, MemRefType::get(tt.getShape(), tt.getElementType()), table, true);
      func::CallOp::create(b, loc, symbol, TypeRange{}, ValueRange{memref::CastOp::create(b, loc, ids2, ib),
          memref::CastOp::create(b, loc, table2, tb), memref::CastOp::create(b, loc, out3, out)});
      Value result = bufferization::ToTensorOp::create(b, loc, ot, out, true, true);
      b.replaceOp(g, result);
    }
    if (expectedCount && hits.size() != expectedCount) {
      module.emitError() << "embedding rows: expected " << expectedCount.getValue() << ", got " << hits.size();
      return signalPassFailure();
    }
    llvm::errs() << "embedding rows: " << hits.size() << "\n";
  }
};
