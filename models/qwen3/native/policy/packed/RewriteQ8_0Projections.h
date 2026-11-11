// SPDX-License-Identifier: BSD-3-Clause
// Q8_0 projections (llama.cpp hmx-tiled numerics): replace each fused W8A8
// projection __hexagon_i8_prepacked_v1_matmul_dequant[_residual|_product] by
// __hexagon_q8_0_matmul[_residual|_product](x, id, [side,] out) on the FP32
// activation x that fed the per-token INT8 quantization. The activation
// quantization becomes dead. id is the call ordinal in module walk order; the
// weight behind each id is supplied by the linked Q8_0 weight table and is
// identified offline from the INT8 constants written with dump-dir.
//
// The LM head (the only projection wider than 3072 columns, a single row)
// becomes __hexagon_q8_0_matvec_q8(x, id, out) instead: llama.cpp computes it
// on the CPU as Q8_0 x Q8_0 (activation quantized per 32-value block), and its
// weight table entry uses the matching HVX vrmpy layout.
//
// With single-row (the decode domain: every projection one static row) the
// other projections take the same Q8_0 x Q8_0 path,
// __hexagon_q8_0_matvec_q8[_residual|_product], on the HMX-layout weights.
//
// The per-token scale S of the removed quantization is guarded by torch-mlir
// zero checks (scf.for { cf.assert S != 0 }) that keep the whole absmax chain
// alive; they only protect the removed 1 / S, so they are erased with it once
// S has no other user.
//
// Matched operand 0 chain (all tensor ops pure, intermediates single-use):
//   x[1, M, K] f32 -> mulf(x, broadcast (c / S)) -> generic(round, clamp -> i8)
//   -> collapse_shape to [M, K] -> to_buffer -> memref.cast
struct RewriteQ8_0ProjectionsPass
    : PassWrapper<RewriteQ8_0ProjectionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteQ8_0ProjectionsPass)
  RewriteQ8_0ProjectionsPass() = default;
  RewriteQ8_0ProjectionsPass(const RewriteQ8_0ProjectionsPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(0)};
  Option<std::string> dumpDir{*this, "dump-dir", llvm::cl::init("")};
  Option<bool> singleRow{*this, "single-row", llvm::cl::init(false)};
  StringRef getArgument() const final { return "qwen-q8-0-projections"; }
  StringRef getDescription() const final {
    return "Replace fused W8A8 projections by Q8_0 FP16-activation runtime calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }

  static ArrayRef<char> constantBytes(Value v) {
    if (auto c = v.getDefiningOp<memref::CastOp>()) v = c.getSource();
    auto b = v.getDefiningOp<bufferization::ToBufferOp>();
    auto c = b ? b.getTensor().getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
    if (!c) return {};
    if (auto r = dyn_cast<DenseResourceElementsAttr>(c.getValue())) return r.getData();
    if (auto d = dyn_cast<DenseElementsAttr>(c.getValue()); d && !d.isSplat()) return d.getRawData();
    return {};
  }
  // The FP32 activation x[1, M, K] behind the INT8 activation operand; scale
  // receives S[1, M, 1].
  static Value activation(Value operand, Value &scale) {
    auto castOp = operand.getDefiningOp<memref::CastOp>();
    auto buffer = castOp ? castOp.getSource().getDefiningOp<bufferization::ToBufferOp>() : nullptr;
    auto collapse = buffer ? buffer.getTensor().getDefiningOp<tensor::CollapseShapeOp>() : nullptr;
    // [[0, 1], [2]], or [[0], [1, 2]] for the single last row (LM head).
    if (!collapse) return {};
    auto groups = collapse.getReassociationIndices();
    bool rows = groups == SmallVector<ReassociationIndices>{{0, 1}, {2}};
    if (!rows && !(groups == SmallVector<ReassociationIndices>{{0}, {1, 2}} &&
                   cast<RankedTensorType>(collapse.getSrc().getType()).getDimSize(1) == 1))
      return {};
    auto quant = collapse.getSrc().getDefiningOp<linalg::GenericOp>();
    if (!quant || quant.getNumDpsInputs() != 1 || quant->getNumResults() != 1 ||
        !cast<RankedTensorType>(quant->getResult(0).getType()).getElementType().isSignlessInteger(8))
      return {};
    auto yield = cast<linalg::YieldOp>(quant.getBody()->getTerminator());
    if (!yield.getOperand(0).getDefiningOp<arith::TruncIOp>()) return {};
    auto scaled = quant.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!scaled || scaled.getNumDpsInputs() != 2 || scaled->getNumResults() != 1) return {};
    Block &body = *scaled.getBody();
    auto mul = dyn_cast<arith::MulFOp>(&body.front());
    if (body.getOperations().size() != 2 || !mul || mul.getLhs() != body.getArgument(0) ||
        mul.getRhs() != body.getArgument(1))
      return {};
    auto maps = scaled.getIndexingMapsArray();
    Value x = scaled.getDpsInputs()[0];
    auto type = dyn_cast<RankedTensorType>(x.getType());
    if (!maps[0].isIdentity() || !maps[2].isIdentity() || maps[1].isIdentity() || !type ||
        type.getRank() != 3 || type.getDimSize(0) != 1 || !type.getElementType().isF32() ||
        type.getShape() != cast<RankedTensorType>(quant->getResult(0).getType()).getShape())
      return {};
    // Broadcast factor = c / S, S[1, M, 1].
    auto inverse = scaled.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
    if (!inverse || inverse.getNumDpsInputs() != 1 || inverse->getNumResults() != 1) return {};
    Block &ib = *inverse.getBody();
    auto div = dyn_cast<arith::DivFOp>(&ib.front());
    if (ib.getOperations().size() != 2 || !div || div.getRhs() != ib.getArgument(0) ||
        !div.getLhs().getDefiningOp<arith::ConstantOp>())
      return {};
    scale = inverse.getDpsInputs()[0];
    return x;
  }

  // scf.for %i = 0 to dim(S, 1) { cf.assert extract(S[0, %i, 0]) != 0 }
  static bool zeroCheck(scf::ForOp loop, Value s) {
    Block &b = *loop.getBody();
    if (b.getOperations().size() != 4 || loop.getNumResults()) return false;
    auto it = b.begin();
    auto extract = dyn_cast<tensor::ExtractOp>(&*it++);
    auto compare = dyn_cast<arith::CmpFOp>(&*it++);
    auto check = dyn_cast<cf::AssertOp>(&*it);
    return extract && compare && check && extract.getTensor() == s &&
           compare.getPredicate() == arith::CmpFPredicate::ONE && compare.getLhs() == extract.getResult() &&
           matchPattern(compare.getRhs(), m_AnyZeroFloat()) && check.getArg() == compare.getResult() &&
           check.getMsg() == "unimplemented: tensor with zero element";
  }

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter rw(ctx);
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp c) {
      if (c.getCallee().starts_with("__hexagon_i8_prepacked_v1_matmul_dequant")) calls.push_back(c);
    });
    auto f32 = rw.getF32Type();
    auto matrix = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32,
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {ShapedType::kDynamic, ShapedType::kDynamic}));
    std::string manifest;
    llvm::raw_string_ostream m(manifest);
    unsigned id = 0, failed = 0;
    llvm::SetVector<Value> scales;
    for (auto call : calls) {
      StringRef callee = call.getCallee();
      bool residual = callee.ends_with("_residual"), product = callee.ends_with("_product");
      if (!residual && !product && callee != "__hexagon_i8_prepacked_v1_matmul_dequant") { ++failed; continue; }
      unsigned operands = residual || product ? 6 : 5;
      Value scale;
      Value x = call.getNumOperands() == operands ? activation(call.getOperand(0), scale) : Value();
      ArrayRef<char> weight = constantBytes(call.getOperand(1)), column = constantBytes(call.getOperand(3));
      auto wcast = call.getOperand(1).getDefiningOp<memref::CastOp>();
      auto wt = wcast ? dyn_cast<MemRefType>(wcast.getSource().getType()) : MemRefType();
      if (!x || weight.empty() || column.empty() || !wt || wt.getRank() != 2 ||
          cast<RankedTensorType>(x.getType()).getDimSize(2) != wt.getDimSize(0)) {
        call.emitError("unmatched Q8_0 projection");
        ++failed;
        continue;
      }
      int64_t k = wt.getDimSize(0), n = wt.getDimSize(1);
      m << (id ? ",\n" : "") << "{\"id\": " << id << ", \"n\": " << n << ", \"k\": " << k
        << ", \"mode\": \"" << (residual ? "residual" : product ? "product" : "plain") << "\"}";
      if (!dumpDir.empty()) {
        for (auto [suffix, bytes] : {std::pair<StringRef, ArrayRef<char>>{"w", weight}, {"s", column}}) {
          std::error_code ec;
          llvm::raw_fd_ostream f(dumpDir + "/" + std::to_string(id) + "." + suffix.str(), ec);
          if (ec) { module.emitError("dump-dir: ") << ec.message(); return signalPassFailure(); }
          f.write(bytes.data(), bytes.size());
        }
      }
      rw.setInsertionPoint(call);
      Location loc = call.getLoc();
      auto groups = call.getOperand(0).getDefiningOp<memref::CastOp>().getSource()
          .getDefiningOp<bufferization::ToBufferOp>().getTensor()
          .getDefiningOp<tensor::CollapseShapeOp>().getReassociationIndices();
      Value flat = rw.create<tensor::CollapseShapeOp>(loc, x, groups);
      auto ft = cast<RankedTensorType>(flat.getType());
      Value xb = rw.create<bufferization::ToBufferOp>(loc, MemRefType::get(ft.getShape(), f32), flat, true);
      SmallVector<Value> args{rw.create<memref::CastOp>(loc, matrix, xb),
          rw.create<arith::ConstantIntOp>(loc, id, 64)};
      if (residual || product) args.push_back(call.getOperand(4));
      args.push_back(call.getOperand(operands - 1));
      bool lmHead = n > 3072;
      if (lmHead && (residual || product || cast<RankedTensorType>(x.getType()).getDimSize(1) != 1)) {
        call.emitError("Q8_0 LM head must be a plain single-row projection");
        ++failed;
        continue;
      }
      if (singleRow && cast<RankedTensorType>(x.getType()).getDimSize(1) != 1) {
        call.emitError("single-row Q8_0 projection needs one static row");
        ++failed;
        continue;
      }
      std::string suffix = callee.drop_front(strlen("__hexagon_i8_prepacked_v1_matmul_dequant")).str();
      std::string symbol = lmHead ? std::string("__hexagon_q8_0_matvec_q8")
          : (singleRow ? "__hexagon_q8_0_matvec_q8" : "__hexagon_q8_0_matmul") + suffix;
      if (!module.lookupSymbol(symbol)) {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToStart(module.getBody());
        SmallVector<Type> inputs;
        for (Value v : args) inputs.push_back(v.getType());
        auto fn = rw.create<func::FuncOp>(module.getLoc(), symbol, rw.getFunctionType(inputs, {}));
        fn.setPrivate();
        fn->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      rw.create<func::CallOp>(loc, symbol, TypeRange{}, args);
      SmallVector<Value> old(call.getOperands());
      rw.eraseOp(call);
      // Drop the now unused INT8 activation, weight and scale buffers.
      for (Value v : old)
        if (auto c = v.getDefiningOp<memref::CastOp>(); c && c->use_empty()) {
          Value s = c.getSource();
          rw.eraseOp(c);
          if (auto b = s.getDefiningOp<bufferization::ToBufferOp>(); b && b->use_empty()) rw.eraseOp(b);
        }
      scales.insert(scale);
      ++id;
    }
    // Erase the dead quantization, then the zero checks that kept S alive.
    // Post-order walks: a dead op is collected (and erased) before its parent;
    // repeat until users erased in one round expose no further dead producers.
    auto dce = [&] {
      for (bool changed = true; changed;) {
        SmallVector<Operation *> dead;
        module.walk([&](Operation *op) { if (isOpTriviallyDead(op)) dead.push_back(op); });
        changed = !dead.empty();
        for (Operation *op : dead) rw.eraseOp(op);
      }
    };
    dce();
    unsigned checks = 0;
    for (Value s : scales) {
      SmallVector<scf::ForOp> loops;
      bool other = false;
      for (Operation *user : s.getUsers()) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp()); loop && zeroCheck(loop, s)) loops.push_back(loop);
        else if (auto dim = dyn_cast<tensor::DimOp>(user);
                 dim && llvm::all_of(dim->getUsers(), [&](Operation *u) {
                   auto l = dyn_cast<scf::ForOp>(u); return l && zeroCheck(l, s); }))
          ;
        else other = true;
      }
      if (other) continue;
      for (auto loop : loops) { rw.eraseOp(loop); ++checks; }
    }
    dce();
    for (StringRef s : {"__hexagon_i8_prepacked_v1_matmul_dequant", "__hexagon_i8_prepacked_v1_matmul_dequant_residual",
                        "__hexagon_i8_prepacked_v1_matmul_dequant_product"})
      if (auto fn = module.lookupSymbol<func::FuncOp>(s); fn && fn.symbolKnownUseEmpty(module)) fn.erase();
    if (!dumpDir.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream f(dumpDir + "/manifest.json", ec);
      f << "[\n" << m.str() << "\n]\n";
    }
    if (failed || (expectedCount && id != expectedCount)) {
      module.emitError() << "Q8_0 projection coverage: expected " << expectedCount.getValue() << ", got "
                         << id << " (" << failed << " unmatched)";
      return signalPassFailure();
    }
    if (checks != scales.size()) {
      module.emitError() << "Q8_0 projections: " << checks << " dead scale checks erased for "
                         << scales.size() << " activation scales";
      return signalPassFailure();
    }
    llvm::errs() << "Q8_0 projections: " << id << ", activation quantizations removed: " << scales.size() << "\n";
  }
};
