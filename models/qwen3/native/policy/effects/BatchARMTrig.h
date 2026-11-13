// SPDX-License-Identifier: BSD-3-Clause
// Reuse all four results of the existing SLEEF range-reduction group.
struct BatchARMTrigPass
    : PassWrapper<BatchARMTrigPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchARMTrigPass)
  BatchARMTrigPass() = default;
  BatchARMTrigPass(const BatchARMTrigPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(2)};
  StringRef getArgument() const final { return "qwen-batch-arm-rope"; }
  StringRef getDescription() const final {
    return "Reuse each original four-lane ARM RoPE trigonometric result";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }
  static func::CallOp match(linalg::GenericOp op) {
    if (!op->getAttrOfType<UnitAttr>("hexagon.fp32_arm_rope_trig4") ||
        !op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
        op.getNumDpsInputs() != 4 || op.getNumDpsInits() != 1 ||
        op.getNumResults() != 1 || op.getNumLoops() != 3 ||
        op.getNumReductionLoops())
      return {};
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || type.getRank() != 3 || type.getEncoding() ||
        !type.getElementType().isF32() || type.getDimSize(0) != 1 ||
        type.getDimSize(2) != 128 ||
        (!type.isDynamicDim(1) &&
         (type.getDimSize(1) < 1 || type.getDimSize(1) > 32)) ||
        op.getDpsInits()[0].getType() != type)
      return {};
    auto maps = op.getIndexingMapsArray();
    auto *ctx = op.getContext();
    auto d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx),
         d2 = getAffineDimExpr(2, ctx);
    for (unsigned lane = 0; lane < 4; ++lane)
      if (op.getDpsInputs()[lane] != op.getDpsInputs()[0] ||
          op.getDpsInputs()[lane].getType() != type ||
          maps[lane] !=
              AffineMap::get(3, 0, {d0, d1, d2.floorDiv(4) * 4 + lane}, ctx))
        return {};
    if (!maps.back().isIdentity())
      return {};
    auto yield = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    auto call = yield.getOperand(0).getDefiningOp<func::CallOp>();
    if (!call || call.getNumOperands() != 5 || call.getNumResults() != 1 ||
        !call.getResult(0).getType().isF32() ||
        (call.getCallee() != "__hexagon_arm_sin4_lane" &&
         call.getCallee() != "__hexagon_arm_cos4_lane"))
      return {};
    for (unsigned lane = 0; lane < 4; ++lane)
      if (call.getOperand(lane) != op.getBody()->getArgument(lane))
        return {};
    auto cast = call.getOperand(4).getDefiningOp<arith::IndexCastOp>();
    auto rem =
        cast ? cast.getIn().getDefiningOp<arith::RemUIOp>() : arith::RemUIOp();
    auto index =
        rem ? rem.getLhs().getDefiningOp<linalg::IndexOp>() : linalg::IndexOp();
    auto four = rem ? rem.getRhs().getDefiningOp<arith::ConstantIndexOp>()
                    : arith::ConstantIndexOp();
    if (!cast || !cast.getType().isInteger(32) || !index ||
        index.getDim() != 2 || !four || four.value() != 4 ||
        op.getBody()->getOperations().size() !=
            5u + unsigned(four->getBlock() == op.getBody()))
      return {};
    return call;
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::GenericOp> work;
    module.walk([&](linalg::GenericOp op) {
      if (op->hasAttr("hexagon.fp32_arm_rope_trig4"))
        work.push_back(op);
    });
    if (!expectedCount || work.size() != expectedCount) {
      module.emitError("batched ARM RoPE coverage differs");
      return signalPassFailure();
    }
    // Validate the entire input before creating declarations or changing IR.
    for (auto op : work) {
      auto call = match(op);
      auto helper = call ? module.lookupSymbol<func::FuncOp>(call.getCallee())
                         : func::FuncOp();
      if (!call || !helper || !helper.isPrivate() || !helper.isExternal() ||
          helper.getFunctionType() !=
              FunctionType::get(&getContext(), call.getOperandTypes(),
                                call.getResultTypes())) {
        op.emitError("unsupported original ARM RoPE four-lane expression");
        return signalPassFailure();
      }
    }
    for (StringRef name : {"__hexagon_arm_sin4", "__hexagon_arm_cos4"})
      if (module.lookupSymbol(name)) {
        module.emitError("batched ARM RoPE runtime symbol already exists");
        return signalPassFailure();
      }
    IRRewriter rw(&getContext());
    auto resultBuffer = MemRefType::get({4}, rw.getF32Type());
    for (auto op : work) {
      auto original = match(op);
      StringRef name = original.getCallee() == "__hexagon_arm_sin4_lane"
                           ? "__hexagon_arm_sin4"
                           : "__hexagon_arm_cos4";
      auto helper = module.lookupSymbol<func::FuncOp>(name);
      if (!helper) {
        rw.setInsertionPointToStart(module.getBody());
        helper = func::FuncOp::create(
            rw, op.getLoc(), name,
            rw.getFunctionType({resultBuffer, rw.getF32Type(), rw.getF32Type(),
                                rw.getF32Type(), rw.getF32Type()},
                               {}));
        helper.setPrivate();
        helper->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      rw.setInsertionPoint(op);
      auto loc = op.getLoc();
      Value x = op.getDpsInputs()[0];
      // RoPE duplicates the exact same 64-element frequency vector. Four-lane
      // helper grouping repeats too; reuse its results only with this SSA proof.
      auto concat = x.getDefiningOp<tensor::ConcatOp>();
      bool repeatedHalf = concat && concat.getDim() == 2 &&
          concat.getInputs().size() == 2 &&
          concat.getInputs()[0] == concat.getInputs()[1] &&
          cast<RankedTensorType>(concat.getInputs()[0].getType()).getDimSize(2) == 64;
      Value scratch = memref::AllocaOp::create(rw, loc, resultBuffer);
      Value zero = arith::ConstantIndexOp::create(rw, loc, 0);
      Value one = arith::ConstantIndexOp::create(rw, loc, 1);
      Value four = arith::ConstantIndexOp::create(rw, loc, 4);
      Value width = arith::ConstantIndexOp::create(rw, loc, repeatedHalf ? 64 : 128);
      Value rows = tensor::DimOp::create(rw, loc, x, 1);
      auto row = scf::ForOp::create(rw, loc, zero, rows, one, op.getDpsInits());
      rw.setInsertionPointToStart(row.getBody());
      auto col = scf::ForOp::create(rw, loc, zero, width, four,
                                    row.getRegionIterArgs());
      rw.setInsertionPointToStart(col.getBody());
      SmallVector<Value> args{scratch}, indices;
      for (int lane = 0; lane < 4; ++lane) {
        Value i = arith::ConstantIndexOp::create(rw, loc, lane);
        indices.push_back(
            arith::AddIOp::create(rw, loc, col.getInductionVar(), i));
        args.push_back(tensor::ExtractOp::create(
            rw, loc, x,
            ValueRange{zero, row.getInductionVar(), indices.back()}));
      }
      func::CallOp::create(rw, loc, helper, args);
      Value output = col.getRegionIterArgs()[0];
      for (int lane = 0; lane < 4; ++lane) {
        Value i = arith::ConstantIndexOp::create(rw, loc, lane);
        Value value = memref::LoadOp::create(rw, loc, scratch, ValueRange{i});
        output = tensor::InsertOp::create(
            rw, loc, value, output,
            ValueRange{zero, row.getInductionVar(), indices[lane]});
        if (repeatedHalf) {
          Value half = arith::ConstantIndexOp::create(rw, loc, 64);
          Value duplicate = arith::AddIOp::create(rw, loc, indices[lane], half);
          output = tensor::InsertOp::create(
              rw, loc, value, output,
              ValueRange{zero, row.getInductionVar(), duplicate});
        }
      }
      scf::YieldOp::create(rw, loc, output);
      rw.setInsertionPointAfter(col);
      scf::YieldOp::create(rw, loc, col.getResults());
      row->setAttr("hexagon.fp32_arm_rope_trig4_batch", rw.getUnitAttr());
      rw.replaceOp(op, row.getResults());
    }
    llvm::errs() << "Batched ARM RoPE expressions: " << work.size() << "\n";
  }
};
