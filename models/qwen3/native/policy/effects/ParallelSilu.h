// SPDX-License-Identifier: BSD-3-Clause
// Admit the original marked SiLU expression, preserving negate/exp/add/divide.
struct ParallelSiluPass
    : PassWrapper<ParallelSiluPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelSiluPass)
  ParallelSiluPass() = default;
  ParallelSiluPass(const ParallelSiluPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(28)};
  StringRef getArgument() const final { return "qwen-parallel-silu"; }
  StringRef getDescription() const final {
    return "Run marked strict SiLU on independent workers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    cf::ControlFlowDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }
  static bool elementwise(linalg::GenericOp op, RankedTensorType type,
                          unsigned inputs) {
    return op && op.hasPureTensorSemantics() && !op->hasAttr("library_call") &&
           op.getNumDpsInputs() == inputs && op.getNumDpsInits() == 1 &&
           op.getNumResults() == 1 && op.getResult(0).getType() == type &&
           op.getDpsInits()[0].getType() == type &&
           llvm::all_of(op.getDpsInputs(),
                        [&](Value v) { return v.getType() == type; }) &&
           llvm::all_of(op.getIndexingMapsArray(),
                        [](AffineMap m) { return m.isIdentity(); }) &&
           llvm::all_of(op.getIteratorTypesArray(), [](auto i) {
             return i == utils::IteratorType::parallel;
           });
  }
  void runOnOperation() final {
    auto expHelper = getOperation().lookupSymbol<func::FuncOp>(
        "__hexagon_silu_exp_compat");
    auto scalarType = FunctionType::get(&getContext(),
        {Float32Type::get(&getContext())}, {Float32Type::get(&getContext())});
    if (!expHelper || !expHelper.isPrivate() || !expHelper.isExternal() ||
        expHelper.getFunctionType() != scalarType) {
      getOperation().emitError("original private external SiLU helper required");
      return signalPassFailure();
    }
    SmallVector<linalg::GenericOp> work;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op->hasAttr("hexagon.fp32_arm_silu_division"))
        work.push_back(op);
    });
    if (!expectedCount || work.size() != expectedCount) {
      getOperation().emitError("parallel SiLU coverage differs");
      return signalPassFailure();
    }
    if (getOperation().lookupSymbol("qwen_ordered_silu_v1")) {
      getOperation().emitError("SiLU runtime symbol already exists");
      return signalPassFailure();
    }
    IRRewriter rw(&getContext());
    for (auto op : work) {
      auto owner = op->getParentOfType<func::FuncOp>();
      bool single = owner && (owner.getSymName() == "Qwen3DynamicSingle0" ||
                              owner.getSymName() == "Qwen3DynamicCache1");
      if (!owner || (owner.getSymName() != "Qwen3DynamicCache0" &&
                     owner.getSymName() != "Qwen3DynamicPrefill" && !single)) {
        op.emitError("parallel SiLU requires an admitted resident entry");
        return signalPassFailure();
      }
      auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
      if (!type || type.getRank() != 3 || type.getEncoding() ||
          !type.getElementType().isF32() || type.getDimSize(0) != 1 ||
          type.getDimSize(2) != 3072 ||
          (single && type.getDimSize(1) != 1) ||
          (!single && !type.isDynamicDim(1) &&
           (type.getDimSize(1) < 2 || type.getDimSize(1) > 32)) ||
          !elementwise(op, type, 2) ||
          op.getBody()->getOperations().size() != 2) {
        op.emitError("unsupported marked SiLU shape/body");
        return signalPassFailure();
      }
      auto denom = op.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
      auto div = dyn_cast<arith::DivFOp>(op.getBody()->front());
      if (!elementwise(denom, type, 1) || !denom.getResult(0).hasOneUse() ||
          !denom->hasAttr("hexagon.fp32_arm_silu_denominator") ||
          denom.getDpsInputs()[0] != op.getDpsInputs()[1] ||
          denom.getBody()->getOperations().size() != 4 || !div ||
          div.getFastmath() != arith::FastMathFlags::none ||
          div.getLhs() != op.getBody()->getArgument(1) ||
          div.getRhs() != op.getBody()->getArgument(0) ||
          cast<linalg::YieldOp>(op.getBody()->getTerminator()).getOperand(0) !=
              div.getResult()) {
        op.emitError("unsupported strict SiLU division");
        return signalPassFailure();
      }
      auto it = denom.getBody()->begin();
      auto neg = dyn_cast<arith::NegFOp>(*it++);
      auto exp = dyn_cast<func::CallOp>(*it++);
      auto add = dyn_cast<arith::AddFOp>(*it++);
      auto yield = dyn_cast<linalg::YieldOp>(*it);
      auto constant = add ? add.getRhs().getDefiningOp<arith::ConstantOp>()
                          : arith::ConstantOp();
      auto oneAttr =
          constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
      if (!neg || neg.getFastmath() != arith::FastMathFlags::none ||
          neg.getOperand() != denom.getBody()->getArgument(0) || !exp ||
          exp.getCallee() != "__hexagon_silu_exp_compat" ||
          exp.getNumOperands() != 1 || exp.getNumResults() != 1 ||
          exp.getOperand(0) != neg.getResult() ||
          !exp->hasAttr("hexagon.fp32_silu_exp_compat") || !add ||
          add.getFastmath() != arith::FastMathFlags::none ||
          add.getLhs() != exp.getResult(0) || !oneAttr ||
          !oneAttr.getValue().isExactlyValue(1.0) || !yield ||
          yield.getOperand(0) != add.getResult()) {
        op.emitError("unsupported strict SiLU denominator");
        return signalPassFailure();
      }
      rw.setInsertionPoint(op);
      Location loc = op.getLoc();
      Value input = denom.getDpsInputs()[0];
      Value rows = rw.create<tensor::DimOp>(loc, input, 1);
      auto memref = MemRefType::get(type.getShape(), type.getElementType());
      Value source =
          rw.create<bufferization::ToBufferOp>(loc, memref, input, true);
      SmallVector<Value> dynamic;
      if (type.isDynamicDim(1))
        dynamic.push_back(rows);
      Value output = rw.create<memref::AllocOp>(loc, memref, dynamic,
                                                rw.getI64IntegerAttr(128));
      auto layout =
          StridedLayoutAttr::get(&getContext(), ShapedType::kDynamic,
                                 SmallVector<int64_t>(3, ShapedType::kDynamic));
      auto inputType =
          MemRefType::get(type.getShape(), type.getElementType(), layout);
      Value strided = rw.create<memref::CastOp>(loc, inputType, source);
      auto fnType = rw.getFunctionType({inputType, memref}, {rw.getI32Type()});
      auto callee =
          getOperation().lookupSymbol<func::FuncOp>("qwen_ordered_silu_v1");
      if (!callee) {
        OpBuilder::InsertionGuard guard(rw);
        rw.setInsertionPointToStart(getOperation().getBody());
        callee = rw.create<func::FuncOp>(loc, "qwen_ordered_silu_v1", fnType);
        callee.setPrivate();
        callee->setAttr("llvm.emit_c_interface", rw.getUnitAttr());
      }
      if (callee.getFunctionType() != fnType) {
        op.emitError("mismatched SiLU ABI");
        return signalPassFailure();
      }
      Value status =
          rw.create<func::CallOp>(loc, callee, ValueRange{strided, output})
              .getResult(0);
      Value ok =
          rw.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, status,
                                   rw.create<arith::ConstantIntOp>(loc, 0, 32));
      rw.create<cf::AssertOp>(loc, ok, "ordered SiLU failed");
      Value tensor =
          rw.create<bufferization::ToTensorOp>(loc, type, output, true, true);
      rw.replaceOp(op, tensor);
      rw.eraseOp(denom);
    }
    llvm::errs() << "Parallel strict SiLU tensors: " << work.size() << "\n";
  }
};
