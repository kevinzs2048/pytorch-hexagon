// SPDX-License-Identifier: BSD-3-Clause
// FP16 resident KV storage, llama.cpp's default KV cache type (F16).
// Runs on the prepared graph after qwen-resident-kv and attention fusion:
//   * every caller-owned KV plane argument becomes tensor<1x8x?x128xf16>;
//   * the append copy of new rows becomes qwen_kv_append_f16_v1, which rounds
//     them once (IEEE round-to-nearest-even, the HVX vcvt conversion the HMX
//     attention kernels already applied to FP32 KV operands) into the plane.
//     An in-graph arith.truncf lowers to scalar code (about 1 ns per value,
//     68 us per prefill layer); the runtime uses the HVX conversion;
//   * the fused attention call receives the FP16 planes directly
//     (qwen_ordered_fp16kv_attention_v1).
//   * attention over fresh in-graph K/V (initial domain, no caller planes)
//     rounds those operands the same way, so every attention reads F16 KV.
// Returned KV deltas stay FP32; the session rounds them with the same
// conversion when it owns the copy. Any other use of a plane is rejected.
struct FP16KVStoragePass
    : PassWrapper<FP16KVStoragePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FP16KVStoragePass)
  FP16KVStoragePass() = default;
  FP16KVStoragePass(const FP16KVStoragePass &p) : PassWrapper(p) {}
  Option<unsigned> expectedPlanes{*this, "expected-planes", llvm::cl::init(0)};
  Option<unsigned> expectedCalls{*this, "expected-calls", llvm::cl::init(0)};
  StringRef getArgument() const final { return "qwen-fp16-kv-storage"; }
  StringRef getDescription() const final {
    return "Store resident KV planes as FP16 and pass them to FP16-KV attention";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect, tensor::TensorDialect>();
  }
  static constexpr StringLiteral FP32Attention = "qwen_ordered_fp32_attention_v1";
  static constexpr StringLiteral FP16Attention = "qwen_ordered_fp16kv_attention_v1";
  static constexpr StringLiteral AppendF16 = "qwen_kv_append_f16_v1";

  // memref<1x8x?x128xT, strided<[?, ?, ?, ?], offset: ?>>: one declaration for
  // decode (one row) and prefill appends; the runtime checks unit inner strides.
  static MemRefType rows(MLIRContext *ctx, Type element) {
    int64_t d = ShapedType::kDynamic;
    return MemRefType::get({1, 8, d, 128}, element, StridedLayoutAttr::get(ctx, d, {d, d, d, d}));
  }
  static func::FuncOp appendDeclaration(IRRewriter &b, ModuleOp module, Location loc) {
    MLIRContext *ctx = module.getContext();
    auto type = FunctionType::get(ctx, {rows(ctx, Float32Type::get(ctx)), rows(ctx, Float16Type::get(ctx))}, {});
    if (auto existing = module.lookupSymbol<func::FuncOp>(AppendF16))
      return existing.getFunctionType() == type ? existing : func::FuncOp();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    auto declaration = func::FuncOp::create(b, loc, AppendF16, type);
    declaration.setPrivate();
    declaration->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    return declaration;
  }

  static bool planeType(Type type, bool f16) {
    auto t = dyn_cast<RankedTensorType>(type);
    return t && !t.getEncoding() && t.getRank() == 4 && t.getDimSize(0) == 1 &&
           t.getDimSize(1) == 8 && t.isDynamicDim(2) && t.getDimSize(3) == 128 &&
           (f16 ? t.getElementType().isF16() : t.getElementType().isF32());
  }
  static Type half(Type type) {
    auto f16 = Float16Type::get(type.getContext());
    if (auto t = dyn_cast<RankedTensorType>(type)) return t.clone(f16);
    return MemRefType::Builder(cast<MemRefType>(type)).setElementType(f16);
  }
  static Value elementwise(IRRewriter &b, Location loc, Value input, Type element, bool narrow) {
    auto type = cast<RankedTensorType>(input.getType());
    SmallVector<Value> dynamic;
    for (int64_t i = 0; i < type.getRank(); ++i)
      if (type.isDynamicDim(i)) dynamic.push_back(tensor::DimOp::create(b, loc, input, i));
    // Used only where the result is consumed (copied) before any later write:
    // equal-shaped empties may be CSE'd into one buffer.
    Value init = tensor::EmptyOp::create(b, loc, type.getShape(), element, dynamic);
    SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(type.getRank()));
    SmallVector<utils::IteratorType> iterators(type.getRank(), utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{input},
        ValueRange{init}, maps, iterators,
        [&](OpBuilder &nb, Location l, ValueRange args) {
          Value v = narrow ? Value(arith::TruncFOp::create(nb, l, element, args[0]))
                           : Value(arith::ExtFOp::create(nb, l, element, args[0]));
          linalg::YieldOp::create(nb, l, v);
        });
    return generic.getResult(0);
  }

  // Round fresh K and V together with one two-result elementwise generic,
  // placed directly before the call. The results stay tensor-level so the
  // conversion fuses and vectorizes (a memref-level generic lowered to scalar
  // half conversions). Separate generics were wrong: the pipeline's empty-tensor
  // CSE gave both destinations one buffer, so V's conversion overwrote K's after
  // to_buffer had already handed it to the call (initial-domain K == V). Two
  // results of one op cannot share a buffer, and nothing writes between the
  // to_buffer views and the call. The FP16 copies keep the [1,8,T,128] layout.
  static void roundFresh(IRRewriter &b, func::CallOp call, Value key, Value value) {
    Location loc = call.getLoc();
    Type f16 = Float16Type::get(b.getContext());
    b.setInsertionPoint(call);
    SmallVector<Value> inits;
    for (Value input : {key, value}) {
      auto type = cast<RankedTensorType>(input.getType());
      Value tokens = tensor::DimOp::create(b, loc, input, 2);
      inits.push_back(tensor::EmptyOp::create(b, loc, type.getShape(), f16, ValueRange{tokens}));
    }
    SmallVector<AffineMap> maps(4, b.getMultiDimIdentityMap(4));
    SmallVector<utils::IteratorType> iterators(4, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(b, loc, TypeRange{inits[0].getType(), inits[1].getType()},
        ValueRange{key, value}, inits, maps, iterators,
        [&](OpBuilder &nb, Location l, ValueRange args) {
          linalg::YieldOp::create(nb, l, ValueRange{Value(arith::TruncFOp::create(nb, l, f16, args[0])),
                                                    Value(arith::TruncFOp::create(nb, l, f16, args[1]))});
        });
    for (unsigned index : {1u, 2u}) {
      auto operand = cast<MemRefType>(call.getOperand(index).getType());
      MemRefType view = MemRefType::Builder(operand).setElementType(f16);
      Value buffer = bufferization::ToBufferOp::create(b, loc, view, generic.getResult(index - 1),
                                                       /*readOnly=*/true);
      call->setOperand(index, buffer);
    }
  }

  void runOnOperation() final {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    IRRewriter b(ctx);
    Type f16 = Float16Type::get(ctx), f32 = Float32Type::get(ctx);
    unsigned planes = 0, appends = 0;
    llvm::SetVector<Operation *> calls;
    auto fail = [&](Operation *op, const Twine &message) {
      op->emitError(message);
      signalPassFailure();
    };
    for (auto f : module.getOps<func::FuncOp>()) {
      if (!f->hasAttr("hexagon.resident_kv") || f.isDeclaration()) continue;
      for (BlockArgument arg : f.getArguments()) {
        if (!f.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "hexagon.kv_plane")) continue;
        if (!planeType(arg.getType(), false))
          return fail(f, "FP16 KV: plane argument is not tensor<1x8x?x128xf32>");
        arg.setType(half(arg.getType()));
        ++planes;
        // Retype the plane's buffer views; rewrite their FP32 boundaries.
        SmallVector<Value> work{arg};
        while (!work.empty()) {
          Value value = work.pop_back_val();
          for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
            Operation *user = use.getOwner();
            if (isa<tensor::DimOp, memref::DimOp>(user)) continue;
            if (isa<bufferization::ToBufferOp, memref::ReinterpretCastOp, memref::SubViewOp,
                    memref::CastOp>(user) && use.getOperandNumber() == 0) {
              Value result = user->getResult(0);
              result.setType(half(result.getType()));
              work.push_back(result);
              continue;
            }
            if (auto copy = dyn_cast<memref::CopyOp>(user)) {
              auto source = copy.getSource().getDefiningOp<bufferization::ToBufferOp>();
              if (use.getOperandNumber() != 1 || !source || !source.getReadOnly())
                return fail(copy, "FP16 KV: append copy source is not a read-only tensor buffer");
              auto append = appendDeclaration(b, module, copy.getLoc());
              if (!append)
                return fail(copy, "FP16 KV: inconsistent qwen_kv_append_f16_v1 declaration");
              b.setInsertionPoint(copy);
              Value rowsF32 = memref::CastOp::create(b, copy.getLoc(), rows(ctx, f32), source.getResult());
              Value planeF16 = memref::CastOp::create(b, copy.getLoc(), rows(ctx, f16), copy.getTarget());
              func::CallOp::create(b, copy.getLoc(), append, ValueRange{rowsF32, planeF16});
              copy.erase();
              ++appends;
              continue;
            }
            if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
              Value result = toTensor.getResult();
              SmallVector<OpOperand *> uses;
              for (OpOperand &u : result.getUses()) uses.push_back(&u);
              result.setType(half(result.getType()));
              b.setInsertionPointAfter(toTensor);
              Value widened = elementwise(b, toTensor.getLoc(), result, f32, false);
              for (OpOperand *u : uses) u->set(widened);
              continue;
            }
            if (auto call = dyn_cast<func::CallOp>(user)) {
              if (call.getCallee() != FP32Attention ||
                  (use.getOperandNumber() != 1 && use.getOperandNumber() != 2))
                return fail(call, "FP16 KV: plane reaches an unexpected call operand");
              calls.insert(call);
              continue;
            }
            return fail(user, "FP16 KV: unsupported plane use");
          }
        }
      }
      SmallVector<Type> inputs(f.front().getArgumentTypes());
      f.setType(FunctionType::get(ctx, inputs, f.getResultTypes()));
    }
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == FP32Attention) calls.insert(call);
    });
    for (Operation *op : calls) {
      auto call = cast<func::CallOp>(op);
      SmallVector<bufferization::ToBufferOp> fresh;
      for (unsigned index : {1u, 2u}) {
        Value operand = call.getOperand(index);
        if (!cast<MemRefType>(operand.getType()).getElementType().isF32()) continue;
        auto source = operand.getDefiningOp<bufferization::ToBufferOp>();
        if (!source || !source.getReadOnly() || !planeType(source.getTensor().getType(), false))
          return fail(call, "FP16 KV: fresh attention K/V is not a read-only FP32 tensor buffer");
        fresh.push_back(source);
      }
      if (fresh.size() == 1)
        return fail(call, "FP16 KV: attention mixes a resident plane with a fresh K/V operand");
      if (fresh.size() == 2) {
        roundFresh(b, call, fresh[0].getTensor(), fresh[1].getTensor());
        for (auto source : fresh)
          if (source->use_empty()) source.erase();
      }
      auto type = FunctionType::get(ctx, call.getOperandTypes(), call.getResultTypes());
      auto k = cast<MemRefType>(type.getInput(1)), v = cast<MemRefType>(type.getInput(2));
      if (!k.getElementType().isF16() || !v.getElementType().isF16())
        return fail(call, "FP16 KV: attention K and V must both be FP16 planes");
      auto declaration = module.lookupSymbol<func::FuncOp>(FP16Attention);
      if (!declaration) {
        b.setInsertionPointToStart(module.getBody());
        declaration = func::FuncOp::create(b, call.getLoc(), FP16Attention, type);
        declaration.setPrivate();
        declaration->setAttr("llvm.emit_c_interface", b.getUnitAttr());
      } else if (declaration.getFunctionType() != type) {
        return fail(call, "FP16 KV: inconsistent FP16-KV attention signatures");
      }
      b.setInsertionPoint(call);
      auto replacement = func::CallOp::create(b, call.getLoc(), declaration, call.getOperands());
      b.replaceOp(call, replacement.getResults());
    }
    if (auto old = module.lookupSymbol<func::FuncOp>(FP32Attention))
      if (old.isDeclaration() && SymbolTable::symbolKnownUseEmpty(old, module)) old.erase();
    if ((expectedPlanes && planes != expectedPlanes) || (expectedCalls && calls.size() != expectedCalls)) {
      module.emitError() << "FP16 KV: rewrote " << planes << " planes and " << calls.size()
                         << " attention calls; expected " << expectedPlanes.getValue() << " and "
                         << expectedCalls.getValue();
      signalPassFailure();
      return;
    }
    llvm::errs() << "fp16 kv storage planes: " << planes << " attention calls: " << calls.size()
                 << " appends: " << appends << "\n";
  }
};
