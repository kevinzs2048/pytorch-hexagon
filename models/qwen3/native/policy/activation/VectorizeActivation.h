// SPDX-License-Identifier: BSD-3-Clause
// Preserve checked scale division and IEEE ties-to-even while exposing SIMD.
// Included after ReuseActivation.h, whose strict reciprocal matcher is shared.
#include "FuseBoundedQuant.h"

namespace {
struct VectorizeActivationPass
    : PassWrapper<VectorizeActivationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeActivationPass)
  VectorizeActivationPass() = default;
  VectorizeActivationPass(const VectorizeActivationPass &other)
      : PassWrapper(other) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(113)};
  StringRef getArgument() const final { return "qwen-vectorize-activation"; }
  StringRef getDescription() const final {
    return "Hoist row scale checks and expand exact f32 ties-to-even";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, tensor::TensorDialect, arith::ArithDialect,
                    cf::ControlFlowDialect>();
  }
  void runOnOperation() final {
    SmallVector<linalg::GenericOp> work;
    SmallVector<math::RoundEvenOp> rounds;
    bool invalid = false;
    auto admittedOwner = [](Operation *op) {
      auto owner = op->getParentOfType<func::FuncOp>();
      return owner && (owner.getName() == "Qwen3DynamicCache0" ||
                       owner.getName() == "Qwen3DynamicPrefill" ||
                       owner.getName() == "Qwen3DynamicSingle0" ||
                       owner.getName() == "Qwen3DynamicCache1");
    };
    getOperation().walk([&](linalg::GenericOp op) {
      if (checkedActivationReciprocal(op)) {
        if (!admittedOwner(op))
          invalid = true;
        work.push_back(op);
      }
    });
    getOperation().walk([&](math::RoundEvenOp op) {
      if (!admittedOwner(op) || !op->getParentOfType<linalg::GenericOp>() ||
          !op.getType().isF32() ||
          op.getFastmath() != arith::FastMathFlags::none)
        invalid = true;
      rounds.push_back(op);
    });
    if (invalid || work.size() != expectedCount ||
        rounds.size() != expectedCount) {
      getOperation().emitError()
          << "activation vectorization scope/count mismatch: " << work.size()
          << " reciprocals, " << rounds.size() << " rounding operations";
      return signalPassFailure();
    }
    // Validate the complete transaction before modifying any operation.
    for (auto op : work) {
      auto ty = cast<RankedTensorType>(op.getResult(0).getType());
      auto owner = op->getParentOfType<func::FuncOp>();
      bool single = owner.getName() == "Qwen3DynamicSingle0" ||
                    owner.getName() == "Qwen3DynamicCache1";
      if (ty.getRank() != 3 || ty.getDimSize(0) != 1 || ty.getDimSize(2) != 1 ||
          ty.getEncoding() || (single && ty.getDimSize(1) != 1) ||
          (!ty.isDynamicDim(1) &&
           (ty.getDimSize(1) < 1 || ty.getDimSize(1) > 32)) ||
          op.getDpsInputs()[0].getType() != ty ||
          op.getDpsInits()[0].getType() != ty ||
          llvm::any_of(op.getIndexingMapsArray(),
                       [](AffineMap m) { return !m.isIdentity(); })) {
        op.emitError("unsupported activation reciprocal shape/maps");
        return signalPassFailure();
      }
    }
    // Check the same immutable scale tensor before division. Failed checks
    // still abort; successful rows retain the exact original division and
    // threshold.
    IRRewriter rw(&getContext());
    for (auto op : work) {
      rw.setInsertionPoint(op);
      auto loc = op.getLoc();
      Value input = op.getDpsInputs()[0];
      Value rows = rw.create<tensor::DimOp>(loc, input, 1);
      Value zero = rw.create<arith::ConstantIndexOp>(loc, 0),
            one = rw.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = rw.create<scf::ForOp>(loc, zero, rows, one);
      rw.setInsertionPointToStart(loop.getBody());
      Value x = rw.create<tensor::ExtractOp>(
          loc, input, ValueRange{zero, loop.getInductionVar(), zero});
      auto cmp = cast<arith::CmpFOp>(op.getBody()->front());
      auto check = cast<cf::AssertOp>(*std::next(op.getBody()->begin()));
      IRMapping mapping;
      mapping.map(op.getBody()->getArgument(0), x);
      rw.clone(*cmp, mapping);
      rw.clone(*check, mapping);
      check.erase();
      cmp.erase();
    }
    // For 0 <= |x| < 2**23, adding 2**23 rounds to the nearest integer
    // with ties to even. Subtracting it is exact. Keep both IEEE operations
    // (no reassociation), restore the sign including -0, and select the original
    // bits for larger integral values and all infinities/NaNs. This preserves
    // the pinned roundevenf value contract without variable integer shifts.
    IRRewriter b(&getContext());
    unsigned fused = 0;
    for (auto op : rounds)
      fused += fuseBoundedQuant(op, b);
    llvm::errs() << "qwen bounded quant fusion: " << fused << "\n";
    for (auto op : rounds) {
      b.setInsertionPoint(op);
      auto loc = op.getLoc();
      auto c = [&](uint32_t value) -> Value {
        return b.create<arith::ConstantIntOp>(loc, value, 32);
      };
      Value bits =
          b.create<arith::BitcastOp>(loc, b.getI32Type(), op.getOperand());
      Value mag = b.create<arith::AndIOp>(loc, bits, c(0x7fffffff));
      Value sign = b.create<arith::AndIOp>(loc, bits, c(0x80000000));
      Value absolute = b.create<arith::BitcastOp>(loc, b.getF32Type(), mag);
      Value magic = b.create<arith::ConstantOp>(
          loc, b.getF32FloatAttr(8388608.0));
      Value shifted = b.create<arith::AddFOp>(loc, absolute, magic);
      Value rounded = b.create<arith::SubFOp>(loc, shifted, magic);
      Value result = b.create<arith::BitcastOp>(loc, b.getI32Type(), rounded);
      result = b.create<arith::OrIOp>(loc, result, sign);
      Value large = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::uge, mag, c(0x4b000000));
      result = b.create<arith::SelectOp>(loc, large, bits, result);
      b.replaceOp(
          op, b.create<arith::BitcastOp>(loc, b.getF32Type(), result).getResult());
    }
    llvm::errs() << "qwen vectorized activation rows: " << work.size() << "\n";
  }
};
} // namespace
