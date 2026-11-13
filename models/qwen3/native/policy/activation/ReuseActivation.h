// SPDX-License-Identifier: BSD-3-Clause
// Share identical checked reciprocals without dropping the first assertion.
// This allows ordinary CSE to share Q/K/V and gate/up activation quantization.
namespace {
bool activationScalar(Value value, double expected) {
  llvm::APFloat number(0.0f);
  return matchPattern(value, m_ConstantFloat(&number)) &&
         number.isExactlyValue(expected);
}
bool checkedActivationReciprocal(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
      op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 ||
      !isa<RankedTensorType>(op.getResult(0).getType()) ||
      !isa<RankedTensorType>(op.getDpsInputOperand(0)->get().getType()) ||
      llvm::any_of(op.getIteratorTypesArray(), [](auto kind) {
        return kind != utils::IteratorType::parallel;
      }))
    return false;
  auto &body = op.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getArgument(1).use_empty() ||
      body.getOperations().size() != 4)
    return false;
  auto it = body.begin();
  auto cmp = dyn_cast<arith::CmpFOp>(&*it++);
  auto check = dyn_cast<cf::AssertOp>(&*it++);
  auto div = dyn_cast<arith::DivFOp>(&*it++);
  auto yield = dyn_cast<linalg::YieldOp>(&*it);
  Value input = body.getArgument(0);
  return input.getType().isF32() && cmp && check && div && yield &&
         cmp.getPredicate() == arith::CmpFPredicate::ONE &&
         cmp.getLhs() == input && activationScalar(cmp.getRhs(), 0.0) &&
         cmp.getFastmath() == arith::FastMathFlags::none &&
         check.getArg() == cmp.getResult() &&
         activationScalar(div.getLhs(), 1.0) && div.getRhs() == input &&
         div.getFastmath() == arith::FastMathFlags::none &&
         yield.getNumOperands() == 1 && yield.getOperand(0) == div.getResult();
}
struct ReuseActivationPass
    : PassWrapper<ReuseActivationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseActivationPass)
  ReuseActivationPass() = default;
  ReuseActivationPass(const ReuseActivationPass &other) : PassWrapper(other) {}
  Option<unsigned> expectedCount{*this, "expected-count", llvm::cl::init(84)};
  StringRef getArgument() const final { return "qwen-reuse-activation"; }
  StringRef getDescription() const final {
    return "Reuse an already checked identical tensor reciprocal; preserve FP "
           "arithmetic";
  }
  void runOnOperation() final {
    SmallVector<linalg::GenericOp> candidates, seen;
    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>> work;
    getOperation().walk([&](linalg::GenericOp op) {
      if (checkedActivationReciprocal(op))
        candidates.push_back(op);
    });
    for (auto op : candidates) {
      linalg::GenericOp prior;
      for (auto old : seen)
        if (old->getBlock() == op->getBlock() && old->isBeforeInBlock(op) &&
            old.getDpsInputOperand(0)->get() ==
                op.getDpsInputOperand(0)->get() &&
            old.getDpsInitOperand(0)->get() == op.getDpsInitOperand(0)->get() &&
            old->getAttrDictionary() == op->getAttrDictionary() &&
            old.getResult(0).getType() == op.getResult(0).getType() &&
            old.getIndexingMaps() == op.getIndexingMaps() &&
            old.getIteratorTypes() == op.getIteratorTypes()) {
          prior = old;
          break;
        }
      if (prior)
        work.emplace_back(op, prior);
      else
        seen.push_back(op);
    }
    if (work.size() != expectedCount) {
      getOperation().emitError()
          << "activation reuse coverage differs: " << work.size()
          << " != " << expectedCount.getValue();
      return signalPassFailure();
    }
    // Same block, same tensor operands and iteration bounds, same pure scalar
    // body. If the first assertion fails, the duplicate is never reached; if
    // it succeeds, every duplicate sees the same values and succeeds as well.
    // Keep the original division result instead of recomputing it. No FP
    // reduction, scale, rounding operation or quantization threshold changes.
    for (auto [op, prior] : work) {
      op.getResult(0).replaceAllUsesWith(prior.getResult(0));
      op.erase();
    }
    llvm::errs() << "qwen reused activation reciprocals: " << work.size()
                 << "\n";
  }
};
} // namespace
