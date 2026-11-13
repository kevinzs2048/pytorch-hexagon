// SPDX-License-Identifier: BSD-3-Clause
// Explicit scalar helper contracts; preserve every arithmetic operation.
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
namespace {
#include "ParallelSilu.h"
#include "BatchARMTrig.h"
struct Snapshot {
  Operation *op;
  DictionaryAttr attrs;
  Attribute properties;
  SmallVector<Value> operands;
  SmallVector<Type> resultTypes;
  SmallVector<Block *> successors;
  SmallVector<Value> arguments;
};
struct ScalarHelperEffectsPass:PassWrapper<ScalarHelperEffectsPass,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScalarHelperEffectsPass)
  ScalarHelperEffectsPass()=default;
  ScalarHelperEffectsPass(const ScalarHelperEffectsPass &other):PassWrapper(other) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  Option<unsigned> expectedCount{*this,"expected-count",llvm::cl::desc("Exact admitted marked query1 SiLU call count"),llvm::cl::init(28)};
  Option<bool> multiRow{*this,"multi-row",llvm::cl::desc("Opt-in existing query2..32 SiLU calls; default retains original query1 scope"),llvm::cl::init(false)};
  Option<bool> residentGraphs{*this,"resident-graphs",llvm::cl::desc("Admit the four pinned resident graph entry names"),llvm::cl::init(false)};
  StringRef getArgument() const final {return "qwen-silu-effects";}
  StringRef getDescription() const final {return "Pinned SiLU readnone/nounwind/willreturn; all original operations preserved";}
  void runOnOperation() final {
    ModuleOp module=getOperation();
    constexpr StringLiteral name="__hexagon_silu_exp_compat";
    auto callee=module.lookupSymbol<func::FuncOp>(name);
    auto expected=FunctionType::get(module.getContext(),{Float32Type::get(module.getContext())},
                                   {Float32Type::get(module.getContext())});
    if (!callee || !callee.isExternal() || !callee.isPrivate() ||
        callee.getFunctionType()!=expected || !expectedCount ||
        callee->hasAttr("llvm.readnone") || callee->hasAttr("memory_effects") ||
        callee->hasAttr("passthrough")) {
      module.emitError("unannotated private external pinned scalar SiLU declaration required");
      return signalPassFailure();
    }
    auto uses=SymbolTable::getSymbolUses(callee,module);
    if (!uses || uses->empty()) {
      module.emitError("complete direct SiLU symbol uses required");return signalPassFailure();
    }
    unsigned count=0;
    for (auto use:*uses) {
      auto call=dyn_cast<func::CallOp>(use.getUser());
      auto parent=call ? call->getParentOfType<linalg::GenericOp>() : linalg::GenericOp();
      auto owner=call ? call->getParentOfType<func::FuncOp>() : func::FuncOp();
      auto type=parent && parent.getNumResults()==1 ?
          dyn_cast<RankedTensorType>(parent.getResult(0).getType()) : RankedTensorType();
      bool shape=false;
      bool ownerMatches = owner && owner.getSymName()=="Qwen3DynamicBoundProbe";
      if (residentGraphs && owner) {
        StringRef name = owner.getSymName();
        ownerMatches |= multiRow
            ? (name=="Qwen3DynamicCache0" || name=="Qwen3DynamicPrefill")
            : (name=="Qwen3DynamicSingle0" || name=="Qwen3DynamicCache1");
      }
      if (type && type.getRank()==3 && type.getDimSize(0)==1 && type.getDimSize(2)==3072) {
        int64_t rows=type.getDimSize(1);
        shape=multiRow ? (rows==ShapedType::kDynamic || (rows>=2 && rows<=32)) : rows==1;
      }
      if (!call || call.getCallee()!=name || !isa_and_nonnull<UnitAttr>(call->getAttr("hexagon.fp32_silu_exp_compat")) ||
          call.getNumOperands()!=1 || call.getOperand(0).getType()!=Float32Type::get(module.getContext()) ||
          call.getResultTypes()!=expected.getResults() || !ownerMatches ||
          !parent || !parent->hasAttr("hexagon.fp32_arm_silu_denominator") || !type ||
          !shape || !type.getElementType().isF32()) {
        module.emitError("only original marked query1 SiLU scalar direct calls are admitted");
        return signalPassFailure();
      }
      ++count;
    }
    if (count!=expectedCount) {
      module.emitError()<<"SiLU effect coverage differs: "<<count<<" != "<<expectedCount.getValue();
      return signalPassFailure();
    }
    SmallVector<Snapshot,0> before;
    module.walk([&](Operation *op) {
      Snapshot s{op,op->getAttrDictionary(),op->getPropertiesAsAttribute(),
        SmallVector<Value>(op->getOperands()),SmallVector<Type>(op->getResultTypes()),
        SmallVector<Block *>(op->getSuccessors()),{}};
      for (auto &region:op->getRegions()) for (auto &block:region)
        llvm::append_range(s.arguments,block.getArguments());
      before.push_back(std::move(s));
    });
    Builder builder(module.getContext());
    callee->setAttr("llvm.readnone",builder.getUnitAttr());
    if (residentGraphs) {
      // FuncToLLVM filters llvm.readnone when creating an external C wrapper.
      // This scalar-only ABI also carries the modern attribute through it.
      callee->setAttr("memory_effects", LLVM::MemoryEffectsAttr::get(module.getContext(),
          {LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
           LLVM::ModRefInfo::NoModRef}));
    }
    callee->setAttr("passthrough",builder.getStrArrayAttr({"nounwind","willreturn"}));
    DictionaryAttr changed=callee->getAttrDictionary();
    SmallVector<Operation *> after;
    module.walk([&](Operation *op) {after.push_back(op);});
    if (after.size()!=before.size()) {
      module.emitError("original operation count changed");return signalPassFailure();
    }
    for (auto [i,s]:llvm::enumerate(before)) {
      Operation *op=after[i];SmallVector<Value> arguments;
      for (auto &region:op->getRegions()) for (auto &block:region)
        llvm::append_range(arguments,block.getArguments());
      if (op!=s.op || op->getAttrDictionary()!=(op==callee.getOperation() ? changed : s.attrs) ||
          op->getPropertiesAsAttribute()!=s.properties ||
          SmallVector<Value>(op->getOperands())!=s.operands ||
          SmallVector<Type>(op->getResultTypes())!=s.resultTypes ||
          SmallVector<Block *>(op->getSuccessors())!=s.successors || arguments!=s.arguments) {
        module.emitError("original operation, property, FP order or SSA dependency changed");
        return signalPassFailure();
      }
    }
    llvm::errs()<<"qwen scoped "<<(multiRow ? "query2..32" : "query1")<<" SiLU effect calls: "<<count<<"\n";
    llvm::errs()<<"qwen preserved original operations: "<<before.size()<<"\n";
  }
};
}
int main(int argc,char **argv) {
  DialectRegistry registry;
  registry.insert<affine::AffineDialect,arith::ArithDialect,cf::ControlFlowDialect,
      func::FuncDialect,linalg::LinalgDialect,LLVM::LLVMDialect,math::MathDialect,
      bufferization::BufferizationDialect,memref::MemRefDialect,scf::SCFDialect,
      tensor::TensorDialect,vector::VectorDialect,tm_tensor::TmTensorDialect>();
  PassRegistration<ScalarHelperEffectsPass>();
  PassRegistration<ParallelSiluPass>();
  PassRegistration<BatchARMTrigPass>();
  return asMainReturnCode(MlirOptMain(argc,argv,"Scoped original Qwen SiLU effect contract\n",registry));
}
