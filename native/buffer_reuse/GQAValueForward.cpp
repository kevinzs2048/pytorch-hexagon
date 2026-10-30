// SPDX-License-Identifier: BSD-3-Clause
// Forward scalar reads through immutable pure V copies and shape views.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#ifdef QWEN_GQA_KEY_COMPACT
unsigned compactQwenGQAKeys(mlir::func::FuncOp function);
#endif

std::unique_ptr<mlir::Pass> originalCSEFactory()
    asm("__real__ZN4mlir13createCSEPassEv");
unsigned forwardQwenGQARepeatSlices(mlir::func::FuncOp function);
#ifdef QWEN_GQA_DEAD_REPEAT
unsigned eraseQwenDeadGQARepeats(mlir::func::FuncOp function);
#endif
std::unique_ptr<mlir::Pass> wrappedCSEFactory()
    asm("__wrap__ZN4mlir13createCSEPassEv");

namespace {
using namespace mlir;
bool admitted(func::FuncOp function) {
  if (function.isDeclaration()) return false;
  if (function.getSymName()=="qwen_gqa_value_forward" &&
      function->hasAttr("qwen.diagnostic_gqa_value_forward")) return true;
#ifdef QWEN_GQA_KEY_COMPACT
  if (function.getSymName()=="qwen_gqa_key_compact" &&
      function->hasAttr("qwen.diagnostic_gqa_key_compact")) return true;
#endif
  auto order=function->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
  auto viewReady=function->getAttrOfType<BoolAttr>("qwen.gqa_identity_view_complete");
  auto module=function->getParentOfType<ModuleOp>();
  auto preflight=module ? module->getAttrOfType<BoolAttr>("qwen.diagnostic_v_repeat_preflight") : BoolAttr();
  auto tokens=function.getNumArguments() ? dyn_cast<RankedTensorType>(function.getArgument(0).getType()) : RankedTensorType();
  return function.getSymName()=="Qwen3DynamicBoundProbe" && function.getNumArguments()==62 &&
      function.getNumResults()==57 && tokens && tokens.getShape()==ArrayRef<int64_t>({1,1}) &&
      tokens.getElementType().isInteger(64) && order && order.getValue()=="qk-fma-pv-openblas-armv8-k1024" &&
      ((viewReady && viewReady.getValue()) || (preflight && preflight.getValue()));
}
bool pureValueCopy(linalg::GenericOp generic) {
  if (!generic || generic.getNumResults()!=1 || generic.getNumDpsInits()!=1 ||
      generic.getNumDpsInputs()>1 || generic.getNumReductionLoops()) return false;
  auto type=dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!type || !type.getElementType().isF32() ||
      (type.getShape()!=ArrayRef<int64_t>({8,2,ShapedType::kDynamic,128}) &&
       type.getShape()!=ArrayRef<int64_t>({16,ShapedType::kDynamic,128}) &&
       type.getShape()!=ArrayRef<int64_t>({1,8,2,ShapedType::kDynamic,128}) &&
       type.getShape()!=ArrayRef<int64_t>({1,16,ShapedType::kDynamic,128}))) return false;
  auto maps=generic.getIndexingMapsArray();
  if (generic.getNumLoops()!=unsigned(type.getRank()) || !maps.back().isIdentity() ||
      !llvm::all_of(generic.getIteratorTypesArray(),[](utils::IteratorType t){return t==utils::IteratorType::parallel;}) ||
      !generic.getBody()->getArgument(generic.getNumDpsInputs()).use_empty()) return false;
  for (auto item:llvm::enumerate(generic.getDpsInputs())) {
    auto input=dyn_cast<RankedTensorType>(item.value().getType());
    if (!input || input.getRank()!=0 || !input.getElementType().isInteger(1) || maps[item.index()].getNumResults()) return false;
  }
  auto yield=cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  if (yield.getValues().size()!=1 || !yield.getValues()[0].getDefiningOp<tensor::ExtractOp>()) return false;
  for (Operation &op:*generic.getBody())
    if (!isa<linalg::IndexOp,linalg::YieldOp,tensor::ExtractOp,
             arith::ConstantIndexOp,arith::CmpIOp,arith::SelectOp,
             arith::AddIOp,arith::MulIOp,arith::DivUIOp,arith::RemUIOp>(op)) return false;
  return true;
}
bool staticGroups(RankedTensorType type,ArrayRef<ReassociationIndices> groups) {
  for (auto group:groups) if (group.size()>1) {
    int64_t product=1;
    for (int64_t axis:group) {
      int64_t size=type.getDimSize(axis);
      if (size<=0 || product>std::numeric_limits<int64_t>::max()/size) return false;
      product*=size;
    }
  }
  return true;
}
bool forward(tensor::ExtractOp extract,unsigned &repeatReads,unsigned &layoutReads) {
  Value source=extract.getTensor();SmallVector<Operation *> views;
  for (unsigned depth=0;depth<8;++depth) {
    if (auto collapse=source.getDefiningOp<tensor::CollapseShapeOp>()) {
      if (!staticGroups(cast<RankedTensorType>(collapse.getSrc().getType()),collapse.getReassociationIndices())) return false;
      views.push_back(collapse);source=collapse.getSrc();continue;
    }
    if (auto expand=source.getDefiningOp<tensor::ExpandShapeOp>()) {
      if (!staticGroups(cast<RankedTensorType>(expand.getResult().getType()),expand.getReassociationIndices())) return false;
      views.push_back(expand);source=expand.getSrc();continue;
    }
    break;
  }
  auto copy=source.getDefiningOp<linalg::GenericOp>();
  if (!pureValueCopy(copy)) return false;
  auto shape=cast<RankedTensorType>(copy.getResult(0).getType()).getShape();
  bool repeat=shape==ArrayRef<int64_t>({8,2,ShapedType::kDynamic,128}) ||
              shape==ArrayRef<int64_t>({1,8,2,ShapedType::kDynamic,128});
  OpBuilder builder(extract);Location loc=extract.getLoc();SmallVector<Value> indices(extract.getIndices());
  auto constant=[&](int64_t n)->Value {return arith::ConstantIndexOp::create(builder,loc,n);};
  for (Operation *op:views) {
    SmallVector<Value> next;
    if (auto collapse=dyn_cast<tensor::CollapseShapeOp>(op)) {
      auto type=cast<RankedTensorType>(collapse.getSrc().getType());
      for (auto item:llvm::enumerate(collapse.getReassociationIndices())) {
        auto group=item.value();Value flat=indices[item.index()];
        if (group.size()==1) {next.push_back(flat);continue;}
        for (unsigned i=0;i<group.size();++i) {
          int64_t stride=1;for (unsigned j=i+1;j<group.size();++j) stride*=type.getDimSize(group[j]);
          Value coordinate=flat;
          if (stride!=1) coordinate=arith::DivUIOp::create(builder,loc,flat,constant(stride));
          if (i) coordinate=arith::RemUIOp::create(builder,loc,coordinate,constant(type.getDimSize(group[i])));
          next.push_back(coordinate);
        }
      }
    } else {
      auto expand=cast<tensor::ExpandShapeOp>(op);auto type=cast<RankedTensorType>(expand.getResult().getType());
      for (auto group:expand.getReassociationIndices()) {
        Value flat=indices[group.front()];
        for (unsigned i=1;i<group.size();++i) {
          Value multiplied=arith::MulIOp::create(builder,loc,flat,constant(type.getDimSize(group[i])));
          flat=arith::AddIOp::create(builder,loc,multiplied,indices[group[i]]);
        }
        next.push_back(flat);
      }
    }
    indices=std::move(next);
  }
  IRMapping mapping;
  for (auto item:llvm::enumerate(copy.getDpsInputs())) {
    auto scalar=tensor::ExtractOp::create(builder,loc,item.value(),ValueRange{});
    mapping.map(copy.getBody()->getArgument(item.index()),scalar.getResult());
  }
  for (Operation &op:*copy.getBody()) {
    if (auto index=dyn_cast<linalg::IndexOp>(op)) mapping.map(index.getResult(),indices[index.getDim()]);
    else if (!isa<linalg::YieldOp>(op)) builder.clone(op,mapping);
  }
  auto yield=cast<linalg::YieldOp>(copy.getBody()->getTerminator());
  extract.getResult().replaceAllUsesWith(mapping.lookupOrDefault(yield.getValues()[0]));extract.erase();
  if (repeat) ++repeatReads;else ++layoutReads;
  return true;
}
struct GQAValueForwardPass:PassWrapper<GQAValueForwardPass,OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GQAValueForwardPass)
  StringRef getArgument() const final {return "qwen-gqa-value-scalar-forward";}
  void runOnOperation() final {
    unsigned changed=0,repeatReads=0,layoutReads=0;
    getOperation()->walk([&](func::FuncOp function) {
      if (!admitted(function)) return;
      forwardQwenGQARepeatSlices(function);
#ifdef QWEN_GQA_KEY_COMPACT
      compactQwenGQAKeys(function);
#endif
      for (unsigned round=0;round<8;++round) {
        SmallVector<tensor::ExtractOp> extracts;function.walk([&](tensor::ExtractOp op){extracts.push_back(op);});
        unsigned step=0;for (auto extract:extracts) step+=forward(extract,repeatReads,layoutReads);
        changed+=step;if (!step) break;
      }
    });
    if (changed) llvm::errs()<<"QWEN_GQA_VALUE_FORWARD reads="<<changed<<" FP-arithmetic-changed=0 K-transpose-retained=1 model_repeat_reads="<<repeatReads<<" model_layout_reads="<<layoutReads<<"\n";
    OpPassManager original(getOperation()->getName().getStringRef());original.addPass(originalCSEFactory());
    if (failed(runPipeline(original,getOperation()))) {signalPassFailure();return;}
#ifdef QWEN_GQA_DEAD_REPEAT
    // CSE first removes the old forwarded layout's consumers. Only then is
    // the repeat result's closed unused shape-view graph visible.
    getOperation()->walk([&](func::FuncOp function) {
      if(admitted(function)) eraseQwenDeadGQARepeats(function);
    });
#endif
  }
};
} // namespace

std::unique_ptr<mlir::Pass> wrappedCSEFactory() {return std::make_unique<GQAValueForwardPass>();}
