// SPDX-License-Identifier: BSD-3-Clause
// Redirect readonly component slices through a fully proved repeat copy.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace {
using namespace mlir;
bool integer(Value value,int64_t n) {
  auto c=value.getDefiningOp<arith::ConstantIndexOp>();return c && c.value()==n;
}
bool integer(OpFoldResult value,int64_t n) {
  if(auto a=dyn_cast<Attribute>(value)) {auto i=dyn_cast<IntegerAttr>(a);return i && i.getInt()==n;}
  return integer(cast<Value>(value),n);
}
bool yielded(scf::ForOp loop,Value value) {
  auto y=cast<scf::YieldOp>(loop.getBody()->getTerminator());return y.getNumOperands()==1 && y.getOperand(0)==value;
}
bool groupsSafe(RankedTensorType type,ArrayRef<ReassociationIndices> groups) {
  if(groups.empty() || groups.back().size()!=1 || groups.back()[0]!=type.getRank()-1) return false;
  for(auto group:groups) if(group.size()>1) {
    int64_t product=1;
    for(int64_t axis:group) {int64_t n=type.getDimSize(axis);if(n<=0 || product>std::numeric_limits<int64_t>::max()/n) return false;product*=n;}
  }
  return type.getDimSize(type.getRank()-1)==128;
}
struct Copy {scf::ForOp root;SmallVector<scf::ForOp> loops;linalg::GenericOp scalar;tensor::ExtractOp load;};
bool prove(scf::ForOp root,Copy &copy) {
  if(root.getInitArgs().size()!=1 || root.getNumResults()!=1) return false;
  auto type=dyn_cast<RankedTensorType>(root.getResult(0).getType());
  if(!type || type.getEncoding() || !type.getElementType().isF32() || type.getShape()!=ArrayRef<int64_t>({8,2,ShapedType::kDynamic,128})) return false;
  auto empty=root.getInitArgs()[0].getDefiningOp<tensor::EmptyOp>();
  if(!empty || empty.getDynamicSizes().size()!=1) return false;
  SmallVector<scf::ForOp> loops{root};
  for(unsigned i=0;i<2;++i) {
    SmallVector<scf::ForOp> children;for(auto child:loops.back().getBody()->getOps<scf::ForOp>()) children.push_back(child);
    if(children.size()!=1) return false;loops.push_back(children[0]);
  }
  Value bounds[]={Value(),Value(),empty.getDynamicSizes()[0]};
  for(unsigned i=0;i<3;++i) {
    auto loop=loops[i];if(loop.getInitArgs().size()!=1 || loop.getNumResults()!=1 || !integer(loop.getLowerBound(),0) || !integer(loop.getStep(),1)) return false;
    if(i<2 ? !integer(loop.getUpperBound(),i==0?8:2) : loop.getUpperBound()!=bounds[2]) return false;
    if(i && (loop.getInitArgs()[0]!=loops[i-1].getRegionIterArgs()[0] || !yielded(loops[i-1],loop.getResult(0)))) return false;
  }
  linalg::GenericOp scalar;tensor::InsertSliceOp insert;bool valid=true;
  root.walk([&](Operation *op) {
    if(!isa<scf::ForOp,scf::YieldOp,tensor::ExtractSliceOp,tensor::InsertSliceOp,tensor::ExtractOp,
        linalg::GenericOp,linalg::IndexOp,linalg::YieldOp,arith::SelectOp,arith::ConstantIndexOp>(op)) valid=false;
    if(auto g=dyn_cast<linalg::GenericOp>(op)) {if(scalar) valid=false;scalar=g;}
    if(auto i=dyn_cast<tensor::InsertSliceOp>(op)) {if(insert) valid=false;insert=i;}
  });
  auto inner=loops.back();
  if(!valid || !scalar || !insert || scalar.getNumDpsInputs()>1 || scalar.getNumDpsInits()!=1 || scalar.getNumResults()!=1 ||
      scalar.getNumLoops()!=1 || scalar.getNumReductionLoops() || scalar.getIteratorTypesArray()[0]!=utils::IteratorType::parallel ||
      !scalar.getIndexingMapsArray().back().isIdentity() || !scalar.getBody()->getArgument(scalar.getNumDpsInputs()).use_empty() ||
      scalar->getBlock()!=inner.getBody() || insert->getBlock()!=inner.getBody()) return false;
  auto slice=scalar.getDpsInits()[0].getDefiningOp<tensor::ExtractSliceOp>();
  if(!slice || slice.getSource()!=inner.getRegionIterArgs()[0] || insert.getSource()!=scalar.getResult(0) || insert.getDest()!=inner.getRegionIterArgs()[0] ||
      !yielded(inner,insert.getResult()) || slice.getMixedOffsets()!=insert.getMixedOffsets() || slice.getMixedSizes()!=insert.getMixedSizes() || slice.getMixedStrides()!=insert.getMixedStrides()) return false;
  auto offsets=slice.getMixedOffsets(),sizes=slice.getMixedSizes(),strides=slice.getMixedStrides();
  if(offsets.size()!=4 || !integer(offsets[3],0) || !integer(sizes[3],128)) return false;
  for(unsigned i=0;i<3;++i) if(dyn_cast<Value>(offsets[i])!=loops[i].getInductionVar() || !integer(sizes[i],1)) return false;
  if(!llvm::all_of(strides,[](OpFoldResult s){return integer(s,1);})) return false;
  for(auto input:scalar.getDpsInputs()) {
    auto t=dyn_cast<RankedTensorType>(input.getType());if(!t || t.getRank()!=0 || !t.getElementType().isInteger(1) || !root.isDefinedOutsideOfLoop(input)) return false;
  }
  auto y=cast<linalg::YieldOp>(scalar.getBody()->getTerminator());
  if(y.getValues().size()!=1) return false;auto load=y.getValues()[0].getDefiningOp<tensor::ExtractOp>();
  if(!load || !root.isDefinedOutsideOfLoop(load.getTensor()) || load.getIndices().size()!=5) return false;
  auto source=dyn_cast<RankedTensorType>(load.getTensor().getType());
  if(!source || source.getEncoding() || source.getShape()!=ArrayRef<int64_t>({1,8,1,ShapedType::kDynamic,128})) return false;
  auto indices=load.getIndices();auto component=indices[4].getDefiningOp<linalg::IndexOp>();
  if(!integer(indices[0],0) || indices[1]!=loops[0].getInductionVar() || !integer(indices[2],0) || !component || component.getDim()!=0) return false;
  Value column=indices[3];
  if(column!=loops[2].getInductionVar()) {
    auto select=column.getDefiningOp<arith::SelectOp>();
    if(!select || scalar.getNumDpsInputs()!=1 || select.getCondition()!=scalar.getBody()->getArgument(0) || !integer(select.getTrueValue(),0) || select.getFalseValue()!=loops[2].getInductionVar()) return false;
  }
  copy={root,loops,scalar,load};return true;
}
bool redirect(tensor::ExtractSliceOp slice,Operation *&producer) {
  auto result=dyn_cast<RankedTensorType>(slice.getResult().getType());auto sourceType=dyn_cast<RankedTensorType>(slice.getSource().getType());
  if(!result || !sourceType || result.getRank()>1 || !result.getElementType().isF32() || sourceType.getRank()<3) return false;
  auto lengths=slice.getMixedSizes();auto strides=slice.getMixedStrides();
  if(!llvm::all_of(strides,[](OpFoldResult s){return integer(s,1);})) return false;
  for(unsigned i=0;i+1<lengths.size();++i) if(!integer(lengths[i],1)) return false;
  int64_t span=result.getRank()==0?1:result.getDimSize(0);
  if(span<=0 || span>128 || !integer(lengths.back(),span)) return false;
  Value source=slice.getSource();SmallVector<Operation *> views;
  for(unsigned depth=0;depth<16;++depth) {
    if(auto parent=source.getDefiningOp<tensor::ExtractSliceOp>()) {
      if(!llvm::all_of(parent.getMixedStrides(),[](OpFoldResult s){return integer(s,1);})) return false;
      if(parent.getDroppedDims().test(cast<RankedTensorType>(parent.getSource().getType()).getRank()-1)) return false;
      views.push_back(parent);source=parent.getSource();continue;
    }
    if(auto collapse=source.getDefiningOp<tensor::CollapseShapeOp>()) {
      if(!groupsSafe(cast<RankedTensorType>(collapse.getSrc().getType()),collapse.getReassociationIndices())) return false;
      views.push_back(collapse);source=collapse.getSrc();continue;
    }
    if(auto expand=source.getDefiningOp<tensor::ExpandShapeOp>()) {
      if(!groupsSafe(cast<RankedTensorType>(expand.getResult().getType()),expand.getReassociationIndices())) return false;
      views.push_back(expand);source=expand.getSrc();continue;
    }
    break;
  }
  auto root=source.getDefiningOp<scf::ForOp>();Copy copy;if(!root || !prove(root,copy)) return false;
  OpBuilder builder(slice);auto loc=slice.getLoc();
  auto constant=[&](int64_t n)->Value{return arith::ConstantIndexOp::create(builder,loc,n);};
  auto value=[&](OpFoldResult v)->Value {if(auto x=dyn_cast<Value>(v)) return x;return constant(cast<IntegerAttr>(cast<Attribute>(v)).getInt());};
  SmallVector<Value> indices;for(auto offset:slice.getMixedOffsets()) indices.push_back(value(offset));
  for(Operation *op:views) {
    SmallVector<Value> next;
    if(auto parent=dyn_cast<tensor::ExtractSliceOp>(op)) {
      auto dropped=parent.getDroppedDims();unsigned current=0;
      for(auto item:llvm::enumerate(parent.getMixedOffsets())) {
        Value offset=value(item.value());
        next.push_back(dropped.test(item.index())?offset:Value(arith::AddIOp::create(builder,loc,offset,indices[current++])));
      }
    } else if(auto collapse=dyn_cast<tensor::CollapseShapeOp>(op)) {
      auto type=cast<RankedTensorType>(collapse.getSrc().getType());
      for(auto item:llvm::enumerate(collapse.getReassociationIndices())) {
        auto group=item.value();Value flat=indices[item.index()];
        if(group.size()==1) {next.push_back(flat);continue;}
        for(unsigned i=0;i<group.size();++i) {
          int64_t stride=1;for(unsigned j=i+1;j<group.size();++j) stride*=type.getDimSize(group[j]);
          Value coordinate=flat;if(stride!=1) coordinate=arith::DivUIOp::create(builder,loc,flat,constant(stride));
          if(i) coordinate=arith::RemUIOp::create(builder,loc,coordinate,constant(type.getDimSize(group[i])));next.push_back(coordinate);
        }
      }
    } else {
      auto expand=cast<tensor::ExpandShapeOp>(op);auto type=cast<RankedTensorType>(expand.getResult().getType());
      for(auto group:expand.getReassociationIndices()) {
        Value flat=indices[group.front()];
        for(unsigned i=1;i<group.size();++i) {Value scaled=arith::MulIOp::create(builder,loc,flat,constant(type.getDimSize(group[i])));flat=arith::AddIOp::create(builder,loc,scaled,indices[group[i]]);}
        next.push_back(flat);
      }
    }
    indices=std::move(next);
  }
  if(indices.size()!=4) return false;
  IRMapping mapping;for(unsigned i=0;i<3;++i) mapping.map(copy.loops[i].getInductionVar(),indices[i]);
  for(auto item:llvm::enumerate(copy.scalar.getDpsInputs())) mapping.map(copy.scalar.getBody()->getArgument(item.index()),tensor::ExtractOp::create(builder,loc,item.value(),ValueRange{}).getResult());
  for(Operation &op:*copy.scalar.getBody()) {
    if(auto index=dyn_cast<linalg::IndexOp>(op)) mapping.map(index.getResult(),indices[3]);
    else if(!isa<linalg::YieldOp,tensor::ExtractOp>(op)) builder.clone(op,mapping);
  }
  SmallVector<OpFoldResult> offsets;for(Value index:copy.load.getIndices()) offsets.push_back(mapping.lookupOrDefault(index));
  SmallVector<OpFoldResult> sizes(5,builder.getIndexAttr(1)),ones(5,builder.getIndexAttr(1));sizes.back()=builder.getIndexAttr(span);
  auto replacement=tensor::ExtractSliceOp::create(builder,loc,result,copy.load.getTensor(),offsets,sizes,ones);
  slice.getResult().replaceAllUsesWith(replacement.getResult());slice.erase();producer=root;return true;
}
#ifdef QWEN_GQA_DEAD_REPEAT
// Only an already proved, tensor-only repeat producer may be erased. Ordinary
// SCF DCE conservatively retains its tensor reads after forwarding its users.
// Authenticate the entire remaining result-use closure before any deletion.
bool eraseDeadRepeat(Operation *producer,unsigned &viewCount) {
  auto root=dyn_cast<scf::ForOp>(producer);Copy copy;
  if(!root || !prove(root,copy)) return false;
  SmallVector<std::pair<Value,unsigned>> pending{{root.getResult(0),0}};
  SmallVector<Operation *> views;
  llvm::SmallPtrSet<Operation *,32> seen;
  while(!pending.empty()) {
    auto item=pending.pop_back_val();
    if(item.second>16 || seen.size()>256) return false;
    for(Operation *user:item.first.getUsers()) {
      Value source,result;
      if(auto op=dyn_cast<tensor::ExpandShapeOp>(user)) {source=op.getSrc();result=op.getResult();}
      else if(auto op=dyn_cast<tensor::CollapseShapeOp>(user)) {source=op.getSrc();result=op.getResult();}
      else if(auto op=dyn_cast<tensor::CastOp>(user)) {source=op.getSource();result=op.getResult();}
      else return false;
      if(source!=item.first) return false;
      if(seen.insert(user).second) {views.push_back(user);pending.push_back({result,item.second+1});}
    }
  }
  // A closed tensor shape-use DAG has no observable reads or output escapes.
  // Erase only its dead leaves, then the proved scalar copy itself.
  while(!views.empty()) {
    bool changed=false;
    for(unsigned i=views.size();i>0;--i) if(views[i-1]->use_empty()) {
      views[i-1]->erase();views.erase(views.begin()+i-1);++viewCount;changed=true;
    }
    if(!changed) return false;
  }
  if(!root->use_empty()) return false;
  root.erase();return true;
}
#endif
} // namespace

unsigned forwardQwenGQARepeatSlices(mlir::func::FuncOp function) {
  using namespace mlir;SmallVector<tensor::ExtractSliceOp> slices;function.walk([&](tensor::ExtractSliceOp slice){slices.push_back(slice);});
  unsigned changed=0;llvm::SmallPtrSet<Operation *,32> copies;
  for(auto slice:slices) {Operation *producer=nullptr;if(redirect(slice,producer)){++changed;copies.insert(producer);}}
  if(changed) llvm::errs()<<"QWEN_GQA_REPEAT_SLICE_FORWARD slices="<<changed<<" copies="<<copies.size()<<" FP-arithmetic-changed=0 vector-load-and-FMA-order-retained=1\n";
  return changed;
}

#ifdef QWEN_GQA_DEAD_REPEAT
unsigned eraseQwenDeadGQARepeats(mlir::func::FuncOp function) {
  using namespace mlir;SmallVector<Operation *> candidates;
  function.walk([&](scf::ForOp root) {
    Copy copy;if(prove(root,copy)) candidates.push_back(root);
  });
  unsigned erased=0,views=0;
  for(Operation *producer:candidates) erased+=eraseDeadRepeat(producer,views);
  if(erased) llvm::errs()<<"QWEN_GQA_DEAD_REPEAT erased="<<erased<<" shape-views="<<views<<" tensor-only-copy-proved=1 remaining-use-closure-proved=1 FP-arithmetic-changed=0\n";
  return erased;
}
#endif
