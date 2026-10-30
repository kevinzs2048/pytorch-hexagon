// SPDX-License-Identifier: BSD-3-Clause
// Keep the contiguous K transpose but store only its8 distinct KV heads.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

namespace {
using namespace mlir;
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
struct Read {tensor::ExtractOp extract;SmallVector<Operation *> views;};
bool collect(Value value,SmallVector<Operation *> views,SmallVector<Read> &reads,
             SmallVector<tensor::DimOp> &dimensions,unsigned depth=0,
             llvm::SmallPtrSetImpl<Operation *> *admittedViews=nullptr) {
  if (depth>8) return false;
  for (Operation *user:value.getUsers()) {
    if (auto extract=dyn_cast<tensor::ExtractOp>(user)) {
      if (extract.getTensor()!=value) return false;reads.push_back({extract,views});
    } else if (auto dim=dyn_cast<tensor::DimOp>(user)) {
      auto axis=dim.getIndex().getDefiningOp<arith::ConstantIndexOp>();
      auto type=cast<RankedTensorType>(value.getType());
      if (!axis || axis.value()<0 || axis.value()>=type.getRank()) return false;dimensions.push_back(dim);
    } else if (auto expand=dyn_cast<tensor::ExpandShapeOp>(user)) {
      if (expand.getSrc()!=value || !staticGroups(cast<RankedTensorType>(expand.getResult().getType()),expand.getReassociationIndices())) return false;
      if(admittedViews) admittedViews->insert(expand);
      auto nested=views;nested.insert(nested.begin(),expand);
      if (!collect(expand.getResult(),nested,reads,dimensions,depth+1,admittedViews)) return false;
    } else if (auto collapse=dyn_cast<tensor::CollapseShapeOp>(user)) {
      if (collapse.getSrc()!=value || !staticGroups(cast<RankedTensorType>(collapse.getSrc().getType()),collapse.getReassociationIndices())) return false;
      if(admittedViews) admittedViews->insert(collapse);
      auto nested=views;nested.insert(nested.begin(),collapse);
      if (!collect(collapse.getResult(),nested,reads,dimensions,depth+1,admittedViews)) return false;
    } else return false;
  }
  return true;
}
void redirectReads(Value compact,Value n,ArrayRef<Read> reads,
                   ArrayRef<tensor::DimOp> dimensions,OpBuilder &builder,Location loc,
                   Value directSource={},Value directFlag={},
                   ArrayRef<Operation *> admittedViews={}) {
  for (auto read:reads) {
    builder.setInsertionPoint(read.extract);SmallVector<Value> indices(read.extract.getIndices());
    auto constant=[&](int64_t v)->Value {return arith::ConstantIndexOp::create(builder,loc,v);};
    for (Operation *view:read.views) {
      SmallVector<Value> next;
      if (auto collapse=dyn_cast<tensor::CollapseShapeOp>(view)) {
        auto source=cast<RankedTensorType>(collapse.getSrc().getType());
        for (auto item:llvm::enumerate(collapse.getReassociationIndices())) {
          auto group=item.value();Value flat=indices[item.index()];
          if (group.size()==1) {next.push_back(flat);continue;}
          for (unsigned i=0;i<group.size();++i) {
            int64_t stride=1;for (unsigned j=i+1;j<group.size();++j) stride*=source.getDimSize(group[j]);
            Value coordinate=flat;
            if (stride!=1) coordinate=arith::DivUIOp::create(builder,loc,flat,constant(stride));
            if (i) coordinate=arith::RemUIOp::create(builder,loc,coordinate,constant(source.getDimSize(group[i])));
            next.push_back(coordinate);
          }
        }
      } else {
        auto expand=cast<tensor::ExpandShapeOp>(view);auto result=cast<RankedTensorType>(expand.getResult().getType());
        for (auto group:expand.getReassociationIndices()) {
          Value flat=indices[group.front()];
          for (unsigned i=1;i<group.size();++i) {
            Value scaled=arith::MulIOp::create(builder,loc,flat,constant(result.getDimSize(group[i])));
            flat=arith::AddIOp::create(builder,loc,scaled,indices[group[i]]);
          }
          next.push_back(flat);
        }
      }
      indices=std::move(next);
    }
    tensor::ExtractOp load;
    if (directSource) {
      // The proved SCF body is exactly source[0, head, 0,
      // select(flag, 0, position), component]. Reconstruct that source
      // coordinate at each admitted scalar read; no arithmetic is cloned.
      auto flag=tensor::ExtractOp::create(builder,loc,directFlag,ValueRange{});
      Value position=arith::SelectOp::create(builder,loc,flag.getResult(),constant(0),indices[3]);
      load=tensor::ExtractOp::create(builder,loc,directSource,
          ValueRange{constant(0),indices[0],constant(0),position,indices[2]});
    } else {
      load=tensor::ExtractOp::create(builder,loc,compact,ValueRange{indices[0],indices[2],indices[3]});
    }
    read.extract.getResult().replaceAllUsesWith(load.getResult());read.extract.erase();
  }
  for (auto dim:dimensions) {
    builder.setInsertionPoint(dim);auto source=cast<RankedTensorType>(dim.getSource().getType());
    int64_t axis=dim.getIndex().getDefiningOp<arith::ConstantIndexOp>().value();Value size=n;
    if (!source.isDynamicDim(axis)) size=arith::ConstantIndexOp::create(builder,loc,source.getDimSize(axis));
    dim.getResult().replaceAllUsesWith(size);dim.erase();
  }
  // Remove only shape views after every admitted read and dimension is gone.
  llvm::SmallPtrSet<Operation *,32> views;
  for(Operation *view:admittedViews) views.insert(view);
  for(auto read:reads) for(auto view:read.views) views.insert(view);
  bool changed=true;
  while(changed) {
    SmallVector<Operation *> dead;
    for(Operation *view:views) if(view->use_empty()) dead.push_back(view);
    changed=!dead.empty();
    for(Operation *view:dead) {views.erase(view);view->erase();}
  }
}

bool compact(linalg::GenericOp original) {
  if (original.getNumResults()!=1 || original.getNumDpsInputs() || original.getNumDpsInits()!=1 ||
      original.getNumLoops()!=4 || original.getNumReductionLoops() || !original.getIndexingMapsArray().back().isIdentity()) return false;
  auto type=dyn_cast<RankedTensorType>(original.getResult(0).getType());
  if (!type || type.getEncoding() || !type.getElementType().isF32() || type.getShape()!=ArrayRef<int64_t>({8,2,128,ShapedType::kDynamic}) ||
      !original.getBody()->getArgument(0).use_empty() ||
      !llvm::all_of(original.getIteratorTypesArray(),[](utils::IteratorType t){return t==utils::IteratorType::parallel;})) return false;
  auto empty=original.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
  if (!empty || empty.getDynamicSizes().size()!=1) return false;
  auto yield=cast<linalg::YieldOp>(original.getBody()->getTerminator());
  if (yield.getValues().size()!=1 || !yield.getValues()[0].getDefiningOp<tensor::ExtractOp>()) return false;
  for (Operation &op:*original.getBody()) {
    if (!isa<linalg::IndexOp,linalg::YieldOp,tensor::ExtractOp,
             arith::ConstantIndexOp,arith::CmpIOp,arith::SelectOp>(op)) return false;
    if (auto index=dyn_cast<linalg::IndexOp>(op))
      if (index.getDim()==1 && !index.getResult().use_empty()) return false;
  }
  SmallVector<Read> reads;SmallVector<tensor::DimOp> dimensions;
  if (!collect(original.getResult(0),{},reads,dimensions) || reads.empty()) return false;
  Value n=empty.getDynamicSizes()[0];OpBuilder builder(original);Location loc=original.getLoc();
  auto compactType=RankedTensorType::get({8,128,ShapedType::kDynamic},type.getElementType());
  auto storage=tensor::EmptyOp::create(builder,loc,compactType.getShape(),type.getElementType(),ValueRange{n});
  SmallVector<AffineMap> maps{builder.getMultiDimIdentityMap(3)};
  SmallVector<utils::IteratorType> iterators(3,utils::IteratorType::parallel);
  auto copy=linalg::GenericOp::create(builder,loc,TypeRange{compactType},ValueRange{},ValueRange{storage.getResult()},maps,iterators,
      [&](OpBuilder &body,Location bodyLoc,ValueRange args) {
        IRMapping mapping;mapping.map(original.getBody()->getArgument(0),args[0]);
        for (Operation &op:*original.getBody()) {
          if (auto index=dyn_cast<linalg::IndexOp>(op)) {
            if (index.getDim()==1) continue;
            auto dimension=linalg::IndexOp::create(body,bodyLoc,index.getDim()==0 ? 0 : index.getDim()-1);
            mapping.map(index.getResult(),dimension.getResult());
          } else body.clone(op,mapping);
        }
      });
  for (auto attribute:original->getDiscardableAttrs()) copy->setDiscardableAttr(attribute.getName(),attribute.getValue());
  redirectReads(copy.getResult(0),n,reads,dimensions,builder,loc);
  if(original->use_empty()) original.erase();
  llvm::errs()<<"QWEN_GQA_KEY_COMPACT heads=8 repeats=2 contiguous-N-retained FP-arithmetic-changed=0 reads="<<reads.size()<<"\n";
  return true;
}

bool integer(Value value,int64_t n) {
  auto c=value.getDefiningOp<arith::ConstantIndexOp>();return c && c.value()==n;
}
bool integer(OpFoldResult value,int64_t n) {
  if(auto attr=dyn_cast<Attribute>(value)) {auto i=dyn_cast<IntegerAttr>(attr);return i && i.getInt()==n;}
  return integer(cast<Value>(value),n);
}
bool yielded(scf::ForOp loop,Value value) {
  auto y=dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  return y && y.getNumOperands()==1 && y.getOperand(0)==value;
}
struct SCFCopy {
  SmallVector<scf::ForOp> loops;
  linalg::GenericOp scalar;
  tensor::ExtractSliceOp slice;
  tensor::InsertSliceOp insert;
  affine::AffineMinOp tail;
  Value n;
};
bool proveSCF(scf::ForOp root,SCFCopy &copy) {
  if(root.getInitArgs().size()!=1 || root.getNumResults()!=1) return false;
  auto type=dyn_cast<RankedTensorType>(root.getResult(0).getType());
  if(!type || type.getEncoding() || !type.getElementType().isF32() ||
      type.getShape()!=ArrayRef<int64_t>({8,2,128,ShapedType::kDynamic})) return false;
  auto empty=root.getInitArgs()[0].getDefiningOp<tensor::EmptyOp>();
  if(!empty || empty.getDynamicSizes().size()!=1) return false;
  Value n=empty.getDynamicSizes()[0];SmallVector<scf::ForOp> loops{root};
  for(unsigned i=0;i<3;++i) {
    auto body=loops.back().getBody();
    if(body->getOperations().size()!=2) return false;
    auto child=dyn_cast<scf::ForOp>(&body->front());if(!child) return false;
    loops.push_back(child);
  }
  for(unsigned i=0;i<4;++i) {
    auto loop=loops[i];
    if(loop.getNumResults()!=1 || loop.getInitArgs().size()!=1 ||
        loop.getResult(0).getType()!=type || !integer(loop.getLowerBound(),0) ||
        !integer(loop.getStep(),i==3?128:1) ||
        (i==3 ? loop.getUpperBound()!=n : !integer(loop.getUpperBound(),i==0?8:i==1?2:128))) return false;
    if(i && (loop.getInitArgs()[0]!=loops[i-1].getRegionIterArgs()[0] ||
        !yielded(loops[i-1],loop.getResult(0)))) return false;
  }
  auto inner=loops.back();if(inner.getBody()->getOperations().size()!=5) return false;
  affine::AffineMinOp tail;tensor::ExtractSliceOp slice;tensor::InsertSliceOp insert;linalg::GenericOp scalar;
  for(Operation &op:*inner.getBody()) {
    if(auto x=dyn_cast<affine::AffineMinOp>(op)) {if(tail) return false;tail=x;}
    else if(auto x=dyn_cast<tensor::ExtractSliceOp>(op)) {if(slice) return false;slice=x;}
    else if(auto x=dyn_cast<tensor::InsertSliceOp>(op)) {if(insert) return false;insert=x;}
    else if(auto x=dyn_cast<linalg::GenericOp>(op)) {if(scalar) return false;scalar=x;}
    else if(!isa<scf::YieldOp>(op)) return false;
  }
  if(!tail || !slice || !insert || !scalar || !yielded(inner,insert.getResult()) ||
      tail.getOperands()!=ValueRange{inner.getInductionVar(),n}) return false;
  auto map=tail.getAffineMap();auto context=root.getContext();
  auto d=getAffineDimExpr(0,context),s=getAffineSymbolExpr(0,context);
  if(map.getNumDims()!=1 || map.getNumSymbols()!=1 || map.getNumResults()!=2 ||
      !llvm::is_contained(map.getResults(),s-d) ||
      !llvm::is_contained(map.getResults(),getAffineConstantExpr(128,context))) return false;
  if(slice.getSource()!=inner.getRegionIterArgs()[0] || insert.getDest()!=inner.getRegionIterArgs()[0] ||
      scalar.getNumResults()!=1 || scalar.getNumDpsInputs()!=1 || scalar.getNumDpsInits()!=1 ||
      scalar.getDpsInits()[0]!=slice.getResult() || insert.getSource()!=scalar.getResult(0) ||
      scalar.getNumLoops()!=1 || scalar.getNumReductionLoops() ||
      scalar.getIteratorTypesArray()[0]!=utils::IteratorType::parallel ||
      !scalar.getIndexingMapsArray().back().isIdentity() || scalar.getIndexingMapsArray()[0].getNumResults() ||
      scalar.getBody()->getNumArguments()!=2 || !scalar.getBody()->getArgument(1).use_empty() ||
      slice.getMixedOffsets()!=insert.getMixedOffsets() || slice.getMixedSizes()!=insert.getMixedSizes() ||
      slice.getMixedStrides()!=insert.getMixedStrides()) return false;
  auto flag=dyn_cast<RankedTensorType>(scalar.getDpsInputs()[0].getType());
  if(!flag || flag.getEncoding() || flag.getRank()!=0 || !flag.getElementType().isInteger(1) ||
      !root.isDefinedOutsideOfLoop(scalar.getDpsInputs()[0])) return false;
  auto offsets=slice.getMixedOffsets(),sizes=slice.getMixedSizes(),strides=slice.getMixedStrides();
  if(offsets.size()!=4 || sizes.size()!=4 || strides.size()!=4 ||
      dyn_cast<Value>(sizes[3])!=tail.getResult() ||
      !llvm::all_of(strides,[](OpFoldResult v){return integer(v,1);})) return false;
  for(unsigned i=0;i<4;++i) if(dyn_cast<Value>(offsets[i])!=loops[i].getInductionVar() ||
      (i<3 && !integer(sizes[i],1))) return false;
  if(scalar.getBody()->getOperations().size()!=5) return false;
  linalg::IndexOp column;affine::AffineApplyOp position;arith::SelectOp select;tensor::ExtractOp load;
  for(Operation &op:*scalar.getBody()) {
    if(auto x=dyn_cast<linalg::IndexOp>(op)) {if(column) return false;column=x;}
    else if(auto x=dyn_cast<affine::AffineApplyOp>(op)) {if(position) return false;position=x;}
    else if(auto x=dyn_cast<arith::SelectOp>(op)) {if(select) return false;select=x;}
    else if(auto x=dyn_cast<tensor::ExtractOp>(op)) {if(load) return false;load=x;}
    else if(!isa<linalg::YieldOp>(op)) return false;
  }
  if(!column || column.getDim()!=0 || !position || !select || !load ||
      position.getAffineMap()!=AffineMap::get(1,1,d+s,context) ||
      position.getOperands()!=ValueRange{inner.getInductionVar(),column.getResult()} ||
      select.getCondition()!=scalar.getBody()->getArgument(0) || !integer(select.getTrueValue(),0) ||
      select.getFalseValue()!=position.getResult() || !root.isDefinedOutsideOfLoop(load.getTensor())) return false;
  auto source=dyn_cast<RankedTensorType>(load.getTensor().getType());
  auto yield=dyn_cast<linalg::YieldOp>(scalar.getBody()->getTerminator());
  if(!source || source.getEncoding() || !source.getElementType().isF32() ||
      source.getShape()!=ArrayRef<int64_t>({1,8,1,ShapedType::kDynamic,128}) ||
      load.getIndices().size()!=5 || !yield || yield.getValues().size()!=1 || yield.getValues()[0]!=load.getResult()) return false;
  auto indices=load.getIndices();
  if(!integer(indices[0],0) || indices[1]!=loops[0].getInductionVar() || !integer(indices[2],0) ||
      indices[3]!=select.getResult() || indices[4]!=loops[2].getInductionVar()) return false;
  // No use of the repeat IV remains anywhere in the copied scalar body.
  copy={loops,scalar,slice,insert,tail,n};return true;
}
bool compactSCF(scf::ForOp root) {
  SCFCopy copy;if(!proveSCF(root,copy)) return false;
  SmallVector<Read> reads;SmallVector<tensor::DimOp> dimensions;
#ifdef QWEN_GQA_KEY_SCALAR_FORWARD
  // Include every admitted shape view, including dimension-only and empty
  // branches, so the complete producer deletion follows from use closure.
  llvm::SmallPtrSet<Operation *,32> allViews;
  if(!collect(root.getResult(0),{},reads,dimensions,0,&allViews) || reads.empty()) return false;
  SmallVector<Operation *> admittedViews(allViews.begin(),allViews.end());
#else
  if(!collect(root.getResult(0),{},reads,dimensions) || reads.empty()) return false;
#endif
  OpBuilder builder(root);auto loc=root.getLoc();auto type=RankedTensorType::get({8,128,ShapedType::kDynamic},builder.getF32Type());
  auto storage=tensor::EmptyOp::create(builder,loc,type.getShape(),type.getElementType(),ValueRange{copy.n});
  IRMapping mapping;SmallVector<scf::ForOp> created;
#ifdef QWEN_GQA_KEY_TILE_N32
  // The admitted producer writes each (head, component, position) once to
  // fresh storage. Its body only selects an immutable source coordinate.
  // Partition positions into 32-element tiles; traverse all components
  // within a tile. The complete tensor and every consumer remain unchanged.
  unsigned order[] = {0u,3u,2u};
#else
  unsigned order[] = {0u,2u,3u};
#endif
  for(unsigned old: order) {
    auto original=copy.loops[old];Value initial=created.empty()?Value(storage.getResult()):Value(created.back().getRegionIterArgs()[0]);
    Value step=original.getStep();
#ifdef QWEN_GQA_KEY_TILE_N32
    if(old==3u) step=arith::ConstantIndexOp::create(builder,loc,32);
#endif
    auto loop=scf::ForOp::create(builder,loc,original.getLowerBound(),original.getUpperBound(),step,ValueRange{initial});
    mapping.map(original.getInductionVar(),loop.getInductionVar());created.push_back(loop);
    builder.setInsertionPointToStart(loop.getBody());
  }
#ifdef QWEN_GQA_KEY_TILE_N32
  auto context=builder.getContext();
  auto tailMap=AffineMap::get(1,1,{getAffineConstantExpr(32,context),
      getAffineSymbolExpr(0,context)-getAffineDimExpr(0,context)},context);
  auto amount=affine::AffineMinOp::create(builder,loc,tailMap,
      ValueRange{created[1].getInductionVar(),copy.n}).getResult();
  SmallVector<OpFoldResult> offsets{created[0].getInductionVar(),created[2].getInductionVar(),created[1].getInductionVar()};
#else
  auto amount=cast<affine::AffineMinOp>(builder.clone(*copy.tail.getOperation(),mapping)).getResult();
  SmallVector<OpFoldResult> offsets{created[0].getInductionVar(),created[1].getInductionVar(),created[2].getInductionVar()};
#endif
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1),builder.getIndexAttr(1),amount};
  SmallVector<OpFoldResult> strides(3,builder.getIndexAttr(1));
  auto slice=tensor::ExtractSliceOp::create(builder,loc,copy.slice.getType(),created.back().getRegionIterArgs()[0],offsets,sizes,strides);
  mapping.map(copy.slice.getResult(),slice.getResult());
  auto scalar=cast<linalg::GenericOp>(builder.clone(*copy.scalar.getOperation(),mapping));
  auto insert=tensor::InsertSliceOp::create(builder,loc,scalar.getResult(0),created.back().getRegionIterArgs()[0],offsets,sizes,strides);
  for(int i=2;i>=0;--i) {
    auto loop=created[i];builder.setInsertionPointToEnd(loop.getBody());
    if(!loop.getBody()->empty())
      if(auto oldYield=dyn_cast<scf::YieldOp>(&loop.getBody()->back())) oldYield.erase();
    scf::YieldOp::create(builder,loc,ValueRange{i==2?insert.getResult():created[i+1].getResult(0)});
  }
#ifdef QWEN_GQA_KEY_SCALAR_FORWARD
  // The source and flag are immutable tensor SSA values dominating the
  // original producer. collect() has already rejected escapes and all
  // non-scalar consumers. The old K/FMA consumers keep their operations.
  Value source;
  for (Operation &op:*copy.scalar.getBody())
    if(auto load=dyn_cast<tensor::ExtractOp>(op)) source=load.getTensor();
  redirectReads(created.front().getResult(0),copy.n,reads,dimensions,builder,loc,
      source,copy.scalar.getDpsInputs()[0],admittedViews);
#else
  redirectReads(created.front().getResult(0),copy.n,reads,dimensions,builder,loc);
#endif
#ifdef QWEN_GQA_KEY_SCALAR_FORWARD
  if(!root->use_empty()) llvm_unreachable("All original K reads, dimensions and shape views were proved redirected");
  root.erase();
#else
  if(root->use_empty()) root.erase();
#endif
  llvm::errs()<<"QWEN_GQA_KEY_COMPACT heads=8 repeats=2 contiguous-N-retained FP-arithmetic-changed=0 reads="<<reads.size()<<" SCF-producer-proved=1\n";
#ifdef QWEN_GQA_KEY_TILE_N32
  llvm::errs()<<"QWEN_GQA_KEY_TILE positions=32 head-position-component-order fresh-complete-destination-proved=1 FP-arithmetic-changed=0\n";
#endif
#ifdef QWEN_GQA_KEY_SCALAR_FORWARD
  if(!created.front()->use_empty()) llvm_unreachable("All K consumers were proved scalar and redirected");
  created.front().erase();
  llvm::errs()<<"QWEN_GQA_KEY_SCALAR_FORWARD immutable-source-index-equivalence-proved=1 scalar-consumers-only=1 removed-complete-K-copy=1 FP-arithmetic-changed=0 reads="<<reads.size()<<"\n";
#endif
  return true;
}
} // namespace

unsigned compactQwenGQAKeys(mlir::func::FuncOp function) {
  using namespace mlir;SmallVector<linalg::GenericOp> candidates;
  function.walk([&](linalg::GenericOp op){candidates.push_back(op);});
  unsigned changed=0;for (auto copy:candidates) changed+=compact(copy);
  SmallVector<scf::ForOp> roots;
  function.walk([&](scf::ForOp op) {
    if(op.getInitArgs().size()==1 && op.getInitArgs()[0].getDefiningOp<tensor::EmptyOp>()) roots.push_back(op);
  });
  for(auto root:roots) changed+=compactSCF(root);return changed;
}
