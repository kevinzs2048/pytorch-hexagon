// SPDX-License-Identifier: BSD-3-Clause
// Immutable weight byte permutation, paired with the prepacked HVX runtime.
struct PrepackInt8WeightsPass
    : PassWrapper<PrepackInt8WeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepackInt8WeightsPass)
  PrepackInt8WeightsPass() = default;
  PrepackInt8WeightsPass(const PrepackInt8WeightsPass &p) : PassWrapper(p) {}
  Option<unsigned> expectedCount{*this,"expected-count",llvm::cl::init(0)};
  StringRef getArgument() const final { return "hexagon-prepack-i8-weights"; }
  StringRef getDescription() const final { return "Pack immutable INT8 weights once for HVX; preserve integer values"; }
  void runOnOperation() final {
    auto module=getOperation();
    IRRewriter rw(module.getContext());
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp c){
      if(c.getCallee()=="__qwen_i8_multi_matmul")calls.push_back(c);
    });
    unsigned changed=0;
    for(auto call:calls) {
      if(call.getNumOperands()!=3)continue;
      Value b=call.getOperand(1);
      if(auto cast=b.getDefiningOp<memref::CastOp>())b=cast.getSource();
      auto buffer=b.getDefiningOp<bufferization::ToBufferOp>();
      if(!buffer)continue;
      auto transpose=buffer.getTensor().getDefiningOp<linalg::TransposeOp>();
      if(!transpose || transpose.getPermutation()!=ArrayRef<int64_t>({1,0}))continue;
      auto constant=transpose.getInput().getDefiningOp<arith::ConstantOp>();
      if(!constant)continue;
      auto type=dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
      if(!type || type.getRank()!=2 || !type.hasStaticShape() || !type.getElementType().isSignlessInteger(8))continue;
      int64_t k=type.getDimSize(0),n=type.getDimSize(1);
      if(k<1 || k>131071 || k%4 || n<128 || n>151936 || n%128)continue;
      ArrayRef<char> bytes;
      if(auto resource=dyn_cast<DenseResourceElementsAttr>(constant.getValue()))bytes=resource.getData();
      else if(auto dense=dyn_cast<DenseElementsAttr>(constant.getValue())) {
        if(dense.isSplat())continue;
        bytes=dense.getRawData();
      } else continue;
      if(bytes.size()!=size_t(k*n))continue;
      std::vector<char> packed(k*n);
      for(int64_t col=0;col<n;col+=128)
        for(int64_t red=0;red<k;red+=4)
          for(int64_t j=0;j<128;++j)
            for(int64_t r=0;r<4;++r)
              packed[(col/128)*k*128+(red/4)*512+j*4+r]=bytes[(col+j)*k+red+r];
      rw.setInsertionPoint(call);
      auto data=DenseElementsAttr::getFromRawBuffer(type,ArrayRef<char>(packed));
      Value value=rw.create<arith::ConstantOp>(call.getLoc(),type,data);
      auto memref=MemRefType::get(type.getShape(),type.getElementType());
      Value packedBuffer=rw.create<bufferization::ToBufferOp>(call.getLoc(),memref,value,true);
      Value cast=rw.create<memref::CastOp>(call.getLoc(),call.getOperand(1).getType(),packedBuffer);
      rw.modifyOpInPlace(call,[&]{
        call->setOperand(1,cast);
        call.setCallee("__hexagon_i8_prepacked_v1_matmul");
      });
      ++changed;
    }
    if(expectedCount && changed!=expectedCount) {
      module.emitError()<<"prepacked weight coverage: expected "<<expectedCount.getValue()<<", got "<<changed;
      return signalPassFailure();
    }
    if(changed && !module.lookupSymbol("__hexagon_i8_prepacked_v1_matmul")) {
      auto old=module.lookupSymbol<func::FuncOp>("__qwen_i8_multi_matmul");
      rw.setInsertionPointToStart(module.getBody());
      auto fn=rw.create<func::FuncOp>(module.getLoc(),"__hexagon_i8_prepacked_v1_matmul",old.getFunctionType());
      fn.setPrivate();fn->setAttr("llvm.emit_c_interface",rw.getUnitAttr());
    }
    llvm::errs()<<"Hexagon immutable prepacked weights: "<<changed<<"\n";
  }
};
