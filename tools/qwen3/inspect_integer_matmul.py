"""Compile tiny widened/mixed INT8 matmuls with the pinned existing backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def module(size,columns,widen,packed=False,named=False,explicit=False):
    a=f'tensor<1x{size}xi8>';b=f'tensor<{size}x{columns}xi8>';c=f'tensor<1x{columns}xi32>'
    if explicit:
        ma=f'memref<1x{size}xi8>';mb=f'memref<{size}x{columns}xi8>';mc=f'memref<1x{columns}xi32>'
        da='memref<?x?xi8, strided<[?, ?], offset: ?>>'
        dc='memref<?x?xi32, strided<[?, ?], offset: ?>>'
        return '\n'.join([
            'module {',
            f'func.func private @__qwen_i8_matmul({da}, {da}, {dc}) attributes {{llvm.emit_c_interface}}',
            f'func.func @probe(%a: {a}, %b: {b}) -> {c} attributes {{llvm.emit_c_interface}} {{',
            f'%ab = bufferization.to_buffer %a read_only : {a} to {ma}',
            f'%bb = bufferization.to_buffer %b read_only : {b} to {mb}',
            f'%cb = memref.alloc() : {mc}',
            '%zero = arith.constant 0 : i32',
            f'linalg.fill ins(%zero : i32) outs(%cb : {mc})',
            f'%ad = memref.cast %ab : {ma} to {da}',
            f'%bd = memref.cast %bb : {mb} to {da}',
            f'%cd = memref.cast %cb : {mc} to {dc}',
            f'func.call @__qwen_i8_matmul(%ad, %bd, %cd) : ({da}, {da}, {dc}) -> ()',
            f'%result = bufferization.to_tensor %cb restrict writable : {mc} to {c}',
            f'return %result : {c}', '}}', ''])
    lines=[f'module {{ func.func @probe(%a: {a}, %b: {b}) -> {c} attributes {{llvm.emit_c_interface}} {{',
        '%zero = arith.constant 0 : i32',f'%empty = tensor.empty() : {c}',
        f'%init = linalg.fill ins(%zero : i32) outs(%empty : {c}) -> {c}']
    left,right,atypes,btypes='%a','%b',a,b
    if widen:
        for name,type8,rows,cols in (('a',a,1,size),('b',b,size,columns)):
            type32=f'tensor<{rows}x{cols}xi32>'
            lines += [f'%{name}_empty = tensor.empty() : {type32}',
                f'%{name}_wide = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}} ins(%{name} : {type8}) outs(%{name}_empty : {type32}) {{',
                '^bb0(%v: i8, %unused: i32):', '%extended = arith.extsi %v : i8 to i32',
                'linalg.yield %extended : i32',f'}} -> {type32}']
        left,right='%a_wide','%b_wide';atypes=f'tensor<1x{size}xi32>';btypes=f'tensor<{size}x{columns}xi32>'
    if packed and named:
        lines += [f'%result = linalg.matmul {{library_call = "__qwen_i8_vecmat"}} ins(%a, %b : {a}, {b}) outs(%init : {c}) -> {c}']
    elif packed:
        lines += [f'%result = linalg.generic {{library_call = "__qwen_i8_matmul", indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%a, %b : {a}, {b}) outs(%init : {c}) {{',
            '^bb0(%left: i8, %right: i8, %initial: i32):',
            '%left32 = arith.extsi %left : i8 to i32',
            '%right32 = arith.extsi %right : i8 to i32',
            '%product = arith.muli %left32, %right32 : i32',
            '%sum = arith.addi %initial, %product : i32',
            'linalg.yield %sum : i32',f'}} -> {c}']
    else:
        lines += [f'%result = linalg.matmul ins({left}, {right} : {atypes}, {btypes}) outs(%init : {c}) -> {c}']
    lines += [f'return %result : {c}','}}']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=1024);p.add_argument('--columns',type=int,default=17)
    p.add_argument('--mode',choices=('widened','mixed','packed','packed-named','packed-explicit'),required=True)
    p.add_argument('--disable-fusion',action='store_true',help='Isolated call-survival experiment, not a model option change')
    p.add_argument('--input-module',type=Path)
    p.add_argument('--input-module-sha256')
    args=p.parse_args()
    if (not args.output.is_absolute() or args.output.exists() or not 1<=args.size<=131071
            or not 1<=args.columns<=151936 or args.size*(args.columns+1)>256*1024**2):
        p.error('Fresh output and bounded exact integer primitive shapes required')
    if (bool(args.input_module)!=bool(args.input_module_sha256)
            or (args.input_module and (args.mode!='packed-explicit' or sha(args.input_module)!=args.input_module_sha256))):
        p.error('External tiny IR requires explicit-call mode and its exact identity')
    config=json.loads(args.config.read_text())
    sys.path[:0]=[str(ROOT/'qwen3/runtime/arm'),str(ROOT)]
    from hexagon_python_source import configure_triton_python
    from compile_qwen3_artifact import load_extension
    from build import verify_dependencies
    lock=Path(__file__).with_name('dependencies.lock.json')
    verified=verify_dependencies(config,json.loads(lock.read_text()))
    configure_triton_python(Path(config['triton_python']));load_extension(Path(config['compiler']))
    from triton._C.libtriton import ir,qcom_hexagon_backend as backend
    from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
    raw=json.loads(Path(__file__).with_name('options_result.json').read_text())['options']
    defaults=HexagonOptions().__dict__
    if set(defaults)-set(raw)!={'outlineCheckedVectorExp'} or set(raw)-set(defaults)!={'keepAssertMessagesInMain'}:
        raise ValueError('Existing backend option schema changed')
    options={key:str(value) for key,value in raw.items()}
    options['outlineCheckedVectorExp']=str(defaults['outlineCheckedVectorExp'])
    if args.disable_fusion:options['fusion']='False'
    args.output.mkdir(parents=True)
    input_path=args.output/'input.mlir'
    if args.input_module:input_path.write_bytes(args.input_module.read_bytes())
    else:input_path.write_text(module(args.size,args.columns,args.mode=='widened',args.mode in ('packed','packed-named'),args.mode=='packed-named',args.mode=='packed-explicit'))
    record=dict(status='compiling_integer_primitive_NOT_model',model_inference=False,
        kind=args.mode,size=args.size,columns=args.columns,compiler_sha256=sha(config['compiler']),
        input_sha256=sha(input_path),source_sha256=sha(__file__),options=options,
        SDK_verification=dict(files=len(verified),lock_sha256=sha(lock)))
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    previous_cwd=Path.cwd();os.chdir(args.output)
    try:
        context=ir.context();backend.load_dialects(context)
        parsed=backend.parse_mlir_module_from_file(str(input_path),context)
        start=time.monotonic();objects=backend.translate_linalg_to_obj(parsed,options)
        record['elapsed_seconds']=time.monotonic()-start
        record['objects']=[]
        for index,data in enumerate(objects):
            target=args.output/('primitive-'+str(index)+'.o');target.write_bytes(bytes(data))
            record['objects'].append(dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size))
        nm=Path(config['hexagon_tools'])/'bin/llvm-nm'
        symbols=subprocess.run([str(nm),'-u',*[row['path'] for row in record['objects']]],
            check=True,capture_output=True,text=True,timeout=30).stdout
        (args.output/'undefined-symbols.txt').write_text(symbols)
        record['runtime_call_import_present']=any(line.split()[-1:] == ['_mlir_ciface___qwen_i8_matmul']
            for line in symbols.splitlines())
        if args.mode=='packed-explicit' and not record['runtime_call_import_present']:
            raise ValueError('Explicit packed runtime call was removed during compilation')
        record['status']='integer_primitive_objects_built_NOT_device_or_model_acceptance'
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        os.chdir(previous_cwd);(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record['status'])


if __name__=='__main__':main()
