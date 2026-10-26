"""Check actual extracted QK fallback, original FMA body and guarded copies."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from build import ROOT,HERE,entry,write
from profile_model_memory import checked
from inspect_q2_dispatch_fixture import SSA,bindings,canonical
from revisit_native_compile_session_abba import require_normal_owner


def loops(text):
    lines=text.splitlines(keepends=True);result=[]
    for start,line in enumerate(lines):
        if '"scf.for"'not in line:continue
        depth=0
        for end in range(start,len(lines)):
            depth+=lines[end].count('{')-lines[end].count('}')
            if depth==0:
                result.append(''.join(lines[start:end+1]));break
        else:raise ValueError('Unclosed actual loop region')
    return result


def operands(line,operation):
    match=re.search('"'+re.escape(operation)+r'"\(([^)]*)\)',line)
    if match is None:raise ValueError('Missing actual operation '+operation)
    values=[x.strip()for x in match[1].split(',')if x.strip()]
    if any(re.fullmatch(SSA,x)is None for x in values):raise ValueError('Unknown operation operands')
    return values


def compute_loop(head,module):
    candidates=[x for x in loops(head)if x.count('"scf.for"')==2 and x.count('"math.fma"')==1]
    if len(candidates)!=1:raise ValueError('Exactly one original column/component loop required')
    body=candidates[0];header=body.splitlines()[0];args=operands(header,'scf.for')
    if len(args)!=4:raise ValueError('Original single tensor accumulator required')
    indices=re.findall(r'\^bb0\(('+SSA+r'): index, ('+SSA+r'): tensor<16x\?x\?xf32>\)',head)
    if len(indices)<3:raise ValueError('Original head/query/column tensor loops required')
    external=bindings(module);external.update({indices[0][0]:'HEAD_INDEX',indices[1][0]:'QUERY_INDEX',
        indices[1][1]:'QUERY_CARRY','%COLUMN_BOUND':'COLUMN_BOUND'})
    normalized=header.replace('"scf.for"('+', '.join(args)+')',
        '"scf.for"('+', '.join([args[0],'%COLUMN_BOUND',args[2],args[3]])+')')
    if normalized==header:raise ValueError('Column-bound normalization failed')
    body=normalized+body[len(header):]
    result=re.match(r'\s*('+SSA+r') = ',header)
    if result is None:raise ValueError('Column result binding missing')
    return canonical(body,external),args[1],result[1],indices


def require_tail(module,optimized,reference,reference_module):
    if module.count('"math.fma"')!=2 or module.count('"scf.if"')!=2 or '#arith.fastmath<fast>'in module:
        raise ValueError('One original fallback and one guarded ordered FMA branch required')
    for anchor in ['%9 = "arith.constant"() <{value = 8 : index}>',
            '%10 = "arith.constant"() <{value = 32 : index}>',
            '%11 = "arith.cmpi"(%7, %9) <{predicate = 9 : i64}>',
            '%12 = "arith.cmpi"(%7, %10) <{predicate = 7 : i64}>',
            '%13 = "arith.andi"(%11, %12)', '%14 = "scf.if"(%13)']:
        if module.count(anchor)!=1:raise ValueError('Original M8..32 selection changed')
    actual,upper,nresult,indices=compute_loop(optimized,module)
    original,_,_,_=compute_loop(reference,reference_module)
    if actual!=original:raise ValueError('Original register8 FMA body or operand/seed/mask binding changed')
    definitions={m[1]:m[2]for m in re.finditer(r'^\s*('+SSA+r') = (.+)$',module,re.M)}
    ext=bindings(module)
    def line(token):
        if token not in definitions:raise ValueError('Unknown guarded operand '+token)
        return definitions[token]
    def semantic(token):
        if token in ext:return ext[token]
        text=line(token)
        constant=re.search(r'"arith.constant"\(\) <\{value = (-?\d+) : index\}>',text)
        if constant:return ('constant',int(constant[1]))
        if '"tensor.dim"'in text:
            source,index=operands(text,'tensor.dim')
            if semantic(source)!='SEED':raise ValueError('Guard score extent comes from another tensor')
            value=semantic(index)
            if value not in [('constant',1),('constant',2)]:raise ValueError('Wrong seed dimension')
            return ('seed_dimension',value[1])
        if '"bufferization.to_buffer"'in text:
            values=operands(text,'bufferization.to_buffer')
            if len(values)!=1 or '<{read_only}>'not in text:raise ValueError('Read-only exact input views required')
            return ('buffer',semantic(values[0]))
        if '"func.call"'in text and 'callee = @qwen_q2_qk_positive_zero_extent' in text:
            values=operands(text,'func.call')
            if [semantic(x)for x in values]!=[('buffer','KEYS'),('buffer','SEED'),'SELECTOR']:
                raise ValueError('Guard must inspect the actual original keys, seed and selector')
            return 'PROVEN_EXTENT'
        raise ValueError('Unknown guard expression '+text)
    select=operands(line(upper),'arith.select')
    if len(select)!=3 or semantic(select[2])!='CAPACITY':raise ValueError('Fallback must retain original capacity')
    skip,represented,_=select;addition=operands(line(represented),'arith.addi')
    if [semantic(x)for x in addition]!=['PROVEN_EXTENT','LANES_8']:
        raise ValueError('Exactly one original eight-column representative tile required')
    extent=addition[0]
    def leaves(token,seen):
        if token in seen:raise ValueError('Repeated or cyclic guard condition')
        seen.add(token);text=line(token)
        if '"arith.andi"'in text:
            a,b=operands(text,'arith.andi');return leaves(a,seen)+leaves(b,seen)
        args=operands(text,'arith.cmpi');predicate=re.search(r'predicate = (\d+) : i64',text)
        if len(args)!=2 or predicate is None:raise ValueError('Unknown guard predicate')
        return [(int(predicate[1]),semantic(args[0]),semantic(args[1]))]
    expected=[(0,'CAPACITY',('constant',1024)),(0,'QUERY_EXTENT',('seed_dimension',1)),
        (0,'CAPACITY',('seed_dimension',2)),(6,'PROVEN_EXTENT','CAPACITY')]
    if leaves(skip,set())!=expected:raise ValueError('Exact capacity, row/column and proven-prefix conditions required')
    choices=re.findall(r'('+SSA+r') = "scf.if"\(('+SSA+r')\)',optimized)
    if len(choices)!=1 or choices[0][1]!=skip:raise ValueError('Copying must use precisely the same eligibility condition')
    tile_lines=[x for x in optimized.splitlines()if '"tensor.extract_slice"'in x and '-> tensor<8xf32>'in x]
    if len(tile_lines)!=1:raise ValueError('Exactly one completed representative tile must be copied')
    tile_line=tile_lines[0];tile=re.match(r'\s*('+SSA+r') = ',tile_line)[1]
    if operands(tile_line,'tensor.extract_slice')!=[nresult,indices[0][0],indices[1][0],extent]:
        raise ValueError('Representative tile must come from the completed original recurrence')
    for anchor in ['static_sizes = array<i64: 1, 1, 8>','static_strides = array<i64: 1, 1, 1>']:
        if anchor not in tile_line:raise ValueError('Representative shape/stride changed')
    copies=[x for x in loops(optimized)if x.count('"scf.for"')==1 and '"tensor.insert_slice"'in x and '"math.fma"'not in x]
    if len(copies)!=1:raise ValueError('Exactly one bit-copy loop required')
    copy=copies[0];args=operands(copy.splitlines()[0],'scf.for')
    if len(args)!=4 or args[0]!=upper or semantic(args[1])!='CAPACITY' or semantic(args[2])!='LANES_8' or args[3]!=nresult:
        raise ValueError('Copy loop must cover precisely the uncomputed tail')
    index,carry=re.findall(r'\^bb0\(('+SSA+r'): index, ('+SSA+r'): tensor<16x\?x\?xf32>\)',copy)[0]
    insert=[x for x in copy.splitlines()if '"tensor.insert_slice"'in x]
    if len(insert)!=1 or operands(insert[0],'tensor.insert_slice')!=[tile,carry,indices[0][0],indices[1][0],index]:
        raise ValueError('Only the completed tile bits may fill the selected original row')
    for anchor in ['static_sizes = array<i64: 1, 1, 8>','static_strides = array<i64: 1, 1, 1>']:
        if anchor not in insert[0]:raise ValueError('Bit-copy shape or stride changed')
    if re.search(r'\}, \{\s*"scf.yield"\('+re.escape(nresult)+r'\)',optimized) is None:
        raise ValueError('Noneligible path must return the complete original recurrence')
    return dict(compute_body_original_exact_except_guarded_column_upper=True,
        original128_FMA_order_and_seed_mask_bindings_preserved=True,
        guard_keys_seed_selector_and_all_four_conditions_verified=True,
        completed_representative8_and_exact_remaining_tail_copy_verified=True,
        upper=upper,skip=skip,extent=extent)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','SDK','SDK-controller','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text());c=read(a.config)
    if not a.worker:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path'])));return
    sdk=read(a.SDK);owner=read(a.SDK_controller);require_normal_owner(a.SDK_controller,owner['authorized_deadline_unix'])
    if (a.output.exists() or not a.output.is_absolute()
            or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
            or sdk['status']!='Q2_identical_tail_structural_SDK_STOP_BEFORE_MODEL_CODEGEN_NOT_target'
            or owner['source_revision']!=sdk['source_revision']
            or owner['command'][owner['command'].index('--output')+1]!=str(a.SDK.parent)
            or sdk['dispatch']!=entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp')
            or sdk['identical_tail']!=entry(compiler_tools()/'buffer_reuse/DynamicQKIdenticalTail.inc')
            or sdk['integer_guard']!=entry(compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent.c')):
        raise ValueError('Exact normally owned frozen identical-tail SDK required')
    base=read(checked(sdk['base_sdk']));fixture=checked(base['fixture']);original=fixture.read_text()
    mutations=[
        ('component_step','"scf.for"(%217, %12, %13, %arg14590)','"scf.for"(%217, %12, %two, %arg14590)'),
        ('column_step','"scf.for"(%217, %795, %12, %arg14592)','"scf.for"(%217, %795, %two, %arg14592)'),
        ('key_component','%arg14587, %arg14591, %arg14593, %34409','%arg14587, %arg14589, %arg14593, %34409'),
        ('query_component','%34412, %arg14591)','%34412, %arg14593)'),
        ('broadcast_row','"arith.select"(%797, %217, %arg14589)','"arith.select"(%797, %two, %arg14589)'),
        ('FMA_accumulator','"math.fma"(%34418, %34416, %34417)','"math.fma"(%34418, %34416, %34416)'),
        ('FMA_query','"math.fma"(%34418, %34416, %34417)','"math.fma"(%34417, %34416, %34417)'),
        ('fast_FMA','#arith.fastmath<none>','#arith.fastmath<fast>'),
        ('tail_extent','(-d0 + s0, 128)','(-d0 + s0, 64)'),
        ('mask_extent','"vector.create_mask"(%34414)','"vector.create_mask"(%795)')]
    cases=[('original',original,True),('SSA_renamed',original.replace('%arg','%local'),True)]
    for name,before,after in mutations:
        if original.count(before)!=1:raise ValueError('Actual mutation anchor not unique: '+name)
        cases.append((name,original.replace(before,after),False))
    library=ctypes.CDLL(str(checked(sdk['candidate'])));transform=library.qwenQ2QKDispatchPhysicalPreflight
    reference=library.qwenQ2QKRegisterReference
    for function in (transform,reference):function.argtypes=[ctypes.c_char_p,ctypes.c_char_p];function.restype=ctypes.c_int
    a.output.mkdir();rows=[]
    with (a.output/'diagnostics.txt').open('xb')as log:
        previous=os.dup(2);os.dup2(log.fileno(),2)
        try:
            for name,text,expected in cases:
                source=a.output/(name+'.input.mlir');target=a.output/(name+'.output.mlir');source.write_text(text)
                rc=transform(os.fsencode(source),os.fsencode(target))
                if (rc==0)!=expected:raise ValueError('Actual structural admission disagrees: '+name)
                row=dict(name=name,input=entry(source),accepted=rc==0,returncode=rc)
                if expected:
                    ref=a.output/(name+'.register-reference.txt')
                    if reference(os.fsencode(source),os.fsencode(ref)):raise ValueError('Original register reference failed')
                    module=target.read_text();before=Path(str(target)+'.original-head.txt');fallback=Path(str(target)+'.fallback-head.txt')
                    optimized=Path(str(target)+'.optimized-head.txt');refmodule=Path(str(ref)+'.module.mlir')
                    if canonical(before.read_text(),bindings(module))!=canonical(fallback.read_text(),bindings(module)):
                        raise ValueError('Original M2..7 fallback changed')
                    proof=require_tail(module,optimized.read_text(),ref.read_text(),refmodule.read_text())
                    row.update(output=entry(target),original=entry(before),fallback=entry(fallback),
                        optimized=entry(optimized),reference=entry(ref),reference_module=entry(refmodule),
                        fallback_original_exact=True,proof=proof)
                rows.append(row)
        finally:os.dup2(previous,2);os.close(previous)
    write(a.output/'result.json',dict(status='actual_Q2_identical_tail_fixture_original_FMA_guard_and_copy_verified_HOST_ONLY_NOT_target',
        source_root=str(ROOT),source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source=entry(__file__),SDK=entry(a.SDK),SDK_controller=entry(a.SDK_controller),fixture=entry(fixture),
        cases=rows,positive=2,unsafe_rejected=10,model_PRIMARY_compilations=0,DSP_calls=0,
        model_inference_acceptance=False,performance_acceptance=False,completed_unix=time.time()))


if __name__=='__main__':main()
