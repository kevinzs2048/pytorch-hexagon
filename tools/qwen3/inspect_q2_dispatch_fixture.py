"""Check real runtime branches against the unchanged original/proven register IR."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from build import ROOT, HERE, entry, write
from profile_model_memory import checked
from q2_dispatch_contract import require_proven_qk
from revisit_native_compile_session_abba import require_normal_owner


SSA = r'%[A-Za-z0-9_$.-]+'


def bindings(module):
    # Outer operands retain their meaning during renumbering. Local SSA values
    # may be renamed; different external keys/query/seed operands may not.
    expected = [('0', 'index'), ('1', 'index'), ('16', 'index'), ('128', 'index'),
                ('0.000000e+00', 'f32'), ('2', 'index')]
    result = {'%arg0':'KEYS','%arg1':'QUERY','%arg2':'SEED','%arg3':'SELECTOR_TENSOR'}
    for number, (value, kind) in enumerate(expected):
        anchor = f'%{number} = "arith.constant"() <{{value = {value} : {kind}}}>'
        if module.count(anchor) != 1:
            raise ValueError('Original external constant identity changed')
        result[f'%{number}'] = 'CONSTANT_'+str(number)
    for token, anchor in [('%6','"tensor.dim"(%arg0, %5)'),
                           ('%7','"tensor.dim"(%arg2, %1)'),
                           ('%8','"tensor.extract"(%arg3)')]:
        if module.count(token+' = '+anchor) != 1:
            raise ValueError('Original runtime extent or selector binding changed')
        result[token] = {'%6':'CAPACITY','%7':'QUERY_EXTENT','%8':'SELECTOR'}[token]
    for token in re.findall('('+SSA+r') = "arith.constant"\(\) <\{value = 8 : index\}>',module):
        result[token] = 'LANES_8'
    return result


def canonical(head, external):
    # Only rename values defined inside this head; unknown outside bindings fail.
    definitions = set(re.findall('('+SSA+r') = ',head))
    definitions.update(re.findall('('+SSA+r'): (?:index|tensor<[^>]+>|vector<[^>]+>)',head))
    names = {}
    def replace(match):
        token = match[0]
        if token in definitions:
            return names.setdefault(token,'%LOCAL_'+str(len(names)))
        if token not in external:
            raise ValueError('Unbound external operand in transformed recurrence: '+token)
        return '%'+external[token]
    return re.sub(SSA,replace,head)


def require_condition(module):
    for anchor in ['%9 = "arith.constant"() <{value = 8 : index}>',
                   '%10 = "arith.constant"() <{value = 32 : index}>',
                   '%11 = "arith.cmpi"(%7, %9) <{predicate = 9 : i64}>',
                   '%12 = "arith.cmpi"(%7, %10) <{predicate = 7 : i64}>',
                   '%13 = "arith.andi"(%11, %12)', '%14 = "scf.if"(%13)']:
        if module.count(anchor) != 1:
            raise ValueError('Runtime query extent must select register8 only within 8..32')
    if module.count('"math.fma"') != 2 or module.count('"scf.if"') != 1:
        raise ValueError('Exactly one original and one optimized FMA recurrence required')
    if '#arith.fastmath<fast>' in module:
        raise ValueError('Fast floating-point reassociation is forbidden')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','SDK','SDK-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a = p.parse_args(); read = lambda path: json.loads(Path(path).read_text())
    c = read(a.config)
    if not a.worker:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path']))); return
    sdk = read(a.SDK); owner = read(a.SDK_controller)
    require_normal_owner(a.SDK_controller,owner['authorized_deadline_unix'])
    if (a.output.exists() or not a.output.is_absolute()
            or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
            or sdk['status'] != 'Q2_dispatch_preflight_SDK_built_STOP_BEFORE_MODEL_CODEGEN'
            or owner['source_revision'] != sdk['source_revision']
            or owner['command'][owner['command'].index('--output')+1] != str(a.SDK.parent)
            or sdk['dispatch'] != entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp')):
        raise ValueError('Frozen source and exact normally owned structural SDK required')
    base = read(checked(sdk['base_sdk']))
    require_proven_qk(checked(sdk['qk_review']),checked(sdk['qk_review_controller']),base)
    original = checked(sdk['fixture']).read_text()
    cases = [('original',original,True),('SSA_renamed',original.replace('%arg','%local'),True)]
    mutations = [
        ('component_step','"scf.for"(%217, %12, %13, %arg14590)','"scf.for"(%217, %12, %two, %arg14590)'),
        ('column_step','"scf.for"(%217, %795, %12, %arg14592)','"scf.for"(%217, %795, %two, %arg14592)'),
        ('key_component','%arg14587, %arg14591, %arg14593, %34409','%arg14587, %arg14589, %arg14593, %34409'),
        ('query_component','%34412, %arg14591)','%34412, %arg14593)'),
        ('broadcast_column','"arith.select"(%797, %217, %arg14589)','"arith.select"(%797, %two, %arg14589)'),
        ('FMA_accumulator','"math.fma"(%34418, %34416, %34417)','"math.fma"(%34418, %34416, %34416)'),
        ('FMA_query','"math.fma"(%34418, %34416, %34417)','"math.fma"(%34417, %34416, %34417)'),
        ('fast_FMA','#arith.fastmath<none>','#arith.fastmath<fast>'),
        ('tail_extent','(-d0 + s0, 128)','(-d0 + s0, 64)'),
        ('mask_extent','"vector.create_mask"(%34414)','"vector.create_mask"(%795)')
    ]
    # The mask is validated by the old proof; perturb it with another valid mask
    # whose length no longer derives from the selected original key slice.
    for name, before, after in mutations:
        if original.count(before) != 1:
            raise ValueError('Actual fixture mutation anchor must be unique: '+name)
        cases.append((name,original.replace(before,after),False))
    library = ctypes.CDLL(str(checked(sdk['candidate'])))
    transform = library.qwenQ2QKDispatchPhysicalPreflight
    reference = library.qwenQ2QKRegisterReference
    for function in (transform, reference):
        function.argtypes = [ctypes.c_char_p,ctypes.c_char_p]; function.restype = ctypes.c_int
    a.output.mkdir(); rows = []
    with (a.output/'diagnostics.txt').open('xb') as log:
        previous = os.dup(2); os.dup2(log.fileno(),2)
        try:
            for name,text,expected in cases:
                source = a.output/(name+'.input.mlir'); target = a.output/(name+'.output.mlir')
                source.write_text(text)
                rc = transform(os.fsencode(source),os.fsencode(target))
                if (rc == 0) != expected:
                    raise ValueError('Actual structural guard disagrees: '+name+' rc='+str(rc))
                row = dict(name=name,accepted=rc==0,returncode=rc,input=entry(source))
                if expected:
                    standalone = a.output/(name+'.reference-head.txt')
                    if reference(os.fsencode(source),os.fsencode(standalone)) != 0:
                        raise ValueError('Unchanged reviewed reference transform failed')
                    module = target.read_text(); require_condition(module)
                    external = bindings(module)
                    before = Path(str(target)+'.original-head.txt')
                    fallback = Path(str(target)+'.fallback-head.txt')
                    optimized = Path(str(target)+'.optimized-head.txt')
                    reference_module = Path(str(standalone)+'.module.mlir')
                    if canonical(before.read_text(),external) != canonical(fallback.read_text(),external):
                        raise ValueError('Fallback recurrence changed')
                    if canonical(optimized.read_text(),external) != canonical(standalone.read_text(),bindings(reference_module.read_text())):
                        raise ValueError('Optimized branch differs from the exact reviewed standalone transform')
                    row.update(output=entry(target),original=entry(before),fallback=entry(fallback),
                        optimized=entry(optimized),reference=entry(standalone),reference_module=entry(reference_module),
                        fallback_original_exact=True,optimized_reviewed_exact=True)
                rows.append(row)
        finally:
            os.dup2(previous,2); os.close(previous)
    write(a.output/'result.json',dict(status='actual_Q2_dispatch_positive2_unsafe10_original_fallback_and_reviewed_register_branches_verified_NOT_target_acceptance',
        SDK=entry(a.SDK),SDK_controller=entry(a.SDK_controller),source=entry(__file__),
        source_root=str(ROOT),source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        cases=rows,positive=2,unsafe_rejected=10,runtime_query_bounds=[8,32],
        model_PRIMARY_compilations=0,DSP_calls=0,model_acceptance=False,performance_acceptance=False))


if __name__ == '__main__':
    main()
