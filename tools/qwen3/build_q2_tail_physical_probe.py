"""Compile one extracted QK operator and pair it with the tested current object."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from build import ROOT,HERE,entry,write
from profile_model_memory import checked
from revisit_native_compile_session_abba import require_normal_owner


def require_probe_exports(text):
    names={line.split()[-1]for line in text.splitlines()if line.split()}
    if not {'main','hexagon_forward_buffers'}<=names:
        raise ValueError('Both actual DSP loader entrypoints must be dynamically exported')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','fixture','fixture-controller','all28','all28-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text());c=read(a.config)
    if not a.worker:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path'])));return
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain') or a.deadline_unix-time.time()<1800:
        raise ValueError('Fresh committed isolated operator build and bounded budget required')
    fixture,all28=read(a.fixture),read(a.all28)
    for controller,result,script,directory in (
        (a.fixture_controller,fixture,'inspect_q2_tail_fixture.py',a.fixture.parent),
        (a.all28_controller,all28,'inspect_q2_tail_model.py',a.all28.parent)):
        owner=read(controller);require_normal_owner(controller,owner['authorized_deadline_unix'])
        if (owner['source_revision']!=result['source_revision'] or Path(owner['command'][1]).name!=script
                or owner['command'][owner['command'].index('--output')+1]!=str(directory)):
            raise ValueError('Actual normal fixture and all28 owners required')
    if (fixture['status']!='actual_Q2_identical_tail_fixture_original_FMA_guard_and_copy_verified_HOST_ONLY_NOT_target'
            or fixture['positive']!=2 or fixture['unsafe_rejected']!=10
            or all28['status']!='actual_all28_Q2_identical_tail_guards_original_FMA_and_fallback_verified_NO_MODEL_CODEGEN_NOT_target'
            or all28['fixture']!=entry(a.fixture) or all28['all28_QK_regions']!=28
            or fixture['model_PRIMARY_compilations'] or all28['model_PRIMARY_compilations']
            or fixture['DSP_calls'] or all28['DSP_calls']):
        raise ValueError('Actual exact recurrence/guard/copy proof and all28 pre-codegen check required')
    sdk=read(checked(fixture['SDK']));base=read(checked(sdk['base_sdk']))
    for key,name in [('dispatch','buffer_reuse/DynamicQKRegisterDispatch.cpp'),
            ('identical_tail','buffer_reuse/DynamicQKIdenticalTail.inc'),
            ('integer_guard','runtime/profile/q2_qk_positive_zero_extent.c')]:
        if sdk[key]!=entry(HERE/name):raise ValueError('The actually checked transform/guard changed')
    old_review=read(checked(base['physical_review']));old_owner=checked(base['physical_review_controller'])
    require_normal_owner(old_owner,read(old_owner)['authorized_deadline_unix'])
    if (old_review['status']!='actual_Q2_runtime_dispatch_ABBA64_all_FP32_words_full_FNV_and_normal_owners_independently_revisited_NOT_model_acceptance'
            or old_review['DSP_calls']!=64 or old_review['model_inference_acceptance']):
        raise ValueError('Actual independently target-tested current register8 dispatch required')
    original=read(checked(old_review['receipt']));baseline=original['variants']['dispatch']
    options=original['options']
    if options['addFastMath']!='False' or original['original_compiler']!=entry(c['compiler']):
        raise ValueError('Original non-reassociated compiler/options required')
    for key in ('input','llvm','object','assembly','library'):checked(baseline[key])
    source=checked(fixture['cases'][0]['output'])
    a.output.mkdir();record=dict(status='building_isolated_Q2_identical_tail_NOT_model',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        fixture=entry(a.fixture),fixture_controller=entry(a.fixture_controller),all28=entry(a.all28),
        all28_controller=entry(a.all28_controller),SDK=fixture['SDK'],current_baseline_review=base['physical_review'],
        current_baseline_library=baseline['library'],
        original_compiler=entry(c['compiler']),options=options,variants={},sources=[],commands=[],
        isolated_operator_compilations=0,new_model_PRIMARY_compilations=0,DSP_calls=0,
        model_inference_acceptance=False,performance_acceptance=False,deadline_unix=a.deadline_unix)
    save=lambda:write(a.output/'result.json',record)
    def run(argv,label):
        row=dict(argv=argv,started_unix=time.time());record['commands'].append(row);save()
        with (a.output/(label+'.log')).open('xb')as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        row.update(returncode=result.returncode,completed_unix=time.time());save();result.check_returncode()
    previous=Path.cwd();save()
    try:
        for name in ('q2_tail_probe.c','q2_qk_positive_zero_extent.c','zero_tail_hvx_guard.c','dynamic_qk_reference.c'):
            target=a.output/name;shutil.copy2(compiler_tools()/'runtime/profile'/name,target);record['sources'].append(entry(target))
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_python_source import configure_triton_python
        from compile_qwen3_artifact import load_extension
        configure_triton_python(Path(c['triton_python']));load_extension(Path(c['compiler']))
        from triton._C.libtriton import ir,qcom_hexagon_backend as backend
        context=ir.context();context.disable_multithreading();backend.load_dialects(context)
        folder=a.output/'tail';folder.mkdir();os.chdir(folder)
        path=folder/'operator.mlir';shutil.copy2(source,path)
        module=backend.parse_mlir_module_from_file(str(path),context)
        llvm=folder/'operator.ll';llvm.write_text(backend.translate_linalg_to_llvmir(module,options))
        if llvm.read_text().count('call i64 @_mlir_ciface_qwen_q2_qk_positive_zero_extent')!=1:
            raise ValueError('Exactly one real runtime guard call in the actual candidate LLVM required')
        module=backend.parse_mlir_module_from_file(str(path),context)
        record['isolated_operator_compilations']=1;save();objects=backend.translate_linalg_to_obj(module,options)
        if len(objects)!=1:raise ValueError('One isolated operator object required')
        obj=folder/'operator.o';obj.write_bytes(bytes(objects[0]));assembly=folder/'operator.asm'
        with assembly.open('xb')as log:subprocess.run([str(Path(c['hexagon_tools'])/'bin/hexagon-llvm-objdump'),'-d',str(obj)],stdout=log,check=True)
        os.chdir(previous)
        tools=Path(c['hexagon_tools']);sdkroot=Path(c['hexagon_sdk'])
        flags=[str(tools/'bin/hexagon-clang'),'-O2','-Wall','-Wextra','-Werror','-fPIC','-fvisibility=hidden',
            '-fno-builtin','-mv73','-mhvx','-mhvx-length=128B','-G0','-I',str(sdkroot/'incs'),'-I',str(sdkroot/'incs/stddef')]
        helpers=[]
        for name in ('q2_tail_probe.c','q2_qk_positive_zero_extent.c','zero_tail_hvx_guard.c'):
            helper=a.output/(name+'.o');run([*flags,'-c',str(a.output/name),'-o',str(helper)],name+'-compile');helpers.append(str(helper))
        rows=dict(baseline=baseline,tail=dict(input=entry(path),actual_fixture=entry(source),
            llvm=entry(llvm),object=entry(obj),assembly=entry(assembly)))
        for kind,row in rows.items():
            library=a.output/(kind+'.so');run([*flags,*helpers,row['object']['path'],'-shared','--ld-path='+c['lld'],
                '-Wl,-z,separate-loadable-segments','-Wl,--no-as-needed',
                '-Wl,--wrap=_mlir_ciface_qwen_q2_qk_positive_zero_extent',
                str(tools/'target/hexagon/lib/v73/G0/pic/libc.so'),'-o',str(library)],kind+'-link')
            undefined=subprocess.check_output([str(tools/'bin/llvm-nm'),'-u',str(library)],text=True)
            if '__real_'in undefined or '_mlir_ciface_qwen_q2_qk_positive_zero_extent'in undefined:
                raise ValueError('Diagnostic guard or target must bind locally')
            exported=subprocess.check_output([str(tools/'bin/llvm-nm'),'-D','--defined-only',str(library)],text=True)
            require_probe_exports(exported)
            record['variants'][kind]=dict(row,library=entry(library));save()
        record.update(status='actual_Q2_identical_tail_and_current_dispatch_objects_linked_NOT_target_or_model',
            current_baseline_object_reused_exact=True,completed_unix=time.time());save()
    except BaseException as error:record.update(status='failed',error=repr(error));save();raise
    finally:os.chdir(previous)


if __name__=='__main__':main()
