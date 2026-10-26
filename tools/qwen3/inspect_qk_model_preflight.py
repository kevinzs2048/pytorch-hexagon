"""Apply the target-proved QK pass to actual 28-layer q1 and stop before codegen."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from build import ROOT,entry,write
from revisit_native_compile_session_abba import require_normal_owner
from revisit_release_index import indexed_files
from run_regression import require_configured_python

MARKER='QWEN_Q1_QK_IDENTICAL_TAIL verified=28 query_FP_cloned_unchanged=1 original_FMA_per_component=1 read_only_guard=1 model_ABI_unchanged=1'
STOP='QWEN_Q1_QK_PREFLIGHT_STOP_BEFORE_MODEL_CODEGEN'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','sdk','sdk-controller','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();read=lambda x:json.loads(Path(x).read_text());c=read(a.config)
    require_configured_python(c)
    if not a.worker:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path']),PYTHONDONTWRITEBYTECODE='1'))
        return
    git=lambda *x:subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip()
    sdk=read(a.sdk);owner=require_normal_owner(a.sdk_controller,sdk['deadline_unix']);refs=[];indexed_files(sdk,refs)
    model=read(a.model);physical=read(sdk['physical_review']['path'])
    linked=read(physical['linked_build']['path']);lower=read(linked['build']['path']);observed=read(lower['preflight']['path'])
    if (not a.output.is_absolute() or a.output.exists() or git('status','--porcelain')
            or a.deadline_unix-time.time()<1800
            or sdk['status']!='q1_QK_preflight_SDK_built_STOP_BEFORE_MODEL_CODEGEN'
            or not sdk['preflight_only'] or not sdk['base_replay_byte_identical']
            or owner['source_revision']!=sdk['source_revision']
            or observed['model']!=entry(a.model) or len(model['objects'])!=4):
        raise ValueError('Exact actual q1 model and normally owned target-proved preflight SDK required')
    for key in ('linalg','options_result','compiler'):
        if entry(model['inputs'][key]['path'])!=model['inputs'][key]:raise ValueError('Original q1 input changed')
    options=observed['original_compiler_options']
    if options['addFastMath']!='False':raise ValueError('Original strict FP policy required')
    a.output.mkdir();record=dict(status='inspecting_target_proved_QK_in_actual_q1_NOT_model_codegen',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        SDK=entry(a.sdk),SDK_controller=entry(a.sdk_controller),actual_model=entry(a.model),
        physical_review=sdk['physical_review'],CPP_source=sdk['CPP_source'],model_hook=sdk['model_hook'],
        options=options,model_PRIMARY_compilations=0,DSP_calls=0,model_acceptance=False,
        deadline_unix=a.deadline_unix,started_unix=time.time())
    save=lambda:write(a.output/'result.json',record);save()
    try:
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_python_source import configure_triton_python
        from compile_qwen3_artifact import load_extension
        configure_triton_python(Path(model['triton_python']['root']));load_extension(Path(sdk['candidate']['path']))
        from triton._C.libtriton import ir,qcom_hexagon_backend as backend
        context=ir.context();context.disable_multithreading();backend.load_dialects(context)
        module=backend.parse_mlir_module_from_file(model['inputs']['linalg']['path'],context)
        diagnostics=a.output/'all28.stderr.txt';error=None
        with diagnostics.open('xb') as log:
            previous=os.dup(2);os.dup2(log.fileno(),2)
            try:backend.translate_linalg_to_obj(module,options)
            except RuntimeError as exc:error=str(exc)
            finally:os.dup2(previous,2);os.close(previous)
        text=diagnostics.read_text()
        if (error is None or text.splitlines().count(MARKER)!=1 or text.splitlines().count(STOP)!=1
                or list(a.output.glob('PRIMARY*'))):
            raise ValueError('Exactly 28 guarded QK recurrences and deliberate pre-codegen stop required')
        first=text.split('QWEN_Q1_QK_GUARDED_FIRST_BEGIN\n',1)[1].split('\nQWEN_Q1_QK_GUARDED_FIRST_END',1)[0]
        excerpt=a.output/'actual-guarded-qk-loop.mlir.txt';excerpt.write_text(first+'\n')
        record.update(status='actual_q1_all28_target_proved_QK_preflight_STOP_BEFORE_MODEL_CODEGEN',
            guarded_regions=28,diagnostics=entry(diagnostics),actual_guarded_loop=entry(excerpt),
            deliberate_stop_error=error,completed_unix=time.time())
    except BaseException as exc:record.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
