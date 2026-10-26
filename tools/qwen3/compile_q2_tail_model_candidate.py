"""Compile one existing dynamic q2 category with the exact target-proved identical tail."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from build import ROOT,entry,write
from compile_zero_tail_q1_model_candidate import verify_constant_output
from profile_model_memory import checked
from revisit_native_compile_session_abba import require_normal_owner
from revisit_release_index import indexed_files
from run_regression import require_configured_python


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','SDK','SDK-controller','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text());c=read(a.config)
    require_configured_python(c)
    if not a.worker:
        subprocess.run([c['python'],__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path']),PYTHONDONTWRITEBYTECODE='1'));return
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    sdk=read(a.SDK);model=read(a.model);owner=read(a.SDK_controller)
    require_normal_owner(a.SDK_controller,owner['authorized_deadline_unix'])
    refs=[];indexed_files(sdk,refs)
    physical=read(checked(sdk['target_review']))
    all28=read(checked(sdk['all28']))
    base=read(checked(sdk['base_sdk']))
    original=read(checked(base['base_sdk']))
    original_qk=read(checked(original['original_QK_sdk_receipt']))
    if (a.output.exists() or not a.output.is_absolute() or git('status','--porcelain')
            or a.deadline_unix-time.time()<3600
            or sdk['status']!='Q2_identical_tail_target_proved_MODEL_CANDIDATE_SDK_NOT_model_acceptance'
            or owner['source_revision']!=sdk['source_revision']
            or sdk['model_PRIMARY_compilations'] or sdk['DSP_calls']
            or physical['DSP_calls']!=256 or physical['new_DSP_calls']
            or physical['model_inference_acceptance']
            or all28['original_producer']!=entry(a.model)
            or all28['all28_QK_regions']!=28 or len(model['objects'])!=4
            or model['inputs']['compiler']['sha256']!=original_qk['candidate']['sha256']
            or not sdk['base_replay_byte_identical'] or not sdk['original_PV_objects_unchanged']):
        raise ValueError('Exact target256/all28, original q2 model and unchanged PV with one-hour budget required')
    for key in ('linalg','compiler','options_result'):checked(model['inputs'][key])
    old=read(read(base['qk_review']['path'])['receipt']['path'])
    lowering=read(checked(old['lowering']));options=lowering['options']
    if options['addFastMath']!='False':raise ValueError('Strict original arithmetic options required')
    a.output.mkdir()
    record=dict(status='compiling_one_existing_Q2_identical_tail_PRIMARY_NOT_model_acceptance',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        SDK=entry(a.SDK),SDK_controller=entry(a.SDK_controller),original_model=entry(a.model),
        actual_linalg=model['inputs']['linalg'],options=options,physical_review=sdk['target_review'],
        all28=sdk['all28'],model_PRIMARY_compilation_attempts=0,model_PRIMARY_compilations=0,
        DSP_calls=0,model_acceptance=False,performance_acceptance=False,query_bounds=[2,32],
        deadline_unix=a.deadline_unix,started_unix=time.time())
    save=lambda:write(a.output/'result.json',record);save();previous=Path.cwd()
    try:
        os.chdir(a.output);sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_python_source import configure_triton_python
        from compile_qwen3_artifact import load_extension
        configure_triton_python(Path(model['triton_python']['root']))
        load_extension(Path(sdk['candidate']['path']))
        from triton._C.libtriton import ir,qcom_hexagon_backend as backend
        context=ir.context();context.disable_multithreading();backend.load_dialects(context)
        module=backend.parse_mlir_module_from_file(model['inputs']['linalg']['path'],context)
        diagnostic=a.output/'compiler-diagnostics.txt'
        record['model_PRIMARY_compilation_attempts']=1;save()
        with diagnostic.open('xb')as log:
            oldfd=os.dup(2);os.dup2(log.fileno(),2)
            try:objects=backend.translate_linalg_to_obj(module,options)
            finally:os.dup2(oldfd,2);os.close(oldfd)
        if len(objects)!=4:raise ValueError('Original primary plus three constant members required')
        primary=a.output/'q2-identical-tail-primary.o';primary.write_bytes(objects[0])
        record.update(generated_primary=entry(primary),compiler_objects_returned=4,model_PRIMARY_compilations=1);save()
        text=diagnostic.read_text()
        markers=['QWEN_Q2_IDENTICAL_TAIL verified=28 positive_zero_keys_and_seeds=1 representative_lanes=8 original128_FMA_order=1 fallback_unchanged=1',
            'QWEN_DYNAMIC_QK_INVARIANT_VECTOR matched=28',
            'QWEN_DYNAMIC_QK_DISPATCH verified=28 runtime_query=8..32 original_else=1 exact_proven_register_then=1 FP_order_unchanged=1',
            'QWEN_DYNAMIC_PV_REGISTER verified=84 lanes=8 per_partial_k_order=original partial_merges=original no_fast_math=1',
            'QWEN_DYNAMIC_PV_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE']
        if any(text.count(marker)!=1 for marker in markers) or 'STOP_BEFORE_MODEL_CODEGEN'in text:
            raise ValueError('All28 dispatch and all84 original PV must actually reach codegen')
        constants=[verify_constant_output(objects[i],model['objects'][i])for i in (1,2,3)]
        symbols=subprocess.check_output([str(Path(c['hexagon_tools'])/'bin/llvm-nm'),str(primary)],text=True)
        if ('_mlir_ciface_Qwen3DynamicBoundProbe'not in symbols
                or '_mlir_ciface_qwen_q2_qk_positive_zero_extent'not in symbols):
            raise ValueError('Original dynamic model ABI required')
        symbol_file=a.output/'symbols.txt';symbol_file.write_text(symbols)
        record.update(status='one_existing_Q2_identical_tail_PRIMARY_compiled_constants_identical_NOT_model_acceptance',
            primary=entry(primary),unchanged_constant_objects=constants,compiler_diagnostics=entry(diagnostic),
            symbols=entry(symbol_file),elapsed_seconds=time.time()-record['started_unix'],completed_unix=time.time())
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:
        os.chdir(previous);save()


if __name__=='__main__':main()
