"""Compile one existing q1 category once after real C++ target and all28 closure.

The result is an unaccepted diagnostic model candidate. The original three
constant object outputs must be byte-identical; retain their existing files.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from build import ROOT,entry,write
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner
from run_regression import require_configured_python

def verify_constant_output(generated, historical):
    """Match actual emitted bytes even when old large objects were not retained."""
    digest=hashlib.sha256(generated).hexdigest()
    if digest!=historical['sha256'] or len(generated)!=historical['bytes']:
        raise ValueError('Actual emitted constant bytes differ from the original compiler output')
    retained=historical.get('retained',historical.get('path') is not None)
    if type(retained) is not bool:
        raise ValueError('Explicit historical constant retention state required')
    if retained:
        path=historical.get('path')
        if (not path or entry(path)!=dict(path=str(Path(path).absolute()),sha256=digest)
                or Path(path).stat().st_size!=len(generated)):
            raise ValueError('Retained original constant file changed')
    elif historical.get('path') is not None:
        raise ValueError('Unretained constants cannot borrow a file path')
    return historical

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','sdk','sdk-controller','model','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    p.add_argument('--qk',action='store_true',help='Use the separately target-proved QK SDK and actual all28 preflight')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    read=lambda path:json.loads(Path(path).read_text());c=read(a.config);require_configured_python(c)
    if not a.worker:
        subprocess.run([c['python'],__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path']),PYTHONDONTWRITEBYTECODE='1'));return
    git=lambda *x:subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip();sdk=read(a.sdk);model=read(a.model)
    owner=require_normal_owner(a.sdk_controller,a.deadline_unix);refs=[];indexed_files(sdk,refs)
    all28=read(sdk['all28_review']['path']);preceding=read(all28['run']['path'])
    if (not a.output.is_absolute() or a.output.exists() or git('status','--porcelain') or a.deadline_unix-time.time()<3600
            or sdk['status']!=('q1_QK_candidate_SDK_built_NOT_model_acceptance' if a.qk else
                'zero_tail_runtime_extent_q1_MODEL_CANDIDATE_SDK_built_NOT_model_acceptance')
            or owner['source_revision']!=sdk['source_revision'] or sdk['model_PRIMARY_compilations'] or sdk['DSP_calls']
            or all28['status']!=('target_proved_QK_and_actual_q1_all28_preflight_independently_revisited_NOT_model' if a.qk else
                'actual_CPP_target24_and_original_q1_all28_preflight_independently_closed_NOT_model_acceptance')
            or all28['actual_model']!=entry(a.model) or preceding['options']['addFastMath']!='False'
            or len(model['objects'])!=4):
        raise ValueError('Exact isolated candidate SDK after actual all28 closure and one-hour compile/diagnostic budget required')
    if a.qk:
        require_normal_owner(sdk['all28_review_controller']['path'],read(sdk['all28_review_controller']['path'])['authorized_deadline_unix'])
        if (sdk['preflight_only'] or sdk['physical_review']!=all28['physical_review']
                or sdk['CPP_source']!=all28['CPP_source'] or sdk['model_hook']!=all28['model_hook']
                or not sdk['base_replay_byte_identical'] or not sdk['original_PV_guard_objects_retained']):
            raise ValueError('QK candidate must retain its exact same-pass target/all28 proofs and original PV pass')
    for key in ['linalg','options_result','compiler']:
        if entry(model['inputs'][key]['path'])!=model['inputs'][key]:raise ValueError('Original compiler inputs changed')
    a.output.mkdir();record=dict(status='compiling_one_existing_q1_graph_ZERO_TAIL_DIAGNOSTIC_NOT_accepted',source_root=str(ROOT),
        source_revision=git('rev-parse','HEAD'),source=entry(__file__),SDK=entry(a.sdk),SDK_controller=entry(a.sdk_controller),
        original_model=entry(a.model),actual_linalg=model['inputs']['linalg'],options=preceding['options'],
        DSP_calls=0,model_PRIMARY_compilations=0,model_acceptance=False,performance_acceptance=False,
        deadline_unix=a.deadline_unix,started_unix=time.time());save=lambda:write(a.output/'result.json',record);save()
    record['QK_guard_candidate']=a.qk;save()
    previous=Path.cwd();os.chdir(a.output)
    try:
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_python_source import configure_triton_python
        from compile_qwen3_artifact import load_extension
        configure_triton_python(Path(model['triton_python']['root']));load_extension(Path(sdk['candidate']['path']))
        from triton._C.libtriton import ir,qcom_hexagon_backend as backend
        context=ir.context();context.disable_multithreading();backend.load_dialects(context)
        module=backend.parse_mlir_module_from_file(model['inputs']['linalg']['path'],context)
        diagnostic=a.output/'compiler-diagnostics.txt';record['model_PRIMARY_compilation_attempts']=1;save()
        with diagnostic.open('xb') as stream:
            old=os.dup(2);os.dup2(stream.fileno(),2)
            try:objects=backend.translate_linalg_to_obj(module,preceding['options'])
            finally:os.dup2(old,2);os.close(old)
        if len(objects)!=4:raise ValueError('Original primary plus exact three constant outputs required')
        # Preserve actual model code before any audit/post-processing can fail.
        # Its mere existence is diagnostic, never numerical/model acceptance.
        primary=a.output/('q1-QK-tail-primary.o' if a.qk else 'q1-zero-tail-primary.o')
        primary.write_bytes(objects[0])
        record.update(generated_primary=entry(primary),model_PRIMARY_compilations=1,
            compiler_objects_returned=4);save()
        text=diagnostic.read_text();marker='QWEN_Q1_ZERO_TAIL_RUNTIME_EXTENT verified=28 original_FMA_clones=3 read_only_buffers=1 model_ABI_unchanged=1'
        if text.splitlines().count(marker)!=1 or 'PREFLIGHT_STOP_BEFORE_MODEL_CODEGEN' in text:raise ValueError('Actual all28 guarded candidate must continue through codegen')
        if a.qk:
            from inspect_qk_model_preflight import MARKER as qk_marker
            if text.splitlines().count(qk_marker)!=1:
                raise ValueError('Actual same target-proved all28 QK transform must reach model codegen')
        constant_rows=[]
        for i in range(1,4):
            constant_rows.append(verify_constant_output(objects[i],model['objects'][i]))
        symbols=subprocess.check_output([str(Path(c['hexagon_tools'])/'bin/llvm-nm'),str(primary)],text=True)
        sym=a.output/'symbols.txt';sym.write_text(symbols)
        if '_mlir_ciface_qwen_hvx_positive_zero_extent' not in symbols or '_mlir_ciface_Qwen3DynamicBoundProbe' not in symbols:
            raise ValueError('Original model C API and actual integer extent dependency required')
        if a.qk and '_mlir_ciface_qwen_qk_positive_zero_extent' not in symbols:
            raise ValueError('Actual target-proved QK integer extent dependency required')
        record.update(status=('one_existing_q1_QK_PRIMARY_compiled_constants_identical_NOT_model_acceptance' if a.qk else
            'one_existing_q1_zero_tail_PRIMARY_compiled_constants_identical_NOT_model_acceptance'),
            primary=entry(primary),unchanged_constant_objects=constant_rows,compiler_diagnostics=entry(diagnostic),symbols=entry(sym),
            model_PRIMARY_compilations=1,query_bounds=[1,1],elapsed_seconds=time.time()-record['started_unix'],completed_unix=time.time())
    except BaseException as exc:record.update(status='failed',error=repr(exc));raise
    finally:os.chdir(previous);save()

if __name__=='__main__':main()
