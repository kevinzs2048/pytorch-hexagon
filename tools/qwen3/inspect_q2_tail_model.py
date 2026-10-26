"""Check identical-tail QK guards on the actual28-layer model, with codegen stopped."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from build import ROOT, HERE, entry, write
from profile_model_memory import checked
from revisit_native_compile_session_abba import require_normal_owner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','SDK','SDK-controller','fixture','fixture-controller','producer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a = p.parse_args(); read = lambda path: json.loads(Path(path).read_text()); c = read(a.config)
    if not a.worker:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],check=True,
            env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path']))); return
    sdk, fixture, producer = read(a.SDK), read(a.fixture), read(a.producer)
    owner = read(a.SDK_controller); require_normal_owner(a.SDK_controller,owner['authorized_deadline_unix'])
    inspection_owner = read(a.fixture_controller)
    require_normal_owner(a.fixture_controller,inspection_owner['authorized_deadline_unix'])
    if (a.output.exists() or not a.output.is_absolute()
            or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
            or sdk['status'] != 'Q2_identical_tail_structural_SDK_STOP_BEFORE_MODEL_CODEGEN_NOT_target'
            or sdk['dispatch'] != entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp')
            or sdk['identical_tail'] != entry(compiler_tools()/'buffer_reuse/DynamicQKIdenticalTail.inc')
            or sdk['integer_guard'] != entry(compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent.c')
            or fixture['status'] != 'actual_Q2_identical_tail_fixture_original_FMA_guard_and_copy_verified_HOST_ONLY_NOT_target'
            or owner['source_revision'] != sdk['source_revision']
            or fixture['SDK'] != entry(a.SDK) or fixture['SDK_controller'] != entry(a.SDK_controller)
            or fixture['positive'] != 2 or fixture['unsafe_rejected'] != 10
            or fixture['DSP_calls'] or fixture['model_PRIMARY_compilations']
            or inspection_owner['source_revision'] != fixture['source_revision']
            or inspection_owner['command'][inspection_owner['command'].index('--output')+1] != str(a.fixture.parent)):
        raise ValueError('Same normally owned preflight SDK and real fixture branch checks required')
    base = read(checked(sdk['base_sdk']))
    composed=read(checked(base['base_sdk']))
    original = read(checked(composed['original_QK_sdk_receipt']))
    if entry(checked(producer['inputs']['compiler'])) != entry(checked(original['candidate'])):
        raise ValueError('Actual original dynamic producer with its exact QK SDK required')
    for key in ('linalg','options_result','compiler'):
        checked(producer['inputs'][key])
    checked(sdk['candidate'])
    a.output.mkdir()
    record = dict(status='inspecting_actual_all28_Q2_identical_tail_STOP_BEFORE_CODEGEN',
        source=entry(__file__),source_root=str(ROOT),
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        SDK=entry(a.SDK),SDK_controller=entry(a.SDK_controller),fixture=entry(a.fixture),
        fixture_controller=entry(a.fixture_controller),original_producer=entry(a.producer),
        dispatch=sdk['dispatch'],proven_recurrence=sdk['proven_recurrence'],
        model_PRIMARY_compilations=0,DSP_calls=0,model_acceptance=False,performance_acceptance=False,
        started_unix=time.time())
    save = lambda: write(a.output/'result.json',record); save()
    try:
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_python_source import configure_triton_python
        from compile_qwen3_artifact import load_extension
        configure_triton_python(Path(producer['triton_python']['root']))
        load_extension(Path(sdk['candidate']['path']))
        from triton._C.libtriton import ir, qcom_hexagon_backend as backend
        from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
        raw = read(producer['inputs']['options_result']['path'])['options']
        defaults = HexagonOptions().__dict__
        if set(defaults)-set(raw) != {'outlineCheckedVectorExp'} or set(raw)-set(defaults) != {'keepAssertMessagesInMain'}:
            raise ValueError('Original option reconciliation changed')
        options = {key:str(value)for key,value in raw.items()}
        options['outlineCheckedVectorExp'] = str(defaults['outlineCheckedVectorExp'])
        options.update(producer['overrides'])
        os.environ.pop('MLIR_ENABLE_DUMP',None); os.environ['HEXAGON_MLIR_PROGRESS']='true'
        context = ir.context(); context.disable_multithreading(); backend.load_dialects(context)
        module = backend.parse_mlir_module_from_file(producer['inputs']['linalg']['path'],context)
        diagnostic = a.output/'pass-diagnostics.txt'; error = None; llvm_ir = None
        with diagnostic.open('xb') as log:
            previous = os.dup(2); os.dup2(log.fileno(),2)
            try:
                llvm_ir = backend.translate_linalg_to_llvmir(module,options)
            except Exception as exc:
                error = repr(exc)
            finally:
                os.dup2(previous,2); os.close(previous)
        text = diagnostic.read_text()
        marker = 'QWEN_DYNAMIC_QK_DISPATCH verified=28 runtime_query=8..32 original_else=1 exact_proven_register_then=1 FP_order_unchanged=1'
        if (error is None or llvm_ir is not None or text.count(marker) != 1
                or text.count('QWEN_DYNAMIC_QK_DISPATCH_STOP_BEFORE_MODEL_CODEGEN') != 1
                or text.count('QWEN_DYNAMIC_QK_INVARIANT_VECTOR matched=28') != 1
                or text.count('QWEN_Q2_IDENTICAL_TAIL verified=28 positive_zero_keys_and_seeds=1 representative_lanes=8 original128_FMA_order=1 fallback_unchanged=1') != 1):
            raise ValueError('Exact all28 QK dispatch and deliberate pre-codegen stop required')
        record.update(status='actual_all28_Q2_identical_tail_guards_original_FMA_and_fallback_verified_NO_MODEL_CODEGEN_NOT_target',
            all28_QK_regions=28,runtime_query_bounds=[8,32],expected_frontend_stop=error,
            diagnostics=entry(diagnostic),completed_unix=time.time()); save()
    except BaseException as exc:
        record.update(status='failed',error=repr(exc)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
