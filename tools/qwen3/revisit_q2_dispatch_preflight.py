"""Independently re-read host dispatch branches and all28 source/owner evidence."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build import ROOT, HERE, entry, write
from inspect_q2_dispatch_fixture import bindings, canonical, require_condition
from profile_model_memory import checked
from q2_dispatch_contract import require_proven_qk
from revisit_native_compile_session_abba import require_normal_owner
from revisit_release_index import indexed_files


def actual_owner(path, result, script, output):
    owner = json.loads(Path(path).read_text())
    require_normal_owner(path,owner['authorized_deadline_unix'])
    root = Path(result['source_root'])
    source = root/'tools/qwen3'/script
    if (owner['source_root'] != str(root) or owner['source_revision'] != result['source_revision']
            or owner['command'][1] != str(source)
            or owner['command'][owner['command'].index('--output')+1] != str(output)):
        raise ValueError('Exact normal historical producer and output binding required')
    committed = subprocess.check_output(['git','-C',str(root),'show',
        result['source_revision']+':tools/qwen3/'+script])
    if hashlib.sha256(committed).hexdigest() != result['source']['sha256']:
        raise ValueError('Historical producer bytes differ from its actual receipt')
    return owner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('all28','all28-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args(); read = lambda path: json.loads(Path(path).read_text())
    git = lambda *words: subprocess.check_output(['git','-C',str(ROOT),*words],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status','--porcelain'):
        raise ValueError('Fresh committed independent review required')
    model = read(a.all28)
    if (model['status'] != 'actual_all28_Q2_runtime_dispatch_original_and_proven_register_verified_NO_MODEL_CODEGEN_NOT_target_acceptance'
            or model['all28_QK_regions'] != 28 or model['runtime_query_bounds'] != [8,32]
            or model['model_PRIMARY_compilations'] or model['DSP_calls']
            or model['model_acceptance'] or model['performance_acceptance']):
        raise ValueError('Complete strict host all28 preflight with honest scope required')
    refs = []; indexed_files(model,refs)
    model_owner = actual_owner(a.all28_controller,model,'inspect_q2_dispatch_model.py',a.all28.parent)
    sdk_path, fixture_path = checked(model['SDK']), checked(model['fixture'])
    sdk, fixture = read(sdk_path), read(fixture_path)
    indexed_files(sdk,refs); indexed_files(fixture,refs)
    sdk_owner = actual_owner(checked(model['SDK_controller']),sdk,'build_q2_dispatch_preflight_sdk.py',sdk_path.parent)
    fixture_owner = actual_owner(checked(model['fixture_controller']),fixture,'inspect_q2_dispatch_fixture.py',fixture_path.parent)
    for name, value in [('SDK',sdk_path),('fixture',fixture_path),('producer',checked(model['original_producer']))]:
        if model_owner['command'][model_owner['command'].index('--'+name)+1] != str(value):
            raise ValueError('All28 owner input differs from stored reference')
    if (fixture['SDK'] != model['SDK'] or fixture['SDK_controller'] != model['SDK_controller']
            or fixture['positive'] != 2 or fixture['unsafe_rejected'] != 10
            or len(fixture['cases']) != 12 or fixture['DSP_calls'] or fixture['model_PRIMARY_compilations']
            or sdk['dispatch'] != model['dispatch']
            or sdk['dispatch'] != entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp')
            or sdk['proven_recurrence'] != model['proven_recurrence']
            or sdk['model_PRIMARY_compilations'] or sdk['DSP_calls']
            or not sdk['unmodified_SDK_replay_byte_identical'] or not sdk['original_PV_objects_unchanged']):
        raise ValueError('Exact fixture, all28, SDK and reviewed source binding required')
    base = read(checked(sdk['base_sdk']))
    require_proven_qk(checked(sdk['qk_review']),checked(sdk['qk_review_controller']),base)
    for path,digest in {**base['link_inputs'],**sdk['retained_objects']}.items():
        if entry(path)['sha256'] != digest:
            raise ValueError('Original composed SDK object changed')
    if entry(checked(sdk['original_replay']))['sha256'] != base['candidate']['sha256']:
        raise ValueError('Actual original SDK replay differs')
    checked(sdk['original_shared_sdk'])
    rows = []
    for number, row in enumerate(fixture['cases']):
        positive = number < 2
        if type(row['accepted']) is not bool or row['accepted'] is not positive or row['returncode'] != (0 if positive else 3):
            raise ValueError('Valid original/renamed IR must pass; valid unsafe IR must fail structural proof')
        if positive:
            module = checked(row['output']).read_text(); require_condition(module)
            external = bindings(module)
            if canonical(checked(row['original']).read_text(),external) != canonical(checked(row['fallback']).read_text(),external):
                raise ValueError('Independent original fallback recurrence mismatch')
            if canonical(checked(row['optimized']).read_text(),external) != canonical(
                    checked(row['reference']).read_text(),bindings(checked(row['reference_module']).read_text())):
                raise ValueError('Independent reviewed register recurrence mismatch')
        rows.append(dict(name=row['name'],accepted=positive,returncode=row['returncode']))
    diagnostic = checked(model['diagnostics']).read_text()
    for marker in ['QWEN_DYNAMIC_QK_INVARIANT_VECTOR matched=28',
        'QWEN_DYNAMIC_QK_DISPATCH verified=28 runtime_query=8..32 original_else=1 exact_proven_register_then=1 FP_order_unchanged=1',
        'QWEN_DYNAMIC_QK_DISPATCH_STOP_BEFORE_MODEL_CODEGEN']:
        if diagnostic.count(marker) != 1:
            raise ValueError('Actual all28 pass marker or deliberate frontend stop changed')
    write(a.output,dict(status='Q2_actual_fixture_and_all28_runtime_dispatch_independently_revisited_HOST_ONLY_NOT_target_or_model_acceptance',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        all28=entry(a.all28),all28_controller=entry(a.all28_controller),SDK=entry(sdk_path),fixture=entry(fixture_path),
        checked_file_references=len(refs),cases=rows,all28_QK_regions=28,runtime_query_bounds=[8,32],
        original_PV_objects_unchanged=True,original_fallback_exact=True,optimized_reviewed_exact=True,
        model_PRIMARY_compilations=0,DSP_calls=0,new_DSP_calls=0,model_acceptance=False,performance_acceptance=False))


if __name__ == '__main__':
    main()
