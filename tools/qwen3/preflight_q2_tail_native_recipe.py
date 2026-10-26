"""Rebuild/cache both exact tested copy variants, forbidding any model codegen."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from build import ROOT,entry,sha,write
from profile_model_memory import checked
from build_q2_tail_native_recipe import validate_transition
from revisit_q2_dispatch_preflight import actual_owner
from run_regression import require_configured_python
from run_integer_regressions import candidate_libraries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-cold','recipe-build','recipe-build-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh committed exact-copy preflight required')
    require_configured_python(read(a.config));build=read(a.recipe_build/'result.json')
    actual_owner(a.recipe_build_controller,build,'build_q2_tail_native_recipe.py',a.recipe_build)
    recipe=checked(build['recipe']);data=read(recipe)
    validate_transition(read(checked(build['base_recipe'])),data)
    sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
    from hexagon_q2_identical_tail import authenticate
    proof=authenticate(data['inputs'])
    _,libraries=candidate_libraries(a.baseline_cold)
    expected={(1,1):proof['model']['unchanged_q1_library'],(2,32):proof['manifest']['libraries']['candidate']['sha256']}
    baseline={(1,1):proof['model']['unchanged_q1_library'],(2,32):proof['manifest']['libraries']['baseline']['sha256']}
    if libraries!=baseline:raise ValueError('Both actual accepted baseline libraries required')
    native=read(a.baseline_cold/'native-result.json')
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    from reuse_accepted_prepared_linalg import frontend_entry
    a.output.mkdir();cache_root=a.output/'machine-cache';cache_root.mkdir();frontend=a.baseline_cold/'frontend-cache'
    cache=CapturedQwenMachineCache(cache_root,recipe,sha(recipe),frontend)
    original_run=subprocess.run
    def guarded_run(argv,*args,**kwargs):
        if any(Path(str(word)).name=='probe_hexagon_dynamic_linalg.py'for word in argv):
            raise ValueError('New model primary codegen is forbidden in byte-copy preflight')
        return original_run(argv,*args,**kwargs)
    subprocess.run=guarded_run;rows=[]
    try:
        for capture in native['captures']:
            bounds=tuple(capture['contract']['variables']['query'])
            inputs=frontend_entry(frontend/(capture['frontend_key']+'.json'));start=time.monotonic()
            result=cache.get_or_compile(inputs,runtime_contract=capture['contract']);folder=Path(result['directory'])
            linked=read(folder/'linked/result.json');reuse=read(folder/'objects/reuse-result.json')
            if (result['cache_hit'] or result['model_object_codegen_invocations']
                    or result['machine_code_compilations']!=1 or sha(linked['library']['path'])!=expected[bounds]
                    or linked['library']['sha256']!=expected[bounds] or reuse['new_machine_code_compilations']
                    or reuse['DSP_calls'] or reuse['query_bounds']!=list(bounds)
                    or linked['global_copy']['original_model_primary_and_constants_unchanged']is not True
                    or linked['global_copy']['original_fixed512_text_unchanged']is not True
                    or linked['global_copy']['all_copy_symbols_locally_bound']is not True):
                raise ValueError('Actual public cache must reproduce the exact fully packet-tested copy libraries')
            if bounds==(2,32):
                objects=read(folder/'objects/result.json')
                if (linked['q2_identical_tail']['primary']['sha256']!=proof['code']['primary']['sha256']
                        or objects['objects'][0]['sha256']!=proof['code']['primary']['sha256']
                        or objects['objects'][1:]!=proof['code']['unchanged_constant_objects']):
                    raise ValueError('Only the exact newly proved primary may replace the accepted q2 object')
            elif 'q2_identical_tail'in linked:
                raise ValueError('New prefill stage must never alter q1')
            second=cache.get_or_compile(inputs,runtime_contract=capture['contract'])
            if not second['cache_hit'] or second['machine_code_compilations'] or second['model_object_codegen_invocations']:
                raise ValueError('Both actual query domains must hit on the second complete lookup')
            rows.append(dict(query_bounds=list(bounds),DSO=linked['library'],
                base_link=linked['global_copy']['base_link'],reuse_receipt=entry(folder/'objects/reuse-result.json'),
                elapsed_seconds=time.monotonic()-start,second_lookup_verified_cache_hit=True))
            write(a.output/'progress.json',dict(rows=rows,DSP_calls=0))
    finally:subprocess.run=original_run
    if len(rows)!=2 or {tuple(row['query_bounds'])for row in rows}!={(1,1),(2,32)}:
        raise ValueError('Exactly both existing dynamic query domains required')
    write(a.output/'result.json',dict(status='Q2_identical_tail_actual_public_two_domain_exact_tested_DSOs_NOT_native_acceptance',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        recipe=entry(recipe),recipe_build=entry(a.recipe_build/'result.json'),recipe_build_controller=entry(a.recipe_build_controller),
        baseline_cold=str(a.baseline_cold),rows=rows,DSP_calls=0,new_model_PRIMARY_compilations=0,
        model_inference_acceptance=False,performance_acceptance=False))


if __name__=='__main__':main()
