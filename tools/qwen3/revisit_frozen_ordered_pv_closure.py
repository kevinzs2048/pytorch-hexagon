"""Independently audit original PV393 without new model execution or compilation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

# Keep the numerical producer frozen; this delivery checkout owns only review.
AUDIT_ROOT = Path(__file__).resolve().parents[2]
_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument("--numerical-source", type=Path, required=True)
_boot_args, _remaining = _boot.parse_known_args()
sys.argv[1:] = _remaining
sys.path.insert(0, str(_boot_args.numerical_source / "tools/qwen3"))

from build import ROOT, entry, sha, write
from build_ordered_pv_recipe import validate_transition
from ordered_pv_model_contract import KIND, require_reference
from run_ordered_pv_iteration import require_early
from revisit_key_tile_model import revisit
from run_helper_effects_iteration import positive_native_gain
from compare_performance import accepted
from run_integer_regressions import candidate_libraries, compare_integer, validate_float_policy, validate_parameter_mode
from bounded_loader_policy import verify_native_layout
from revisit_performance_release import native_abba, states
from audit_prepared_stage_reuse import cold_cache


def read(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','closure','controller','baseline-closure','original-state-suite','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native89-negative',action='store_true',help='Audit only an actually nonpositive original PV timing decision')
    a=p.parse_args();git=lambda *x:subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip()
    revision=git('rev-parse','HEAD');audit_git=lambda *x: subprocess.check_output(['git','-C',str(AUDIT_ROOT),*x],text=True).strip()
    if audit_git('status','--porcelain') or ROOT != _boot_args.numerical_source.resolve():
        raise ValueError('Clean committed independent audit and exact frozen numerical source required')
    result=read(a.closure/'result.json');controller=read(a.controller)
    calls=89 if a.native89_negative else 393
    status=('ordered_PV_deferred_after_nonpositive_original_native89' if a.native89_negative else
            'ordered_PV_original393_complete_NOT_independently_accepted')
    if (not a.output.is_absolute() or a.output.exists() or git('status','--porcelain')
            or result['status']!=status or result['candidate_kind']!=KIND or type(result['DSP_calls']) is not int or result['DSP_calls']!=calls
            or result['source_revision']!=revision or result['model_inference_acceptance'] is not False
            or result['original_state304_executed'] is not (not a.native89_negative)
            or result['original_full393_minimum_remaining_seconds']!=14400
            or result['new_primary_model_codegen_invocations']!=0
            or result['source_sha256']!=sha(ROOT/'tools/qwen3/run_ordered_pv_iteration.py')
            or controller['status']!='completed' or controller['returncode']!=0
            or controller['source_revision']!=revision or controller['source_root']!=str(ROOT)
            or controller['authorized_deadline_unix']-controller['start_unix']<14400
            or controller['completed_unix']>=controller['authorized_deadline_unix']
            or result['deadline_unix']!=controller['authorized_deadline_unix']
            or any(Path('/proc',str(controller[k])).exists() for k in ('controller_pid','child_pid'))):
        raise ValueError('Own matching terminal PV closure and normal original-budget owner required')
    prior=read(a.baseline_closure/'result.json');baseline_audit=read(result['baseline_independent_revisit']['path'])
    require_reference(prior,baseline_audit,entry(a.baseline_closure/'result.json'))
    early=read(result['early_model_revisit']['path']);owners=read(result['owner_revisit']['path'])
    require_early(early,owners)
    if revisit(Path(early['run']['path']),Path(early['plan']['path']),a.baseline_closure/'cold')!=early:
        raise ValueError('Actual original PV8 source, full packet and raw ABBA changed')
    for key in ('baseline_independent_revisit','early_model_revisit','owner_revisit'):
        if entry(result[key]['path'])!=result[key]:raise ValueError('Prerequisite bytes changed')
    if (result['baseline_closure']!=entry(a.baseline_closure/'result.json')
            or owners['packet_revisit']!=result['early_model_revisit']):
        raise ValueError('Exact baseline and early packet lineage required')
    base,libraries=candidate_libraries(a.closure/'cold');old_base,old_libraries=candidate_libraries(a.baseline_closure/'cold')
    build=read(early['build']['path'])
    expected={(1,1):build['library']['sha256'],(2,32):old_libraries[(2,32)]}
    if libraries!=expected or result['candidate_libraries']!={str(k):v for k,v in expected.items()}:
        raise ValueError('Exact tested PV q1 and unchanged QK q2 DSOs required')
    validate_transition(read(old_base['recipe']['path']),read(base['recipe']['path']))
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',build['source_revision'],'HEAD'],check=True)
    allowed={'qwen3/runtime/arm/hexagon_qwen_captured_machine_cache.py'}|{
        'tools/qwen3/'+name+'.py' for name in ('reuse_accepted_compiler_objects','preflight_key_tile_recipe',
        'build_ordered_pv_recipe','test_ordered_pv_recipe','run_ordered_pv_iteration',
        'test_ordered_pv_iteration','revisit_ordered_pv_closure')}
    if not set(git('diff','--name-only',build['source_revision'],revision).splitlines())<=allowed:
        raise ValueError('Numerical/compiler source changed after the actual PV model8')
    cold=read(a.closure/'cold/native-result.json');old_cold=read(a.baseline_closure/'cold/native-result.json')
    acceptance=read(a.closure/'cold/acceptance.json');audit=read(acceptance['audit']['path'])
    if (cold['DSP_calls']!=9 or audit['numerical_acceptance'] is not True
            or entry(acceptance['audit']['path'])!=acceptance['audit']
            or audit['run_sha256']!=sha(a.closure/'cold/native-result.json')
            or read(a.closure/'cold/run-exit.json')!={'board_returncode':0,'worker_returncode':0,'numerical_acceptance':False}):
        raise ValueError('Own full raw ARM cold9 and normal exit required')
    compare_integer(old_cold,cold,libraries);validate_float_policy(cold,True);fingerprint=verify_native_layout(cold)
    sys.path[:0]=[str(ROOT/'qwen3/runtime/arm'),str(ROOT)]
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    compiler=cold_cache(a.closure/'cold',base)
    _,old_warm,_=accepted(a.baseline_closure/'warm16');_,warm,_=accepted(a.closure/'warm16')
    if warm['DSP_calls']!=16:raise ValueError('Original warm16 required')
    compare_integer(old_warm,warm,libraries);validate_float_policy(warm,True);validate_parameter_mode(warm,'exact-bytes')
    native=native_abba(a.closure);positive=positive_native_gain(native['raw_native_measurements'],False)
    if positive==a.native89_negative:raise ValueError('Actual recomputed native decision differs')
    state=None if a.native89_negative else states(a.closure,a.original_state_suite,libraries)
    folders=[a.closure/'cold',a.closure/'warm16']+[a.closure/('native-abba-'+str(i)+'-'+kind)
        for i,kind in enumerate(('baseline','candidate','candidate','baseline'))]
    if not a.native89_negative:folders += [a.closure/'state304'/case for case in ('tails','eos','old-long32','prefill1024','decode993','code64')]
    config=read(a.config);board=config['board'];runtime=result['common_native_runtime_override'];raws=[];pids=set();chains=[]
    for folder in folders:
        processes=read(folder/'active-processes.json')
        if any(type(pid) is not int or Path('/proc',str(pid)).exists() for pid in processes.values()):
            raise ValueError('Every actual host session owner must be gone')
        actual=read(folder/'native-result.json');commands=read(folder/'commands.json') if (folder/'commands.json').exists() else read(folder/'result.json')
        command=commands['board']
        if verify_native_layout(actual)!=fingerprint:raise ValueError('Payload-preserving loader fingerprint changed')
        if folder!=a.closure/'cold' and (command[command.index('--catalog')+1]!=runtime['board_path']
                or command[command.index('--catalog-sha256')+1]!=runtime['sha256']):
            raise ValueError('Every native arm/state must use the same common runtime')
        worker=commands['worker'];mailbox=Path(worker[worker.index('--mailbox')+1]);chains.append(audit_compile_handoff(actual,mailbox,rebuilt=True))
        output=Path(command[command.index('--output')+1]);raw=Path(config['observations'])/output.name/'stream.stdout.txt'
        events=[json.loads(line) for line in raw.read_text().splitlines()]
        if (not events or events[-1]['stage']!='close'
                or [e for e in events if e.get('stage')=='invoke']!=[f['invoke'] for f in actual['forwards']]
                or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
            raise ValueError('Complete raw invocation trace and normal close required')
        ready=[e for e in events if e.get('stage')=='ready']
        if len(ready)!=1 or type(ready[0].get('ARM_pid')) is not int:raise ValueError('Actual AP owner identity required')
        pids.add(ready[0]['ARM_pid']);raws.append(entry(raw))
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target'],shlex.join([board['python'],'-c',
        'from pathlib import Path;import sys;assert all(not Path("/proc",p).exists() for p in sys.argv[1:])',*map(str,sorted(pids))])],check=True)
    write(a.output,dict(status='ordered_PV_original_native89_negative_independently_revisited' if a.native89_negative else
        'ordered_PV_original393_independently_revisited',candidate_kind=KIND,closure=entry(a.closure/'result.json'),
        controller=entry(a.controller),source_root=str(ROOT),source_revision=revision,reviewer_source=entry(__file__),
        audit_source_root=str(AUDIT_ROOT),audit_source_revision=audit_git("rev-parse","HEAD"),
        preparation_reviewer_source=entry(Path(__file__).with_name("audit_prepared_stage_reuse.py")),
        DSP_calls=calls,new_DSP_calls=0,candidate_DSP_calls=57 if a.native89_negative else 361,paired_baseline_DSP_calls=32,
        model_inference_acceptance=not a.native89_negative,positive_native_gain=positive,
        candidate_libraries={str(k):v for k,v in libraries.items()},compiler_cache=compiler,native=native,states=state,
        raw_invocation_traces=raws,actual_AP_owners_gone=sorted(pids),compile_chains=chains,
        production_TUI_speedup_claimed=False,completed_unix=time.time()))


if __name__=='__main__':main()
