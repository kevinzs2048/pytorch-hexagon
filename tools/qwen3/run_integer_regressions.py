"""Revisit existing state boundaries with independently accepted integer code.

Changed model library identity is allowed only when bound to the accepted cold
candidate. Complete buffers, tokens and checksums still have to be identical.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from compare_performance import accepted, sha
from run_performance_regressions import derive_command, save
from run_regression import accepted_base, replace, require_configured_python

ROOT=Path(__file__).resolve().parents[2]


def candidate_libraries(cold):
    base=accepted_base(cold)
    audit=json.loads(Path(json.loads((cold/'acceptance.json').read_text())['audit']['path']).read_text())
    native=json.loads((cold/'native-result.json').read_text())
    if (sha(cold/'native-result.json')!=audit['run_sha256']
            or native['status']!='ARM_native_dynamic_compiler_handoff_generate_completed'
            or native['machine_code_compilations']!=2 or native['frontend_invocations']!=2
            or native['process_exit_code']!=0 or native['CPU_model_fallback']):
        raise ValueError('Actual fresh two-graph candidate cold acceptance required')
    recipe=json.loads(Path(base['recipe']['path']).read_text())
    if recipe.get('packed_i8_query1') is not True:
        raise ValueError('Explicit integer candidate recipe required')
    libraries={}
    for capture in native['captures']:
        bounds=tuple(capture['contract']['variables']['query'])
        if bounds not in ((1,1),(2,32)) or bounds in libraries:
            raise ValueError('Exactly the original two query categories required')
        libraries[bounds]=capture['DSO_sha256']
    if set(libraries)!={(1,1),(2,32)}:
        raise ValueError('Cold candidate did not actually execute both query categories')
    return base,libraries


def compare_integer(baseline,candidate,libraries):
    for run in (baseline,candidate):
        if len(run['forwards'])!=run['DSP_calls']:
            raise ValueError('Complete actual forward list required')
    if ([(r['prompt_ids'],r['generated_ids'],r['terminal_observation']) for r in baseline['requests']]
            !=[(r['prompt_ids'],r['generated_ids'],r['terminal_observation']) for r in candidate['requests']]):
        raise ValueError('Prompt, selected tokens or termination changed')
    if len(baseline['forwards'])!=len(candidate['forwards']):
        raise ValueError('Different actual forward count')
    for old,new in zip(baseline['forwards'],candidate['forwards'],strict=True):
        for key in ('query','request_index','cache_positions','input_sha256','output_sha256'):
            if old[key]!=new[key]:
                raise ValueError('Changed complete model input/output or state: '+key)
        bounds=(1,1) if new['query']==1 else (2,32)
        if type(new['query']) is not int or not 1<=new['query']<=32 or new['DSO_sha256']!=libraries[bounds]:
            raise ValueError('Actual library is not bound to accepted cold candidate')
        for key in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'):
            if old['invoke'][key]!=new['invoke'][key]:
                raise ValueError('Input checksum changed: '+key)
    return True


def validate_parameter_mode(native,mode):
    if mode=='sha256':
        if native.get('parameter_validation',{}).get('mode','sha256')!='sha256':
            raise ValueError('Actual parameter checker differs from requested SHA mode')
        return
    validation=native.get('parameter_validation',{})
    if (mode!='exact-bytes' or validation.get('mode')!=mode
            or validation.get('all_logical_parameter_bytes_checked_each_forward') is not True
            or validation.get('version_only_check') is not False
            or validation.get('initial_compiler_SHA_pins_preserved') is not True
            or type(validation.get('owned_immutable_snapshot_bytes')) is not int
            or validation['owned_immutable_snapshot_bytes']<=0):
        raise ValueError('Actual exact-byte parameter validation evidence required')


def validate_float_policy(native,specialize):
    actual=native.get('generation_options',{}).get('specialize_python_floats',False)
    if type(actual) is not bool or actual is not specialize:
        raise ValueError('Actual native float capture policy differs from requested mode')
    if specialize:
        validation=native.get('parameter_validation',{})
        ledger=validation.get('guarded_python_float_constants',{})
        if (ledger.get('count')!=142 or ledger.get('FP64_bytes_per_constant')!=8
                or ledger.get('owner_identity_and_all_bytes_checked_each_forward') is not True
                or type(validation.get('guarded_python_float_forwards_checked')) is not int
                or validation['guarded_python_float_forwards_checked']!=native['DSP_calls']
                or type(ledger.get('ledger_sha256')) is not str or len(ledger['ledger_sha256'])!=64):
            raise ValueError('Actual original module float byte guards missing')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-suite','accepted-run','runtime-catalog','board-runtime-catalog','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--cases',nargs='+',default=['tails','eos','old-long32','prefill1024','decode993','code64'])
    parser.add_argument('--deadline-unix',type=int,required=True)
    parser.add_argument('--parameter-check',choices=('sha256','exact-bytes'),default='sha256')
    parser.add_argument('--specialize-floats',action='store_true')
    args=parser.parse_args()
    if (not args.output.is_absolute() or args.output.exists() or args.output.is_symlink()
            or not args.name or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for x in args.name)
            or len(args.cases)!=len(set(args.cases)) or time.time()>=args.deadline_unix):
        parser.error('Fresh output, bounded deadline and unique named state cases required')
    base,libraries=candidate_libraries(args.accepted_run)
    final=json.loads((args.baseline_suite/'final-result.json').read_text())
    if final['status']!='all_post_cleanup_regressions_passed_and_independently_rechecked':
        raise ValueError('Fully accepted original state suite required')
    original=json.loads((args.baseline_suite/'plan.json').read_text())
    available={case['case']:case for case in original['cases']}
    if not args.cases or set(args.cases)-available.keys():
        raise ValueError('Only existing accepted state cases may run')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed diagnostic source required')
    baselines={};rows=[]
    for name in args.cases:
        old=available[name];command=old['command']
        require_configured_python(json.loads(Path(command[command.index('--config')+1]).read_text()))
        _,native,_=accepted(args.baseline_suite/name)
        if native['DSP_calls']!=old['expected_DSP_calls']:
            raise ValueError('Original state count differs')
        baselines[name]=native
        command=derive_command(command,source=ROOT,output=args.output/name,
            name=args.name+'-'+name,host_catalog=args.runtime_catalog,board_catalog=args.board_runtime_catalog)
        replace(command,'--accepted-run',args.accepted_run)
        if args.parameter_check!='sha256':command+=['--parameter-check',args.parameter_check]
        if args.specialize_floats:command+=['--specialize-floats']
        rows.append(dict(case=name,expected_DSP_calls=old['expected_DSP_calls'],command=command))
    args.output.mkdir(parents=True)
    save(args.output/'plan.json',dict(source_revision=revision,accepted_cold_source_revision=base['revision'],
        baseline_plan_sha256=sha(args.baseline_suite/'plan.json'),baseline_final_sha256=sha(args.baseline_suite/'final-result.json'),
        candidate_acceptance_sha256=sha(args.accepted_run/'acceptance.json'),
        runtime_catalog_sha256=sha(args.runtime_catalog),deadline_unix=args.deadline_unix,cases=rows,
        parameter_check=args.parameter_check,
        specialize_python_floats=args.specialize_floats,
        candidate_model_libraries={str(k):v for k,v in libraries.items()}))
    result=dict(status='integer_candidate_state_regressions_running',cases=[],DSP_calls=0)
    save(args.output/'result.json',result)
    try:
        for row in rows:
            if time.time()>=args.deadline_unix:
                result['status']='deadline_reached_before_next_case';return
            if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=revision:
                raise ValueError('Pinned diagnostic queue source changed')
            name=row['case'];save(args.output/'progress.json',dict(stage='real_board_case',case=name,completed_cases=len(result['cases'])))
            with (args.output/(name+'.driver.log')).open('xb') as log:
                subprocess.run(row['command'],check=True,stdout=log,stderr=subprocess.STDOUT)
            acceptance,native,_=accepted(args.output/name)
            validate_float_policy(native,args.specialize_floats)
            if acceptance['numerical_revision']!=base['revision'] or native['DSP_calls']!=row['expected_DSP_calls']:
                raise ValueError('Accepted candidate source/count changed')
            compare_integer(baselines[name],native,libraries)
            validate_parameter_mode(native,args.parameter_check)
            result['cases'].append(dict(case=name,DSP_calls=native['DSP_calls'],
                full_inputs_outputs_tokens_and_FNV_identical=True,actual_libraries_bound_to_candidate=True,
                acceptance=dict(path=str(args.output/name/'acceptance.json'),sha256=sha(args.output/name/'acceptance.json'))))
            result['DSP_calls']+=native['DSP_calls'];save(args.output/'result.json',result)
        result.update(status='all_integer_candidate_state_regressions_passed',numerical_acceptance=True,
            production_throughput_acceptance=False,completed_unix=time.time())
        save(args.output/'progress.json',dict(stage='complete',completed_cases=len(result['cases'])))
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        save(args.output/'result.json',result)
    print(result['status'],result['DSP_calls'])


if __name__=='__main__':main()
