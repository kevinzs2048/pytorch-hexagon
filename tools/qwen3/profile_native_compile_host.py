"""Profile original host cache-hit transactions, forbidding compiler work.

Replays already closed, trusted ARM graph carriers. Full original admission,
parameters, contracts and library bytes are checked by the frozen transaction.
No graph capture, model forward, transport or new compiler output is permitted.
"""
import argparse
import json
from pathlib import Path
import pstats
import runpy
import subprocess
import sys
import time

from build import ROOT, entry, sha, write


FORBIDDEN = {'probe_hexagon_dynamic_linalg.py', 'clang', 'clang++', 'hexagon-clang',
             'hexagon-clang++', 'mlir-opt', 'triton-opt', 'triton-translate', 'ld.lld'}


def require_cache_hit(result, original):
    if (result.get('status') != 'carried_ARM_Qwen_graph_bound_to_Hexagon_library_NOT_DSP'
            or result.get('CPU_model_forward') is not False
            or result.get('numerical_acceptance') is not False
            or any(type(result.get(key)) is not int or result[key] != 0 for key in
                   ('DSP_calls', 'frontend_invocations', 'machine_code_compilations'))
            or result['frontend_cache']['cache_hit'] is not True
            or result['machine_cache']['cache_hit'] is not True
            or result['compiled_entry'] != original['compiled_entry']
            or result['immutable_tensors_verified'] != original['immutable_tensors_verified']
            or result['buffer_capacities'] != original['buffer_capacities']):
        raise ValueError('Exact original full-parameter cache-hit transaction required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'numerical-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    git = lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    if (not args.output.is_absolute() or args.output.exists() or git(ROOT,'status','--porcelain')
            or git(args.numerical_source,'status','--porcelain')):
        raise ValueError('Fresh scoped profile and frozen clean sources required')
    acceptance = json.loads((args.oracle/'acceptance.json').read_text())
    commands = json.loads((args.oracle/'commands.json').read_text())
    if (acceptance['numerical_acceptance'] is not True
            or commands['numerical_worker_source'] != str(args.numerical_source)
            or acceptance['numerical_revision'] != git(args.numerical_source,'rev-parse','HEAD')):
        raise ValueError('Own closed numerical oracle and exact original host compiler required')
    worker = commands['worker']; mailbox = Path(worker[worker.index('--mailbox')+1])
    script = args.numerical_source/'qwen3/runtime/arm/compile_qwen_symbolic_carrier.py'
    sys.path[:0] = [str(script.parent), str(args.numerical_source)]
    args.output.mkdir(); rows = []
    record = dict(status='running', rows=rows, source_revision=git(ROOT,'rev-parse','HEAD'),
        source_sha256=sha(__file__), original_compiler=entry(script), oracle=entry(args.oracle/'acceptance.json'),
        DSP_calls=0, CPU_model_forward_calls=0, new_model_PRIMARY_compilations=0,
        model_inference_acceptance=False, performance_acceptance=False,
        profiled_with_shared_interpreter=True, profiler_used_for_speedup_claim=False)
    def save():write(args.output/'result.json', record)
    save(); original_popen = subprocess.Popen; old_argv = sys.argv
    def no_compiler(command, *positional, **kwargs):
        if isinstance(command, (str, bytes)) or any(Path(str(word)).name in FORBIDDEN for word in command):
            raise ValueError('Cache-hit host profile forbids shell commands and all compiler work')
        return original_popen(command,*positional,**kwargs)
    try:
        for folder in sorted(mailbox.glob('capture-*')):
            prior_path = folder/'compiled/result.json'
            prior = json.loads(prior_path.read_text()); require_cache_hit(prior, prior)
            if sha(prior['producer']['path']) != prior['producer']['sha256']:
                raise ValueError('Actual original ARM graph producer changed')
            local = args.output/folder.name; local.mkdir(); output = local/'compiled'
            command = [str(script),'--producer',prior['producer']['path'],
                '--producer-sha256',prior['producer']['sha256'],'--trusted-local-producer','--output',str(output)]
            for key in ('context','catalog','catalog-sha256','frontend-cache','machine-cache','machine-recipe','machine-recipe-sha256'):
                command.extend(['--'+key, worker[worker.index('--'+key)+1]])
            write(local/'command.json',command)
            import cProfile
            profile = cProfile.Profile(); start = time.perf_counter()
            sys.argv = command; subprocess.Popen = no_compiler
            try:profile.enable();runpy.run_path(str(script),run_name='__main__')
            finally:
                profile.disable(); subprocess.Popen = original_popen; sys.argv = old_argv
                profile.dump_stats(str(local/'compile.pstats'))
            result = json.loads((output/'result.json').read_text());require_cache_hit(result,prior)
            library = result['compiled_entry']['library']
            if sha(library['path']) != library['sha256']:
                raise ValueError('Actual original model library bytes changed')
            stats = pstats.Stats(profile)
            hotspots = [dict(file=name[0],line=name[1],function=name[2],primitive_calls=value[0],
                total_calls=value[1],exclusive_seconds=value[2],nested_cumulative_seconds=value[3])
                for name,value in sorted(stats.stats.items(),key=lambda item:item[1][2],reverse=True)[:30]]
            rows.append(dict(query_bounds=result['compiled_entry']['contract']['variables']['query'],
                original=entry(prior_path), result=entry(output/'result.json'), profile=entry(local/'compile.pstats'),
                wall_seconds_under_profiler=time.perf_counter()-start, total_profile_calls=stats.total_calls,
                hottest_exclusive_functions=hotspots, exact_original_compiled_entry=True))
            save()
        if {tuple(row['query_bounds']) for row in rows} != {(1,1),(2,32)} or len(rows) != 2:
            raise ValueError('Exactly the original two actual dynamic query categories required')
        record.update(status='original_host_two_cache_hit_transactions_cProfile_complete_NOT_performance_acceptance',
                      completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        subprocess.Popen = original_popen; sys.argv = old_argv;save()


if __name__ == '__main__':
    main()
