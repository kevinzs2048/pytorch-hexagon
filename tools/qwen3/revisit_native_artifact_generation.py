"""Revisit original warm or live generation against an existing packet oracle.

No CPU reference or DSP work runs here. The original native selector proves
full graph and immutable binding identity; the existing packet comparison
checks all outputs, selected tokens, cache positions and transport checksums.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from build import ROOT, entry, sha, write
from run_native_aot_generate_arm import (require_artifact_catalog, require_generation_protocol,
    require_original_guard_configuration)
from revisit_native_compile_session_abba import require_normal_owner


def require_artifact_execution(native, catalog, libraries):
    if (native['status'] != 'ARM_native_dynamic_warm_generate_completed'
            or native['live_compiler_handoff'] is not False
            or native['compiler_transactions'] != []
            or type(native['compile_requests_submitted']) is not int
            or native['compile_requests_submitted'] != 0):
        raise ValueError('Original warm selector with no live compiler transactions required')
    rows = {row['graph']['sha256']: row for row in catalog['entries']}
    for capture in native['captures']:
        row = rows.get(capture['graph']['sha256'])
        if (row is None or any(capture[key] != row[key] for key in
                ('graph', 'retained_input_indices', 'contract', 'frontend_key', 'machine_key'))
                or libraries.get(str(tuple(capture['contract']['variables']['query']))) != capture['DSO_sha256']):
            raise ValueError('Actual complete graph, original input binding, ABI, keys and model DSO required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'index', 'oracle', 'native', 'controller', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('artifact-catalog', 'artifact-controller'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--compiler-mode', choices=('original_warm_catalog', 'original_live_handoff'),
        default='original_warm_catalog')
    a = p.parse_args()
    warm = a.compiler_mode == 'original_warm_catalog'
    if (bool(a.artifact_catalog) != warm or bool(a.artifact_controller) != warm):
        p.error('Warm mode requires both artifact proofs; live mode uses its actual mailbox')
    read = lambda path: json.loads(Path(path).read_text())
    git = lambda root, *words: subprocess.check_output(['git', '-C', str(root), *words], text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git(ROOT, 'status', '--porcelain'):
        raise ValueError('Fresh independent report and committed frozen reviewer required')
    producer = read(a.native/'result.json'); run = read(a.native/'native-result.json')
    cache = read(a.native/'AOT-cache-report.json'); index = read(a.index)
    artifact = read(a.artifact_catalog/'result.json') if warm else None
    catalog = read(a.artifact_catalog/'catalog.json') if warm else None
    model = index['accepted_versions'][index['current_default_accepted_version']]
    owner = require_normal_owner(a.controller, read(a.controller)['authorized_deadline_unix'])
    if warm:
        artifact_owner = require_normal_owner(a.artifact_controller,
            read(a.artifact_controller)['authorized_deadline_unix'])
        builder_relative = 'tools/qwen3/build_native_artifact_catalog.py'
        builder_blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            artifact_owner['source_revision']+':'+builder_relative])
        if (artifact_owner['source_root'] != str(ROOT)
                or artifact_owner['source_revision'] != artifact['source_revision']
                or hashlib.sha256(builder_blob).hexdigest() != artifact['source']['sha256']):
            raise ValueError('Actual normal reply-derived catalog builder and original committed source required')
    numerical = Path(model['source_root'])
    if (git(numerical, 'rev-parse', 'HEAD') != model['source_revision']
            or git(numerical, 'status', '--porcelain')
            or owner['source_root'] != str(ROOT)
            or producer['source_revision'] != owner['source_revision']
            or producer['source'] != entry(Path(__file__).with_name('generate_native_aot.py'))
            or producer['accepted_model'] != model
            or producer['accepted_model_version'] != index['current_default_accepted_version']
            or producer['status'] != 'accepted_dynamic_model_native_AOT_generation_completed'
            or producer['compiler_mode'] != a.compiler_mode
            or producer['CPU_reference_executed'] is not False
            or producer['new_prompt_reference_equality_claimed'] is not False
            or (producer['worker'] is None) != warm
            or read(a.native/'run-exit.json') != {'board_returncode': 0, 'worker_returncode': None if warm else 0}
            or any(Path('/proc', str(pid)).exists() for pid in producer['active_processes'].values())):
        raise ValueError('Actual current-model warm generation and normal source-bound owner required')
    manifest = producer['manifest']
    if (manifest['adapter_sha256'] != sha((Path(__file__).resolve().parents[2]/'qwen3/runtime/support/native_aot_output.py'))
            or manifest['numerical_revision'] != model['source_revision']
            or manifest['recipe_sha256'] != index['machine_recipe']['sha256']):
        raise ValueError('Actual current numerical model and unchanged adapter and recipe required')
    if warm and (artifact['accepted_model'] != model
            or artifact['source'] != entry(Path(__file__).with_name('build_native_artifact_catalog.py'))
            or artifact['catalog'] != entry(a.artifact_catalog/'catalog.json')
            or producer['artifact_catalog_preparation'] != entry(a.artifact_catalog/'result.json')
            or catalog['base_runtime_catalog'] != index['runtime_catalog']
            or artifact['full_graph_and_immutable_binding_selector_source'] !=
                entry(numerical/'qwen3/runtime/arm/hexagon_arm_dynamic_catalog.py')):
        raise ValueError('Current original graph selector, accepted artifacts and separate namespace required')
    libraries = require_artifact_catalog(artifact, catalog, manifest['runtime_catalog_sha256'], manifest) if warm else None
    if not warm:
        if (manifest['runtime_catalog_sha256'] != index['runtime_catalog']['sha256']
                or producer['artifact_catalog_preparation'] is not None
                or cache['artifact_catalog_proof_sha256'] is not None):
            raise ValueError('Live execution must retain its original runtime and actual compiler mailbox')
    for deployed in producer['deployed_sources']:
        source = deployed['source']; relative = str(Path(source['path']).relative_to(ROOT))
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', owner['source_revision']+':'+relative])
        if sha(source['path']) != source['sha256'] or hashlib.sha256(blob).hexdigest() != source['sha256']:
            raise ValueError('Actual original integration source changed')
    sys.path[:0] = [str(numerical/'qwen3/runtime/arm'), str(numerical)]
    from compare_performance import accepted, compare
    from hexagon_host_timing import audit_arm_forward_timings
    from bounded_loader_policy import verify_native_layout
    from tools.qwen3.run_integer_regressions import validate_parameter_mode, validate_float_policy
    from analyze_performance import summarize
    acceptance, oracle, _ = accepted(a.oracle)
    if (acceptance['numerical_revision'] != model['source_revision']
            or producer['native_sha256'] != sha(a.native/'native-result.json')
            or producer['DSP_calls'] != run['DSP_calls']
            or cache['status'] != 'original_native_AOT_cached_generation_completed'
            or cache['factory_restored'] is not True or cache['original_configuration_restored'] is not True
            or cache['namespace'] != hashlib.sha256(json.dumps(manifest,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            or run['source_sha256'] != manifest['native_source_sha256']
            or cache['native_source_sha256'] != run['source_sha256']
            or cache['adapter_sha256'] != manifest['adapter_sha256']
            or cache['wrapper_sha256'] != sha(Path(__file__).with_name('run_native_aot_generate_arm.py'))
            or (warm and cache['artifact_catalog_proof_sha256'] != sha(a.artifact_catalog/'result.json'))):
        raise ValueError('Actual complete warm cache protocol and current source identity required')
    if warm:
        require_artifact_execution(run, catalog, libraries)
    else:
        from audit_arm_dynamic_warm_run import audit_compile_handoff
        chain = audit_compile_handoff(run, Path(producer['host_mailbox']), rebuilt=True)
        if chain != producer['compile_chain']:
            raise ValueError('Actual complete original live compiler transactions differ')
    require_original_guard_configuration(cache['original_configuration'])
    protocol = require_generation_protocol(run, cache['events'], cache['framework_cache_counters_delta'],
        compiler_mode=a.compiler_mode)
    if protocol != cache['protocol'] or protocol != producer['cache_protocol']:
        raise ValueError('Actual framework counters and current model rebinding differ')
    validate_parameter_mode(run, 'exact-bytes'); validate_float_policy(run, True)
    if (run['parameter_validation']['guarded_python_float_constants'] !=
            oracle['parameter_validation']['guarded_python_float_constants']
            or run['parameter_validation']['guarded_python_float_constants']['ledger_sha256'] !=
                manifest['module_constants_sha256']):
        raise ValueError('Original complete module constants and current namespace must agree')
    compare(oracle, run); audit_arm_forward_timings(run)
    summarize(run)  # Validate every original generation timer; retain raw spans below.
    if (verify_native_layout(oracle) != verify_native_layout(run)
            or [r['terminal_observation'] for r in oracle['requests']] !=
                [r['terminal_observation'] for r in run['requests']]):
        raise ValueError('All original loader payloads and native stopping behavior required')
    raw = a.native/'stream.stdout.txt'; events = [json.loads(line) for line in raw.read_text().splitlines()]
    if (not events or events[-1]['stage'] != 'close'
            or [e for e in events if e.get('stage') == 'invoke'] != [f['invoke'] for f in run['forwards']]
            or any(e.get(k, 0) for e in events for k in ('rpc_result', 'DSP_result', 'output_result', 'loader_result'))
            or any(len({e[k] for k in ('AP_input_fnv1a32', 'RPC_input_fnv1a32', 'DSP_input_fnv1a32')}) != 1
                   for e in events if e.get('stage') == 'invoke')):
        raise ValueError('Every original raw invocation, triple FNV and normal close required')
    ready = [e for e in events if e.get('stage') == 'ready']
    if len(ready) != 1 or type(ready[0]['ARM_pid']) is not int:
        raise ValueError('Actual AP owner identity required')
    ap = ready[0]['ARM_pid']; config = read(a.config)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', config['board']['ssh_target'],
        shlex.join([config['board']['python'], '-c',
            'from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()', str(ap)])], check=True)
    write(a.output, dict(status=('original_warm_catalog' if warm else 'original_live_handoff')+
        '_native_AOT_all_original_packets_guards_and_owners_independently_revisited_DIAGNOSTIC',
        compiler_mode=a.compiler_mode,
        DSP_calls=run['DSP_calls'], new_DSP_calls=0, native=entry(a.native/'result.json'),
        controller=entry(a.controller), oracle=entry(a.oracle/'acceptance.json'), raw_invocation_trace=entry(raw),
        artifact_catalog_preparation=entry(a.artifact_catalog/'result.json') if warm else None,
        cache_protocol=protocol,
        complete_input_output_token_and_three_FNV_equality=True, original_full_parameter_guards_retained=True,
        actual_AP_owner_gone=ap, generation_timings=[r['generation_timing'] for r in run['requests']],
        model_inference_acceptance=False, performance_acceptance=False, TUI_startup_claimed=False,
        reviewer_source=entry(__file__), reviewer_revision=git(ROOT, 'rev-parse', 'HEAD'), completed_unix=time.time()))


if __name__ == '__main__': main()
