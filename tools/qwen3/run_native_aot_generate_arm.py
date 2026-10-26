"""Reuse native AOT artifacts for the original generation arguments.

The immutable native entry owns model loading, all parameter guards, the
compiler, DSP callbacks and generation. A request may capture q1, q2..32, or
both. This entry never runs a CPU reference or declares new-prompt equality.
"""
from qwen3.runtime_checks import require_original_guard_configuration, require_generation_protocol

import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import signal
import sys
import time


def require_artifact_catalog(proof, catalog, catalog_sha, manifest):
    if (proof.get('status') != 'accepted_dynamic_model_artifact_catalog_prepared_NOT_device_acceptance'
            or proof.get('catalog', {}).get('sha256') != catalog_sha
            or proof.get('accepted_model', {}).get('source_revision') != manifest['numerical_revision']
            or proof.get('original_runtime_artifacts_preserved') is not True
            or any(type(proof.get(k)) is not int or proof[k] != 0 for k in
                   ('DSP_calls', 'new_model_PRIMARY_compilations', 'new_frontend_invocations'))
            or catalog.get('schema') != 1
            or catalog.get('status') != 'warm_ARM_dynamic_catalog_NOT_executed'
            or len(catalog.get('entries', [])) != 2):
        raise ValueError('Actual reply-derived current-model catalog and unchanged runtime required')
    rows = catalog['entries']
    libraries = {str(tuple(r['contract']['variables']['query'])): r['library']['sha256'] for r in rows}
    if (set(libraries) != {'(1, 1)', '(2, 32)'} or libraries != proof['candidate_libraries']
            or len({r['graph']['sha256'] for r in rows}) != 2
            or len({r['binding_content_sha256'] for r in rows}) != 1):
        raise ValueError('Both original full-identity graphs and exact current accepted libraries required')
    return libraries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-source', 'namespace-manifest', 'cache-root', 'cache-report'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--artifact-catalog-proof', type=Path)
    p.add_argument('--artifact-catalog-proof-sha256')
    a, remaining = p.parse_known_args()
    if (platform.machine() != 'aarch64' or not a.native_source.is_absolute()
            or not a.native_source.is_file() or not a.cache_report.is_absolute()
            or a.cache_report.exists() or remaining.count('--output') != 1):
        p.error('Actual ARM, original source, fresh report and one native output required')
    output_index = remaining.index('--output')
    if output_index + 1 >= len(remaining):
        p.error('Native output path is missing')
    native_output = Path(remaining[output_index + 1])
    if not native_output.is_absolute() or native_output.exists():
        p.error('Fresh absolute original native output required')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = json.loads(a.namespace_manifest.read_text())
    if sha(a.native_source) != manifest['native_source_sha256']:
        raise ValueError('Original native source differs from namespace')
    for flag, expected in (('--catalog-sha256', manifest['runtime_catalog_sha256']),
                           ('--parameter-check', 'exact-bytes')):
        if (remaining.count(flag) != 1 or remaining.index(flag) + 1 >= len(remaining)
                or remaining[remaining.index(flag) + 1] != expected):
            p.error('Original runtime and full parameter-check arguments required')
    if '--specialize-floats' not in remaining:
        p.error('Original complete float guards required')
    if bool(a.artifact_catalog_proof) != bool(a.artifact_catalog_proof_sha256):
        p.error('Paired artifact catalog preparation proof and exact digest required')
    artifact_libraries = None
    if a.artifact_catalog_proof:
        if ('--compiler-mailbox' in remaining or '--startup-timing' in remaining
                or remaining.count('--catalog') != 1
                or sha(a.artifact_catalog_proof) != a.artifact_catalog_proof_sha256):
            p.error('Original warm catalog mode and unchanged owned preparation proof required')
        catalog_path = Path(remaining[remaining.index('--catalog') + 1])
        if sha(catalog_path) != manifest['runtime_catalog_sha256']:
            raise ValueError('Actual original artifact catalog differs from namespace')
        artifact_libraries = require_artifact_catalog(json.loads(a.artifact_catalog_proof.read_text()),
            json.loads(catalog_path.read_text()), sha(catalog_path), manifest)
    elif '--compiler-mailbox' not in remaining:
        p.error('Original live compiler mailbox required')
    native_root = a.native_source.parents[3]
    sys.path[:0] = [str(a.native_source.parent), str(native_root/'tools/qwen3'), str(native_root)]
    import torch
    import torch._dynamo.backends.common as common
    import torch._functorch.config as config
    from torch._dynamo.utils import counters
    from qwen3.runtime.support.native_aot_output import native_cache_scope, namespace_digest
    from tools.qwen3.run_integer_regressions import validate_parameter_mode, validate_float_policy
    def settings():
        return dict(bundled_autograd_cache=config.bundled_autograd_cache,
            enable_autograd_cache=config.enable_autograd_cache,
            bypass_autograd_cache_key=config.bypass_autograd_cache_key,
            unsafe_skip_cache_dynamic_shape_guards=torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards,
            caching_precompile=torch._dynamo.config.caching_precompile)
    events = []
    before = dict(counters['aot_autograd'])
    original = common.aot_autograd
    old_argv = sys.argv
    previous_handlers = {sig: signal.getsignal(sig) for sig in
                         (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    def interrupted(signum, frame):
        raise InterruptedError('Original native generation interrupted: ' + str(signum))
    original_config = settings()
    require_original_guard_configuration(original_config)
    record = dict(status='running', namespace=namespace_digest(manifest), events=events,
        native_source_sha256=sha(a.native_source), wrapper_sha256=sha(__file__),
        adapter_sha256=manifest['adapter_sha256'], manifest_sha256=sha(a.namespace_manifest),
        cache_root=str(a.cache_root), original_configuration=original_config,
        installed_sources_modified=False, production_configuration_changed=False,
        CPU_reference_executed=False, new_prompt_reference_equality_claimed=False,
        performance_acceptance=False, start_unix=time.time())
    record['compiler_mode'] = 'original_warm_catalog' if artifact_libraries is not None else 'original_live_handoff'
    record['artifact_catalog_proof_sha256'] = a.artifact_catalog_proof_sha256
    def save():
        temporary = a.cache_report.with_suffix('.pending.json')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(a.cache_report)
    save()
    try:
        for sig in previous_handlers:
            signal.signal(sig, interrupted)
        sys.argv = [str(a.native_source), *remaining]
        with native_cache_scope(manifest, a.cache_root, events):
            runpy.run_path(str(a.native_source), run_name='__main__')
        native = json.loads((native_output / 'result.json').read_text())
        delta = {k: v - before.get(k, 0) for k, v in counters['aot_autograd'].items()}
        validate_parameter_mode(native, 'exact-bytes')
        validate_float_policy(native, True)
        if (native['source_sha256'] != manifest['native_source_sha256']
                or native['parameter_validation']['guarded_python_float_constants']['ledger_sha256']
                != manifest['module_constants_sha256']):
            raise ValueError('Actual original source and module constants differ from namespace')
        if artifact_libraries is not None:
            if (native['live_compiler_handoff'] is not False or native['compiler_transactions']
                    or native['compile_requests_submitted'] != 0
                    or any(artifact_libraries.get(str(tuple(c['contract']['variables']['query']))) != c['DSO_sha256']
                           for c in native['captures'])):
                raise ValueError('Original full-binding warm lookup and exact accepted library execution required')
        record.update(status='original_native_AOT_cached_generation_completed',
            protocol=require_generation_protocol(native, events, delta,
                compiler_mode=record['compiler_mode']), DSP_calls=native['DSP_calls'])
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        sys.argv = old_argv
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        record.update(completed_unix=time.time(), factory_restored=common.aot_autograd is original,
            original_configuration_restored=original_config == settings(),
            framework_cache_counters_delta={k: v - before.get(k, 0) for k, v in counters['aot_autograd'].items()})
        save()


if __name__ == '__main__':
    main()
