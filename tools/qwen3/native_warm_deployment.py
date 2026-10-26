"""Admit a warm deployment from completed original state and paired receipts."""
import json
import hashlib
from pathlib import Path
import subprocess

from build import HERE, ROOT, entry
from revisit_native_compile_session_abba import require_normal_owner
from revisit_release_index import indexed_files


STATES = [('artifact-state-tails', 6), ('artifact-state-eos', 9),
    ('artifact-state-old-long32', 33), ('artifact-state-prefill1024', 33),
    ('artifact-state-decode993', 95), ('artifact-state-code64', 128)]
WARM = 'original_warm_catalog'
LIVE = 'original_live_handoff'
REVIEW_SUFFIX = '_native_AOT_all_original_packets_guards_and_owners_independently_revisited_DIAGNOSTIC'


def require_closure_shape(state, paired):
    """Require all original transitions and four independently reviewed arms."""
    if (state.get('status') != 'original_warm_catalog_state304_completed_DIAGNOSTIC'
            or type(state.get('DSP_calls')) is not int or state['DSP_calls'] != 304
            or [(row.get('case'), row.get('DSP_calls')) for row in state.get('rows', [])] != STATES
            or paired.get('status') != 'same_input_live_warm_warm_live_ABBA64_completed_DIAGNOSTIC'
            or type(paired.get('DSP_calls')) is not int or paired['DSP_calls'] != 64
            or paired.get('startup_samples_per_mode') != 2
            or [(row.get('order'), row.get('compiler_mode'), row.get('DSP_calls'))
                for row in paired.get('rows', [])] !=
                [(i, mode, 16) for i, mode in enumerate((LIVE, WARM, WARM, LIVE))]):
        raise ValueError('Complete original warm state304 and same-input paired64 required')
    for closure in (state, paired):
        if (closure.get('CPU_reference_executed') is not False
                or closure.get('production_launcher_switched') is not False):
            raise ValueError('Original opt-in execution boundary required')
        for row in closure['rows']:
            if (type(row['DSP_calls']) is not int
                    or not row.get('independent_review') or not row.get('reviewer_controller')
                    or row.get('cache_protocol') != dict(captures=2, hits=2, misses=0, guard_misses=0)):
                raise ValueError('Each original arm needs complete independent review and real cache hits')


def require_warm_deployment(deployment):
    """Rehash normal receipts and bind the currently installed model and sources.

    This is a deployment check, never a per-prompt CPU oracle or a new model
    acceptance. The delegated generator retains its current-model admission.
    """
    read = lambda path: json.loads(Path(path).read_text())
    references = []
    validation = deployment['warm_validation']
    indexed_files(validation, references)
    state = read(validation['state304']['path'])
    paired = read(validation['ABBA64']['path'])
    require_closure_shape(state, paired)
    catalog_root = Path(deployment['artifact_catalog'])
    artifact = read(catalog_root/'result.json')
    catalog = entry(catalog_root/'result.json')
    if catalog != validation['artifact_preparation']:
        raise ValueError('Warm closure must bind this exact artifact preparation')
    builder_owner = require_normal_owner(validation['artifact_controller']['path'],
        read(validation['artifact_controller']['path'])['authorized_deadline_unix'])
    if (builder_owner['source_root'] != str(ROOT)
            or builder_owner['source_revision'] != artifact['source_revision']
            or artifact['source'] != entry(HERE/'build_native_artifact_catalog.py')):
        raise ValueError('Original normal source-bound artifact builder required')
    builder_blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        builder_owner['source_revision']+':tools/qwen3/build_native_artifact_catalog.py'])
    if hashlib.sha256(builder_blob).hexdigest() != artifact['source']['sha256']:
        raise ValueError('Artifact builder must match its actual committed executable source')
    index = read(deployment['index'])
    model = index['accepted_versions'][index['current_default_accepted_version']]
    namespaces = {}
    paired_oracles = []
    for closure in (state, paired):
        indexed_files(closure, references)
        for row in closure['rows']:
            proof = read(row['independent_review']['path'])
            producer = read(row['native']['path'])
            indexed_files(proof, references)
            owner = require_normal_owner(row['controller']['path'], closure['deadline_unix'])
            reviewer = require_normal_owner(row['reviewer_controller']['path'], closure['deadline_unix'])
            mode = row.get('compiler_mode', WARM)
            if (proof['status'] != mode+REVIEW_SUFFIX or proof['compiler_mode'] != mode
                    or proof['DSP_calls'] != row['DSP_calls'] or proof['new_DSP_calls'] != 0
                    or proof['native'] != row['native'] or proof['controller'] != row['controller']
                    or proof['complete_input_output_token_and_three_FNV_equality'] is not True
                    or proof['original_full_parameter_guards_retained'] is not True
                    or proof['cache_protocol'] != row['cache_protocol']
                    or proof['generation_timings'] != row['generation_timings']
                    or proof['reviewer_source'] != entry(HERE/'revisit_native_artifact_generation.py')
                    or proof['reviewer_revision'] != reviewer['source_revision']
                    or owner['source_root'] != str(ROOT) or reviewer['source_root'] != str(ROOT)
                    or producer['source_revision'] != owner['source_revision']
                    or producer['source'] != entry(HERE/'generate_native_aot.py')
                    or producer['accepted_model'] != model
                    or producer['compiler_mode'] != mode
                    or producer['artifact_catalog_preparation'] != (catalog if mode == WARM else None)
                    or proof['artifact_catalog_preparation'] != (catalog if mode == WARM else None)):
                raise ValueError('Normal independent original packets must bind current model, catalog and sources')
            for deployed in producer['deployed_sources']:
                if deployed['source'] != entry(deployed['source']['path']):
                    raise ValueError('Tested ARM integration source changed')
            manifest = producer['manifest']
            previous = namespaces.setdefault(mode, manifest)
            if manifest != previous:
                raise ValueError('Full backend namespace must match in every original arm')
            if closure is paired:
                paired_oracles.append(proof['oracle'])
    if (len(paired_oracles) != 4 or any(o != paired_oracles[0] for o in paired_oracles)
            or artifact['accepted_model'] != model):
        raise ValueError('Same original input oracle and current accepted model required')
    live = dict(namespaces[LIVE]); warm = dict(namespaces[WARM])
    if (live.pop('runtime_catalog_sha256') != index['runtime_catalog']['sha256']
            or warm.pop('runtime_catalog_sha256') != artifact['catalog']['sha256'] or live != warm):
        raise ValueError('Live and warm namespaces may differ only in their proven runtime catalog')
    return catalog_root
