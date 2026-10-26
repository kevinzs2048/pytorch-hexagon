"""Stage only the exact q1 zero-tail object already checked on the target.

This is compiler-object reuse, not numerical or deployment acceptance.
The old q1 reuse stage must authenticate its complete original policy first.
"""
import copy
import json
from pathlib import Path
import shutil

from hexagon_compiled_artifact_cache import canonical, digest_file

CONTRACT = 'original_q1_PV_runtime_bitwise_positive_zero_suffix_ordered_FMA_v1'
INPUTS = {'zero_tail_code', 'zero_tail_code_owner', 'zero_tail_link',
          'zero_tail_model_review', 'zero_tail_model_review_owner',
          'zero_tail_native_review', 'zero_tail_native_review_owner',
          'zero_tail_runtime'}


def checked(row):
    path = Path(row['path'])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or digest_file(path) != row['sha256']):
        raise ValueError('Pinned zero-tail artifact changed')
    return path


def normal_owner(row, revision):
    owner = json.loads(checked(row).read_text())
    if (owner['status'] != 'completed' or type(owner['returncode']) is not int
            or owner['returncode'] != 0 or owner['source_revision'] != revision
            or any(Path('/proc', str(owner[k])).exists() for k in ('controller_pid', 'child_pid'))):
        raise ValueError('Actually closed normal producer/reviewer required')


def validate_link_runtime(runtime, link_row):
    link = json.loads(checked(link_row).read_text())
    code = json.loads(checked(link['model_code']).read_text())
    checked(runtime)
    checked(link['runtime_source'])
    checked(code['primary'])
    if (link['status'] != 'actual_q1_runtime_extent_model_linked_DIAGNOSTIC_NOT_model_acceptance'
            or link['query_bounds'] != [1, 1]
            or runtime['sha256'] != link['runtime']['sha256']):
        raise ValueError('Exact target-tested q1 integer runtime required')
    return code


def authenticate(inputs):
    for key in INPUTS:
        checked(inputs[key])
    code = validate_link_runtime(inputs['zero_tail_runtime'], inputs['zero_tail_link'])
    if (code['status'] != 'one_existing_q1_zero_tail_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or code['query_bounds'] != [1, 1] or code['model_PRIMARY_compilations'] != 1):
        raise ValueError('One actual original q1 category compilation required')
    normal_owner(inputs['zero_tail_code_owner'], code['source_revision'])
    sdk = json.loads(checked(code['SDK']).read_text())
    checked(sdk['candidate'])
    checked(sdk['runtime_helper'])
    diagnostics = checked(code['compiler_diagnostics'])
    if inputs['zero_tail_code']['sha256'] != json.loads(checked(inputs['zero_tail_link']).read_text())['model_code']['sha256']:
        raise ValueError('Code and tested link provenance differ')
    reviews = [('model', 24, 'actual_q1_zero_tail_original24_full_words_raw_RPC_FNV_ABBA_and_owner_independently_revisited_DIAGNOSTIC'),
               ('native', 16, 'actual_q1_runtime_extent_native16_all_full_words_tokens_KV_3FNV_raw_RPC_and_owner_independently_revisited_DIAGNOSTIC')]
    for kind, count, status in reviews:
        review = json.loads(checked(inputs['zero_tail_' + kind + '_review']).read_text())
        normal_owner(inputs['zero_tail_' + kind + '_review_owner'], review['source_revision'])
        if (review['status'] != status or review['DSP_calls'] != count
                or review['new_DSP_calls'] != 0 or review['model_acceptance']):
            raise ValueError('Full target packet and native generation diagnostic gates required')
        if kind == 'model' and review['build']['sha256'] != inputs['zero_tail_link']['sha256']:
            raise ValueError('Model review belongs to another library')
        if kind == 'native' and review['q1_review']['sha256'] != inputs['zero_tail_model_review']['sha256']:
            raise ValueError('Native review belongs to another model review')
    return code, sdk, diagnostics


def stage(inputs, objects, bounds):
    if tuple(bounds) != (1, 1):
        raise ValueError('Runtime zero-tail staging applies only to q1')
    code, sdk, diagnostics = authenticate(inputs)
    folder = Path(objects)
    old = json.loads((folder / 'result.json').read_text())
    reuse = json.loads((folder / 'reuse-result.json').read_text())
    producer = json.loads(checked(code['original_model']).read_text())
    if (reuse['status'] != 'authenticated_exact_compiler_stage_objects_reused_NOT_model_acceptance'
            or reuse['query_bounds'] != [1, 1] or reuse['new_machine_code_compilations'] != 0
            or old['inputs']['linalg']['sha256'] != code['actual_linalg']['sha256']
            or old['objects'][0]['sha256'] != producer['objects'][0]['sha256']
            or old['objects'][1:] != code['unchanged_constant_objects']
            or digest_file(old['objects'][0]['path']) != producer['objects'][0]['sha256']):
        raise ValueError('Complete original preparation and ordered-PV object must be authenticated first')
    for key in ('overrides', 'option_reconciliation', 'expected_objects_result', 'triton_python'):
        if old[key] != producer[key]:
            raise ValueError('Original compiler options or reference changed')
    checked(code['actual_linalg'])
    (folder / 'before-zero-tail-result.json').write_bytes((folder / 'result.json').read_bytes())
    shutil.copyfile(checked(code['primary']), folder / 'dynamic-hexagon-0.o')
    shutil.copyfile(diagnostics, folder / 'compiler-diagnostics.txt')
    result = copy.deepcopy(old)
    result['objects'][0].update(sha256=code['primary']['sha256'], bytes=(folder / 'dynamic-hexagon-0.o').stat().st_size)
    result['inputs']['compiler'] = sdk['candidate']
    result['compiler_diagnostics']['sha256'] = code['compiler_diagnostics']['sha256']
    result['zero_tail_producer'] = inputs['zero_tail_code']
    result['new_model_object_codegen_invocations'] = 0
    (folder / 'result.json').write_bytes(canonical(result) + b'\n')
    proof = dict(status='exact_q1_zero_tail_object_staged_NOT_model_or_delivery_acceptance',
                 contract=CONTRACT, producer=inputs['zero_tail_code'],
                 model_review=inputs['zero_tail_model_review'], native_review=inputs['zero_tail_native_review'],
                 new_model_object_codegen_invocations=0, DSP_calls=0)
    (folder / 'zero-tail-reuse-result.json').write_bytes(canonical(proof) + b'\n')
