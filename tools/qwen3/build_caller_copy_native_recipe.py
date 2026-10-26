"""Admit only the separately checked owned caller copy to the existing recipe."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
from build import ROOT, entry, sha, write
from dynamic_pv_native_contract import checked
from caller_byte_copy_contract import CONTRACT

OWNED_REUSE = {'compiler_stage_reuse_source', 'prepared_stage_reuse_source'}


def validate_transition(original, recipe):
    added = {'caller_copy_runtime', 'caller_copy_source'}
    if (original.get('ordered_pv_register_query2_to32') is not True
            or original.get('reuse_exact_compiler_objects') is not True
            or original.get('reuse_exact_prepared_linalg') is not True
            or set(recipe) != set(original) | {'caller_hvx_byte_copy', 'caller_hvx_byte_copy_contract'}
            or recipe['caller_hvx_byte_copy'] is not True
            or recipe['caller_hvx_byte_copy_contract'] != CONTRACT
            or set(recipe['inputs']) != set(original['inputs']) | added
            or any(recipe[k] != v for k, v in original.items() if k != 'inputs')
            or any(recipe['inputs'][k] != v for k, v in original['inputs'].items() if k not in OWNED_REUSE)
            or any(recipe['inputs'][k]['sha256'] != original['inputs'][k]['sha256'] for k in OWNED_REUSE)):
        raise ValueError('Only owned byte copying may extend the exact accepted numerical recipe')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe', 'model-review', 'primitive-review', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if (not a.output.is_absolute() or a.output.exists()
            or subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Fresh output and clean committed source required')
    review = json.loads(a.model_review.read_text())
    primitive = json.loads(a.primitive_review.read_text())
    if (review['status'] != 'caller_only_HVX_copy_original_q1_q8_ABBA8_full_output_FNV_raw_RPC_and_normal_owner_independently_revisited_NOT_native393_acceptance'
            or review['DSP_calls'] != 8 or review['new_DSP_calls']
            or review['positive_early_entry_gain'] is not True
            or review['model_math_unchanged'] is not True
            or type(review['transfer_calls_only_changed']) is not int
            or review['transfer_calls_only_changed'] != 117
            or review['model_inference_acceptance'] or review['performance_acceptance']
            or primitive['status'] != 'actual_DSP_caller_copy_498_bounded_cases_complete_bytes_canaries_3FNV_and_normal_owner_independently_revisited_NOT_model_acceptance'
            or primitive['case_count'] != 498 or primitive['DSP_primitive_calls'] != 1
            or primitive['DSP_model_calls'] or primitive['new_DSP_calls']):
        raise ValueError('Actual byte primitive and positive original model gates required')
    model = json.loads(checked(review['build']).read_text())
    for row in model['models']:
        for key in ('original_wrapper', 'candidate_wrapper', 'primary', 'library'):
            checked(row[key])
    original = json.loads(a.recipe.read_text())
    for row in original['inputs'].values():
        checked(row)
    recipe = copy.deepcopy(original)
    recipe.update(caller_hvx_byte_copy=True, caller_hvx_byte_copy_contract=CONTRACT)
    recipe['inputs'].update(caller_copy_runtime=model['runtime'], caller_copy_source=model['copy_source'])
    for key, name in (('compiler_stage_reuse_source', 'reuse_accepted_compiler_objects.py'),
                      ('prepared_stage_reuse_source', 'reuse_accepted_prepared_linalg.py')):
        recipe['inputs'][key] = entry(ROOT/'tools/qwen3'/name)
    for key in ('caller_copy_runtime', 'caller_copy_source'):
        checked(recipe['inputs'][key])
    validate_transition(original, recipe)
    a.output.mkdir()
    write(a.output/'machine-recipe.json', recipe)
    write(a.output/'result.json', dict(
        status='owned_caller_copy_exact_two_graph_recipe_NOT_native393_acceptance',
        source=entry(__file__), source_revision=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        base_recipe=entry(a.recipe), recipe=entry(a.output/'machine-recipe.json'),
        tested_model=review['build'], model_review=entry(a.model_review), primitive_review=entry(a.primitive_review),
        original_model_math_objects_and_owned_lifetimes_unchanged=True,
        new_model_PRIMARY_compilations=0, DSP_calls=0,
        minimum_original_full393_start_budget_hours=4, model_inference_acceptance=False))


if __name__ == '__main__':
    main()
