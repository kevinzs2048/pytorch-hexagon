"""Prepare both original dynamic graphs after the distinct positive PV model8.

This does not admit native393; that entry retains its four-hour budget.
Every baseline dependency stays pinned, including the unchanged QK q2 graph.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

from build import ROOT, entry, sha, write
from ordered_pv_model_contract import KIND, require_reference, require_compiler
from revisit_key_tile_model import revisit
from run_integer_regressions import candidate_libraries

CONTRACT = 'original_Qwen_query1_ordered_K1024_PV_columns8_after_N32_GQA_v1'
ADDED = {'query1_ordered_pv_compiler', 'query1_ordered_pv_sdk_receipt',
         'query1_ordered_pv_objects_receipt', 'query1_ordered_pv_contract_source'}
REPLACED = {'compiler_stage_reuse_source', 'prepared_stage_reuse_source',
            'compiler_stage_reuse_config'}


def validate_transition(original, recipe):
    if (len(original['inputs']) != 53
            or original.get('dynamic_qk_invariant_query2_to32') is not True
            or original.get('gqa_key_tile_query1') is not True
            or original.get('reuse_exact_prepared_linalg') is not True
            or original.get('reuse_exact_compiler_objects') is not True
            or set(recipe) != set(original) | {'ordered_pv_register_query1', 'ordered_pv_register_contract'}
            or recipe['ordered_pv_register_query1'] is not True
            or recipe['ordered_pv_register_contract'] != CONTRACT
            or set(recipe['inputs']) != set(original['inputs']) | ADDED
            or any(recipe[k] != v for k, v in original.items() if k != 'inputs')
            or any(recipe['inputs'][k] != v for k, v in original['inputs'].items() if k not in REPLACED)):
        raise ValueError('Only the distinct ordered q1 PV producer and owned reuse paths may extend QK53')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'baseline-cold', 'model-revisit', 'owner-revisit',
                 'accepted-closure', 'accepted-model-audit', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); read = lambda path: json.loads(Path(path).read_text())
    if (not a.output.is_absolute() or a.output.exists()
            or subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Fresh recipe and frozen clean source required')
    reference = entry(a.accepted_closure); closure = read(a.accepted_closure)
    require_reference(closure, read(a.accepted_model_audit), reference)
    base, libraries = candidate_libraries(a.baseline_cold)
    original_path = Path(base['recipe']['path']); original = read(original_path)
    proof = read(a.model_revisit); owners = read(a.owner_revisit)
    if (proof['candidate_kind'] != KIND or type(proof['DSP_calls']) is not int or proof['DSP_calls'] != 8
            or proof['positive_early_model_gain'] is not True
            or proof['full_original_logits_and_56_KV_byteexact'] is not True
            or proof['model_inference_acceptance'] is not False
            or Path(proof['accepted_cold']['path']).parent != a.baseline_cold
            or revisit(Path(proof['run']['path']), Path(proof['plan']['path']), a.baseline_cold) != proof
            or owners['status'] != 'ordered_PV_register_original_QK393_q1_packet8_raw_FNV_and_normal_owners_independently_revisited'
            or owners['packet_revisit'] != entry(a.model_revisit)
            or owners['candidate_kind'] != KIND or type(owners['DSP_calls']) is not int or owners['DSP_calls'] != 8
            or owners['positive_early_model_gain'] is not True
            or owners['model_inference_acceptance'] is not False
            or owners['complete_input_FNV']['all_three_FNV_recomputed'] is not True):
        raise ValueError('Own positive PV packet8 and distinct normally closed owners/FNV required')
    build = read(proof['build']['path']); sdk = read(build['compiler_receipt']['path'])
    code = read(build['objects_receipt']['path'])
    for row in (proof['build'], build['compiler_receipt'], build['objects_receipt']):
        if sha(row['path']) != row['sha256']: raise ValueError('Actually tested PV producer changed')
    if (build['accepted_baseline_closure'] != {k: reference[k] for k in ('path', 'sha256')}
            or build['accepted_original_R393_calls'] != 393
            or build['original_model_library']['sha256'] != libraries[(1, 1)]
            or closure['candidate_libraries'] != {str(k): v for k, v in libraries.items()}):
        raise ValueError('Actual PV q1 producer must derive from this exact complete QK393')
    require_compiler(sdk, Path(code['compiler_diagnostics']['path']).read_text())
    for row in original['inputs'].values():
        if sha(row['path']) != row['sha256']: raise ValueError('Original pinned numerical dependency changed')
    recipe = copy.deepcopy(original)
    recipe.update(ordered_pv_register_query1=True, ordered_pv_register_contract=CONTRACT)
    recipe['inputs'].update(
        query1_ordered_pv_compiler={k: sdk['candidate'][k] for k in ('path', 'sha256')},
        query1_ordered_pv_sdk_receipt=build['compiler_receipt'],
        query1_ordered_pv_objects_receipt=build['objects_receipt'],
        query1_ordered_pv_contract_source=entry(ROOT/'tools/qwen3/ordered_pv_model_contract.py'),
        compiler_stage_reuse_source=entry(ROOT/'tools/qwen3/reuse_accepted_compiler_objects.py'),
        prepared_stage_reuse_source=entry(ROOT/'tools/qwen3/reuse_accepted_prepared_linalg.py'),
        compiler_stage_reuse_config=entry(a.config))
    validate_transition(original, recipe)
    a.output.mkdir(); write(a.output/'machine-recipe.json', recipe)
    write(a.output/'result.json', dict(
        status='ordered_PV_exact_original_two_graph_recipe_built_NOT_native_or_state_acceptance',
        candidate_kind=KIND, source_revision=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=sha(__file__), base_recipe=entry(original_path),
        accepted_closure=reference, accepted_model_audit=entry(a.accepted_model_audit),
        model_revisit=entry(a.model_revisit), owner_revisit=entry(a.owner_revisit),
        recipe=entry(a.output/'machine-recipe.json'),
        original_QK_and_arithmetic_dependencies_unchanged=True,
        changed_graph_candidate_SDK_fallback_forbidden=True,
        new_primary_model_codegen=0, DSP_calls=0,
        minimum_original_full393_start_budget_hours=4, model_inference_acceptance=False))


if __name__ == '__main__': main()
