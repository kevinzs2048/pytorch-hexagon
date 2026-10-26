"""Extend accepted PV57 with the separately reviewed dynamic PV producer.

This prepares a candidate only. Full original native393 remains required.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
from build import ROOT, entry, sha, write
from dynamic_pv_native_contract import CONTRACT, checked, checked_references, require_compiler

ADDED={'query2_ordered_pv_compiler','query2_ordered_pv_sdk_receipt',
       'query2_ordered_pv_objects_receipt','query2_ordered_pv_contract_source'}
REPLACED={'compiler_stage_reuse_source','prepared_stage_reuse_source','compiler_stage_reuse_config'}


def validate_transition(original, recipe):
    if (len(original['inputs'])!=57 or original.get('ordered_pv_register_query1') is not True
            or original.get('dynamic_qk_invariant_query2_to32') is not True
            or original.get('reuse_exact_compiler_objects') is not True
            or original.get('reuse_exact_prepared_linalg') is not True
            or set(recipe)!=set(original)|{'ordered_pv_register_query2_to32','ordered_pv_register_query2_contract'}
            or recipe['ordered_pv_register_query2_to32'] is not True
            or recipe['ordered_pv_register_query2_contract']!=CONTRACT
            or set(recipe['inputs'])!=set(original['inputs'])|ADDED
            or any(recipe[k]!=v for k,v in original.items() if k!='inputs')
            or any(recipe['inputs'][k]!=v for k,v in original['inputs'].items() if k not in REPLACED)):
        raise ValueError('Only separately reviewed dynamic PV and owned reuse paths may extend PV57')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','recipe','model-review','long-state-review','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if a.output.exists() or not a.output.is_absolute() or subprocess.check_output(
            ['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Fresh output and clean committed numerical source required')
    proof=read(a.model_review);long=read(a.long_state_review)
    if (proof['status']!='actual_dynamic_PV_candidate_model_ABBA12_full_logits_KV_FNV_original_cold_oracle_and_normal_owners_independently_revisited_NOT_generation_or_state_acceptance'
            or proof['DSP_calls']!=12 or proof['new_DSP_calls'] or proof['model_inference_acceptance'] or proof['performance_acceptance']
            or [row['query'] for row in proof['cases']]!=[8,2,32]
            or long['status']!='synthetic_long_KV_q32_position992_ABBA4_all_full_words_FNV_and_normal_owners_independently_revisited_NOT_generation_or_state_acceptance'
            or long['DSP_calls']!=4 or long['new_DSP_calls'] or long['model_inference_acceptance'] or long['performance_acceptance']):
        raise ValueError('Own actual model12 and synthetic long-state4 normal independent gates required')
    checked_references(proof);checked_references(long)
    model=read(checked(proof['receipt']));sdk=read(checked(model['sdk']));code=read(checked(model['code']))
    require_compiler(sdk,checked(code['compiler_diagnostics']).read_text())
    from ordered_pv_release_acceptance import require_pv393
    require_pv393(read(checked(model['original_model']['acceptance'])),read(checked(model['original_model']['numerical_audit'])))
    original=read(a.recipe)
    for row in original['inputs'].values():checked(row)
    recipe=copy.deepcopy(original)
    recipe.update(ordered_pv_register_query2_to32=True,ordered_pv_register_query2_contract=CONTRACT)
    recipe['inputs'].update(query2_ordered_pv_compiler={k:sdk['candidate'][k] for k in ('path','sha256')},
        query2_ordered_pv_sdk_receipt=model['sdk'],query2_ordered_pv_objects_receipt=model['code'],
        query2_ordered_pv_contract_source=entry(ROOT/'tools/qwen3/dynamic_pv_native_contract.py'),
        compiler_stage_reuse_source=entry(ROOT/'tools/qwen3/reuse_accepted_compiler_objects.py'),
        prepared_stage_reuse_source=entry(ROOT/'tools/qwen3/reuse_accepted_prepared_linalg.py'),
        compiler_stage_reuse_config=entry(a.config))
    validate_transition(original,recipe)
    a.output.mkdir();write(a.output/'machine-recipe.json',recipe)
    write(a.output/'result.json',dict(status='dynamic_PV_exact_original_two_graph_candidate_recipe_NOT_native393_acceptance',
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source=entry(__file__),base_recipe=entry(a.recipe),model_review=entry(a.model_review),long_state_review=entry(a.long_state_review),
        tested_model=proof['receipt'],recipe=entry(a.output/'machine-recipe.json'),
        original_q1_and_all_numerical_dependencies_unchanged=True,changed_graph_SDK_fallback_forbidden=True,
        new_primary_model_codegen=0,DSP_calls=0,minimum_original_full393_start_budget_hours=4,model_inference_acceptance=False))


if __name__=='__main__':main()
