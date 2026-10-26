"""Prepare an explicit diagnostic q1 zero-tail recipe after actual target gates."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from build import ROOT, entry, write

sys.path.insert(0, str(ROOT/'qwen3/runtime/arm'))
from hexagon_zero_tail import CONTRACT, INPUTS, authenticate, checked


def validate_transition(original, recipe):
    if (original.get('caller_hvx_byte_copy') is not True
            or original.get('ordered_pv_register_query2_to32') is not True
            or set(recipe) != set(original) | {'zero_tail_query1', 'zero_tail_contract'}
            or recipe['zero_tail_query1'] is not True or recipe['zero_tail_contract'] != CONTRACT
            or set(recipe['inputs']) != set(original['inputs']) | INPUTS
            or any(recipe[k] != v for k, v in original.items() if k != 'inputs')):
        raise ValueError('Only the explicit q1 runtime-extent recipe may extend the qualified two-graph recipe')
    for name, row in original['inputs'].items():
        current = recipe['inputs'][name]
        if current != row and (name not in {'compiler_stage_reuse_source', 'prepared_stage_reuse_source'}
                              or current['sha256'] != row['sha256']):
            raise ValueError('Original preparation, q2, runtime and numerical inputs must remain identical')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe', 'output', *sorted(INPUTS)):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status', '--porcelain'):
        raise ValueError('Fresh output and clean committed source required')
    original = json.loads(a.recipe.read_text())
    for row in original['inputs'].values():
        checked(row)
    recipe = copy.deepcopy(original)
    recipe.update(zero_tail_query1=True, zero_tail_contract=CONTRACT)
    recipe['inputs'].update({name: entry(getattr(a, name)) for name in INPUTS})
    for name, filename in [('compiler_stage_reuse_source', 'reuse_accepted_compiler_objects.py'),
                           ('prepared_stage_reuse_source', 'reuse_accepted_prepared_linalg.py')]:
        recipe['inputs'][name] = entry(ROOT/'tools/qwen3'/filename)
    validate_transition(original, recipe)
    authenticate(recipe['inputs'])
    a.output.mkdir()
    write(a.output/'machine-recipe.json', recipe)
    write(a.output/'result.json', dict(status='explicit_q1_zero_tail_recipe_NOT_full393_or_delivery_acceptance',
          source=entry(__file__), source_revision=git('rev-parse', 'HEAD'), base_recipe=entry(a.recipe),
          recipe=entry(a.output/'machine-recipe.json'), DSP_calls=0, new_model_PRIMARY_compilations=0,
          model_inference_acceptance=False, minimum_original_full393_start_budget_hours=4))


if __name__ == '__main__':
    main()
