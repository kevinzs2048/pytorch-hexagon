"""Extend the accepted QK recipe only with target-checked dynamic Q2 object reuse."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from build import ROOT, entry, write

sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
from hexagon_q2_qk_dispatch import CONTRACT, INPUTS, authenticate
from hexagon_zero_tail import checked


def validate_transition(original, recipe):
    if (original.get('qk_tail_query1') is not True
            or original.get('ordered_pv_register_query2_to32') is not True
            or original.get('reuse_exact_compiler_objects') is not True
            or original.get('reuse_exact_prepared_linalg') is not True
            or set(recipe) != set(original)|{'q2_qk_dispatch','q2_qk_dispatch_contract'}
            or recipe['q2_qk_dispatch'] is not True or recipe['q2_qk_dispatch_contract'] != CONTRACT
            or set(recipe['inputs']) != set(original['inputs'])|INPUTS
            or any(recipe[k] != v for k,v in original.items() if k != 'inputs')):
        raise ValueError('Only explicit q2 QK dispatch may extend the complete unchanged accepted QK/PV recipe')
    for name,row in original['inputs'].items():
        current = recipe['inputs'][name]
        if current != row and (name not in {'compiler_stage_reuse_source','prepared_stage_reuse_source'}
                or current['sha256'] != row['sha256']):
            raise ValueError('Only byte-identical owned compiler reuse scripts may relocate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','output',*sorted(INPUTS)):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a = p.parse_args()
    git = lambda *x: subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status','--porcelain'):
        raise ValueError('Fresh recipe and frozen committed source required')
    original = json.loads(a.recipe.read_text())
    for row in original['inputs'].values():
        checked(row)
    recipe = copy.deepcopy(original)
    recipe.update(q2_qk_dispatch=True,q2_qk_dispatch_contract=CONTRACT)
    recipe['inputs'].update({key:entry(getattr(a,key)) for key in INPUTS})
    for name,filename in (('compiler_stage_reuse_source','reuse_accepted_compiler_objects.py'),
                          ('prepared_stage_reuse_source','reuse_accepted_prepared_linalg.py')):
        recipe['inputs'][name] = entry(ROOT/'tools/qwen3'/filename)
    validate_transition(original,recipe); authenticate(recipe['inputs'])
    a.output.mkdir(); write(a.output/'machine-recipe.json',recipe)
    write(a.output/'result.json',dict(status='explicit_Q2_QK_dispatch_recipe_NOT_generation_full393_or_delivery_acceptance',
          source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
          base_recipe=entry(a.recipe),recipe=entry(a.output/'machine-recipe.json'),
          DSP_calls=0,model_PRIMARY_compilations=0,model_inference_acceptance=False))


if __name__ == '__main__':
    main()
