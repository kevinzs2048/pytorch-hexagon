"""Extend only the accepted linker with the independently tested byte dispatcher."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from build import ROOT,entry,write
sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
from hexagon_global_memcpy import CONTRACT,INPUTS,authenticate,checked


def validate_transition(original,recipe):
    flags={'global_hvx_memcpy','global_hvx_memcpy_contract'}
    if (original.get('q2_qk_dispatch')is not True or original.get('caller_hvx_byte_copy')is not True
            or original.get('global_hvx_memcpy',False)is not False
            or recipe.get('global_hvx_memcpy')is not True or recipe.get('global_hvx_memcpy_contract')!=CONTRACT
            or set(recipe)!=set(original)|flags or set(recipe['inputs'])!=set(original['inputs'])|INPUTS
            or any(recipe[k]!=v for k,v in original.items()if k not in flags|{'inputs'})):
        raise ValueError('Only explicit model-local memcpy may extend the complete accepted recipe')
    for name,row in original['inputs'].items():
        current=recipe['inputs'][name]
        if current!=row and (name not in {'compiler_stage_reuse_source','prepared_stage_reuse_source'}
                or current['sha256']!=row['sha256']):
            raise ValueError('Only byte-identical owned compiler reuse scripts may relocate')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','output',*sorted(INPUTS)):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status','--porcelain'):
        raise ValueError('Fresh committed global-copy recipe required')
    original=json.loads(a.recipe.read_text())
    for row in original['inputs'].values():checked(row)
    recipe=copy.deepcopy(original);recipe.update(global_hvx_memcpy=True,global_hvx_memcpy_contract=CONTRACT)
    recipe['inputs'].update({name:entry(getattr(a,name))for name in INPUTS})
    for name,filename in [('compiler_stage_reuse_source','reuse_accepted_compiler_objects.py'),
                          ('prepared_stage_reuse_source','reuse_accepted_prepared_linalg.py')]:
        recipe['inputs'][name]=entry(ROOT/'tools/qwen3'/filename)
    validate_transition(original,recipe);_,build=authenticate(recipe['inputs'])
    index=json.loads(checked(build['accepted_index']).read_text())
    if entry(a.recipe)['sha256']!=index['machine_recipe']['sha256']:
        raise ValueError('Actual currently accepted baseline recipe must precede global copying')
    a.output.mkdir();write(a.output/'machine-recipe.json',recipe)
    write(a.output/'result.json',dict(status='explicit_model_local_global_copy_recipe_NOT_model_or_generation_acceptance',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        base_recipe=entry(a.recipe),recipe=entry(a.output/'machine-recipe.json'),
        DSP_calls=0,new_model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False))


if __name__=='__main__':main()
