"""Add only the separately proved q2 tail stage to the fully accepted two-domain recipe."""
from qwen3.compiler_dependency import compiler_root
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from build import ROOT,entry,write
from native_model_acceptance import require_model_acceptance
sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
from hexagon_q2_identical_tail import CONTRACT,INPUTS,authenticate,checked

RELOCATABLE={'compiler_stage_reuse_source','prepared_stage_reuse_source','global_copy_linker','global_copy_preserver','global_copy_source'}

def validate_transition(original,recipe):
    flags={'q2_qk_identical_tail','q2_qk_identical_tail_contract'}
    if (original.get('global_hvx_memcpy')is not True or original.get('q2_qk_dispatch')is not True
            or original.get('q2_qk_identical_tail',False)is not False
            or recipe.get('q2_qk_identical_tail')is not True or recipe.get('q2_qk_identical_tail_contract')!=CONTRACT
            or set(recipe)!=set(original)|flags or set(recipe['inputs'])!=set(original['inputs'])|INPUTS
            or any(recipe[k]!=v for k,v in original.items()if k not in flags|{'inputs'})):
        raise ValueError('Only explicit q2 identical-tail proof may extend the accepted recipe')
    for name,row in original['inputs'].items():
        current=recipe['inputs'][name]
        if current!=row and (name not in RELOCATABLE or current['sha256']!=row['sha256']):
            raise ValueError('Only byte-identical owned reuse/linker/source scripts may relocate')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','index','output',*sorted(INPUTS)):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists()or git('status','--porcelain'):raise ValueError('Fresh frozen tail recipe required')
    original=json.loads(a.recipe.read_text());index=json.loads(a.index.read_text());require_model_acceptance(index)
    if index['machine_recipe']!=entry(a.recipe)or index['deployment_selection']['release_key']!='qualified_global_copy_native_release':
        raise ValueError('The actual fully accepted current global-copy recipe required')
    for row in original['inputs'].values():checked(row)
    recipe=copy.deepcopy(original);recipe.update(q2_qk_identical_tail=True,q2_qk_identical_tail_contract=CONTRACT)
    recipe['inputs'].update({name:entry(getattr(a,name))for name in INPUTS})
    local={'compiler_stage_reuse_source':ROOT/'tools/qwen3/reuse_accepted_compiler_objects.py',
        'prepared_stage_reuse_source':ROOT/'tools/qwen3/reuse_accepted_prepared_linalg.py',
        'global_copy_linker':ROOT/'qwen3/runtime/arm/hexagon_global_memcpy.py',
        'global_copy_preserver':ROOT/'qwen3/runtime/arm/hexagon_preserved_memcpy.py',
        'global_copy_source':compiler_root()/'tools/qwen3/runtime/profile/global_hvx_byte_copy.cpp'}
    recipe['inputs'].update({name:entry(path)for name,path in local.items()})
    validate_transition(original,recipe);proof=authenticate(recipe['inputs'])
    if proof['manifest']['index']!=entry(a.index):raise ValueError('Actual model controls must bind this exact qualified baseline')
    a.output.mkdir();write(a.output/'machine-recipe.json',recipe)
    write(a.output/'result.json',dict(status='explicit_Q2_identical_tail_recipe_NOT_model_or_generation_acceptance',source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        base_recipe=entry(a.recipe),accepted_index=entry(a.index),recipe=entry(a.output/'machine-recipe.json'),DSP_calls=0,new_model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False))

if __name__=='__main__':main()
