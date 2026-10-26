"""Select only an independently accepted full393 Q2 model into an opt-in index."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
from build import ROOT, entry, write
from native_model_acceptance import require_model_acceptance, selected_dynamic_model
from profile_model_memory import checked
from q2_dispatch_release import FULL_STATUS, REVIEW_STATUS
from revisit_q2_dispatch_preflight import actual_owner
from revisit_native_compile_session_abba import require_normal_owner

RELEASE_KEY = 'qualified_q2_qk_dispatch_native_release'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-index','closure','controller','review','review-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    if a.output.exists() or not a.output.is_absolute() or subprocess.check_output(
            ['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Fresh committed explicit Q2 release selection required')
    result = read(a.closure/'result.json'); proof = read(a.review)
    actual_owner(a.controller,result,'run_q2_dispatch_native393.py',a.closure)
    reviewer = read(a.review_controller)
    require_normal_owner(a.review_controller,reviewer['authorized_deadline_unix'])
    source = checked(proof['reviewer_source'])
    if (result['status'] != FULL_STATUS or proof['status'] != REVIEW_STATUS
            or proof['Q2_QK_dispatch_candidate'] is not True or proof['model_inference_acceptance'] is not True
            or proof['closure'] != entry(a.closure/'result.json') or proof['controller'] != entry(a.controller)
            or proof['positive_prefill_native_gain'] is not True
            or reviewer['source_revision'] != proof['reviewer_revision']
            or reviewer['source_root'] != str(source.parents[2]) or reviewer['command'][1] != str(source)
            or reviewer['command'][reviewer['command'].index('--output')+1] != str(a.review)):
        raise ValueError('Own exact independent Q2 model393 and genuine normal whole/model-review owners required')
    parent = copy.deepcopy(read(a.parent_index))
    version = 'ordered_q2_runtime_QK_dispatch_dynamic_Qwen_W8A8'
    parent[RELEASE_KEY] = dict(version=version,source_root=result['numerical_source_root'],
        source_revision=result['numerical_source_revision'],closure=entry(a.closure/'result.json'),
        independent_audit=entry(a.review),candidate_libraries=result['candidate_libraries'],
        recipe=result['recipe'],model_inference_acceptance=True,QK_guard_candidate=True,Q2_QK_dispatch_candidate=True)
    a.output.mkdir(); write(a.output/'parent-index.json',parent)
    index = copy.deepcopy(parent)
    index.update(deployment_selection=dict(parent_index=entry(a.output/'parent-index.json'),release_key=RELEASE_KEY),
                 current_default_accepted_version=version,candidate_libraries=result['candidate_libraries'],
                 machine_recipe=result['recipe'])
    index['accepted_versions'][version] = selected_dynamic_model(parent,RELEASE_KEY)
    require_model_acceptance(index)
    write(a.output/'index.json',index)
    write(a.output/'result.json',dict(status='independently_qualified_Q2_QK_dispatch_opt_in_index_prepared_NOT_warm_delivery',
          source_root=str(ROOT),source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
          source=entry(__file__),parent_index=entry(a.parent_index),index=entry(a.output/'index.json'),
          shared_default_changed=False,DSP_calls=0,model_PRIMARY_compilations=0))


if __name__ == '__main__':
    main()
