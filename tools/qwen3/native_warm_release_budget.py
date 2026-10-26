"""Reserve time from two completed identical800-call protocols, retaining margin."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from build import entry
from revisit_native_compile_session_abba import require_normal_owner

ORIGINAL_SOURCE_SHA='aca5eace7b4445d61b66b49673af96c4de2f53e07b31e31fff7876c85910cf22'


def reserve_seconds(durations):
    if (len(durations)!=2 or any(type(x) not in (int,float) or not math.isfinite(x)
            or not 0<x<=10800 for x in durations)):
        raise ValueError('Two actual positive completed durations below three hours required')
    return max(18000,math.ceil(max(durations)*1.8/3600)*3600)


def admission_budget(index,references,deadline,now):
    rows=[]
    if references:
        if (len(references)!=2 or index.get('deployment_selection',{}).get('release_key')
                !='qualified_global_copy_native_release'):
            raise ValueError('Measured reserve is explicit and limited to the qualified byte-copy successor')
        for value in references:
            path=Path(value);result=json.loads(path.read_text());folder=path.parent
            owner_path=folder.with_name(folder.name+'-process.json');owner=json.loads(owner_path.read_text())
            require_normal_owner(owner_path,owner['authorized_deadline_unix'])
            root=Path(owner['source_root']);script=root/'tools/qwen3/run_native_warm_release.py'
            blob=subprocess.check_output(['git','-C',str(root),'show',
                owner['source_revision']+':tools/qwen3/run_native_warm_release.py'])
            if (result['status']!='qualified_dynamic_PV_warm_deployment_closed'
                    or result['qualified_DSP_calls']!=800 or result['newly_executed_DSP_calls']!=800
                    or result['reused_completed_DSP_calls']!=0
                    or result['new_model_PRIMARY_compilations']!=0
                    or result['source_root']!=str(root) or result['source_revision']!=owner['source_revision']
                    or owner['command'][1]!=str(script)
                    or owner['command'][owner['command'].index('--output')+1]!=str(folder)
                    or hashlib.sha256(blob).hexdigest()!=ORIGINAL_SOURCE_SHA
                    or owner['authorized_deadline_unix']-owner['start_unix']<21600):
                raise ValueError('Actual normal original unchanged800-call protocol required for time planning')
            rows.append(dict(result=entry(path),controller=entry(owner_path),
                numerical_revision=result['accepted_model']['source_revision'],
                actual_seconds=owner['completed_unix']-owner['start_unix']))
        if len({r['numerical_revision']for r in rows})!=2:
            raise ValueError('Two distinct accepted models must support the reserve estimate')
        required=reserve_seconds([r['actual_seconds']for r in rows])
    else:required=21600
    if deadline-now<required:
        raise ValueError('Full admitted warm protocol reserve required before starting')
    return dict(minimum_remaining_seconds=required,historical_completed_protocols=rows,
        measured_duration_margin=1.8 if rows else None,
        full_original800_and_all_independent_reviews_preserved=True)
