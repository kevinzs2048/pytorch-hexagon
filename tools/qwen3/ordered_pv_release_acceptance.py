"""Accept PV only through its distinct full393 independent audit.

The frozen producer intentionally never grants itself numerical acceptance.
Keep its receipt unchanged; negative89 and packet8 cannot satisfy this gate.
"""
from pathlib import Path
import subprocess
from build import entry, sha

KIND = 'ordered-pv-register-q1'
COMPLETE = 'ordered_PV_original393_complete_NOT_independently_accepted'
AUDITED = 'ordered_PV_original393_independently_revisited'


def require_pv393(closure, proof):
    if (closure.get('candidate_kind') != KIND or closure.get('status') != COMPLETE
            or closure.get('model_inference_acceptance') is not False
            or closure.get('original_state304_executed') is not True
            or closure.get('original_numerical_dependencies_unchanged') is not True
            or type(closure.get('original_full393_minimum_remaining_seconds')) is not int
            or closure['original_full393_minimum_remaining_seconds'] != 14400
            or type(closure.get('new_primary_model_codegen_invocations')) is not int
            or closure['new_primary_model_codegen_invocations'] != 0
            or any(type(closure.get(k)) is not int or closure[k] != n for k, n in
                (('DSP_calls',393),('candidate_DSP_calls',361),('paired_baseline_DSP_calls',32)))):
        raise ValueError('Own original PV393 producer, unchanged budget and state304 required')
    if (proof is None or proof.get('status') != AUDITED or proof.get('candidate_kind') != KIND
            or proof.get('model_inference_acceptance') is not True
            or proof.get('positive_native_gain') is not True
            or proof.get('production_TUI_speedup_claimed') is not False
            or proof.get('source_revision') != closure['source_revision']
            or proof.get('candidate_libraries') != closure['candidate_libraries']
            or any(type(proof.get(k)) is not int or proof[k] != n for k, n in
                (('DSP_calls',393),('new_DSP_calls',0),('candidate_DSP_calls',361),('paired_baseline_DSP_calls',32)))
            or len(proof.get('raw_invocation_traces',[])) != 12
            or len(proof.get('compile_chains',[])) != 12
            or type(proof.get('states',{}).get('DSP_calls')) is not int
            or proof['states']['DSP_calls'] != 304 or len(proof['states'].get('proofs',[])) != 6
            or type(proof.get('native',{}).get('DSP_calls')) is not int
            or proof['native']['DSP_calls'] != 64
            or proof['native'].get('raw_native_measurements') != closure['native_model_timing']):
        raise ValueError('Distinct positive independent PV393, twelve raw sessions and six states required')


def require_pv_sources(closure, proof, source, baseline):
    source = Path(source)
    reviewer = entry(source/'tools/qwen3/revisit_ordered_pv_closure.py')
    if proof.get('audit_source_root') is not None:
        audit = Path(__file__).resolve().parents[2]
        git = lambda *args: subprocess.check_output(['git','-C',str(audit),*args],text=True).strip()
        if (proof['audit_source_root'] != str(audit)
                or not isinstance(proof.get('audit_source_revision'), str)
                or subprocess.run(['git','-C',str(audit),'merge-base','--is-ancestor',
                    proof['audit_source_revision'],'HEAD'],capture_output=True).returncode
                or git('status','--porcelain')
                or proof.get('preparation_reviewer_source') != entry(audit/'tools/qwen3/audit_prepared_stage_reuse.py')):
            raise ValueError('Exact committed independent preparation reviewer required')
        reviewer = entry(audit/'tools/qwen3/revisit_frozen_ordered_pv_closure.py')
        for relative in ('tools/qwen3/audit_prepared_stage_reuse.py', 'tools/qwen3/revisit_frozen_ordered_pv_closure.py'):
            if subprocess.check_output(['git','-C',str(audit),'show',proof['audit_source_revision']+':'+relative]) != (audit/relative).read_bytes():
                raise ValueError('Independent reviewer differs from its actual committed Git blob')
    if (not source.is_absolute() or proof['source_root'] != str(source)
            or proof['reviewer_source'] != reviewer
            or closure['source_sha256'] != sha(source/'tools/qwen3/run_ordered_pv_iteration.py')
            or closure['baseline_closure'] != {k:baseline[k] for k in ('path','sha256')}):
        raise ValueError('Exact frozen PV reviewer, driver and directly paired QK393 required')
