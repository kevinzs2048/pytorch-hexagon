"""Deploy a committed host packer while proving all numerical files unchanged."""
import json
from pathlib import Path
import shlex
import subprocess

from build import ROOT, entry, sha

ALLOWED = {
    'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py',
    'tools/qwen3/joined_input_packet.py',
    'tools/qwen3/test_joined_input_packet.py',
    'tools/qwen3/deploy_input_packing.py',
    'tools/qwen3/probe_joined_input_packet.py',
    'tools/qwen3/run_online_regression.py',
    'tools/qwen3/benchmark_accepted_generation.py',
    'tools/qwen3/run_input_packing_host_probe.py',
    'tools/qwen3/run_native.py',
    'tools/qwen3/run_integer_regressions.py',
    'tools/qwen3/run_input_packing_iteration.py',
    'tools/qwen3/revisit_input_packing_closure.py',
    'tools/qwen3/test_input_packing_gain.py',
}

COMPARE = '''
import hashlib,json,sys
from pathlib import Path
old,new=map(Path,sys.argv[1:3]);expected=json.loads(sys.argv[3])
def inventory(root):
 result={}
 for p in root.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
   if p.is_symlink():raise ValueError('Ordinary committed source required')
   result[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result
a,b=inventory(old),inventory(new)
changed={p for p in set(a)|set(b) if a.get(p)!=b.get(p)}
if changed!=set(expected) or any(b.get(p)!=h for p,h in expected.items()):
 raise ValueError('Input packer changed an unproved numerical or runtime source')
print(json.dumps(dict(all_other_numerical_runtime_files_identical=True,
 changed_files={p:b[p] for p in sorted(changed)})))
'''


def reuse_committed_packer(config, original_source, native_sha256):
    """Activate an existing, byte-authenticated packer without redeployment.

    The numerical deployment already contains this exact runner and joined
    serializer. Switching its explicit runtime option changes no source file;
    the benchmark still verifies every complete input/output against its oracle.
    """
    native = ROOT/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'
    packer = ROOT/'tools/qwen3/joined_input_packet.py'
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Committed packer source required')
    if sha(native) != native_sha256:
        raise ValueError('The existing native runner must match the committed packer interface')
    original_source = Path(original_source)
    board = config['board']
    ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    verified = {}
    for relative, expected in (
            ('qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py', native_sha256),
            ('tools/qwen3/joined_input_packet.py', sha(packer))):
        actual = subprocess.check_output(ssh+[shlex.join(['sha256sum',str(original_source/relative)])],text=True).split()[0]
        if actual != expected:
            raise ValueError('The unchanged numerical deployment has a different packer source: '+relative)
        verified[relative] = expected
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    return dict(status='exact_committed_packer_reused_from_existing_numerical_deployment_NOT_model_acceptance',
        source_root=str(ROOT), source_revision=revision, original_source=str(original_source),
        deployment=str(original_source), verified_unchanged_sources=verified,
        packer_source=entry(packer), native_source=entry(native),
        deployed_source_files_changed=0, DSP_calls=0, new_primary_model_codegen=0,
        model_inference_acceptance=False)


def deploy(config, base_revision, original_source, destination):
    destination = Path(destination)
    if (destination.parent != Path(original_source).parent
            or destination.is_relative_to(Path('/dev/shm'))
            or not destination.name.endswith('-input-packer-sources')
            or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()):
        raise ValueError('Fresh scoped deployment and committed host packer required')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',base_revision,revision],check=True)
    changed=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',base_revision,revision],text=True).splitlines()
    if not changed or not set(changed)<=ALLOWED:
        raise ValueError('Only the explicit host packing and verification files may change')
    expected={path:sha(ROOT/path) for path in changed}
    board=config['board'];ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    subprocess.run(ssh+[shlex.join(['mkdir',str(destination)])],check=True)
    archive=subprocess.Popen(['git','-C',str(ROOT),'archive',revision],stdout=subprocess.PIPE)
    with archive.stdout:
        extracted=subprocess.run(ssh+[shlex.join(['tar','-C',str(destination),'-xf','-'])],stdin=archive.stdout)
    if archive.wait() or extracted.returncode:
        raise ValueError('Committed host packer deployment failed')
    proof=json.loads(subprocess.check_output(ssh+[shlex.join([board['python'],'-c',COMPARE,
        str(original_source),str(destination),json.dumps(expected)])],text=True))
    return dict(status='committed_host_input_packer_deployed_all_other_sources_identical_NOT_model_acceptance',
        source_root=str(ROOT),source_revision=revision,base_revision=base_revision,
        original_source=str(original_source),deployment=str(destination),comparison=proof,
        packer_source=entry(ROOT/'tools/qwen3/joined_input_packet.py'),
        native_source=entry(ROOT/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'),
        DSP_calls=0,new_primary_model_codegen=0,model_inference_acceptance=False)
