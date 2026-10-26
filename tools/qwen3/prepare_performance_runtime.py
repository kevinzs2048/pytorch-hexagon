"""Stage a separately built transport while preserving the numerical catalog."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def validate_transport_override(original, candidate):
    mutable = {'runtime', 'control_deployment'}
    if ({k:v for k,v in original.items() if k not in mutable} !=
            {k:v for k,v in candidate.items() if k not in mutable}):
        raise ValueError('Transport override changed model entries or numerical catalog metadata')
    control = Path(candidate['control_deployment'])
    runtime = candidate.get('runtime', {})
    artifacts = runtime.get('artifacts', [])
    if (control.parent != Path('/dev/shm') or not control.name.startswith('qwen3-arm-')
            or any(not (x.isalnum() or x in '-_') for x in control.name)
            or runtime.get('shared_memory_packets') is not True
            or [x.get('name') for x in artifacts] != ['buffer_stream', 'libhexagon_buffer_runner_skel.so']):
        raise ValueError('Scoped explicit shared-packet runtime required')
    for digest in [runtime.get('sha256'), *(x.get('sha256') for x in artifacts)]:
        if type(digest) is not str or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Pinned runtime receipt and artifact SHA256 required')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transport_catalog(base, runtime, receipt, receipt_sha, control):
    artifacts = runtime.get('artifacts', [])
    if (base.get('status') != 'warm_ARM_dynamic_catalog_NOT_executed'
            or runtime.get('status') != 'profiled_buffer_transport_built_not_executed'
            or runtime.get('model_math_changed') is not False
            or runtime.get('input_checks_removed') is not False
            or runtime.get('shared_memory_packets') is not True
            or [Path(x['path']).name for x in artifacts] != ['buffer_stream', 'libhexagon_buffer_runner_skel.so']
            or Path(control).parent != Path('/dev/shm') or not Path(control).name.startswith('qwen3-arm-')):
        raise ValueError('Explicit profiled transport and unchanged numerical catalog required')
    catalog = copy.deepcopy(base)
    catalog['control_deployment'] = str(control)
    catalog['runtime'] = dict(path=str(receipt), sha256=receipt_sha, shared_memory_packets=True,
        artifacts=[dict(name=Path(row['path']).name, sha256=row['sha256']) for row in artifacts])
    validate_transport_override(base, catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'runtime', 'output', 'board-output', 'control-deployment'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.exists()
            or args.board_output.parent != Path('/mnt/nvme/qwen3-work/qwen3-performance')
            or not args.board_output.name.startswith('runtime-')):
        parser.error('Fresh host output and scoped persistent board runtime directory required')
    config = json.loads(args.config.read_text())
    base = json.loads(Path(config['catalog']).read_text())
    runtime = json.loads(args.runtime.read_text())
    catalog = transport_catalog(base, runtime, args.runtime, sha(args.runtime), args.control_deployment)
    for row in runtime['artifacts']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Built runtime artifact changed')
    args.output.mkdir(parents=True)
    catalog_path = args.output/'catalog.json'
    catalog_path.write_text(json.dumps(catalog, indent=2)+'\n')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', config['board']['ssh_target']]
    # Create only a fresh persistent directory. The small /dev/shm alias is
    # recorded explicitly and can be recreated after reboot.
    script = """import json,sys
from pathlib import Path
a=json.loads(sys.argv[1]);p=Path(a['output']);alias=Path(a['alias'])
if p.exists() or alias.exists() or alias.is_symlink():raise ValueError('Fresh runtime paths required')
p.mkdir(parents=True);(p/'rfsa').mkdir()
alias.symlink_to(p,target_is_directory=True)
old=Path(a['constants_source'])
for i in range(1,4):
 name='lib_mlir_ciface_Qwen3DynamicBoundProbe-consts-'+str(i)+'.so'
 source=old/name
 if not source.is_file():raise ValueError('Accepted model constant missing')
 (p/name).symlink_to(source)
"""
    payload = json.dumps(dict(output=str(args.board_output), alias=str(args.control_deployment),
        constants_source=base['control_deployment']))
    subprocess.run(ssh+[shlex.join(['python', '-c', script, payload])], check=True)
    board_files = []
    for row in runtime['artifacts']:
        name = Path(row['path']).name
        destination = args.board_output/(Path('rfsa')/name if name.endswith('.so') else Path(name))
        subprocess.run(['scp', '-q', row['path'], config['board']['ssh_target']+':'+str(destination)], check=True)
        remote_sha = subprocess.check_output(ssh+[shlex.join(['sha256sum', str(destination)])], text=True).split()[0]
        if remote_sha != row['sha256']:
            raise ValueError('Staged board runtime changed')
        board_files.append(dict(path=str(destination), sha256=remote_sha))
    board_catalog = args.board_output/'catalog.json'
    subprocess.run(['scp', '-q', str(catalog_path), config['board']['ssh_target']+':'+str(board_catalog)], check=True)
    if subprocess.check_output(ssh+[shlex.join(['sha256sum', str(board_catalog)])], text=True).split()[0] != sha(catalog_path):
        raise ValueError('Board catalog changed in staging')
    result = dict(status='profiled_transport_prepared_NOT_numerically_accepted',
        model_entries_unchanged=catalog['entries'] == base['entries'],
        base_catalog_sha256=sha(config['catalog']), runtime_receipt_sha256=sha(args.runtime),
        catalog=dict(path=str(catalog_path), board_path=str(board_catalog), sha256=sha(catalog_path)),
        control_alias=str(args.control_deployment), persistent_board_directory=str(args.board_output),
        board_artifacts=board_files, numerical_acceptance=False, performance_acceptance=False)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'])


if __name__ == '__main__':
    main()
