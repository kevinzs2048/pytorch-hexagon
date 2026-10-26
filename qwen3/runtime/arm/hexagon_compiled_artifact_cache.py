"""Backend machine-code cache; does not replace Dynamo or model/KV semantics.

The caller computes identities from the actual captured graph, frozen weights,
target, compiler policy and ABI. A cache hit only authenticates compiler output;
it is neither numerical acceptance nor a CPU fallback. New failed build folders
are retained. This module does not execute binaries or launch a compiler itself.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def cache_key(request):
    fields={'schema','graph_sha256','parameters_sha256','compiler_sha256','linker_sha256',
            'lowering_policy_sha256','runtime_abi_sha256','framework_sha256','target','options'}
    if (type(request) is not dict or set(request)!=fields or type(request['schema']) is not int
            or request['schema']!=1 or type(request['target']) is not str or not request['target']
            or type(request['options']) is not dict):
        raise ValueError('Complete explicit graph/weights/toolchain/ABI cache identity required')
    for key in fields:
        if key.endswith('_sha256'):
            value=request[key]
            if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Exact SHA256 identities required')
    return hashlib.sha256(canonical(request)).hexdigest()


def artifact_member(root,relative):
    if (type(relative) is not str or not relative or '\\' in relative
            or any(c in relative for c in '\x00\r\n')):raise ValueError('Explicit artifact member required')
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts or str(path)!=relative or relative=='.':
        raise ValueError('Artifact must stay inside its new build directory')
    target=Path(root)/path
    if target.is_symlink() or not target.is_file() or Path(root).resolve() not in target.resolve().parents:
        raise ValueError('Regular build artifact required; symlink escape forbidden')
    return target


def describe_artifacts(root,relative_paths):
    if type(relative_paths) is not list or not relative_paths or len(set(relative_paths))!=len(relative_paths):
        raise ValueError('Unique explicit compiler output members required')
    rows=[]
    for relative in relative_paths:
        path=artifact_member(root,relative)
        rows.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest_file(path)))
    return rows


class CompiledArtifactCache:
    def __init__(self,root):
        supplied=Path(root)
        if not supplied.is_absolute() or supplied.is_symlink() or not supplied.is_dir():
            raise ValueError('Existing explicit cache directory required')
        self.root=supplied.resolve()

    def _read(self,key,request):
        entry=self.root/(key+'.json')
        if not entry.exists():
            if entry.is_symlink():raise ValueError('Dangling cache entry is not a miss')
            return None
        if entry.is_symlink() or not entry.is_file():raise ValueError('Regular cache entry required')
        value=json.loads(entry.read_text())
        if (value.get('status')!='compiler_artifacts_cached_NOT_numerical_acceptance'
                or value.get('key')!=key or value.get('request')!=request
                or type(value.get('build')) is not str or Path(value['build']).name!=value['build']
                or not value['build'].startswith(key+'-')):
            raise ValueError('Cache identity or transaction binding changed')
        build=self.root/value['build']
        if build.is_symlink() or not build.is_dir():raise ValueError('Original build directory required')
        rows=value.get('artifacts')
        if (type(rows) is not list or not rows
                or rows!=describe_artifacts(build,[r['path'] for r in rows])):
            raise ValueError('Cached compiler bytes drifted; do not silently rebuild or execute')
        return dict(value,directory=str(build))

    def get_or_compile(self,request,compiler):
        request=json.loads(canonical(request));key=cache_key(request)
        # Per-key serialization: a second process checks again after the first
        # finishes. We do not conflate this with PyTorch's graph/guard cache.
        flags=os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW
        fd=os.open(self.root/(key+'.lock'),flags,0o600)
        with os.fdopen(fd,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            hit=self._read(key,request)
            if hit is not None:return dict(hit,cache_hit=True,compiler_invocations=0)
            build=Path(tempfile.mkdtemp(prefix=key+'-',dir=self.root))
            try:
                relative_paths=compiler(build)
                rows=describe_artifacts(build,relative_paths)
                value=dict(status='compiler_artifacts_cached_NOT_numerical_acceptance',key=key,
                    request=request,build=build.name,artifacts=rows)
                pending=build/'cache-entry.pending.json'
                with pending.open('xb') as stream:
                    stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
                # Publish only after complete output hashing, never a partial
                # compiler result. Hardlink is atomic and refuses replacement.
                os.link(pending,self.root/(key+'.json'))
                result=self._read(key,request)
                return dict(result,cache_hit=False,compiler_invocations=1)
            except BaseException as error:
                failure=build/'cache-build-failure.json'
                if not failure.exists():
                    with failure.open('x') as stream:
                        json.dump(dict(key=key,error=repr(error),published=(self.root/(key+'.json')).exists()),stream)
                raise
