"""Experimental cache-hit host worker retaining one Python interpreter.

The original worker owns mailbox admission, replies and session closure. Every
request still calls the original model loader and full parameter validation.
Cache misses are rejected before lowering/code generation. This is not the
general production compiler worker or a numerical/performance acceptance.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from build import ROOT, sha
from profile_native_compile_host import FORBIDDEN


def reject_build(build):
    raise ValueError('Experimental cache-hit worker forbids frontend and machine cache misses')


def cached_only(original):
    def invoke(cache, request, compiler):
        # Keep original key construction, locking and complete artifact hashing.
        return original(cache, request, reject_build)
    return invoke


def cached_frontend_only(original):
    def read(cache, *args):
        result=original(cache,*args)
        if result is None:
            raise ValueError('Experimental worker requires the authenticated original frontend cache entry')
        return result
    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numerical-source', type=Path, required=True)
    parser.add_argument('--numerical-revision', required=True)
    parser.add_argument('--deadline-unix', type=int, required=True)
    args, remaining = parser.parse_known_args()
    git = lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    if (git(ROOT,'status','--porcelain') or git(args.numerical_source,'status','--porcelain')
            or git(args.numerical_source,'rev-parse','HEAD') != args.numerical_revision
            or args.deadline_unix-time.time()<300):
        raise ValueError('Frozen clean worker and numerical source within actual deadline required')
    start=time.perf_counter();folder=args.numerical_source/'qwen3/runtime/arm'
    sys.path[:0]=[str(folder),str(args.numerical_source)]
    import torch
    from transformers import Qwen3ForCausalLM, AutoTokenizer
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    import compile_qwen_symbolic_carrier as compiler
    import serve_qwen_compile_mailbox as worker
    from hexagon_compiled_artifact_cache import CompiledArtifactCache
    from hexagon_carrier_parameter_binding import CarrierParameterBinding
    from hexagon_captured_linalg_cache import CapturedLinalgCache
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    original_compile=worker.run_compiler;original_cache=CompiledArtifactCache.get_or_compile
    original_frontend=CapturedLinalgCache._read
    original_popen=subprocess.Popen;original_argv=sys.argv
    def no_compiler(command,*positional,**kwargs):
        if isinstance(command,(str,bytes)) or any(Path(str(word)).name in FORBIDDEN for word in command):
            raise ValueError('Cache-hit session forbids external compiler or shell work')
        return original_popen(command,*positional,**kwargs)
    def inline(command,log,timeout):
        if (len(command)<3 or command[0]!=sys.executable
                or Path(command[1]).resolve()!=Path(compiler.__file__).resolve()):
            raise ValueError('Exact original owned compiler transaction required')
        from contextlib import redirect_stdout, redirect_stderr
        previous_argv=sys.argv;previous_handler=signal.getsignal(signal.SIGALRM)
        previous_timer=signal.getitimer(signal.ITIMER_REAL)
        def expired(*unused):raise TimeoutError('Owned cache-hit transaction deadline reached')
        try:
            signal.signal(signal.SIGALRM,expired)
            signal.setitimer(signal.ITIMER_REAL,max(1,min(timeout,args.deadline_unix-time.time()-30)))
            sys.argv=command[1:]
            with redirect_stdout(log),redirect_stderr(log):compiler.main()
            output=Path(command[command.index('--output')+1])
            result=json.loads((output/'result.json').read_text())
            if (result['frontend_invocations']!=0 or result['machine_code_compilations']!=0
                    or result['frontend_cache']['cache_hit'] is not True
                    or result['machine_cache']['cache_hit'] is not True
                    or result['DSP_calls']!=0 or result['CPU_model_forward'] is not False):
                raise ValueError('Experimental worker must preserve original zero-codegen admission')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)
            if previous_timer[0]:signal.setitimer(signal.ITIMER_REAL,*previous_timer)
            sys.argv=previous_argv
    print(json.dumps(dict(status='experimental_cache_hit_interpreter_dependencies_ready',
        dependency_preload_seconds=time.perf_counter()-start,
        original_worker_sha256=sha(worker.__file__),original_compiler_sha256=sha(compiler.__file__),
        original_model_loader_called_each_request=True,all_original_parameter_and_artifact_checks_preserved=True)),flush=True)
    try:
        worker.run_compiler=inline;CompiledArtifactCache.get_or_compile=cached_only(original_cache)
        CapturedLinalgCache._read=cached_frontend_only(original_frontend)
        subprocess.Popen=no_compiler;sys.argv=[str(worker.__file__),*remaining]
        worker.main()
    finally:
        worker.run_compiler=original_compile;CompiledArtifactCache.get_or_compile=original_cache
        CapturedLinalgCache._read=original_frontend
        subprocess.Popen=original_popen;sys.argv=original_argv


if __name__=='__main__':main()
