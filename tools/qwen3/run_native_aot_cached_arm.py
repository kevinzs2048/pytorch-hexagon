"""Opt in to AOT artifact reuse around the immutable native generation entry.

The original entry continues to own model loading, full parameter checks,
compiler handoff, loader, DSP, generation and stopping behavior. This wrapper
only installs the native OutputCode protocol in this diagnostic process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-source','namespace-manifest','cache-root','cache-report'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--aot-mode',choices=('original','cache'),required=True)
    args,remaining=parser.parse_known_args()
    if (platform.machine()!='aarch64' or not args.native_source.is_absolute()
            or not args.native_source.is_file() or not args.cache_report.is_absolute()
            or args.cache_report.exists()):
        parser.error('Actual ARM, immutable native source and fresh report required')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads(args.namespace_manifest.read_text())
    if sha(args.native_source)!=manifest['native_source_sha256']:
        raise ValueError('Exact original native source required')
    sys.path[:0]=[str(args.native_source.parent),str(args.native_source.parents[3])]
    import torch
    import torch._dynamo.backends.common as common
    import torch._functorch.config as config
    from torch._dynamo.utils import counters
    from native_aot_output import native_cache_scope,namespace_digest
    namespace=namespace_digest(manifest);events=[]
    before=dict(counters['aot_autograd']);original=common.aot_autograd;old_argv=sys.argv
    original_config=dict(bundled_autograd_cache=config.bundled_autograd_cache,
        enable_autograd_cache=config.enable_autograd_cache,
        bypass_autograd_cache_key=config.bypass_autograd_cache_key,
        unsafe_skip_cache_dynamic_shape_guards=torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards)
    record=dict(status='running',aot_mode=args.aot_mode,namespace=namespace,events=events,
        native_source_sha256=sha(args.native_source),wrapper_sha256=sha(__file__),
        adapter_sha256=manifest['adapter_sha256'],manifest_sha256=sha(args.namespace_manifest),
        cache_root=str(args.cache_root),original_configuration=original_config,
        installed_sources_modified=False,production_configuration_changed=False,
        model_inference_acceptance=False,performance_acceptance=False,
        original_native_model_loader_and_all_parameter_guards_retained=True,
        original_native_math_and_DSP_entry_retained=True,start_unix=time.time())
    def save():
        temporary=args.cache_report.with_suffix('.pending.json')
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(args.cache_report)
    save()
    try:
        sys.argv=[str(args.native_source),*remaining]
        if args.aot_mode=='cache':
            with native_cache_scope(manifest,args.cache_root,events):
                runpy.run_path(str(args.native_source),run_name='__main__')
            if len(events)!=2:
                raise ValueError('Exactly the original two native captures must use the adapter')
        else:
            runpy.run_path(str(args.native_source),run_name='__main__')
            if events:raise ValueError('Original arm must not use cached native callbacks')
        record['status']='original_native_entry_completed_with_recorded_AOT_cache_protocol_DIAGNOSTIC'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        sys.argv=old_argv
        record.update(completed_unix=time.time(),factory_restored=common.aot_autograd is original,
            framework_cache_counters_delta={key:value-before.get(key,0) for key,value in counters['aot_autograd'].items()},
            original_configuration_restored=original_config==dict(bundled_autograd_cache=config.bundled_autograd_cache,
                enable_autograd_cache=config.enable_autograd_cache,bypass_autograd_cache_key=config.bypass_autograd_cache_key,
                unsafe_skip_cache_dynamic_shape_guards=torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards))
        save()


if __name__=='__main__':main()
