"""Install native AOT reuse around the unchanged original packet observer."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-source','observer-source','namespace-manifest','cache-root','cache-report','native-output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--observer-sha256',required=True)
    a,remaining=p.parse_known_args()
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if (platform.machine()!='aarch64' or a.cache_report.exists()
            or not all(path.is_absolute() for path in (a.native_source,a.observer_source,a.cache_report,a.native_output))
            or sha(a.observer_source)!=a.observer_sha256):
        raise ValueError('Actual ARM, original observer bytes and fresh absolute report required')
    manifest=json.loads(a.namespace_manifest.read_text())
    if sha(a.native_source)!=manifest['native_source_sha256']:
        raise ValueError('Original observed native entry differs from namespace')
    sys.path[:0]=[str(a.observer_source.parent),str(a.native_source.parent),str(a.native_source.parents[3])]
    import torch
    import torch._dynamo.backends.common as common
    import torch._functorch.config as config
    from torch._dynamo.utils import counters
    from native_aot_output import native_cache_scope,namespace_digest
    original=common.aot_autograd;old_argv=sys.argv;events=[];before=dict(counters['aot_autograd'])
    def settings():
        return dict(bundled_autograd_cache=config.bundled_autograd_cache,
            enable_autograd_cache=config.enable_autograd_cache,bypass_autograd_cache_key=config.bypass_autograd_cache_key,
            unsafe_skip_cache_dynamic_shape_guards=torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards)
    original_config=settings()
    record=dict(status='running',aot_mode='cache',namespace=namespace_digest(manifest),events=events,
        native_source_sha256=sha(a.native_source),observer_source_sha256=sha(a.observer_source),
        wrapper_sha256=sha(__file__),adapter_sha256=manifest['adapter_sha256'],
        manifest_sha256=sha(a.namespace_manifest),cache_root=str(a.cache_root),
        original_configuration=original_config,installed_sources_modified=False,
        production_configuration_changed=False,model_inference_acceptance=False,performance_acceptance=False,
        original_online_CPU_reference_and_actual_DSP_observer_retained=True,start_unix=time.time())
    def save():
        temporary=a.cache_report.with_suffix('.pending.json')
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(a.cache_report)
    save()
    try:
        sys.argv=[str(a.observer_source),*remaining]
        with native_cache_scope(manifest,a.cache_root,events):
            runpy.run_path(str(a.observer_source),run_name='__main__')
        native=json.loads((a.native_output/'result.json').read_text())
        if (len(events)!=len(native['captures']) or not 1<=len(events)<=2
                or [e['graph'] for e in events]!=[c['graph'] for c in native['captures']]):
            raise ValueError('Every actually observed original capture must use the cache protocol')
        record['status']='original_native_observer_completed_with_recorded_AOT_cache_protocol_DIAGNOSTIC'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        sys.argv=old_argv
        record.update(completed_unix=time.time(),factory_restored=common.aot_autograd is original,
            original_configuration_restored=original_config==settings(),
            framework_cache_counters_delta={k:v-before.get(k,0) for k,v in counters['aot_autograd'].items()})
        save()


if __name__=='__main__':main()
