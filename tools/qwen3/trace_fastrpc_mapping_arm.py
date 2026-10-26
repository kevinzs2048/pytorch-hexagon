"""Capture FastRPC mapping arguments/errors in a private ARM trace instance.

Diagnostic intervals only; stop before formal native performance measurements.
No fault injection, driver replacement, global trace reset or model byte reads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time


TRACE=Path('/sys/kernel/tracing')
GROUP='qwen3_loader_20261008'
INSTANCE=TRACE/'instances'/GROUP
MODULE_SHA='2662f5becdd8f96ea17a926fdf76d568122b01a6ee590e785f7870fe8cc55b8a'
PROBES={
    'map_entry':'p:'+GROUP+'/map_entry fastrpc:fastrpc_map_create fd=$arg2:s32 length=$arg4:u64 attr=$arg5:x32',
    'map_exit':'r64:'+GROUP+'/map_exit fastrpc:fastrpc_map_create result=$retval:s32',
    'attach_error':'r64:'+GROUP+'/attach_error dma_buf_attach result=$retval:s64',
    'dma_map_error':'r64:'+GROUP+'/dma_map_error dma_buf_map_attachment_unlocked result=$retval:s64',
}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def append_probe(line):
    # Tracefs is nonseekable; Python's append-mode file wrapper seeks to EOF.
    # Raw O_APPEND writes preserve every unrelated existing probe definition.
    fd=os.open(TRACE/'kprobe_events',os.O_WRONLY|os.O_APPEND)
    try:
        data=(line+'\n').encode()
        if os.write(fd,data)!=len(data):raise OSError('Incomplete tracefs probe definition')
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','stop'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if (os.geteuid()!=0 or platform.machine()!='aarch64'
            or platform.release()!='6.8.12-4-qc2509'
            or not a.output.is_absolute() or a.output.is_symlink()
            or not a.output.parent.is_dir()):
        p.error('Pinned actual ARM kernel, root and absolute diagnostic output with an existing parent required')
    module=Path('/lib/modules',platform.release(),'kernel/drivers/misc/fastrpc.ko.xz')
    if sha(module)!=MODULE_SHA:raise ValueError('Actual FastRPC module bytes changed')
    def save(name,value):(a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    if a.action=='start':
        before=(TRACE/'kprobe_events').read_text()
        if a.output.exists() or INSTANCE.exists() or GROUP in before:
            raise ValueError('Fresh private trace instance and output required')
        a.output.mkdir();INSTANCE.mkdir();added=[]
        try:
            (INSTANCE/'tracing_on').write_text('0')
            (INSTANCE/'buffer_size_kb').write_text('4096')
            (INSTANCE/'trace_clock').write_text('mono')
            for name,line in PROBES.items():
                append_probe(line);added.append(name)
                event=INSTANCE/'events'/GROUP/name
                if name.endswith('_error'):
                    (event/'filter').write_text('result >= -4095 && result < 0')
                (event/'enable').write_text('1')
            record=dict(status='private_FastRPC_mapping_trace_active',source_sha256=sha(Path(__file__)),
                instance=str(INSTANCE),group=GROUP,probes=PROBES,
                kernel=platform.release(),module_sha256=MODULE_SHA,
                preexisting_kprobe_events=before,global_tracing_on=(TRACE/'tracing_on').read_text(),
                global_current_tracer=(TRACE/'current_tracer').read_text(),start_unix=time.time(),
                DSP_model_calls=0,performance_acceptance=False,
                use_scope='numerical_loader_diagnostic_only_stop_before_formal_ABBA')
            save('start.json',record);(INSTANCE/'tracing_on').write_text('1')
        except BaseException:
            (INSTANCE/'tracing_on').write_text('0')
            for name in added:
                (INSTANCE/'events'/GROUP/name/'enable').write_text('0')
            INSTANCE.rmdir()
            for name in added:append_probe('-:'+GROUP+'/'+name)
            raise
    else:
        record=json.loads((a.output/'start.json').read_text())
        if (record['status']!='private_FastRPC_mapping_trace_active'
                or record['source_sha256']!=sha(Path(__file__)) or record['probes']!=PROBES
                or record['instance']!=str(INSTANCE) or not INSTANCE.is_dir()
                or (a.output/'stop.json').exists()):
            raise ValueError('Exact owned active trace receipt required')
        (INSTANCE/'tracing_on').write_text('0')
        for name in PROBES:(INSTANCE/'events'/GROUP/name/'enable').write_text('0')
        (a.output/'trace.txt').write_text((INSTANCE/'trace').read_text())
        # This kernel reports unqualified event names in kprobe_profile.
        profile='\n'.join(x for x in (TRACE/'kprobe_profile').read_text().splitlines()
            if x.split() and x.split()[0] in PROBES)
        (a.output/'probe-profile.txt').write_text(profile+'\n')
        INSTANCE.rmdir()
        for name in PROBES:append_probe('-:'+GROUP+'/'+name)
        after=(TRACE/'kprobe_events').read_text()
        if after!=record['preexisting_kprobe_events']:
            raise ValueError('Preexisting probe definitions changed; own events removed')
        save('stop.json',dict(status='private_FastRPC_mapping_trace_stopped_and_own_probes_removed',
            trace_sha256=sha(a.output/'trace.txt'),probe_profile=profile,completed_unix=time.time(),
            original_probe_definitions_preserved=True,instance_removed=not INSTANCE.exists(),
            global_tracing_on_unchanged=(TRACE/'tracing_on').read_text()==record['global_tracing_on'],
            global_current_tracer_unchanged=(TRACE/'current_tracer').read_text()==record['global_current_tracer'],
            performance_acceptance=False,DSP_model_calls=0))
    print(a.action+' private FastRPC mapping trace')


if __name__=='__main__':main()
