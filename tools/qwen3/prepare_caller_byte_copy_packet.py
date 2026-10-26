"""Independent CPU byte reference for bounded alignment/length classes."""
import argparse
import struct
from pathlib import Path

from build import entry, write


def prepare(output):
    if output.exists() or not output.is_absolute():raise ValueError('Fresh absolute packet directory required')
    cases=[(a,0,1025) for a in range(128)]+[(0,a,1025) for a in range(128)]+[(a,a,1025) for a in range(128)]
    cases += [(a,a,n) for n in (0,1,127,128,129,511,512,1023,1024,1151,1152,1153,4095,4096,4097,65536) for a in (0,1,7,16,63,64,127)]
    cases += [(16,0,4194304),(0,0,4194304)]
    data=bytearray();expected=bytearray();rows=[];header=16+12*len(cases);base=(header+127)&~127
    pattern=bytes((i*97+13)&255 for i in range(256))
    for sa,da,n in cases:
        si=(len(data)+127)&~127;di=(len(expected)+127)&~127
        data.extend(bytes(si+128+sa+n+128-len(data)));expected.extend(b'\xa7'*(di+128+da+n+128-len(expected)))
        full=(pattern*((128+sa+n+128+255)//256))[:128+sa+n+128];data[si:si+len(full)]=full
        start=base+si+128+sa;dest=di+128+da;rows.append((start,dest,n))
        expected[dest:dest+n]=data[si+128+sa:si+128+sa+n]
    payload=b'QWCPY1\0\0'+struct.pack('<II',len(cases),0)+b''.join(struct.pack('<III',*r) for r in rows)+bytes(base-header)+data
    output.mkdir();packet=output/'input.bin';reference=output/'expected-output.bin'
    packet.write_bytes(payload);reference.write_bytes(expected)
    result=dict(status='independent_CPU_byte_reference_prepared_NOT_DSP',case_count=len(cases),
        packet=entry(packet),expected=entry(reference),source=entry(__file__),
        cases=[dict(source_alignment=sa,destination_alignment=da,bytes=n) for sa,da,n in cases],
        model_calls=0,DSP_calls=0,model_inference_acceptance=False)
    write(output/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
