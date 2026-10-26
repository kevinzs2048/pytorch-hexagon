"""Bound readonly constant PT_LOAD reads while preserving code and weight sections.

Diagnostic only until actual board loading and full native numerics pass. A
program-header table stays in the firmware-required first page. Its expansion
displaces only .dynsym into verified unused readonly padding; the corresponding
DT_SYMTAB and section-header pointers follow it. Symbol values, code, weights,
entry point and all data addresses stay unchanged. The byte patch is reversible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct


EHDR = struct.Struct('<16sHHIIIIIHHHHHH')
PHDR = struct.Struct('<8I')
SHDR = struct.Struct('<10I')
FIRMWARE_MAX_PROGRAM_HEADERS = 10
MAX_READ_BYTES = 256*1024*1024


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def headers(path):
    with Path(path).open('rb') as f:
        raw=f.read(EHDR.size)
        if len(raw)!=EHDR.size:
            raise ValueError('Complete ELF32 header required')
        e=list(EHDR.unpack(raw))
        if (e[0][:7]!=b'\x7fELF\x01\x01\x01' or e[1:4]!=[3,164,1]
                or e[8]!=EHDR.size or e[9]!=PHDR.size or not 1<=e[10]<=64
                or e[11]!=SHDR.size or not 1<=e[12]<65535):
            raise ValueError('Ordinary little-endian Hexagon ELF32 DSO required')
        f.seek(e[5]);raw=f.read(e[10]*PHDR.size)
        if len(raw)!=e[10]*PHDR.size:
            raise ValueError('Complete program headers required')
        ph=[list(PHDR.unpack_from(raw,i*PHDR.size)) for i in range(e[10])]
        f.seek(e[6]);raw=f.read(e[12]*SHDR.size)
        if len(raw)!=e[12]*SHDR.size:
            raise ValueError('Complete section headers required')
        sh=[list(SHDR.unpack_from(raw,i*SHDR.size)) for i in range(e[12])]
    return e,ph,sh


def patches(path, maximum):
    e,ph,sh=headers(path)
    readonly=[(i,p) for i,p in enumerate(ph) if p[0]==1 and p[6]==4]
    descriptor=[(i,p) for i,p in enumerate(ph) if p[0]==6]
    if (len(readonly)!=1 or len(descriptor)!=1 or maximum<65536
            or maximum>MAX_READ_BYTES or maximum%65536):
        raise ValueError('One readonly LOAD and PHDR, bounded 64KiB-aligned chunks required')
    index,p=readonly[0];pi,old_table=descriptor[0]
    if (p[1:4]!=[0,0,0] or p[4]!=p[5]
            or p[7]!=65536 or old_table[1]!=e[5]
            or old_table[2]!=e[5] or old_table[4]!=e[10]*PHDR.size):
        raise ValueError('Original flat readonly image and covered PHDR required')
    if (pi!=0 or index!=1 or e[5]!=EHDR.size
            or len(ph)>FIRMWARE_MAX_PROGRAM_HEADERS):
        raise ValueError('First-page PHDR followed by readonly LOAD and at most ten headers required')
    if p[4]<=maximum:
        return e,ph,sh,[]
    chunks=[]
    for offset in range(0,p[4],maximum):
        size=min(maximum,p[4]-offset)
        chunks.append([1,offset,offset,offset,size,size,4,p[7]])
    new_ph=[q[:] for q in ph[:index]]+chunks+[q[:] for q in ph[index+1:]]
    if len(new_ph)>FIRMWARE_MAX_PROGRAM_HEADERS:
        raise ValueError('Firmware permits at most ten program headers; increase the read bound')
    table_size=len(new_ph)*PHDR.size
    table_offset=e[5];table_end=table_offset+table_size
    symbols=[(i,q) for i,q in enumerate(sh) if q[1]==11]
    dynamics=[q for q in ph if q[0]==2]
    if len(symbols)!=1 or len(dynamics)!=1:
        raise ValueError('One original dynamic symbol table and dynamic segment required')
    si,sym=symbols[0];dynamic=dynamics[0]
    symbol_offset=(p[4]+3)&~3;symbol_end=symbol_offset+sym[5]
    if (len(new_ph)>64 or table_end>4096 or symbol_end>path.stat().st_size
            or chunks[-1][4]+symbol_end-p[4]>maximum or sym[2]!=2
            or sym[3]!=sym[4] or sym[4]!=e[5]+e[10]*PHDR.size
            or not 0<sym[5]<=65536 or sym[9]!=16 or sym[5]%16
            or not sym[4]<table_end<=sym[4]+sym[5]):
        raise ValueError('First-page table expanding only into original .dynsym required')
    for q in ph:
        if q[0] not in (1,6):continue
        if q[0]==1 and q is not p and q[4] and q[1]<symbol_end and q[1]+q[4]>symbol_offset:
            raise ValueError('Relocated symbols overlap another LOAD')
    for q in sh:
        if q[1]==8 or not q[5]:continue
        if q[4]<symbol_end and q[4]+q[5]>symbol_offset:
            raise ValueError('Relocated symbols overlap an original ELF section')
        if q is not sym and q[4]<table_end and q[4]+q[5]>table_offset:
            raise ValueError('Expanded headers overlap another original ELF section')
    if e[6]<symbol_end and e[6]+e[12]*SHDR.size>symbol_offset:
        raise ValueError('Relocated symbols overlap original section headers')
    with path.open('rb') as f:
        f.seek(p[4]);padding=f.read(symbol_end-p[4])
        f.seek(sym[4]);symbol_bytes=f.read(sym[5])
        f.seek(dynamic[1]);dynamic_bytes=f.read(dynamic[4])
    if len(padding)!=symbol_end-p[4] or any(padding):
        raise ValueError('Verified zero padding required')
    if dynamic[4]%8 or len(dynamic_bytes)!=dynamic[4]:
        raise ValueError('Complete ELF32 dynamic entries required')
    symbol_tags=[i for i in range(0,len(dynamic_bytes),8)
        if struct.unpack_from('<II',dynamic_bytes,i)==(6,sym[3])]
    if len(symbol_tags)!=1:
        raise ValueError('Original DT_SYMTAB pointer must match .dynsym')
    # PHDR precedes every LOAD in this supported input; it stays first.
    if pi!=0 or index!=1:
        raise ValueError('Original PHDR followed by readonly LOAD required')
    new_ph[0]=[6,table_offset,table_offset,table_offset,table_size,table_size,4,4]
    last=1+len(chunks)-1
    new_ph[last][4]=new_ph[last][5]=symbol_end-new_ph[last][1]
    new_e=e[:];new_e[10]=len(new_ph)
    new_sh=[q[:] for q in sh];new_sh[si][3]=new_sh[si][4]=symbol_offset
    edits=[(44,struct.pack('<H',len(new_ph))),
           (table_offset,b''.join(PHDR.pack(*q) for q in new_ph)),
           (symbol_offset,symbol_bytes),
           (e[6]+si*SHDR.size+12,struct.pack('<II',symbol_offset,symbol_offset)),
           (dynamic[1]+symbol_tags[0]+4,struct.pack('<I',symbol_offset))]
    with path.open('rb') as f:
        rows=[]
        for offset,data in edits:
            f.seek(offset);old=f.read(len(data))
            rows.append(dict(offset=offset,original_hex=old.hex(),replacement_hex=data.hex()))
    return new_e,new_ph,new_sh,rows


def verify(source,output,rows,maximum):
    """Derive permitted edits independently, then compare every resulting byte."""
    expected_e,expected_ph,expected_sh,expected_rows=patches(source,maximum)
    if rows!=expected_rows:raise ValueError('Saved patch differs from independently derived metadata edits')
    if source.stat().st_size!=output.stat().st_size:
        raise ValueError('ELF file length changed')
    with source.open('rb') as a,output.open('rb') as b:
        offset=0
        while True:
            old=a.read(4*1024*1024);new=b.read(len(old))
            if not old:break
            wanted=bytearray(old)
            for row in rows:
                start=row['offset'];data=bytes.fromhex(row['replacement_hex'])
                lo=max(offset,start);hi=min(offset+len(old),start+len(data))
                if lo<hi:wanted[lo-offset:hi-offset]=data[lo-start:hi-start]
            if new!=wanted:raise ValueError('Unexpected payload or metadata byte changed')
            offset+=len(old)
    old_e,old_ph,old_sh=headers(source);new_e,new_ph,new_sh=headers(output)
    if new_sh!=expected_sh or new_e!=expected_e or new_ph!=expected_ph:
        raise ValueError('Unsupported section, program header or ELF ABI change')
    if any(q[4]>maximum for q in new_ph if q[0]==1 and q[6]==4):
        raise ValueError('Readonly LOAD exceeds bound')
    if [q for q in old_ph if q[0] not in (1,6) or q[6]!=4]!=[q for q in new_ph if q[0] not in (1,6) or q[6]!=4]:
        raise ValueError('Executable, writable or other original segment changed')
    return dict(all_other_file_bytes_identical=True,
        only_dynsym_section_location_and_DT_SYMTAB_pointer_changed=True,
        all_dynamic_symbol_values_and_table_bytes_identical=True,
        first_page_program_header_table=True,
        original_symbol_code_constant_addresses_and_bytes_unchanged=True,
        readonly_LOAD_filesz_max=max(q[4] for q in new_ph if q[0]==1 and q[6]==4),
        original_program_headers=len(old_ph),candidate_program_headers=len(new_ph))


def verified_proof(source,output,proof_path,expected_sha):
    proof=json.loads(proof_path.read_text())
    actual=sha(output)
    if (proof['status']!='constant_readonly_LOAD_split_byte_preservation_verified_diagnostic'
            or proof['original']['sha256']!=expected_sha or sha(source)!=expected_sha
            or proof['candidate']['sha256']!=actual
            or proof['checks']!=verify(source,output,proof['patch'],proof['max_load_bytes'])):
        raise ValueError('Actual original and preserved constant payload proof required')
    return dict(original_sha256=expected_sha,candidate_sha256=actual,
        proof_path=str(proof_path),proof_sha256=sha(proof_path),
        max_load_bytes=proof['max_load_bytes'],checks=proof['checks'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','proof'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--max-load-bytes',type=int,default=MAX_READ_BYTES)
    a=p.parse_args()
    if (not a.output.is_absolute() or a.output.exists() or a.output.is_symlink()
            or not a.proof.is_absolute() or a.proof.exists() or a.proof.is_symlink()
            or sha(a.input)!=a.expected_sha256):
        p.error('Pinned original constant file, fresh absolute output and proof required')
    _,_,_,rows=patches(a.input,a.max_load_bytes)
    shutil.copyfile(a.input,a.output)
    with a.output.open('r+b') as f:
        for row in rows:
            f.seek(row['offset']);f.write(bytes.fromhex(row['replacement_hex']))
    checks=verify(a.input,a.output,rows,a.max_load_bytes)
    if sha(a.input)!=a.expected_sha256:raise ValueError('Original constant bytes changed during copy')
    record=dict(status='constant_readonly_LOAD_split_byte_preservation_verified_diagnostic',
        source_sha256=sha(__file__),original=dict(path=str(a.input),sha256=a.expected_sha256),
        candidate=dict(path=str(a.output),sha256=sha(a.output),bytes=a.output.stat().st_size),
        patch=rows,checks=checks,max_load_bytes=a.max_load_bytes,
        firmware_program_header_limit=FIRMWARE_MAX_PROGRAM_HEADERS,
        DSP_calls=0,numerical_acceptance=False,
        loader_failure_fixed=False)
    a.proof.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':main()
