import struct
from pathlib import Path
import pytest

from split_constant_load_segments import EHDR,PHDR,SHDR,headers,patches,verify


def fixture(path):
    ident=b'\x7fELF\x01\x01\x01'+b'\0'*9
    ph=[[6,52,52,52,8*32,8*32,4,4],
        [1,0,0,0,0x3507c,0x3507c,4,65536],
        [1,0x40000,0x40000,0x40000,0x100,0x100,5,65536],
        [1,0x50000,0x50000,0x50000,0x100,0x1000,6,65536],
        [1,0x60000,0x60000,0x60000,0x100,0x100,6,65536],
        [2,0x50020,0x50020,0x50020,0x80,0x80,6,4],
        [0x6474e552,0x50000,0x50000,0x50000,0x100,0x1000,4,1],
        [0x6474e551,0,0,0,0,0,6,0]]
    sh=[[0]*10,[0,11,2,0x134,0x134,0x1a0,0,0,4,16],
        [0,1,2,0x1000,0x1000,0x3407c,0,0,16,0],
        [0,1,6,0x40000,0x40000,0x100,0,0,16,0],
        [0,6,3,0x50020,0x50020,0x80,0,0,4,8]]
    e=[ident,3,164,1,0,52,0x61000,0,52,32,len(ph),40,len(sh),0]
    data=bytearray(0x62000);data[:52]=EHDR.pack(*e)
    for i,q in enumerate(ph):data[52+i*32:52+(i+1)*32]=PHDR.pack(*q)
    data[0x1000:0x3507c]=bytes((i*17+11)&255 for i in range(0x3407c))
    data[0x40000:0x40100]=bytes(range(256));data[0x50000:0x50100]=b'W'*256
    data[0x134:0x2d4]=bytes((i*23)&255 for i in range(0x1a0))
    data[0x50020:0x500a0]=struct.pack('<II',6,0x134)+b'\0'*0x78
    for i,q in enumerate(sh):data[0x61000+i*40:0x61000+(i+1)*40]=SHDR.pack(*q)
    path.write_bytes(data)


def apply(source,output):
    _,_,_,rows=patches(source,131072);data=bytearray(source.read_bytes())
    for row in rows:
        start=row['offset'];new=bytes.fromhex(row['replacement_hex'])
        data[start:start+len(new)]=new
    output.write_bytes(data);return rows


def test_split_preserves_all_sections_payload_and_reverses_exactly(tmp_path):
    source=tmp_path/'old.so';output=tmp_path/'new.so';fixture(source)
    rows=apply(source,output);checks=verify(source,output,rows,131072)
    assert checks['all_other_file_bytes_identical']
    assert checks['readonly_LOAD_filesz_max']==131072
    _,ph,sh=headers(output);_,_,old_sh=headers(source)
    assert len(ph)==9
    # Every original constant, instruction and writable section remains exact.
    for old,q in zip(old_sh,sh):
        if q[1]==6:continue  # Only the typed DT_SYMTAB pointer follows .dynsym.
        assert source.read_bytes()[old[4]:old[4]+old[5]]==output.read_bytes()[q[4]:q[4]+q[5]]
    assert headers(output)[0][5]==52
    assert source.read_bytes()[0x1000:0x3507c]==output.read_bytes()[0x1000:0x3507c]
    reverted=bytearray(output.read_bytes())
    for row in rows:
        data=bytes.fromhex(row['original_hex']);reverted[row['offset']:row['offset']+len(data)]=data
    assert reverted==source.read_bytes()


@pytest.mark.parametrize('damage',['machine','executable','BSS','alignment','nonzero_gap','section_in_gap','PHDR_order','symbol_tag'])
def test_unsafe_or_unsupported_images_refused(tmp_path,damage):
    source=tmp_path/'old.so';fixture(source);data=bytearray(source.read_bytes())
    e,ph,sh=headers(source)
    if damage=='machine':e[2]=183
    elif damage=='executable':e[1]=2
    elif damage=='BSS':ph[1][5]+=4096
    elif damage=='alignment':ph[1][7]=4096
    elif damage=='nonzero_gap':data[0x3507c]=1
    elif damage=='section_in_gap':sh[2][5]+=4
    elif damage=='symbol_tag':data[0x50024:0x50028]=struct.pack('<I',0x444)
    else:ph[0],ph[1]=ph[1],ph[0]
    data[:52]=EHDR.pack(*e)
    for i,q in enumerate(ph):data[52+i*32:52+(i+1)*32]=PHDR.pack(*q)
    for i,q in enumerate(sh):data[0x61000+i*40:0x61000+(i+1)*40]=SHDR.pack(*q)
    source.write_bytes(data)
    with pytest.raises(ValueError):patches(source,131072)


@pytest.mark.parametrize('offset',[0x1234,0x40010,0x50024,0x61000,24])
def test_constant_code_dynamic_metadata_or_entry_corruption_refused(tmp_path,offset):
    source=tmp_path/'old.so';output=tmp_path/'new.so';fixture(source);rows=apply(source,output)
    data=bytearray(output.read_bytes());data[offset]^=1;output.write_bytes(data)
    with pytest.raises(ValueError):verify(source,output,rows,131072)


@pytest.mark.parametrize('bound',[0,4096,65537,256*1024*1024+65536])
def test_unbounded_or_misaligned_chunk_size_refused(tmp_path,bound):
    source=tmp_path/'old.so';fixture(source)
    with pytest.raises(ValueError):patches(source,bound)


def test_forged_patch_cannot_hide_a_changed_weight(tmp_path):
    source=tmp_path/'old.so';output=tmp_path/'new.so';fixture(source);rows=apply(source,output)
    data=bytearray(output.read_bytes());old=data[0x1234];data[0x1234]^=1;output.write_bytes(data)
    rows.append(dict(offset=0x1234,original_hex=bytes([old]).hex(),replacement_hex=bytes([data[0x1234]]).hex()))
    with pytest.raises(ValueError,match='independently derived'):verify(source,output,rows,131072)


def test_firmware_segment_limit_rejects_too_small_read_bound(tmp_path):
    source=tmp_path/'old.so';fixture(source)
    with pytest.raises(ValueError,match='at most ten program headers'):
        patches(source,65536)


def test_small_constant_library_remains_byte_identical(tmp_path):
    source=tmp_path/'old.so';output=tmp_path/'new.so';fixture(source)
    e,ph,sh,rows=patches(source,256*1024*1024)
    assert rows==[] and (e,ph,sh)==headers(source)
    output.write_bytes(source.read_bytes())
    checks=verify(source,output,rows,256*1024*1024)
    assert checks['candidate_program_headers']==checks['original_program_headers']==8
    data=bytearray(output.read_bytes());data[0x1234]^=1;output.write_bytes(data)
    with pytest.raises(ValueError,match='Unexpected payload'):
        verify(source,output,rows,256*1024*1024)
