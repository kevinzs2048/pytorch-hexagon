"""The scalar guard must inspect bits without calling an external memcpy."""
import ctypes
import shutil
import subprocess

import pytest

from models.qwen3.paths import native_root


def test_guard_bits_and_no_external_word_copy(tmp_path):
    cc, nm = shutil.which('cc'), shutil.which('nm')
    if not cc or not nm:
        pytest.skip('C compiler and nm required')
    library = tmp_path / 'guard.so'
    subprocess.run([cc, '-O2', '-fno-builtin', '-shared', '-fPIC',
                    str(native_root() / 'runtime/profile/zero_tail_hvx_guard.c'),
                    '-o', str(library)], check=True)
    undefined = subprocess.check_output([nm, '-u', str(library)], text=True)
    assert 'memcpy' not in undefined
    guard = ctypes.CDLL(str(library)).qwen_zero_words
    guard.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    guard.restype = ctypes.c_bool
    data = (ctypes.c_uint32 * 320)()
    address = ctypes.addressof(data)
    assert not guard(None, 0)
    assert not guard(address + 1, 4)
    assert not guard(address, 3)
    # Four-byte alignment, vector boundaries, tails, and exact bit predicates.
    for offset in range(32):
        for words in (0, 1, 31, 32, 33, 127, 128, 129):
            start = address + 4 * offset
            assert guard(start, 4 * words)
            for index in {offset, offset + words - 1} if words else ():
                for bits in (1, 0x80000000, 0x7fc00001, 0x3f800000, 0xffffffff):
                    data[index] = bits
                    assert not guard(start, 4 * words)
                    data[index] = 0
            # A nonzero word outside the requested span must not affect it.
            data[offset + words] = 0xffffffff
            assert guard(start, 4 * words)
            data[offset + words] = 0
