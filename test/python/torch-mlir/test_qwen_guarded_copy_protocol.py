"""Verify independent copy oracles and rejection of incomplete DSP evidence."""
import importlib.util
from pathlib import Path
import struct

import pytest

SOURCE = Path(__file__).resolve().parents[3] / 'tools/qwen3/probe_guarded_copy_arm.py'
spec = importlib.util.spec_from_file_location('guarded_copy_probe', SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_all_descriptors_have_valid_packet_bounds_and_unchanged_sentinels():
    for case in probe.cases():
        payload, expected = probe.packet_and_reference(case, 2)
        assert len(payload) == probe.INPUT_BYTES and len(expected) == 4096
        assert payload[:8] == b'QWCOPY1\0'
        repeats, rank, element, src_offset, dst_offset = struct.unpack_from('<IIqqq', payload, 8)
        assert repeats == 2 and rank == len(case[1]) and element == case[-1]
        if case[0] == 'hot-512':
            assert expected[:512] == payload[256:768] and expected[512:] == bytes([0xa5] * (4096 - 512))
        if case[0] == 'empty':
            assert expected == bytes([0xa5] * 4096)
        if case[0] == 'scalar':
            assert expected[dst_offset*element:(dst_offset+1)*element] == payload[256+src_offset*element:256+(src_offset+1)*element]
        # Full destination is idempotent, as required by the repeated timing loop.
        assert probe.packet_and_reference(case, 32768)[1] == expected


def test_integer_oracle_distinguishes_native_32bit_extent_from_signed_product():
    payload, expected = probe.guard_packet_and_reference(1024)
    assert len(payload) == 128 + 32*16 and len(expected) == 32*24
    assert len(set(probe.guard_pairs())) == 32
    for index, (a, b) in enumerate(probe.guard_pairs()):
        signed, extent, product, count = struct.unpack_from('<IIqQ', expected, index*24)
        assert bool(signed) == (a*b <= (1 << 63)-1)
        assert bool(extent) == (a*b <= (1 << 32)-1)
        assert product == (a*b if signed else -123)
        assert count == (a*b if extent else 123)
    pair_index = probe.guard_pairs().index((1 << 32, 1))
    assert struct.unpack_from('<II', expected, pair_index*24) == (1, 0)


@pytest.mark.parametrize('guard', [False, True])
def test_actual_timing_header_and_dsp_word_size_required(guard):
    packet = bytearray(128 + 32*24 if guard else probe.OUTPUT_BYTES)
    struct.pack_into('<QIIQ', packet, 0, 50, 2, 4 if guard else 0, 48000)
    assert probe.parse_response(packet, 2, guard)['elapsed_us'] == 50
    for offset, value in [(0, 0), (8, 1), (12, 8), (16, 0), (24, 1)]:
        changed = bytearray(packet)
        struct.pack_into('<Q' if offset in (0, 16) else '<I', changed, offset, value)
        with pytest.raises(ValueError):
            probe.parse_response(changed, 2, guard)
    with pytest.raises(ValueError):
        probe.parse_response(packet[:-1], 2, guard)


def test_oracle_rejects_address_outside_packet():
    with pytest.raises(ValueError):
        probe.packet_and_reference(('outside', [2], [-1], [1], 0, 0, 4), 1)
