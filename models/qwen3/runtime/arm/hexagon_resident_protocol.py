"""Shared wire handling for the admitted Qwen resident-model ABI."""
import struct
import time

import torch

VOCAB = 151936
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
MAX_QUERY = 32
MAX_LENGTH = 1024


def header(mode, position, query=0, chunk=0):
    return b'QWMS001\0' + struct.pack('<IIII', mode, position, query, chunk) + bytes(40)


def response_statistics(raw, size, length):
    if len(raw) != size or raw[:8] != b'QWMS001\0' or any(raw[56:64]):
        raise ValueError('Invalid resident response header or size')
    stats = dict(zip(('length', 'capacity', 'appended_bytes', 'relocated_bytes', 'update_us', 'model_us'),
                     struct.unpack_from('<6Q', raw, 8)))
    if stats['length'] != length or not stats['length'] <= stats['capacity'] <= MAX_LENGTH:
        raise ValueError('Resident response cache extent differs')
    return stats


def forward_batch(stream, ids, position, chunk):
    """One DSP call for a logical batch (mode 5, sessions that advertise it).
    The DSP runs the forwards a caller would send one ``chunk`` at a time and
    returns the last one's logits; the statistics are their sums."""
    if (ids.device.type != 'cpu' or ids.dtype != torch.int64 or ids.ndim != 2
            or ids.shape[0] != 1):
        raise ValueError('Resident tokens must be a batch-one CPU int64 tensor')
    query = ids.shape[1]
    if (not 1 <= chunk <= MAX_QUERY or not 1 <= query
            or not 0 <= position <= MAX_LENGTH - query):
        raise ValueError('Resident query/cache extent is unsupported')
    if bool(((ids < 0) | (ids >= VOCAB)).any()):
        raise ValueError('Token ID is outside the compiled vocabulary')
    size = 64 + VOCAB * 4
    raw, event = stream.invoke_packet(header(5, position, query, chunk) + ids.contiguous().numpy().tobytes(), size)
    exchanged = time.perf_counter()
    stats = response_statistics(raw, size, position + query)
    logits = torch.frombuffer(bytearray(raw[64:]), dtype=torch.float32).reshape(1, 1, VOCAB).clone()
    return logits, stats, event, exchanged


def forward(stream, ids, position, *, resident_only=False, logits=True):
    """One DSP forward. ``logits=False`` (mode 4, resident-only sessions that
    advertise it) appends to the DSP cache and returns no prediction."""
    if not logits and not resident_only:
        raise ValueError('Only a resident-only forward may omit its logits')
    if (ids.device.type != 'cpu' or ids.dtype != torch.int64 or ids.ndim != 2
            or ids.shape[0] != 1):
        raise ValueError('Resident tokens must be a batch-one CPU int64 tensor')
    query = ids.shape[1]
    if not 1 <= query <= MAX_QUERY or not 0 <= position <= MAX_LENGTH - query:
        raise ValueError('Resident query/cache extent is unsupported')
    if bool(((ids < 0) | (ids >= VOCAB)).any()):
        raise ValueError('Token ID is outside the compiled vocabulary')
    size = 64 + ((VOCAB if logits else 0) + (0 if resident_only else 2 * LAYERS * query * KV_HEADS * HEAD_DIM)) * 4
    mode = 1 if not resident_only else 3 if logits else 4
    raw, event = stream.invoke_packet(header(mode, position, query) + ids.contiguous().numpy().tobytes(), size)
    exchanged = time.perf_counter()
    stats = response_statistics(raw, size, position + query)
    if not logits:
        return None, None, stats, event, exchanged
    data = torch.frombuffer(bytearray(raw[64:]), dtype=torch.float32)
    logits = data[:VOCAB].reshape(1, 1, VOCAB).clone()
    if resident_only:
        return logits, None, stats, event, exchanged
    delta = data[VOCAB:].reshape(2 * LAYERS, query, KV_HEADS, HEAD_DIM).transpose(1, 2).unsqueeze(1)
    return logits, delta, stats, event, exchanged


def verify_cache(stream, position, planes):
    size = 64 + 2 * LAYERS * position * KV_HEADS * HEAD_DIM * 4
    raw, _ = stream.invoke_packet(header(2, position), size)
    if len(raw) != size or raw[:8] != b'QWMS001\0' or struct.unpack_from('<Q', raw, 8)[0] != position:
        raise ValueError('Invalid resident cache snapshot')
    resident = torch.frombuffer(bytearray(raw[64:]), dtype=torch.float32).reshape(2 * LAYERS, position, KV_HEADS, HEAD_DIM)
    shadow = torch.stack(planes).squeeze(1).transpose(1, 2).contiguous()
    if not torch.equal(resident.view(torch.int32), shadow.view(torch.int32)):
        raise ValueError('DSP history and native cache outputs differ')
