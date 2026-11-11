# SPDX-License-Identifier: BSD-3-Clause
"""Q8_0 weights for the --qwen-q8-0-projections ids, from the HF checkpoint.

The rewrite pass dumps, per call ordinal (id), the prepacked INT8 weight and
its column scales. Each id is identified as the unique HF linear weight whose
per-channel INT8 rounding it is (|q * s - w| <= s / 2), quantized to Q8_0 with
gguf-py (the quantizer of llama.cpp's convert_hf_to_gguf.py, so the bytes equal
a Q8_0 GGUF of the same checkpoint) and packed for the DSP runtime: the only
projection wider than 3072 outputs (the LM head) in the vrmpy layout, every
other one in the HMX row-pair layout. An optional GGUF is a cross-check only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .quantize import gguf_blocks, pack, pack_vrmpy

PROJECTIONS = (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
               ("self_attn.o_proj", "attn_output"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
               ("mlp.down_proj", "ffn_down"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack_prepacked_int8(packed, n, k):
    """Undo PrepackInt8Weights: packed[(c/128)*k*128 + (r/4)*512 + j*4 + t] = q[c + j][r + t]."""
    return packed.reshape(n // 128, k // 4, 128, 4).transpose(0, 2, 1, 3).reshape(n, k)


def build(dump, hf_checkpoint, gguf_py, out, cross_check_gguf=None, log=print):
    """dump: --qwen-q8-0-projections dump-dir; writes <id>.q8 blobs and weights.json to out."""
    dump, out = Path(dump), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(gguf_py)))
    import gguf
    from safetensors.torch import load_file
    hf = {k: v.float().numpy() for k, v in load_file(str(hf_checkpoint)).items()}
    layers = 1 + max(int(k.split(".")[2]) for k in hf if k.startswith("model.layers."))
    names = {f"model.layers.{i}.{h}.weight": f"blk.{i}.{g}.weight" for i in range(layers) for h, g in PROJECTIONS}
    names["lm_head.weight" if "lm_head.weight" in hf else "model.embed_tokens.weight"] = "output.weight"
    manifest = json.loads((dump / "manifest.json").read_text())
    reference = {t.name: t for t in gguf.GGUFReader(str(cross_check_gguf)).tensors} if cross_check_gguf else {}
    used, rows, cross = set(), [], 0
    for e in manifest:
        i, n, k = e["id"], e["n"], e["k"]
        s = np.fromfile(dump / f"{i}.s", np.float32)
        w8 = unpack_prepacked_int8(np.fromfile(dump / f"{i}.w", np.int8), n, k).astype(np.float32) * s[:, None]
        candidates = [h for h in names if h not in used and hf[h].shape == (n, k)]
        if not candidates:
            raise ValueError(f"Q8_0 id {i}: no unused HF projection of shape {n}x{k}")
        error = {h: float((np.abs(w8 - hf[h]) / s[:, None]).max()) for h in candidates}
        name = min(error, key=error.get)
        if error[name] > 0.5 + 1e-3 or sum(v <= 0.5 + 1e-3 for v in error.values()) != 1:
            raise ValueError(f"Q8_0 id {i}: no unique HF weight for the INT8 constant ({error[name]:.4f})")
        used.add(name)
        raw = gguf.quants.quantize(hf[name].astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
        raw = np.asarray(raw).reshape(-1).view(np.uint8)
        if cross_check_gguf:
            t = reference[names[name]]
            if t.tensor_type != gguf.GGMLQuantizationType.Q8_0 or not np.array_equal(
                    np.asarray(t.data).reshape(-1).view(np.uint8), raw):
                raise ValueError("GGUF cross-check differs: " + names[name])
            cross += 1
        q, d = gguf_blocks(raw, n, k)
        layout = "hvx_vrmpy" if n > 3072 else "hmx_f16"
        blob = pack_vrmpy(q, d) if layout == "hvx_vrmpy" else pack(q, d)
        blob.tofile(out / f"{i}.q8")
        rows.append(dict(id=i, n=n, k=k, mode=e["mode"], layout=layout, hf=name, gguf_name=names[name],
                         int8_max_err_over_scale=error[name], q8_0_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                         packed_sha256=hashlib.sha256(blob.tobytes()).hexdigest(), packed_bytes=int(blob.size)))
        log(f"Q8_0 id {i}: {name}")
    if not (len(used) == len(manifest) == len(names)):
        raise ValueError(f"Q8_0 coverage: {len(used)} identified, {len(manifest)} ids, {len(names)} HF projections")
    record = dict(source="HF safetensors -> gguf-py quants.quantize(Q8_0)", hf=str(hf_checkpoint), hf_sha256=sha(hf_checkpoint),
                  manifest_sha256=sha(dump / "manifest.json"),
                  gguf_cross_check=dict(path=str(cross_check_gguf), sha256=sha(cross_check_gguf), tensors_identical=cross)
                  if cross_check_gguf else None, projections=rows)
    (out / "weights.json").write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--hf", type=Path, required=True, help="model.safetensors")
    p.add_argument("--gguf-py", type=Path, required=True, help="llama.cpp gguf-py directory")
    p.add_argument("--cross-check-gguf", type=Path)
    a = p.parse_args()
    r = build(a.dump, a.hf, a.gguf_py, a.out, a.cross_check_gguf)
    print(len(r["projections"]), "projections,", sum(x["packed_bytes"] for x in r["projections"]), "bytes")
