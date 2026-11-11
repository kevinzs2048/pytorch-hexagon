# SPDX-License-Identifier: BSD-3-Clause
"""Q8_0 runtime board check (board side; run from the build.py output directory).

Each selected id is run as a prefill projection (m = 32, HMX FP16 path) and as
a single row (Q8_0 x Q8_0 path) with its graph mode (plain, residual, product);
the LM head (n > 3072) only as a single row. References: quantize.hmx_matmul
for the HMX path (the HMX accumulator differs from an FP32 sum: within one FP16
ULP of the largest output magnitude, times |side| for the product) and
quantize.q8_q8_matvec for the single row (bit-exact). Exits non-zero on any
mismatch.

Usage: python3 check.py --runtime BUNDLE_RUNTIME_DIR --application APP_DIR [--ids ...]
"""
import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from quantize import hmx_matmul, q8_q8_matvec, unpack  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True, help="directory with buffer_stream and the skel")
    p.add_argument("--application", type=Path, required=True, help="directory providing hexagon_runtime")
    p.add_argument("--ids", default=None)
    p.add_argument("--lock", default="/tmp/qwen3-qcs9075-probe.lock")
    a = p.parse_args()
    sys.path.insert(0, str(a.application))
    from hexagon_runtime.hexagon_shared_packet_transport import HexagonSharedPacketStream
    meta = {r["id"]: r for r in json.loads((here / "weights.json").read_text())["projections"]}
    ids = [int(i) for i in a.ids.split(",")] if a.ids else sorted(int(f.stem) for f in here.glob("*.q8"))
    rng = np.random.default_rng(7)
    cases = []
    for i in ids:
        r = meta[i]
        base = {"plain": 0, "residual": 1, "product": 2}[r["mode"]]
        cases += ([(i, 32, base)] if r["n"] <= 3072 else []) + [(i, 1, 3 + base)]
    in_bytes = 64 + 4 * max(m * meta[i]["k"] + m * meta[i]["n"] for i, m, _ in cases)
    out_bytes = 64 + 4 * max(m * meta[i]["n"] for i, m, _ in cases)
    work = Path("/dev/shm/qwen3-q8-0-check")
    shutil.rmtree(work, ignore_errors=True)
    (work / "rfsa").mkdir(parents=True)
    (work / "buffer_stream").symlink_to(a.runtime / "buffer_stream")
    (work / "rfsa/libhexagon_buffer_runner_skel.so").symlink_to(a.runtime / "libhexagon_buffer_runner_skel.so")
    shutil.copy2(here / "probe.so", work / "model.so")
    shutil.copy2(here / "libqwen_q8_0_weights.so", work / "libqwen_q8_0_weights.so")
    env = dict(os.environ, LD_LIBRARY_PATH=str(work), ADSP_LIBRARY_PATH=f"{work};{work / 'rfsa'};/usr/lib/rfsa/adsp;"
               "/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp")
    argv = ["flock", "-n", a.lock, str(work / "buffer_stream"), str(work / "model.so"), str(in_bytes), str(out_bytes)]
    shutil.rmtree(here / "stream", ignore_errors=True)  # the transport writes fresh logs here
    (here / "stream").mkdir()
    failures, rows, previous = 0, [], Path.cwd()
    try:
        os.chdir(work)
        with HexagonSharedPacketStream(argv, env, here / "stream", input_bytes=in_bytes, output_bytes=out_bytes,
                                       initial_library=str(work / "model.so"), library_path_prefix="/dev/shm/qwen3-",
                                       timeout=120) as stream:
            stream.start()
            for i, m, mode in cases:
                n, k = meta[i]["n"], meta[i]["k"]
                q, d = unpack(np.fromfile(here / f"{i}.q8", np.uint8), n, k, vrmpy=n > 3072)
                x = rng.standard_normal((m, k)).astype(np.float32)
                side = rng.standard_normal((m, n)).astype(np.float32)
                y = hmx_matmul(x, q, d) if mode < 3 else q8_q8_matvec(x[0], q, d)[None]
                ref = (side + y if mode % 3 == 1 else side * y if mode % 3 == 2 else y).astype(np.float32)
                raw, _ = stream.invoke_packet(b"QWQ8002\0" + struct.pack("<5I", m, n, k, mode, i) + bytes(36) +
                                              x.tobytes() + side.tobytes(), 64 + 4 * m * n)
                got = np.frombuffer(raw, np.float32, m * n, 64).reshape(m, n)
                if mode < 3:  # one FP16 ULP at the largest projection magnitude, before the epilogue
                    ulp = float(np.spacing(np.float16(np.abs(y).max())))
                    scale = np.abs(side) if mode == 2 else 1
                    ok = bool((np.abs(got - ref) <= ulp * scale + np.abs(ref) * 2 ** -22).all())
                else:
                    ok = bool(np.array_equal(got, ref))
                failures += not ok
                rows.append(dict(id=i, m=m, n=n, k=k, mode=mode, ok=ok, us=struct.unpack_from("<Q", raw, 16)[0],
                                 max_abs_err=float(np.abs(got - ref).max())))
                print(rows[-1], flush=True)
            stream.close()
    finally:
        os.chdir(previous)
        shutil.rmtree(work, ignore_errors=True)
    (here / "result.json").write_text(json.dumps(dict(cases=rows, failures=failures), indent=1))
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
