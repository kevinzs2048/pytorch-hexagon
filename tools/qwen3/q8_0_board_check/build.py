# SPDX-License-Identifier: BSD-3-Clause
"""Build the Q8_0 board check (host side).

Compiles models/qwen3/native/runtime/q8_0/q8_0_hmx.cpp and probe.cpp into a
DSP library linked against the weight library of models/qwen3/q8_0/library.py,
and collects what check.py needs on the board: probe.so, the weight library,
weights.json, the packed tiles of the selected ids and quantize.py.

Usage: build.py --weights W --library L --link-command JSON --out DIR [--ids 0,...]
--link-command is the model link command; its -I/-L/--ld-path flags, the
parallel_for source and the attention archive (HMX dispatcher runtime) are reused.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True, help="models/qwen3/q8_0/weights.py output")
    p.add_argument("--library", type=Path, required=True, help="models/qwen3/q8_0/library.py output")
    p.add_argument("--link-command", type=Path, required=True)
    p.add_argument("--hexagon-tools", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids", default="0,1,2,3,4,5,6,196")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = json.loads(a.link_command.read_text())
    flags = [x for x in model if x.startswith(("-I", "--ld-path"))]
    archive = next(x for x in model if x.endswith("libqwen_triton_experimental_attention.a"))
    hexkl = Path(next(x for x in model if x.endswith("libhexkl_micro.a")))
    flags.append("-I" + str(hexkl.parents[2] / "include"))  # hexkl_addon/lib/<toolchain>/ -> hexkl_addon/include
    tools = a.hexagon_tools
    cc = [str(tools / "bin/hexagon-clang++"), "-O3", "-mv73", "-mhvx", "-mhvx-length=128B", "-mhmx", "-fPIC", "-G0",
          "-mlong-calls", "-std=c++17", "-fno-math-errno", "-Drestrict=__restrict", *flags]
    lib = tools / "target/hexagon/lib/v73/G0/pic"
    subprocess.run(cc + ["-mhvx-ieee-fp", "-c", str(ROOT / "models/qwen3/native/runtime/q8_0/q8_0_hmx.cpp"),
                         "-o", str(a.out / "q8_0_hmx.o")], check=True)
    subprocess.run(cc + ["-shared", "-nostdlib++", str(Path(__file__).with_name("probe.cpp")), str(a.out / "q8_0_hmx.o"),
                         str(a.library / "q8_0_weight_table.cpp"), str(ROOT / "hexagon_runtime/native/parallel_for.cpp"),
                         archive, str(ROOT / "models/qwen3/native/runtime/aligned_alloc_shim.cpp"),
                         str(lib / "libc++.a"), str(lib / "libc++abi.a"),
                         *[x for x in model if x.endswith("libhexkl_micro.a")], str(lib / "libgcc.a"),
                         *[x for x in model if x.startswith("-L") or x == "-lqhmath_hvx"],
                         "-L" + str(a.library), "-lqwen_q8_0_weights", "-o", str(a.out / "probe.so")], check=True)
    for f in (a.library / "libqwen_q8_0_weights.so", a.weights / "weights.json",
              ROOT / "models/qwen3/q8_0/quantize.py", Path(__file__).with_name("check.py")):
        shutil.copy2(f, a.out / f.name)
    for i in a.ids.split(","):
        shutil.copy2(a.weights / f"{int(i)}.q8", a.out / f"{int(i)}.q8")
    print("built", a.out)


if __name__ == "__main__":
    main()
