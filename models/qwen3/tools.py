"""Run maintainer tools with repository paths inherited by child processes.

Example: python -m models.qwen3.tools build --config local.json --output build/policy
These tools build or inspect artifacts; they are not the inference entry.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

from hexagon_runtime.compiler_dependency import compiler_root

ROOT = Path(__file__).resolve().parents[2]
# Public maintainer commands. Helper modules and retired recipe launchers are
# deliberately not executable through this entry.
COMMANDS = (
    "build", "verify", "analyze_performance", "compare_performance",
    "profile_model_memory", "probe_model_memory_arm", "run_staged_probe",
    "probe_guarded_copy_arm", "probe_helper_effects_arm",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=COMMANDS, help="Maintained build or diagnostic command")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = ROOT / "tools/qwen3" / (args.tool + ".py")
    paths = [str(ROOT / p) for p in ("tools/qwen3", "models/qwen3/runtime/arm", ".")]
    paths.append(str(compiler_root()))
    previous_path, previous_argv = sys.path[:], sys.argv
    previous_env = os.environ.get("PYTHONPATH")
    try:
        sys.path[:0] = paths
        os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([previous_env] if previous_env else []))
        sys.argv = [str(script), *args.arguments]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path[:] = previous_path
        sys.argv = previous_argv
        if previous_env is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_env


if __name__ == "__main__":
    main()
