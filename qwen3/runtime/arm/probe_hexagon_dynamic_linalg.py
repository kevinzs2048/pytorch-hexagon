"""Locate the first Hexagon compiler result for a real dynamic Qwen Linalg graph.

The pinned graph may be unbound or have frozen weights. No object is linked
and no DSP session is opened: this is not model execution acceptance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("linalg", "compiler", "options-result"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--triton-python", type=Path, required=True)
    parser.add_argument("--disable-vectorization", action="store_true")
    parser.add_argument("--disable-split-reduction", action="store_true")
    parser.add_argument("--emit", choices=("llvmir", "obj"), default="llvmir")
    parser.add_argument("--code-only", action="store_true",
                        help="Retain only code object; hash/check unmodified constant objects in memory")
    parser.add_argument("--allow-code-difference", action="store_true",
                        help="With code-only, permit code-object changes while pinning all constants")
    parser.add_argument("--expected-objects-result", type=Path)
    parser.add_argument("--expected-objects-result-sha256")
    parser.add_argument("--graph-kind", choices=("dynamic", "dynamic-bound", "static-regression"),
                        default="dynamic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("linalg", "compiler", "options_result"):
        path, expected = getattr(args, name), getattr(args, name + "_sha256")
        if digest(path) != expected:
            raise ValueError("Pinned compiler input drift: " + name)
    if args.allow_code_difference and not args.code_only:
        raise ValueError("Changed code can only be admitted by a code-only replay")
    if args.code_only:
        if (args.emit != "obj" or args.expected_objects_result is None or
                args.expected_objects_result_sha256 is None or
                digest(args.expected_objects_result) != args.expected_objects_result_sha256):
            raise ValueError("Code-only requires pinned reference objects and object emission")
        expected = json.loads(args.expected_objects_result.read_text())
        if (expected.get("status") !=
                "dynamic-bound_q17_lifted_graph_Hexagon_objects_compiled_NOT_model_or_DSP"
                or len(expected.get("objects", [])) != 4):
            raise ValueError("Exact four-object dynamic-bound reference required")
    else:
        expected = None
    minimum_space = (1024 if args.code_only else 2048) * 1024**2
    if args.emit == "obj" and shutil.disk_usage(args.output.parent).free < minimum_space:
        raise ValueError("Insufficient artifact-output space")
    args.output.mkdir(exist_ok=False)
    record = {
        "status": "admitting_dynamic_linalg",
        "source_sha256": digest(__file__),
        "inputs": {name: {"path": str(getattr(args, name)),
                          "sha256": getattr(args, name + "_sha256")}
                   for name in ("linalg", "compiler", "options_result")},
        "machine_code_compilations": 0,
        "DSP_calls": 0,
        "model_execution": False,
        "graph_kind": args.graph_kind,
    }
    if args.code_only:
        record["expected_objects_result"] = {
            "path": str(args.expected_objects_result),
            "sha256": args.expected_objects_result_sha256,
        }
        record["allow_code_difference"] = args.allow_code_difference

    def save():
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")

    diagnostic = args.output / "compiler-diagnostics.txt"

    def translate(action):
        # The Hexagon extension emits MLIR verifier failures directly to the
        # process stderr file descriptor, not Python's sys.stderr.
        with diagnostic.open("xb") as stream:
            previous = os.dup(2)
            os.dup2(stream.fileno(), 2)
            try:
                return action()
            finally:
                os.dup2(previous, 2)
                os.close(previous)

    save()
    try:
        from hexagon_python_source import configure_triton_python
        record["triton_python"] = configure_triton_python(args.triton_python)
        from compile_qwen3_artifact import load_extension

        load_extension(args.compiler)
        from triton._C.libtriton import ir, qcom_hexagon_backend as backend
        from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions

        previous = json.loads(args.options_result.read_text())
        raw = previous["options"]
        defaults = HexagonOptions().__dict__
        if ((set(defaults) - set(raw)) != {"outlineCheckedVectorExp"}
                or (set(raw) - set(defaults)) != {"keepAssertMessagesInMain"}):
            raise ValueError("Recorded q32 compiler option keys differ: " +
                             f"missing={sorted(set(defaults) - set(raw))}, " +
                             f"extra={sorted(set(raw) - set(defaults))}")
        options = {key: str(value) for key, value in raw.items()}
        options["outlineCheckedVectorExp"] = str(defaults["outlineCheckedVectorExp"])
        if args.disable_vectorization:
            options["enableVectorization"] = "False"
        if args.disable_split_reduction:
            options["enableSplitReduction"] = "False"
        record["overrides"] = {
            "enableVectorization": options["enableVectorization"],
            "enableSplitReduction": options["enableSplitReduction"],
        }
        record["option_reconciliation"] = {
            "only_missing_default": "outlineCheckedVectorExp",
            "default_value": options["outlineCheckedVectorExp"],
            "recorded_extra": "keepAssertMessagesInMain",
        }
        context = ir.context()
        if raw.get("lowerConstantsInSeparateSharedObjects") is True:
            context.disable_multithreading()
        backend.load_dialects(context)
        module = backend.parse_mlir_module_from_file(str(args.linalg), context)
        record["status"] = "lowering_q17_Linalg_to_Hexagon_" + args.emit
        save()
        start = time.monotonic()
        if args.emit == "llvmir":
            output = translate(lambda: backend.translate_linalg_to_llvmir(module, options))
            target = args.output / "dynamic-hexagon.ll"
            with target.open("x") as stream:
                stream.write(output)
            record.update(status=args.graph_kind +
                          "_q17_lifted_graph_lowered_to_Hexagon_LLVM_NOT_model_or_DSP",
                          elapsed_seconds=time.monotonic() - start,
                          llvm_ir={"path": str(target), "sha256": digest(target),
                                   "bytes": target.stat().st_size})
        else:
            objects = translate(lambda: backend.translate_linalg_to_obj(module, options))
            if args.code_only and len(objects) != 4:
                raise ValueError("Expected four object members in code-only replay")
            members = []
            for index, obj in enumerate(objects):
                target = args.output / f"dynamic-hexagon-{index}.o"
                data = bytes(obj)
                if data[:4] != b"\x7fELF":
                    raise ValueError("Hexagon backend returned non-ELF object")
                member_hash = hashlib.sha256(data).hexdigest()
                if args.code_only:
                    reference = expected["objects"][index]
                    if (index != 0 or not args.allow_code_difference) and (
                            member_hash != reference["sha256"] or
                            len(data) != reference["bytes"]):
                        raise ValueError(f"Pinned object {index} changed")
                if index == 0 or not args.code_only:
                    if (len(data) > 3 * 1024**3 or
                            shutil.disk_usage(args.output).free < len(data) + 512 * 1024**2):
                        raise ValueError("Insufficient scoped space for Hexagon object member")
                    with target.open("xb") as stream:
                        stream.write(data)
                members.append({"path": str(target) if index == 0 or not args.code_only else None,
                                "retained": index == 0 or not args.code_only,
                                "sha256": member_hash, "bytes": len(data)})
            record.update(status=args.graph_kind +
                          ("_q17_lifted_graph_Hexagon_code_candidate_constants_identical_NOT_DSP"
                           if args.code_only and args.allow_code_difference else
                           "_q17_lifted_graph_Hexagon_code_replayed_identical_NOT_DSP"
                           if args.code_only else
                           "_q17_lifted_graph_Hexagon_objects_compiled_NOT_model_or_DSP"),
                          elapsed_seconds=time.monotonic() - start,
                          objects=members, machine_code_compilations=1)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__,
                      error=str(error)[:2000])
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if diagnostic.exists():
            record["compiler_diagnostics"] = {
                "path": str(diagnostic), "sha256": digest(diagnostic),
                "bytes": diagnostic.stat().st_size,
            }
        save()
    print(record["status"], flush=True)


if __name__ == "__main__":
    main()
