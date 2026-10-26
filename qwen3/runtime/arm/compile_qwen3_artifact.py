"""Compile exported Linalg through the actual Python Hexagon backend.

An optional explicit libtriton path selects an isolated build without replacing
the installed extension. This is an AOT compilation driver, not model inference.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def load_extension(path):
    name = "triton._C.libtriton"
    if name in sys.modules:
        raise RuntimeError("Select the extension before importing triton/torch")
    spec = importlib.util.spec_from_file_location(name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def resolve_options(defaults, reference, overrides):
    """Replay a complete recorded option set; never inherit missing baseline keys."""
    options = {key: str(value) for key, value in defaults.items()}
    provenance = None
    if reference is not None:
        raw = reference.read_bytes()
        metadata = json.loads(raw)
        if not isinstance(metadata, dict):
            raise ValueError("Recorded compilation metadata must be an object")
        recorded = metadata.get("options")
        if not isinstance(recorded, dict) or set(recorded) != set(options):
            raise ValueError("Recorded options must contain exactly the current option keys")
        if any(not isinstance(value, str) for value in recorded.values()):
            raise ValueError("Recorded option values must be strings")
        options = recorded.copy()
        provenance = {"path": str(reference.resolve()),
                      "sha256": hashlib.sha256(raw).hexdigest()}
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or key not in options or not value:
            raise ValueError(f"Unknown or malformed option: {item}")
        if value.lower() in ("true", "false"):
            value = value.title()
        options[key] = value
    return options, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="Single object or LLVM IR output")
    destination.add_argument("--output-bundle", type=Path,
                             help="Fresh directory containing a sealed main/constant object bundle")
    parser.add_argument("--emit", choices=("llvmir", "obj"), default="obj")
    parser.add_argument("--libtriton", type=Path)
    parser.add_argument("--triton-python", type=Path)
    parser.add_argument("--backend-source", type=Path,
                        help="Compiler source root supplying options; use the explicit extension without an installed Triton package")
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--options-from", type=Path,
                        help="Replay all options from an existing compilation JSON; explicit --option overrides apply last")
    parser.add_argument("--progress", action="store_true", help="Print compiler phase boundaries without dumping model weights")
    parser.add_argument("--mlir-progress", action="store_true", help="Trace individual MLIR passes without dumping operations or weights")
    parser.add_argument("--codegen-timing", action="store_true", help="Report LLVM machine-code pass timings")
    args = parser.parse_args()
    if args.output_bundle is not None and args.emit != "obj":
        parser.error("Object bundles require --emit obj")
    metadata_path = Path(str(args.output) + ".json") if args.output is not None else None
    destination_path = args.output if args.output is not None else args.output_bundle
    if destination_path.exists() or (metadata_path is not None and metadata_path.exists()):
        parser.error("Choose a new output path; existing artifacts are preserved")
    if args.backend_source and (not args.libtriton or args.triton_python):
        parser.error("backend-source requires libtriton and cannot use triton-python")
    if args.triton_python:
        sys.path.insert(0, str(args.triton_python.resolve()))
    selected_identity = selected_digest = None
    if args.libtriton:
        selected_identity = file_identity(args.libtriton)
        selected_digest = digest(args.libtriton)
        selected_extension = load_extension(args.libtriton)
        if file_identity(args.libtriton) != selected_identity:
            raise RuntimeError("Extension changed while it was being loaded; retry after the build")
    if args.progress:
        os.environ["HEXAGON_COMPILE_PROGRESS"] = "true"
    if args.mlir_progress:
        os.environ["HEXAGON_MLIR_PROGRESS"] = "true"
    if args.codegen_timing:
        os.environ["HEXAGON_CODEGEN_TIMING"] = "true"

    if args.backend_source:
        libtriton = selected_extension
        ir, backend = libtriton.ir, libtriton.qcom_hexagon_backend
        options_path = args.backend_source / "qcom_hexagon_backend/backend/hexagon_options.py"
        spec = importlib.util.spec_from_file_location("qwen_build_options", options_path)
        options_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = options_module
        spec.loader.exec_module(options_module)
        HexagonOptions = options_module.HexagonOptions
    else:
        from triton._C import libtriton
        from triton._C.libtriton import ir, qcom_hexagon_backend as backend
        from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions

    try:
        options, option_source = resolve_options(
            HexagonOptions().__dict__, args.options_from, args.option)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    extension = Path(libtriton.__file__).resolve()
    if args.libtriton and extension != args.libtriton.resolve():
        raise RuntimeError(f"Wrong extension loaded: {extension}")
    input_identity = file_identity(args.input)
    metadata = {
        "input": str(args.input.resolve()),
        "input_sha256": digest(args.input),
        "extension": str(extension),
        "extension_sha256": selected_digest if selected_digest else digest(extension),
        "extension_identity_at_load": selected_identity,
        "codegen_timing": args.codegen_timing,
        "mlir_progress": args.mlir_progress,
        "options": options,
        "options_from": option_source,
        "emit": args.emit,
        "scope": "Python backend AOT compilation only; not device validation",
    }
    print(json.dumps(metadata, indent=2), flush=True)
    context = ir.context()
    if options.get("lowerConstantsInSeparateSharedObjects") == "True":
        context.disable_multithreading()
    backend.load_dialects(context)
    parse_started = time.monotonic()
    module = backend.parse_mlir_module_from_file(str(args.input), context)
    if file_identity(args.input) != input_identity:
        raise RuntimeError("Input changed while parsing")
    metadata["parse_seconds"] = time.monotonic() - parse_started
    started = time.monotonic()
    if args.emit == "llvmir":
        output = backend.translate_linalg_to_llvmir(module, options)
        data = output.encode("utf-8")
        if options.get("enableHexagonBitCopyGeneration") == "False":
            if '"hexagon-no-bit-copy-generation"' not in output:
                raise RuntimeError("Requested safety attribute was not emitted")
        if options.get("enableHexagonGenInsert") == "False":
            if '"hexagon-no-gen-insert"' not in output:
                raise RuntimeError("Requested GenInsert policy was not emitted")
    else:
        objects = backend.translate_linalg_to_obj(module, options)
        if args.output_bundle is not None:
            from qwen3_object_bundle import write_bundle
            metadata["compile_seconds"] = time.monotonic() - started
            metadata = write_bundle(args.output_bundle, objects, metadata)
            print(f"Wrote {args.output_bundle}: {len(objects)} modules, {metadata['output_bytes']} bytes", flush=True)
            return
        if len(objects) != 1:
            raise RuntimeError("This probe expects one object module")
        data = bytes(objects[0])
        if data[:4] != b"\x7fELF":
            raise RuntimeError("Backend did not return an ELF object")
    metadata["compile_seconds"] = time.monotonic() - started
    metadata["output_sha256"] = hashlib.sha256(data).hexdigest()
    metadata["output_bytes"] = len(data)
    with args.output.open("xb") as stream:
        stream.write(data)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output}: {len(data)} bytes", flush=True)


if __name__ == "__main__":
    main()
