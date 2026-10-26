"""Sealed main/constant object bundles for the existing multi-module backend."""
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_bundle(directory, objects, metadata):
    """Backend order is main first, followed by constants-only modules."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Choose a fresh bundle directory")
    if not objects or any(bytes(obj[:4]) != b"\x7fELF" for obj in objects):
        raise ValueError("Every bundle module must be an ELF object")
    records = [{"file": f"module-{index:04d}.o", "role": "main" if index == 0 else "constants",
                "sha256": hashlib.sha256(obj).hexdigest(), "bytes": len(obj)}
               for index, obj in enumerate(objects)]
    record = dict(metadata, bundle_format="qwen3-hexagon-objects-v1", objects=records,
                  output_bytes=sum(item["bytes"] for item in records))
    directory.mkdir()
    for obj, item in zip(objects, records):
        with (directory / item["file"]).open("xb") as stream:
            stream.write(obj)
    with (directory / "bundle.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def load_bundle(directory):
    directory = Path(directory)
    record = json.loads((directory / "bundle.json").read_text())
    if not isinstance(record, dict) or record.get("bundle_format") != "qwen3-hexagon-objects-v1":
        raise ValueError("Unsupported object bundle format")
    items = record.get("objects")
    if not isinstance(items, list) or not items:
        raise ValueError("Bundle must contain at least a main object")
    paths = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("file") != f"module-{index:04d}.o" or \
                item.get("role") != ("main" if index == 0 else "constants"):
            raise ValueError("Bundle object order, filename or role is invalid")
        path = directory / item["file"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Bundle objects must be regular nonsymlink files")
        if type(item.get("bytes")) is not int or path.stat().st_size != item["bytes"] or \
                digest(path) != item.get("sha256"):
            raise ValueError("Bundle object bytes drifted")
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("Bundle object is not ELF")
        paths.append(path)
    if record.get("output_bytes") != sum(item["bytes"] for item in items):
        raise ValueError("Bundle total byte count is inconsistent")
    return record, paths


def load_constant_libraries(directory, compilation, linker_sha256):
    """Reuse data DSOs only when graph, ordered constant objects and linker match.

    A previous numerical failure does not invalidate immutable constant files.
    No accuracy result is reused; the new main object and wrapper still run.
    The selected directory must itself contain the original linked libraries.
    """
    directory = Path(directory)
    metadata = directory / "result.json"
    raw = metadata.read_bytes()
    previous = json.loads(raw)
    source = previous.get("compilation", {})
    if source.get("input_sha256") != compilation.get("input_sha256"):
        raise ValueError("Constant-library source belongs to another graph")
    if previous.get("linker_sha256") != linker_sha256:
        raise ValueError("Constant-library linker differs")
    expected = compilation.get("objects", [])[1:]
    recorded = source.get("objects", [])[1:]
    if not expected or len(recorded) != len(expected):
        raise ValueError("Constant-object count differs")
    for actual, wanted in zip(recorded, expected):
        if actual.get("role") != "constants" or wanted.get("role") != "constants" or \
                any(actual.get(key) != wanted.get(key) for key in ("sha256", "bytes")):
            raise ValueError("Constant-object bytes or order differ")
    libraries = previous.get("constant_libraries", [])
    if len(libraries) != len(expected):
        raise ValueError("Linked constant-library count differs")
    paths = []
    for item in libraries:
        name = item.get("file", "")
        if not isinstance(name, str) or re.fullmatch(r"lib[A-Za-z0-9_.-]+\.so", name) is None:
            raise ValueError("Invalid constant-library basename")
        path = directory / name
        if path in paths or path.is_symlink() or not path.is_file():
            raise ValueError("Constant libraries must be unique regular nonsymlink files")
        if type(item.get("bytes")) is not int or path.stat().st_size != item["bytes"] or \
                digest(path) != item.get("sha256"):
            raise ValueError("Linked constant-library bytes drifted")
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("Constant library is not ELF")
        paths.append(path.resolve())
    return paths, {"source_result": str(metadata.resolve()),
                   "source_result_sha256": hashlib.sha256(raw).hexdigest(),
                   "scope": "immutable constant DSOs only; no numerical result reused"}
