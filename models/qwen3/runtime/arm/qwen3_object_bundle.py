"""Sealed main/constant object bundles for the existing multi-module backend."""
import hashlib
import json
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
