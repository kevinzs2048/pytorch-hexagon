"""Relocatable, hash-checked execution assets. No historical run directories."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qwen3-hexagon-bundle-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def inside(root, relative):
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError("Bundle paths must be relative and contained: " + str(relative))
    result = (root / part).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Bundle symlink escapes its root: " + str(relative))
    return result


def validate(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported execution bundle schema")
    for name, row in manifest["files"].items():
        path = inside(root, name)
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Missing or changed bundle file: " + name)
    catalog = json.loads(inside(root, "catalog.json").read_text())
    lock = json.loads((ROOT / "models/qwen3/dependencies.lock.json").read_text())
    if manifest["files"].get("catalog.json", {}).get("sha256") != lock["portable_catalog_sha256"]:
        raise ValueError("Complete captured graph/ABI catalog differs from the pinned release")
    libraries = {str(tuple(e["contract"]["variables"]["query"])): e["library"]["sha256"]
                 for e in catalog["entries"]}
    if libraries != lock["model_libraries"] or len(catalog["entries"]) != 2:
        raise ValueError("The bundle must contain both pinned execution domains")
    for logical, expected in lock["runtime_files"].items():
        if manifest["files"].get(logical, {}).get("sha256") != expected:
            raise ValueError("Pinned execution asset differs: " + logical)
    for entry in catalog["entries"]:
        name = "libraries/" + entry["library"]["sha256"] + ".so"
        if manifest["files"].get(name, {}).get("sha256") != entry["library"]["sha256"]:
            raise ValueError("Model library missing from file inventory")
    context = json.loads(inside(root, "context.json").read_text())
    if manifest["files"].get("hf/model.safetensors", {}).get("sha256") != context["checkpoint"]["sha256"]:
        raise ValueError("Model checkpoint differs from its pinned context")
    if manifest.get("namespace") != lock["namespace"]:
        raise ValueError("Framework/model namespace differs from the pinned release")
    return manifest, catalog


def export(spec_path, output):
    """Copy explicitly supplied assets; paths in the resulting bundle are relative."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)

    def source(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else spec_path.parent / p

    files = {}

    def add(src, relative):
        dst = inside(output, relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[relative] = dict(bytes=dst.stat().st_size, sha256=sha(dst))

    catalog = json.loads(source(spec["catalog"]).read_text())
    # Historical reports remain evidence, not executable dependencies.
    keys = ("graph", "parameters_sha256", "retained_input_indices", "contract",
            "binding_content_sha256", "frontend_key", "machine_key")
    entries = []
    for old in catalog["entries"]:
        digest = old["library"]["sha256"]
        add(source(spec["libraries"][digest]), "libraries/" + digest + ".so")
        entries.append({**{k: old[k] for k in keys}, "library": {"sha256": digest}})
    portable = dict(schema=1, status="warm_ARM_dynamic_catalog_NOT_executed",
                    entries=entries, runtime=copy.deepcopy(catalog["runtime"]))
    portable["runtime"] = {k: portable["runtime"][k] for k in ("shared_memory_packets", "artifacts")}
    write(output / "catalog.json", portable)
    files["catalog.json"] = dict(bytes=(output / "catalog.json").stat().st_size, sha256=sha(output / "catalog.json"))
    add(source(spec["context"]), "context.json")
    add(source(spec["decompositions"]), "policy/fx_decomp_util.py")
    for p in sorted(source(spec["hf"]).iterdir()):
        if p.is_file():
            add(p, "hf/" + p.name)
    for name, path in spec["runtime"].items():
        add(source(path), "runtime/" + name)
    for index in (1, 2, 3):
        name = f"lib_mlir_ciface_Qwen3DynamicBoundProbe-consts-{index}.so"
        add(source(spec["constants"]) / name, "constants/" + name)
    # Preserve the exact installed ARM packages, including metadata and licenses.
    # Later roots are explicit overlays (e.g. the patched Transformers release).
    # Python .pth files are not executed: the CLI inserts this directory directly.
    for directory in spec.get("python_packages", []):
        folder = source(directory)
        if not folder.is_dir():
            raise ValueError("Python package source directory is missing: " + str(folder))
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            if (path.is_file() and "__pycache__" not in rel.parts
                    and path.suffix not in (".pyc", ".pth")
                    and not rel.parts[0].startswith("__editable__")):
                add(path, "python/" + str(rel))
    lock = json.loads((ROOT / "models/qwen3/dependencies.lock.json").read_text())
    write(output / "bundle.json", dict(schema=SCHEMA, files=files, namespace=lock["namespace"],
          scope=lock["scope"], compiler_rebuilt=False))
    validate(output)
    return output / "bundle.json"


def prepare(root, prepared):
    """Recreate volatile DSP loader paths from portable durable assets."""
    root = Path(root).resolve()
    manifest, catalog = validate(root)
    prepared = Path(prepared).absolute()
    if prepared.parent != Path("/dev/shm") or not prepared.name.startswith("qwen3-arm-dynamic-"):
        raise ValueError("Expected a scoped /dev/shm/qwen3-arm-dynamic-* directory")
    prepared.mkdir(exist_ok=False)
    (prepared / "rfsa").mkdir()
    (prepared / "proofs").mkdir()
    (prepared / "buffer_stream").symlink_to(root / "runtime/buffer_stream")
    (prepared / "rfsa/libhexagon_buffer_runner_skel.so").symlink_to(root / "runtime/libhexagon_buffer_runner_skel.so")
    from hexagon_runtime.split_constant_load_segments import patches, verify, MAX_READ_BYTES
    # Original constant files live in the bundle; bounded ELF views are generated
    # here and independently byte-checked by the unchanged native staging code.
    control = Path(str(prepared) + "-control")
    control.mkdir(exist_ok=False)
    (control / "rfsa").mkdir()
    (control / "buffer_stream").symlink_to(root / "runtime/buffer_stream")
    (control / "rfsa/libhexagon_buffer_runner_skel.so").symlink_to(root / "runtime/libhexagon_buffer_runner_skel.so")
    for index in (1, 2, 3):
        name = f"lib_mlir_ciface_Qwen3DynamicBoundProbe-consts-{index}.so"
        original, candidate = root / "constants" / name, prepared / name
        (control / name).symlink_to(original)
        _, _, _, edits = patches(original, MAX_READ_BYTES)
        shutil.copyfile(original, candidate)
        with candidate.open("r+b") as stream:
            for edit in edits:
                stream.seek(edit["offset"])
                stream.write(bytes.fromhex(edit["replacement_hex"]))
        proof = dict(status="constant_readonly_LOAD_split_byte_preservation_verified_diagnostic",
                     original=dict(path=str(original), sha256=sha(original)),
                     candidate=dict(path=str(candidate), sha256=sha(candidate)),
                     patch=edits, max_load_bytes=MAX_READ_BYTES,
                     checks=verify(original, candidate, edits, MAX_READ_BYTES))
        write(prepared / "proofs" / f"proof-const{index}.json", proof)
    catalog["control_deployment"] = str(control)
    for entry in catalog["entries"]:
        name = entry["library"]["sha256"] + ".so"
        (control / name).symlink_to(root / "libraries" / name)
        entry["library"]["remote_path"] = str(control / name)
    write(prepared / "catalog.json", catalog)
    return manifest, prepared / "catalog.json"
