"""Public commands; framework imports stay lazy so inspection works on x86."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import signal
import subprocess
import sys
import uuid

from .bundle import ROOT, export, prepare, sha, validate, write
from .compiler_dependency import activate, compiler_root


def runtime_paths():
    activate()
    sys.path[:0] = [str(ROOT / "qwen3/runtime/arm"), str(ROOT / "tools/qwen3"), str(ROOT)]


def doctor(bundle, *, execution=False):
    compiler_root()
    manifest, _ = validate(bundle)
    result = dict(bundle=str(Path(bundle).resolve()), files_checked=len(manifest["files"]),
                  bundle_valid=True, execution_environment_checked=execution,
                  clean_compiler_rebuild_verified=False)
    if execution:
        if platform.machine() != "aarch64":
            raise ValueError("Generate runs on the ARM board; use doctor without --execution on the host")
        lock = json.loads((ROOT / "qwen3/dependencies.lock.json").read_text())
        packages = Path(bundle).resolve() / "python"
        if packages.is_dir():
            sys.path.insert(0, str(packages))
        for name in ("torch", "torchao", "transformers"):
            if importlib.metadata.version(name) != lock["framework"][name]:
                raise ValueError("Pinned framework version required: " + name)
        from transformers.generation import utils
        if sha(utils.__file__) != lock["framework"]["transformers_generation_sha256"]:
            raise ValueError("Transformers generation source differs; see docs/qwen3/REPRODUCE.md")
        import torch
        if torch.version.git_version != manifest["namespace"]["torch_git"]:
            raise ValueError("PyTorch source revision differs")
        if platform.python_version() != manifest["namespace"]["python_version"]:
            raise ValueError("Pinned ARM Python version required")
        import torchao
        import transformers
        roots = {"torch": Path(torch.__file__).parent, "torchao": Path(torchao.__file__).parent,
                 "transformers": Path(transformers.__file__).parent,
                 "context": Path(bundle).resolve() / "context.json",
                 "decompositions": Path(bundle).resolve() / "policy/fx_decomp_util.py"}
        framework_lock = json.loads((ROOT / "tools/qwen3/board-framework.lock.json").read_text())
        for row in framework_lock["files"]:
            path = roots[row["root"]] / row["relative"] if row["relative"] else roots[row["root"]]
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Framework file differs: " + row["root"] + "/" + row["relative"])
        result["framework_files_checked"] = len(framework_lock["files"])
        for command in ("flock", "pgrep"):
            if shutil.which(command) is None:
                raise ValueError("Missing runtime command: " + command)
    return result


def implementation_digest():
    digest = hashlib.sha256()
    for directory in ("qwen3", "tools/qwen3", "third_party/hexagon-mlir/qcom_hexagon_backend/backend"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def generate(args):
    doctor(args.bundle, execution=True)
    runtime_paths()
    output = args.output.absolute()
    if output.exists():
        raise ValueError("Output must be a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = args.bundle.resolve()
    prepared = Path("/dev/shm/qwen3-arm-dynamic-unified-" + uuid.uuid4().hex)
    workdir = Path(str(prepared) + "-run")
    paths = (workdir, Path(str(prepared) + "-control"), prepared)
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def interrupted(signum, frame):
        raise InterruptedError("Generation interrupted by signal " + str(signum))

    for s in previous_handlers:
        signal.signal(s, interrupted)
    try:
        manifest, catalog = prepare(bundle, prepared)
        native = ROOT / "qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py"
        namespace = dict(manifest["namespace"], runtime_catalog_sha256=sha(bundle / "catalog.json"))
        namespace["native_source_sha256"] = sha(native)
        namespace["adapter_sha256"] = sha(ROOT / "tools/qwen3/native_aot_output.py")
        # A source change invalidates cached adapters even if model libraries stay identical.
        namespace["recipe_sha256"] = hashlib.sha256(
            (namespace["recipe_sha256"] + implementation_digest()).encode()).hexdigest()
        argv = [str(native), "--catalog", str(catalog), "--catalog-sha256", sha(catalog),
                "--context", str(bundle / "context.json"), "--hf", str(bundle / "hf"),
                "--decompositions", str(bundle / "policy/fx_decomp_util.py"),
                "--output", str(output), "--workdir", str(workdir),
                "--constant-load-split-proofs", str(prepared / "proofs"),
                "--constant-loader-mode", "verified-payload-preserving",
                "--packet-recording", args.packet_recording, "--local-packet-transport", "shared-memory",
                "--output-decoder", "owned-once", "--input-packer", "joined",
                "--parameter-check", "exact-bytes", "--specialize-floats", "--generation-timing",
                "--max-new-tokens", str(args.max_new_tokens), "--chunk-size", str(args.chunk_size)]
        if args.requests_json is not None:
            argv += ["--requests-json", args.requests_json]
        else:
            argv += ["--prompts-json", args.prompts_json if args.prompts_json is not None else json.dumps([args.prompt])]
        if args.chat_template:
            argv += ["--chat-template"]
        import torch
        from torch._dynamo.utils import counters
        from native_aot_output import native_cache_scope
        from run_native_aot_generate_arm import require_generation_protocol, require_original_guard_configuration
        from run_integer_regressions import validate_float_policy, validate_parameter_mode
        before = dict(counters["aot_autograd"])
        import torch._functorch.config as aot_config
        require_original_guard_configuration(dict(
            bypass_autograd_cache_key=aot_config.bypass_autograd_cache_key,
            unsafe_skip_cache_dynamic_shape_guards=torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards,
            caching_precompile=torch._dynamo.config.caching_precompile))
        events = []
        old_argv = sys.argv
        try:
            sys.argv = argv
            with native_cache_scope(namespace, args.cache_dir.absolute(), events):
                runpy.run_path(str(native), run_name="__main__")
        finally:
            sys.argv = old_argv
        result = json.loads((output / "result.json").read_text())
        delta = {k: v - before.get(k, 0) for k, v in counters["aot_autograd"].items()}
        protocol = require_generation_protocol(result, events, delta, compiler_mode="original_warm_catalog")
        validate_float_policy(result, True)
        validate_parameter_mode(result, "exact-bytes")
        if result["parameter_validation"]["guarded_python_float_constants"]["ledger_sha256"] != namespace["module_constants_sha256"]:
            raise ValueError("Actual guarded module constants differ from the release namespace")
        write(output / "execution.json", dict(status="completed", bundle_sha256=sha(bundle / "bundle.json"),
              implementation_sha256=implementation_digest(), namespace=namespace, AOT_protocol=protocol,
              DSP_calls=result["DSP_calls"], events=events, compiler_rebuilt=False))
        for request in result["requests"]:
            print(request.get("generated_text", request["generated_ids"]), flush=True)
        print("Results: " + str(output), flush=True)
    finally:
        # All native context managers have exited before removing this session's files.
        for path in paths:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        for s, handler in previous_handlers.items():
            signal.signal(s, handler)


def compare(reference, candidate):
    """Compare all packed logits/KV bytes, request state, tokens and DSP checksums."""
    runtime_paths()
    from run_integer_regressions import compare_integer
    old, new = [json.loads(Path(p).read_text()) for p in (reference, candidate)]
    for result in (old, new):
        if (result.get("process_exit_code") != 0 or result.get("CPU_model_fallback") is not False
                or result.get("native_generate") is not True or not result.get("forwards")):
            raise ValueError("Both runs must contain completed native DSP generation")
    libraries = {tuple(c["contract"]["variables"]["query"]): c["DSO_sha256"] for c in old["captures"]}
    compare_integer(old, new, libraries)
    return dict(status="identical", forwards=new["DSP_calls"],
                compared="all input/output packet hashes (logits + all KV), positions, tokens, FNV",
                reference_sha256=sha(reference), candidate_sha256=sha(candidate))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B on QCS9075 Hexagon")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="Create a portable asset bundle from an explicit file mapping")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("doctor", help="Verify assets and optionally the ARM execution environment")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--execution", action="store_true")
    p = sub.add_parser("generate", help="Run native Transformers generation on the DSP")
    p.add_argument("--bundle", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompts-json")
    group.add_argument("--requests-json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen3"))
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--packet-recording", choices=("hashes", "full"), default="hashes")
    p = sub.add_parser("compare", help="Check a moved installation against a completed reference run")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("regress", help="Execute the included same-input 16-forward reference and compare every output")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen3"))
    sub.add_parser("test", help="Run the maintained host regression suite")
    args = parser.parse_args()
    try:
        if args.command == "export":
            print(export(args.spec, args.output))
        elif args.command == "doctor":
            print(json.dumps(doctor(args.bundle, execution=args.execution), indent=2))
        elif args.command == "compare":
            result = compare(args.reference, args.candidate)
            if args.output:
                write(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "regress":
            config = json.loads((ROOT / "qwen3/config/regression16.json").read_text())
            args.prompts_json = json.dumps(config["prompts"])
            args.requests_json = None
            args.prompt = None
            args.chat_template = False
            args.max_new_tokens = config["max_new_tokens"]
            args.chunk_size = config["chunk_size"]
            args.packet_recording = "hashes"
            generate(args)
            result = compare(ROOT / "qwen3/validation/reference16.json", args.output / "result.json")
            write(args.output / "comparison.json", result)
            print(json.dumps(result, indent=2))
        elif args.command == "test":
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
                       ("qwen3/runtime/arm", "tools/qwen3", ".", "third_party/hexagon-mlir")))
            tests = ["test/python/qwen3", "test/python/torch-mlir/test_qwen_generation_budget.py",
                     "test/python/torch-mlir/test_qwen_query_shape_scope.py",
                     "test/python/torch-mlir/test_arm_dynamic_warm_audit.py",
                     "test/python/torch-mlir/test_hexagon_library_lifecycle.py"]
            tests += ["tools/qwen3/" + name for name in (
                "test_native_aot_output.py", "test_native_aot_generation_protocol.py",
                "test_joined_input_packet.py", "test_exact_parameter_bytes.py",
                "test_constant_load_segments.py")]
            raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, env=env))
        else:
            if not 1 <= args.chunk_size <= 32 or args.max_new_tokens < 1:
                raise ValueError("chunk-size must be 1..32 and max-new-tokens must be positive")
            if args.chat_template and args.requests_json is not None:
                raise ValueError("Chat templates require text input")
            generate(args)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, "qwen3: " + str(error) + "\n")
