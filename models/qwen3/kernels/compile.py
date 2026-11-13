# SPDX-License-Identifier: BSD-3-Clause
"""Compile a maintained Triton model kernel to a Hexagon object.

This builds an object, not a deployable model. The caller supplies the DSP
launcher, worker pool, SDK libraries and validated model precision policy.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("strict-decode", "qfloat-decode", "experimental-decode-fp16", "experimental-prefill-fp16", "strict-silu",
                                            "experimental-silu-fp32", "llama-silu-fp32",
                                            "experimental-quantized-silu-up", "strict-int8-pack",
                                            "strict-int8-accumulate", "strict-prefill-pack",
                                            "strict-prefill-qk", "strict-prefill-softmax",
                                            "strict-prefill-pv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libtriton", type=Path, required=True)
    parser.add_argument("--triton-shared-opt", type=Path, required=True)
    parser.add_argument("--linalg-hexagon-opt", type=Path, required=True)
    parser.add_argument("--llc", type=Path,
                        help="Explicit LLVM code generator for quantized SiLU or HMX decode")
    parser.add_argument("--llvm-extract", type=Path, help="HMX decode: extract kernel without a duplicate HMX runtime")
    parser.add_argument("--llvm-opt", type=Path, help="HMX decode: internalize private outlined helpers")
    parser.add_argument("--reuse-hmx-workspace", action="store_true",
                        help="Experimental prefill only: retain owner-thread VTCM scratch")
    parser.add_argument("--prepare-hmx-matmul", action="store_true",
                        help="Experimental prefill only: pack full tiles before HMX dispatch")
    parser.add_argument("--external-allocator", action="store_true",
                        help="Experimental prefill only: use the adapter's temporary-buffer allocator")
    parser.add_argument("--schedule-fma-chains", action="store_true",
                        help="Strict SiLU/decode: schedule exact FP32 FMA chains; decode uses 32-lane softmax blocks")
    parser.add_argument("--lwp", action="store_true",
                        help="Diagnostic LWP instrumentation; requires serialized DSP execution")
    parser.add_argument("--lwp-loop-depth", type=int, default=1)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--backend-source", type=Path,
                        default=root / "third_party/hexagon-mlir")
    args = parser.parse_args()
    if args.lwp_loop_depth < 0 or (not args.lwp and args.lwp_loop_depth != 1):
        parser.error("--lwp-loop-depth requires --lwp and a nonnegative depth")
    hmx_decode = args.profile == "experimental-decode-fp16"
    qfloat_decode = args.profile == "qfloat-decode"
    decode = args.profile in ("strict-decode", "qfloat-decode")
    fast_silu = args.profile == "experimental-silu-fp32"
    llama_silu = args.profile == "llama-silu-fp32"
    silu = args.profile == "strict-silu" or fast_silu or llama_silu
    quantized_silu = args.profile == "experimental-quantized-silu-up"
    int8 = args.profile in ("strict-int8-pack", "strict-int8-accumulate")
    strict_prefill = args.profile.startswith("strict-prefill-")
    prefill = args.profile == "experimental-prefill-fp16"
    if (quantized_silu or hmx_decode) != (args.llc is not None):
        parser.error("--llc is required for experimental-quantized-silu-up or experimental-decode-fp16 only")
    if hmx_decode != (args.llvm_extract is not None) or hmx_decode != (args.llvm_opt is not None):
        parser.error("experimental-decode-fp16 requires --llvm-extract and --llvm-opt")
    hmx = prefill or hmx_decode
    if hmx_decode and not (args.reuse_hmx_workspace and args.prepare_hmx_matmul and args.external_allocator):
        parser.error("HMX decode requires --reuse-hmx-workspace --prepare-hmx-matmul --external-allocator")
    if args.schedule_fma_chains and not (silu or decode or strict_prefill):
        parser.error("--schedule-fma-chains requires strict-silu, strict-decode or strict-prefill")
    if not hmx and args.reuse_hmx_workspace:
        parser.error("--reuse-hmx-workspace requires an experimental HMX profile")
    if not hmx and args.prepare_hmx_matmul:
        parser.error("--prepare-hmx-matmul requires an experimental HMX profile")
    if not hmx and args.external_allocator:
        parser.error("--external-allocator requires an experimental HMX profile")
    source_path = Path(__file__).with_name("strict_prefill.py" if strict_prefill else
                                          "int8_prefill.py" if int8 else
                                          "quantized_silu_up.py" if quantized_silu else
                                          "llama_silu.py" if llama_silu else
                                          "fast_silu.py" if fast_silu else
                                          "silu.py" if silu else "hvx_decode.py" if qfloat_decode else
                                          "strict_decode.py" if decode else "hmx_decode.py" if hmx_decode else "attention.py")
    for name in ("libtriton", "triton_shared_opt", "linalg_hexagon_opt", "backend_source"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tracked = [Path(__file__).resolve(), source_path, args.libtriton,
               args.triton_shared_opt, args.linalg_hexagon_opt]
    if (silu and not llama_silu) or quantized_silu or strict_prefill:
        tracked.append(Path(__file__).with_name("strict_decode.py"))
    if fast_silu:
        tracked += [Path(__file__).with_name(name) for name in
                    ("silu.py", "quantized_silu_up.py")]
    if qfloat_decode:
        tracked += [Path(__file__).with_name(name) for name in ("strict_decode.py", "llama_silu.py")]
    if hmx_decode:
        tracked.append(Path(__file__).with_name("attention.py"))
    for name in ("llc", "llvm_extract", "llvm_opt"):
        path = getattr(args, name)
        if path:
            path = path.resolve(strict=True)
            setattr(args, name, path)
            tracked.append(path)
    identities = {str(path): identity(path) for path in tracked}
    hashes = {str(path): digest(path) for path in tracked}
    commands = []

    def run(command, stage):
        command = [str(item) for item in command]
        commands.append({"stage": stage, "argv": command})
        print(stage, flush=True)
        with (output / (stage + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)

    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.qcom_hexagon_backend.compiler import HexagonBackend
    from triton._C.libtriton import ir

    frontend_extension = Path(sys.modules["triton._C.libtriton"].__file__).resolve()
    if frontend_extension not in tracked:
        tracked.append(frontend_extension)
        identities[str(frontend_extension)] = identity(frontend_extension)
        hashes[str(frontend_extension)] = digest(frontend_extension)

    spec = importlib.util.spec_from_file_location("hexagon_attention_kernel", source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = (getattr(module, "qwen_prefill_" + args.profile.removeprefix("strict-prefill-"))
              if strict_prefill else module.qwen_int8_pack_activation if args.profile == "strict-int8-pack" else
              module.qwen_int8_accumulate_result if int8 else
              module.silu_product if quantized_silu else module.silu_forward if silu else
              module.attention_decode if decode else
              module.attention_forward_q1_hmx_g2 if hmx_decode else module.attention_forward)
    signature = {name: "*fp32" if name in ("Q", "K", "V", "Out", "Scratch", "X", "Y") else "i32"
                 for index, name in enumerate(kernel.arg_names) if index not in kernel.constexprs}
    constants = ({"BLOCK_N": 64, "APPROX_EXP": False, "HEADS_PER_PROGRAM": 4,
                  "SOFTMAX_BLOCK": 32 if args.schedule_fma_chains or qfloat_decode else 8} if decode else
                 {"BLOCK_M": 64, "BLOCK_N": 128, "FP16_DOT": True, "APPROX_EXP": True})
    if hmx_decode:
        constants = {"BLOCK_N": 128}
    if silu:
        constants = {"BLOCK": 32}
    if quantized_silu:
        signature = {"G": "*fp32", "U": "*i32", "S": "*fp32", "W": "*fp32",
                     "Y": "*fp32", "rows": "i32"}
        constants = {"BLOCK": 32, "MODE": 1}
    if int8:
        signature = ({"A": "*i8", "Packed": "*i8", "rows": "i32", "width": "i32"}
                     if args.profile == "strict-int8-pack" else
                     {"Acc": "*i32", "Out": "*i32", "rows": "i32",
                      "columns": "i32", "column": "i32"})
        constants = {}
    if strict_prefill:
        pointers = 6 if args.profile == "strict-prefill-pack" else 5
        signature = {name: "*fp32" if index < pointers else "i32"
                     for index, name in enumerate(kernel.arg_names) if index not in kernel.constexprs}
        constants = {"SOFTMAX_BLOCK": 32}
    target = GPUTarget("hexagon", "v73", 32)
    frontend = HexagonBackend(target)
    features = "+hvxv73,+hvx-length128b" + (",+hvx-qfloat" if llama_silu or qfloat_decode else "")
    options = frontend.parse_options({"arch_features": features})
    context = ir.context()
    ir.load_dialects(context)
    ast = ASTSource(kernel, signature, constants)
    ttir = ast.make_ir(target, options, frontend.get_codegen_implementation(options), {}, context)
    ttir = frontend.make_ttir(ttir, {}, options)
    (output / "kernel.ttir").write_text(str(ttir))
    shared_input = output / "kernel.ttir"
    shared_flags = []
    if llama_silu or qfloat_decode:
        # Generic syntax preserves LLVM/vector operations across triton-shared
        # versions which do not register those dialects. The Hexagon compiler
        # subsequently parses and verifies them as registered operations.
        shared_input = output / "packed.ttir"
        run([args.linalg_hexagon_opt, output / "kernel.ttir", "--lower-libdevice",
             "--canonicalize", "--cse", "--mlir-print-op-generic",
             "-o", shared_input], "lower-packed-hvx")
        shared_flags = ["--allow-unregistered-dialect"]
    run([args.triton_shared_opt, shared_input, *shared_flags, "--triton-to-linalg-experimental",
         "-o", output / "raw.mlir"], "triton-to-linalg")
    passes = []
    if args.prepare_hmx_matmul:
        passes.append("--hexagon-outline-matmul=prepare-hmx-transpose=true")
    passes.append("--hexagon-vectorize-tile-copies")
    if hmx:
        passes.append("--hexagon-optimize-row-reductions")
    if silu or quantized_silu:
        (output / "kernel.mlir").write_bytes((output / "raw.mlir").read_bytes())
    else:
        run([args.linalg_hexagon_opt, output / "raw.mlir", *passes,
             "-o", output / "kernel.mlir"], "prepare")
    overrides = {
        "arch_features": features,
        "enableConvertToHexagonmem": False,
        "enableSplitReduceGeneric": True,
        "enableVTCMTiling": False,
        "enableHexKL": hmx,
        "outlineMatmul": hmx,
        "dispatchHMX": hmx,
        "reuseHMXWorkspace": args.reuse_hmx_workspace,
        "prepareHMXMatmul": args.prepare_hmx_matmul,
        "externalKernelAllocator": args.external_allocator,
        "scheduleFMAChains": args.schedule_fma_chains or quantized_silu or strict_prefill,
        # The 128-byte SiLU temporary costs 191 bytes including alignment.
        # Existing bounded stack promotion lets LLVM eliminate its allocation
        # and per-vector copy; attention keeps its current allocation policy.
        "stackBufferBudget": 4096 if quantized_silu or int8 or silu or qfloat_decode else 0,
        "enableLWP": args.lwp,
        "LWPloopDepth": args.lwp_loop_depth,
        "addFastMath": False,
        "enableConversionToFp16": False,
    }
    emitted = output / ("kernel.ll" if args.llc else "kernel.o")
    command = [sys.executable, root / "models/qwen3/runtime/arm/compile_qwen3_artifact.py",
               "--input", output / "kernel.mlir", "--output", emitted,
               "--libtriton", args.libtriton, "--backend-source", args.backend_source]
    if args.llc:
        command += ["--emit", "llvmir"]
    for name, value in overrides.items():
        command += ["--option", f"{name}={value}"]
    if args.lwp:
        # The upstream pass writes a process-global pathname. Serialize builds
        # through this entry and preserve the previous dump for other workflows.
        import fcntl
        mapping = Path("/tmp/lwp_infodump.txt")
        with open("/tmp/hexagon-mlir-lwp.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = mapping.read_bytes() if mapping.exists() else None
            try:
                mapping.unlink(missing_ok=True)
                run(command, "compile")
                from hexagon_runtime.lwp import read_map
                read_map(mapping)  # Reject duplicate IDs before exporting.
                (output / "lwp-map.txt").write_bytes(mapping.read_bytes())
            finally:
                if previous is None:
                    mapping.unlink(missing_ok=True)
                else:
                    mapping.write_bytes(previous)
    else:
        run(command, "compile")
    if hmx_decode:
        # Prefill supplies one HMX owner, VTCM pool and allocation runtime for
        # the whole model. Export only this entry; helpers remain internal.
        run([args.llvm_extract, "--func=attention_forward_q1_hmx_g2",
             "--rfunc=^__hexagon_", "--keep-const-init", emitted,
             "-o", output / "selected.bc"], "extract-kernel")
        emitted = output / "kernel.bc"
        run([args.llvm_opt, "-passes=internalize,globaldce",
             "--internalize-public-api-list=attention_forward_q1_hmx_g2",
             output / "selected.bc", "-o", emitted], "internalize-kernel")
    if args.llc:
        run([args.llc, "-O3", "-hexagon-insert=false", "-verify-machineinstrs",
             "-filetype=obj", "-relocation-model=pic", "-code-model=small",
             emitted, "-o", output / "kernel.o"], "codegen")
    for path in tracked:
        if identity(path) != identities[str(path)]:
            raise RuntimeError(f"Build input changed during compilation: {path}")
    manifest = {
        "profile": args.profile,
        "reuse_hmx_workspace": args.reuse_hmx_workspace,
        "external_allocator": args.external_allocator,
        "prepare_hmx_matmul": args.prepare_hmx_matmul,
        "schedule_fma_chains": overrides["scheduleFMAChains"],
        "stack_buffer_budget_bytes": overrides["stackBufferBudget"],
        "lwp": {"enabled": args.lwp, "loop_depth": args.lwp_loop_depth,
                "requires_serial_execution": args.lwp},
        "status": "object-built; device/model validation is separate",
        "constants": constants,
        "grid": [1, 4 if decode else 8],
        "scratch_fp32_elements": module.SCRATCH_FP32_ELEMENTS if decode else 0,
        "contract": {"batch": 1, "query": [1, 1] if decode or hmx_decode else [2, 32],
                     "query_heads": 16, "kv_heads": 8, "head_dim": 128,
                     "kv_length_max": 1024, "storage": "fp32",
                     "mask": "causal, no padding, final consecutive query positions"},
        "source_and_tools_sha256": hashes,
        "commands": commands,
        "object_sha256": digest(output / "kernel.o"),
        "precision": ("ordered FP32; accepted finite-input model policy" if decode else
                      "experimental FP16 dot operands/results and probabilities; FP32 online state; "
                      "pairwise sums, approximate exp, reciprocal normalization; NOT model accepted"),
    }
    if hmx and args.prepare_hmx_matmul:
        manifest["prepared_hmx_slot_bytes"] = 192 * 1024
    if hmx_decode:
        manifest.update(entry_point="attention_forward_q1_hmx_g2",
                        grid=[1, 4], kv_heads_per_program=2,
                        shared_runtime="experimental-prefill-fp16",
                        raw_buffer_access="dsp-load-store",
                        runtime_query_tiles=[{"query": [1, 1],
                                              "physical_hmx_rows": 32,
                                              "live_softmax_rows": 4}])
    if qfloat_decode:
        manifest["precision"] = (
            "FP32 Q/K/V and output; HVX QFloat QK/PV with tree QK reduction; "
            "SLEEF-compatible softmax exponent, strict denominator/division; "
            "not bitwise strict decode; device/model validation is separate")
    if silu:
        manifest.update(grid=[4], scratch_fp32_elements=0,
                        contract={"batch": 1, "rows": [1, 32], "columns": 3072,
                                  "input_row_stride_min": 3072, "output": "contiguous FP32"},
                        precision="SLEEF-compatible FP32 exponent/FMA order and strict division; device/model validation required")
        if fast_silu:
            manifest.update(
                precision="bounded FP32 approximation, strict exceptional vectors; not bitwise reference",
                approximation={"domain": "x == 0 or 2^-64 <= abs(x) <= 16",
                               "reciprocal_refinements": 3,
                               "input_output_aliasing": "forbidden; runtime rejects overlap",
                               "ordinary_error_check": "abs(error)/max(1,abs(reference)) <= 2e-6",
                               "fallback": "whole vector uses strict SiLU outside the domain"})
    if llama_silu:
        manifest.update(
            precision="llama.cpp Hexagon SDK QFloat SiLU; not strict PyTorch bitwise reference",
            approximation={
                "reference_revision": "f818065d753700341b5355ce36ca676ecfb5d139",
                "reference": "htp/hvx-sigmoid.h and hvx-inverse.h, SDK QFloat conversion folds",
                "sigmoid_guards": "x <= -87 -> 0; x >= 87 -> 1; unordered comparisons -> 0",
                "reciprocal_refinements": 3,
                "strict_rounding_repair": False,
                "input_output_aliasing": "forbidden; runtime rejects overlap"})
    if quantized_silu:
        manifest.update(grid=[4], scratch_fp32_elements=0,
                        contract={"batch": 1, "rows": [4, 32], "columns": 3072,
                                  "inputs": "contiguous G/S/Y FP32[rows,3072], U INT32[rows,3072], W FP32[3072]",
                                  "up_order": "(S * float(U)) * W",
                                  "allowed_consumers": "symmetric row INT8 quantizer and its scale only",
                                  "scale": "max(max(abs(Y))/127, float32(1e-5))",
                                  "quantization": "roundeven(Y * (1/scale)), clamp [-127,127]",
                                  "small_queries": "retain the existing strict graph for rows below 4"},
                        precision="experimental approximate FP32 product with strict row/max/boundary repair; independent numerical review and model acceptance pending",
                        llvm_codegen={"path": str(args.llc), "sha256": hashes[str(args.llc)],
                                      "required_fix": "HEX-PRED-001; device verification is separate"})
    if int8:
        manifest.update(grid=[1], scratch_fp32_elements=0,
                        contract={"query_rows": [1, 32], "reduction_widths": [1024, 2048, 3072],
                                  "output_columns": [128, 3072], "column_multiple": 128,
                                  "activation": "signed INT8, contiguous",
                                  "packed_activation": "u8[width/32,64,32]; padded rows are 128",
                                  "accumulator": "i32[64,32]; row32 is the zero-point bias",
                                  "output": "C += signed A @ W, contiguous INT32"},
                        precision="exact INT8/INT32; no floating-point conversion; model validation is separate")
    if strict_prefill:
        stage = args.profile.removeprefix("strict-prefill-")
        manifest.update(grid=[1, 4],
                        runtime_worker_counts=[4, 8] if stage in ("qk", "pv") else [4],
                        scratch_fp32_elements=module.SCRATCH_FP32_ELEMENTS,
                        contract={"batch": 1, "query": [16, 32], "kv_length": [32, 256],
                                  "minimum_query_key_pairs": 1024,
                                  "query_heads": 16, "kv_heads": 8, "head_dim": 128,
                                  "storage": "fp32", "input_absolute_bound": "strictly below 2^40",
                                  "mask": "causal, no padding, final consecutive query positions",
                                  "fallback": "strict reference for unsupported shape or numeric bound"},
                        precision="ordered FP32; guarded causal pruning; unchanged per-result FMA order")
    if prefill:
        manifest["raw_buffer_access"] = (
            "dsp-load-store" if not overrides["enableConvertToHexagonmem"] else "dma")
        manifest["runtime_query_tiles"] = [
            {"query": [2, 16], "rows_including_two_heads": 32},
            {"query": [17, 32], "rows_including_two_heads": 64},
        ]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "kernel.o", flush=True)


if __name__ == "__main__":
    main()
