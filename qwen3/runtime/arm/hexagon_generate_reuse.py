"""Strict reuse checks for the bounded Qwen generation diagnostic runner.

Model code and weights are not recompiled or relinked. The SDK wrapper embeds
remote filenames, so a reused artifact requires its original remote namespace.
Previous LOCAL per-call evidence must remain untouched; the runner uses a new
local output directory and serializes board operations with the shared lock.
"""
import hashlib
import errno
import json
import os
from pathlib import Path
import re
import shutil
import stat


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_library(source, target, *, expected_sha256, expected_bytes):
    """Stage immutable bytes beside NEW execution evidence, never overwrite.

    Prefer a hardlink; only EXDEV permits a verified copy. A symlink is not
    sufficient: the executor resolves paths before choosing its manifest folder.
    The caller must not edit staged libraries, including on the hardlink path.
    """
    source, target = Path(source), Path(target)
    if not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("Compiled library must be a regular file")
    created = False
    try:
        try:
            os.link(source, target)
            created = True
            method = "hardlink"
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # Exclusive creation also rejects pre-existing and dangling symlinks.
            with target.open("xb") as output:
                created = True
                with source.open("rb") as input_stream:
                    shutil.copyfileobj(input_stream, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            method = "verified-copy"
        if target.stat().st_size != expected_bytes or sha(target) != expected_sha256:
            raise ValueError("Staged compiled library bytes drifted")
        return method
    except BaseException:
        if created:
            target.unlink()
        raise


def historical_parameter_digest(record):
    """Same named-parameter SHA formula, in legacy and generation records.

Never substitute parameters_before_quantization_sha256: that field hashes
state_dict with a DIFFERENT algorithm in the generation runner.
"""
    values = [record[key] for key in ("model_parameter_sha256_before_quantization",
                                     "historical_parameter_sha256") if key in record]
    if (not values or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values)
            or len(set(values)) != 1):
        raise ValueError("Missing, invalid or conflicting historical named-parameter digest")
    return values[0]


def validate_reuse_record(previous, current, *, require_compiled_graphs=True):
    # Completed compilation may be reused after a failed numerical run; this
    # never upgrades that run's outcome. Partial graphs are rejected below.
    # Configs such as id2label use integer dictionary keys in memory, while
    # JSON restores string keys. Compare their lossless persisted form.
    current = json.loads(json.dumps(current))
    for key in ("capacity", "precision", "prefill_policy", "full_checkpoint", "versions",
                "model_config", "parameters_before_quantization_sha256"):
        if previous.get(key) != current.get(key):
            raise ValueError("Compiled reuse context differs: " + key)
    if require_compiled_graphs:
        if previous.get("options") != current.get("options"):
            raise ValueError("Compiled reuse context differs: options")
        for key in ("compiler", "linker"):
            if previous[key]["sha256"] != current[key]["sha256"]:
                raise ValueError("Compiled reuse toolchain differs: " + key)
    # Exported Linalg is the INPUT to Hexagon compilation. Graph-only reuse
    # deliberately recompiles and relinks it, so compiler/options may differ.
    # Forward sources, weights, versions and exported graph bytes remain bound.
    required_sources = ("qwen3_public_static_forward.py",)
    if current["full_checkpoint"]:
        required_sources += ("fp32_silu_decomposition.py",)
        if previous.get("historical_parameter_sha256") != current.get("historical_parameter_sha256"):
            raise ValueError("Full model parameter identity differs")
        if previous["checkpoint"]["sha256"] != current["checkpoint"]["sha256"]:
            raise ValueError("Full checkpoint bytes differ")
    for name in required_sources:
        old = [p["sha256"] for p in previous["sources"] if Path(p["path"]).name == name]
        new = [p["sha256"] for p in current["sources"] if Path(p["path"]).name == name]
        if len(old) != 1 or old != new:
            raise ValueError("Compiled model source differs: " + name)
    for policy in ("fp32_sum_policy", "fp32_attention_dot_policy", "fp32_context_dot_policy", "fp32_softmax_policy"):
        old_policy = previous.get(policy, "none")
        new_policy = current.get(policy, "none")
        if require_compiled_graphs and old_policy != new_policy:
            raise ValueError("Compiled numerical policy differs: " + policy)
        if not require_compiled_graphs and old_policy not in ("none", new_policy):
            raise ValueError("Cannot undo a numerical pass by reusing its transformed graph: " + policy)
        if not require_compiled_graphs and old_policy != "none" and not previous.get("graphs"):
            raise ValueError("Requested numerical policy has no bound completed graph: " + policy)
    if require_compiled_graphs and previous.get("scalar_exp_policy", "sdk") != current.get("scalar_exp_policy", "sdk"):
        raise ValueError("Compiled scalar exponential binding differs")
    if previous.get('rotary_policy', 'sdk') != current.get('rotary_policy', 'sdk'):
        raise ValueError('Compiled rotary policy differs; use the explicit bounded candidate builder')
    if require_compiled_graphs:
        expected_counts = {1} if current["prefill_policy"] == "tokenwise" else {1, 4}
        actual_counts = [g["tokens_per_call"] for g in previous["graphs"]]
        if len(actual_counts) != len(expected_counts) or set(actual_counts) != expected_counts:
            raise ValueError("Missing or duplicate compiled graphs")


def inherited_scalar_exp_binding(previous, control_path):
    """Preserve actual link provenance when reusing, rather than relinking, a DSO."""
    policy = previous.get('scalar_exp_policy', 'sdk')
    if policy not in ('sdk', 'sleef', 'softmax-only-sleef'):
        raise ValueError('Unknown scalar exponential binding')
    if policy == 'sdk':
        if control_path is not None:
            raise ValueError('SDK reuse must not supply an exponential control')
        return None
    binding = previous.get('scalar_exp_binding')
    if not binding or control_path is None or sha(control_path) != binding['control_record_sha256']:
        raise ValueError('Reused SLEEF binding lacks the original sealed control')
    if len(previous['graphs']) != 1 or previous['graphs'][0]['libraries'][0]['sha256'] != binding['linked_library_sha256']:
        raise ValueError('Reused SLEEF binding does not identify the compiled main library')
    if policy == 'softmax-only-sleef':
        if (binding.get('expf_interposed') is not False
                or binding.get('explicit_symbol') != '__hexagon_softmax_exp_compat'
                or set(binding.get('defined_runtime_symbols', [])) != {
                    '__hexagon_softmax_exp_compat', '_mlir_ciface___hexagon_softmax_exp_compat'}
                or not binding.get('link_command')
                or any('--wrap' in word for word in binding['link_command'])):
            raise ValueError('Selective Softmax reuse lacks the no-interposition ABI binding')
    return json.loads(json.dumps(binding))


def inherited_rotary_binding(previous):
    """Do not let a finite native-table specialization masquerade as SDK RoPE."""
    policy = previous.get('rotary_policy', 'sdk')
    if policy == 'sdk':
        if previous.get('native_rotary_table_binding') is not None:
            raise ValueError('SDK rotary policy has an unexpected table binding')
        return None
    capacities = {f'native-table-capacity{n}':n for n in (24,32,64)}
    if policy not in capacities: raise ValueError('Unknown rotary policy')
    capacity = capacities[policy]
    binding = previous.get('native_rotary_table_binding')
    if (previous.get('capacity') != capacity or not binding or len(previous.get('graphs', [])) != 1
            or binding.get('output_graph_sha256') != previous['graphs'][0]['graph_sha256']
            or binding.get('capacity') != capacity or binding.get('positions_independently_verified') != capacity
            or binding.get('DSP_outputs_bitexact') != 2*capacity or binding.get('table_bytes') != 2*capacity*128*4
            or binding.get('runtime_CPU_inference') is not False
            or binding.get('original_CPU_reference_unchanged') is not True
            or binding.get('forward_ABI_unchanged') is not True
            or binding.get('cache_and_generation_unchanged') is not True
            or not binding.get('control_path')
            or sha(binding['control_path']) != binding.get('control_sha256')):
        raise ValueError('Native rotary table reuse lacks its verified finite-capacity binding')
    return json.loads(json.dumps(binding))


def reuse_forward(previous, *, count, inputs, expected, directory, remote, graph_archive=None):
    rotary_binding = inherited_rotary_binding(previous)
    if rotary_binding is not None and (count != 1 or inputs[1].numel() != 1
            or not 0 <= int(inputs[1].item()) < rotary_binding['capacity']):
        raise ValueError('Native rotary table input is outside its verified capacity')
    from triton._C.libtriton import ir, qcom_hexagon_backend as backend
    from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMlirHexagonWrapperGenerator
    from triton.backends.qcom_hexagon_backend.utils import parse_return_types, profile_torch_mlir_inputs
    info = next(g for g in previous["graphs"] if g["tokens_per_call"] == count)
    libraries = [str(Path(lib["path"]).resolve()) for lib in info["libraries"]]
    for path, lib in zip(libraries, info["libraries"]):
        if sha(path) != lib["sha256"] or Path(path).stat().st_size != lib["bytes"]:
            raise ValueError("Compiled library bytes drifted")
    name = f"Qwen3{'Full' if previous['full_checkpoint'] else 'Tiny'}Static{count}"
    entry = "_mlir_ciface_" + name
    origin = Path(libraries[0]).parent
    graph = Path(info.get("graph_path", origin / "graph.mlirbc"))
    context = ir.context(); context.disable_multithreading(); backend.load_dialects(context)
    from hexagon_archived_graph import verified_graph_file
    with verified_graph_file(graph, info['graph_sha256'], graph_archive) as parse_path:
        module = backend.parse_mlir_module_from_file(parse_path, context)
    return_profiles = parse_return_types(backend.get_return_list(module, name))
    if len(return_profiles) != len(expected) or any(p.dtype != "float" or p.rank != t.ndim
                                                  for p, t in zip(return_profiles, expected)):
        raise ValueError("Reused output ABI differs from the expected FP32 tensors")
    ownership = info.get('returned_buffer_policy')
    if ownership != previous.get('returned_buffer_policy'):
        raise ValueError('Graph and run disagree on returned-buffer ownership')
    owner_options = {}
    if ownership is not None:
        from hexagon_wrapper_relink import verify_owned_wrapper_relink
        verify_owned_wrapper_relink(previous)
        owner_options['release_returned_buffers'] = True
    wrapper = TorchMlirHexagonWrapperGenerator(profile_torch_mlir_inputs(list(inputs)), 1,
                                               entry, return_profiles, previous["options"], **owner_options)
    original_wrapper = Path(info.get("original_wrapper_path", origin / f"{entry}_wrapper.cpp"))
    # Exact comparison covers types, input/output filenames, ranks, benchmark
    # iteration count and the wrapper's embedded remote execution directory.
    generated = wrapper.generate_cpp_wrapper(entry, remote)
    if previous.get('buffer_interface') is not None:
        from hexagon_buffer_binding import buffer_entry_suffix
        generated += buffer_entry_suffix(previous,wrapper)
    if original_wrapper.read_text() != generated:
        raise ValueError("Reused wrapper differs; use the ORIGINAL --remote directory and ABI")
    # Executor writes its execution manifest beside the FIRST library, not
    # beside the input tensors. Stage immutable libraries into the fresh run
    # directory using hardlinks, or a verified copy across filesystems.
    # No operation here edits/truncates existing libraries.
    staged = []
    staging = []
    for path, lib in zip(libraries, info["libraries"]):
        target = Path(directory) / Path(path).name
        method = stage_library(path, target, expected_sha256=lib["sha256"], expected_bytes=lib["bytes"])
        staged.append(str(target.resolve()))
        staging.append(dict(source=path, target=str(target.resolve()), method=method))
    metadata = dict(info, reused=True, graph_path=str(graph),
                    original_wrapper_path=str(original_wrapper),
                    original_wrapper_sha256=sha(original_wrapper), compile_link_seconds=0,
                    libraries=[dict(lib, path=path) for lib, path in zip(info["libraries"], staged)])
    metadata['library_staging'] = staging
    if graph_archive is not None:
        metadata['verified_graph_archive'] = dict(path=str(graph_archive), sha256=sha(graph_archive))
    return wrapper, staged, metadata
