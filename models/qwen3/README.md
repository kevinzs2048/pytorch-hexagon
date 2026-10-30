# Qwen3 adapter

This directory contains the Qwen3-0.6B adapter for PyTorch Hexagon. Its model
capture, generation, KV lifecycle and numerical rules are not generic model APIs.

- `cli.py`, `bundle.py`: Qwen commands and validation of its execution assets.
- `runtime/arm/`: model capture, dynamic shape policy, model binding and generation.
- `runtime/support/`: Qwen scalar/parameter guards and AOT serialization.
- `native/`: Qwen compiler passes, numerical tools and model-linked DSP helpers.
- `build.py`, `build_model.py`: build this adapter and its model libraries.
- `config/`, `dependencies.lock.json`: Qwen configuration and execution contracts.
- `validation/`: historical evidence and the included 16-forward reference.
- `runtime-sources.json`: execution-source inventory used for AOT cache invalidation.

Shared transport, packet packing, artifact caching and timing live in
`hexagon_runtime/`. The adapter supplies the staged-library namespace required by
its existing ARM runner. It retains the same numerical policies and model ABI.

Commands from the repository root:

```bash
python -m hexagon_runtime qwen3 --help
python -m models.qwen3.build --help
python -m models.qwen3.tools verify --help
```

Current support: QCS9075, full 28-layer Qwen3-0.6B, batch 1, W8A8 linear layers,
FP32 intermediates and KV, capacity 1024. q1 decoding and q2–32 prefill reuse
compiled libraries. No CPU model-forward fallback is supplied.

See the [Chinese guide](../../USAGE.zh-CN.md),
[source-build instructions](../../docs/qwen3/SOURCE_BUILD.md) and
[qualification history](../../docs/qwen3/BUILD_STATUS.md). Source moves invalidate
old AOT caches; existing compiled model bundles retain their original identities.
