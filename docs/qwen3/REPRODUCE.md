# Reproducing Qwen3 execution

## Required inputs

1. This application repository, branch `main`, with the pinned `third_party/hexagon-mlir` submodule initialized (see the root README).
2. The matching `qwen3-hexagon-bundle-v1` execution asset directory.
3. QCS9075 with the qualified Linux/FastRPC/DSP environment, ARM64 Python 3.12.3,
   `flock`, `pgrep`, and writable shared memory. A host-only machine can inspect
   the bundle and run host tests; it cannot execute DSP inference.

Use a trusted execution bundle: it contains executable Python/native code. It
is platform-specific. Copying it to an arbitrary OS or a different DSP is not a
supported deployment. No model or SDK binaries are embedded in Git.

The supplied bundle includes exact installed ARM Python packages and their
licenses. The runner inserts its `python/` directory itself; no private
`PYTHONPATH`, editable install, old report, SSHFS mount, or compiler-host daemon
is needed. `.pth` startup hooks and Python caches are excluded when exporting.
The operating system still supplies Python's standard library and system/FastRPC
libraries. A new-board OS/firmware qualification is separate from directory relocation.

## One public entry

```bash
cd /path/to/qwen3-hexagon-runtime
# Optional: prove that no old PYTHONPATH is involved.
unset PYTHONPATH
python3 -m qwen3 doctor --bundle /path/to/bundle --execution
python3 -m qwen3 generate --bundle /path/to/bundle \
  --prompt '你好，请用一句话介绍你自己。' --max-new-tokens 32 \
  --chat-template --output results/chinese
```

`doctor` hashes every asset, verifies the pinned model/constant/runtime identities,
then validates framework versions and all files in the 4,622-entry framework
lock. A mismatch stops execution. Changing a prompt does not compile a new
prompt-specific graph. Runtime input values select a matching captured q1 or
q2–32 graph through the original graph/weight guards; an unknown graph fails.

For a controlled token-ID test, use `--requests-json '[[...], [...]]'` instead of
`--prompt`. For multiple texts use `--prompts-json '["first", "second"]'`.
Requests are handled by native Transformers generation/reset. The capacity bound
is `prompt_tokens + max_new_tokens - 1 <= 1024`. Batch size is one.

Outputs are in the directory passed to `--output`. `result.json` records every
forward, packet hash and request. `execution.json` records the portable bundle,
source digest and AOT cache protocol. `--packet-recording full` additionally saves
complete tensors for independent replay and uses substantially more disk space.
Compiled DSP libraries are already present; first-use Python/AOT capture still
has startup cost. The default AOT cache is `.cache/qwen3`, namespaced by source,
model and framework identities; it can be deleted when no process is using it.

## Produce the asset bundle

An explicit local file mapping is used once by the maintainer:

```bash
python3 -m qwen3 export --spec /path/to/local-assets.json --output /path/to/bundle
python3 -m qwen3 doctor --bundle /path/to/bundle
```

See [the example mapping](../../qwen3/config/asset-map.example.json). Paths may be
absolute or relative to the mapping file. Export copies assets; it does not
compile the model or confer new numerical acceptance. The generated `bundle.json`
contains only relative file names and hashes. Original experiment paths are not
runtime prerequisites. `python_packages` is an ordered list: later directories
overlay earlier ones, allowing the pinned patched Transformers/torchao source to
replace the base environment's packages. Export retains distribution metadata
and licenses. It does not run `.pth` files from that snapshot.

All weight-bearing binaries/checkpoints and platform runtimes remain in the
asset bundle. Preserve their licenses and normal access/distribution constraints
when sharing. SDK/compiler toolchains are separate build dependencies; see
[build status](BUILD_STATUS.md).

## Verify relocation and regression

```bash
# Host Python with the project's test dependencies; no DSP needed.
python -m qwen3 test

# Board: prompts and the trusted tensor-hash reference are included in the repo.
python3 -m qwen3 regress --bundle /path/to/bundle --output results/regression

# Run exactly the reference prompts, max-new-tokens and chunk-size first.
python3 -m qwen3 compare --reference /path/to/reference-result.json \
  --candidate results/relocated/result.json --output results/comparison.json
```

The comparison checks all input/output packet SHA256 values (including logits
and every KV tensor), positions, request boundaries, generated tokens and
ARM/RPC/DSP input checksums. A token-only match is insufficient. It compares a
saved trusted reference; it does not run a fresh ARM reference itself.

Keep outputs and local mappings in `results/`, `.local/` or outside the checkout;
these paths are ignored by Git. Application source, patches, tests and documentation belong here. Compiler
source and DSP kernels belong in the compiler fork. Historical experiments remain
in the preserved compiler-history commits.
