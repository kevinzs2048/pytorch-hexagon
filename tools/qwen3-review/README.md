# Downstream inference diagnostics

These tools inspect an already accepted downstream Qwen3 model. They require its pinned clean source, recorded machine-cache receipts, original constants and executable libraries. They are not a model quick start for this public review branch.

`relink_inference_profile.py` first reproduces the exact original model DSO byte for byte. It then links unchanged model primary/runtime objects with diagnostic wrappers. It supports `--mode external`, `--mode attention` and `--mode guard`; the latter two require the bounded dynamic prefill model. It does not compile the model graph or claim to restore the entire historical compiler build.

```bash
python tools/qwen3-review/relink_inference_profile.py \
  --source /absolute/pinned-downstream-source --revision FULL_COMMIT \
  --machine /absolute/accepted-machine-cache-entry \
  --config /absolute/local-config.json \
  --mode attention --output /absolute/fresh-diagnostic-build
```

Execution uses the downstream `probe_model_memory_arm.py` and recorded `run_staged_probe.py` plan. The normal board probe checks original ARM-qualified full input/output identities, all K/V tensors, unchanged constants, and actual runtime exits. Diagnostic binaries must pass those checks before their counters are interpreted.

`analyze_inference_gap.py` verifies completed native runs against their accepted oracle and independently decodes raw profile packets. Its steady decode partition subtracts nested clocks before averaging, excludes the first decode transition in each process, and keeps local validation/SHA work separate from framework scheduling and DSP execution. Every native prompt used for this analysis must have a multi-token prefill, followed by one forward for each inter-token interval.

```bash
python tools/qwen3-review/analyze_inference_gap.py \
  --runs /absolute/warm-run1 /absolute/warm-run2 \
  --profiles /absolute/q1-external-run /absolute/q2-external-run \
    /absolute/q2-attention-run /absolute/q2-guard-run \
  --oracle /absolute/CPU-qualified-warm-oracle \
  --downstream /absolute/pinned-downstream-source \
  --output /absolute/fresh-analysis.json
```

Measured instrumentation overhead is reported separately. Attention intervals include arithmetic, layouts and guards; DSP execution time alone does not establish an upstream compiler defect. See [the support-case scope](../../docs/qwen3/PERFORMANCE_GAPS.md).
