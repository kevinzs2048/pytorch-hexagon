# Qwen DSP helpers and optional diagnostics

`../CMakeLists.txt` builds only production helper objects used by the model link.
The following diagnostic sources are intentionally outside that target:

| Source | Purpose |
| --- | --- |
| `model_memory_profile.c` | Count outer memory calls by site, byte size, time and cycles |
| `model_external_profile.c` | Delegate FP/INT/library calls; exclude nested timer counts |
| `model_integer_timeline.c` | Timestamp the 197 prefill INT8 calls and optional attention boundaries |
| `model_fmaf_effects.c` | Query the imported fmaf implementation, result bits and errno |
| `../../diagnostics/guard_profile.c` | Time the existing Q2 extent guard without timing its inner calls |

The application repository owns relinking, packet parsing and comparisons.
Its `tools/qwen3-review/relink_inference_profile.py` supports external, attention
and guard modes. `tools/qwen3/profile_model_memory.py` also supports the FMA query.
These tools require the original model objects, constants and link receipts;
they first reproduce the uninstrumented library byte for byte. They are not
part of ordinary inference or a second model-build route.

Use the same qualified SDK and source revision as the application lock. Compile
with `hexagon-clang -O2 -Wall -Wextra -Werror -fPIC -fvisibility=hidden
-fno-builtin -mv73 -mhvx -G0` and the SDK `incs` / `incs/stddef` includes.
External prefill instrumentation defines `QWEN_PROFILE_MULTI_INT8`; allocation
counters define `QWEN_PROFILE_ALLOCATIONS`; attention boundary instrumentation
defines `QWEN_TIMELINE_ATTENTION_PHASES`. The application supplies the matching
link wrappers and checks unresolved callees and local counter binding.

These bounded, single-threaded counters target the Qwen3-0.6B ABI. Timer overhead
must be reported separately. The attention intervals include layouts and guards;
they are not individual operator measurements. A diagnostic binary must pass
complete logits/KV comparisons before its timings are used as evidence.
