# Qwen native extensions

These sources are owned by the Qwen application, including the C++/MLIR code.
They are not part of the general Hexagon-MLIR backend.

- `compiler/Profiles.cmake`: q1/q2 compiler targets, loaded through CMake's
  standard `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` entry.
- `buffer_reuse/`: the selected attention/layout transformations.
- `policy/`: first/sums numerical stages, INT8 matching and SiLU effects.
  Its tests use `qwen-policy-sums-opt`, not the general backend optimizer.
- `runtime/`: model-linked DSP helpers and their include closure.
- `arm_rope_trig_runtime/`: pinned SLEEF preparation and RoPE adaptation.
- `diagnostics/` and `runtime/profile/model_*`: optional profiling support.

Build with `python -m models.qwen3.build` from the application root. General MLIR types,
dialects and lowering support are taken from the explicitly selected compiler
checkout. Build receipts check both that checkout and these native sources.
There is no dependency from the compiler repository back into this directory.

The first/sums stages retain their existing numerical implementations. This
ownership change does not merge algorithms or change their floating-point order.
