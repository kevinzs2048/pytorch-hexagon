# Requested Hexagon-MLIR support for full-model Qwen3 inference

We would like to collaborate on a publicly reproducible Qwen3 path that retains PyTorch/Transformers generation semantics and runs complete model forward passes on QCS9075 Hexagon DSP.

The immediate engineering priorities are measured attention execution and KV ownership/buffer reuse. See [the runtime measurements and specific assistance requested](PERFORMANCE_GAPS.md). Version alignment below supports that work.

## 1. A supported public source and dependency baseline

Please identify the recommended public commit or release and the tested combination of LLVM, Triton, triton-shared, torch-mlir, PyTorch, torchao, Hexagon SDK, Hexagon Tools and HexKL. Please also confirm QCS9075 target/runtime support and required board-side components. If applicable support or fixes exist only in a newer internal tree, could the relevant source, build instructions and minimal tests be published?

We have verified that public main has already advanced to `9ec15e80d9d53aa7fb97654da00bfd7047a77ead`. Our request is to align on a reproducible full-model baseline; it is not a claim that the public project has not been updated. A version table and a clean-build check would help more than an unqualified request for the latest binaries.

## 2. Bounded dynamic input and native generation

We need runtime token IDs, positions, masks and KV values, with reusable query=1 decoding and query=2–32 chunked prefill. Changing prompt contents or supported query lengths should not require one program per prompt. Please advise on the intended public API for AOT/Dynamo graph preparation, dynamic-shape guards, compiled artifact reuse and persistent execution sessions.

Acceptance should include changing inputs without new model compilation, native EOS and request reset, and correct capacity rejection. Support for arbitrary batch sizes or arbitrary model architectures is outside the initial request.

## 3. Explicit numerical control

Our reference and DSP both use W8A8 linear layers. Differences in FP32 reduction order, FMA contraction and RoPE/transcendental rounding can be amplified at requantization boundaries. We need documented controls for these transformations and a way to apply explicit numerical policies without silently enabling incompatible fast-math transformations.

We can provide same-input operator and model reproducers. We are not asking that a platform-specific OpenBLAS order become the compiler's universal default. We would appreciate guidance on which policies belong in the frontend, Hexagon-MLIR, math runtime or LLVM, and what accuracy contract the supported backend promises.

## 4. Large-model compilation, loading and runtime

We need reliable constant splitting, preservation of constant payloads, scalable bufferization/code generation and a documented loading strategy for large model libraries. The initial patch in this branch addresses independently reproducible partitioning errors.

We would also like guidance or public implementation support for persistent model constants and DSP-resident KV, to reduce repeated transfers. These are requested capabilities, not completed features of our current deployment. Historical loader failures remain open; we can provide a minimal failure record rather than asserting an unproven compiler or driver root cause.

## 5. A maintained full-model example and contribution route

Could we agree on a small Qwen3 example or supported reference model, with end-to-end compilation, generation, same-input numerical checks and reproducible dependency pins?

The current [contribution policy](https://github.com/qualcomm/hexagon-mlir/blob/9ec15e80d9d53aa7fb97654da00bfd7047a77ead/CONTRIBUTING.md) says external PRs are not yet accepted. Please confirm whether patches by email, a review branch or external PRs are now preferred, and who should review compiler, runtime and LLVM-related changes. We can send small patches and minimal reproducers in that order.
