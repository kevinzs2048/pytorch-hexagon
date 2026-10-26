# Proposed contribution sequence

The complete downstream work spans framework integration, compiler policies, runtime transport, diagnostics and local environment management. It should not be submitted as one accumulated experimental diff.

| Order | Proposed contribution | Destination | Current state |
| --- | --- | --- | --- |
| 1 | Constant partition size accounting and resource/module preservation | Hexagon-MLIR | Extracted onto current public main in this branch; standalone host pass regression included |
| 2 | Small compiler scalability reproducers and generic fixes | LLVM/MLIR or Hexagon-MLIR, according to ownership | Historical local patches exist; rebase and verify against the agreed baseline before submission |
| 3 | Bounded dynamic graph/tensor contracts and native generation example | Hexagon-MLIR frontend/runtime, with maintainer input | Runs in pinned downstream implementation; dependency extraction and API review required |
| 4 | Explicit numerical policy mechanisms, with positive and rejection tests | Frontend/compiler/math runtime, according to ownership | Downstream model validated; ARM-specific policy must remain opt-in and clearly scoped |
| 5 | Runtime copy/loading optimizations with complete-model evidence | Hexagon-MLIR runtime | Accepted downstream changes and diagnostic-only candidates must be separated |

Before submitting each patch, record its exact base, dependencies, tests and known limits. Generic compiler fixes should have model-independent reproducers. Model integration should have a portable configuration example; local SSH addresses, absolute build paths, generated caches and machine binaries do not belong in the submission.

The initial patch deliberately excludes retention of private assertion strings, newer guard candidates and asynchronous diagnostic hashing. It also does not merge the original dirty development workspace. Their provenance and release status require separate review.

The downstream repository already contains local commits for accepted model/runtime work. Local commits do not mean that upstream has accepted, reviewed or even received that code. No remote push or external PR is implied by this review branch.
