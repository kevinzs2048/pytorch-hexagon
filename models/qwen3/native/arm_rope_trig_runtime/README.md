# ARM-compatible RoPE math dependency

The Qwen compatibility path uses SLEEF 3.8's FP32 sin/cos implementation.
Third-party source is fetched into the build directory, not copied into this
repository. `Sleef.cmake` fixes the download URL and SHA-256, verifies each of
the seven selected source files, and applies `helper-adaptation.patch` only to
a build-local copy. The adapted helper is also checked by SHA-256.

Upstream: https://github.com/shibatch/sleef/releases/tag/3.8

Archive: https://codeload.github.com/shibatch/sleef/tar.gz/refs/tags/3.8

Archive SHA-256: `a12ccd50f57083c530e1c76f10d52865defbd19fc9e2c85b483493065709874a`

The Boost Software License is retained in `LICENSE.txt` and copied alongside
the prepared sources. Upstream source copyright notices are preserved.

## Build and offline use

The runtime CMake build prepares this dependency automatically. For offline
builds, unpack the pinned archive and configure with
`-DQWEN_SLEEF_SOURCE_DIR=/path/to/sleef-3.8`. The environment variable of the
same name supplies the default for a new build directory, including builds
launched through `python -m models.qwen3.build`. This path must contain the
unmodified upstream source; it is validated and never patched in place.
No network access occurs when this option is supplied. CMake and the `patch`
utility are required. Only the selected runtime translation units are built;
upstream SLEEF's own CMake project is not configured.

`trig_lane.c` exports only the four-lane sin/cos entry points used by the compiler
rewrite. The adaptation retains grouped range reduction, ties-to-even
rounding and explicit fused operations; compile with `-ffp-contract=off`.
The SLEEF source and numerical algorithm are unchanged from the previous
vendored version. This dependency arrangement does not itself qualify full-model output.

The selected model policy admits FP32 `[1, q, 128]` angles for q=1..32 and
requires explicit declarations and linkage. Model/ARM numerical validation
remains in the application repository.

To test offline preparation and input-integrity checks:

```sh
QWEN_SLEEF_SOURCE_DIR=/path/to/sleef-3.8 python -m pytest -q \
  test/python/torch-mlir/test_sleef_dependency.py
```
