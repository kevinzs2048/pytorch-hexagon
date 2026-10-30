> Historical record. The combined-repository organization below was superseded
> by [the repository split](REPOSITORY_SPLIT.md).

# Repository consolidation — 2026-10-26

## Canonical project

The existing `hexagon-mlir-qwen3` repository, branch `models/qwen3/unified`, is the single
source entry. We did not create another application repository. Compiler source,
framework/model/KV integration, policies, tests, patches and instructions are
versioned together. Large execution assets are exported to a relocatable bundle.

The previous working tree had 1,940 untracked files and modifications to 77
tracked files. It is preserved in archive commit
`4c174dce80553f6cd781222c73817271edfa0370` on
`archive/qwen3-before-consolidation-20261026`. This snapshot precedes any removal
or source relocation. Do not treat its experimental entry points as current APIs.

The maintained tree imports the required runtime/build support files from:

- Runtime/policies: `feb45fd85274d395117f6ba75dfd33433256bba0`.
- Native AOT adapter: `7ec99369749f52ffbe0ed3b81df51ca407cdfb5b`.
- Existing local compiler: preserved in full by the archive snapshot; not
  overwritten by the accepted runtime checkout's incomplete compiler snapshot.
- Compiler-team diagnostics/docs: upstream-review branch, collected here.
- Retained LLVM local changes: one patch against `064f02d`, including new tests.

`models/qwen3/source-origins.json` records the original paths and byte hashes of the
imported sources. New public packaging code is ordinary Git-tracked source.
Numerical code, the native generation entry, and the two model DSOs are retained;
the integration refactor changes how assets/dependencies are found and staged.

## Removed runtime dependencies

The public CLI no longer traverses historical release-index/state304/ABBA report
chains to discover its executable paths. Those records remain historical evidence.
A bundle contains the full captured graph/ABI catalog, pinned model/runtime hashes,
checkpoint/tokenizer and decomposition policy. Original graph/weight/float guards,
full KV transport and native Transformers generation remain active.

A clean environment check exposed another dependency: the old virtualenv did not
contain the active torchao and patched Transformers packages. They came from a
separate `PYTHONPATH` directory. Export now captures exact installed package files,
metadata and licenses in the bundle; execution verifies the existing framework
lock. No old `PYTHONPATH` is required.

## Validation

- `python -m qwen3 test`: 107 host tests passed, including two-process AOT cache,
  parameter byte guards, packing, loader ELF-byte preservation and bundle failure
  cases. Framework deprecation warnings remain; they are not test failures.
- Bundle environment check on ARM: 19,793 asset files and 4,622 framework/context
  files verified, using `/usr/bin/python3` and an unset `PYTHONPATH`.
- Device: two migration runs (32 forwards) and the final public `regress` entry
  (16 forwards) matched every reference input/output hash, position, token and
  ARM/RPC/DSP input checksum. The final source digest was checked against the
  executed board source. See [the validation record](../../models/qwen3/validation/consolidation.json).
- The copied host bundle also passed verification: 19,793 files, 3,502,496,726
  bytes. It includes the ARM package snapshot and needs no old PYTHONPATH.

This validation qualifies repository/asset relocation, not a new compiler build,
new numerical algorithm or performance claim. See [build status](BUILD_STATUS.md)
for the remaining source-build work.

## Rollback and future work

Inspect the archive without overwriting current work:

```bash
git show archive/qwen3-before-consolidation-20261026:path/to/file
```

For a full historical checkout, use a separate recovery directory. The frozen
runtime/deployment worktrees and previously accepted binaries were not modified.
Do not revert by deleting or moving their artifacts.

Keep new executable code in this repository. Add user-facing operations to
`python -m qwen3`; do not add timestamped public launchers or runtime dependencies
on report directories. Keep local paths in `.local/` configurations and results
in ignored output directories. Compiler-only legacy diagnostics remain internal
maintenance tools; they are not portable execution entry points.
