# Reproduce and identify the LLVM dependency

`recipe.json` pins the upstream commit, complete patched Git tree, patch hash,
build options and targets. `qualified.patch` restores the 29 source/test changes
present in the qualified local LLVM checkout. It is a compatibility snapshot,
not a claim that all 29 changes are necessary for Qwen or ready for upstream.
The separately reviewed four correctness patches remain an upstream proposal;
they do not yet replace this complete toolchain baseline.

The snapshot includes correctness fixes, old-baseline backports, compile-time
scalability changes, per-function controls and the existing BF16 source changes.
Keeping the latter here preserves the actual dependency; it does not enable or
qualify BF16 model execution. No generated binaries, reports or machine paths
are included in the patch.

## Source build

Use Linux, Python 3.11+, Git, CMake, Ninja and a compatible host Clang/Clang++.
The configured host must provide the zlib, zstd, libxml2 and libedit development
dependencies. Start with an LLVM Git repository containing the pinned commit and
its blobs; an existing checkout or an offline Git repository is sufficient.
Preparation creates a separate worktree and never edits the supplied checkout.

```sh
python dependencies/llvm/toolchain.py prepare \
  --repository /path/to/llvm-git --source /path/to/llvm-qwen-source
```

Copy `models/qwen3/config/build.example.json` in this application repository. Set `llvm_source` to
that worktree, `llvm_build` to a fresh build directory, `lld` to that build's
`bin/ld.lld`, and `llvm_receipt` to an output JSON file outside the source tree.
Set the host compiler paths and any required host `library_path` entries.

```sh
python dependencies/llvm/toolchain.py build --config /path/to/toolchain.json --jobs 8
python dependencies/llvm/toolchain.py verify --config /path/to/toolchain.json
```

The recipe preserves the four target backends, assertions, RTTI, exceptions and
O2 optimization. It omits debug information to reduce build space. CMake/Ninja
reuse unchanged outputs when rerun with the same inputs. A directory owned by
different recipe/source/compiler inputs is rejected rather than overwritten.
`configure.log` and `build.log` retain the commands' output. The receipt is
written only after the required build targets and source checks complete.

Use the same configuration with the application's `python -m models.qwen3.build`.
The application requires the receipt before configuring the backend and records
its identity in `build.json`. Model builds and host verification check that same
identity again. A new source-built dependency still needs compiler/model
qualification before replacing a qualified deployment; creating a receipt does
not assert numerical acceptance.

## What is checked

- The upstream revision and actual complete working-source tree, including
  uncommitted patches and untracked source files. The real Git index is untouched.
- LLVM/MLIR/LLD static libraries, shared libraries, generated public headers,
  CMake package files and the CMake cache.
- Compiler/tablegen/validation executables and the configured model linker,
  including an external linker used by an existing qualified installation.
- Host compiler/build executables and their resolved shared-library dependencies.

File content hashes and symlink identities are checked, not just version strings
or timestamps. Hashes are cached only within a process while file identity,
mtime and ctime remain unchanged. LLVM's full Git source history and build
intermediates are not copied into the application repository.

## Preserve an existing qualified installation

For an already accepted installation, choose a fresh `llvm_receipt` path and run:

```sh
python dependencies/llvm/toolchain.py seal-existing --config /path/to/toolchain.json
```

This checks the exact patched source tree and snapshots the existing binaries;
it neither rebuilds them nor proves that those binaries came from that source.
The receipt is explicitly labelled `qualified_prebuilt_snapshot`, whereas a
successful build is labelled `source_build`. Sealing is an explicit maintainer
action, never an automatic fallback when an identity check fails. Subsequent
changes to libraries, generated headers, host dependencies or the linker are
rejected. These receipts detect accidental dependency changes; they are not
signatures that establish trust in an unknown third-party binary.

Do not reuse an old compiler build receipt with a new LLVM identity. Rerun the
unified compiler build in its existing output directory, compare the resulting
products, and qualify any changed model code. An unchanged SDK helper need not
be rebuilt merely to record an LLVM dependency identity.
