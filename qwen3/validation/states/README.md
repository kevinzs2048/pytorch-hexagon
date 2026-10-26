# Accepted state-transition references

These six compact references preserve the existing 304-forward regression:

| Case | Forward calls | Purpose |
| --- | ---: | --- |
| tails | 6 | q32 prefill, q1/q2 remainder and a new request |
| eos | 9 | Original 32-token budget with natural early EOS |
| old-long32 | 33 | Continuous 32-token generation |
| prefill1024 | 33 | Full 1024-position prefill, then request reset |
| decode993 | 95 | Decode through the KV boundary, then request reset |
| code64 | 128 | Original two 64-token generation requests |

The source run, its acceptance record and the independent oracle are identified
by SHA256 in each file. These are the original inputs and outputs, not outputs
produced by the new source-built compiler. Only fields needed by the maintained
comparison entry are retained; timings and obsolete deployment paths are omitted.

Each forward retains whole input/output packet hashes and all three input FNV
checksums. The output hash covers logits and all 56 KV tensors. The actual model
library identities are recorded separately. `index.json` seals the six files.

On ARM, pass a reference to `python3 -m qwen3.verify_build --reference ...` with
the qualified asset bundle and an explicit `--libraries ...` map for q1 and q2.
The original state runs used an older accepted q1 library. Consequently, even a
check of the current accepted bundle needs its actual library paths and hashes
in this map. This changes the expected library identities; it never changes the
reference input/output hashes. The smaller `reference16.json` already identifies
the current bundle and can be used without a library override.
Use a fresh output directory for each case. The comparison honors the original
generation budget and prefill chunk, including early-EOS requests.

For long sequences, add `--packet-recording hashes` to avoid retaining hundreds
of large KV packets. The native runner still hashes every byte of every input
and output, and the same comparison runs. `full` (the default) also saves raw
packets for diagnosis. A mismatch requires investigation before qualification;
hash equality is never replaced by token-only comparison.

These are state-transition checks, not an exhaustive prompt corpus or a language
quality benchmark. They supplement the compiler/operator tests and source review.
