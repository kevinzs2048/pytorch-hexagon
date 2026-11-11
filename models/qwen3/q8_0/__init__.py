# SPDX-License-Identifier: BSD-3-Clause
"""Q8_0 projections (llama.cpp Q8_0 weights) for the resident Qwen3 session.

quantize: Q8_0 tile packing and the bit-exact numpy references of the DSP runtime
weights:  identify each --qwen-q8-0-projections id against the HF checkpoint and
          pack its gguf-py Q8_0 bytes
library:  link the packed tiles into libqwen_q8_0_weights.so plus the id table
"""
