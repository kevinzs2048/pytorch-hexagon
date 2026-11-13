// SPDX-License-Identifier: BSD-3-Clause
// Resident full-model session; bindings come from the captured model ABI.
#pragma clang fp contract(off)
#include "CRunnerUtils.cpp"
#include "common.h"
#include "hexagon_benchmark.h"
#include "kv_cache.h"
#ifdef QWEN_KV_FP16
#include "kv_fp16.h"
#endif
#include "qwen3_sleef_exp_candidate.h"
#include "qwen_resident_bindings.h"
#include "returned_memref_owner.h"
#include "tensor.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifdef QWEN_PREPACKED_FORWARD_SCOPE
extern "C" void qwen_prepacked_forward_begin();
extern "C" void qwen_prepacked_forward_end();
#endif
#ifdef QWEN_RESIDENT_NO_LOGITS
// Q8_0 runtime: leave the vocabulary projection of the next forwards uncomputed.
extern "C" void qwen_q8_0_skip_vocabulary(int skip);
// Last single-forward mode. Mode 5 (a batch of them) is dispatched separately.
constexpr uint32_t LastMode = 4;
#else
constexpr uint32_t LastMode = 3;
#endif
#ifdef QWEN_ATTENTION_FORWARD_SCOPE
// Attention K/V staging: one VTCM lease per forward instead of one per layer.
extern "C" void qwen_attention_forward_begin();
extern "C" void qwen_attention_forward_end();
#endif
#ifdef QWEN_PARALLEL_KV_EXPORT
extern "C" void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t),
                                          void *, uint64_t);
#endif
extern "C" float __hexagon_silu_exp_compat(float x) {
  return qwen3_sleef_exp_candidate(x);
}
extern "C" float _mlir_ciface___hexagon_silu_exp_compat(float x) {
  return qwen3_sleef_exp_candidate(x);
}
extern "C" float __hexagon_softmax_exp_compat(float x) {
  return qwen3_sleef_exp_candidate(x);
}
extern "C" float _mlir_ciface___hexagon_softmax_exp_compat(float x) {
  return qwen3_sleef_exp_candidate(x);
}
#ifdef QWEN_KV_FP16
// FP16 KV storage (llama.cpp's default F16 KV cache type). New rows are
// rounded once, IEEE round-to-nearest-even, exactly like the graph appends.
// Packets stay FP32: mode 1 deltas and mode 2 snapshots carry the widened
// stored values, so host-side cache reconstruction sees what attention reads.
using KVElement = uint16_t;
static void write_rows(KVElement *dst, const float *src, size_t n) {
  pytorch_hexagon::kv_round_f16(dst, src, n);
}
static void write_rows(float *dst, const float *src, size_t n) {
  uint16_t half[256];
  for (size_t i = 0; i < n; i += 256) {
    size_t count = n - i < 256 ? n - i : 256;
    pytorch_hexagon::kv_round_f16(half, src + i, count);
    pytorch_hexagon::kv_widen_f16(dst + i, half, count);
  }
}
static void export_rows(float *dst, const KVElement *src, size_t n) {
  pytorch_hexagon::kv_widen_f16(dst, src, n);
}
static void write_one(KVElement *dst, float value) { write_rows(dst, &value, 1); }
static void write_one(float *dst, float value) { write_rows(dst, &value, 1); }
#else
using KVElement = float;
static void write_rows(float *dst, const float *src, size_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}
static void export_rows(float *dst, const float *src, size_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}
static void write_one(float *dst, float value) { *dst = value; }
#endif
static pytorch_hexagon::BasicKVCache<KVElement> cache(28, 8, 128, 1024, 32);
using CacheView = pytorch_hexagon::BasicKVCache<KVElement>::View;
// The generated bindings type every plane as a float descriptor; descriptors
// pass only pointers and element-unit extents, and the FP16-KV graph addresses
// its planes with its own f16 signature.
static float *plane_pointer(KVElement *plane) {
  return reinterpret_cast<float *>(plane);
}
// One destination plane of the returned FP32 KV deltas.
template <typename Destination>
static void copy_delta_plane(Destination *target, const MemRefDescriptor<float, 4> &v,
                             unsigned query) {
  if (v.strides[3] == 1 && v.strides[1] == 128 && (query == 1 || v.strides[2] == 1024)) {
    write_rows(target, v.aligned + v.offset, query * 1024);
    return;
  }
  for (unsigned h = 0; h < 8; ++h)
    for (unsigned t = 0; t < query; ++t) {
      const float *row = v.aligned + v.offset + h * v.strides[1] + t * v.strides[2];
      Destination *dst = target + t * 1024 + h * 128;
      if (v.strides[3] == 1)
        write_rows(dst, row, 128);
      else
        for (unsigned d = 0; d < 128; ++d)
          write_one(dst + d, row[d * v.strides[3]]);
    }
}
#ifdef QWEN_TEMPORARY_BUFFER_BYTES
#include "temporary_buffer_pool.h"
// Like KVCache, this storage belongs to the compiled session and requires
// serialized forwards. Only main graph object references use these hooks.
using TemporaryBuffers = pytorch_hexagon::TemporaryBufferPool;
// Keep short decode allocations out of the larger prefill free lists. Both
// pools share the configured retention budget; neither owns persistent KV.
// A paired forward (mode 5) keeps the temporaries of two prefill graphs alive
// at once, so sessions with that mode allow the prefill pool twice its share.
// The pool retains only what forwards actually used.
static constexpr size_t decode_temporary_bytes = QWEN_TEMPORARY_BUFFER_BYTES / 16;
#ifdef QWEN_RESIDENT_NO_LOGITS
static constexpr size_t prefill_graphs = 2;
#else
static constexpr size_t prefill_graphs = 1;
#endif
static TemporaryBuffers decode_temporary_buffers(decode_temporary_bytes);
static TemporaryBuffers prefill_temporary_buffers(
    prefill_graphs * (QWEN_TEMPORARY_BUFFER_BYTES - decode_temporary_bytes));
static TemporaryBuffers *temporary_buffers = &prefill_temporary_buffers;
extern "C" __attribute__((visibility("hidden"))) void *qwen_graph_malloc(size_t n) {
  return temporary_buffers->allocate(n);
}
extern "C" __attribute__((visibility("hidden"))) void qwen_graph_free(void *p) {
  temporary_buffers->release(p);
}
#endif
#ifdef QWEN_BENCHMARK_MAX_PERFORMANCE
extern "C" bool pytorch_hexagon_request_benchmark_performance();
#endif
int main() { return 2; }
static int forward_packet(const unsigned char *input, int input_len,
                          unsigned char *output, int output_len,
                          uint64_t *kernel_us) {
  if (!input || !output || !kernel_us || input_len < 64 || output_len < 64 ||
      std::memcmp(input, "QWMS001\0", 8))
    return 2;
  uint32_t mode, position, query;
  std::memcpy(&mode, input + 8, 4);
  std::memcpy(&position, input + 12, 4);
  std::memcpy(&query, input + 16, 4);
  // Mode 4 is mode 3 without logits: the forward appends to the resident
  // cache and the response is the 64-byte header only. Callers use it for
  // prompt chunks whose prediction they discard.
  const bool forward = mode == 1 || mode == 3 || mode == 4;
  if (mode > LastMode || position > 1024 ||
      (forward ? (query < 1 || query > 32 || query > 1024 - position)
                 : query != 0))
    return 2;
  if (mode == 0 && position)
    return 2;
  for (int i = 20; i < 64; ++i)
    if (input[i])
      return 2;
  size_t history_bytes = 56 * size_t(position) * 1024 * 4;
  if (input_len != 64 + (forward ? query * 8 : 0) ||
      output_len != 64 + (mode == 1   ? (151936 + 56 * query * 1024) * 4
                          : mode == 3 ? 151936 * 4
                          : mode == 2 ? history_bytes
                                      : 0))
    return 2;
#ifdef QWEN_BENCHMARK_MAX_PERFORMANCE
  if (!pytorch_hexagon_request_benchmark_performance())
    return 7;
#endif
  std::memset(output, 0, 64);
  std::memcpy(output, "QWMS001\0", 8);
  *kernel_us = 0;
  uint64_t update_us = 0, model_us = 0;
  try {
    if (mode == 0) {
      cache.reset();
    } else if (mode == 2) {
      if (position != cache.length())
        return 2;
      for (unsigned l = 0; l < 28; ++l) {
        CacheView v{};
        if (!cache.view(l, v))
          return 3;
        size_t bytes = position * 1024 * 4;
        export_rows(reinterpret_cast<float *>(output + 64 + 2 * l * bytes), v.keys,
                    position * 1024);
        export_rows(reinterpret_cast<float *>(output + 64 + (2 * l + 1) * bytes),
                    v.values, position * 1024);
      }
    } else {
      if (position != cache.length() || position == 1024)
        return 2;
      bool ok = false;
      update_us =
          benchmark_time_us(1, [&] { ok = cache.begin(position, query); });
      if (!ok)
        return 3;
      MemRefDescriptor<float, 4> kv[56];
      for (unsigned l = 0; l < 28; ++l) {
        CacheView v{};
        if (!cache.history(l, v)) {
          cache.cancel();
          return 3;
        }
        kv[2 * l] = {plane_pointer(v.keys),
                     plane_pointer(v.keys),
                     0,
                     {1, 8, position, 128},
                     {position * 1024, 128, 1024, 1}};
        kv[2 * l + 1] = {plane_pointer(v.values),
                         plane_pointer(v.values),
                         0,
                         {1, 8, position, 128},
                         {position * 1024, 128, 1024, 1}};
      }
      int64_t tokens[32], pos[32], ones[1025];
      std::memcpy(tokens, input + 64, query * 8);
      for (unsigned i = 0; i < query; ++i) {
        if (tokens[i] < 0 || tokens[i] >= 151936) {
          cache.cancel();
          return 2;
        }
        pos[i] = position + i;
      }
      for (unsigned i = 0; i < position + query; ++i)
        ones[i] = 1;
      MemRefDescriptor<int64_t, 2> ids{
          tokens, tokens, 0, {1, query}, {query, 1}};
      MemRefDescriptor<int64_t, 1> cache_position{pos, pos, 0, {query}, {1}};
      MemRefDescriptor<int64_t, 2> positions{
          pos, pos, 0, {1, query}, {query, 1}};
      MemRefDescriptor<int64_t, 2> mask{
          ones, ones, 0, {1, position + query}, {position + query, 1}};
      Results results{};
#ifdef QWEN_TEMPORARY_BUFFER_BYTES
      temporary_buffers = query == 1 ? &decode_temporary_buffers
                                     : &prefill_temporary_buffers;
      if (!temporary_buffers->begin()) {
        cache.cancel();
        return 3;
      }
      // Returned values are either copied out or retained in the resident cache.
      // Destroy their owner first, then return any remaining temporaries to the
      // pool on both normal and exceptional exits. No graph buffer escapes.
      struct PoolScope {
        TemporaryBuffers &pool;
        ~PoolScope() { pool.cancel(); }
      } pool_scope{*temporary_buffers};
      OwnedReturnedMemrefs<57> owner(qwen_graph_free);
#else
      OwnedReturnedMemrefs<57> owner;
#endif
      model_us = benchmark_time_us(1, [&] {
#ifdef QWEN_PREPACKED_FORWARD_SCOPE
        qwen_prepacked_forward_begin();
        struct ForwardResources {
          ~ForwardResources() { qwen_prepacked_forward_end(); }
        } resources;
#endif
#ifdef QWEN_ATTENTION_FORWARD_SCOPE
        qwen_attention_forward_begin();
        struct AttentionResources {
          ~AttentionResources() { qwen_attention_forward_end(); }
        } attention_resources;
#endif
#ifdef QWEN_RESIDENT_NO_LOGITS
        struct SkipLogits {
          bool active;
          explicit SkipLogits(bool on) : active(on) {
            if (active) qwen_q8_0_skip_vocabulary(1);
          }
          ~SkipLogits() {
            if (active) qwen_q8_0_skip_vocabulary(0);
          }
        } skip_logits(mode == 4);
#endif
        qwen_resident_forward(&results, query, position, cache.capacity(), ids,
                              cache_position, positions, mask, kv);

      });
      if (!qwen_adopt_results(owner, results, kv, ids, cache_position,
                              positions, mask)) {
        cache.cancel();
        return 4;
      }
      auto &logits = results.logits;
      if (!logits.aligned || logits.sizes[0] != 1 || logits.sizes[1] != 1 ||
          logits.sizes[2] != 151936) {
        cache.cancel();
        return 4;
      }
      float *out = reinterpret_cast<float *>(output + 64);
      if (mode == 4)
        ;  // Unwritten logits buffer; the response carries no payload.
      else if (logits.strides[2] == 1)
        std::memcpy(out, logits.aligned + logits.offset, 151936 * sizeof(float));
      else
        for (unsigned i = 0; i < 151936; ++i)
          out[i] = logits.aligned[logits.offset + i * logits.strides[2]];
      // Mode 1 exports new KV rows for the generic PyTorch/diagnostic path.
      // Mode 3 keeps them on the DSP and returns only the prediction logits.
      // Mode 4 keeps them on the DSP and returns nothing.
      float *delta = mode == 1 ? out + 151936 : nullptr;
      // Validate every returned view before workers write disjoint planes.
      for (unsigned i = 0; i < 56; ++i) {
        const auto &v = results.delta[i];
        if (!v.aligned || v.sizes[0] != 1 || v.sizes[1] != 8 ||
            v.sizes[2] != query || v.sizes[3] != 128) {
          cache.cancel();
          return 4;
        }
      }
      // Mode 1 writes FP32 packet rows; position 0 writes cache planes.
      float *destinations[56]{};
      KVElement *cache_planes[56]{};
      if (mode == 1) {
        for (unsigned i = 0; i < 56; ++i)
          destinations[i] = delta + i * query * 1024;
      } else if (!position) {
        for (unsigned l = 0; l < 28; ++l) {
          CacheView view{};
          if (!cache.history(l, view)) {
            cache.cancel();
            return 3;
          }
          cache_planes[2 * l] = view.keys;
          cache_planes[2 * l + 1] = view.values;
        }
      }
      struct CopyJob {
        Results *results;
        float **destinations;
        KVElement **cache_planes;
        unsigned query;
      } copy_job{&results, destinations, cache_planes, query};
      auto copy_planes = [](void *opaque, uint64_t begin, uint64_t count) {
        const auto &job = *static_cast<CopyJob *>(opaque);
        for (unsigned i = unsigned(begin); i < unsigned(begin + count); ++i) {
          const auto &v = job.results->delta[i];
          unsigned plane = output_plane[i];
          if (job.destinations[plane])
            copy_delta_plane(job.destinations[plane], v, job.query);
          else
            copy_delta_plane(job.cache_planes[plane], v, job.query);
        }
      };
      auto copy_delta = [&] {
#ifdef QWEN_PARALLEL_KV_EXPORT
        // output_plane is a validated permutation. Workers write disjoint
        // planes and join before either packet or cache storage is consumed.
        if (query >= 16)
          __hexagon_parallel_for_v1(copy_planes, &copy_job, 56);
        else
#endif
          copy_planes(&copy_job, 0, 56);
      };
      if (mode == 1)
        copy_delta();
      else if (!position)
        update_us += benchmark_time_us(1, copy_delta);
      if (!position && mode == 1)
        update_us += benchmark_time_us(1, [&] {
          for (unsigned l = 0; l < 28; ++l) {
            CacheView v{};
            if (!cache.history(l, v)) {
              ok = false;
              return;
            }
            // Packet rows already hold stored values; FP16 rounding is exact.
            write_rows(v.keys, delta + 2 * l * query * 1024, query * 1024);
            write_rows(v.values, delta + (2 * l + 1) * query * 1024, query * 1024);
          }
        });
      if (!ok) {
        cache.cancel();
        return 3;
      }
      // Full KV contents are checked by the opt-in --verify-resident-cache
      // snapshot comparison. Normal inference does not rescan returned rows.
      update_us += benchmark_time_us(1, [&] {
        for (unsigned l = 0; ok && l < 28; ++l)
          ok = cache.finish_layer(l);
        if (ok)
          ok = cache.commit();
      });
      if (!ok) {
        cache.cancel();
        return 4;
      }
    }
    uint64_t metadata[] = {
        cache.length(),          cache.capacity(), cache.appended_bytes(),
        cache.relocated_bytes(), update_us,        model_us};
    std::memcpy(output + 8, metadata, sizeof(metadata));
    *kernel_us = update_us + model_us;
    return 0;
  } catch (...) {
#ifdef QWEN_TEMPORARY_BUFFER_BYTES
    if (temporary_buffers->active()) temporary_buffers->cancel();
#endif
    cache.cancel();
    return 5;
  }
}
#ifdef QWEN_RESIDENT_NO_LOGITS
// ---- Paired chunks ----
// Two consecutive prompt chunks of a mode 5 batch run as two instances of the
// prefill graph on this thread: the earlier chunk on its own stack, the later
// one on the caller's. The Q8_0 runtime makes them meet at every projection
// (qwen_q8_0_pair_begin), so each weight is read and dequantized once for both
// chunks. The earlier chunk always advances first: its K/V rows of a layer are
// in the cache before the later chunk's attention of that layer reads them.
// Every row's arithmetic is that of separate forwards.
extern "C" void qwen_q8_0_pair_begin(void (*)(void *), void *);
extern "C" int qwen_q8_0_pair_end();
// Callee-saved registers of a suspended context: r16..r27, r29:28, r31:30 and
// the stack limit checked by allocframe.
struct Context {
  uint64_t r[8];
  uint32_t framelimit, reserved;
};
extern "C" void qwen_context_switch(Context *from, Context *to);
asm(R"(
    .text
    .p2align 4
    .globl qwen_context_switch
    .type qwen_context_switch, @function
qwen_context_switch:
    memd(r0+#0) = r17:16
    memd(r0+#8) = r19:18
    memd(r0+#16) = r21:20
    memd(r0+#24) = r23:22
    memd(r0+#32) = r25:24
    memd(r0+#40) = r27:26
    memd(r0+#48) = r29:28
    memd(r0+#56) = r31:30
    r2 = framelimit
    memw(r0+#64) = r2
    r17:16 = memd(r1+#0)
    r19:18 = memd(r1+#8)
    r21:20 = memd(r1+#16)
    r23:22 = memd(r1+#24)
    r25:24 = memd(r1+#32)
    r27:26 = memd(r1+#40)
    r2 = memw(r1+#64)
    framelimit = r2
    r29:28 = memd(r1+#48)
    r31:30 = memd(r1+#56)
    jumpr r31
    .size qwen_context_switch, .-qwen_context_switch
)");
// The inputs and results of one graph of a pair.
struct GraphRun {
  int64_t tokens[32], pos[32], ones[1025];
  MemRefDescriptor<float, 4> kv[56];
  MemRefDescriptor<int64_t, 2> ids, positions, mask;
  MemRefDescriptor<int64_t, 1> cache_position;
  Results results;
  uint32_t query, position;
  // False for a token outside the vocabulary or a missing cache layer.
  bool prepare(const unsigned char *raw, uint32_t count, uint32_t at) {
    query = count; position = at;
    std::memcpy(tokens, raw, count * 8);
    for (unsigned i = 0; i < count; ++i) {
      if (tokens[i] < 0 || tokens[i] >= 151936) return false;
      pos[i] = at + i;
    }
    for (unsigned i = 0; i < at + count; ++i) ones[i] = 1;
    for (unsigned l = 0; l < 28; ++l) {
      CacheView v{};
      if (!cache.history(l, v)) return false;
      // The planes hold this graph's history: for the later chunk that
      // includes the rows the earlier chunk appends layer by layer.
      kv[2 * l] = {plane_pointer(v.keys), plane_pointer(v.keys), 0, {1, 8, at, 128}, {at * 1024, 128, 1024, 1}};
      kv[2 * l + 1] = {plane_pointer(v.values), plane_pointer(v.values), 0, {1, 8, at, 128}, {at * 1024, 128, 1024, 1}};
    }
    ids = {tokens, tokens, 0, {1, count}, {count, 1}};
    cache_position = {pos, pos, 0, {count}, {1}};
    positions = {pos, pos, 0, {1, count}, {count, 1}};
    mask = {ones, ones, 0, {1, at + count}, {at + count, 1}};
    results = Results{};
    return true;
  }
  void run() {
    qwen_resident_forward(&results, query, position, cache.capacity(), ids, cache_position, positions, mask, kv);
  }
  bool valid() const {
    const auto &logits = results.logits;
    if (!logits.aligned || logits.sizes[0] != 1 || logits.sizes[1] != 1 || logits.sizes[2] != 151936) return false;
    for (const auto &v : results.delta)
      if (!v.aligned || v.sizes[0] != 1 || v.sizes[1] != 8 || v.sizes[2] != query || v.sizes[3] != 128) return false;
    return true;
  }
};
// Stack of the earlier chunk's graph (the compiled graph and the runtime
// kernels it calls; measured use on QCS9075: 30 KB).
#ifndef QWEN_PAIR_STACK_BYTES
#define QWEN_PAIR_STACK_BYTES (1u << 20)
#endif
extern "C" void *memalign(size_t, size_t);
constexpr size_t PairStackBytes = QWEN_PAIR_STACK_BYTES;
struct PairState {
  Context caller, earlier;
  GraphRun runs[2];  // earlier chunk (own stack), later chunk (caller's stack)
  unsigned char *stack;
  bool on_earlier, earlier_done, earlier_failed, skip_later_logits;
};
static PairState pair_state;
struct PairFailure {};
// Called by the Q8_0 runtime from whichever graph is running.
static void pair_yield(void *) {
  PairState &s = pair_state;
  if (s.on_earlier) {
    s.on_earlier = false;
    qwen_q8_0_skip_vocabulary(s.skip_later_logits);
    qwen_context_switch(&s.earlier, &s.caller);
  } else {
    // The earlier graph failed: unwind the later one into forward_pair.
    if (s.earlier_failed) throw PairFailure{};
    if (s.earlier_done) std::abort();  // the graphs did not reach the same projection
    s.on_earlier = true;
    qwen_q8_0_skip_vocabulary(1);  // the earlier chunk's prediction is never used
    qwen_context_switch(&s.caller, &s.earlier);
  }
}
// Entry of the earlier chunk's context; it never returns.
static void pair_earlier_entry() {
  PairState &s = pair_state;
  try {
    s.runs[0].run();
  } catch (...) {
    s.earlier_failed = true;
  }
  s.earlier_done = true;
  s.on_earlier = false;
  qwen_q8_0_skip_vocabulary(s.skip_later_logits);
  qwen_context_switch(&s.earlier, &s.caller);
  std::abort();
}
// Two prefill forwards at `position`: counts[0] tokens, then counts[1]. The
// response is the 64-byte header, followed by the later chunk's logits when
// `logits`. Status codes as forward_packet.
static int forward_pair(uint32_t position, const unsigned char *tokens, const uint32_t counts[2], bool logits,
                        unsigned char *output, uint64_t *kernel_us) {
  PairState &s = pair_state;
  const uint32_t total = counts[0] + counts[1];
  if (position != cache.length() || !position || counts[0] < 2 || counts[0] > 32 || counts[1] < 2 ||
      counts[1] > 32 || total > 1024 - position)
    return 2;
#ifdef QWEN_BENCHMARK_MAX_PERFORMANCE
  if (!pytorch_hexagon_request_benchmark_performance())
    return 7;
#endif
  if (!s.stack && !(s.stack = static_cast<unsigned char *>(memalign(4096, PairStackBytes))))
    return 3;
  std::memset(output, 0, 64);
  std::memcpy(output, "QWMS001\0", 8);
  *kernel_us = 0;
  uint64_t update_us = 0, model_us = 0;
  try {
    bool ok = false;
    update_us = benchmark_time_us(1, [&] { ok = cache.begin(position, total); });
    if (!ok)
      return 3;
    if (!s.runs[0].prepare(tokens, counts[0], position) ||
        !s.runs[1].prepare(tokens + counts[0] * 8, counts[1], position + counts[0])) {
      cache.cancel();
      return 2;
    }
#ifdef QWEN_TEMPORARY_BUFFER_BYTES
    temporary_buffers = &prefill_temporary_buffers;
    if (!temporary_buffers->begin()) {
      cache.cancel();
      return 3;
    }
    // As forward_packet: owners first, then every remaining temporary of
    // either graph (also of an abandoned one) returns to the pool.
    struct PoolScope {
      TemporaryBuffers &pool;
      ~PoolScope() { pool.cancel(); }
    } pool_scope{*temporary_buffers};
    OwnedReturnedMemrefs<57> owners[2]{OwnedReturnedMemrefs<57>(qwen_graph_free),
                                       OwnedReturnedMemrefs<57>(qwen_graph_free)};
#else
    OwnedReturnedMemrefs<57> owners[2];
#endif
    bool met = false;
    model_us = benchmark_time_us(1, [&] {
#ifdef QWEN_PREPACKED_FORWARD_SCOPE
      qwen_prepacked_forward_begin();
      struct ForwardResources {
        ~ForwardResources() { qwen_prepacked_forward_end(); }
      } resources;
#endif
#ifdef QWEN_ATTENTION_FORWARD_SCOPE
      qwen_attention_forward_begin();
      struct AttentionResources {
        ~AttentionResources() { qwen_attention_forward_end(); }
      } attention_resources;
#endif
      struct Meeting {
        bool &met;
        explicit Meeting(bool &result) : met(result) { qwen_q8_0_pair_begin(pair_yield, nullptr); }
        ~Meeting() {
          met = qwen_q8_0_pair_end();
          qwen_q8_0_skip_vocabulary(0);
        }
      } meeting(met);
      s.earlier_done = s.earlier_failed = false;
      s.skip_later_logits = !logits;
      s.earlier = Context{};
      const uint32_t top = uint32_t(uintptr_t(s.stack + PairStackBytes - 64)) & ~15u;
      s.earlier.r[6] = uint64_t(top) << 32;                                           // r29 (sp)
      s.earlier.r[7] = uint64_t(uint32_t(uintptr_t(&pair_earlier_entry))) << 32;      // r31 (lr)
      s.earlier.framelimit = uint32_t(uintptr_t(s.stack));
      s.on_earlier = true;
      qwen_q8_0_skip_vocabulary(1);
      qwen_context_switch(&s.caller, &s.earlier);  // until its first projection
      if (!s.earlier_failed)
        s.runs[1].run();
    });
    // The earlier graph must have finished, and no projection may be parked.
    if (!met || !s.earlier_done || s.earlier_failed) {
      cache.cancel();
      return 4;
    }
    for (unsigned g = 0; g < 2; ++g) {
      GraphRun &run = s.runs[g];
      if (!qwen_adopt_results(owners[g], run.results, run.kv, run.ids, run.cache_position, run.positions,
                              run.mask) ||
          !run.valid()) {
        cache.cancel();
        return 4;
      }
    }
    if (logits) {
      const auto &v = s.runs[1].results.logits;
      float *out = reinterpret_cast<float *>(output + 64);
      if (v.strides[2] == 1)
        std::memcpy(out, v.aligned + v.offset, 151936 * sizeof(float));
      else
        for (unsigned i = 0; i < 151936; ++i)
          out[i] = v.aligned[v.offset + i * v.strides[2]];
    }
    // Both graphs appended their rows in place (position > 0).
    update_us += benchmark_time_us(1, [&] {
      for (unsigned l = 0; ok && l < 28; ++l)
        ok = cache.finish_layer(l);
      if (ok)
        ok = cache.commit();
    });
    if (!ok) {
      cache.cancel();
      return 4;
    }
    uint64_t metadata[] = {
        cache.length(),          cache.capacity(), cache.appended_bytes(),
        cache.relocated_bytes(), update_us,        model_us};
    std::memcpy(output + 8, metadata, sizeof(metadata));
    *kernel_us = update_us + model_us;
    return 0;
  } catch (...) {
#ifdef QWEN_TEMPORARY_BUFFER_BYTES
    if (temporary_buffers->active()) temporary_buffers->cancel();
#endif
    cache.cancel();
    return 5;
  }
}
// Mode 5: a whole logical batch in one call. Header word 3 (offset 20) is the
// chunk size. The batch runs as the same forwards a caller would send one by
// one: mode 4 for every chunk but the last, mode 3 for the last, whose logits
// are the response. Chunks commit one by one; after a failure the cache holds
// the chunks that completed, as it would for separate calls (a pair commits
// or fails as one). Two consecutive
// chunks that both run the prefill graph (position > 0, two or more tokens)
// are executed as a pair unless header word 4 (offset 24) is 1.
static int forward_batch(const unsigned char *input, int input_len,
                         unsigned char *output, int output_len,
                         uint64_t *kernel_us) {
  uint32_t position, query, chunk, sequential;
  std::memcpy(&position, input + 12, 4);
  std::memcpy(&query, input + 16, 4);
  std::memcpy(&chunk, input + 20, 4);
  std::memcpy(&sequential, input + 24, 4);
  if (position > 1024 || query < 1 || query > 1024 - position || chunk < 1 ||
      chunk > 32 || sequential > 1 || input_len != int(64 + query * 8) ||
      output_len != 64 + 151936 * 4)
    return 2;
  for (int i = 28; i < 64; ++i)
    if (input[i])
      return 2;
  uint64_t update_us = 0, model_us = 0;
  for (uint32_t offset = 0, count; offset < query; offset += count) {
    count = query - offset < chunk ? query - offset : chunk;
    const uint32_t at = position + offset;
    const uint32_t next = query - offset - count < chunk ? query - offset - count : chunk;
    unsigned char header[64];
    if (!sequential && at && count >= 2 && next >= 2) {
      const uint32_t counts[2] = {count, next};
      const bool last = offset + count + next == query;
      unsigned char *response = last ? output : header;
      uint64_t pair_us = 0;
      if (int status = forward_pair(at, input + 64 + offset * 8, counts, last, response, &pair_us))
        return status;
      uint64_t times[2];
      std::memcpy(times, response + 8 + 4 * 8, sizeof(times));
      update_us += times[0];
      model_us += times[1];
      count += next;
      continue;
    }
    const bool last = offset + count == query;
    const uint32_t mode = last ? 3 : 4;
    unsigned char packet[64 + 32 * 8]{};
    std::memcpy(packet, "QWMS001\0", 8);
    std::memcpy(packet + 8, &mode, 4);
    std::memcpy(packet + 12, &at, 4);
    std::memcpy(packet + 16, &count, 4);
    std::memcpy(packet + 64, input + 64 + offset * 8, count * 8);
    unsigned char *response = last ? output : header;
    uint64_t chunk_us = 0;
    if (int status = forward_packet(packet, 64 + count * 8, response,
                                    last ? output_len : 64, &chunk_us))
      return status;
    uint64_t times[2];
    std::memcpy(times, response + 8 + 4 * 8, sizeof(times));
    update_us += times[0];
    model_us += times[1];
  }
  const uint64_t times[2] = {update_us, model_us};
  std::memcpy(output + 8 + 4 * 8, times, sizeof(times));
  *kernel_us = update_us + model_us;
  return 0;
}
#endif
extern "C" int hexagon_forward_buffers(const unsigned char *input,
                                       int input_len, unsigned char *output,
                                       int output_len, uint64_t *kernel_us) {
#ifdef QWEN_RESIDENT_NO_LOGITS
  uint32_t mode;
  if (input && output && kernel_us && input_len >= 64 &&
      !std::memcmp(input, "QWMS001\0", 8) &&
      (std::memcpy(&mode, input + 8, 4), mode == 5))
    return forward_batch(input, input_len, output, output_len, kernel_us);
#endif
  return forward_packet(input, input_len, output, output_len, kernel_us);
}
