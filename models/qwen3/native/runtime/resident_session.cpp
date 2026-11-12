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
constexpr uint32_t LastMode = 4;
#else
constexpr uint32_t LastMode = 3;
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
static constexpr size_t decode_temporary_bytes = QWEN_TEMPORARY_BUFFER_BYTES / 16;
static TemporaryBuffers decode_temporary_buffers(decode_temporary_bytes);
static TemporaryBuffers prefill_temporary_buffers(
    QWEN_TEMPORARY_BUFFER_BYTES - decode_temporary_bytes);
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
extern "C" int hexagon_forward_buffers(const unsigned char *input,
                                       int input_len, unsigned char *output,
                                       int output_len, uint64_t *kernel_us) {
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
