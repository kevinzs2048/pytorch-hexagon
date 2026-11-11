// SPDX-License-Identifier: BSD-3-Clause
#ifndef PYTORCH_HEXAGON_KV_CACHE_H
#define PYTORCH_HEXAGON_KV_CACHE_H

#include <cstddef>
#include <cstdint>

namespace pytorch_hexagon {

// Session-owned KV storage. Token-major storage makes every occupied
// prefix contiguous, regardless of spare capacity. Attention receives length,
// never capacity. A session is used by one synchronous forward at a time.
// Element is float (FP32 KV) or uint16_t (IEEE binary16 bits, FP16 KV).
template <typename Element> class BasicKVCache {
public:
  struct View {
    Element *keys;
    Element *values;
    size_t length;
    size_t heads;
    size_t width;
  };

  BasicKVCache(size_t layers, size_t heads, size_t width, size_t max_tokens,
               size_t initial_capacity = 128);
  ~BasicKVCache();
  BasicKVCache(const BasicKVCache &) = delete;
  BasicKVCache &operator=(const BasicKVCache &) = delete;

  // Begin one contiguous append. Reject stale positions, overflow and an
  // unfinished prior forward before touching cache data.
  bool begin(size_t position, size_t query);
  // Source layout is [head, query, width]; only new tokens are copied.
  bool append(size_t layer, const Element *keys, const Element *values);
  // For compiled in-place appends: the caller has written exactly the reserved
  // new rows of BOTH planes returned by history(). Mark completion without
  // copying those rows a second time. Old rows must remain unchanged.
  bool finish_layer(size_t layer);
  // The prior prefix remains readable while an append is pending. Its backing
  // allocation has already been reserved by begin(), including the new tail.
  bool history(size_t layer, View &result) const;
  // During a forward, only layers already appended can expose the new prefix.
  bool view(size_t layer, View &result) const;
  bool commit();
  void cancel();
  void reset();

  size_t length() const { return length_; }
  size_t capacity() const { return capacity_; }
  uint64_t appended_bytes() const { return appended_bytes_; }
  uint64_t relocated_bytes() const { return relocated_bytes_; }

private:
  bool reserve(size_t required);
  Element *plane(size_t layer, size_t kind) const;
  size_t layers_, heads_, width_, max_tokens_, initial_capacity_;
  size_t capacity_ = 0, length_ = 0, query_ = 0;
  uint64_t written_ = 0, appended_bytes_ = 0, relocated_bytes_ = 0;
  Element *storage_ = nullptr;
  bool valid_;
};

using KVCache = BasicKVCache<float>;
extern template class BasicKVCache<float>;
extern template class BasicKVCache<uint16_t>;
} // namespace pytorch_hexagon
#endif
