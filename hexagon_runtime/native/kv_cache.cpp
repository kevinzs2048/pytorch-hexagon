// SPDX-License-Identifier: BSD-3-Clause
#include "kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace pytorch_hexagon {
template <typename Element>
BasicKVCache<Element>::BasicKVCache(size_t layers, size_t heads, size_t width,
                                    size_t max_tokens, size_t initial_capacity)
    : layers_(layers), heads_(heads), width_(width), max_tokens_(max_tokens),
      initial_capacity_(initial_capacity),
      valid_(layers && layers <= 64 && heads && width && max_tokens && initial_capacity) {
  // Bound multiplication and alignment padding, including on a 32-bit DSP.
  size_t available = std::numeric_limits<size_t>::max() - 127;
  for (size_t factor : {layers, size_t(2), heads, width, max_tokens, sizeof(Element)}) {
    if (!factor || factor > available) {
      valid_ = false;
      break;
    }
    available /= factor;
  }
}

template <typename Element> BasicKVCache<Element>::~BasicKVCache() { std::free(storage_); }

template <typename Element>
Element *BasicKVCache<Element>::plane(size_t layer, size_t kind) const {
  return storage_ + (2 * layer + kind) * capacity_ * heads_ * width_;
}

template <typename Element>
bool BasicKVCache<Element>::reserve(size_t required) {
  if (required <= capacity_) return true;
  size_t next = capacity_ ? capacity_ :
      (initial_capacity_ < max_tokens_ ? initial_capacity_ : max_tokens_);
  while (next < required)
    next = next > max_tokens_ / 2 ? max_tokens_ : next * 2;
  size_t plane_words = next * heads_ * width_;
  size_t bytes = layers_ * 2 * plane_words * sizeof(Element);
  auto *replacement = static_cast<Element *>(std::aligned_alloc(128, (bytes + 127) & ~size_t(127)));
  if (!replacement) return false;
  size_t prefix_bytes = length_ * heads_ * width_ * sizeof(Element);
  if (prefix_bytes) {
    for (size_t layer = 0; layer < layers_; ++layer)
      for (size_t kind = 0; kind < 2; ++kind)
        std::memcpy(replacement + (2 * layer + kind) * plane_words,
                    plane(layer, kind), prefix_bytes);
    relocated_bytes_ += 2 * layers_ * prefix_bytes;
  }
  std::free(storage_);
  storage_ = replacement;
  capacity_ = next;
  return true;
}

template <typename Element>
bool BasicKVCache<Element>::begin(size_t position, size_t query) {
  if (!valid_ || query_ || position != length_ || !query ||
      query > max_tokens_ - length_) return false;
  if (!reserve(length_ + query)) return false;
  query_ = query;
  written_ = 0;
  return true;
}

template <typename Element>
bool BasicKVCache<Element>::append(size_t layer, const Element *keys,
                                   const Element *values) {
  if (!query_ || layer >= layers_ || !keys || !values ||
      (written_ & (uint64_t(1) << layer))) return false;
  const Element *sources[] = {keys, values};
  for (size_t kind = 0; kind < 2; ++kind) {
    Element *destination = plane(layer, kind) + length_ * heads_ * width_;
    for (size_t token = 0; token < query_; ++token)
      for (size_t head = 0; head < heads_; ++head)
        std::memcpy(destination + (token * heads_ + head) * width_,
                    sources[kind] + (head * query_ + token) * width_,
                    width_ * sizeof(Element));
  }
  return finish_layer(layer);
}

template <typename Element>
bool BasicKVCache<Element>::finish_layer(size_t layer) {
  if (!query_ || layer >= layers_ || (written_ & (uint64_t(1) << layer)))
    return false;
  written_ |= uint64_t(1) << layer;
  appended_bytes_ += 2 * query_ * heads_ * width_ * sizeof(Element);
  return true;
}

template <typename Element>
bool BasicKVCache<Element>::history(size_t layer, View &result) const {
  result = {};
  if (!valid_ || !storage_ || layer >= layers_) return false;
  result = {plane(layer, 0), plane(layer, 1), length_, heads_, width_};
  return true;
}

template <typename Element>
bool BasicKVCache<Element>::view(size_t layer, View &result) const {
  result = {};
  if (query_ && (layer >= layers_ || !(written_ & (uint64_t(1) << layer)))) return false;
  if (!history(layer, result)) return false;
  result.length += query_;
  return true;
}

template <typename Element> bool BasicKVCache<Element>::commit() {
  if (!valid_) return false;
  const uint64_t complete = layers_ == 64 ? ~uint64_t(0) : (uint64_t(1) << layers_) - 1;
  if (!query_ || written_ != complete) return false;
  length_ += query_;
  cancel();
  return true;
}

template <typename Element> void BasicKVCache<Element>::cancel() { query_ = 0; written_ = 0; }
template <typename Element> void BasicKVCache<Element>::reset() { cancel(); length_ = 0; }

template class BasicKVCache<float>;
template class BasicKVCache<uint16_t>;
} // namespace pytorch_hexagon
