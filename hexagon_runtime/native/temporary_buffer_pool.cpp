// SPDX-License-Identifier: BSD-3-Clause
#include "temporary_buffer_pool.h"
#include <cstdlib>
#if defined(__hexagon__)
extern "C" void *memalign(size_t, size_t);
#endif
namespace pytorch_hexagon {
namespace {
void *allocateBuffer(size_t bytes) {
  // Keep storage on an HVX vector boundary even when a caller's memref only
  // requests 64-byte alignment. Both paths remain compatible with free().
#if defined(__hexagon__)
  return memalign(128, bytes);
#else
  void *pointer = nullptr;
  return posix_memalign(&pointer, 128, bytes) ? nullptr : pointer;
#endif
}
} // namespace
struct TemporaryBufferPool::Storage {
  struct Entry { void *pointer; size_t capacity; unsigned next_pointer, next_size, state; };
  Entry entries[2048];
  unsigned pointers[4096], sizes[256], empty, count;
  size_t limit, retained;
  bool active;
  Statistics stats;
  static unsigned pointer_hash(void *p) { return (reinterpret_cast<uintptr_t>(p) >> 5) & 4095; }
  // Ordered capacity buckets avoid scanning live and never-used slots when
  // an exact-size cached allocation is unavailable.
  static unsigned size_hash(size_t n) { return n ? 63U - unsigned(__builtin_clzll(n)) : 0U; }
  unsigned find(void *p) const {
    unsigned i = pointers[pointer_hash(p)];
    while (i && entries[i - 1].pointer != p) i = entries[i - 1].next_pointer;
    return i;
  }
  void unlink_size(unsigned i) {
    auto &e = entries[i - 1]; unsigned *link = &sizes[size_hash(e.capacity)];
    while (*link && *link != i) link = &entries[*link - 1].next_size;
    if (!*link) std::abort();
    *link = e.next_size; e.next_size = 0;
  }
  void detach(unsigned i) {
    auto &e = entries[i - 1]; unsigned *link = &pointers[pointer_hash(e.pointer)];
    while (*link && *link != i) link = &entries[*link - 1].next_pointer;
    if (!*link) std::abort();
    *link = e.next_pointer; retained -= e.capacity;
    e = {}; e.next_pointer = empty; empty = i;
  }
  unsigned best_fit(size_t n) const {
    for (unsigned i = sizes[size_hash(n)]; i; i = entries[i - 1].next_size)
      if (entries[i - 1].capacity == n) return i;
    for (unsigned bucket = size_hash(n); bucket < sizeof(size_t) * 8; ++bucket) {
      unsigned best = 0;
      for (unsigned i = sizes[bucket]; i; i = entries[i - 1].next_size) {
        const auto &e = entries[i - 1];
        if (e.capacity >= n && (!best || e.capacity < entries[best - 1].capacity ||
            (e.capacity == entries[best - 1].capacity && i < best))) best = i;
      }
      // Every later bucket has larger capacities. Keep the original lowest
      // slot tie-break within this bucket, as well as the exact-size fast path.
      if (best) return best;
    }
    return 0;
  }
  bool evict() {
    unsigned best = 0;
    for (unsigned i = 1; i <= count; ++i)
      if (entries[i - 1].state == 2 &&
          (!best || entries[i - 1].capacity > entries[best - 1].capacity)) best = i;
    if (!best) return false;
    std::free(entries[best - 1].pointer); unlink_size(best); detach(best);
    ++stats.evicted; return true;
  }
};
TemporaryBufferPool::TemporaryBufferPool(size_t bytes, unsigned entries)
    : storage_(nullptr) {
  if (!bytes || !entries || entries > 2048) return;
  storage_ = static_cast<Storage *>(std::calloc(1, sizeof(Storage)));
  if (!storage_) return;
  storage_->limit = bytes; storage_->count = entries; storage_->empty = 1;
  for (unsigned i = 0; i < entries; ++i)
    storage_->entries[i].next_pointer = i + 1 == entries ? 0 : i + 2;
}
TemporaryBufferPool::~TemporaryBufferPool() {
  if (!storage_) return;
  if (storage_->active) std::abort();
  for (unsigned i = 0; i < storage_->count; ++i)
    if (storage_->entries[i].state == 2) std::free(storage_->entries[i].pointer);
  std::free(storage_);
}
bool TemporaryBufferPool::begin() {
  if (!storage_) return true; // Disabled or metadata allocation failed: malloc/free.
  if (storage_->active) return false;
  storage_->active = true; storage_->stats = {};
  storage_->stats.peak_bytes = storage_->retained;
  return true;
}
bool TemporaryBufferPool::active() const { return storage_ && storage_->active; }
void *TemporaryBufferPool::allocate(size_t n) {
  if (!storage_) return allocateBuffer(n);
  auto &s = *storage_; if (!s.active) std::abort(); ++s.stats.calls;
  if (n && n <= s.limit) {
    unsigned i = s.best_fit(n);
    if (i) {
      s.unlink_size(i); s.entries[i - 1].state = 1; ++s.stats.reused;
      return s.entries[i - 1].pointer;
    }
    while ((!s.empty || s.retained > s.limit - n) && s.evict()) {}
  }
  void *p = allocateBuffer(n); ++s.stats.fresh;
  if (!p || !n || n > s.limit || !s.empty || s.retained > s.limit - n) {
    ++s.stats.fallback; return p;
  }
  unsigned i = s.empty; auto &e = s.entries[i - 1]; s.empty = e.next_pointer;
  unsigned h = Storage::pointer_hash(p);
  e = {p, n, s.pointers[h], 0, 1}; s.pointers[h] = i; s.retained += n;
  if (s.retained > s.stats.peak_bytes) s.stats.peak_bytes = s.retained;
  return p;
}
void TemporaryBufferPool::release(void *p) {
  if (!storage_) { std::free(p); return; }
  auto &s = *storage_; unsigned i = s.find(p);
  if (!i) { if (s.active) ++s.stats.foreign; std::free(p); return; }
  if (!s.active || s.entries[i - 1].state != 1) std::abort();
  auto &e = s.entries[i - 1]; e.state = 2;
  unsigned h = Storage::size_hash(e.capacity); e.next_size = s.sizes[h]; s.sizes[h] = i;
  ++s.stats.released;
}
void TemporaryBufferPool::end() {
  if (!storage_) return;
  auto &s = *storage_; if (!s.active) std::abort();
  for (unsigned i = 1; i <= s.count; ++i) {
    auto &e = s.entries[i - 1]; if (e.state != 1) continue;
    ++s.stats.escaped; s.stats.escaped_bytes += e.capacity; s.detach(i);
  }
  s.stats.retained_bytes = s.retained; s.active = false;
}
void TemporaryBufferPool::cancel() {
  if (!storage_) return;
  auto &s = *storage_; if (!s.active) std::abort();
  for (unsigned i = 1; i <= s.count; ++i)
    if (s.entries[i - 1].state == 1) release(s.entries[i - 1].pointer);
  end();
}
TemporaryBufferPool::Statistics TemporaryBufferPool::statistics() const {
  return storage_ ? storage_->stats : Statistics{};
}
} // namespace pytorch_hexagon
