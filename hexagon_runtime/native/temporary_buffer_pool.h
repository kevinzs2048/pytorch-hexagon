// SPDX-License-Identifier: BSD-3-Clause
#ifndef PYTORCH_HEXAGON_TEMPORARY_BUFFER_POOL_H
#define PYTORCH_HEXAGON_TEMPORARY_BUFFER_POOL_H
#include <cstddef>
#include <cstdint>
namespace pytorch_hexagon {
// One synchronous graph at a time, owned by its session. Only explicit releases
// permit reuse. end() transfers still-live allocations to the original caller.
// Storage is 128-byte aligned and free()-compatible, including disabled/fallback
// allocation. The byte limit bounds requested pool-owned capacity, excluding
// allocator overhead, escaped outputs and fallback allocations.
class TemporaryBufferPool {
public:
  struct Statistics {
    uint64_t calls, reused, fresh, fallback, released, foreign, escaped;
    uint64_t escaped_bytes, retained_bytes, peak_bytes, evicted;
  };
  explicit TemporaryBufferPool(size_t byte_limit, unsigned entry_limit = 2048);
  ~TemporaryBufferPool();
  TemporaryBufferPool(const TemporaryBufferPool &) = delete;
  TemporaryBufferPool &operator=(const TemporaryBufferPool &) = delete;
  bool begin();
  void *allocate(size_t bytes);
  void release(void *pointer);
  void end();
  // Use only when no result has been transferred to a caller. Fallback
  // allocations remain the graph's responsibility, just as with malloc.
  void cancel();
  bool active() const;
  Statistics statistics() const;
private:
  struct Storage;
  Storage *storage_;
};
} // namespace pytorch_hexagon
#endif
