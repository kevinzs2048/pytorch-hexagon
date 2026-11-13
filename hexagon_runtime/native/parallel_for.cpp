// SPDX-License-Identifier: Apache-2.0
// Synchronous implementation of the compiler's optional row-parallel ABI.
// Numerical work stays in the compiler-generated callback. This adapter only
// partitions rows and uses the pinned llama.cpp worker pool.
#include "worker-pool.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace {
using Callback = void (*)(void *, uint64_t, uint64_t);
worker_pool_context_t pool;
std::atomic_flag busy = ATOMIC_FLAG_INIT;

struct Job {
  Callback callback;
  void *context;
  uint64_t rows;
};

void worker(unsigned workers, unsigned id, void *opaque) {
  const auto &job = *static_cast<const Job *>(opaque);
  const uint64_t start = job.rows * id / workers;
  const uint64_t end = job.rows * (id + 1) / workers;
  if (end > start)
    job.callback(job.context, start, end - start);
}

__attribute__((destructor)) void releasePool() {
  if (pool)
    worker_pool_release(&pool);
}
} // namespace

extern "C" void __hexagon_parallel_for_v1(Callback callback, void *context,
                                          uint64_t rows) {
  // Nested/concurrent calls use the serial path; the owner keeps its pool.
  if (rows < 16 || rows > UINT32_MAX || busy.test_and_set()) {
    callback(context, 0, rows);
    return;
  }
  struct Unlock {
    ~Unlock() { busy.clear(); }
  } unlock;
  if (!pool && worker_pool_init(&pool, 4)) {
    callback(context, 0, rows);
    return;
  }
  Job job{callback, context, rows};
  // A failed dispatch may have already executed some rows: never replay it.
  if (worker_pool_run_func(pool, worker, &job, 4))
    std::abort();
}
