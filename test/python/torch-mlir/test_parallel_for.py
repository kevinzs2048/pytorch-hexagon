"""The synchronous callback adapter covers rows once and handles nesting."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_parallel_for_fallback_partition_and_nested_calls(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    root = Path(__file__).resolve().parents[3]
    (tmp_path/'worker-pool.h').write_text('''
#include <stdint.h>
typedef void *worker_pool_context_t;
typedef void (*worker_callback_t)(unsigned, unsigned, void *);
int worker_pool_init(worker_pool_context_t *, uint32_t);
void worker_pool_release(worker_pool_context_t *);
int worker_pool_run_func(worker_pool_context_t, worker_callback_t, void *, unsigned);
''')
    source = tmp_path/'test.cpp'
    source.write_text(r'''
#include "worker-pool.h"
#include <cassert>
#include <vector>
using Callback = void (*)(void *, uint64_t, uint64_t);
extern "C" void __hexagon_parallel_for_v1(Callback, void *, uint64_t);
static bool failInit = true;
static unsigned dispatches, calls;
int worker_pool_init(worker_pool_context_t *p, uint32_t n) {
  assert(n == 4); if (failInit) return 1; *p = reinterpret_cast<void *>(1); return 0;
}
void worker_pool_release(worker_pool_context_t *p) { *p = nullptr; }
int worker_pool_run_func(worker_pool_context_t, worker_callback_t f, void *ctx, unsigned n) {
  ++dispatches; for (unsigned id=0; id<n; ++id) f(n, id, ctx); return 0;
}
void count(void *p, uint64_t start, uint64_t size) {
  ++calls; auto &rows=*static_cast<std::vector<unsigned> *>(p);
  assert(start+size<=rows.size()); for(uint64_t i=start;i<start+size;++i) ++rows[i];
}
void nested(void *p, uint64_t start, uint64_t size) {
  std::vector<unsigned> inner(32); unsigned before=dispatches;
  __hexagon_parallel_for_v1(count, &inner, 32); assert(dispatches==before);
  for(auto x:inner) assert(x==1);
  count(p,start,size);
}
int main() {
  for(bool failure:{true,false}) { failInit=failure;
    for(unsigned n:{0,1,15,16,17,31,32,65,1024}) {
      std::vector<unsigned> rows(n); calls=0; unsigned before=dispatches;
      __hexagon_parallel_for_v1(count,&rows,n);
      for(auto x:rows) assert(x==1);
      assert(dispatches-before==unsigned(!failure && n>=16));
      assert(calls==unsigned(!failure && n>=16 ? 4 : 1));
    }
  }
  std::vector<unsigned> rows(33); __hexagon_parallel_for_v1(nested,&rows,33);
  for(auto x:rows) assert(x==1);
}
''')
    executable = tmp_path/'test'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-I'+str(tmp_path), str(source),
                    str(root/'hexagon_runtime/native/parallel_for.cpp'),
                    '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
