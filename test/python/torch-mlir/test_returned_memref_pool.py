"""Returned tensor owners may recycle buffers after their contents are copied."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_returned_owner_recycles_unique_buffers_and_unwinds(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / 'test.cpp'
    source.write_text(r'''
#include "returned_memref_owner.h"
#include "temporary_buffer_pool.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
using pytorch_hexagon::TemporaryBufferPool;
static TemporaryBufferPool *current;
static unsigned releases;
static void release(void *p) { ++releases; current->release(p); }
int main() {
  for (unsigned budget : {0U, 1024U, 16384U}) {
    TemporaryBufferPool pool(budget); current=&pool;
    for (bool fail : {false, true}) {
      assert(pool.begin()); releases=0;
      void *a=pool.allocate(128), *b=pool.allocate(4096);
      assert(a && b && a!=b); std::memset(a,37,128); std::memset(b,91,4096);
      unsigned char copied[128];
      try {
        OwnedReturnedMemrefs<4> owner(release);
        assert(owner.adopt({a,b,a,nullptr}, {}));
        assert(owner.unique_buffers()==2);
        std::memcpy(copied,a,128);
        if(fail)throw std::runtime_error("after adoption");
      } catch(const std::runtime_error &) {}
      assert(releases==2);
      pool.cancel();
      assert(!pool.active() && pool.statistics().escaped==0);
      for(auto x:copied)assert(x==37);
      if (budget >= 16384) {
        assert(pool.begin());assert(pool.allocate(128)==a);assert(pool.allocate(4096)==b);pool.cancel();
      }
    }
    assert(pool.begin());void *a=pool.allocate(128);
    { OwnedReturnedMemrefs<1> owner(release); assert(!owner.adopt({a},{a})); }
    pool.release(a);pool.end();
  }
  { OwnedReturnedMemrefs<2> owner;void *p=std::malloc(16);assert(owner.adopt({p,p},{})); }
}
''')
    executable = tmp_path / 'test'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined',
                    '-I'+str(root/'models/qwen3/native/runtime/dsp-include'),
                    '-I'+str(root/'hexagon_runtime/native'), str(source),
                    str(root/'hexagon_runtime/native/temporary_buffer_pool.cpp'),
                    '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
