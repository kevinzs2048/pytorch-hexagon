"""Temporary storage ownership, mixed sizes, bounded eviction and failure cleanup."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_temporary_buffer_pool(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    native = Path(__file__).resolve().parents[3] / 'hexagon_runtime/native'
    source = tmp_path / 'test.cpp'
    source.write_text(r'''
#include "temporary_buffer_pool.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
using pytorch_hexagon::TemporaryBufferPool;
static bool aligned(const void *p) {
  return p && reinterpret_cast<uintptr_t>(p) % 128 == 0;
}
int main() {
  void *escaped;
  {
    TemporaryBufferPool pool(1024, 2);
    assert(pool.begin() && !pool.begin());
    auto *a = pool.allocate(600), *b = pool.allocate(600);
    assert(aligned(a) && aligned(b) && a != b); memset(a, 37, 600);
    pool.release(a); pool.release(b); pool.end();
    assert(pool.statistics().fallback == 1 && pool.statistics().retained_bytes == 600);
    assert(pool.begin());
    escaped = pool.allocate(400); // A larger free allocation is safe to reuse.
    assert(aligned(escaped) && escaped == a && static_cast<unsigned char *>(escaped)[399] == 37);
    auto *small = pool.allocate(128); assert(aligned(small) && small != escaped);
    pool.release(small); pool.end();
    assert(pool.statistics().escaped == 1 && pool.statistics().escaped_bytes == 600);
    assert(pool.statistics().retained_bytes == 128);
    assert(pool.begin());
    a = pool.allocate(300); b = pool.allocate(300);
    assert(aligned(a) && aligned(b) && a != b); // Evict the unused 128-byte slot, not the live 300-byte block.
    assert(pool.statistics().evicted == 1 && pool.statistics().fallback == 0);
    memset(a, 11, 300); memset(b, 22, 300);
    pool.release(a); pool.release(b); pool.end();
    assert(pool.statistics().retained_bytes == 600);
    assert(pool.begin());
    a = pool.allocate(800); // Byte budget requires cached allocations to be evicted.
    assert(a && pool.statistics().evicted == 2); pool.release(a); pool.end();
    for (int i = 0; i < 100; ++i) {
      assert(pool.begin());
      a = pool.allocate(700); assert(a);
      memset(a, i, 700); pool.cancel();
      assert(!pool.active() && pool.statistics().escaped == 0);
      assert(pool.statistics().retained_bytes == 800);
    }
    assert(static_cast<unsigned char *>(escaped)[399] == 37);
  }
  assert(static_cast<unsigned char *>(escaped)[399] == 37); free(escaped);
  {
    TemporaryBufferPool pool(8192, 8);
    assert(pool.begin());
    auto *a = pool.allocate(600), *b = pool.allocate(700), *c = pool.allocate(1800);
    pool.release(a); pool.release(b); pool.release(c);
    // Best fit within one capacity bucket, then across bucket boundaries;
    // live allocations must not be returned again.
    assert(pool.allocate(500) == a);
    assert(pool.allocate(650) == b);
    assert(pool.allocate(1500) == c);
    pool.release(a); pool.release(b); pool.release(c); pool.end();
    assert(pool.begin());
    assert(pool.allocate(700) == b); // Preserve exact-size preference.
    pool.cancel();
  }
  TemporaryBufferPool first(1024), second(1024);
  assert(first.begin() && second.begin());
  auto *a = first.allocate(32), *b = second.allocate(32); assert(a != b);
  first.release(a); second.release(b); first.end(); second.end();
  TemporaryBufferPool disabled(0);
  assert(disabled.begin()); a = disabled.allocate(32); assert(aligned(a));
  disabled.release(a); disabled.end();
}
''')
    executable = tmp_path / 'test'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-I'+str(native), str(source),
                    str(native/'temporary_buffer_pool.cpp'), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
