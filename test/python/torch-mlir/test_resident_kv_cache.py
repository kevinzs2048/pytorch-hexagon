"""Native cache lifetime, incremental writes and geometric growth."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_resident_kv_cache(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    native = Path(__file__).resolve().parents[3] / 'hexagon_runtime/native'
    source = tmp_path / 'test.cpp'
    source.write_text(r'''
#include "kv_cache.h"
#include <cassert>
#include <cstdint>
#include <limits>
using pytorch_hexagon::KVCache;
int main() {
  KVCache cache(2, 2, 2, 2048, 4);
  KVCache::View view{};
  float k[] = {1,2,3,4, 5,6,7,8}; // [2 heads, 2 tokens, 2 words]
  float v[] = {11,12,13,14, 15,16,17,18};
  assert(!cache.begin(1,2));
  assert(cache.begin(0,2));
  assert(cache.history(0,view) && view.length==0 && view.keys);
  assert(!cache.begin(0,2));
  assert(!cache.view(0,view));
  assert(cache.append(0,k,v));
  assert(!cache.append(0,k,v));
  assert(!cache.commit());
  assert(cache.view(0,view) && view.length==2);
  float expected[] = {1,2,5,6,3,4,7,8};
  for(int i=0;i<8;++i) assert(view.keys[i]==expected[i]);
  assert(cache.append(1,k,v) && cache.commit());
  assert(cache.length()==2 && cache.capacity()==4);
  auto *address=view.keys;
  assert(cache.begin(2,2));
  assert(cache.append(0,k,v) && cache.append(1,k,v) && cache.commit());
  assert(cache.view(0,view) && view.keys==address && view.length==4);
  assert(cache.relocated_bytes()==0 && cache.appended_bytes()==256);
  assert(cache.begin(4,2)); // Grow once; copy only the occupied prefix.
  assert(cache.append(0,k,v) && cache.append(1,k,v) && cache.commit());
  assert(cache.view(0,view) && view.length==6 && cache.capacity()==8);
  assert(cache.relocated_bytes()==256 && cache.appended_bytes()==384);
  for(int i=0;i<24;++i) assert(view.keys[i]==expected[i%8]);
  assert(!cache.begin(6,2043));
  assert(cache.begin(6,2) && cache.append(0,k,v));
  cache.cancel();
  assert(cache.length()==6 && cache.view(0,view) && view.length==6);
  cache.reset();
  assert(cache.length()==0 && cache.capacity()==8);
  assert(cache.begin(0,1));
  assert(!cache.view(1,view)); // Never expose stale data from the prior request.
  assert(cache.append(0,k,v) && cache.append(1,k,v) && cache.commit());
  assert(cache.view(1,view) && view.length==1);
  KVCache invalid(65,1,1,1);
  assert(!invalid.begin(0,1) && !invalid.commit());
  KVCache overflow(28,8,128,std::numeric_limits<size_t>::max());
  assert(!overflow.begin(0,1));
  KVCache edge(64,1,1,1);
  assert(edge.begin(0,1));
  for(size_t layer=0;layer<64;++layer) assert(edge.append(layer,k,v));
  assert(edge.commit());
  // A compiled function writes only the reserved tail, then acknowledges both
  // planes. No second append/copy is needed; earlier tokens remain intact.
  KVCache compiled(1,1,2,8,4);
  assert(!compiled.finish_layer(0));
  assert(compiled.begin(0,1) && compiled.append(0,k,v) && compiled.commit());
  assert(compiled.begin(1,1) && compiled.history(0,view));
  view.keys[2]=21;view.keys[3]=22;view.values[2]=31;view.values[3]=32;
  assert(!compiled.finish_layer(1) && !compiled.commit());
  assert(compiled.finish_layer(0) && !compiled.finish_layer(0));
  assert(!compiled.append(0,k,v) && compiled.commit());
  assert(compiled.view(0,view) && view.length==2);
  assert(view.keys[0]==1 && view.keys[1]==2 && view.keys[2]==21 && view.keys[3]==22);
  assert(view.values[0]==11 && view.values[1]==12 && view.values[2]==31 && view.values[3]==32);
  assert(compiled.appended_bytes()==32 && compiled.relocated_bytes()==0);
}
''')
    executable = tmp_path / 'test'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-I'+str(native),
                    str(source), str(native/'kv_cache.cpp'), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
