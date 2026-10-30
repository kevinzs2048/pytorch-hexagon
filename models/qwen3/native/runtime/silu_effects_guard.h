// Scoped contract prerequisite; never modifies the original scalar helper.
#ifndef QWEN3_SILU_EFFECTS_GUARD_H
#define QWEN3_SILU_EFFECTS_GUARD_H
#include <cmath>
#include <cstring>
static inline bool qwen3_silu_effects_runtime_matches() {
  // Complete normal SDK FMA implementation, not the distinct :lib opcode.
  const unsigned char expected[]={0x82,0xc1,0x00,0xef,0xc0,0x3f,0x20,0x50};
  auto actual=reinterpret_cast<const unsigned char *>(std::fmaf);
  return std::memcmp(actual,expected,sizeof(expected))==0;
}
#endif
