/* Integer-only proof for the existing rank3 prefill QK score recurrence.
 * A returned prefix is a multiple of128. Beyond it, every key component
 * and every initial score is bitwise positive zero. Each query row must still
 * execute its original128 ordered FMAs for one representative tile; only the
 * resulting identical column bits may be copied. No literal zero replaces FP.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

struct q2_qk_m3 {void *allocated,*aligned;int64_t offset,sizes[3],strides[3];};
extern bool qwen_zero_words(const void *,size_t);

static const unsigned char *canonical(void *pointer,int64_t offset,size_t words) {
  uintptr_t address=(uintptr_t)pointer;
  if(!pointer || offset || (address&3) || words>SIZE_MAX/4
      || words*4>UINTPTR_MAX-address)return NULL;
  return (const unsigned char *)pointer;
}

int64_t _mlir_ciface_qwen_q2_qk_positive_zero_extent(struct q2_qk_m3 *keys,
    struct q2_qk_m3 *seed,bool broadcast) {
  if(!keys || !seed || broadcast || keys->sizes[0]!=16 || keys->sizes[1]!=128
      || keys->sizes[2]!=1024 || keys->strides[0]!=128*1024
      || keys->strides[1]!=1024 || keys->strides[2]!=1
      || seed->sizes[0]!=16 || seed->sizes[1]<2 || seed->sizes[1]>32
      || seed->sizes[2]!=1024 || seed->strides[0]!=seed->sizes[1]*1024
      || seed->strides[1]!=1024 || seed->strides[2]!=1)return 1024;
  size_t score_rows=(size_t)seed->sizes[0]*(size_t)seed->sizes[1];
  size_t key_bytes=16u*128u*1024u*4u,score_bytes=score_rows*1024u*4u;
  const unsigned char *k=canonical(keys->aligned,keys->offset,key_bytes/4);
  const unsigned char *s=canonical(seed->aligned,seed->offset,score_bytes/4);
  if(!k || !s)return 1024;
  uintptr_t ka=(uintptr_t)k,sa=(uintptr_t)s;
  if(ka<sa+score_bytes && sa<ka+key_bytes)return 1024;
  /* Invariant: every processed row is positive zero at columns >=prefix.
   * Increasing prefix preserves that invariant. Read each new row contiguously
   * once; only a row disproving the current suffix needs backward refinement.
   * This changes integer load order only, never the accepted bit predicate.
   */
  int64_t prefix=0;
  for(unsigned operand=0;operand<2;++operand) {
    const unsigned char *base=operand?k:s;
    size_t rows=operand?16u*128u:score_rows;
    for(size_t row=0;row<rows;++row) {
      const unsigned char *p=base+4*row*1024;
      if(qwen_zero_words(p+4*(size_t)prefix,4*(size_t)(1024-prefix)))continue;
      for(int64_t start=896;start>=prefix;start-=128) {
        if(!qwen_zero_words(p+4*(size_t)start,128*4)) {
          prefix=start+128;
          if(prefix>=768)return 1024;
          break;
        }
      }
    }
  }
  return prefix;
}
