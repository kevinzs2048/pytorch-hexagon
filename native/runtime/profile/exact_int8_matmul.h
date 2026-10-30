/* Exact signed I8 matrix multiply, C += A*B in the INT32 ring.
 * A has one row, B is [K,N]. Preserve weight layout and all FP operations.
 * |integer dot| <= 16384*K for K<=131071. Initial C is unrestricted: updates
 * use defined unsigned arithmetic to preserve Linalg wraparound semantics.
 */
#ifndef QWEN_EXACT_INT8_MATMUL_H
#define QWEN_EXACT_INT8_MATMUL_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#endif
typedef signed char qwen_bytes128 __attribute__((vector_size(128)));
#define QWEN_BYTES_LOW 0,128,1,129,2,130,3,131,4,132,5,133,6,134,7,135,8,136,9,137,10,138,11,139,12,140,13,141,14,142,15,143,16,144,17,145,18,146,19,147,20,148,21,149,22,150,23,151,24,152,25,153,26,154,27,155,28,156,29,157,30,158,31,159,32,160,33,161,34,162,35,163,36,164,37,165,38,166,39,167,40,168,41,169,42,170,43,171,44,172,45,173,46,174,47,175,48,176,49,177,50,178,51,179,52,180,53,181,54,182,55,183,56,184,57,185,58,186,59,187,60,188,61,189,62,190,63,191
#define QWEN_BYTES_HIGH 64,192,65,193,66,194,67,195,68,196,69,197,70,198,71,199,72,200,73,201,74,202,75,203,76,204,77,205,78,206,79,207,80,208,81,209,82,210,83,211,84,212,85,213,86,214,87,215,88,216,89,217,90,218,91,219,92,220,93,221,94,222,95,223,96,224,97,225,98,226,99,227,100,228,101,229,102,230,103,231,104,232,105,233,106,234,107,235,108,236,109,237,110,238,111,239,112,240,113,241,114,242,115,243,116,244,117,245,118,246,119,247,120,248,121,249,122,250,123,251,124,252,125,253,126,254,127,255
#define QWEN_PAIRS_LOW 0,1,128,129,2,3,130,131,4,5,132,133,6,7,134,135,8,9,136,137,10,11,138,139,12,13,140,141,14,15,142,143,16,17,144,145,18,19,146,147,20,21,148,149,22,23,150,151,24,25,152,153,26,27,154,155,28,29,156,157,30,31,158,159,32,33,160,161,34,35,162,163,36,37,164,165,38,39,166,167,40,41,168,169,42,43,170,171,44,45,172,173,46,47,174,175,48,49,176,177,50,51,178,179,52,53,180,181,54,55,182,183,56,57,184,185,58,59,186,187,60,61,188,189,62,63,190,191
#define QWEN_PAIRS_HIGH 64,65,192,193,66,67,194,195,68,69,196,197,70,71,198,199,72,73,200,201,74,75,202,203,76,77,204,205,78,79,206,207,80,81,208,209,82,83,210,211,84,85,212,213,86,87,214,215,88,89,216,217,90,91,218,219,92,93,220,221,94,95,222,223,96,97,224,225,98,99,226,227,100,101,228,229,102,103,230,231,104,105,232,233,106,107,234,235,108,109,236,237,110,111,238,239,112,113,240,241,114,115,242,243,116,117,244,245,118,119,246,247,120,121,248,249,122,123,250,251,124,125,252,253,126,127,254,255

static inline void qwen_pack_four_rows(const int8_t *b, ptrdiff_t stride,
                                       qwen_bytes128 packed[4]) {
  qwen_bytes128 r0,r1,r2,r3;
  /* Each bounded load remains valid for an unaligned base/row stride. */
  memcpy(&r0,b,128);memcpy(&r1,b+stride,128);
  memcpy(&r2,b+2*stride,128);memcpy(&r3,b+3*stride,128);
  qwen_bytes128 lo01=__builtin_shufflevector(r0,r1,QWEN_BYTES_LOW);
  qwen_bytes128 hi01=__builtin_shufflevector(r0,r1,QWEN_BYTES_HIGH);
  qwen_bytes128 lo23=__builtin_shufflevector(r2,r3,QWEN_BYTES_LOW);
  qwen_bytes128 hi23=__builtin_shufflevector(r2,r3,QWEN_BYTES_HIGH);
  qwen_bytes128 w0=__builtin_shufflevector(lo01,lo23,QWEN_PAIRS_LOW);
  qwen_bytes128 w1=__builtin_shufflevector(lo01,lo23,QWEN_PAIRS_HIGH);
  qwen_bytes128 w2=__builtin_shufflevector(hi01,hi23,QWEN_PAIRS_LOW);
  qwen_bytes128 w3=__builtin_shufflevector(hi01,hi23,QWEN_PAIRS_HIGH);
  memcpy(&packed[0],&w0,128);memcpy(&packed[1],&w1,128);
  memcpy(&packed[2],&w2,128);memcpy(&packed[3],&w3,128);
}
#undef QWEN_BYTES_LOW
#undef QWEN_BYTES_HIGH
#undef QWEN_PAIRS_LOW
#undef QWEN_PAIRS_HIGH

static inline int qwen_exact_int8_matmul(const int8_t *a,const int8_t *b,
    int32_t *c,size_t k,size_t n,ptrdiff_t a_stride,
    ptrdiff_t b_row_stride,ptrdiff_t b_column_stride,ptrdiff_t c_stride) {
  if (!a || !b || !c || !k || k>131071 || !n || n>151936) return 2;
  size_t column=0;
#if defined(__hexagon__)
  if (a_stride==1 && b_column_stride==1 && c_stride==1) {
    for (;n-column>=128;column+=128) {
      HVX_Vector acc[4];
      for (unsigned group=0;group<4;++group)
        memcpy(&acc[group],c+column+32*group,128);
      size_t reduction=0;
      for (;k-reduction>=4;reduction+=4) {
        int32_t activation;
        memcpy(&activation,a+reduction,4);
        HVX_Vector va=Q6_V_vsplat_R(activation);
        qwen_bytes128 weights[4];
        qwen_pack_four_rows(b+(ptrdiff_t)reduction*b_row_stride+(ptrdiff_t)column,
                           b_row_stride,weights);
        for (unsigned group=0;group<4;++group) {
          HVX_Vector weight;
          memcpy(&weight,&weights[group],128);
          acc[group]=Q6_Vw_vrmpyacc_VwVbVb(acc[group],va,weight);
        }
      }
      for (unsigned group=0;group<4;++group)
        memcpy(c+column+32*group,&acc[group],128);
      for (;reduction<k;++reduction) {
        for (unsigned tail=0;tail<128;++tail) {
          uint32_t sum;
          memcpy(&sum,c+column+tail,4);
          sum+=(uint32_t)((int32_t)a[reduction]*(int32_t)b[(ptrdiff_t)reduction*b_row_stride+(ptrdiff_t)column+tail]);
          memcpy(c+column+tail,&sum,4);
        }
      }
    }
  }
#endif
  for (;column<n;++column) {
    uint32_t sum;
    memcpy(&sum,c+(ptrdiff_t)column*c_stride,4);
    for (size_t reduction=0;reduction<k;++reduction)
      sum+=(uint32_t)((int32_t)a[(ptrdiff_t)reduction*a_stride]*
        (int32_t)b[(ptrdiff_t)reduction*b_row_stride+(ptrdiff_t)column*b_column_stride]);
    memcpy(c+(ptrdiff_t)column*c_stride,&sum,4);
  }
  return 0;
}
#endif
