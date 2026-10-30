/* Existing query1..32 integer contractions, with shared weight loads.
 * Each dot is an exact signed-I8 product sum modulo 2^32. Reordering these
 * integer additions is valid even for arbitrary initial C; no FP operation
 * or quantization step is part of this kernel. The original one-row kernel
 * and bounded scalar tails remain the fallback for every strided layout.
 */
#ifndef QWEN_EXACT_INT8_MULTI_MATMUL_H
#define QWEN_EXACT_INT8_MULTI_MATMUL_H
#include "exact_int8_matmul.h"

#if defined(__hexagon__)
static inline __attribute__((always_inline)) void qwen_multi_four_rows(const int8_t *a,const int8_t *b,
    int32_t *c,size_t rows,size_t k,size_t n,ptrdiff_t a_row_stride,
    ptrdiff_t b_row_stride,ptrdiff_t c_row_stride) {
  for (size_t column=0;column<n;column+=128) {
    HVX_Vector acc[4][4];
    #pragma unroll
    for (unsigned row=0;row<rows;++row)
      #pragma unroll
      for (unsigned group=0;group<4;++group)
        memcpy(&acc[row][group],c+(ptrdiff_t)row*c_row_stride+column+32*group,128);
    size_t reduction=0;
    for (;k-reduction>=4;reduction+=4) {
      qwen_bytes128 packed[4];
      qwen_pack_four_rows(b+(ptrdiff_t)reduction*b_row_stride+(ptrdiff_t)column,
                         b_row_stride,packed);
      #pragma unroll
      for (unsigned row=0;row<rows;++row) {
        int32_t activation;
        memcpy(&activation,a+(ptrdiff_t)row*a_row_stride+reduction,4);
        HVX_Vector va=Q6_V_vsplat_R(activation);
        #pragma unroll
        for (unsigned group=0;group<4;++group) {
          HVX_Vector weights;
          memcpy(&weights,&packed[group],128);
          acc[row][group]=Q6_Vw_vrmpyacc_VwVbVb(acc[row][group],va,weights);
        }
      }
    }
    #pragma unroll
    for (unsigned row=0;row<rows;++row) {
      #pragma unroll
      for (unsigned group=0;group<4;++group)
        memcpy(c+(ptrdiff_t)row*c_row_stride+column+32*group,&acc[row][group],128);
      for (size_t tail=reduction;tail<k;++tail)
        for (unsigned offset=0;offset<128;++offset) {
          int32_t *slot=c+(ptrdiff_t)row*c_row_stride+column+offset;
          uint32_t sum;memcpy(&sum,slot,4);
          sum+=(uint32_t)((int32_t)a[(ptrdiff_t)row*a_row_stride+(ptrdiff_t)tail]
              *(int32_t)b[(ptrdiff_t)tail*b_row_stride+(ptrdiff_t)column+(ptrdiff_t)offset]);
          memcpy(slot,&sum,4);
        }
    }
  }
}
#endif

static inline int qwen_exact_int8_multi_matmul(const int8_t *a,const int8_t *b,
    int32_t *c,size_t m,size_t k,size_t n,ptrdiff_t a_row_stride,
    ptrdiff_t a_column_stride,ptrdiff_t b_row_stride,
    ptrdiff_t b_column_stride,ptrdiff_t c_row_stride,ptrdiff_t c_column_stride) {
  if (!a || !b || !c || !m || m>32 || !k || k>131071 || !n || n>151936) return 2;
  size_t row=0;
#if defined(__hexagon__)
  /* A complete 128-column tile prevents speculative/out-of-bounds loads.
   * Dynamic row tails dispatch 4, then 2, then the original one-row kernel.
   */
  if (a_column_stride==1 && b_column_stride==1 && c_column_stride==1 && n%128==0
      && (c_row_stride>=(ptrdiff_t)n || c_row_stride<=-(ptrdiff_t)n)) {
    for (;m-row>=4;row+=4)
      qwen_multi_four_rows(a+(ptrdiff_t)row*a_row_stride,b,c+(ptrdiff_t)row*c_row_stride,
          4,k,n,a_row_stride,b_row_stride,c_row_stride);
    if (m-row>=2) {
      qwen_multi_four_rows(a+(ptrdiff_t)row*a_row_stride,b,c+(ptrdiff_t)row*c_row_stride,
          2,k,n,a_row_stride,b_row_stride,c_row_stride);
      row+=2;
    }
  }
#endif
  for (;row<m;++row) {
    int result=qwen_exact_int8_matmul(a+(ptrdiff_t)row*a_row_stride,b,
        c+(ptrdiff_t)row*c_row_stride,k,n,a_column_stride,
        b_row_stride,b_column_stride,c_column_stride);
    if (result) return result;
  }
  return 0;
}
#endif
