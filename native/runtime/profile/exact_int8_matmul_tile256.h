/* Same signed I8 products and INT32 ring as the existing query1 runtime.
 * Two adjacent 128-column tiles share each activation load and K loop.
 * All strided layouts and remaining columns use the original runtime.
 */
#ifndef QWEN_EXACT_INT8_MATMUL_TILE256_H
#define QWEN_EXACT_INT8_MATMUL_TILE256_H
#include "exact_int8_matmul.h"

static inline int qwen_exact_int8_matmul_tile256(const int8_t *a,const int8_t *b,
    int32_t *c,size_t k,size_t n,ptrdiff_t a_stride,
    ptrdiff_t b_row_stride,ptrdiff_t b_column_stride,ptrdiff_t c_stride) {
  if (!a || !b || !c || !k || k>131071 || !n || n>151936) return 2;
  size_t column=0;
#if defined(__hexagon__)
  if (a_stride==1 && b_column_stride==1 && c_stride==1) {
    for (;n-column>=256;column+=256) {
      HVX_Vector acc[8];
      #pragma unroll
      for (unsigned group=0;group<8;++group)
        memcpy(&acc[group],c+column+32*group,128);
      size_t reduction=0;
      for (;k-reduction>=4;reduction+=4) {
        int32_t activation;memcpy(&activation,a+reduction,4);
        HVX_Vector va=Q6_V_vsplat_R(activation);
        #pragma unroll
        for (unsigned block=0;block<2;++block) {
          qwen_bytes128 packed[4];
          qwen_pack_four_rows(b+(ptrdiff_t)reduction*b_row_stride+(ptrdiff_t)column+128*block,
                             b_row_stride,packed);
          #pragma unroll
          for (unsigned group=0;group<4;++group) {
            HVX_Vector weight;memcpy(&weight,&packed[group],128);
            unsigned index=4*block+group;
            acc[index]=Q6_Vw_vrmpyacc_VwVbVb(acc[index],va,weight);
          }
        }
      }
      #pragma unroll
      for (unsigned group=0;group<8;++group)
        memcpy(c+column+32*group,&acc[group],128);
      for (;reduction<k;++reduction)
        for (unsigned tail=0;tail<256;++tail) {
          uint32_t sum;memcpy(&sum,c+column+tail,4);
          sum+=(uint32_t)((int32_t)a[reduction]*
              (int32_t)b[(ptrdiff_t)reduction*b_row_stride+(ptrdiff_t)column+tail]);
          memcpy(c+column+tail,&sum,4);
        }
    }
  }
#endif
  if (column==n) return 0;
  return qwen_exact_int8_matmul(a,b+(ptrdiff_t)column*b_column_stride,
      c+(ptrdiff_t)column*c_stride,k,n-column,a_stride,b_row_stride,b_column_stride,c_stride);
}
#endif
