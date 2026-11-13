// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_EXACT_INT8_PREPACKED_H
#define HEXAGON_EXACT_INT8_PREPACKED_H
#include "exact_int8_matmul.h"
/* Version 1: [N/128][K/4][128 output columns][4 reduction bytes].
 * Byte permutation only; scales and integer values are unchanged.
 * Caller owns immutable packed storage for the weight's lifetime.
 */
#if defined(__hexagon__)
static inline __attribute__((always_inline)) void hexagon_prepacked_four_rows(const int8_t *a,const int8_t *b,
    int32_t *c,size_t rows,size_t k,size_t n,ptrdiff_t a_row_stride,
    ptrdiff_t c_row_stride,int aligned,int prefetch) {
  for (size_t column=0;column<n;column+=128) {
    HVX_Vector acc[4][4];
    #pragma unroll
    for (unsigned row=0;row<rows;++row)
      #pragma unroll
      for (unsigned group=0;group<4;++group)
        memcpy(&acc[row][group],c+(ptrdiff_t)row*c_row_stride+column+32*group,128);
    size_t reduction=0;
    for (;k-reduction>=4;reduction+=4) {
      const int8_t *packed=b+(column/128)*k*128+(reduction/4)*512;
      // Single-row groups prefetch a bounded 16 KiB tile ahead.
      // Only immutable packed bytes within this column tile are touched.
      if (prefetch && rows == 1 && (reduction % 128) == 0 && k - reduction > 128) {
        size_t next = reduction + 128, count = k - next;
        if (count > 128) count = 128;
        uint64_t control = Q6_P_combine_RR(512, Q6_R_combine_RlRl(512, count / 4));
        Q6_l2fetch_AP((void *)(b + (column/128)*k*128 + (next/4)*512), control);
      }
      #pragma unroll
      for (unsigned row=0;row<rows;++row) {
        int32_t activation;
        const int8_t *row_input=a+(ptrdiff_t)row*a_row_stride+reduction;
        if(aligned)row_input=(const int8_t*)__builtin_assume_aligned(row_input,4);
        memcpy(&activation,row_input,4);
        HVX_Vector va=Q6_V_vsplat_R(activation);
        #pragma unroll
        for (unsigned group=0;group<4;++group) {
          HVX_Vector weights;
          const int8_t *weight_input=packed+group*128;
          if(aligned)weight_input=(const int8_t*)__builtin_assume_aligned(weight_input,128);
          memcpy(&weights,weight_input,128);
          acc[row][group]=Q6_Vw_vrmpyacc_VwVbVb(acc[row][group],va,weights);
        }
      }
    }
    #pragma unroll
    for (unsigned row=0;row<rows;++row) {
      #pragma unroll
      for (unsigned group=0;group<4;++group)
        memcpy(c+(ptrdiff_t)row*c_row_stride+column+32*group,&acc[row][group],128);

    }
  }
}
#endif
static int hexagon_i8_prepacked_v1_with_prefetch(const int8_t*a,const int8_t*b,int32_t*c,size_t m,size_t k,size_t n,ptrdiff_t ars,ptrdiff_t crs,int prefetch) {
 if(!a||!b||!c||!m||m>32||!k||k>131071||k%4||!n||n>151936||n%128||k>SIZE_MAX/n||ars<(ptrdiff_t)k||crs<(ptrdiff_t)n)return 2;
#if defined(__hexagon__)
 // Check before assuming alignment; arbitrary valid strides keep the fallback.
 if(((uintptr_t)a%4)==0&&ars%4==0&&((uintptr_t)b%128)==0) {
  // Reuse one weight tile across query groups before advancing columns.
  if(m>4) {
  // A constant row count lets the compiler pipeline the reduction without
  // testing the same tail predicates on every K iteration. Keep the partial
  // group separate; neither packed bytes nor integer arithmetic changes.
  for(size_t column=0;column<n;column+=128) {
   size_t r=0;
   for(;r+4<=m;r+=4)
    hexagon_prepacked_four_rows(a+r*ars,b+column*k,c+r*crs+column,4,k,128,ars,crs,1,prefetch);
   if(r<m)hexagon_prepacked_four_rows(a+r*ars,b+column*k,c+r*crs+column,m-r,k,128,ars,crs,1,prefetch);
  }
 } else { for(size_t r=0;r<m;r+=4)hexagon_prepacked_four_rows(a+r*ars,b,c+r*crs,m-r<4?m-r:4,k,n,ars,crs,1,prefetch); }
 } else {
  if(m>4) {
  for(size_t column=0;column<n;column+=128)
   for(size_t r=0;r<m;r+=4)
    hexagon_prepacked_four_rows(a+r*ars,b+column*k,c+r*crs+column,m-r<4?m-r:4,k,128,ars,crs,0,prefetch);
 } else { for(size_t r=0;r<m;r+=4)hexagon_prepacked_four_rows(a+r*ars,b,c+r*crs,m-r<4?m-r:4,k,n,ars,crs,0,prefetch); }
 }
#else
 (void)prefetch;
 for(size_t r=0;r<m;++r)for(size_t col=0;col<n;++col){
  uint32_t sum;memcpy(&sum,c+r*crs+col,4);
  for(size_t red=0;red<k;++red)sum+=(uint32_t)((int32_t)a[r*ars+red]*(int32_t)b[(col/128)*k*128+(red/4)*512+(col%128)*4+red%4]);
  memcpy(c+r*crs+col,&sum,4);
 }
#endif
 return 0;
}

static int hexagon_i8_prepacked_v1(const int8_t*a,const int8_t*b,int32_t*c,
    size_t m,size_t k,size_t n,ptrdiff_t ars,ptrdiff_t crs) {
 return hexagon_i8_prepacked_v1_with_prefetch(a,b,c,m,k,n,ars,crs,1);
}

#endif
