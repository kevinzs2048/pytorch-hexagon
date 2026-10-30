/* Read-only integer +0-bit guard. No floating point arithmetic/state changes.
   Every load stays inside the exact suffix; failure immediately falls back.
   Leading/trailing words avoid any unaligned vector overread. */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#ifdef __hexagon__
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#endif
struct m2 {void *allocated,*aligned;int64_t offset,sizes[2],strides[2];};
struct m5 {void *allocated,*aligned;int64_t offset,sizes[5],strides[5];};
static uint32_t word(const unsigned char *p){uint32_t v;memcpy(&v,__builtin_assume_aligned(p,4),4);return v;}
bool qwen_zero_words(const void *data,size_t bytes){
  const unsigned char *p=data;
  if(!p || bytes%4 || ((uintptr_t)p&3))return false;
#ifdef __hexagon__
  while(bytes && ((uintptr_t)p&127)){
    if(word(p))return false;p+=4;bytes-=4;
  }
  while(bytes>=512){
    const HVX_Vector *v=(const HVX_Vector *)p;
    HVX_Vector x=Q6_V_vor_VV(Q6_V_vor_VV(v[0],v[1]),Q6_V_vor_VV(v[2],v[3]));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,64));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,32));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,16));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,8));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,4));
    if(Q6_R_vextract_VR(x,0))return false;
    p+=512;bytes-=512;
  }
#endif
  while(bytes){if(word(p))return false;p+=4;bytes-=4;}
  return true;
}
static const unsigned char *base(void *ptr,int64_t offset,size_t words){
  uintptr_t start=(uintptr_t)ptr;
  if(!ptr || (start&3) || offset<0 || (uint64_t)offset>SIZE_MAX/4)return NULL;
  size_t shift=(size_t)offset*4;
  if(shift>UINTPTR_MAX-start || words>SIZE_MAX/4 || words*4>UINTPTR_MAX-start-shift)return NULL;
  return (const unsigned char *)(start+shift);
}
/* Conservative cost eligibility, separate from the exact numerical guard:
   near-full prefixes cannot amortize a scan; huge prefix operands can make
   every recurrence block overflow and fall back after paying for that scan.
   Rejecting eligibility always runs the complete original recurrence. */
static bool small_words(const unsigned char *p,size_t bytes){
#ifdef __hexagon__
  while(bytes && ((uintptr_t)p&127)){
    if((word(p)&0x7fffffffu)>0x4f800000u)return false;p+=4;bytes-=4;
  }
  HVX_Vector mask=Q6_V_vsplat_R(0x7fffffff),limit=Q6_V_vsplat_R(0x4f800000);
  while(bytes>=512){
    const HVX_Vector *vectors=(const HVX_Vector *)p;
    HVX_Vector b0=Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(vectors[0],mask),limit),-1);
    HVX_Vector b1=Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(vectors[1],mask),limit),-1);
    HVX_Vector b2=Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(vectors[2],mask),limit),-1);
    HVX_Vector b3=Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(vectors[3],mask),limit),-1);
    HVX_Vector x=Q6_V_vor_VV(Q6_V_vor_VV(b0,b1),Q6_V_vor_VV(b2,b3));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,64));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,32));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,16));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,8));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,4));
    if(Q6_R_vextract_VR(x,0))return false;
    p+=512;bytes-=512;
  }
  while(bytes>=128){
    HVX_Vector x=Q6_V_vand_VV(*(const HVX_Vector *)p,mask);
    x=Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(x,limit),-1);
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,64));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,32));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,16));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,8));
    x=Q6_V_vor_VV(x,Q6_V_valign_VVR(x,x,4));
    if(Q6_R_vextract_VR(x,0))return false;
    p+=128;bytes-=128;
  }
#endif
  while(bytes){if((word(p)&0x7fffffffu)>0x4f800000u)return false;p+=4;bytes-=4;}
  return true;
}
bool _mlir_ciface_qwen_hvx_positive_zero_suffix(struct m2 *p,struct m5 *v,int64_t end,bool broadcast){
  if(!p||!v||broadcast||end<0||end>=768
      ||p->sizes[0]!=16||p->sizes[1]!=1024||p->strides[0]!=1024||p->strides[1]!=1
      ||v->sizes[0]!=1||v->sizes[1]!=8||v->sizes[2]!=1||v->sizes[3]!=1024||v->sizes[4]!=128
      ||v->strides[1]!=1024*128||v->strides[3]!=128||v->strides[4]!=1)return false;
  const unsigned char *pb=base(p->aligned,p->offset,16*1024),*vb=base(v->aligned,v->offset,8*1024*128);
  if(!pb||!vb)return false;
  size_t suffix=(size_t)(1024-end);
  for(size_t h=0;h<16;h++)
    if(!qwen_zero_words(pb+4*(h*1024+(size_t)end),4*suffix))return false;
  for(size_t h=0;h<16;h++)
    if(!small_words(pb+4*h*1024,4*(size_t)end))return false;
  for(size_t h=0;h<8;h++)
    if(!small_words(vb+4*h*1024*128,4*(size_t)end*128))return false;
  for(size_t h=0;h<8;h++)
    if(!qwen_zero_words(vb+4*(h*1024+(size_t)end)*128,4*suffix*128))return false;
  return true;
}

/* Derive the occupied extent from actual P bits, never caller metadata.
   A nonzero bit pattern (including -0/NaN) extends the prefix. The exact V
   suffix and per-block finite-prefix checks still decide whether to skip. */
int64_t _mlir_ciface_qwen_hvx_positive_zero_extent(struct m2 *p,struct m5 *v,bool broadcast){
  if(!p||!v||broadcast||p->sizes[0]!=16||p->sizes[1]!=1024
      ||p->strides[0]!=1024||p->strides[1]!=1)return 1024;
  const unsigned char *pb=base(p->aligned,p->offset,16*1024);
  if(!pb)return 1024;
  int64_t end=0;
  for(int h=0;h<16;h++){
    int64_t k=1024;
    while(k>end){
      if(word(pb+4*(h*1024+k-1))){end=k;break;}
      --k;
    }
    if(end>=768)return 1024;
  }
  return _mlir_ciface_qwen_hvx_positive_zero_suffix(p,v,end,broadcast)?end:1024;
}
