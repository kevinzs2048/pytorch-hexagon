// Four-lane ARM-compatible sin/cos entry points used by the DSP RoPE rewrite.
#define ENABLE_PUREC
#include "sleefsimdsp.c"

float __hexagon_arm_sin4_lane(float a,float b,float c,float d,int lane) {vfloat x; x.f[0]=a;x.f[1]=b;x.f[2]=c;x.f[3]=d;return xsinf_u1(x).f[lane];}
float __hexagon_arm_cos4_lane(float a,float b,float c,float d,int lane) {vfloat x; x.f[0]=a;x.f[1]=b;x.f[2]=c;x.f[3]=d;return xcosf_u1(x).f[lane];}
float _mlir_ciface___hexagon_arm_sin4_lane(float a,float b,float c,float d,int lane) {return __hexagon_arm_sin4_lane(a,b,c,d,lane);}
float _mlir_ciface___hexagon_arm_cos4_lane(float a,float b,float c,float d,int lane) {return __hexagon_arm_cos4_lane(a,b,c,d,lane);}
