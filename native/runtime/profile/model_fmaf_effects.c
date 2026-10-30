/* Diagnostic query in a relink of the unchanged accepted model object.
 * Inspect the same imported fmaf symbol used by the original scalar helper.
 * No replacement, interposition, clock vote, or model arithmetic is added.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <dlfcn.h>

static void copy_name(unsigned char *out, size_t capacity, const char *name) {
  if (!name) return;
  size_t n = 0;
  while (n + 1 < capacity && name[n]) ++n;
  memcpy(out, name, n);
}

int qwen_fmaf_effects_query(const unsigned char *input, int input_len,
    unsigned char *output, int output_len, uint64_t *kernel_us) {
  if (!input || input_len < 8 || memcmp(input,"QWFMA01\0",8)) return -1;
  if (input_len < 16 || !output || !kernel_us) return 2;
  uint32_t count, reserved;
  memcpy(&count,input+8,4);memcpy(&reserved,input+12,4);
  if (!count || count > 8192 || reserved || input_len != (int)(16+count*12)
      || output_len != (int)(256+count*4)) return 2;
  memset(output,0,(size_t)output_len);*kernel_us=0;
  memcpy(output,"QWFMA01\0",8);
  uint32_t version=1;
  memcpy(output+8,&version,4);memcpy(output+12,&count,4);
  float (*callee)(float,float,float)=fmaf;
  void *resolved=dlsym(RTLD_DEFAULT,"fmaf");
  Dl_info info;memset(&info,0,sizeof(info));
  int found=dladdr((const void *)(uintptr_t)callee,&info);
  uint32_t address=(uint32_t)(uintptr_t)callee;
  uint32_t default_address=(uint32_t)(uintptr_t)resolved;
  uint32_t symbol_address=(uint32_t)(uintptr_t)info.dli_saddr;
  memcpy(output+16,&address,4);memcpy(output+20,&default_address,4);
  memcpy(output+24,&found,4);memcpy(output+28,&symbol_address,4);
  /* The eight bytes include the complete pinned SDK two-instruction fmaf.
   * They establish identity only if the actual bytes match that whole object.
   */
  memcpy(output+32,(const void *)(uintptr_t)callee,8);
  if (found) {
    copy_name(output+48,160,info.dli_fname);
    copy_name(output+208,32,info.dli_sname);
  }
  int saved_errno=errno;
  errno=0x1234;
  int before=errno;
  for (uint32_t i=0;i<count;++i) {
    float a,b,c;
    memcpy(&a,input+16+i*12,4);memcpy(&b,input+20+i*12,4);
    memcpy(&c,input+24+i*12,4);
    float result=callee(a,b,c);
    memcpy(output+256+i*4,&result,4);
  }
  int after=errno;
  errno=saved_errno;
  memcpy(output+40,&before,4);memcpy(output+44,&after,4);
  return 0;
}
