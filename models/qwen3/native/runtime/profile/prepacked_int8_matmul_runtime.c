/* Explicit integer-only ABI for the existing bounded multi-token graph. */
#include <stdint.h>
#include <stdlib.h>
#include "exact_int8_prepacked.h"

#ifdef HEXAGON_PREPACK_LLAMA_WORKERS
#include "worker-pool.h"
#include "prepacked_dma.h"
#include <qurt_memory.h>
#include <stdatomic.h>
static worker_pool_context_t prepacked_pool;
static atomic_flag prepacked_busy=ATOMIC_FLAG_INIT;
struct prepacked_job {
  const int8_t *a,*b;int32_t *c;size_t m,k,n;ptrdiff_t ars,crs;int dma;
};
__attribute__((destructor)) static void release_prepacked_pool(void) {
  if(prepacked_pool)worker_pool_release(&prepacked_pool);
  qwen_dma_destroy();
}
static void prepacked_worker(unsigned workers,unsigned id,void *data) {
  const struct prepacked_job *j=data;
  size_t tiles=j->n/128,start=tiles*id/workers,end=tiles*(id+1)/workers;
  if(start==end)return;
  if(j->dma) {
    dma_queue *queue=&qwen_dma_workers[id].queue;
    unsigned char *workspace=qwen_dma_storage+id*2*QWEN_DMA_TILE_BYTES;
    size_t pushed=start;
    for(unsigned slot=0;slot<2 && pushed<end;++slot,++pushed)
      if(!dma_queue_push_single_1d(queue,dma_make_ptr(
          workspace+slot*QWEN_DMA_TILE_BYTES,j->b+pushed*128*j->k),128*j->k))abort();
    for(size_t tile=start;tile<end;++tile) {
      dma_ptr ready=dma_queue_pop(queue);
      if(!ready.dst || hexagon_i8_prepacked_v1_with_prefetch(j->a,ready.dst,
          j->c+tile*128,j->m,j->k,128,j->ars,j->crs,0))abort();
      if(pushed<end) {
        if(!dma_queue_push_single_1d(queue,dma_make_ptr(
            ready.dst,j->b+pushed*128*j->k),128*j->k))abort();
        ++pushed;
      }
    }
    return;
  }
  if(hexagon_i8_prepacked_v1(j->a,j->b+start*128*j->k,j->c+start*128,
      j->m,j->k,(end-start)*128,j->ars,j->crs))abort();
}
static int prepacked_parallel(const int8_t*a,const int8_t*b,int32_t*c,
    size_t m,size_t k,size_t n,ptrdiff_t ars,ptrdiff_t crs) {
  if (!a || !b || !c || !m || m>32 || !k || k>131071 || k%4 ||
      !n || n>151936 || n%128 || k>SIZE_MAX/n ||
      ars<(ptrdiff_t)k || crs<(ptrdiff_t)n) return 2;
  // Small matrices cannot amortize worker dispatch; preserve the serial path.
  if(k*n<65536 || n<512)return hexagon_i8_prepacked_v1(a,b,c,m,k,n,ars,crs);
  if(atomic_flag_test_and_set(&prepacked_busy))return 3;
  int status=0,dma=0;
  if(!prepacked_pool && worker_pool_init(&prepacked_pool,4)){status=4;goto done;}
  if(qurt_mem_cache_clean((uintptr_t)a,(m-1)*ars+k,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE)||
     qurt_mem_cache_clean((uintptr_t)c,((m-1)*crs+n)*4,QURT_MEM_CACHE_FLUSH_INVALIDATE,QURT_MEM_DCACHE)) {
    status=5;goto done;
  }
  dma=qwen_dma_acquire(k);
  struct prepacked_job job={a,b,c,m,k,n,ars,crs,dma};
  if(worker_pool_run_func(prepacked_pool,prepacked_worker,&job,4)){status=6;goto done;}
  if(qurt_mem_cache_clean((uintptr_t)c,((m-1)*crs+n)*4,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE))status=5;
done:
  if(dma)qwen_dma_release();
  atomic_flag_clear(&prepacked_busy);return status;
}
#define PREPACKED_EXECUTE prepacked_parallel
#else
#define PREPACKED_EXECUTE hexagon_i8_prepacked_v1
#endif

struct qwen_multi_i8_memref2 { int8_t *allocated,*aligned; int64_t offset,sizes[2],strides[2]; };
struct qwen_multi_i32_memref2 { int32_t *allocated,*aligned; int64_t offset,sizes[2],strides[2]; };

static int valid_span(int64_t offset,int64_t rows,int64_t columns,const int64_t strides[2]) {
  if (offset<INT32_MIN || offset>INT32_MAX || rows<1 || columns<1
      || strides[0]<INT32_MIN || strides[0]>INT32_MAX
      || strides[1]<INT32_MIN || strides[1]>INT32_MAX) return 0;
  /* All dimension bounds are checked first. Products fit signed int64 even
   * on native DSP32; their final pointer-index range is checked separately.
   */
  int64_t row=(rows-1)*strides[0],column=(columns-1)*strides[1];
  int64_t lo=offset+(row<0?row:0)+(column<0?column:0);
  int64_t hi=offset+(row>0?row:0)+(column>0?column:0);
  return lo>=INT32_MIN && hi<=INT32_MAX;
}


void _mlir_ciface___hexagon_i8_prepacked_v1_matmul(struct qwen_multi_i8_memref2 *a,
    struct qwen_multi_i8_memref2 *b,struct qwen_multi_i32_memref2 *c) {
  if (!a || !b || !c || !a->aligned || !b->aligned || !c->aligned
      || a->sizes[0]<1 || a->sizes[0]>32 || c->sizes[0]!=a->sizes[0]
      || a->sizes[1]<1 || a->sizes[1]>131071 || a->sizes[1]!=b->sizes[0]
      || c->sizes[1]<1 || c->sizes[1]>151936 || b->sizes[1]!=c->sizes[1]
      || !valid_span(a->offset,a->sizes[0],a->sizes[1],a->strides)
      || !valid_span(b->offset,b->sizes[0],b->sizes[1],b->strides)
      || !valid_span(c->offset,c->sizes[0],c->sizes[1],c->strides)) abort();
  if (a->strides[1]!=1 || a->strides[0]<a->sizes[1] || b->strides[1]!=1 ||
      b->strides[0]!=b->sizes[1] || c->strides[1]!=1 ||
      c->strides[0]<c->sizes[1]) abort();
  if (PREPACKED_EXECUTE(a->aligned+a->offset,b->aligned+b->offset,
      c->aligned+c->offset,(size_t)a->sizes[0],(size_t)a->sizes[1],(size_t)c->sizes[1],
      (ptrdiff_t)a->strides[0],(ptrdiff_t)c->strides[0])) abort();
}
