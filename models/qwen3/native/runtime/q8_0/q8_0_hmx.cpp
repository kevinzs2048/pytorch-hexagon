// SPDX-License-Identifier: BSD-3-Clause
// Q8_0 x FP16 matrix multiplication on HMX, the arithmetic of llama.cpp's
// Hexagon hmx-tiled path (f818065): Q8_0 weights (32 values per block, FP16
// scale) are dequantized to FP16 on HVX, FP32 activations are rounded to FP16,
// HMX accumulates all of K over 32x32 FP16 tiles and its accumulator is read
// back as FP16 (then FP32, plus the fused residual or SwiGLU product of the
// graph). Conversions here are IEEE round-to-nearest-even; llama.cpp's v73
// qf16/qf32 sequences can differ by one FP16 ULP.
//
// Pipeline (as llama.cpp overlaps HVX dequantization with its HMX queue):
// the caller leases a prepared VTCM slot for the activation, which all four
// pool jobs pack in parallel (FP32 -> FP16 tiles), then submits one
// HMX job that consumes 32-column weight panels in order. Each panel is split
// along K into chunks of at most 32 tiles (64 KB of FP16), so every producer
// slot holds three chunk buffers whatever K is; three pool workers each hold
// one prepared slot and dequantize chunks g = w (mod 3) ahead of the HMX
// thread (counter handshake per worker). When the row-parallel pool is not
// available (busy or failed to start) the job runs once on the caller: it then
// packs everything itself and the HMX thread dequantizes each panel inline.
//
// Single-row projections (decode) and the LM head instead follow llama.cpp's
// CPU Q8_0 x Q8_0 semantics (ggml_vec_dot_q8_0_q8_0, see matvec_q8 below):
// the row is quantized per 32 values and dotted on HVX vrmpy, 32 outputs per
// vector, panels streamed into VTCM by UDMA and split over the pool.
//
// Weight panels are prepacked offline (models/qwen3/q8_0/quantize.py) in HMX tile
// order: for panel nt and reduction tile kt, 1024 int8 in row-pair order
// (pair p, column c, parity r -> q[n=c][k=2p+r]) then 64 FP16 scales expanded
// to the same lanes (lane 2c+r -> d[c]).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>
#include <qurt_hvx.h>
#ifndef restrict
#define restrict __restrict
#endif
#include "hexkl_micro.h"

extern "C" {
void hexagon_runtime_dispatch_hmx_dsp(void (*)(void *), void *);
void *hexagon_runtime_acquire_hmx_workspace_dsp(size_t, uint64_t, bool);
void hexagon_runtime_release_hmx_workspace_dsp(void *);
void *hexagon_runtime_acquire_prepared_hmx_workspace_dsp(size_t, uint64_t, bool);
void hexagon_runtime_release_prepared_hmx_workspace_dsp(void *);
int hexagon_runtime_configure_vtcm_pool_dsp(unsigned);
int hexagon_runtime_configure_hmx_performance_dsp(unsigned);
void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t), void *, uint64_t);
// Generated weight object: panel-major packed Q8_0 tiles of weight `id`.
const uint8_t *qwen_q8_0_weight(int64_t id, uint32_t *n, uint32_t *k);
}

struct MemF32 {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

namespace {
constexpr unsigned Tile = 2048;              // 32 x 32 FP16
constexpr unsigned PackedTile = 1024 + 128;  // int8 values + expanded FP16 scales
constexpr unsigned MaxK = 3072;
constexpr unsigned SlotBytes = 192 * 1024;   // largest prepared HMX lease
constexpr unsigned Producers = 3;

struct alignas(64) Producer {
  std::atomic<uint8_t *> slot{nullptr};
  std::atomic<unsigned> produced{0}, consumed{0};
};

struct Call {
  const float *x; int64_t x_stride;
  const float *side; int64_t side_stride;  // residual or factor, may be null
  bool product;
  float *out; int64_t out_stride;
  const uint8_t *weight;
  unsigned m, n, k, tiles, panels, buffers, chunk, chunks;  // chunk tiles, chunks per panel
  uint8_t *activation;
  unsigned owner;
  bool serial;  // no pool workers: dequantize on the HMX thread
  std::atomic<unsigned> packed{0};  // pool jobs done packing their activation tiles
  Producer producers[Producers];
};

inline void spin() { __asm__ __volatile__("pause(#32)" ::: "memory"); }

constexpr unsigned Jobs = Producers + 1;

// Activation tiles kt = part (mod Jobs).
void pack_activation(const Call &c, unsigned part) {
  for (unsigned kt = part; kt < c.tiles; kt += Jobs) {
    auto *tile = reinterpret_cast<HVX_Vector *>(c.activation + kt * Tile);
    for (unsigned p = 0; p < 16; ++p) {
      unsigned r0 = 2 * p, r1 = r0 + 1;
      HVX_Vector a = r0 < c.m ? *reinterpret_cast<const HVX_UVector *>(c.x + r0 * c.x_stride + kt * 32) : Q6_V_vzero();
      HVX_Vector b = r1 < c.m ? *reinterpret_cast<const HVX_UVector *>(c.x + r1 * c.x_stride + kt * 32) : Q6_V_vzero();
      tile[p] = Q6_Vhf_vcvt_VsfVsf(a, b);  // row-pair interleave of rows r0, r1
    }
  }
}

inline void prefetch_tiles(const uint8_t *panel, unsigned tiles) {
  // l2fetch control: stride (bytes) << 32 | width (bytes) << 16 | rows.
  uint64_t control = (uint64_t(PackedTile) << 32) | (uint64_t(PackedTile) << 16) | tiles;
  Q6_l2fetch_AP(const_cast<uint8_t *>(panel), control);
}

void dequantize_tiles(const uint8_t *panel, unsigned tiles, uint8_t *buffer) {
  for (unsigned kt = 0; kt < tiles; ++kt) {
    const uint8_t *src = panel + kt * PackedTile;
    HVX_Vector scale = *reinterpret_cast<const HVX_Vector *>(src + 1024);
    auto *dst = reinterpret_cast<HVX_Vector *>(buffer + kt * Tile);
    for (unsigned i = 0; i < 8; ++i) {
      HVX_VectorPair h = Q6_Wh_vunpack_Vb(reinterpret_cast<const HVX_Vector *>(src)[i]);
      dst[2 * i] = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vcvt_Vh(Q6_V_lo_W(h)), scale);
      dst[2 * i + 1] = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vcvt_Vh(Q6_V_hi_W(h)), scale);
    }
  }
}

// HMX owner thread: consume every panel in order.
void consume(void *opaque) {
  auto &c = *static_cast<Call *>(opaque);
  struct Hvx {  // the HMX owner holds no HVX context by default
    bool locked = false;
    ~Hvx() { if (locked) qurt_hvx_unlock(); }
  } hvx;
  if (qurt_hvx_get_lock_val() != QURT_HVX_LOCKED) {
    if (qurt_hvx_lock(QURT_HVX_MODE_128B)) std::abort();
    hvx.locked = true;
  }
  unsigned config = (hexkl_micro_hmx_config_size() + 2047) & ~2047u;
  const unsigned chunk_bytes = c.chunk * Tile;
  // Serial mode: one weight buffer per chunk of a panel after acc and config.
  auto *scratch = static_cast<uint8_t *>(hexagon_runtime_acquire_hmx_workspace_dsp(
      2 * Tile + config + (c.serial ? c.chunks * chunk_bytes : 0), 128, true));
  if (!scratch) std::abort();
  uint8_t *accumulator = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(scratch) + 2047) & ~uintptr_t(2047));
  uint8_t *inline_weights = accumulator + Tile + config;
  uint8_t *slots[Producers] = {};
  if (!c.serial)
    for (unsigned w = 0; w < Producers; ++w) slots[w] = c.producers[w].slot.load(std::memory_order_acquire);
  uint8_t *low = accumulator;
  if (c.activation < low) low = c.activation;
  for (auto *s : slots) if (s && s < low) low = s;
  uint32_t acc = uint32_t(accumulator - low), cfg = acc + Tile, a = uint32_t(c.activation - low);
  if (hexkl_micro_hmx_setup_acc_read_f16(low, cfg)) std::abort();
  const size_t chunk_stride = size_t(c.chunk) * PackedTile;
  const uint64_t side_rows = c.side ? (uint64_t(c.side_stride * 4) << 32) | (128ull << 16) | c.m : 0;
  for (unsigned p = 0, g = 0; p < c.panels; ++p) {
    if (c.side && p + 1 < c.panels) Q6_l2fetch_AP(const_cast<float *>(c.side + (p + 1) * 32), side_rows);
    hexkl_micro_hmx_acc_clear_f16();
    for (unsigned j = 0; j < c.chunks; ++j, ++g) {
      uint32_t weights;
      if (c.serial) {  // previous panel's acc read finished its weight reads
        dequantize_tiles(c.weight + g * chunk_stride, c.chunk, inline_weights + j * chunk_bytes);
        weights = uint32_t(inline_weights + j * chunk_bytes - low);
      } else {
        Producer &w = c.producers[g % Producers];
        unsigned index = g / Producers;
        while (w.produced.load(std::memory_order_acquire) <= index) spin();
        weights = uint32_t(slots[g % Producers] + (index % c.buffers) * chunk_bytes - low);
      }
      uint32_t x = a + j * chunk_bytes;
      for (unsigned kt = 0; kt < c.chunk; ++kt)
        if (hexkl_micro_hmx_mm_f16(low, x + kt * Tile, weights + kt * Tile)) std::abort();
    }
    // The accumulator read completes every HMX weight read of this panel (as
    // llama.cpp releases its buffers after the job's output store).
    if (hexkl_micro_hmx_acc_read_f16(low, cfg, acc)) std::abort();
    if (!c.serial)
      for (unsigned j = g - c.chunks; j < g; ++j)
        c.producers[j % Producers].consumed.store(j / Producers + 1, std::memory_order_release);
    auto *result = reinterpret_cast<const HVX_Vector *>(accumulator);
    const unsigned column = p * 32;
    for (unsigned q = 0; q < 16; ++q) {
      HVX_VectorPair rows = Q6_Wsf_vcvt_Vhf(result[q]);  // even lanes row 2q, odd 2q+1
      for (unsigned r = 0; r < 2; ++r) {
        unsigned row = 2 * q + r;
        if (row >= c.m) break;
        HVX_Vector y = r ? Q6_V_hi_W(rows) : Q6_V_lo_W(rows);
        if (c.side) {
          HVX_Vector s = *reinterpret_cast<const HVX_UVector *>(c.side + row * c.side_stride + column);
          y = c.product ? Q6_Vsf_vmpy_VsfVsf(s, y) : Q6_Vsf_vadd_VsfVsf(s, y);
        }
        *reinterpret_cast<HVX_UVector *>(c.out + row * c.out_stride + column) = y;
      }
    }
  }
  hexagon_runtime_release_hmx_workspace_dsp(scratch);
}

void produce(Call &c, unsigned w) {
  auto *slot = static_cast<uint8_t *>(
      hexagon_runtime_acquire_prepared_hmx_workspace_dsp(c.buffers * c.chunk * Tile, 2048, true));
  if (!slot) std::abort();
  Producer &self = c.producers[w];
  self.slot.store(slot, std::memory_order_release);
  pack_activation(c, w + 1);
  c.packed.fetch_add(1, std::memory_order_release);
  // Chunk g = panel g / chunks, K tiles [(g % chunks) * chunk, +chunk); panels
  // are contiguous, so chunk g starts at g * chunk packed tiles.
  const size_t chunk_stride = size_t(c.chunk) * PackedTile;
  const unsigned total = c.panels * c.chunks;
  unsigned index = 0;
  for (unsigned g = w; g < total; g += Producers, ++index) {
    if (g + Producers < total) prefetch_tiles(c.weight + (g + Producers) * chunk_stride, c.chunk);
    while (index >= self.consumed.load(std::memory_order_acquire) + c.buffers) spin();
    dequantize_tiles(c.weight + g * chunk_stride, c.chunk, slot + (index % c.buffers) * c.chunk * Tile);
    self.produced.store(index + 1, std::memory_order_release);
  }
  while (self.consumed.load(std::memory_order_acquire) < index) spin();
  hexagon_runtime_release_prepared_hmx_workspace_dsp(slot);
}

void job(void *opaque, uint64_t start, uint64_t rows) {
  auto &c = *static_cast<Call *>(opaque);
  if (start == 0 && rows == 64) {  // serial fallback: the only job, on the caller
    if (qurt_thread_get_id() != c.owner) std::abort();
    c.serial = true;
    for (unsigned part = 0; part < Jobs; ++part) pack_activation(c, part);
    hexagon_runtime_dispatch_hmx_dsp(consume, &c);
    return;
  }
  if (rows != 16 || start % 16) std::abort();  // four jobs of 16 rows
  // The pool runs job 0 on the calling thread, which holds the activation slot
  // (prepared leases may not nest): it submits the HMX consumer; each of the
  // three pool threads produces one interleaved third of the panels.
  if (qurt_thread_get_id() == c.owner) {
    // Producers may resize their slots through the HMX owner: start the
    // consumer only after all three hold their leases.
    pack_activation(c, 0);
    c.packed.fetch_add(1, std::memory_order_release);
    for (auto &w : c.producers)
      while (!w.slot.load(std::memory_order_acquire)) spin();
    while (c.packed.load(std::memory_order_acquire) < Jobs) spin();
    hexagon_runtime_dispatch_hmx_dsp(consume, &c);
    return;
  }
  if (start == 0 || start / 16 > Producers) std::abort();  // job 0 is the caller's
  produce(c, unsigned(start / 16) - 1);
}

void matmul(MemF32 *x, int64_t id, MemF32 *side, bool product, MemF32 *out) {
  uint32_t n = 0, k = 0;
  const uint8_t *weight = qwen_q8_0_weight(id, &n, &k);
  int64_t m = x ? x->sizes[0] : 0;
  if (!weight || !x || !out || m < 1 || m > 32 || x->sizes[1] != k || x->strides[1] != 1 ||
      k % 32 || k > MaxK || n % 32 || out->sizes[0] != m || out->sizes[1] != n || out->strides[1] != 1 ||
      (side && (side->sizes[0] != m || side->sizes[1] != n || side->strides[1] != 1)))
    std::abort();
  if (hexagon_runtime_configure_hmx_performance_dsp(1) ||
      hexagon_runtime_configure_vtcm_pool_dsp(2 * 1024 * 1024))
    std::abort();
  Call c;
  c.x = x->aligned + x->offset; c.x_stride = x->strides[0];
  c.side = side ? side->aligned + side->offset : nullptr; c.side_stride = side ? side->strides[0] : 0;
  c.product = product;
  c.serial = false;
  c.out = out->aligned + out->offset; c.out_stride = out->strides[0];
  c.weight = weight; c.m = unsigned(m); c.n = n; c.k = k;
  c.tiles = k / 32; c.panels = n / 32;
  c.chunks = (c.tiles + 31) / 32;
  c.chunk = c.tiles / c.chunks;
  if (c.chunk * c.chunks != c.tiles) std::abort();  // K in {1024, 2048, 3072}: 32-tile chunks
  c.buffers = SlotBytes / (c.chunk * Tile);
  if (c.buffers > 4) c.buffers = 4;
  c.activation = static_cast<uint8_t *>(
      hexagon_runtime_acquire_prepared_hmx_workspace_dsp(c.tiles * Tile, 2048, true));
  if (!c.activation) std::abort();
  // l2fetch control: stride << 32 | width << 16 | rows (rows of k FP32).
  Q6_l2fetch_AP(const_cast<float *>(c.x), (uint64_t(c.x_stride * 4) << 32) | (uint64_t(k * 4) << 16) | uint64_t(m));
  c.owner = qurt_thread_get_id();
  // 64 rows -> four pool jobs of 16 (fewer than 16 rows would run serially).
#ifdef Q8_0_TEST_SERIAL  // diagnostic builds only: exercise the pool-less path
  job(&c, 0, 64);
#else
  __hexagon_parallel_for_v1(job, &c, 64);
#endif
  hexagon_runtime_release_prepared_hmx_workspace_dsp(c.activation);
}
// ---- Q8_0 x Q8_0 rows: LM head and single-row (decode) projections ----
// llama.cpp CPU semantics (ggml_vec_dot_q8_0_q8_0): the FP32 row is quantized
// per 32 values (d = amax / 127, q = round-half-even(x * (1 / d)), FP16 d) and
// each output is sum_b float(int32 block dot) * (fp32(d_w) * fp32(d_x)), FP32
// accumulation in block order, then the graph's residual addition (R + y) or
// SwiGLU product (M * y) in FP32. Bit-exact with models/qwen3/q8_0/quantize.py q8_q8_matvec.
// Weights: the LM head (only n > 3072) is stored in the vrmpy layout; every
// other projection is the HMX row-pair layout, turned into vrmpy words on the
// fly (llama.cpp accum_q8_0_32x1). Each worker streams its column panels in
// chunks of <= 32 tiles into a prepared VTCM slot (UDMA, 4 chunks in flight).
struct MatVec {
  const uint8_t *weight;
  unsigned tiles, chunk, chunks, panels;
  bool vrmpy_layout;
  const float *side; bool product;
  float *out;
  const uint32_t *words;  // activation int8, 4 per word, k order
  const float *scales;    // fp32(fp16 d_x) per block
};

struct alignas(64) Dma {  // v73 2D UDMA descriptor (16-bit fields)
  void *next;
  uint32_t reserved0 : 24, desc_size : 2, dst_comp : 1, src_comp : 1,
      dst_bypass : 1, src_bypass : 1, order : 1, done : 1;
  const void *src;
  void *dst;
  uint32_t desc_type : 8, reserved1 : 24;
  uint32_t row_size : 16, nrows : 16;
  uint32_t src_stride : 16, dst_stride : 16;
  uint32_t src_offset : 16, dst_offset : 16;
};
static_assert(sizeof(Dma) == 64, "UDMA descriptor");
constexpr unsigned InFlight = 4;  // chunk buffers per worker = descriptors per ring
struct alignas(128) DmaRing {  // one per pool job; tail starts at a zero descriptor
  Dma d[InFlight];
  Dma *tail;
  unsigned next;
};
DmaRing rings[4];

Dma *dma_push(DmaRing &r, const void *src, void *dst, unsigned bytes) {
  if (!r.tail) r.tail = &r.d[InFlight - 1];
  Dma *d = &r.d[r.next++ % InFlight];  // reused only after its transfer was waited for
  *d = {};
  d->desc_size = 1;   // 2D, one row
  d->src_bypass = 1;  // as llama.cpp hex-dma: stream weights past L2
  d->dst_bypass = 1;
  d->src = src;
  d->dst = dst;
  d->row_size = bytes;
  d->nrows = 1;
  d->src_stride = bytes;
  d->dst_stride = bytes;
  asm volatile(" release(%0):at" : : "r"(d));
  asm volatile(" dmlink(%0, %1)" : : "r"(r.tail), "r"(d));
  r.tail = d;
  return d;
}
void dma_wait(const Dma *d) {
  while (!static_cast<const volatile Dma *>(d)->done) {
    unsigned status;
    asm volatile(" %0 = dmpoll" : "=r"(status) : : "memory");
  }
}

// One chunk of a panel: tiles [k0, k0 + v.chunk) accumulated into acc.
inline HVX_Vector matvec_chunk(const MatVec &v, const uint8_t *chunk, unsigned k0, HVX_Vector acc) {
  for (unsigned t = 0; t < v.chunk; ++t) {
    const unsigned kt = k0 + t;
    const auto *w = reinterpret_cast<const HVX_Vector *>(chunk + t * PackedTile);
    HVX_Vector dot = Q6_V_vzero();
    for (unsigned g = 0; g < 8; ++g) {
      // Word lane c: q_w[c][32 kt + 4 g + j], j = 0..3.
      HVX_Vector words = v.vrmpy_layout ? w[g]
          : Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(w[g], 64), w[g], -2));
      dot = Q6_Vw_vrmpyacc_VwVbVb(dot, words, Q6_V_vsplat_R(v.words[kt * 8 + g]));
    }
    HVX_Vector dw = Q6_V_lo_W(Q6_Wsf_vcvt_Vhf(w[8]));  // even FP16 lanes 2c -> d_w[c]
    uint32_t dx;
    std::memcpy(&dx, &v.scales[kt], 4);
    HVX_Vector scale = Q6_Vsf_vmpy_VsfVsf(dw, Q6_V_vsplat_R(dx));
    acc = Q6_Vsf_vadd_VsfVsf(acc, Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(dot), scale));
  }
  return acc;
}

void matvec_panels(void *opaque, uint64_t start, uint64_t count) {
  const auto &v = *static_cast<const MatVec *>(opaque);
  // The pool splits rows as rows * id / 4; the serial fallback is job 0.
  unsigned id = 0;
  while (id < 3 && uint64_t(v.panels) * id / 4 != start) ++id;
  if (uint64_t(v.panels) * id / 4 != start) std::abort();
  const unsigned chunk_bytes = v.chunk * PackedTile;
  auto *slot = static_cast<uint8_t *>(
      hexagon_runtime_acquire_prepared_hmx_workspace_dsp(InFlight * chunk_bytes, 2048, true));
  if (!slot) std::abort();
  DmaRing &ring = rings[id];
  // Units u = panel * chunks + j are contiguous in the panel-major weight.
  const uint64_t first = start * v.chunks, units = count * v.chunks;
  Dma *pending[InFlight] = {};
  for (unsigned u = 0; u < InFlight && u < units; ++u)
    pending[u] = dma_push(ring, v.weight + (first + u) * chunk_bytes, slot + u * chunk_bytes, chunk_bytes);
  HVX_Vector acc = Q6_V_vzero();
  for (uint64_t u = 0; u < units; ++u) {
    uint8_t *buffer = slot + (u % InFlight) * chunk_bytes;
    const unsigned j = unsigned((first + u) % v.chunks);
    dma_wait(pending[u % InFlight]);
    if (!j) acc = Q6_V_vzero();  // +0.0f
    acc = matvec_chunk(v, buffer, j * v.chunk, acc);
    if (u + InFlight < units)  // the chunk was read above: refill its buffer
      pending[u % InFlight] = dma_push(ring, v.weight + (first + u + InFlight) * chunk_bytes, buffer, chunk_bytes);
    if (j + 1 == v.chunks) {
      const uint64_t column = ((first + u) / v.chunks) * 32;
      HVX_Vector y = acc;
      if (v.side) {
        HVX_Vector s = *reinterpret_cast<const HVX_UVector *>(v.side + column);
        y = v.product ? Q6_Vsf_vmpy_VsfVsf(s, y) : Q6_Vsf_vadd_VsfVsf(s, y);
      }
      *reinterpret_cast<HVX_UVector *>(v.out + column) = y;
    }
  }
  hexagon_runtime_release_prepared_hmx_workspace_dsp(slot);
}

// quantize_row_q8_0 on HVX, one vector per 32-value block: exact integer
// absolute maximum, IEEE d = amax / 127 and 1 / d on the scalar unit, IEEE
// x * (1 / d) and round-half-even by adding and subtracting 1.5 * 2^23.
void quantize_row(const float *row, unsigned k, uint32_t *words, float *scales) {
  const HVX_Vector magnitude = Q6_V_vsplat_R(0x7fffffff), round = Q6_V_vsplat_R(0x4b400000);  // 1.5 * 2^23
  alignas(128) int32_t q[32];
  auto *qa = reinterpret_cast<int8_t *>(words);
  for (unsigned b = 0; b < k / 32; ++b) {
    HVX_Vector x = *reinterpret_cast<const HVX_UVector *>(row + b * 32);
    HVX_Vector m = Q6_V_vand_VV(x, magnitude);  // |x| bits order as |x| (no NaN)
    for (unsigned width = 64; width >= 4; width >>= 1) m = Q6_Vw_vmax_VwVw(m, Q6_V_vror_VR(m, width));
    uint32_t bits = uint32_t(Q6_R_vextract_VR(m, 0));
    float amax;
    std::memcpy(&amax, &bits, 4);
    const float d = amax / 127.0f;
    const float inverse = d != 0.0f ? 1.0f / d : 0.0f;
    uint32_t inv;
    std::memcpy(&inv, &inverse, 4);
    HVX_Vector p = Q6_Vsf_vmpy_VsfVsf(x, Q6_V_vsplat_R(inv));
    p = Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vadd_VsfVsf(p, round), round);
    *reinterpret_cast<HVX_Vector *>(q) = Q6_Vw_equals_Vsf(p);
    for (unsigned j = 0; j < 32; ++j) qa[b * 32 + j] = int8_t(q[j]);
    scales[b] = float(static_cast<_Float16>(d));
  }
}

// Set by the resident session around a forward whose logits the caller
// discards (protocol mode 4). The vocabulary projection is then validated but
// not computed, and its output buffer is left unwritten.
bool skip_vocabulary = false;

void matvec_q8(MemF32 *x, int64_t id, MemF32 *side, bool product, MemF32 *out) {
  uint32_t n = 0, k = 0;
  const uint8_t *weight = qwen_q8_0_weight(id, &n, &k);
  if (!weight || !x || !out || x->sizes[0] != 1 || x->sizes[1] != k || x->strides[1] != 1 || k % 32 ||
      k > MaxK || n % 32 || out->sizes[0] != 1 || out->sizes[1] != n || out->strides[1] != 1 ||
      (side && (side->sizes[0] != 1 || side->sizes[1] != n || side->strides[1] != 1)))
    std::abort();
  const bool vocabulary = n > 3072;  // q8_0/weights.py: only the LM head is this wide
  if (vocabulary && !side && skip_vocabulary) return;
  alignas(128) uint32_t words[MaxK / 4];
  float scales[MaxK / 32];
  quantize_row(x->aligned + x->offset, k, words, scales);
  // Same VTCM pool configuration as the HMX path: the prepared slots below are
  // allocated from it, and it may not change once slots exist.
  if (hexagon_runtime_configure_hmx_performance_dsp(1) ||
      hexagon_runtime_configure_vtcm_pool_dsp(2 * 1024 * 1024))
    std::abort();
  MatVec v{};
  v.weight = weight; v.tiles = k / 32; v.panels = n / 32;
  v.chunks = (v.tiles + 31) / 32; v.chunk = v.tiles / v.chunks;
  if (v.chunk * v.chunks != v.tiles) std::abort();  // K in {1024, 2048, 3072}
  v.vrmpy_layout = vocabulary;  // q8_0/weights.py: only the LM head uses the vrmpy layout
  v.side = side ? side->aligned + side->offset : nullptr; v.product = product;
  v.out = out->aligned + out->offset; v.words = words; v.scales = scales;
  __hexagon_parallel_for_v1(matvec_panels, &v, n / 32);
}
} // namespace

extern "C" void qwen_q8_0_skip_vocabulary(int skip) { skip_vocabulary = skip != 0; }

extern "C" void _mlir_ciface___hexagon_q8_0_matvec_q8(MemF32 *x, int64_t id, MemF32 *out) {
  matvec_q8(x, id, nullptr, false, out);
}
extern "C" void _mlir_ciface___hexagon_q8_0_matvec_q8_residual(MemF32 *x, int64_t id, MemF32 *residual, MemF32 *out) {
  matvec_q8(x, id, residual, false, out);
}
extern "C" void _mlir_ciface___hexagon_q8_0_matvec_q8_product(MemF32 *x, int64_t id, MemF32 *factor, MemF32 *out) {
  matvec_q8(x, id, factor, true, out);
}

extern "C" void _mlir_ciface___hexagon_q8_0_matmul(MemF32 *x, int64_t id, MemF32 *out) {
  matmul(x, id, nullptr, false, out);
}
extern "C" void _mlir_ciface___hexagon_q8_0_matmul_residual(MemF32 *x, int64_t id, MemF32 *residual, MemF32 *out) {
  matmul(x, id, residual, false, out);
}
extern "C" void _mlir_ciface___hexagon_q8_0_matmul_product(MemF32 *x, int64_t id, MemF32 *factor, MemF32 *out) {
  matmul(x, id, factor, true, out);
}
