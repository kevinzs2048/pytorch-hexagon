//===- HexagonMemrefCopy.cpp - MLIR memref copy on DSP --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Implements the MLIR unranked memrefCopy ABI without the host runner library.
// Like memref.copy, source and destination have the same shape and element type.
// Overlapping source and destination are not supported.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
struct UnrankedMemRef {
  int64_t rank;
  void *descriptor;
};
struct RankedPrefix {
  char *allocated;
  char *aligned;
  int64_t offset;
};
struct View {
  RankedPrefix prefix;
  const int64_t *sizes;
  const int64_t *strides;
  explicit View(const UnrankedMemRef &value) {
    std::memcpy(&prefix, value.descriptor, sizeof(prefix));
    sizes = reinterpret_cast<const int64_t *>(
        static_cast<const char *>(value.descriptor) + sizeof(prefix));
    strides = sizes + value.rank;
  }
};
} // namespace

// For positive factors, the native 32x32 product is exact in uint64_t.
// Large factors retain the original division guard. No valid/invalid product
// classification changes, and no FP values are interpreted by this runtime.
static bool positive_product(int64_t a,int64_t b,int64_t *out) {
  if (uint64_t(a)<=UINT32_MAX && uint64_t(b)<=UINT32_MAX) {
    uint64_t value=uint64_t(uint32_t(a))*uint64_t(uint32_t(b));
    if (value>uint64_t(std::numeric_limits<int64_t>::max())) return false;
    *out=int64_t(value);return true;
  }
  if (b>std::numeric_limits<int64_t>::max()/a) return false;
  *out=a*b;return true;
}
static bool block_extent(int64_t elements,int64_t elementSize,size_t *out) {
  if (uint64_t(elements)<=UINT32_MAX && uint64_t(elementSize)<=UINT32_MAX) {
    uint64_t bytes=uint64_t(uint32_t(elements))*uint64_t(uint32_t(elementSize));
    if (bytes>uint64_t(SIZE_MAX)) return false;
    *out=size_t(bytes);return true;
  }
  if (uint64_t(elements)>SIZE_MAX/uint64_t(elementSize)) return false;
  *out=size_t(elements)*size_t(elementSize);return true;
}
#ifdef QWEN_COPY_CHECKS_TEST_ONLY
extern "C" bool qwen_copy_positive_product(int64_t a,int64_t b,int64_t *out) {
  if (a<=0 || b<=0 || !out) return false;
  return positive_product(a,b,out);
}
extern "C" bool qwen_copy_block_extent(int64_t a,int64_t b,size_t *out) {
  if (a<=0 || b<=0 || !out) return false;
  return block_extent(a,b,out);
}
#endif

extern "C" __attribute__((visibility("default"), used))
void __wrap_memrefCopy(int64_t elementSize, UnrankedMemRef *source,
                UnrankedMemRef *destination) {
  if (!source || !destination || source->rank < 0 ||
      source->rank != destination->rank || !source->descriptor ||
      !destination->descriptor || elementSize <= 0 ||
      uint64_t(source->rank) > SIZE_MAX / sizeof(int64_t) ||
      uint64_t(elementSize) > SIZE_MAX)
    std::abort();
  const View src(*source), dst(*destination);
  for (int64_t axis = 0; axis < source->rank; ++axis) {
    if (src.sizes[axis] < 0 || src.sizes[axis] != dst.sizes[axis])
      std::abort();
    if (src.sizes[axis] == 0)
      return;
  }
  if (!src.prefix.aligned || !dst.prefix.aligned)
    std::abort();

  // Coalesce the shared contiguous suffix into one memcpy. This includes the
  // common whole-buffer case; noncontiguous/negative/zero strides still use
  // the general indexed traversal below.
  int64_t outerRank = source->rank;
  int64_t blockElements = 1;
  while (outerRank > 0 &&
         src.strides[outerRank - 1] == blockElements &&
         dst.strides[outerRank - 1] == blockElements) {
    int64_t product;
    if (!positive_product(blockElements,src.sizes[outerRank - 1],&product))
      std::abort();
    blockElements=product;--outerRank;
  }
  size_t blockBytes;
  if (!block_extent(blockElements,elementSize,&blockBytes))
    std::abort();
  char *srcData = src.prefix.aligned + src.prefix.offset * elementSize;
  char *dstData = dst.prefix.aligned + dst.prefix.offset * elementSize;
  if (outerRank == 0) {
    std::memcpy(dstData, srcData, blockBytes);
    return;
  }

  auto *indices = static_cast<int64_t *>(std::calloc(outerRank, sizeof(int64_t)));
  if (!indices)
    std::abort();
  int64_t read = 0, write = 0;
  for (;;) {
    std::memcpy(dstData + write * elementSize,
                srcData + read * elementSize, blockBytes);
    int64_t axis = outerRank - 1;
    for (; axis >= 0; --axis) {
      ++indices[axis];
      read += src.strides[axis];
      write += dst.strides[axis];
      if (indices[axis] < src.sizes[axis])
        break;
      indices[axis] = 0;
      read -= src.sizes[axis] * src.strides[axis];
      write -= dst.sizes[axis] * dst.strides[axis];
    }
    if (axis < 0)
      break;
  }
  std::free(indices);
}
