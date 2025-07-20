// benchmarks/utils.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_BENCHMARKS_UTILS
#define INCLUDED_BENCHMARKS_UTILS

#include <limits>
#include <sys/mman.h> // mmamp/munmap
#include <unistd.h>   // sysconf

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <experimental/mdspan>
#include <limits>
#include <string_view>
#include <type_traits>

#include <fmt/core.h>
#include <nanobench.h>
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#include <omp.h>
#endif

template <class MdSpan>
concept IsMdSpan2D = 2 == MdSpan::rank();

template <class MdSpan>
concept IsMdSpan3D = 3 == MdSpan::rank();

template <class MdSpan>
concept IsLayoutRight = std::is_same_v<typename MdSpan::layout_type, std::layout_right>;

template <class MdSpan>
concept IsLayoutLeft = std::is_same_v<typename MdSpan::layout_type, std::layout_left>;

/// A heap-allocated, contiguous memory slice.
///
/// The underlying memory allocation is backed by `mmap` and is left uninitialized to allow leveraging
/// "first-touch" policy in parallel execution contexts.
template <class T>
struct MemorySlice {
  /// Constructs a new slice of size `len * sizeof(T)` bytes.
  explicit MemorySlice(size_t len) : len_(len * sizeof(T)) {
    // Allocate a page-aligned block of `len` bytes w/ read-write privileges & shared, anonymous mapping.
    ptr_ = static_cast<T*>(mmap(nullptr, len_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    assert(MAP_FAILED != ptr_);
  }

  ~MemorySlice() {
    [[maybe_unused]] int _ret = munmap(static_cast<void*>(ptr_), len_);
    assert(0 == _ret);
  }

  // Prevent deep copies.
  MemorySlice(const MemorySlice& rhs)                    = delete;
  auto operator=(const MemorySlice& rhs) -> MemorySlice& = delete;

  // Allow moves.
  MemorySlice(MemorySlice&& rhs) noexcept : ptr_(rhs.ptr_), len_(rhs.len_) {
    rhs.ptr_ = nullptr;
  }
  auto operator=(MemorySlice&& rhs) -> MemorySlice& {
    [[maybe_unused]] int _ret = munmap(static_cast<void*>(ptr_), len_);
    assert(0 == _ret);
    ptr_     = rhs.ptr_;
    len_     = rhs.len_;
    rhs.ptr_ = nullptr;
    return *this;
  }

  /// Returns a raw pointer to the underlying allocation.
  [[nodiscard]] auto get() -> T* {
    return ptr_;
  }

private:
  /// Pointer to the underlying memory allocation.
  T* ptr_ = nullptr;
  /// Length in bytes of the memory slice.
  size_t len_ = 0;
};

namespace benchmark_utils {

template <class T>
  requires std::is_floating_point_v<T>
auto check(T reference, T result, const std::string_view label, T big_o = 1.0) -> void {
  if (std::abs(result - reference) > (big_o * std::numeric_limits<T>::epsilon())) {
    fmt::print(stderr, "error({}):\n  expected {:.18f}\n       got {:.18f}\n", label, reference, result);
  }
}

/// Computes the L2-norm of an `mdspan`.
template <class MdSpan>
  requires std::is_floating_point_v<typename MdSpan::value_type>
[[nodiscard]] auto l2_norm(MdSpan m) -> double {
  const auto ptr = m.data_handle();
  double sum     = 0.0;
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for reduction(+ : sum)
#endif
  for (size_t e = 0; e < m.size(); ++e) {
    sum += ptr[e];
  }
  return std::sqrt(sum);
}

/// Fills an `mdspan` with zeroes.
template <class MdSpan>
  requires std::is_floating_point_v<typename MdSpan::value_type>
auto fill_zeroes(MdSpan m) -> void {
  auto ptr = m.data_handle();
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t e = 0; e < m.size(); ++e) {
    ptr[e] = 0.0;
  }
}

/// Fills a 2D `mdspan` of `double`-typed elements with values of an other `mdspan` of identical extents.
template <class MdSpanSrc, class MdSpanDst>
  requires IsMdSpan2D<MdSpanSrc> && IsMdSpan2D<MdSpanDst> && std::is_floating_point_v<typename MdSpanSrc::value_type>
           && std::is_floating_point_v<typename MdSpanDst::value_type>
auto fill_2d_copy(MdSpanDst dst, const MdSpanSrc src) -> void {
  assert(src.extent(0) == dst.extent(0));
  assert(src.extent(1) == dst.extent(1));

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < dst.extent(0); ++i) {
    for (size_t j = 0; j < dst.extent(1); ++j) {
      dst[i, j] = src[i, j];
    }
  }
}

/// Fills a 2D `mdspan` of `double`-typed elements with random values of uniform distribution in range [0, 1).
template <class MdSpan>
  requires std::is_floating_point_v<typename MdSpan::value_type> && (2 == MdSpan::rank())
auto fill_2d_uniform01(MdSpan m, ankerl::nanobench::Rng& rng) -> void {
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < m.extent(0); ++i) {
    for (size_t j = 0; j < m.extent(1); ++j) {
      m[i, j] = rng.uniform01();
    }
  }
}

/// Fills a 3D `mdspan` of `double`-typed elements with random values of uniform distribution in range [0, 1).
template <class MdSpan>
  requires std::is_floating_point_v<typename MdSpan::value_type> && (3 == MdSpan::rank())
auto fill_3d_uniform01(MdSpan m, ankerl::nanobench::Rng& rng) -> void {
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t i = 0; i < m.extent(0); ++i) {
    for (size_t j = 0; j < m.extent(1); ++j) {
      for (size_t k = 0; k < m.extent(2); ++k) {
        m[i, j, k] = rng.uniform01();
      }
    }
  }
}

} // namespace benchmark_utils

#endif // INCLUDED_BENCHMARKS_UTILS
