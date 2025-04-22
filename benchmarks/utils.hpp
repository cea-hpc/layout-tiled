// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/mman.h> // mmamp/munmap
#include <unistd.h>   // sysconf

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <nanobench.h>
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#include <omp.h>
#endif

/// A heap-allocated, contiguous memory slice.
///
/// The underlying memory allocation is backed by `mmap` and is left uninitialized to allow leveraging
/// "first touch" policy in parallel execution contexts.
template <class T>
struct MemorySlice {
  /// Constructs a new slice of size `len * sizeof(T)` bytes.
  explicit MemorySlice(size_t len)
      : len_(len * sizeof(T)) {
    // Allocate a page-aligned block of `len` bytes w/ read-write privileges & shared, anonymous mapping.
    ptr_ = static_cast<T*>(mmap(nullptr, len_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    assert(MAP_FAILED != ptr_);
  }

  ~MemorySlice() {
    int ret = munmap(static_cast<void*>(ptr_), len_);
    assert(0 == ret);
  }
  // Prevent deep copies.
  MemorySlice(MemorySlice const& rhs)                    = delete;
  auto operator=(MemorySlice const& rhs) -> MemorySlice& = delete;
  // Allow moves.
  MemorySlice(MemorySlice&& rhs) noexcept
      : ptr_(rhs.ptr_), len_(rhs.len_) {
    rhs.ptr_ = nullptr;
  }
  auto operator=(MemorySlice&& rhs) -> MemorySlice& {
    int ret = munmap(static_cast<void*>(ptr_), len_);
    assert(0 == ret);
    ptr_     = rhs.ptr_;
    len_     = rhs.len_;
    rhs.ptr_ = nullptr;
    return *this;
  }

  /// Returns a raw pointer to the underlying allocation.
  auto get() -> T* {
    return ptr_;
  }

private:
  /// Pointer to the underlying memory allocation.
  T* ptr_ = nullptr;
  /// Length in Bytes of the memory slice.
  size_t len_ = 0;
};

namespace benchmark_utils {

/// Fill a 2D `mdspan` of `double`-typed elements with random values of uniform distribution in range [0, 1).
/// Computes the L2-norm of an `mdspan`.
template <class MdSpan>
auto l2_norm(MdSpan m) -> double
  requires(std::is_floating_point_v<typename MdSpan::value_type>)
{
  auto ptr = m.data();
  double sum = 0.0;
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel reduction(+:sum) schedule(static)
#endif
  for (size_t e = 0; e < m.size(); ++e) {
    sum += ptr[e];
  }
  return std::sqrt(sum);
}

auto fill_2d_uniform01(MdSpan m, ankerl::nanobench::Rng& rng) -> void
  requires(std::is_same_v<double, typename MdSpan::value_type> && 2 == MdSpan::rank())
{
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
auto fill_3d_uniform01(MdSpan m, ankerl::nanobench::Rng& rng) -> void
  requires(std::is_floating_point_v<typename MdSpan::value_type>, 3 == MdSpan::rank())
{
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
