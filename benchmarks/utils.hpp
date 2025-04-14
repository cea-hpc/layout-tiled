// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/mman.h>

#include <array>
#include <cstddef>
#include <cstdlib>

#include <nanobench.h>
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#include <omp.h>
#endif

template <class T>
struct MemorySlice {
  explicit MemorySlice(size_t len)
      : len_(len * sizeof(T)) {
    ptr_ = (T*)mmap(nullptr, len_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    assert(MAP_FAILED != ptr_);
  }

  ~MemorySlice() {
    munmap((void*)ptr_, len_);
  }
  MemorySlice(MemorySlice const& rhs) = delete;
  MemorySlice(MemorySlice&& rhs) noexcept
      : ptr_(rhs.ptr_), len_(rhs.len_) {
    rhs.ptr_ = nullptr;
  }
  auto operator=(MemorySlice const& rhs) -> MemorySlice& = delete;
  auto operator=(MemorySlice&& rhs) -> MemorySlice& {
    munmap((void*)ptr_, len_);
    ptr_     = rhs.ptr_;
    len_     = rhs.len_;
    rhs.ptr_ = nullptr;
    return *this;
  }

  auto get() -> T* {
    return ptr_;
  }

private:
  T* ptr_;
  size_t len_;
};

namespace benchmark_utils {

template <class MdSpan>
auto fill_2d_uniform01(MdSpan&& m, ankerl::nanobench::Rng& rng) -> void
// requires(std::is_same_v<double, typename MdSpan::value_type> && 2 == MdSpan::rank())
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

} // namespace benchmark_utils
