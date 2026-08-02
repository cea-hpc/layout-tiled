// benchmarks/gemm.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <getopt.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <experimental/mdspan>
#include <fstream>
#include <iostream>
#include <string>

#include <fmt/core.h>
#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_rtr.hpp>
#include <layout_tiled/reduce.hpp>
#include <nanobench.h>
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#include <omp.h>
#endif
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
#include <cblas.h>
#endif

#include "utils.hpp"

using namespace std::chrono_literals;
using Type = double;
using Exts = std::dextents<size_t, 2>;

constexpr size_t TILE_L1 = 32;
constexpr size_t TILE_L2 = 128;

struct Config {
  uint64_t n_ = 512;
  uint64_t t_ = TILE_L1;
  std::string output_;
  std::string implementation_;
};

namespace {

/// A view of a rank-2 tiled mdspan as a rank-2 grid of affine mdspan tiles.
///
/// The source mapping owns the global-to-tile decomposition. Each lookup pays
/// that mapping cost once and returns an mdspan whose mapping is the tiled
/// layout's affine inner layout.
template <class MdSpan>
  requires IsMdSpan2D<MdSpan>
class basic_tiled_view_2d {
public:
  using source_type       = MdSpan;
  using mapping_type      = typename source_type::mapping_type;
  using element_type      = typename source_type::element_type;
  using index_type        = typename source_type::index_type;
  using inner_layout_type = typename mapping_type::inner_layout_type;

  static constexpr size_t tile_extent_0 = mapping_type::textent(0);
  static constexpr size_t tile_extent_1 = mapping_type::textent(1);

  using tile_extents_type = std::extents<index_type, tile_extent_0, tile_extent_1>;
  using tile_type         = std::mdspan<element_type, tile_extents_type, inner_layout_type>;

  explicit constexpr basic_tiled_view_2d(source_type source) noexcept : source_(source) {}

  [[nodiscard]] constexpr auto extent(size_t r) const noexcept -> index_type {
    assert(r < source_type::rank());
    return source_.mapping().ntiles_in_extent(r);
  }

  [[nodiscard]] constexpr auto operator[](index_type tile_i, index_type tile_j) const noexcept -> tile_type {
    assert(tile_i < extent(0));
    assert(tile_j < extent(1));

    const index_type i      = tile_i * static_cast<index_type>(tile_extent_0);
    const index_type j      = tile_j * static_cast<index_type>(tile_extent_1);
    const index_type offset = source_.mapping().tile_offset_outer(i, j);
    return tile_type(source_.data_handle() + offset);
  }

private:
  source_type source_;
};

template <class MdSpan>
  requires IsMdSpan2D<MdSpan>
[[nodiscard]] constexpr auto tiled_view(MdSpan source) noexcept -> basic_tiled_view_2d<MdSpan> {
  return basic_tiled_view_2d<MdSpan>(source);
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto naive_ijk(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
      for (size_t k = 0; k < O; ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto naive_ikj(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < M; ++i) {
    if (0 == O) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
      for (size_t j = 0; j < N; ++j) {
        C[i, j] = 0.0;
      }
      continue;
    }

    const Type a0 = A[i, 0];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
    for (size_t j = 0; j < N; ++j) {
      C[i, j] = a0 * B[0, j];
    }

    for (size_t k = 1; k < O; ++k) {
      const Type a = A[i, k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
      for (size_t j = 0; j < N; ++j) {
        C[i, j] += a * B[k, j];
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto cache_blocked_ijk(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      for (size_t z = 0; z < O; z += TO) {
        for (size_t i = x; i < std::min(x + TM, M); ++i) {
          for (size_t j = y; j < std::min(y + TN, N); ++j) {
            Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t k = z; k < std::min(z + TO, O); ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] += acc;
          }
        }
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC> && IsLayoutRight<MdSpanA>
           && IsLayoutRight<MdSpanB> && IsLayoutRight<MdSpanC>
auto cache_blocked_ikj(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      const size_t i_end = std::min(x + TM, M);
      const size_t j_end = std::min(y + TN, N);
      const size_t tj    = j_end - y;
      for (size_t z = 0; z < O; z += TO) {
        const size_t k_end = std::min(z + TO, O);
        for (size_t i = x; i < i_end; ++i) {
          Type* c_row = C.data_handle() + C.mapping()(i, y);
          for (size_t k = z; k < k_end; ++k) {
            const Type a      = *(A.data_handle() + A.mapping()(i, k));
            const Type* b_row = B.data_handle() + B.mapping()(k, y);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t j = 0; j < tj; ++j) {
              c_row[j] += a * b_row[j];
            }
          }
        }
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto data_tiled_ijk(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  using ATile = std::mdspan<Type, std::extents<size_t, TM, TO>, std::layout_right>;
  using BTile = std::mdspan<Type, std::extents<size_t, TO, TN>, std::layout_right>;
  using CTile = std::mdspan<Type, std::extents<size_t, TM, TN>, std::layout_right>;

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      auto tC = CTile(C.data_handle() + C.mapping().tile_offset_outer(x, y));
      for (size_t i = 0; i < std::min(TM, M - x); ++i) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t j = 0; j < std::min(TN, N - y); ++j) {
          tC[i, j] = Type{0};
        }
      }
      for (size_t z = 0; z < O; z += TO) {
        const auto tA = ATile(A.data_handle() + A.mapping().tile_offset_outer(x, z));
        const auto tB = BTile(B.data_handle() + B.mapping().tile_offset_outer(z, y));
        for (size_t i = 0; i < std::min(TM, M - x); ++i) {
          for (size_t j = 0; j < std::min(TN, N - y); ++j) {
            Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < std::min(TO, O - z); ++k) {
              acc += tA[i, k] * tB[k, j];
            }
            tC[i, j] += acc;
          }
        }
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto data_tiled_ikj(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  using ATile = std::mdspan<Type, std::extents<size_t, TM, TO>, std::layout_right>;
  using BTile = std::mdspan<Type, std::extents<size_t, TO, TN>, std::layout_right>;
  using CTile = std::mdspan<Type, std::extents<size_t, TM, TN>, std::layout_right>;

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      auto tC = CTile(C.data_handle() + C.mapping().tile_offset_outer(x, y));
      for (size_t i = 0; i < std::min(TM, M - x); ++i) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t j = 0; j < std::min(TN, N - y); ++j) {
          tC[i, j] = Type{0};
        }
      }
      for (size_t z = 0; z < O; z += TO) {
        const auto tA = ATile(A.data_handle() + A.mapping().tile_offset_outer(x, z));
        const auto tB = BTile(B.data_handle() + B.mapping().tile_offset_outer(z, y));
        for (size_t i = 0; i < std::min(TM, M - x); ++i) {
          for (size_t k = 0; k < std::min(TO, O - z); ++k) {
            const Type a = tA[i, k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t j = 0; j < std::min(TN, N - y); ++j) {
              tC[i, j] += a * tB[k, j];
            }
          }
        }
      }
    }
  }
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto tiled_view_ikj(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  const auto tiled_A = tiled_view(A);
  const auto tiled_B = tiled_view(B);
  auto tiled_C       = tiled_view(C);

  assert(tiled_A.extent(1) == tiled_B.extent(0));
  assert(tiled_A.extent(0) == tiled_C.extent(0));
  assert(tiled_B.extent(1) == tiled_C.extent(1));

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t tile_i = 0; tile_i < tiled_C.extent(0); ++tile_i) {
    for (size_t tile_j = 0; tile_j < tiled_C.extent(1); ++tile_j) {
      auto C_tile        = tiled_C[tile_i, tile_j];
      const size_t j_end = std::min(
        static_cast<size_t>(C_tile.extent(1)),
        static_cast<size_t>(C.extent(1)) - tile_j * static_cast<size_t>(C_tile.extent(1))
      );
      for (size_t i = 0; i < C_tile.extent(0); ++i) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t j = 0; j < j_end; ++j) {
          C_tile[i, j] = Type{0};
        }
      }
      for (size_t tile_k = 0; tile_k < tiled_A.extent(1); ++tile_k) {
        const auto A_tile = tiled_A[tile_i, tile_k];
        const auto B_tile = tiled_B[tile_k, tile_j];
        for (size_t i = 0; i < C_tile.extent(0); ++i) {
          for (size_t k = 0; k < A_tile.extent(1); ++k) {
            const Type a = A_tile[i, k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t j = 0; j < j_end; ++j) {
              C_tile[i, j] += a * B_tile[k, j];
            }
          }
        }
      }
    }
  }
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto layout_aware_reduce(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  enum Axes { I, J, K };
  layout_tiled::reduce<K>(
    layout_tiled::out<I, J>(C, Type{0}),
    layout_tiled::in<I, K>(A),
    layout_tiled::in<K, J>(B),
    [](auto& c, auto a, auto b) { c += a * b; }
  );
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto data_tiled_b_left_ijk(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  using ATile = std::mdspan<Type, std::extents<size_t, TM, TO>, std::layout_right>;
  using BTile = std::mdspan<Type, std::extents<size_t, TO, TN>, std::layout_left>;
  using CTile = std::mdspan<Type, std::extents<size_t, TM, TN>, std::layout_right>;

  const size_t M = A.extent(0);
  const size_t N = B.extent(1);
  const size_t O = A.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      auto tC = CTile(C.data_handle() + C.mapping().tile_offset_outer(x, y));
      for (size_t z = 0; z < O; z += TO) {
        const auto tA = ATile(A.data_handle() + A.mapping().tile_offset_outer(x, z));
        const auto tB = BTile(B.data_handle() + B.mapping().tile_offset_outer(z, y));
        for (size_t i = 0; i < std::min(TM, M - x); ++i) {
          for (size_t j = 0; j < std::min(TN, N - y); ++j) {
            Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < std::min(TO, O - z); ++k) {
              acc += tA[i, k] * tB[k, j];
            }
            tC[i, j] += acc;
          }
        }
      }
    }
  }
}

#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC> && IsLayoutRight<MdSpanA>
           && IsLayoutRight<MdSpanB> && IsLayoutRight<MdSpanC>
auto blas(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == B.extent(0));
  assert(B.extent(1) == C.extent(1));

  const int M            = static_cast<int>(A.extent(0));
  const int N            = static_cast<int>(B.extent(1));
  const int O            = static_cast<int>(A.extent(1));
  const double alpha     = 1.0;
  const double* A_handle = A.data_handle();
  const int lda          = static_cast<int>(A.stride(0));
  const double* B_handle = B.data_handle();
  const int ldb          = static_cast<int>(B.stride(0));
  const double beta      = 0.0;
  double* C_handle       = C.data_handle();
  const int ldc          = static_cast<int>(C.stride(0));

  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    M,
    N,
    O,
    alpha,
    A_handle,
    lda,
    B_handle,
    ldb,
    beta,
    C_handle,
    ldc
  );
}
#endif

auto run_bench(const Config& cfg) -> int {
#if defined(LAYOUT_TILED_BENCH_OPENBLAS)
  if (cfg.implementation_.empty() || cfg.implementation_ == "blas") {
    const int blas_threads = openblas_get_num_threads();
    fmt::println(stderr, "OpenBLAS threads: {}", blas_threads);
    if (blas_threads != 4) {
      fmt::println(stderr, "error: GEMM comparison requires OPENBLAS_NUM_THREADS=4");
      return 1;
    }
  }
#endif
  // Standard layout matrices
  auto rA      = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto rB      = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto rC_ref  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto rC_blas = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lB      = MemorySlice<Type>(cfg.n_ * cfg.n_);
  // Tiled layout matrices (L1 tiles)
  auto trA_l1        = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trB_l1        = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trC_dt_l1     = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trC_tv_l1     = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trC_reduce_l1 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto tlB_l1        = MemorySlice<Type>(cfg.n_ * cfg.n_);
  // Tiled layout matrices (L2 tiles)
  auto trA_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trB_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trC_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto tlB_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);

  // Associated MDSPANs
  auto mrA                       = std::mdspan<Type, Exts>(rA.get(), cfg.n_, cfg.n_);
  auto mrB                       = std::mdspan<Type, Exts>(rB.get(), cfg.n_, cfg.n_);
  [[maybe_unused]] auto mrC_ref  = std::mdspan<Type, Exts>(rC_ref.get(), cfg.n_, cfg.n_);
  [[maybe_unused]] auto mrC_blas = std::mdspan<Type, Exts>(rC_blas.get(), cfg.n_, cfg.n_);
  auto mlB                       = std::mdspan<Type, Exts, std::layout_left>(lB.get(), cfg.n_, cfg.n_);
  auto mtrA_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trA_l1.get(), cfg.n_, cfg.n_);
  auto mtrB_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trB_l1.get(), cfg.n_, cfg.n_);
  auto mtrC_dt_l1
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trC_dt_l1.get(), cfg.n_, cfg.n_);
  auto mtrC_tv_l1
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trC_tv_l1.get(), cfg.n_, cfg.n_);
  auto mtrC_reduce_l1
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trC_reduce_l1.get(), cfg.n_, cfg.n_);
  auto mtlB_l1 = std::mdspan<Type, Exts, layout_tiled::layout_ltl<TILE_L1, TILE_L1>>(tlB_l1.get(), cfg.n_, cfg.n_);
  auto mtrA_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(trA_l2.get(), cfg.n_, cfg.n_);
  auto mtrB_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(trB_l2.get(), cfg.n_, cfg.n_);
  auto mtrC_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(trC_l2.get(), cfg.n_, cfg.n_);
  auto mtlB_l2 = std::mdspan<Type, Exts, layout_tiled::layout_ltl<TILE_L2, TILE_L2>>(tlB_l2.get(), cfg.n_, cfg.n_);

  // Matrix initializations
  ankerl::nanobench::Rng rng_naive(42);
  benchmark_utils::fill_2d_uniform01(mrA, rng_naive);
  benchmark_utils::fill_2d_uniform01(mrB, rng_naive);
  benchmark_utils::fill_2d_copy(mlB, mrB);
  benchmark_utils::fill_2d_copy(mtrA_l1, mrA);
  benchmark_utils::fill_2d_copy(mtrB_l1, mrB);
  benchmark_utils::fill_2d_copy(mtlB_l1, mlB);
  benchmark_utils::fill_2d_copy(mtrA_l2, mrA);
  benchmark_utils::fill_2d_copy(mtrB_l2, mrB);
  benchmark_utils::fill_2d_copy(mtlB_l2, mlB);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  // Checks
  const auto n       = static_cast<double>(cfg.n_);
  const double big_o = 2.0 * n * n * n;
  int res            = 0;

  naive_ijk(mrA, mrB, mrC_ref);
  double ref = benchmark_utils::l2_norm(mrC_ref);
  benchmark_utils::fill_zeroes(mrC_ref);

  cache_blocked_ikj<TILE_L1, TILE_L1, TILE_L1>(mrA, mrB, mrC_ref);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC_ref), "CB-IKJ-SRRR-L1", big_o);
  benchmark_utils::fill_zeroes(mrC_ref);
  cache_blocked_ikj<TILE_L2, TILE_L2, TILE_L2>(mrA, mrB, mrC_ref);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC_ref), "CB-IKJ-SRRR-L2", big_o);
  benchmark_utils::fill_zeroes(mrC_ref);

  naive_ikj(mtrA_l1, mtrB_l1, mtrC_dt_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_dt_l1), "N-IKJ-TRRR-L1", big_o);
  benchmark_utils::fill_zeroes(mtrC_dt_l1);
  naive_ikj(mtrA_l2, mtrB_l2, mtrC_l2);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "N-IKJ-TRRR-L2", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);
  naive_ijk(mtrA_l1, mtlB_l1, mtrC_dt_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_dt_l1), "N-IJK-TRLR-L1", big_o);
  benchmark_utils::fill_zeroes(mtrC_dt_l1);
  naive_ijk(mtrA_l2, mtlB_l2, mtrC_l2);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "N-IJK-TRLR-L2", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  data_tiled_ijk<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_dt_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_dt_l1), "DT-IJK-TRRR-L1", big_o);
  benchmark_utils::fill_zeroes(mtrC_dt_l1);
  data_tiled_ijk<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "DT-IJK-TRRR-L2", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  data_tiled_ikj<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_dt_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_dt_l1), "DT-IKJ-TRRR-L1", big_o);
  tiled_view_ikj(mtrA_l1, mtrB_l1, mtrC_tv_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_tv_l1), "TV-IKJ-TRRR-L1", big_o);
  layout_aware_reduce(mtrA_l1, mtrB_l1, mtrC_reduce_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "REDUCE-TRRR-L1", big_o);
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  blas(mrA, mrB, mrC_blas);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC_blas), "BLAS-SRRR", big_o);
#endif
  data_tiled_ikj<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "DT-IKJ-TRRR-L2", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  return res;
#else
  // Benchmarks
  ankerl::nanobench::Bench bs;
  bs.output(nullptr);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
  bs.title(fmt::format("{0}x{0} {1}C", cfg.n_, omp_get_max_threads())).timeUnit(1s, "s").relative(true);
#else
  bs.title(fmt::format("{0}x{0} 1C", cfg.n_)).timeUnit(1s, "s").relative(true);
#endif
  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }

  const auto selected = [&](const std::string_view implementation) {
    return cfg.implementation_.empty() || cfg.implementation_ == implementation;
  };
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  if (selected("blas")) {
    bs.run("BLAS-SRRR", [&]() { blas(mrA, mrB, mrC_blas); });
  }
#endif

  // bs.run("CB-IJK-SRLR-L1", [&]() { cache_blocked_ijk<TILE_L1, TILE_L1, TILE_L1>(mrA, mlB, mrC); });
  // bs.run("CB-IJK-SRLR-L2", [&]() { cache_blocked_ijk<TILE_L2, TILE_L2, TILE_L2>(mrA, mlB, mrC); });
  // bs.run("CB-IKJ-SRRR-L1", [&]() { cache_blocked_ikj<TILE_L1, TILE_L1, TILE_L1>(mrA, mrB, mrC); });
  // bs.run("CB-IKJ-SRRR-L2", [&]() { cache_blocked_ikj<TILE_L2, TILE_L2, TILE_L2>(mrA, mrB, mrC); });

  // bs.run("N-IKJ-TRRR-L1", [&]() { naive_ikj(mtrA_l1, mtrB_l1, mtrC_l1); });
  // bs.run("N-IKJ-TRRR-L2", [&]() { naive_ikj(mtrA_l2, mtrB_l2, mtrC_l2); });
  // bs.run("N-IJK-TRLR-L1", [&]() { naive_ijk(mtrA_l1, mtlB_l1, mtrC_l1); });
  // bs.run("N-IJK-TRLR-L2", [&]() { naive_ijk(mtrA_l2, mtlB_l2, mtrC_l2); });

  // bs.run("DT-IJK-TRRR-L1", [&]() { data_tiled_ijk<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1); });
  // bs.run("DT-IJK-TRRR-L2", [&]() { data_tiled_ijk<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2); });
  if (selected("dt")) {
    bs.run("DT-IKJ-TRRR-L1", [&]() { data_tiled_ikj<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_dt_l1); });
  }
  if (selected("tv")) {
    bs.run("TV-IKJ-TRRR-L1", [&]() { tiled_view_ikj(mtrA_l1, mtrB_l1, mtrC_tv_l1); });
  }
  if (selected("reduce")) {
    bs.run("REDUCE-TRRR-L1", [&]() { layout_aware_reduce(mtrA_l1, mtrB_l1, mtrC_reduce_l1); });
  }
  // bs.run("DT-IKJ-TRRR-L2", [&]() { data_tiled_ikj<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2); });
  // bs.run("DT-IJK-TRLR-L1", [&]() { data_tiled_b_left_ijk<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtlB_l1, mtrC_l1); });
  // bs.run("DT-IJK-TRLR-L2", [&]() { data_tiled_b_left_ijk<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtlB_l2, mtrC_l2); });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();

  return 0;
#endif
}

} // namespace

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute GEMM benchmark\n\n"
                                "Usage: layout_tiled.benchmarks.gemm [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Matrix size [default: 512]\n"
                                "  -t, --tile <SIZE>             Tile size [default: 32]\n"
                                "  -i, --implementation <NAME>   Run one of: dt, tv, reduce, blas\n"
                                "  -o, --output <FILENAME>       Output file name\n"
                                "  -h, --help                    Show this help\n";

  static struct option long_options[] = {
    {.name = "size", .has_arg = required_argument, .flag = nullptr, .val = 'n'},
    {.name = "tile", .has_arg = required_argument, .flag = nullptr, .val = 't'},
    {.name = "implementation", .has_arg = required_argument, .flag = nullptr, .val = 'i'},
    {.name = "output", .has_arg = required_argument, .flag = nullptr, .val = 'o'},
    {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'},
    {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}
  };

  Config cfg;
  int opt{};
  while ((opt = getopt_long(argc, argv, "n:t:i:o:h", static_cast<const struct option*>(long_options), nullptr))
         != -1) {
    switch (opt) {
    case 'n':
      cfg.n_ = std::stoul(optarg);
      if (0 == cfg.n_) {
        fmt::println(stderr, "error: matrix size must be a positive integer (got {})", optarg);
        return 1;
      }
      break;
    case 't':
      cfg.t_ = std::stoul(optarg);
      if (0 == cfg.t_) {
        fmt::println(stderr, "error: tile size must be a positive integer (got {})", optarg);
        return 1;
      }
      break;
    case 'o':
      cfg.output_ = optarg;
      break;
    case 'i':
      cfg.implementation_ = optarg;
      if (cfg.implementation_ != "dt" && cfg.implementation_ != "tv" && cfg.implementation_ != "reduce"
          && cfg.implementation_ != "blas") {
        fmt::println(stderr, "error: unknown implementation '{}'", cfg.implementation_);
        return 1;
      }
      break;
    case 'h':
      fmt::print("{}", HELP_MSG);
      return 0;
    default:
      fmt::print("{}", HELP_MSG);
      return 1;
    }
  }

  return run_bench(cfg);
}
