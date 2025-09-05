// benchmarks/gemm.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <getopt.h>
#include <immintrin.h>

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
};

namespace {

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto naive(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
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

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto cache_blocked(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
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
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto manual_blocked_tiled(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
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
    const size_t tx = std::min(TM, M - x);
    for (size_t y = 0; y < N; y += TN) {
      double* tC      = C.data_handle() + C.mapping().tile_offset_outer(x, y);
      const size_t ty = std::min(TN, N - y);
      for (size_t z = 0; z < O; z += TO) {
        const size_t tz  = std::min(TO, O - z);
        const double* tA = A.data_handle() + A.mapping().tile_offset_outer(x, z);
        const double* tB = B.data_handle() + B.mapping().tile_offset_outer(z, y);
        for (size_t i = 0; i < tx; ++i) {
          size_t A_inoff = i * TO;
          size_t C_inoff = i * TN;
          for (size_t j = 0; j < ty; ++j) {
            Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < tz; ++k) {
              acc += tA[A_inoff + k] * tB[k * TN + j];
            }
            tC[C_inoff + j] += acc;
          }
        }
      }
    }
  }
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto manual_blocked_tiled(const MdSpanA A, const MdSpanB B, MdSpanC C, size_t TM, size_t TN, size_t TO) -> void {
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
    const size_t tx = std::min(TM, M - x);
    for (size_t y = 0; y < N; y += TN) {
      double* tC      = C.data_handle() + C.mapping().tile_offset_outer(x, y);
      const size_t ty = std::min(TN, N - y);
      for (size_t z = 0; z < O; z += TO) {
        const size_t tz  = std::min(TO, O - z);
        const double* tA = A.data_handle() + A.mapping().tile_offset_outer(x, z);
        const double* tB = B.data_handle() + B.mapping().tile_offset_outer(z, y);
        for (size_t i = 0; i < tx; ++i) {
          size_t A_inoff = i * TO;
          size_t C_inoff = i * TN;
          for (size_t j = 0; j < ty; ++j) {
            Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < tz; ++k) {
              acc += tA[A_inoff + k] * tB[k * TN + j];
            }
            tC[C_inoff + j] += acc;
          }
        }
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto manual_blocked_tiled_subview(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  using ATile = std::mdspan<Type, std::extents<size_t, TM, TO>, typename MdSpanA::mapping_type::inner_layout_type>;
  using BTile = std::mdspan<Type, std::extents<size_t, TO, TN>, typename MdSpanB::mapping_type::inner_layout_type>;
  using CTile = std::mdspan<Type, std::extents<size_t, TM, TN>, typename MdSpanC::mapping_type::inner_layout_type>;

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

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan2D<MdSpanA> && IsMdSpan2D<MdSpanB> && IsMdSpan2D<MdSpanC>
auto manual_blocked_tiled_subview_unroll_vec_k(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  using ATile = std::mdspan<Type, std::extents<size_t, TM, TO>, typename MdSpanA::mapping_type::inner_layout_type>;
  using BTile = std::mdspan<Type, std::extents<size_t, TO, TN>, typename MdSpanB::mapping_type::inner_layout_type>;
  using CTile = std::mdspan<Type, std::extents<size_t, TM, TN>, typename MdSpanC::mapping_type::inner_layout_type>;

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
#if defined(__AVX__) && defined(__FMA__)
            // 8x unrolled and vectorized kernel (processing 32 elements/iter).
            // If TO == 32, this effectively fully unrolls the `k` loop.
            //
            // On Zen3, `vfmaddXXXpd` is 4 lat / 0.5 tp (CPI) => at least 2 FMAs needed to achieve max throughput.
            // Unrolling more than that allows to reduce the cost of the horizontal reduction at the end.
            //
            // TODO(dssgabriel): Attempt to vectorize the `j` loop to remove the need for the horizontal reduction.
            if constexpr (TO >= 32 and TO % 32 == 0) {
              __m256d vacc = _mm256_setzero_pd();
              for (size_t k = 0; k < std::min(TO, O - z); k += 32) {
                __m256d va0 = _mm256_load_pd(&tA[i, k]);
                __m256d vb0 = _mm256_load_pd(&tB[k, j]);
                vacc        = _mm256_fmadd_pd(va0, vb0, vacc);
                __m256d va1 = _mm256_load_pd(&tA[i, k + 4]);
                __m256d vb1 = _mm256_load_pd(&tB[k + 4, j]);
                vacc        = _mm256_fmadd_pd(va1, vb1, vacc);
                __m256d va2 = _mm256_load_pd(&tA[i, k + 8]);
                __m256d vb2 = _mm256_load_pd(&tB[k + 8, j]);
                vacc        = _mm256_fmadd_pd(va2, vb2, vacc);
                __m256d va3 = _mm256_load_pd(&tA[i, k + 12]);
                __m256d vb3 = _mm256_load_pd(&tB[k + 12, j]);
                vacc        = _mm256_fmadd_pd(va3, vb3, vacc);
                __m256d va4 = _mm256_load_pd(&tA[i, k + 16]);
                __m256d vb4 = _mm256_load_pd(&tB[k + 16, j]);
                vacc        = _mm256_fmadd_pd(va4, vb4, vacc);
                __m256d va5 = _mm256_load_pd(&tA[i, k + 20]);
                __m256d vb5 = _mm256_load_pd(&tB[k + 20, j]);
                vacc        = _mm256_fmadd_pd(va5, vb5, vacc);
                __m256d va6 = _mm256_load_pd(&tA[i, k + 24]);
                __m256d vb6 = _mm256_load_pd(&tB[k + 24, j]);
                vacc        = _mm256_fmadd_pd(va6, vb6, vacc);
                __m256d va7 = _mm256_load_pd(&tA[i, k + 28]);
                __m256d vb7 = _mm256_load_pd(&tB[k + 28, j]);
                vacc        = _mm256_fmadd_pd(va7, vb7, vacc);
              }

              __m128d hsum = _mm_add_pd(_mm256_castpd256_pd128(vacc), _mm256_extractf128_pd(vacc, 1));
              hsum         = _mm_hadd_pd(hsum, hsum);
              acc          = _mm_cvtsd_f64(hsum);
            } else {
              for (size_t k = 0; k < std::min(TO, O - z); ++k) {
                acc += tA[i, k] * tB[k, j];
              }
            }
            tC[i, j] += acc;
#else
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < std::min(TO, O - z); ++k) {
              acc += tA[i, k] * tB[k, j];
            }
            tC[i, j] += acc;
#endif
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
  // Standard layout matrices
  auto rA = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto rB = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto rC = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lB = MemorySlice<Type>(cfg.n_ * cfg.n_);
  // Tiled layout matrices (L1 tiles)
  auto trA_l1 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trB_l1 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trC_l1 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto tlB_l1 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  // Tiled layout matrices (L2 tiles)
  auto trA_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trB_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto trC_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto tlB_l2 = MemorySlice<Type>(cfg.n_ * cfg.n_);

  // Associated MDSPANs
  auto mrA     = std::mdspan<Type, Exts>(rA.get(), cfg.n_, cfg.n_);
  auto mrB     = std::mdspan<Type, Exts>(rB.get(), cfg.n_, cfg.n_);
  auto mrC     = std::mdspan<Type, Exts>(rC.get(), cfg.n_, cfg.n_);
  auto mlB     = std::mdspan<Type, Exts, std::layout_left>(lB.get(), cfg.n_, cfg.n_);
  auto mtrA_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trA_l1.get(), cfg.n_, cfg.n_);
  auto mtrB_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trB_l1.get(), cfg.n_, cfg.n_);
  auto mtrC_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(trC_l1.get(), cfg.n_, cfg.n_);
  auto mtlB_l1 = std::mdspan<Type, Exts, layout_tiled::layout_ltl<TILE_L1, TILE_L1>>(tlB_l1.get(), cfg.n_, cfg.n_);
  auto mtrA_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(trA_l2.get(), cfg.n_, cfg.n_);
  auto mtrB_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(trB_l2.get(), cfg.n_, cfg.n_);
  auto mtrC_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(trC_l2.get(), cfg.n_, cfg.n_);
  auto mtlB_l2 = std::mdspan<Type, Exts, layout_tiled::layout_ltl<TILE_L2, TILE_L2>>(tlB_l2.get(), cfg.n_, cfg.n_);

  // Matrix initializations
  ankerl::nanobench::Rng rng(42);
  benchmark_utils::fill_2d_uniform01(mrA, rng);
  benchmark_utils::fill_2d_uniform01(mrB, rng);
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

  naive(mrA, mrB, mrC);
  double ref = benchmark_utils::l2_norm(mrC);
  benchmark_utils::fill_zeroes(mrC);

  cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mrA, mrB, mrC);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "R_cb-L1", big_o);
  benchmark_utils::fill_zeroes(mrC);
  cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mrA, mrB, mrC);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "R_cb-L2", big_o);
  benchmark_utils::fill_zeroes(mrC);

  naive(mtrA_l1, mtrB_l1, mtrC_l1);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-L1_naive", big_o);
  benchmark_utils::fill_zeroes(mtrC_l1);
  naive(mtrA_l2, mtrB_l2, mtrC_l2);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-L2_naive", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  manual_blocked_tiled<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-L1_mbt", big_o);
  benchmark_utils::fill_zeroes(mtrC_l1);
  manual_blocked_tiled<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-L2_mbt", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  manual_blocked_tiled_subview<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-L1_mbtsv", big_o);
  benchmark_utils::fill_zeroes(mtrC_l1);
  manual_blocked_tiled_subview<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-L2_mbtsv", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  manual_blocked_tiled_subview<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtlB_l1, mtrC_l1);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-BLTL-L1_mbtsv", big_o);
  benchmark_utils::fill_zeroes(mtrC_l1);
  manual_blocked_tiled_subview<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtlB_l2, mtrC_l2);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-BLTL-L2_mbtsv", big_o);
  benchmark_utils::fill_zeroes(mtrC_l2);

  manual_blocked_tiled_subview_unroll_vec_k<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtlB_l1, mtrC_l1);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-BLTL-L1_mbtsv_uvk", big_o);
  benchmark_utils::fill_zeroes(mtrC_l1);
  manual_blocked_tiled_subview_unroll_vec_k<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtlB_l2, mtrC_l2);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-BLTL-L2_mbtsv_uvk", big_o);
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

  /*
  bs.run("R naive", [&]() { naive(mrA, mrB, mrC); });
  bs.run("R_BL naive", [&]() { naive(mrA, mlB, mrC); });
  */
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  bs.run("R blas", [&]() { blas(mrA, mrB, mrC); });
#endif

  /*
  bs.run(fmt::format("R cb-{0}x{0}", TILE_L1), [&]() { cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mrA, mrB, mrC); });
  bs.run(fmt::format("R cb-{0}x{0}", TILE_L2), [&]() { cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mrA, mrB, mrC); });
  bs.run(fmt::format("R_BL cb-{0}x{0}", TILE_L1), [&]() { cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mrA, mlB, mrC); });
  bs.run(fmt::format("R_BL cb-{0}x{0}", TILE_L2), [&]() { cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mrA, mlB, mrC); });

  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L1), [&]() { naive(mtrA_l1, mtrB_l1, mtrC_l1); });
  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L2), [&]() { naive(mtrA_l2, mtrB_l2, mtrC_l2); });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} naive", TILE_L1), [&]() { naive(mtrA_l1, mtlB_l1, mtrC_l1); });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} naive", TILE_L2), [&]() { naive(mtrA_l2, mtlB_l2, mtrC_l2); });

  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L1), [&]() {
    manual_blocked_tiled<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L2), [&]() {
    manual_blocked_tiled<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt (runtime)", cfg.t_), [&]() {
    manual_blocked_tiled(mtrA_l1, mtrB_l1, mtrC_l1, cfg.t_, cfg.t_, cfg.t_);
  });

  bs.run(fmt::format("RTR-{0}x{0} mbtsv", TILE_L1), [&]() {
    manual_blocked_tiled_subview<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbtsv", TILE_L2), [&]() {
    manual_blocked_tiled_subview<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  });
  */
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbtsv", TILE_L1), [&]() {
    manual_blocked_tiled_subview<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtlB_l1, mtrC_l1);
  });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbtsv", TILE_L2), [&]() {
    manual_blocked_tiled_subview<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtlB_l2, mtrC_l2);
  });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbtsv_uvk", TILE_L1), [&]() {
    manual_blocked_tiled_subview_unroll_vec_k<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtlB_l1, mtrC_l1);
  });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbtsv_uvk", TILE_L2), [&]() {
    manual_blocked_tiled_subview_unroll_vec_k<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtlB_l2, mtrC_l2);
  });

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
                                "  -o, --output <FILENAME>       Output file name\n"
                                "  -h, --help                    Show this help\n";

  static struct option long_options[] = {
    {.name = "size", .has_arg = required_argument, .flag = nullptr, .val = 'n'},
    {.name = "tile", .has_arg = required_argument, .flag = nullptr, .val = 't'},
    {.name = "output", .has_arg = required_argument, .flag = nullptr, .val = 'o'},
    {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'},
    {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}
  };

  Config cfg;
  int opt{};
  while ((opt = getopt_long(argc, argv, "n:t:o:h", static_cast<const struct option*>(long_options), nullptr)) != -1) {
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
