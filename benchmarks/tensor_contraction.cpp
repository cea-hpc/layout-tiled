// benchmarks/tensor_contraction.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <getopt.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/core.h>
#include <iai/detail/native_tiled_reduce_policy.hpp>
#include <iai/md_reduce.hpp>
#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_rtr.hpp>
#include <nanobench.h>
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#include <omp.h>
#endif

#include "utils.hpp"

using namespace std::chrono_literals;
using Type = double;
using Exts = std::dextents<size_t, 3>;

constexpr size_t L1D_CACHE_BYTES = 48 * 1024;
constexpr size_t L2_CACHE_BYTES  = 1280 * 1024;

constexpr size_t L1_TB = 2;
constexpr size_t L1_TM = 8;
constexpr size_t L1_TN = 32;
constexpr size_t L1_TK = 16;

constexpr size_t L2_TB = 8;
constexpr size_t L2_TM = 64;
constexpr size_t L2_TN = 64;
constexpr size_t L2_TK = 64;

constexpr size_t RECT_TB = 4;
constexpr size_t RECT_TM = 16;
constexpr size_t RECT_TN = 64;
constexpr size_t RECT_TK = 32;

constexpr size_t BATCH_TB = 1;
constexpr size_t BATCH_TM = 32;
constexpr size_t BATCH_TN = 64;
constexpr size_t BATCH_TK = 32;

constexpr auto contraction_working_set_bytes(size_t TB, size_t TM, size_t TN, size_t TK) -> size_t {
  return sizeof(Type) * (TB * TM * TN + TB * TM * TK + TB * TK * TN);
}

constexpr size_t L1_WORKING_SET_BYTES    = contraction_working_set_bytes(L1_TB, L1_TM, L1_TN, L1_TK);
constexpr size_t L2_WORKING_SET_BYTES    = contraction_working_set_bytes(L2_TB, L2_TM, L2_TN, L2_TK);
constexpr size_t RECT_WORKING_SET_BYTES  = contraction_working_set_bytes(RECT_TB, RECT_TM, RECT_TN, RECT_TK);
constexpr size_t BATCH_WORKING_SET_BYTES = contraction_working_set_bytes(BATCH_TB, BATCH_TM, BATCH_TN, BATCH_TK);

static_assert(L1_WORKING_SET_BYTES <= L1D_CACHE_BYTES);
static_assert(L2_WORKING_SET_BYTES <= L2_CACHE_BYTES);
static_assert(RECT_WORKING_SET_BYTES <= L2_CACHE_BYTES);
static_assert(BATCH_WORKING_SET_BYTES <= L1D_CACHE_BYTES);
static_assert(BATCH_TB == 1);

struct Config {
  uint64_t n_ = 128;
  std::string output_;
  bool l1_only_ = false;
};

namespace {

template <class MdSpan>
concept HasInnerLayout = requires { typename MdSpan::mapping_type::inner_layout_type; };

template <class MdSpan>
concept HasRightInnerLayout
  = HasInnerLayout<MdSpan> && std::is_same_v<typename MdSpan::mapping_type::inner_layout_type, std::layout_right>;

template <class MdSpan>
concept HasLeftInnerLayout
  = HasInnerLayout<MdSpan> && std::is_same_v<typename MdSpan::mapping_type::inner_layout_type, std::layout_left>;

template <class MdSpan, class Extents>
using InnerTileMdSpan
  = std::mdspan<typename MdSpan::value_type, Extents, typename MdSpan::mapping_type::inner_layout_type>;

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC>
auto naive(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(1));
  assert(B.extent(2) == C.extent(2));

  const size_t NB = C.extent(0);
  const size_t M  = C.extent(1);
  const size_t N  = C.extent(2);
  const size_t K  = A.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t b = 0; b < NB; ++b) {
    for (size_t i = 0; i < M; ++i) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
      for (size_t j = 0; j < N; ++j) {
        C[b, i, j] = 0.0;
      }
      for (size_t k = 0; k < K; ++k) {
        const Type a = A[b, i, k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t j = 0; j < N; ++j) {
          C[b, i, j] += a * B[b, k, j];
        }
      }
    }
  }
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC>
auto naive_b_left(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(2));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(0));
  assert(B.extent(1) == C.extent(2));

  const size_t NB = C.extent(0);
  const size_t M  = C.extent(1);
  const size_t N  = C.extent(2);
  const size_t K  = A.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t b = 0; b < NB; ++b) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        Type acc = 0.0;
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t k = 0; k < K; ++k) {
          acc += A[b, i, k] * B[k, j, b];
        }
        C[b, i, j] = acc;
      }
    }
  }
}

template <size_t TB, size_t TM, size_t TN, size_t TK, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC> && IsLayoutRight<MdSpanA>
           && IsLayoutRight<MdSpanB> && IsLayoutRight<MdSpanC>
auto cache_blocked(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(1));
  assert(B.extent(2) == C.extent(2));

  const size_t NB = C.extent(0);
  const size_t M  = C.extent(1);
  const size_t N  = C.extent(2);
  const size_t K  = A.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t bb = 0; bb < NB; bb += TB) {
    for (size_t x = 0; x < M; x += TM) {
      for (size_t y = 0; y < N; y += TN) {
        const size_t bb_end = std::min(bb + TB, NB);
        const size_t x_end  = std::min(x + TM, M);
        const size_t y_end  = std::min(y + TN, N);
        const size_t ty     = y_end - y;
        for (size_t b = bb; b < bb_end; ++b) {
          for (size_t i = x; i < x_end; ++i) {
            Type* c_row = C.data_handle() + C.mapping()(b, i, y);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t j = 0; j < ty; ++j) {
              c_row[j] = 0.0;
            }
          }
        }
        for (size_t z = 0; z < K; z += TK) {
          const size_t z_end = std::min(z + TK, K);
          for (size_t b = bb; b < bb_end; ++b) {
            for (size_t i = x; i < x_end; ++i) {
              Type* c_row = C.data_handle() + C.mapping()(b, i, y);
              for (size_t k = z; k < z_end; ++k) {
                const Type a      = *(A.data_handle() + A.mapping()(b, i, k));
                const Type* b_row = B.data_handle() + B.mapping()(b, k, y);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
                for (size_t j = 0; j < ty; ++j) {
                  c_row[j] += a * b_row[j];
                }
              }
            }
          }
        }
      }
    }
  }
}

template <size_t TB, size_t TM, size_t TN, size_t TK, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC> && IsLayoutRight<MdSpanA>
           && IsLayoutLeft<MdSpanB> && IsLayoutRight<MdSpanC>
auto cache_blocked_b_left(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(2));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(0));
  assert(B.extent(1) == C.extent(2));

  const size_t NB = C.extent(0);
  const size_t M  = C.extent(1);
  const size_t N  = C.extent(2);
  const size_t K  = A.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t bb = 0; bb < NB; bb += TB) {
    for (size_t x = 0; x < M; x += TM) {
      for (size_t y = 0; y < N; y += TN) {
        const size_t bb_end = std::min(bb + TB, NB);
        const size_t x_end  = std::min(x + TM, M);
        const size_t y_end  = std::min(y + TN, N);
        for (size_t b = bb; b < bb_end; ++b) {
          for (size_t i = x; i < x_end; ++i) {
            const Type* a_row = A.data_handle() + A.mapping()(b, i, 0);
            Type* c_row       = C.data_handle() + C.mapping()(b, i, y);
            for (size_t j = y; j < y_end; ++j) {
              Type acc = 0.0;
              for (size_t z = 0; z < K; z += TK) {
                const size_t z_end = std::min(z + TK, K);
                const Type* b_col  = B.data_handle() + B.mapping()(z, j, b);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
                for (size_t k = z; k < z_end; ++k) {
                  acc += a_row[k] * b_col[k - z];
                }
              }
              c_row[j - y] = acc;
            }
          }
        }
      }
    }
  }
}

template <size_t TB, size_t TM, size_t TN, size_t TK, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC> && HasRightInnerLayout<MdSpanA>
           && HasRightInnerLayout<MdSpanB> && HasRightInnerLayout<MdSpanC>
auto data_tiled_ikj(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(1));
  assert(B.extent(2) == C.extent(2));

  using ATile = InnerTileMdSpan<MdSpanA, std::extents<size_t, TB, TM, TK>>;
  using BTile = InnerTileMdSpan<MdSpanB, std::extents<size_t, TB, TK, TN>>;
  using CTile = InnerTileMdSpan<MdSpanC, std::extents<size_t, TB, TM, TN>>;

  const size_t NB = C.extent(0);
  const size_t M  = C.extent(1);
  const size_t N  = C.extent(2);
  const size_t K  = A.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t bb = 0; bb < NB; bb += TB) {
    for (size_t x = 0; x < M; x += TM) {
      for (size_t y = 0; y < N; y += TN) {
        const size_t tb = std::min(TB, NB - bb);
        const size_t tx = std::min(TM, M - x);
        const size_t ty = std::min(TN, N - y);
        auto tC         = CTile(C.data_handle() + C.mapping().tile_offset_outer(bb, x, y));
        for (size_t b = 0; b < tb; ++b) {
          for (size_t i = 0; i < tx; ++i) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t j = 0; j < ty; ++j) {
              tC[b, i, j] = 0.0;
            }
          }
        }
        for (size_t z = 0; z < K; z += TK) {
          const size_t tz = std::min(TK, K - z);
          const auto tA   = ATile(A.data_handle() + A.mapping().tile_offset_outer(bb, x, z));
          const auto tB   = BTile(B.data_handle() + B.mapping().tile_offset_outer(bb, z, y));
          for (size_t b = 0; b < tb; ++b) {
            for (size_t i = 0; i < tx; ++i) {
              for (size_t k = 0; k < tz; ++k) {
                const Type a = tA[b, i, k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
                for (size_t j = 0; j < ty; ++j) {
                  tC[b, i, j] += a * tB[b, k, j];
                }
              }
            }
          }
        }
      }
    }
  }
}

template <size_t TB, size_t TM, size_t TN, size_t TK, class MdSpanA, class MdSpanB, class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC> && HasRightInnerLayout<MdSpanA>
           && HasLeftInnerLayout<MdSpanB> && HasRightInnerLayout<MdSpanC>
auto data_tiled_b_left_ijk(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(2));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(0));
  assert(B.extent(1) == C.extent(2));

  using ATile = InnerTileMdSpan<MdSpanA, std::extents<size_t, TB, TM, TK>>;
  using BTile = InnerTileMdSpan<MdSpanB, std::extents<size_t, TK, TN, TB>>;
  using CTile = InnerTileMdSpan<MdSpanC, std::extents<size_t, TB, TM, TN>>;

  const size_t NB = C.extent(0);
  const size_t M  = C.extent(1);
  const size_t N  = C.extent(2);
  const size_t K  = A.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t bb = 0; bb < NB; bb += TB) {
    for (size_t x = 0; x < M; x += TM) {
      for (size_t y = 0; y < N; y += TN) {
        const size_t tb = std::min(TB, NB - bb);
        const size_t tx = std::min(TM, M - x);
        const size_t ty = std::min(TN, N - y);
        auto tC         = CTile(C.data_handle() + C.mapping().tile_offset_outer(bb, x, y));
        for (size_t b = 0; b < tb; ++b) {
          for (size_t i = 0; i < tx; ++i) {
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
            for (size_t j = 0; j < ty; ++j) {
              tC[b, i, j] = 0.0;
            }
          }
        }
        for (size_t z = 0; z < K; z += TK) {
          const size_t tz = std::min(TK, K - z);
          const auto tA   = ATile(A.data_handle() + A.mapping().tile_offset_outer(bb, x, z));
          const auto tB   = BTile(B.data_handle() + B.mapping().tile_offset_outer(z, y, bb));
          for (size_t b = 0; b < tb; ++b) {
            for (size_t i = 0; i < tx; ++i) {
              for (size_t j = 0; j < ty; ++j) {
                Type acc = tC[b, i, j];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
                for (size_t k = 0; k < tz; ++k) {
                  acc += tA[b, i, k] * tB[k, j, b];
                }
                tC[b, i, j] = acc;
              }
            }
          }
        }
      }
    }
  }
}

template <
  iai::detail::native_tiled_reduction_override Override = iai::detail::native_tiled_reduction_override::automatic,
  class MdSpanA,
  class MdSpanB,
  class MdSpanC>
  requires IsMdSpan3D<MdSpanA> && IsMdSpan3D<MdSpanB> && IsMdSpan3D<MdSpanC>
auto layout_aware_reduce(const MdSpanA A, const MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == C.extent(1));
  assert(A.extent(2) == B.extent(1));
  assert(B.extent(2) == C.extent(2));

  enum : size_t {
    Batch,
    I,
    J,
    K
  };
  auto output = iai::out<Batch, I, J>(C, Type{0});
  auto lhs    = iai::in<Batch, I, K>(A);
  auto rhs    = iai::in<Batch, K, J>(B);
  if constexpr (Override == iai::detail::native_tiled_reduction_override::automatic) {
    iai::md_reduce<K>(output, lhs, rhs, [](auto& result, auto lhs_value, auto rhs_value) {
      result += lhs_value * rhs_value;
    });
  } else {
    auto updater = [](auto& result, auto lhs_value, auto rhs_value) {
      result += lhs_value * rhs_value;
    };
    auto arguments = std::forward_as_tuple(output, lhs, rhs, updater);
    iai::detail::dispatch_native_tiled_reduce<iai::detail::axis_list<K>, Override>(
      arguments,
      std::index_sequence<0, 1>{}
    );
  }
}

auto run_l1_bench(const Config& cfg) -> int {
  const auto n         = static_cast<size_t>(cfg.n_);
  const size_t ten_len = n * n * n;

  auto rA            = MemorySlice<Type>(ten_len);
  auto rB            = MemorySlice<Type>(ten_len);
  auto rC            = MemorySlice<Type>(ten_len);
  auto trA_l1        = MemorySlice<Type>(ten_len);
  auto trB_l1        = MemorySlice<Type>(ten_len);
  auto trC_l1        = MemorySlice<Type>(ten_len);
  auto trC_reduce_l1 = MemorySlice<Type>(ten_len);

  auto mrA     = std::mdspan<Type, Exts>(rA.get(), n, n, n);
  auto mrB     = std::mdspan<Type, Exts>(rB.get(), n, n, n);
  auto mrC     = std::mdspan<Type, Exts>(rC.get(), n, n, n);
  auto mtrA_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TM, L1_TK>>(trA_l1.get(), n, n, n);
  auto mtrB_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TK, L1_TN>>(trB_l1.get(), n, n, n);
  auto mtrC_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TM, L1_TN>>(trC_l1.get(), n, n, n);
  auto mtrC_reduce_l1
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TM, L1_TN>>(trC_reduce_l1.get(), n, n, n);

  ankerl::nanobench::Rng rng_naive(42);
  benchmark_utils::fill_3d_uniform01(mrA, rng_naive);
  benchmark_utils::fill_3d_uniform01(mrB, rng_naive);
  benchmark_utils::fill_3d_copy(mtrA_l1, mrA);
  benchmark_utils::fill_3d_copy(mtrB_l1, mrB);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  const auto dn    = static_cast<double>(n);
  const double tol = 2.0 * dn * dn * dn * dn;
  int res          = 0;

  naive(mrA, mrB, mrC);
  const double ref = benchmark_utils::l2_norm(mrC);
  cache_blocked<L1_TB, L1_TM, L1_TN, L1_TK>(mrA, mrB, mrC);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "CB-IKJ-SRRR-L1", tol);
  data_tiled_ikj<L1_TB, L1_TM, L1_TN, L1_TK>(mtrA_l1, mtrB_l1, mtrC_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "DT-IKJ-TRRR-L1", tol);
  layout_aware_reduce(mtrA_l1, mtrB_l1, mtrC_reduce_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "IAI-AUTO-TRRR-L1", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(
    mtrA_l1,
    mtrB_l1,
    mtrC_reduce_l1
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "IAI-COMPLETE-TRRR-L1", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
    mtrA_l1,
    mtrB_l1,
    mtrC_reduce_l1
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "IAI-BLOCKED-TRRR-L1", tol);
  return res;
#else
  ankerl::nanobench::Bench bs;
  bs.output(nullptr);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
  bs.title(fmt::format("{0}x{0}x{0} tensor contraction {1}C", cfg.n_, omp_get_max_threads()))
    .timeUnit(1s, "s")
    .relative(true);
#else
  bs.title(fmt::format("{0}x{0}x{0} tensor contraction 1C", cfg.n_)).timeUnit(1s, "s").relative(true);
#endif

  bs.run("CB-IKJ-SRRR-L1", [&]() { cache_blocked<L1_TB, L1_TM, L1_TN, L1_TK>(mrA, mrB, mrC); });
  bs.run("DT-IKJ-TRRR-L1", [&]() { data_tiled_ikj<L1_TB, L1_TM, L1_TN, L1_TK>(mtrA_l1, mtrB_l1, mtrC_l1); });
  bs.run("IAI-AUTO-TRRR-L1", [&]() { layout_aware_reduce(mtrA_l1, mtrB_l1, mtrC_reduce_l1); });
  bs.run("IAI-COMPLETE-TRRR-L1", [&]() {
    layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(
      mtrA_l1,
      mtrB_l1,
      mtrC_reduce_l1
    );
  });
  bs.run("IAI-BLOCKED-TRRR-L1", [&]() {
    layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
      mtrA_l1,
      mtrB_l1,
      mtrC_reduce_l1
    );
  });

  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  return 0;
#endif
}

auto run_bench(const Config& cfg) -> int {
  const auto n         = static_cast<size_t>(cfg.n_);
  const size_t ten_len = n * n * n;

  auto rA = MemorySlice<Type>(ten_len);
  auto rB = MemorySlice<Type>(ten_len);
  auto rC = MemorySlice<Type>(ten_len);

  // auto trA_l1        = MemorySlice<Type>(ten_len);
  // auto trB_l1        = MemorySlice<Type>(ten_len);
  // auto trC_l1        = MemorySlice<Type>(ten_len);
  // auto trC_reduce_l1 = MemorySlice<Type>(ten_len);

  auto trA_l2 = MemorySlice<Type>(ten_len);
  auto trB_l2 = MemorySlice<Type>(ten_len);
  // auto trC_l2        = MemorySlice<Type>(ten_len);
  auto trC_reduce_l2 = MemorySlice<Type>(ten_len);

  // auto trA_rect        = MemorySlice<Type>(ten_len);
  // auto trB_rect        = MemorySlice<Type>(ten_len);
  // auto trC_rect        = MemorySlice<Type>(ten_len);
  // auto trC_reduce_rect = MemorySlice<Type>(ten_len);

  auto trA_batch = MemorySlice<Type>(ten_len);
  auto trB_batch = MemorySlice<Type>(ten_len);
  auto trC_batch = MemorySlice<Type>(ten_len);
  // auto trC_reduce_batch = MemorySlice<Type>(ten_len);

  auto mrA = std::mdspan<Type, Exts>(rA.get(), n, n, n);
  auto mrB = std::mdspan<Type, Exts>(rB.get(), n, n, n);
  auto mrC = std::mdspan<Type, Exts>(rC.get(), n, n, n);

  // auto mtrA_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TM, L1_TK>>(trA_l1.get(), n, n, n);
  // auto mtrB_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TK, L1_TN>>(trB_l1.get(), n, n, n);
  // auto mtrC_l1 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TM, L1_TN>>(trC_l1.get(), n, n, n);
  // auto mtrC_reduce_l1
  //   = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L1_TB, L1_TM, L1_TN>>(trC_reduce_l1.get(), n, n, n);

  auto mtrA_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L2_TB, L2_TM, L2_TK>>(trA_l2.get(), n, n, n);
  auto mtrB_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L2_TB, L2_TK, L2_TN>>(trB_l2.get(), n, n, n);
  // auto mtrC_l2 = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L2_TB, L2_TM, L2_TN>>(trC_l2.get(), n, n, n);
  auto mtrC_reduce_l2
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<L2_TB, L2_TM, L2_TN>>(trC_reduce_l2.get(), n, n, n);

  // auto mtrA_rect
  //   = std::mdspan<Type, Exts, layout_tiled::layout_rtr<RECT_TB, RECT_TM, RECT_TK>>(trA_rect.get(), n, n, n);
  // auto mtrB_rect
  //   = std::mdspan<Type, Exts, layout_tiled::layout_rtr<RECT_TB, RECT_TK, RECT_TN>>(trB_rect.get(), n, n, n);
  // auto mtrC_rect
  //   = std::mdspan<Type, Exts, layout_tiled::layout_rtr<RECT_TB, RECT_TM, RECT_TN>>(trC_rect.get(), n, n, n);
  // auto mtrC_reduce_rect
  //   = std::mdspan<Type, Exts, layout_tiled::layout_rtr<RECT_TB, RECT_TM, RECT_TN>>(trC_reduce_rect.get(), n, n, n);

  auto mtrA_batch
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<BATCH_TB, BATCH_TM, BATCH_TK>>(trA_batch.get(), n, n, n);
  auto mtrB_batch
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<BATCH_TB, BATCH_TK, BATCH_TN>>(trB_batch.get(), n, n, n);
  auto mtrC_batch
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<BATCH_TB, BATCH_TM, BATCH_TN>>(trC_batch.get(), n, n, n);
  // auto mtrC_reduce_batch
  // = std::mdspan<Type, Exts, layout_tiled::layout_rtr<BATCH_TB, BATCH_TM, BATCH_TN>>(trC_reduce_batch.get(), n, n,
  // n);

  ankerl::nanobench::Rng rng_naive(42);
  benchmark_utils::fill_3d_uniform01(mrA, rng_naive);
  benchmark_utils::fill_3d_uniform01(mrB, rng_naive);
  // benchmark_utils::fill_3d_copy(mtrA_l1, mrA);
  // benchmark_utils::fill_3d_copy(mtrB_l1, mrB);
  benchmark_utils::fill_3d_copy(mtrA_l2, mrA);
  benchmark_utils::fill_3d_copy(mtrB_l2, mrB);
  // benchmark_utils::fill_3d_copy(mtrA_rect, mrA);
  // benchmark_utils::fill_3d_copy(mtrB_rect, mrB);
  benchmark_utils::fill_3d_copy(mtrA_batch, mrA);
  benchmark_utils::fill_3d_copy(mtrB_batch, mrB);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  const auto dn    = static_cast<double>(n);
  const double tol = 2.0 * dn * dn * dn * dn;
  int res          = 0;

  naive(mrA, mrB, mrC);
  const double ref = benchmark_utils::l2_norm(mrC);

  cache_blocked<L1_TB, L1_TM, L1_TN, L1_TK>(mrA, mrB, mrC);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "CB-IKJ-SRRR-L1", tol);
  cache_blocked<L2_TB, L2_TM, L2_TN, L2_TK>(mrA, mrB, mrC);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "CB-IKJ-SRRR-L2", tol);
  cache_blocked<RECT_TB, RECT_TM, RECT_TN, RECT_TK>(mrA, mrB, mrC);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "CB-IKJ-SRRR-RECT", tol);
  cache_blocked<BATCH_TB, BATCH_TM, BATCH_TN, BATCH_TK>(mrA, mrB, mrC);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "CB-IKJ-SRRR-BATCH1", tol);

  data_tiled_ikj<L1_TB, L1_TM, L1_TN, L1_TK>(mtrA_l1, mtrB_l1, mtrC_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "DT-IKJ-TRRR-L1", tol);
  data_tiled_ikj<L2_TB, L2_TM, L2_TN, L2_TK>(mtrA_l2, mtrB_l2, mtrC_l2);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "DT-IKJ-TRRR-L2", tol);
  data_tiled_ikj<RECT_TB, RECT_TM, RECT_TN, RECT_TK>(mtrA_rect, mtrB_rect, mtrC_rect);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_rect), "DT-IKJ-TRRR-RECT", tol);
  data_tiled_ikj<BATCH_TB, BATCH_TM, BATCH_TN, BATCH_TK>(mtrA_batch, mtrB_batch, mtrC_batch);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_batch), "DT-IKJ-TRRR-BATCH1", tol);
  layout_aware_reduce(mtrA_l1, mtrB_l1, mtrC_reduce_l1);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "IAI-AUTO-TRRR-L1", tol);
  layout_aware_reduce(mtrA_l2, mtrB_l2, mtrC_reduce_l2);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l2), "IAI-AUTO-TRRR-L2", tol);
  layout_aware_reduce(mtrA_rect, mtrB_rect, mtrC_reduce_rect);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_rect), "IAI-AUTO-TRRR-RECT", tol);
  layout_aware_reduce(mtrA_batch, mtrB_batch, mtrC_reduce_batch);
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_batch), "IAI-AUTO-TRRR-BATCH1", tol);

  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(
    mtrA_l1,
    mtrB_l1,
    mtrC_reduce_l1
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "IAI-COMPLETE-TRRR-L1", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(
    mtrA_l2,
    mtrB_l2,
    mtrC_reduce_l2
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l2), "IAI-COMPLETE-TRRR-L2", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(
    mtrA_rect,
    mtrB_rect,
    mtrC_reduce_rect
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_rect), "IAI-COMPLETE-TRRR-RECT", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(
    mtrA_batch,
    mtrB_batch,
    mtrC_reduce_batch
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_batch), "IAI-COMPLETE-TRRR-BATCH1", tol);

  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
    mtrA_l1,
    mtrB_l1,
    mtrC_reduce_l1
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l1), "IAI-BLOCKED-TRRR-L1", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
    mtrA_l2,
    mtrB_l2,
    mtrC_reduce_l2
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_l2), "IAI-BLOCKED-TRRR-L2", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
    mtrA_rect,
    mtrB_rect,
    mtrC_reduce_rect
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_rect), "IAI-BLOCKED-TRRR-RECT", tol);
  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
    mtrA_batch,
    mtrB_batch,
    mtrC_reduce_batch
  );
  res |= benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_reduce_batch), "IAI-BLOCKED-TRRR-BATCH1", tol);

  return res;
#else
  ankerl::nanobench::Bench bs;
  bs.output(nullptr);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
  bs.title(fmt::format("{0}x{0}x{0} tensor contraction {1}C", cfg.n_, omp_get_max_threads()))
    .timeUnit(1s, "s")
    .relative(true);
#else
  bs.title(fmt::format("{0}x{0}x{0} tensor contraction 1C", cfg.n_)).timeUnit(1s, "s").relative(true);
#endif
  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }

  if (cfg.n_ <= 256) {
    bs.run("N-IKJ-SRRR", [&]() { naive(mrA, mrB, mrC); });
    bs.run("CB-IKJ-SRRR", [&]() { cache_blocked<BATCH_TB, BATCH_TM, BATCH_TN, BATCH_TK>(mrA, mrB, mrC); });
  }

  // bs.run("CB-IKJ-SRRR-L1", [&]() { cache_blocked<L1_TB, L1_TM, L1_TN, L1_TK>(mrA, mrB, mrC); });
  // bs.run("CB-IKJ-SRRR-L2", [&]() { cache_blocked<L2_TB, L2_TM, L2_TN, L2_TK>(mrA, mrB, mrC); });
  // bs.run("CB-IKJ-SRRR-RECT", [&]() { cache_blocked<RECT_TB, RECT_TM, RECT_TN, RECT_TK>(mrA, mrB, mrC); });
  // bs.run("CB-IKJ-SRRR-BATCH1", [&]() { cache_blocked<BATCH_TB, BATCH_TM, BATCH_TN, BATCH_TK>(mrA, mrB, mrC); });

  // bs.run("DT-IKJ-TRRR-L1", [&]() { data_tiled_ikj<L1_TB, L1_TM, L1_TN, L1_TK>(mtrA_l1, mtrB_l1, mtrC_l1); });
  // bs.run("DT-IKJ-TRRR-L2", [&]() { data_tiled_ikj<L2_TB, L2_TM, L2_TN, L2_TK>(mtrA_l2, mtrB_l2, mtrC_l2); });
  // bs.run("DT-IKJ-TRRR-RECT", [&]() {
  //   data_tiled_ikj<RECT_TB, RECT_TM, RECT_TN, RECT_TK>(mtrA_rect, mtrB_rect, mtrC_rect);
  // });
  bs.run("CBDT-IKJ-TRRR", [&]() {
    data_tiled_ikj<BATCH_TB, BATCH_TM, BATCH_TN, BATCH_TK>(mtrA_batch, mtrB_batch, mtrC_batch);
  });
  // bs.run("IAI-TRRR-L1", [&]() { layout_aware_reduce(mtrA_l1, mtrB_l1, mtrC_reduce_l1); });
  bs.run("IAI-TRRR", [&]() { layout_aware_reduce(mtrA_l2, mtrB_l2, mtrC_reduce_l2); });
  // bs.run("IAI-TRRR-RECT", [&]() { layout_aware_reduce(mtrA_rect, mtrB_rect, mtrC_reduce_rect); });
  // bs.run("IAI-TRRR-BATCH1", [&]() { layout_aware_reduce(mtrA_batch, mtrB_batch, mtrC_reduce_batch); });

  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();

  return 0;
#endif
}

} // namespace

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute 3D tensor contraction benchmark\n\n"
                                "Computes C[b,i,j] = sum_k A[b,i,k] * B[b,k,j].\n\n"
                                "Usage: layout_tiled.benchmarks.tensor_contraction [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Tensor extent [default: 128]\n"
                                "  -l, --l1-only                 Run only the L1 configurations\n"
                                "  -o, --output <FILENAME>       Output file name\n"
                                "  -h, --help                    Show this help\n";

  static struct option long_options[] = {
    {.name = "size", .has_arg = required_argument, .flag = nullptr, .val = 'n'},
    {.name = "l1-only", .has_arg = no_argument, .flag = nullptr, .val = 'l'},
    {.name = "output", .has_arg = required_argument, .flag = nullptr, .val = 'o'},
    {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'},
    {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}
  };

  Config cfg;
  int opt{};
  while ((opt = getopt_long(argc, argv, "n:lo:h", static_cast<const struct option*>(long_options), nullptr)) != -1) {
    switch (opt) {
    case 'n':
      cfg.n_ = std::stoul(optarg);
      if (0 == cfg.n_) {
        fmt::println(stderr, "error: tensor extent must be a positive integer (got {})", optarg);
        return 1;
      }
      break;
    case 'l':
      cfg.l1_only_ = true;
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

  if (0 != cfg.n_ % L1_TB || 0 != cfg.n_ % L1_TM || 0 != cfg.n_ % L1_TN || 0 != cfg.n_ % L1_TK) {
    fmt::println(
      stderr,
      "error: tensor extent must be divisible by the L1 tile extents {}x{}x{}x{} (got {})",
      L1_TB,
      L1_TM,
      L1_TN,
      L1_TK,
      cfg.n_
    );
    return 1;
  }

  if (cfg.l1_only_ || cfg.n_ == 32) {
    if (cfg.n_ == 32) {
      fmt::println(stderr, "Tensor extent 32: running the compatible L1 variants only");
    } else {
      fmt::println(stderr, "Tensor extent {}: --l1-only selected", cfg.n_);
    }
    return run_l1_bench(cfg);
  }

  if (0 != cfg.n_ % L2_TB || 0 != cfg.n_ % L2_TM || 0 != cfg.n_ % L2_TN || 0 != cfg.n_ % L2_TK) {
    fmt::println(
      stderr,
      "error: tensor extent must be divisible by the L2 tile extents {}x{}x{}x{} (got {})",
      L2_TB,
      L2_TM,
      L2_TN,
      L2_TK,
      cfg.n_
    );
    return 1;
  }

  if (0 != cfg.n_ % RECT_TB || 0 != cfg.n_ % RECT_TM || 0 != cfg.n_ % RECT_TN || 0 != cfg.n_ % RECT_TK) {
    fmt::println(
      stderr,
      "error: tensor extent must be divisible by the rectangular tile extents {}x{}x{}x{} (got {})",
      RECT_TB,
      RECT_TM,
      RECT_TN,
      RECT_TK,
      cfg.n_
    );
    return 1;
  }

  if (0 != cfg.n_ % BATCH_TB || 0 != cfg.n_ % BATCH_TM || 0 != cfg.n_ % BATCH_TN || 0 != cfg.n_ % BATCH_TK) {
    fmt::println(
      stderr,
      "error: tensor extent must be divisible by the batch-one tile extents {}x{}x{}x{} (got {})",
      BATCH_TB,
      BATCH_TM,
      BATCH_TN,
      BATCH_TK,
      cfg.n_
    );
    return 1;
  }

  return run_bench(cfg);
}
