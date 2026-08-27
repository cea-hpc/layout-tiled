// benchmarks/gemv.cpp -*-C++-*-
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
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/core.h>
#include <iai/detail/native_tiled_reduce_policy.hpp>
#include <iai/md_reduce.hpp>
#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_rtl.hpp>
#include <nanobench.h>
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#include <omp.h>
#endif
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
#include <cblas.h>
#endif

#include "utils.hpp"

using namespace std::chrono_literals;
using Type       = double;
using MatrixExts = std::dextents<size_t, 2>;
using VectorExts = std::dextents<size_t, 1>;

constexpr size_t TILE_I = 8;
constexpr size_t TILE_K = 128;

using MatrixLayout  = layout_tiled::layout_rtl<TILE_I, TILE_K>;
using MatrixMapping = MatrixLayout::mapping<MatrixExts>;
static_assert(std::is_same_v<typename MatrixMapping::outer_layout_type, std::layout_right>);
static_assert(std::is_same_v<typename MatrixMapping::inner_layout_type, std::layout_left>);

struct Config {
  uint64_t n_ = 4096;
  std::string output_;
  std::string implementation_;
};

namespace {

template <class Matrix, class Vector, class Output>
auto naive_gemv(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < A.extent(0); ++i) {
    Type value = 0.0;
    for (size_t k = 0; k < A.extent(1); ++k) {
      value += A[i, k] * x[k];
    }
    y[i] = value;
  }
}

template <class Matrix, class Vector, class Output>
auto cache_blocked_gemv(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t tile_i = 0; tile_i < y.extent(0); tile_i += TILE_I) {
    const size_t i_end = std::min(tile_i + TILE_I, y.extent(0));
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
    for (size_t i = tile_i; i < i_end; ++i) {
      y[i] = 0.0;
    }
    for (size_t tile_k = 0; tile_k < x.extent(0); tile_k += TILE_K) {
      const size_t k_end = std::min(tile_k + TILE_K, x.extent(0));
      for (size_t k = tile_k; k < k_end; ++k) {
        const Type xk = x[k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t i = tile_i; i < i_end; ++i) {
          y[i] += A[i, k] * xk;
        }
      }
    }
  }
}

template <class Matrix, class Vector, class Output>
auto data_tiled_gemv(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));

  using ATile = std::mdspan<Type, std::extents<size_t, TILE_I, TILE_K>, std::layout_left>;
  using XTile = std::mdspan<Type, std::extents<size_t, TILE_K>, std::layout_left>;
  using YTile = std::mdspan<Type, std::extents<size_t, TILE_I>, std::layout_left>;

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t tile_i = 0; tile_i < y.extent(0); tile_i += TILE_I) {
    auto y_tile = YTile(y.data_handle() + y.mapping().tile_offset_outer(tile_i));
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < TILE_I; ++i) {
      y_tile[i] = 0.0;
    }
    for (size_t tile_k = 0; tile_k < x.extent(0); tile_k += TILE_K) {
      const auto A_tile = ATile(A.data_handle() + A.mapping().tile_offset_outer(tile_i, tile_k));
      const auto x_tile = XTile(x.data_handle() + x.mapping().tile_offset_outer(tile_k));
      for (size_t k = 0; k < TILE_K; ++k) {
        const Type xk = x_tile[k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t i = 0; i < TILE_I; ++i) {
          y_tile[i] += A_tile[i, k] * xk;
        }
      }
    }
  }
}

template <
  iai::detail::native_tiled_reduction_override Override = iai::detail::native_tiled_reduction_override::automatic,
  class Matrix,
  class Vector,
  class Output>
auto layout_aware_reduce(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));

  enum Axes {
    I,
    J,
  };

  auto Av = iai::in<I, J>(A);
  auto xv = iai::in<J>(x);
  auto yv = iai::out<I>(y, 0.0);

  if constexpr (Override == iai::detail::native_tiled_reduction_override::automatic) {
    iai::md_reduce<J>(yv, Av, xv, [](auto& result, auto lhs_value, auto rhs_value) {
      result += lhs_value * rhs_value;
    });
  } else {
    auto updater = [](auto& result, auto lhs_value, auto rhs_value) {
      result += lhs_value * rhs_value;
    };
    auto arguments = std::forward_as_tuple(yv, Av, xv, updater);
    iai::detail::dispatch_native_tiled_reduce<iai::detail::axis_list<J>, Override>(
      arguments,
      std::index_sequence<0, 1>{}
    );
  }
}

#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
template <class Matrix, class Vector, class Output>
auto blas_gemv(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));
  cblas_dgemv(
    CblasColMajor,
    CblasNoTrans,
    static_cast<int>(A.extent(0)),
    static_cast<int>(A.extent(1)),
    1.0,
    A.data_handle(),
    static_cast<int>(A.stride(1)),
    x.data_handle(),
    1,
    0.0,
    y.data_handle(),
    1
  );
}
#endif

auto run_bench(const Config& cfg) -> int {
#if defined(LAYOUT_TILED_BENCH_OPENBLAS)
  if (cfg.implementation_.empty() || cfg.implementation_ == "blas") {
    const int blas_threads = openblas_get_num_threads();
    fmt::println(stderr, "OpenBLAS threads: {}", blas_threads);
    if (blas_threads < 2) {
      fmt::println(stderr, "error: GEMV comparison requires OPENBLAS_NUM_THREADS>=1");
      return 1;
    }
  }
#endif

  const size_t n = static_cast<size_t>(cfg.n_);

  auto rA = MemorySlice<Type>(n * n);
  auto rx = MemorySlice<Type>(n);
  auto ry = MemorySlice<Type>(n);
  auto tA = MemorySlice<Type>(n * n);
  auto tx = MemorySlice<Type>(n);
  auto ty = MemorySlice<Type>(n);

  auto mrA                  = std::mdspan<Type, MatrixExts, std::layout_left>(rA.get(), n, n);
  auto mrx                  = std::mdspan<Type, VectorExts>(rx.get(), n);
  [[maybe_unused]] auto mry = std::mdspan<Type, VectorExts>(ry.get(), n);
  auto mtA                  = std::mdspan<Type, MatrixExts, MatrixLayout>(tA.get(), n, n);
  auto mtx                  = std::mdspan<Type, VectorExts, layout_tiled::layout_ltl<TILE_K>>(tx.get(), n);
  auto mty                  = std::mdspan<Type, VectorExts, layout_tiled::layout_ltl<TILE_I>>(ty.get(), n);

  ankerl::nanobench::Rng rng(42);
  benchmark_utils::fill_2d_uniform01(mrA, rng);
  benchmark_utils::fill_1d_uniform01(mrx, rng);
  benchmark_utils::fill_2d_copy(mtA, mrA);
  benchmark_utils::fill_1d_copy(mtx, mrx);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  const double big_o = 2.0 * static_cast<double>(n) * static_cast<double>(n);
  int result         = 0;
  naive_gemv(mrA, mrx, mry);
  const double reference = benchmark_utils::l2_norm(mry);
  benchmark_utils::fill_zeroes(mry);

  cache_blocked_gemv(mrA, mrx, mry);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(mry), "CB-TRL", big_o);
  benchmark_utils::fill_zeroes(mry);

  data_tiled_gemv(mtA, mtx, mty);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(mty), "CBDT-TRL", big_o);
  benchmark_utils::fill_zeroes(mry);

  layout_aware_reduce(mtA, mtx, mty);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(mty), "IAI-AUTO-TRL", big_o);
  benchmark_utils::fill_zeroes(mty);

  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(mtA, mtx, mty);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(mty), "IAI-COMPLETE-TRL", big_o);
  benchmark_utils::fill_zeroes(mty);

  layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(mtA, mtx, mty);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(mty), "IAI-BLOCKED-TRL", big_o);
  benchmark_utils::fill_zeroes(mty);

#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  blas_gemv(mrA, mrx, mry);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(mry), "BLAS-SRL", big_o);
  benchmark_utils::fill_zeroes(mry);
#endif
  return result;
#else
  ankerl::nanobench::Bench benchmark;
  benchmark.output(nullptr);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
  benchmark.title(fmt::format("{0}x{0} GEMV {1}C", n, omp_get_max_threads())).timeUnit(1s, "s").relative(true);
#else
  benchmark.title(fmt::format("{0}x{0} GEMV 1C", n)).timeUnit(1s, "s").relative(true);
#endif

  std::ostream* output = &std::cout;
  std::ofstream file;
  if (!cfg.output_.empty()) {
    file.open(cfg.output_, std::ios::app | std::ios::binary);
    output = &file;
  }

  const auto selected = [&](const std::string_view implementation) {
    return cfg.implementation_.empty() || cfg.implementation_ == implementation;
  };
  if (cfg.n_ <= 8192) {
    benchmark.run("N-SRL", [&]() { naive_gemv(mrA, mrx, mry); });
    benchmark.run("CB-SRL", [&]() { cache_blocked_gemv(mrA, mrx, mry); });
  }
  if (selected("dt")) {
    benchmark.run("DT-TRL", [&]() { data_tiled_gemv(mtA, mtx, mty); });
  }
  if (selected("iai-auto") || selected("reduce")) {
    benchmark.run("IAI-TRL", [&]() { layout_aware_reduce(mtA, mtx, mty); });
  }
  if (selected("iai-complete")) {
    benchmark.run("IAI-COMPLETE-TRL", [&]() {
      layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_complete_fragment>(mtA, mtx, mty);
    });
  }
  if (selected("iai-blocked")) {
    benchmark.run("IAI-BLOCKED-TRL", [&]() {
      layout_aware_reduce<iai::detail::native_tiled_reduction_override::force_reduction_blocked_fragment>(
        mtA,
        mtx,
        mty
      );
    });
  }
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  if (selected("blas")) {
    benchmark.run("BLAS-SRL", [&]() { blas_gemv(mrA, mrx, mry); });
  }
#endif

  benchmark.render(ankerl::nanobench::templates::csv(), *output);
  file.close();
  return 0;
#endif
}

} // namespace

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute GEMV benchmark\n\n"
                                "Computes y[i] = sum_k A[i,k] * x[k].\n\n"
                                "Usage: layout_tiled.benchmarks.gemv [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Matrix size [default: 4096]\n"
                                "  -i, --implementation <NAME>   Run one of: dt, tv, iai-auto, iai-complete, "
                                "iai-blocked, blas\n"
                                "  -o, --output <FILENAME>       Output file name\n"
                                "  -h, --help                    Show this help\n";

  static struct option long_options[] = {
    {.name = "size", .has_arg = required_argument, .flag = nullptr, .val = 'n'},
    {.name = "implementation", .has_arg = required_argument, .flag = nullptr, .val = 'i'},
    {.name = "output", .has_arg = required_argument, .flag = nullptr, .val = 'o'},
    {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'},
    {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}
  };

  Config cfg;
  int option{};
  while ((option = getopt_long(argc, argv, "n:i:o:h", long_options, nullptr)) != -1) {
    switch (option) {
    case 'n':
      cfg.n_ = std::stoul(optarg);
      if (cfg.n_ == 0) {
        fmt::println(stderr, "error: matrix size must be a positive integer (got {})", optarg);
        return 1;
      }
      break;
    case 'i':
      cfg.implementation_ = optarg;
      if (cfg.implementation_ != "dt" && cfg.implementation_ != "tv" && cfg.implementation_ != "reduce"
          && cfg.implementation_ != "iai-auto" && cfg.implementation_ != "iai-complete"
          && cfg.implementation_ != "iai-blocked" && cfg.implementation_ != "blas") {
        fmt::println(stderr, "error: unknown implementation '{}'", cfg.implementation_);
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

  if (cfg.n_ % TILE_I != 0 || cfg.n_ % TILE_K != 0) {
    fmt::println(
      stderr,
      "error: matrix size must be divisible by the tile extents {}x{} (got {})",
      TILE_I,
      TILE_K,
      cfg.n_
    );
    return 1;
  }
  return run_bench(cfg);
}
