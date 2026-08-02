// benchmarks/gemv.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <getopt.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/core.h>
#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_rtl.hpp>
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

template <class MdSpan>
class basic_tiled_view_1d {
public:
  using source_type       = MdSpan;
  using mapping_type      = typename source_type::mapping_type;
  using element_type      = typename source_type::element_type;
  using index_type        = typename source_type::index_type;
  using inner_layout_type = typename mapping_type::inner_layout_type;

  static constexpr size_t tile_extent = mapping_type::textent(0);
  using tile_type = std::mdspan<element_type, std::extents<index_type, tile_extent>, inner_layout_type>;

  explicit constexpr basic_tiled_view_1d(source_type source) noexcept : source_(source) {}

  [[nodiscard]] constexpr auto extent() const noexcept -> index_type {
    return source_.mapping().ntiles_in_extent(0);
  }

  [[nodiscard]] constexpr auto operator[](index_type tile_i) const noexcept -> tile_type {
    assert(tile_i < extent());
    const index_type i      = tile_i * static_cast<index_type>(tile_extent);
    const index_type offset = source_.mapping().tile_offset_outer(i);
    return tile_type(source_.data_handle() + offset);
  }

private:
  source_type source_;
};

template <class MdSpan>
class basic_tiled_view_2d {
public:
  using source_type       = MdSpan;
  using mapping_type      = typename source_type::mapping_type;
  using element_type      = typename source_type::element_type;
  using index_type        = typename source_type::index_type;
  using inner_layout_type = typename mapping_type::inner_layout_type;

  static constexpr size_t tile_extent_0 = mapping_type::textent(0);
  static constexpr size_t tile_extent_1 = mapping_type::textent(1);
  using tile_type
    = std::mdspan<element_type, std::extents<index_type, tile_extent_0, tile_extent_1>, inner_layout_type>;

  explicit constexpr basic_tiled_view_2d(source_type source) noexcept : source_(source) {}

  [[nodiscard]] constexpr auto extent(size_t dimension) const noexcept -> index_type {
    assert(dimension < 2);
    return source_.mapping().ntiles_in_extent(dimension);
  }

  [[nodiscard]] constexpr auto operator[](index_type tile_i, index_type tile_k) const noexcept -> tile_type {
    assert(tile_i < extent(0));
    assert(tile_k < extent(1));
    const index_type i      = tile_i * static_cast<index_type>(tile_extent_0);
    const index_type k      = tile_k * static_cast<index_type>(tile_extent_1);
    const index_type offset = source_.mapping().tile_offset_outer(i, k);
    return tile_type(source_.data_handle() + offset);
  }

private:
  source_type source_;
};

template <class MdSpan>
[[nodiscard]] constexpr auto tiled_view(MdSpan source) noexcept {
  if constexpr (MdSpan::rank() == 1) {
    return basic_tiled_view_1d<MdSpan>(source);
  } else {
    static_assert(MdSpan::rank() == 2);
    return basic_tiled_view_2d<MdSpan>(source);
  }
}

template <class Matrix, class Vector, class Output>
auto reference_gemv(const Matrix A, const Vector x, Output y) -> void {
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

template <class Matrix, class Vector, class Output>
auto tiled_view_gemv(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));

  const auto tiled_A = tiled_view(A);
  const auto tiled_x = tiled_view(x);
  auto tiled_y       = tiled_view(y);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t tile_i = 0; tile_i < tiled_y.extent(); ++tile_i) {
    auto y_tile = tiled_y[tile_i];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
    for (size_t i = 0; i < y_tile.extent(0); ++i) {
      y_tile[i] = 0.0;
    }
    for (size_t tile_k = 0; tile_k < tiled_x.extent(); ++tile_k) {
      const auto A_tile = tiled_A[tile_i, tile_k];
      const auto x_tile = tiled_x[tile_k];
      for (size_t k = 0; k < x_tile.extent(0); ++k) {
        const Type xk = x_tile[k];
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp simd
#endif
        for (size_t i = 0; i < y_tile.extent(0); ++i) {
          y_tile[i] += A_tile[i, k] * xk;
        }
      }
    }
  }
}

template <class Matrix, class Vector, class Output>
auto layout_aware_reduce(const Matrix A, const Vector x, Output y) -> void {
  assert(A.extent(1) == x.extent(0));
  assert(A.extent(0) == y.extent(0));

  enum : size_t {
    I,
    K
  };
  layout_tiled::reduce<K>(
    layout_tiled::out<I>(y, Type{0}),
    layout_tiled::in<I, K>(A),
    layout_tiled::in<K>(x),
    [](auto& result, auto matrix, auto vector) { result += matrix * vector; }
  );
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
    if (blas_threads != 4) {
      fmt::println(stderr, "error: GEMV comparison requires OPENBLAS_NUM_THREADS=4");
      return 1;
    }
  }
#endif

  const size_t n   = static_cast<size_t>(cfg.n_);
  auto standard_A  = MemorySlice<Type>(n * n);
  auto standard_x  = MemorySlice<Type>(n);
  auto reference_y = MemorySlice<Type>(n);
  auto blas_y      = MemorySlice<Type>(n);
  auto tiled_A     = MemorySlice<Type>(n * n);
  auto tiled_x     = MemorySlice<Type>(n);
  auto dt_y        = MemorySlice<Type>(n);
  auto tv_y        = MemorySlice<Type>(n);
  auto reduce_y    = MemorySlice<Type>(n);

  auto A                       = std::mdspan<Type, MatrixExts, std::layout_left>(standard_A.get(), n, n);
  auto x                       = std::mdspan<Type, VectorExts>(standard_x.get(), n);
  [[maybe_unused]] auto y_ref  = std::mdspan<Type, VectorExts>(reference_y.get(), n);
  [[maybe_unused]] auto y_blas = std::mdspan<Type, VectorExts>(blas_y.get(), n);
  auto tA                      = std::mdspan<Type, MatrixExts, MatrixLayout>(tiled_A.get(), n, n);
  auto tx                      = std::mdspan<Type, VectorExts, layout_tiled::layout_ltl<TILE_K>>(tiled_x.get(), n);
  auto y_dt                    = std::mdspan<Type, VectorExts, layout_tiled::layout_ltl<TILE_I>>(dt_y.get(), n);
  auto y_tv                    = std::mdspan<Type, VectorExts, layout_tiled::layout_ltl<TILE_I>>(tv_y.get(), n);
  auto y_reduce                = std::mdspan<Type, VectorExts, layout_tiled::layout_ltl<TILE_I>>(reduce_y.get(), n);

  ankerl::nanobench::Rng rng(42);
  for (size_t k = 0; k < n; ++k) {
    x[k]  = rng.uniform01();
    tx[k] = x[k];
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      A[i, k]  = rng.uniform01();
      tA[i, k] = A[i, k];
    }
  }

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  const double tolerance = 2.0 * static_cast<double>(n) * static_cast<double>(n);
  int result             = 0;
  reference_gemv(A, x, y_ref);
  const double reference = benchmark_utils::l2_norm(y_ref);
  data_tiled_gemv(tA, tx, y_dt);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(y_dt), "DT-TRL-L1", tolerance);
  tiled_view_gemv(tA, tx, y_tv);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(y_tv), "TV-TRL-L1", tolerance);
  layout_aware_reduce(tA, tx, y_reduce);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(y_reduce), "REDUCE-TRL-L1", tolerance);
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  blas_gemv(A, x, y_blas);
  result |= benchmark_utils::check(reference, benchmark_utils::l2_norm(y_blas), "BLAS-SL", tolerance);
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
#if defined(LAYOUT_TILED_BENCH_ENABLE_BLAS)
  if (selected("blas")) {
    benchmark.run("BLAS-SL", [&]() { blas_gemv(A, x, y_blas); });
  }
#endif
  if (selected("dt")) {
    benchmark.run("DT-TRL-L1", [&]() { data_tiled_gemv(tA, tx, y_dt); });
  }
  if (selected("tv")) {
    benchmark.run("TV-TRL-L1", [&]() { tiled_view_gemv(tA, tx, y_tv); });
  }
  if (selected("reduce")) {
    benchmark.run("REDUCE-TRL-L1", [&]() { layout_aware_reduce(tA, tx, y_reduce); });
  }

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
                                "  -i, --implementation <NAME>   Run one of: dt, tv, reduce, blas\n"
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
          && cfg.implementation_ != "blas") {
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
