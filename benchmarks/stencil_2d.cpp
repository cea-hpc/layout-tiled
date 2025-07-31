// benchmarks/stencil_2d.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <getopt.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <experimental/mdspan>
#include <fstream>
#include <iostream>
#include <string>

#include <fmt/core.h>
#include <layout_tiled/layout_rtr.hpp>
#include <nanobench.h>
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#include <omp.h>
#endif

#include "utils.hpp"

using namespace std::chrono_literals;
using Type = double;
using Exts = std::dextents<size_t, 2>;

constexpr size_t TILE_L1 = 64;
constexpr size_t TILE_L2 = 256;
constexpr size_t DELTA   = 1;

struct Config {
  uint64_t n_ = 512;
  uint64_t t_ = 64;
  uint64_t d_ = 1;
  std::string output_;
};

namespace {

template <class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto naive(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = D; i < M - D; ++i) {
    for (size_t j = D; j < N - D; ++j) {
      // clang-format off
      out[i, j] =              in[i-D, j] +
                  in[i, j-D] + in[  i, j] + in[i, j+D] +
                               in[i+D, j];
      // clang-format on
    }
  }
}

template <size_t TM, size_t TN, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto cache_blocked(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      for (size_t i = x; i < std::min(x + TM, M - D); ++i) {
        for (size_t j = y; j < std::min(y + TN, N - D); ++j) {
          // clang-format off
          out[i, j] =              in[i-D, j] +
                      in[i, j-D] + in[  i, j] + in[i, j+D] +
                                   in[i+D, j];
          // clang-format on
        }
      }
    }
  }
}

template <class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto cache_blocked(MdSpanIn in, MdSpanOut out, size_t TM, size_t TN, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      for (size_t i = x; i < std::min(x + TM, M - D); ++i) {
        for (size_t j = y; j < std::min(y + TN, N - D); ++j) {
          // clang-format off
          out[i, j] =              in[i-D, j] +
                      in[i, j-D] + in[  i, j] + in[i, j+D] +
                                   in[i+D, j];
          // clang-format on
        }
      }
    }
  }
}

template <size_t TM, size_t TN, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto manual_blocked_tiled(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

  auto hin  = in.data_handle();
  auto hout = out.data_handle();
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      const size_t outoff = in.mapping().tile_offset_outer(x, y);
      for (size_t i = 0; i < std::min(TM, M - D - x); ++i) {
        const size_t ti    = TM * i;
        const size_t ti_mD = TM * (i - D);
        const size_t ti_pD = TM * (i + D);
        for (size_t j = 0; j < std::min(TN, N - D - y); ++j) {
          const size_t inoff      = ti + j;
          const size_t baseoff    = outoff + inoff;
          const size_t inoff_mD   = ti_mD + j;
          const size_t inoff_pD   = ti_pD + j;
          const size_t baseoff_mD = outoff + inoff_mD;
          const size_t baseoff_pD = outoff + inoff_pD;
          // clang-format off
          hout[baseoff] =                  hin[baseoff_mD] +
                          hin[baseoff-D] + hin[baseoff]   + hin[baseoff+D] +
                                           hin[baseoff_pD];
          // clang-format on
        }
      }
    }
  }
}

template <class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto manual_blocked_tiled(MdSpanIn in, MdSpanOut out, size_t TM, size_t TN, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

  auto hin  = in.data_handle();
  auto hout = out.data_handle();
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      const size_t outoff = in.mapping().tile_offset_outer(x, y);
      for (size_t i = 0; i < std::min(TM, M - D - x); ++i) {
        const size_t ti    = TM * i;
        const size_t ti_mD = TM * (i - D);
        const size_t ti_pD = TM * (i + D);
        for (size_t j = 0; j < std::min(TN, N - D - y); ++j) {
          const size_t inoff      = ti + j;
          const size_t baseoff    = outoff + inoff;
          const size_t inoff_mD   = ti_mD + j;
          const size_t inoff_pD   = ti_pD + j;
          const size_t baseoff_mD = outoff + inoff_mD;
          const size_t baseoff_pD = outoff + inoff_pD;
          // clang-format off
          hout[baseoff] =                  hin[baseoff_mD] +
                          hin[baseoff-D] + hin[baseoff]   + hin[baseoff+D] +
                                           hin[baseoff_pD];
          // clang-format on
        }
      }
    }
  }
}

template <size_t TM, size_t TN, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto manual_blocked_tiled_subview(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  using InTile  = std::mdspan<Type, std::extents<size_t, TM, TN>, std::layout_right>;
  using OutTile = std::mdspan<Type, std::extents<size_t, TM, TN>, std::layout_right>;

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      const size_t outoff = in.mapping().tile_offset_outer(x, y);
      const auto t_in     = InTile(in.data_handle() + outoff);
      auto t_out          = OutTile(out.data_handle() + outoff);
      for (size_t i = 0; i < std::min(TM, M - x); ++i) {
        for (size_t j = 0; j < std::min(TN, N - y); ++j) {
          // clang-format off
          t_out[i, j] =                ( t_in[i-D, j]
                        + t_in[i, j-D] + t_in[  i, j] + t_in[i, j+D]
                                       + t_in[i+D, j] ) / 5.0;
          // clang-format on
        }
      }
    }
  }
}

auto run_bench(const Config& cfg) -> int {
  // Standard layout matrices
  auto lr_in   = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lr_out  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlr_in  = std::mdspan<Type, Exts>(lr_in.get(), cfg.n_, cfg.n_);
  auto mlr_out = std::mdspan<Type, Exts>(lr_out.get(), cfg.n_, cfg.n_);

  // Tiled layout matrices
  auto lrtr_l1_in  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lrtr_l1_out = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlrtr_l1_in
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(lrtr_l1_in.get(), cfg.n_, cfg.n_);
  auto mlrtr_l1_out
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1>>(lrtr_l1_out.get(), cfg.n_, cfg.n_);

  auto lrtr_l2_in  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lrtr_l2_out = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlrtr_l2_in
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(lrtr_l2_in.get(), cfg.n_, cfg.n_);
  auto mlrtr_l2_out
    = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2>>(lrtr_l2_out.get(), cfg.n_, cfg.n_);

  // Initializations
  auto rng_naive    = ankerl::nanobench::Rng(42);
  auto rng_tiled_l1 = rng_naive.copy();
  auto rng_tiled_l2 = rng_naive.copy();
  benchmark_utils::fill_2d_uniform01(mlr_in, rng_naive);
  benchmark_utils::fill_2d_uniform01(mlrtr_l1_in, rng_tiled_l1);
  benchmark_utils::fill_2d_uniform01(mlrtr_l2_in, rng_tiled_l2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  // Checks
  const auto n       = static_cast<double>(cfg.n_);
  const double big_o = 5.0 * n * n;
  int res            = 0;

  naive(mlr_in, mlr_out);
  double ref = benchmark_utils::l2_norm(mlr_out);

  cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L1", big_o);
  cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L2", big_o);

  naive(mlrtr_l1_in, mlrtr_l1_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_naive", big_o);
  naive(mlrtr_l2_in, mlrtr_l2_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_naive", big_o);

  manual_blocked_tiled<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_mbt", big_o);
  manual_blocked_tiled<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_mbt", big_o);
  manual_blocked_tiled_subview<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_mbtsv", big_o);
  manual_blocked_tiled_subview<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_mbtsv", big_o);

  return res;
#else
  // Benchmarks
  ankerl::nanobench::Bench bs;
  bs.output(nullptr).timeUnit(1s, "s").relative(true);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
  bs.title(fmt::format("{0}x{0} {1}C", cfg.n_, omp_get_max_threads()));
#else
  bs.title(fmt::format("{0}x{0} 1C", cfg.n_));
#endif
  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }

  bs.run("R naive", [&]() { naive(mlr_in, mlr_out, cfg.d_); });

  bs.run(fmt::format("R cb-{0}x{0}", TILE_L1), [&]() { cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out, cfg.d_); });
  bs.run(fmt::format("R cb-{0}x{0}", TILE_L2), [&]() { cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out, cfg.d_); });
  bs.run(fmt::format("R cb-{0}x{0} (runtime)", cfg.t_), [&]() {
    cache_blocked(mlr_in, mlr_out, TILE_L1, TILE_L1, cfg.d_);
  });

  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L1), [&]() { naive(mlrtr_l1_in, mlrtr_l1_out, cfg.d_); });
  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L2), [&]() { naive(mlrtr_l2_in, mlrtr_l2_out, cfg.d_); });

  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L1), [&]() {
    manual_blocked_tiled<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L2), [&]() {
    manual_blocked_tiled<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt (runtime)", cfg.t_), [&]() {
    manual_blocked_tiled(mlrtr_l1_in, mlrtr_l1_out, TILE_L1, TILE_L1, cfg.d_);
  });

  bs.run(fmt::format("RTR-{0}x{0} mbtsv", TILE_L1), [&]() {
    manual_blocked_tiled_subview<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbtsv", TILE_L2), [&]() {
    manual_blocked_tiled_subview<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out, cfg.d_);
  });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();

  return 0;
#endif
}

} // namespace

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute 2D stencil benchmark\n\n"
                                "Usage: layout_tiled.benchmarks.stencil_2d [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Matrix size [default: 512]\n"
                                "  -t, --tile <SIZE>             Tile size [default: 64]\n"
                                "  -d, --delta <VAL>             Stencil delta (order) [default: 1]\n"
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
  while ((opt = getopt_long(argc, argv, "n:t:d:o:h", static_cast<const struct option*>(long_options), nullptr))
         != -1) {
    switch (opt) {
    case 'n':
      cfg.n_ = std::stoul(optarg);
      if (0 == cfg.n_) {
        fmt::println(stderr, "error: matrix size must be positive (got {})", optarg);
        return 1;
      }
      break;
    case 't':
      cfg.t_ = std::stoul(optarg);
      if (0 == cfg.t_) {
        fmt::println(stderr, "error: tile size must be positive (got {})", optarg);
        return 1;
      }
      break;
    case 'd':
      cfg.d_ = std::stoul(optarg);
      if (0 == cfg.d_) {
        fmt::println(stderr, "error: delta must be positive (got {})", optarg);
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
