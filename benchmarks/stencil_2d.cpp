// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#include <cassert>
#include <chrono>
#include <cstddef>
#include <experimental/mdspan>
#include <fstream>
#include <iostream>
#include <string>

#include <getopt.h>

#include <fmt/core.h>
#include <layout_tiled/layout_right_tiled_right.hpp>
#include <nanobench.h>
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#include <omp.h>
#endif

using namespace std::chrono_literals;
using Type = double;
using Exts = std::dextents<size_t, 2>;

constexpr size_t TILE_L1 = 32;
constexpr size_t TILE_L2 = 128;

struct Config {
  uint64_t n_ = 512;
  std::string output_;
};

namespace stencil_2d {

// clang-format off
template <
  class T,
  class ExtIn, class LayIn, class AccIn,
  class ExtOut, class LayOut, class AccOut
>
auto naive(
  std::mdspan<T, ExtIn, LayIn, AccIn> in,
  std::mdspan<T, ExtOut, LayOut, AccOut> out
) -> void
  requires(ExtIn::rank() == 2 && ExtIn::rank() == ExtOut::rank())
{
  // clang-format on
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 1; i < out.extent(0) - 1; ++i) {
    for (size_t j = 1; j < out.extent(1) - 1; ++j) {
      // clang-format off
      out[i, j] =              in[i-1, j] +
                  in[i, j-1] + in[  i, j] + in[i, j+1] +
                               in[i+1, j];
      // clang-format on
    }
  }
}

// clang-format off
template <
  size_t TILE_I, size_t TILE_J,
  class T,
  class ExtIn, class LayIn, class AccIn,
  class ExtOut, class LayOut, class AccOut
>
auto manual_blocked_tiled(
  std::mdspan<T, ExtIn, LayIn, AccIn> in,
  std::mdspan<T, ExtOut, LayOut, AccOut> out
) -> void
  requires(ExtIn::rank() == 2 && ExtIn::rank() == ExtOut::rank())
{
  // clang-format on
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));

  auto hin  = in.data_handle();
  auto hout = out.data_handle();
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t block_i = 1; block_i < in.extent(0) - 1; block_i += TILE_I) {
    auto i_bound = std::min(TILE_I, in.extent(0) - block_i - 1);
    for (size_t block_j = 1; block_j < in.extent(1) - 1; block_j += TILE_J) {
      auto j_bound = std::min(TILE_J, in.extent(1) - block_j - 1);
      auto outoff  = in.mapping().tile_offset_outer(block_i, block_j);
      for (size_t i = 0; i < i_bound; ++i) {
        auto ti   = TILE_I * i;
        auto tim1 = TILE_I * (i - 1);
        auto tip1 = TILE_I * (i + 1);
        for (size_t j = 0; j < j_bound; ++j) {
          auto inoff     = ti + j;
          auto baseoff   = outoff + inoff;
          auto inm1off   = tim1 + j;
          auto inp1off   = tip1 + j;
          auto basem1off = outoff + inm1off;
          auto basep1off = outoff + inp1off;
          // clang-format off
          hout[baseoff] =                  hin[basem1off] +
                          hin[baseoff-1] + hin[baseoff]   + hin[baseoff+1] +
                                           hin[basep1off];
          // clang-format on
        }
      }
    }
  }
}

// clang-format off
template <
  class T,
  class ExtIn, class LayIn, class AccIn,
  class ExtOut, class LayOut, class AccOut
>
auto manual_blocked_tiled(
  std::mdspan<T, ExtIn, LayIn, AccIn> in,
  std::mdspan<T, ExtOut, LayOut, AccOut> out,
  size_t tile_i, size_t tile_j
) -> void
  requires(ExtIn::rank() == 2 && ExtIn::rank() == ExtOut::rank())
{
  // clang-format on
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));

  auto hin  = in.data_handle();
  auto hout = out.data_handle();
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t block_i = 1; block_i < in.extent(0) - 1; block_i += tile_i) {
    auto i_bound = std::min(tile_i, in.extent(0) - block_i - 1);
    for (size_t block_j = 1; block_j < in.extent(1) - 1; block_j += tile_j) {
      auto j_bound = std::min(tile_j, in.extent(1) - block_j - 1);
      auto outoff  = in.mapping().tile_offset_outer(block_i, block_j);
      for (size_t i = 0; i < i_bound; ++i) {
        auto ti   = tile_i * i;
        auto tim1 = tile_i * (i - 1);
        auto tip1 = tile_i * (i + 1);
        for (size_t j = 0; j < j_bound; ++j) {
          auto inoff     = ti + j;
          auto baseoff   = outoff + inoff;
          auto inm1off   = tim1 + j;
          auto inp1off   = tip1 + j;
          auto basem1off = outoff + inm1off;
          auto basep1off = outoff + inp1off;
          // clang-format off
          hout[baseoff] =                  hin[basem1off] +
                          hin[baseoff-1] + hin[baseoff]   + hin[baseoff+1] +
                                           hin[basep1off];
          // clang-format on
        }
      }
    }
  }
}

// clang-format off
template <
  size_t BLKSZ_I, size_t BLKSZ_J,
  class T,
  class ExtIn, class LayIn, class AccIn,
  class ExtOut, class LayOut, class AccOut
>
auto cache_blocked(
  std::mdspan<T, ExtIn, LayIn, AccIn> in,
  std::mdspan<T, ExtOut, LayOut, AccOut> out
) -> void
  requires(ExtIn::rank() == 2 && ExtIn::rank() == ExtOut::rank())
{
  // clang-format on
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t block_i = 1; block_i < in.extent(0) - 1; block_i += BLKSZ_I) {
    auto i_bound = std::min(block_i + BLKSZ_I, in.extent(0) - 1);
    for (size_t block_j = 1; block_j < in.extent(1) - 1; block_j += BLKSZ_J) {
      auto j_bound = std::min(block_j + BLKSZ_J, in.extent(1) - 1);
      for (size_t i = block_i; i < i_bound; ++i) {
        for (size_t j = block_j; j < j_bound; ++j) {
          // clang-format off
          out[i, j] =              in[i-1, j] +
                      in[i, j-1] + in[  i, j] + in[i, j+1] +
                                   in[i+1, j];
          // clang-format on
        }
      }
    }
  }
}

// clang-format off
template <
  class T,
  class ExtIn, class LayIn, class AccIn,
  class ExtOut, class LayOut, class AccOut
>
auto cache_blocked(
  std::mdspan<T, ExtIn, LayIn, AccIn> in,
  std::mdspan<T, ExtOut, LayOut, AccOut> out,
  size_t blksz_i, size_t blksz_j
) -> void
  requires(ExtIn::rank() == 2 && ExtIn::rank() == ExtOut::rank())
{
  // clang-format on
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t block_i = 1; block_i < in.extent(0) - 1; block_i += blksz_i) {
    auto i_bound = std::min(block_i + blksz_i, in.extent(0) - 1);
    for (size_t block_j = 1; block_j < in.extent(1) - 1; block_j += blksz_j) {
      auto j_bound = std::min(block_j + blksz_j, in.extent(1) - 1);
      for (size_t i = block_i; i < i_bound; ++i) {
        for (size_t j = block_j; j < j_bound; ++j) {
          // clang-format off
          out[i, j] =              in[i-1, j] +
                      in[i, j-1] + in[  i, j] + in[i, j+1] +
                                   in[i+1, j];
          // clang-format on
        }
      }
    }
  }
}

} // namespace stencil_2d

auto run_bench(Config const& cfg) -> void {
  ankerl::nanobench::Bench bs;
  bs.output(nullptr);
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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

  // Standard layout matrices
  auto lr_in   = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lr_out  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlr_in  = std::mdspan<Type, Exts>(lr_in.get(), cfg.n_, cfg.n_);
  auto mlr_out = std::mdspan<Type, Exts>(lr_out.get(), cfg.n_, cfg.n_);

  // Tiled layout matrices
  auto lrtr_l1_in  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lrtr_l1_out = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlrtr_l1_in =
    std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(lrtr_l1_in.get(), cfg.n_, cfg.n_);
  auto mlrtr_l1_out =
    std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(lrtr_l1_out.get(), cfg.n_, cfg.n_);

  auto lrtr_l2_in  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lrtr_l2_out = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlrtr_l2_in =
    std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(lrtr_l2_in.get(), cfg.n_, cfg.n_);
  auto mlrtr_l2_out =
    std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(lrtr_l2_out.get(), cfg.n_, cfg.n_);

  // Matrix initializations
  auto rng_naive    = ankerl::nanobench::Rng(42);
  auto rng_tiled_l1 = rng_naive.copy();
  auto rng_tiled_l2 = rng_naive.copy();
  benchmark_utils::fill_2d_random(mlr_in, rng_naive);
  benchmark_utils::fill_2d_random(mlrtr_l1_in, rng_tiled_l1);
  benchmark_utils::fill_2d_random(mlrtr_l2_in, rng_tiled_l2);

  // Benchmarks
  bs.run("lright naive", [&]() { stencil_2d::naive(mlr_in, mlr_out); });

  bs.run(fmt::format("lrtr naive {0}x{0}", TILE_L1), [&]() { stencil_2d::naive(mlrtr_l1_in, mlrtr_l1_out); });
  bs.run(fmt::format("lrtr naive {0}x{0}", TILE_L2), [&]() { stencil_2d::naive(mlrtr_l2_in, mlrtr_l2_out); });

  bs.run(fmt::format("lrtr manual_tiled {0}x{0}", TILE_L1), [&]() {
    stencil_2d::manual_blocked_tiled<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  });
  bs.run(fmt::format("lrtr manual_tiled {0}x{0}", TILE_L2), [&]() {
    stencil_2d::manual_blocked_tiled<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  });
  bs.run(fmt::format("lrtr manual_tiled_rt {0}x{0}", TILE_L1), [&]() {
    stencil_2d::manual_blocked_tiled(mlrtr_l1_in, mlrtr_l1_out, TILE_L1, TILE_L1);
  });
  bs.run(fmt::format("lrtr manual_tiled_rt {0}x{0}", TILE_L2), [&]() {
    stencil_2d::manual_blocked_tiled(mlrtr_l2_in, mlrtr_l2_out, TILE_L2, TILE_L2);
  });

  bs.run(fmt::format("lr blocked {0}x{0}", TILE_L1), [&]() {
    stencil_2d::cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out);
  });
  bs.run(fmt::format("lr blocked {0}x{0}", TILE_L2), [&]() {
    stencil_2d::cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out);
  });
  bs.run(fmt::format("lr blocked_rt {0}x{0}", TILE_L1), [&]() {
    stencil_2d::cache_blocked(mlr_in, mlr_out, TILE_L1, TILE_L1);
  });
  bs.run(fmt::format("lr blocked_rt {0}x{0}", TILE_L2), [&]() {
    stencil_2d::cache_blocked(mlr_in, mlr_out, TILE_L2, TILE_L2);
  });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();
}

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG         = "Execute 2D stencil benchmark\n\n"
                                        "Usage: layout_tiled.benchmarks.stencil_2d [OPTIONS]\n\n"
                                        "Options:\n"
                                        "  -n, --size <SIZE>             Matrix size [default: 512]\n"
                                        "  -o, --output <FILENAME>       Output file name\n"
                                        "  -h, --help                    Show this help\n";
  static struct option long_options[] = {
    {"size", required_argument, nullptr, 'n'},
    {"output", required_argument, nullptr, 'o'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  Config cfg;
  int opt;
  while ((opt = getopt_long(argc, argv, "n:o:h", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'n':
      cfg.n_ = std::stoul(optarg);
      if (0 == cfg.n_) {
        fmt::println(stderr, "error: matrix size must be positive (got {})", optarg);
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

  run_bench(cfg);

  return 0;
}
