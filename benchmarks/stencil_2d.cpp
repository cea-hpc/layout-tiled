// Copyright (C) 2024-2025 CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
//
// SPDX-License-Identifier: Apache-2.0

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

namespace stencil_2d {

template <class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto naive(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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

} // namespace stencil_2d

auto run_bench(const Config& cfg) -> void {
  ankerl::nanobench::Bench bs;
  bs.output(nullptr).timeUnit(1s, "s").relative(true);
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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

  // Standard layout matrices
  auto lr_in   = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lr_out  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlr_in  = std::mdspan<Type, Exts>(lr_in.get(), cfg.n_, cfg.n_);
  auto mlr_out = std::mdspan<Type, Exts>(lr_out.get(), cfg.n_, cfg.n_);

  // Tiled layout matrices
  auto lrtr_l1_in  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lrtr_l1_out = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlrtr_l1_in
    = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(lrtr_l1_in.get(), cfg.n_, cfg.n_);
  auto mlrtr_l1_out
    = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(lrtr_l1_out.get(), cfg.n_, cfg.n_);

  auto lrtr_l2_in  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto lrtr_l2_out = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mlrtr_l2_in
    = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(lrtr_l2_in.get(), cfg.n_, cfg.n_);
  auto mlrtr_l2_out
    = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(lrtr_l2_out.get(), cfg.n_, cfg.n_);

  // Initializations
  auto rng_naive    = ankerl::nanobench::Rng(42);
  auto rng_tiled_l1 = rng_naive.copy();
  auto rng_tiled_l2 = rng_naive.copy();
  benchmark_utils::fill_2d_uniform01(mlr_in, rng_naive);
  benchmark_utils::fill_2d_uniform01(mlrtr_l1_in, rng_tiled_l1);
  benchmark_utils::fill_2d_uniform01(mlrtr_l2_in, rng_tiled_l2);

  // Checks
#if defined(LAYOUT_TILED_ENABLE_CHECKS)
  stencil_2d::naive(mlr_in, mlr_out);
  double ref = benchmark_utils::l2_norm(mlr_out);

  stencil_2d::naive(mlrtr_l1_in, mlrtr_l1_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_naive");
  stencil_2d::naive(mlrtr_l2_in, mlrtr_l2_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_naive");

  stencil_2d::cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L1");
  stencil_2d::cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L2");

  stencil_2d::manual_blocked_tiled(mlrtr_l1_in, mlrtr_l1_out, TILE_L1, TILE_L1);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_mbt");
  stencil_2d::manual_blocked_tiled(mlrtr_l2_in, mlrtr_l2_out, TILE_L2, TILE_L2);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_mbt");
#endif

  // Benchmarks
  bs.run("R naive", [&]() { stencil_2d::naive(mlr_in, mlr_out, cfg.d_); });

  bs.run(fmt::format("R cb-{0}x{0}", TILE_L1), [&]() {
    stencil_2d::cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out, cfg.d_);
  });
  bs.run(fmt::format("R cb-{0}x{0}", TILE_L2), [&]() {
    stencil_2d::cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out, cfg.d_);
  });
  bs.run(fmt::format("R cb-{0}x{0} (runtime)", cfg.t_), [&]() {
    stencil_2d::cache_blocked(mlr_in, mlr_out, TILE_L1, TILE_L1, cfg.d_);
  });

  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L1), [&]() { stencil_2d::naive(mlrtr_l1_in, mlrtr_l1_out, cfg.d_); });
  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L2), [&]() { stencil_2d::naive(mlrtr_l2_in, mlrtr_l2_out, cfg.d_); });

  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L1), [&]() {
    stencil_2d::manual_blocked_tiled<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L2), [&]() {
    stencil_2d::manual_blocked_tiled<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt (runtime)", cfg.t_), [&]() {
    stencil_2d::manual_blocked_tiled(mlrtr_l1_in, mlrtr_l1_out, TILE_L1, TILE_L1, cfg.d_);
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
                                        "  -t, --tile <SIZE>             Tile size [default: 64]\n"
                                        "  -d, --delta <VAL>             Stencil delta (order) [default: 1]\n"
                                        "  -o, --output <FILENAME>       Output file name\n"
                                        "  -h, --help                    Show this help\n";
  static struct option long_options[] = {
    {"size", required_argument, nullptr, 'n'},
    {"tile", required_argument, nullptr, 't'},
    {"delta", required_argument, nullptr, 'd'},
    {"output", required_argument, nullptr, 'o'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  Config cfg;
  int opt;
  while ((opt = getopt_long(argc, argv, "n:t:d:o:h", long_options, nullptr)) != -1) {
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

  run_bench(cfg);

  return 0;
}
