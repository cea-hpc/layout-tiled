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
using Exts = std::dextents<size_t, 3>;

constexpr size_t TILE_L1 = 8;
constexpr size_t TILE_L2 = 32;
constexpr size_t DELTA   = 1;

struct Config {
  uint64_t n_ = 64;
  uint64_t t_ = 8;
  uint64_t d_ = 1;
  std::string output_;
};

namespace stencil_3d {

template <class MdSpanIn, class MdSpanOut>
  requires(3 == MdSpanIn::extents_type::rank(), 3 == MdSpanOut::extents_type::rank())
auto naive(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  size_t const M = out.extent(0);
  size_t const N = out.extent(1);
  size_t const O = out.extent(2);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t i = D; i < M - D; ++i) {
    for (size_t j = D; j < N - D; ++j) {
      for (size_t k = D; k < O - D; ++k) {
        // clang-format off
        out[i, j, k] =                   in[i-D, j, k] + in[i,   j, k-D] +
                       in[i, j-D,   k] + in[  i, j, k] + in[i, j+D,   k] +
                       in[i,   j, k+D] + in[i+D, j, k];
        // clang-format on
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanIn, class MdSpanOut>
  requires(3 == MdSpanIn::extents_type::rank(), 3 == MdSpanOut::extents_type::rank())
auto cache_blocked(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  size_t const M = out.extent(0);
  size_t const N = out.extent(1);
  size_t const O = out.extent(2);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      for (size_t z = D; z < O - D; z += TO) {
        for (size_t i = x; i < std::min(x + TM, M - D); ++i) {
          for (size_t j = y; j < std::min(y + TN, N - D); ++j) {
            for (size_t k = z; k < std::min(z + TO, O - D); ++k) {
              // clang-format off
              out[i, j, k] =                   in[i-D, j, k] + in[i,   j, k-D] +
                             in[i, j-D,   k] + in[  i, j, k] + in[i, j+D,   k] +
                             in[i,   j, k+D] + in[i+D, j, k];
              // clang-format on
            }
          }
        }
      }
    }
  }
}

template <class MdSpanIn, class MdSpanOut>
  requires(3 == MdSpanIn::extents_type::rank(), 3 == MdSpanOut::extents_type::rank())
auto cache_blocked(MdSpanIn in, MdSpanOut out, size_t TM, size_t TN, size_t TO, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  size_t const M = out.extent(0);
  size_t const N = out.extent(1);
  size_t const O = out.extent(2);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      for (size_t z = D; z < O - D; z += TO) {
        for (size_t i = x; i < std::min(x + TM, M - D); ++i) {
          for (size_t j = y; j < std::min(y + TN, N - D); ++j) {
            for (size_t k = z; k < std::min(z + TO, O - D); ++k) {
              // clang-format off
              out[i, j, k] =                   in[i-D, j, k] + in[i,   j, k-D] +
                             in[i, j-D,   k] + in[  i, j, k] + in[i, j+D,   k] +
                             in[i,   j, k+D] + in[i+D, j, k];
              // clang-format on
            }
          }
        }
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanIn, class MdSpanOut>
  requires(3 == MdSpanIn::extents_type::rank(), 3 == MdSpanOut::extents_type::rank())
auto manual_blocked_tiled(MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  size_t const M = out.extent(0);
  size_t const N = out.extent(1);
  size_t const O = out.extent(2);

  size_t const i_stride = TN * TO;
  size_t const j_stride = TO;
  size_t const k_stride = 1;

  auto hin  = in.data_handle();
  auto hout = out.data_handle();
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      for (size_t z = D; z < O - D; z += TO) {
        size_t const outoff = in.mapping().tile_offset_outer(x, y, z);
        for (size_t i = 0; i < std::min(TM, M - D - x); ++i) {
          size_t const i_off = outoff + i * i_stride;
          for (size_t j = 0; j < std::min(TN, N - D - y); ++j) {
            size_t const ij_off = i_off + j * j_stride;
            for (size_t k = 0; k < std::min(TO, O - D - z); ++k) {
              size_t const center = ij_off + k * k_stride;

              size_t const left   = center - D * i_stride;
              size_t const right  = center + D * i_stride;
              size_t const bottom = center - D * j_stride;
              size_t const top    = center + D * j_stride;
              size_t const back   = center - D * k_stride;
              size_t const front  = center + D * k_stride;

              // clang-format off
                hout[center] =              hin[top]    + hin[back] +
                               hin[left]  + hin[center] + hin[right] +
                               hin[front] + hin[bottom];
              // clang-format on
            }
          }
        }
      }
    }
  }
}

template <class MdSpanIn, class MdSpanOut>
  requires(3 == MdSpanIn::extents_type::rank(), 3 == MdSpanOut::extents_type::rank())
auto manual_blocked_tiled(MdSpanIn in, MdSpanOut out, size_t TM, size_t TN, size_t TO, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  size_t const M = out.extent(0);
  size_t const N = out.extent(1);
  size_t const O = out.extent(2);

  size_t const i_stride = TN * TO;
  size_t const j_stride = TO;
  size_t const k_stride = 1;

  auto hin  = in.data_handle();
  auto hout = out.data_handle();
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M - D; x += TM) {
    for (size_t y = D; y < N - D; y += TN) {
      for (size_t z = D; z < O - D; z += TO) {
        size_t const outoff = in.mapping().tile_offset_outer(x, y, z);
        for (size_t i = 0; i < std::min(TM, M - D - x); ++i) {
          size_t const i_off = outoff + i * i_stride;
          for (size_t j = 0; j < std::min(TN, N - D - y); ++j) {
            size_t const ij_off = i_off + j * j_stride;
            for (size_t k = 0; k < std::min(TO, O - D - z); ++k) {
              size_t const center = ij_off + k * k_stride;

              size_t const left   = center - D * i_stride;
              size_t const right  = center + D * i_stride;
              size_t const bottom = center - D * j_stride;
              size_t const top    = center + D * j_stride;
              size_t const back   = center - D * k_stride;
              size_t const front  = center + D * k_stride;

              // clang-format off
                hout[center] =              hin[top] + hin[back] +
                               hin[left]  + hin[center] + hin[right] +
                               hin[front] + hin[bottom];
              // clang-format on
            }
          }
        }
      }
    }
  }
}

} // namespace stencil_3d

auto run_bench(Config const& cfg) -> void {
  ankerl::nanobench::Bench bs;
  bs.output(nullptr).timeUnit(1s, "s").relative(true);
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
  bs.title(fmt::format("{0}x{0}x{0} {1}C", cfg.n_, omp_get_max_threads()));
#else
  bs.title(fmt::format("{0}x{0}x{0} 1C", cfg.n_));
#endif
  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }

  // Standard layout matrices
  auto lr_in   = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto lr_out  = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto mlr_in  = std::mdspan<Type, Exts>(lr_in.get(), cfg.n_, cfg.n_, cfg.n_);
  auto mlr_out = std::mdspan<Type, Exts>(lr_out.get(), cfg.n_, cfg.n_, cfg.n_);

  // Tiled layout matrices
  auto lrtr_l1_in  = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto lrtr_l1_out = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto mlrtr_l1_in = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1, TILE_L1>>(
    lrtr_l1_in.get(), cfg.n_, cfg.n_, cfg.n_
  );
  auto mlrtr_l1_out = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1, TILE_L1>>(
    lrtr_l1_out.get(), cfg.n_, cfg.n_, cfg.n_
  );

  auto lrtr_l2_in  = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto lrtr_l2_out = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto mlrtr_l2_in = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2, TILE_L2>>(
    lrtr_l2_in.get(), cfg.n_, cfg.n_, cfg.n_
  );
  auto mlrtr_l2_out = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2, TILE_L2>>(
    lrtr_l2_out.get(), cfg.n_, cfg.n_, cfg.n_
  );

  // Initializations
  auto rng_naive    = ankerl::nanobench::Rng(42);
  auto rng_tiled_l1 = rng_naive.copy();
  auto rng_tiled_l2 = rng_naive.copy();
  benchmark_utils::fill_3d_uniform01(mlr_in, rng_naive);
  benchmark_utils::fill_3d_uniform01(mlrtr_l1_in, rng_tiled_l1);
  benchmark_utils::fill_3d_uniform01(mlrtr_l2_in, rng_tiled_l2);

  // Checks
#if defined(LAYOUT_TILED_ENABLE_CHECKS)
  stencil_3d::naive(mlr_in, mlr_out);
  double ref = benchmark_utils::l2_norm(mlr_out);

  stencil_3d::naive(mlrtr_l1_in, mlrtr_l1_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_naive");
  stencil_3d::naive(mlrtr_l2_in, mlrtr_l2_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_naive");

  stencil_3d::cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mlr_in, mlr_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L1");
  stencil_3d::cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mlr_in, mlr_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L2");

  stencil_3d::manual_blocked_tiled(mlrtr_l1_in, mlrtr_l1_out, TILE_L1, TILE_L1, TILE_L1);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_mbt");
  stencil_3d::manual_blocked_tiled(mlrtr_l2_in, mlrtr_l2_out, TILE_L2, TILE_L2, TILE_L2);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_mbt");
#endif

  // Benchmarks
  bs.run("R naive", [&]() { stencil_3d::naive(mlr_in, mlr_out, cfg.d_); });

  bs.run(fmt::format("R cb-{0}x{0}x{0}", TILE_L1), [&]() {
    stencil_3d::cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mlr_in, mlr_out, cfg.d_);
  });
  bs.run(fmt::format("R cb-{0}x{0}x{0}", TILE_L2), [&]() {
    stencil_3d::cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mlr_in, mlr_out, cfg.d_);
  });
  bs.run(fmt::format("R cb-{0}x{0}x{0} (runtime)", cfg.t_), [&]() {
    stencil_3d::cache_blocked(mlr_in, mlr_out, TILE_L1, TILE_L1, TILE_L1, cfg.d_);
  });

  bs.run(fmt::format("RTR-{0}x{0}x{0} naive", TILE_L1), [&]() {
    stencil_3d::naive(mlrtr_l1_in, mlrtr_l1_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0}x{0} naive", TILE_L2), [&]() {
    stencil_3d::naive(mlrtr_l2_in, mlrtr_l2_out, cfg.d_);
  });

  bs.run(fmt::format("RTR-{0}x{0}x{0} mbt", TILE_L1), [&]() {
    stencil_3d::manual_blocked_tiled<TILE_L1, TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0}x{0} mbt", TILE_L2), [&]() {
    stencil_3d::manual_blocked_tiled<TILE_L2, TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out, cfg.d_);
  });
  bs.run(fmt::format("RTR-{0}x{0}x{0} mbt (runtime)", cfg.t_), [&]() {
    stencil_3d::manual_blocked_tiled(mlrtr_l1_in, mlrtr_l1_out, TILE_L1, TILE_L1, TILE_L1, cfg.d_);
  });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();
}

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG         = "Execute 3D stencil benchmark\n\n"
                                        "Usage: layout_tiled.benchmarks.stencil_3d [OPTIONS]\n\n"
                                        "Options:\n"
                                        "  -n, --size <SIZE>             Tensor size [default: 64]\n"
                                        "  -t, --tile <SIZE>             Tile size [default: 8]\n"
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
        fmt::println(stderr, "error: tensor size must be positive (got {})", optarg);
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
