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
#include <fstream>
#include <iostream>
#include <mdspan>
#include <string>
#include <vector>

#include <getopt.h>

#include <fmt/core.h>
#include <nanobench.h>
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#include <omp.h>
#endif

#include <layout_tiled/layout_right_tiled_right.hpp>

using namespace std::chrono_literals;
using ext_t = std::dextents<size_t, 2>;

constexpr size_t TILE_L1D = 32;
constexpr size_t TILE_L2  = 128;

struct Config {
  uint64_t n_ = 512;
  std::string output_;
};

// clang-format off
template <
  class T,
  class ExtA, class ExtB, class ExtC,
  class LayA, class LayB, class LayC,
  class AccA, class AccB, class AccC
>
static auto naive_gemm(
  std::mdspan<T, ExtA, LayA, AccA> A,
  std::mdspan<T, ExtB, LayB, AccB> B,
  std::mdspan<T, ExtC, LayC, AccC> C
) -> void
  requires(ExtA::rank() == 2 && ExtA::rank() == ExtB::rank() && ExtA::rank() == ExtC::rank())
{
  // clang-format on
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < A.extent(0); ++i) {
    for (size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp simd
#endif
      for (size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

// clang-format off
template <
  class T,
  class ExtA, class ExtB, class ExtC,
  class LayA, class LayB, class LayC,
  class AccA, class AccB, class AccC
>
auto block_l1d_gemm(
  std::mdspan<T, ExtA, LayA, AccA> A,
  std::mdspan<T, ExtB, LayB, AccB> B,
  std::mdspan<T, ExtC, LayC, AccC> C
) -> void
  requires(ExtA::rank() == 2 && ExtA::rank() == ExtB::rank() && ExtA::rank() == ExtC::rank())
{
  // clang-format on
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t bi = 0; bi < A.extent(0); bi += TILE_L1D) {
    auto const i_end = std::min(bi + TILE_L1D, A.extent(0));
    for (std::size_t bj = 0; bj < B.extent(1); bj += TILE_L1D) {
      auto const j_end = std::min(bj + TILE_L1D, B.extent(1));
      for (std::size_t bk = 0; bk < A.extent(1); bk += TILE_L1D) {
        auto const k_end = std::min(bk + TILE_L1D, A.extent(1));
        for (std::size_t i = bi; i < i_end; ++i) {
          for (std::size_t j = bj; j < j_end; ++j) {
            T acc{};
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp simd
#endif
            for (std::size_t k = bk; k < k_end; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

// clang-format off
template <
  class T,
  class ExtA, class ExtB, class ExtC,
  class LayA, class LayB, class LayC,
  class AccA, class AccB, class AccC
>
auto block_l2_gemm(
  std::mdspan<T, ExtA, LayA, AccA> A,
  std::mdspan<T, ExtB, LayB, AccB> B,
  std::mdspan<T, ExtC, LayC, AccC> C
) -> void
  requires(ExtA::rank() == 2 && ExtA::rank() == ExtB::rank() && ExtA::rank() == ExtC::rank())
{
  // clang-format on
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t bi = 0; bi < A.extent(0); bi += TILE_L2) {
    auto const i_end = std::min(bi + TILE_L2, A.extent(0));
    for (std::size_t bj = 0; bj < B.extent(1); bj += TILE_L2) {
      auto const j_end = std::min(bj + TILE_L2, B.extent(1));
      for (std::size_t bk = 0; bk < A.extent(1); bk += TILE_L2) {
        auto const k_end = std::min(bk + TILE_L2, A.extent(1));
        for (std::size_t i = bi; i < i_end; ++i) {
          for (std::size_t j = bj; j < j_end; ++j) {
            T acc{};
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp simd
#endif
            for (std::size_t k = bk; k < k_end; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

// clang-format off
template <
  class T,
  class ExtA, class ExtB, class ExtC,
  class LayA, class LayB, class LayC,
  class AccA, class AccB, class AccC
>
static auto omp_tile_l1d_gemm(
  std::mdspan<T, ExtA, LayA, AccA> A,
  std::mdspan<T, ExtB, LayB, AccB> B,
  std::mdspan<T, ExtC, LayC, AccC> C
) -> void
  requires(ExtA::rank() == 2 && ExtA::rank() == ExtB::rank() && ExtA::rank() == ExtC::rank())
{
  // clang-format on
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp parallel for schedule(static)
#pragma omp tile sizes(TILE_L1D, TILE_L1D)
#endif
  for (size_t i = 0; i < A.extent(0); ++i) {
    for (size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp simd
#endif
      for (size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

// clang-format off
template <
  class T,
  class ExtA, class ExtB, class ExtC,
  class LayA, class LayB, class LayC,
  class AccA, class AccB, class AccC
>
static auto omp_tile_l2_gemm(
  std::mdspan<T, ExtA, LayA, AccA> A,
  std::mdspan<T, ExtB, LayB, AccB> B,
  std::mdspan<T, ExtC, LayC, AccC> C
) -> void
  requires(ExtA::rank() == 2 && ExtA::rank() == ExtB::rank() && ExtA::rank() == ExtC::rank())
{
  // clang-format on
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp parallel for schedule(static)
#pragma omp tile sizes(TILE_L2, TILE_L2)
#endif
  for (size_t i = 0; i < A.extent(0); ++i) {
    for (size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp simd
#endif
      for (size_t k = 0; k < A.extent(1); ++k) {
        acc += A[i, k] * B[k, j];
      }
      C[i, j] = acc;
    }
  }
}

static auto run_bench(Config const& cfg) -> void {
  ankerl::nanobench::Bench bs;
  bs.output(nullptr);
#if defined(LAYOUT_TILED_BENCH_OPENMP)
  bs.title(fmt::format("{0}x{0} {1}C", cfg.n_, omp_get_max_threads())).timeUnit(1s, "s").relative(true);
#else
  bs.title(fmt::format("{}x{} 1C", cfg.n_).timeUnit(1s, "s").relative(true);
#endif
  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }

  // Standard layout matrices
  std::vector<double> sA(cfg.n_ * cfg.n_);
  std::vector<double> sB(cfg.n_ * cfg.n_);
  std::vector<double> sC(cfg.n_ * cfg.n_);
  auto msA = std::mdspan<double, ext_t>(sA.data(), cfg.n_, cfg.n_);
  auto msB = std::mdspan<double, ext_t>(sB.data(), cfg.n_, cfg.n_);
  auto msC = std::mdspan<double, ext_t>(sC.data(), cfg.n_, cfg.n_);

  // Tiled layout matrices
  std::vector<double> tA_l1(cfg.n_ * cfg.n_);
  std::vector<double> tB_l1(cfg.n_ * cfg.n_);
  std::vector<double> tC_l1(cfg.n_ * cfg.n_);
  auto mtA_l1 = std::mdspan<double, ext_t, layout_right_tiled_right<TILE_L1D, TILE_L1D>>(tA_l1.data(), cfg.n_, cfg.n_);
  auto mtB_l1 = std::mdspan<double, ext_t, layout_right_tiled_right<TILE_L1D, TILE_L1D>>(tB_l1.data(), cfg.n_, cfg.n_);
  auto mtC_l1 = std::mdspan<double, ext_t, layout_right_tiled_right<TILE_L1D, TILE_L1D>>(tC_l1.data(), cfg.n_, cfg.n_);

  std::vector<double> tA_l2(cfg.n_ * cfg.n_);
  std::vector<double> tB_l2(cfg.n_ * cfg.n_);
  std::vector<double> tC_l2(cfg.n_ * cfg.n_);
  auto mtA_l2 = std::mdspan<double, ext_t, layout_right_tiled_right<TILE_L2, TILE_L2>>(tA_l2.data(), cfg.n_, cfg.n_);
  auto mtB_l2 = std::mdspan<double, ext_t, layout_right_tiled_right<TILE_L2, TILE_L2>>(tB_l2.data(), cfg.n_, cfg.n_);
  auto mtC_l2 = std::mdspan<double, ext_t, layout_right_tiled_right<TILE_L2, TILE_L2>>(tC_l2.data(), cfg.n_, cfg.n_);

  // Matrix initializations
#if defined(LAYOUT_TILED_BENCH_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < cfg.n_; ++i) {
    for (size_t j = 0; j < cfg.n_; ++j) {
      msA[i, j]    = 1.0;
      msB[i, j]    = 1.0;
      msC[i, j]    = 1.0;
      mtA_l1[i, j] = 1.0;
      mtB_l1[i, j] = 1.0;
      mtC_l1[i, j] = 1.0;
      mtA_l2[i, j] = 1.0;
      mtB_l2[i, j] = 1.0;
      mtC_l2[i, j] = 1.0;
    }
  }

  // Benchmarks
  bs.run("right naive", [&]() { naive_gemm(msA, msB, msC); });
  bs.run(fmt::format("tiled naive {0}x{0}", TILE_L1D), [&]() { naive_gemm(mtA_l1, mtB_l1, mtC_l1); });
  bs.run(fmt::format("tiled naive {0}x{0}", TILE_L2), [&]() { naive_gemm(mtA_l2, mtB_l2, mtC_l2); });
  bs.run(fmt::format("right block {0}x{0}", TILE_L1D), [&]() { block_l1d_gemm(msA, msB, msC); });
  bs.run(fmt::format("right block {0}x{0}", TILE_L2), [&]() { block_l2_gemm(msA, msB, msC); });
  bs.run(fmt::format("right omp tile {0}x{0}", TILE_L1D), [&]() { omp_tile_l1d_gemm(msA, msB, msC); });
  bs.run(fmt::format("right omp tile {0}x{0}", TILE_L2), [&]() { omp_tile_l2_gemm(msA, msB, msC); });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();
}

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG         = "Execute GEMM benchmark\n\n"
                                        "Usage: layout_tiled.benchmarks.gemm [OPTIONS]\n\n"
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
