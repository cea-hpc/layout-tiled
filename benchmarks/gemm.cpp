/// Copyright (C) 2024, CEA
///
/// Licensed under the MIT License ("the License" hereafter);
/// You may not use this file except in compliance with the License.
/// You should have received a full copy of the License along with this program;
/// if not, you can obtain a copy at:
///
///    https://opensource.org/license/MIT
///
/// The software is provided “as is”, without warranty of any kind, express
/// or implied, including but not limited to the warranties of merchantability,
/// fitness for a particular purpose and noninfringement. In no event shall
/// the authors or copyright holders be liable for any claim, damages or other
/// liability, whether in an action of contract, tort or otherwise, arising from,
/// out of or in connection with the software or the use or other dealings
/// in the software.
///
/// Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>

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
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#include <omp.h>
#endif

#include <layout_tiled/layout_right_tiled_right.hpp>

using namespace std::chrono_literals;
using extents_type = std::dextents<size_t, 2>;

constexpr size_t TILE_L1D = 32;
constexpr size_t TILE_L2  = 128;

static std::string HELP_MSG = "Usage: layout_tiled.benchmarks.gemm [OPTIONS]\n\n"
                              "Options:\n"
                              "  -x, --x                Matrix X-axis size \n"
                              "  -y, --y                Matrix Y-axis size \n"
                              "  -o, --output           Output file\n"
                              "  -h, --help             Show this help\n";

struct Config {
  uint64_t x_ = 512;
  uint64_t y_ = 512;
  std::string output_;
};

template <class T, class Extents, class Layout, class AccessorPolicy>
static auto naive_gemm(
  std::mdspan<T, Extents, Layout, AccessorPolicy> A,
  std::mdspan<T, Extents, Layout, AccessorPolicy> B,
  std::mdspan<T, Extents, Layout, AccessorPolicy> C
) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < A.extent(0); ++i) {
    for (size_t j = 0; j < B.extent(1); ++j) {
      T acc{};
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
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
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
  bs.title(fmt::format("{}x{} {}C", cfg.x_, cfg.y_, omp_get_max_threads())).timeUnit(1s, "s").relative(true);
#else
  bs.title(fmt::format("{}x{} 1C", cfg.x_, cfg.y_).timeUnit(1s, "s").relative(true);
#endif
  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!cfg.output_.empty()) {
    file_stream.open(cfg.output_, std::ios::app | std::ios::binary);
    out_stream = &file_stream;
  }

  // Standard layout matrices
  std::vector<double> sA(cfg.x_ * cfg.y_);
  std::vector<double> sB(cfg.x_ * cfg.y_);
  std::vector<double> sC(cfg.x_ * cfg.y_);
  auto msA = std::mdspan<double, extents_type, std::layout_right>(sA.data(), cfg.x_, cfg.y_);
  auto msB = std::mdspan<double, extents_type, std::layout_right>(sB.data(), cfg.x_, cfg.y_);
  auto msC = std::mdspan<double, extents_type, std::layout_right>(sC.data(), cfg.x_, cfg.y_);

  // Tiled layout matrices
  std::vector<double> tA_l1(cfg.x_ * cfg.y_);
  std::vector<double> tB_l1(cfg.x_ * cfg.y_);
  std::vector<double> tC_l1(cfg.x_ * cfg.y_);
  auto mtA_l1 =
    std::mdspan<double, extents_type, layout_right_tiled_right<TILE_L1D, TILE_L1D>>(tA_l1.data(), cfg.x_, cfg.y_);
  auto mtB_l1 =
    std::mdspan<double, extents_type, layout_right_tiled_right<TILE_L1D, TILE_L1D>>(tB_l1.data(), cfg.x_, cfg.y_);
  auto mtC_l1 =
    std::mdspan<double, extents_type, layout_right_tiled_right<TILE_L1D, TILE_L1D>>(tC_l1.data(), cfg.x_, cfg.y_);

  std::vector<double> tA_l2(cfg.x_ * cfg.y_);
  std::vector<double> tB_l2(cfg.x_ * cfg.y_);
  std::vector<double> tC_l2(cfg.x_ * cfg.y_);
  auto mtA_l2 =
    std::mdspan<double, extents_type, layout_right_tiled_right<TILE_L2, TILE_L2>>(tA_l2.data(), cfg.x_, cfg.y_);
  auto mtB_l2 =
    std::mdspan<double, extents_type, layout_right_tiled_right<TILE_L2, TILE_L2>>(tB_l2.data(), cfg.x_, cfg.y_);
  auto mtC_l2 =
    std::mdspan<double, extents_type, layout_right_tiled_right<TILE_L2, TILE_L2>>(tC_l2.data(), cfg.x_, cfg.y_);

  // Matrix initializations
#if defined(LAYOUT_TILED_PARALLEL_BENCH)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < cfg.x_; ++i) {
    for (size_t j = 0; j < cfg.y_; ++j) {
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
  bs.run("right", [&]() { naive_gemm(msA, msB, msC); });
  bs.run("tiled 32x32 (L1D)", [&]() { naive_gemm(mtA_l1, mtB_l1, mtC_l1); });
  bs.run("tiled 128x128 (L2)", [&]() { naive_gemm(mtA_l2, mtB_l2, mtC_l2); });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();
}

auto main(int argc, char* argv[]) -> int {
  static struct option long_options[] = {
    {"x-size", required_argument, nullptr, 'x'},
    {"y-size", required_argument, nullptr, 'y'},
    {"output", required_argument, nullptr, 'o'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  Config cfg;
  int opt;
  while ((opt = getopt_long(argc, argv, "x:y:o:h", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'x':
      cfg.x_ = std::stoul(optarg);
      break;
    case 'y':
      cfg.y_ = std::stoul(optarg);
      break;
    case 'o':
      cfg.output_ = optarg;
      break;
    case 'h':
      fmt::println("{}", HELP_MSG);
      return 0;
    default:
      fmt::println("{}", HELP_MSG);
      return 1;
    }
  }

  run_bench(cfg);

  return 0;
}
