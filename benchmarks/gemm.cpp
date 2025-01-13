// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#include <getopt.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <experimental/mdspan>
#include <fstream>
#include <iostream>
#include <string>

#include <fmt/core.h>
#include <layout_tiled/layout_left_tiled_left.hpp>
#include <layout_tiled/layout_right_tiled_right.hpp>
#include <nanobench.h>
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#include <omp.h>
#endif
#if defined(LAYOUT_TILED_ENABLE_BLAS)
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

namespace gemm {

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires(MdSpanA::extents_type::rank() == 2, MdSpanB::extents_type::rank() == 2, MdSpanC::extents_type::rank() == 2)
auto naive(MdSpanA A, MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  size_t const M = A.extent(0);
  size_t const N = B.extent(1);
  size_t const O = A.extent(1);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      Type acc{};
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
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
  requires(MdSpanA::extents_type::rank() == 2, MdSpanB::extents_type::rank() == 2, MdSpanC::extents_type::rank() == 2)
auto cache_blocked(MdSpanA A, MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  size_t const M = A.extent(0);
  size_t const N = B.extent(1);
  size_t const O = A.extent(1);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    size_t const tx = std::min(x + TM, M);
    for (size_t y = 0; y < N; y += TN) {
      size_t const ty = std::min(y + TN, N);
      for (size_t z = 0; z < O; z += TO) {
        size_t const tz = std::min(z + TO, O);
        for (size_t i = x; i < tx; ++i) {
          for (size_t j = y; j < ty; ++j) {
            Type acc{};
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < tz; ++k) {
              acc += A[i, k] * B[k, j];
            }
            C[i, j] = acc;
          }
        }
      }
    }
  }
}

template <size_t TM, size_t TN, size_t TO, class MdSpanA, class MdSpanB, class MdSpanC>
  requires(MdSpanA::extents_type::rank() == 2, MdSpanB::extents_type::rank() == 2, MdSpanC::extents_type::rank() == 2)
auto manual_blocked_tiled(MdSpanA A, MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  size_t const M = A.extent(0);
  size_t const N = B.extent(1);
  size_t const O = A.extent(1);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    size_t const tx = std::min(TM, M - x);
    for (size_t y = 0; y < N; y += TN) {
      double* C_outoff = C.data_handle() + C.mapping().tile_offset_outer(x, y);
      size_t const ty  = std::min(TN, N - y);
      for (size_t z = 0; z < O; z += TO) {
        double const* A_outoff = A.data_handle() + A.mapping().tile_offset_outer(x, z);
        double const* B_outoff = A.data_handle() + B.mapping().tile_offset_outer(z, y);
        size_t const tz        = std::min(TO, O - z);
        for (size_t i = 0; i < tx; ++i) {
          size_t const ti = i * TM;
          size_t A_inoff  = ti;
          size_t C_inoff  = ti;
          for (size_t j = 0; j < ty; ++j) {
            size_t B_inoff = j;
            Type acc       = 0.0;
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < tz; ++k) {
              acc += A_outoff[A_inoff] * B_outoff[B_inoff];
              A_inoff += 1;
              B_inoff += TO;
            }
            C_outoff[C_inoff] = acc;
            C_inoff += 1;
          }
        }
      }
    }
  }
}

template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires(MdSpanA::extents_type::rank() == 2, MdSpanB::extents_type::rank() == 2, MdSpanC::extents_type::rank() == 2)
auto manual_blocked_tiled(MdSpanA A, MdSpanB B, MdSpanC C, size_t TM, size_t TN, size_t TO) -> void {
  assert(A.extent(1) == B.extent(0));
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));

  size_t const M = A.extent(0);
  size_t const N = B.extent(1);
  size_t const O = A.extent(1);

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    size_t const tx = std::min(TM, M - x);
    for (size_t y = 0; y < N; y += TN) {
      double* C_outoff = C.data_handle() + C.mapping().tile_offset_outer(x, y);
      size_t const ty  = std::min(TN, N - y);
      for (size_t z = 0; z < O; z += TO) {
        double const* A_outoff = A.data_handle() + A.mapping().tile_offset_outer(x, z);
        double const* B_outoff = A.data_handle() + B.mapping().tile_offset_outer(z, y);
        size_t const tz        = std::min(TO, O - z);
        for (size_t i = 0; i < tx; ++i) {
          size_t const ti = i * TM;
          size_t A_inoff  = ti;
          size_t C_inoff  = ti;
          for (size_t j = 0; j < ty; ++j) {
            size_t B_inoff = j;
            Type acc       = 0.0;
#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp simd
#endif
            for (size_t k = 0; k < tz; ++k) {
              acc += A_outoff[A_inoff] * B_outoff[B_inoff];
              A_inoff += 1;
              B_inoff += TO;
            }
            C_outoff[C_inoff] = acc;
            C_inoff += 1;
          }
        }
      }
    }
  }
}

#if defined(LAYOUT_TILED_ENABLE_BLAS)
/// Only supported with `LayoutRight` matrices
template <class MdSpanA, class MdSpanB, class MdSpanC>
  requires(MdSpanA::extents_type::rank() == 2, MdSpanB::extents_type::rank() == 2, MdSpanC::extents_type::rank() == 2, std::is_same_v<typename MdSpanA::layout_type, std::layout_right>, std::is_same_v<typename MdSpanB::layout_type, std::layout_right>, std::is_same_v<typename MdSpanC::layout_type, std::layout_right>)
auto blas(MdSpanA A, MdSpanB B, MdSpanC C) -> void {
  assert(A.extent(0) == C.extent(0));
  assert(A.extent(1) == B.extent(0));
  assert(B.extent(1) == C.extent(1));

  int const M            = static_cast<int>(A.extent(0));
  int const N            = static_cast<int>(B.extent(1));
  int const O            = static_cast<int>(A.extent(1));
  const double alpha     = 1.0;
  double const* A_handle = A.data_handle();
  int const lda          = static_cast<int>(A.stride(0));
  double const* B_handle = B.data_handle();
  int const ldb          = static_cast<int>(B.stride(0));
  const double beta      = 0.0;
  double* C_handle       = C.data_handle();
  int const ldc          = static_cast<int>(C.stride(0));

  cblas_dgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, O, alpha, A_handle, lda, B_handle, ldb, beta, C_handle, ldc
  );
}
#endif

} // namespace gemm

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
  auto rA = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto rB = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto lB = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto rC = MemorySlice<double>(cfg.n_ * cfg.n_);
  // Tiled layout matrices (L1 tiles)
  auto trA_l1 = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto trB_l1 = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto trC_l1 = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto tlB_l1 = MemorySlice<double>(cfg.n_ * cfg.n_);
  // Tiled layout matrices (L2 tiles)
  auto trA_l2 = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto trB_l2 = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto trC_l2 = MemorySlice<double>(cfg.n_ * cfg.n_);
  auto tlB_l2 = MemorySlice<double>(cfg.n_ * cfg.n_);

  // Associated MDSPANs
  auto mrA     = std::mdspan<Type, Exts>(rA.get(), cfg.n_, cfg.n_);
  auto mrB     = std::mdspan<Type, Exts>(rB.get(), cfg.n_, cfg.n_);
  auto mlB     = std::mdspan<Type, Exts, std::layout_left>(lB.get(), cfg.n_, cfg.n_);
  auto mrC     = std::mdspan<Type, Exts>(rC.get(), cfg.n_, cfg.n_);
  auto mtrA_l1 = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(trA_l1.get(), cfg.n_, cfg.n_);
  auto mtrB_l1 = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(trB_l1.get(), cfg.n_, cfg.n_);
  auto mtrC_l1 = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L1, TILE_L1>>(trC_l1.get(), cfg.n_, cfg.n_);
  auto mtlB_l1 = std::mdspan<Type, Exts, layout_left_tiled_left<TILE_L1, TILE_L1>>(tlB_l1.get(), cfg.n_, cfg.n_);
  auto mtrA_l2 = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(trA_l2.get(), cfg.n_, cfg.n_);
  auto mtrB_l2 = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(trB_l2.get(), cfg.n_, cfg.n_);
  auto mtrC_l2 = std::mdspan<Type, Exts, layout_right_tiled_right<TILE_L2, TILE_L2>>(trC_l2.get(), cfg.n_, cfg.n_);
  auto mtlB_l2 = std::mdspan<Type, Exts, layout_left_tiled_left<TILE_L2, TILE_L2>>(tlB_l2.get(), cfg.n_, cfg.n_);

  // Matrix initializations
  ankerl::nanobench::Rng rng_naive(42);
  auto rng_tiled_l1 = rng_naive.copy();
  auto rng_tiled_l2 = rng_naive.copy();
  benchmark_utils::fill_2d_uniform01(mrA, rng_naive);
  benchmark_utils::fill_2d_uniform01(mrB, rng_naive);
  benchmark_utils::fill_2d_uniform01(mlB, rng_naive);
  benchmark_utils::fill_2d_uniform01(mrC, rng_naive);
  benchmark_utils::fill_2d_uniform01(mtrA_l1, rng_tiled_l1);
  benchmark_utils::fill_2d_uniform01(mtrB_l1, rng_tiled_l1);
  benchmark_utils::fill_2d_uniform01(mtlB_l1, rng_tiled_l1);
  benchmark_utils::fill_2d_uniform01(mtrC_l1, rng_tiled_l1);
  benchmark_utils::fill_2d_uniform01(mtrA_l2, rng_tiled_l2);
  benchmark_utils::fill_2d_uniform01(mtrB_l2, rng_tiled_l2);
  benchmark_utils::fill_2d_uniform01(mtlB_l2, rng_tiled_l2);
  benchmark_utils::fill_2d_uniform01(mtrC_l2, rng_tiled_l2);

  // Checks
#if defined(LAYOUT_TILED_ENABLE_CHECK_RESULT)
  gemm::naive(mrA, mrB, mrB);
  double ref = frobenius_norm(mrC);

  gemm::cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mrA, mrB, mrC);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "R_cb-L1");
  gemm::cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mrA, mrB, mrC);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mrC), "R_cb-L2");

  gemm::naive(mtrA_l1, mtrB_l1, mtrC_l1);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-L1_naive");
  gemm::naive(mtrA_l2, mtrB_l2, mtrC_l2);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-L2_naive");

  gemm::manual_blocked_tiled<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l1), "RTR-L1_mbt");
  gemm::manual_blocked_tiled<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mtrC_l2), "RTR-L2_mbt");
#endif

  // Benchmarks
  bs.run("R naive", [&]() { gemm::naive(mrA, mrB, mrC); });
  bs.run("R_BL naive", [&]() { gemm::naive(mrA, mlB, mrC); });
#if defined(LAYOUT_TILED_ENABLE_BLAS)
  bs.run("R blas", [&]() { gemm::blas(mrA, mrB, mrC); });
#endif

  bs.run(fmt::format("R cb-{0}x{0}", TILE_L1), [&]() {
    gemm::cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mrA, mrB, mrC);
  });
  bs.run(fmt::format("R cb-{0}x{0}", TILE_L2), [&]() {
    gemm::cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mrA, mrB, mrC);
  });
  bs.run(fmt::format("R_BL cb-{0}x{0}", TILE_L1), [&]() {
    gemm::cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mrA, mlB, mrC);
  });
  bs.run(fmt::format("R_BL cb-{0}x{0}", TILE_L2), [&]() {
    gemm::cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mrA, mlB, mrC);
  });

  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L1), [&]() { gemm::naive(mtrA_l1, mtrB_l1, mtrC_l1); });
  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L2), [&]() { gemm::naive(mtrA_l2, mtrB_l2, mtrC_l2); });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} naive", TILE_L1), [&]() { gemm::naive(mtrA_l1, mtlB_l1, mtrC_l1); });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} naive", TILE_L2), [&]() { gemm::naive(mtrA_l2, mtlB_l2, mtrC_l2); });

  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L1), [&]() {
    gemm::manual_blocked_tiled<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtrB_l1, mtrC_l1);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L2), [&]() {
    gemm::manual_blocked_tiled<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtrB_l2, mtrC_l2);
  });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbt", TILE_L1), [&]() {
    gemm::manual_blocked_tiled<TILE_L1, TILE_L1, TILE_L1>(mtrA_l1, mtlB_l1, mtrC_l1);
  });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbt", TILE_L2), [&]() {
    gemm::manual_blocked_tiled<TILE_L2, TILE_L2, TILE_L2>(mtrA_l2, mtlB_l2, mtrC_l2);
  });

  bs.run(fmt::format("RTR-{0}x{0} mbt (runtime)", cfg.t_), [&]() {
    gemm::manual_blocked_tiled(mtrA_l1, mtrB_l1, mtrC_l1, cfg.t_, cfg.t_, cfg.t_);
  });
  bs.run(fmt::format("RTR_BLTL-{0}x{0} mbt (runtime)", cfg.t_), [&]() {
    gemm::manual_blocked_tiled(mtrA_l1, mtlB_l1, mtrC_l1, cfg.t_, cfg.t_, cfg.t_);
  });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();
}

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute GEMM benchmark\n\n"
                                "Usage: layout_tiled.benchmarks.gemm [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Matrix size [default: 512]\n"
                                "  -t, --tile <SIZE>             Tile size [default: 32]\n"
                                "  -o, --output <FILENAME>       Output file name\n"
                                "  -h, --help                    Show this help\n";

  static struct option long_options[] = {
    {"size", required_argument, nullptr, 'n'},
    {"tile", required_argument, nullptr, 't'},
    {"output", required_argument, nullptr, 'o'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  Config cfg;
  int opt;
  while ((opt = getopt_long(argc, argv, "n:t:o:h", long_options, nullptr)) != -1) {
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

  run_bench(cfg);

  return 0;
}
