// benchmarks/stencil_3d.cpp -*-C++-*-
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

namespace {

template <class MdSpanIn, class MdSpanOut>
  requires IsMdSpan3D<MdSpanIn> && IsMdSpan3D<MdSpanOut>
auto naive(const MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);
  const size_t O = out.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
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
  requires IsMdSpan3D<MdSpanIn> && IsMdSpan3D<MdSpanOut>
auto cache_blocked(const MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);
  const size_t O = out.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
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
  requires IsMdSpan3D<MdSpanIn> && IsMdSpan3D<MdSpanOut>
auto cache_blocked(const MdSpanIn in, MdSpanOut out, size_t TM, size_t TN, size_t TO, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);
  const size_t O = out.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
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
  requires IsMdSpan3D<MdSpanIn> && IsMdSpan3D<MdSpanOut>
auto data_tiled(const MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(in.extent(2) == out.extent(2));
  assert(D != 0);

  using InTile  = std::mdspan<Type, std::extents<size_t, TM, TN, TO>, std::layout_right>;
  using OutTile = std::mdspan<Type, std::extents<size_t, TM, TN, TO>, std::layout_right>;

  const size_t M = out.extent(0);
  const size_t N = out.extent(1);
  const size_t O = out.extent(2);

#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for collapse(3) schedule(static)
#endif
  for (size_t x = 0; x < M; x += TM) {
    for (size_t y = 0; y < N; y += TN) {
      for (size_t z = 0; z < O; z += TO) {
        const size_t tx = std::min(TM, M - x);
        const size_t ty = std::min(TN, N - y);
        const size_t tz = std::min(TO, O - z);

        const size_t gi_begin = std::max(x, D);
        const size_t gi_end   = std::min(x + tx, M - D);
        const size_t gj_begin = std::max(y, D);
        const size_t gj_end   = std::min(y + ty, N - D);
        const size_t gk_begin = std::max(z, D);
        const size_t gk_end   = std::min(z + tz, O - D);
        if (gi_begin >= gi_end || gj_begin >= gj_end || gk_begin >= gk_end) {
          continue;
        }

        const size_t li_begin = gi_begin - x;
        const size_t li_end   = gi_end - x;
        const size_t lj_begin = gj_begin - y;
        const size_t lj_end   = gj_end - y;
        const size_t lk_begin = gk_begin - z;
        const size_t lk_end   = gk_end - z;

        const size_t outer_offset = out.mapping().tile_offset_outer(x, y, z);
        const auto t_in           = InTile(in.data_handle() + outer_offset);
        auto t_out                = OutTile(out.data_handle() + outer_offset);

        size_t fi_begin = 0;
        size_t fi_end   = 0;
        size_t fj_begin = 0;
        size_t fj_end   = 0;
        size_t fk_begin = 0;
        size_t fk_end   = 0;
        if (tx > D && ty > D && tz > D) {
          fi_begin = std::max(li_begin, D);
          fi_end   = std::min(li_end, tx - D);
          fj_begin = std::max(lj_begin, D);
          fj_end   = std::min(lj_end, ty - D);
          fk_begin = std::max(lk_begin, D);
          fk_end   = std::min(lk_end, tz - D);
        }

        for (size_t i = li_begin; i < li_end; ++i) {
          const size_t gi = x + i;
          for (size_t j = lj_begin; j < lj_end; ++j) {
            const size_t gj = y + j;
            for (size_t k = lk_begin; k < lk_end; ++k) {
              if (i >= fi_begin && i < fi_end && j >= fj_begin && j < fj_end && k >= fk_begin && k < fk_end) {
                // clang-format off
                t_out[i, j, k] =                   t_in[i-D, j, k] + t_in[i,   j, k-D] +
                                t_in[i, j-D,   k] + t_in[  i, j, k] + t_in[i, j+D,   k] +
                                t_in[i,   j, k+D] + t_in[i+D, j, k];
                // clang-format on
              } else {
                const size_t gk = z + k;
                // clang-format off
                out[gi, gj, gk] =                       in[gi-D, gj, gk] + in[gi,    gj, gk-D] +
                                  in[gi, gj-D,    gk] + in[  gi, gj, gk] + in[gi, gj+D,    gk] +
                                  in[gi,    gj, gk+D] + in[gi+D, gj, gk];
                // clang-format on
              }
            }
          }
        }
      }
    }
  }
}

auto run_bench(const Config& cfg) -> int {
  // Standard layout matrices
  auto lr_in   = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto lr_out  = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto mlr_in  = std::mdspan<Type, Exts>(lr_in.get(), cfg.n_, cfg.n_, cfg.n_);
  auto mlr_out = std::mdspan<Type, Exts>(lr_out.get(), cfg.n_, cfg.n_, cfg.n_);

  // Tiled layout matrices
  auto lrtr_l1_in  = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto lrtr_l1_out = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto mlrtr_l1_in = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1, TILE_L1>>(
    lrtr_l1_in.get(),
    cfg.n_,
    cfg.n_,
    cfg.n_
  );
  auto mlrtr_l1_out = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L1, TILE_L1, TILE_L1>>(
    lrtr_l1_out.get(),
    cfg.n_,
    cfg.n_,
    cfg.n_
  );

  auto lrtr_l2_in  = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto lrtr_l2_out = MemorySlice<Type>(cfg.n_ * cfg.n_ * cfg.n_);
  auto mlrtr_l2_in = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2, TILE_L2>>(
    lrtr_l2_in.get(),
    cfg.n_,
    cfg.n_,
    cfg.n_
  );
  auto mlrtr_l2_out = std::mdspan<Type, Exts, layout_tiled::layout_rtr<TILE_L2, TILE_L2, TILE_L2>>(
    lrtr_l2_out.get(),
    cfg.n_,
    cfg.n_,
    cfg.n_
  );

  // Initializations
  auto rng_naive = ankerl::nanobench::Rng(42);
  benchmark_utils::fill_3d_uniform01(mlr_in, rng_naive);
  benchmark_utils::fill_3d_copy(mlrtr_l1_in, mlr_in);
  benchmark_utils::fill_3d_copy(mlrtr_l2_in, mlr_in);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  // Checks
  const auto n       = static_cast<double>(cfg.n_);
  const double big_o = 7.0 * n * n * n;
  int res            = 0;

  naive(mlr_in, mlr_out);
  double ref = benchmark_utils::l2_norm(mlr_out);

  cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mlr_in, mlr_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "CB-IJK-SR-L1", big_o);
  cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mlr_in, mlr_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "CB-IJK-SR-L2", big_o);

  naive(mlrtr_l1_in, mlrtr_l1_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "N-IJK-TR-L1", big_o);
  naive(mlrtr_l2_in, mlrtr_l2_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "N-IJK-TR-L2", big_o);

  data_tiled<TILE_L1, TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "DT-IJK-TR-L1", big_o);
  data_tiled<TILE_L2, TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  res = benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "DT-IJK-TR-L2", big_o);

  return res;
#else
  // Benchmarks
  ankerl::nanobench::Bench bs;
  bs.output(nullptr).timeUnit(1s, "s").relative(true);
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
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

  bs.run("N-IJK-SR", [&]() { naive(mlr_in, mlr_out, cfg.d_); });

  bs.run("CB-IJK-SR-L1", [&]() { cache_blocked<TILE_L1, TILE_L1, TILE_L1>(mlr_in, mlr_out, cfg.d_); });
  bs.run("CB-IJK-SR-L2", [&]() { cache_blocked<TILE_L2, TILE_L2, TILE_L2>(mlr_in, mlr_out, cfg.d_); });
  bs.run(fmt::format("CB-IJK-SR-B{}", cfg.t_), [&]() {
    cache_blocked(mlr_in, mlr_out, cfg.t_, cfg.t_, cfg.t_, cfg.d_);
  });

  bs.run("N-IJK-TR-L1", [&]() { naive(mlrtr_l1_in, mlrtr_l1_out, cfg.d_); });
  bs.run("N-IJK-TR-L2", [&]() { naive(mlrtr_l2_in, mlrtr_l2_out, cfg.d_); });

  bs.run("DT-IJK-TR-L1", [&]() { data_tiled<TILE_L1, TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out, cfg.d_); });
  bs.run("DT-IJK-TR-L2", [&]() { data_tiled<TILE_L2, TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out, cfg.d_); });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();

  return 0;
#endif
}

} // namespace

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute 3D stencil benchmark\n\n"
                                "Usage: layout_tiled.benchmarks.stencil_3d [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Tensor size [default: 64]\n"
                                "  -t, --tile <SIZE>             Tile size [default: 8]\n"
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
  while ((opt = getopt_long(argc, argv, "n:t:o:h", static_cast<const struct option*>(long_options), nullptr)) != -1) {
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

  return run_bench(cfg);
}
