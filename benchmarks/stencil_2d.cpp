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
auto naive(const MdSpanIn in, MdSpanOut out, size_t D = DELTA) -> void {
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

template <size_t TM = TILE_L1, size_t TN = TILE_L1, size_t D = DELTA, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto cache_blocked(const MdSpanIn in, MdSpanOut out) -> void {
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
auto cache_blocked(const MdSpanIn in, MdSpanOut out, size_t TM, size_t TN, size_t D = DELTA) -> void {
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

template <size_t TM = TILE_L1, size_t TN = TILE_L1, size_t D = DELTA, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto manual_blocked_tiled(const MdSpanIn in, MdSpanOut out) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  using index_type = typename MdSpanIn::index_type;

  const size_t M                        = in.extent(0);
  const size_t N                        = in.extent(1);
  constexpr index_type tile_size        = in.mapping().tile_size();
  const index_type nb_elems_in_tile_row = tile_size * in.mapping().ntiles_in_extent(0);

  const auto hin = in.data_handle();
  auto hout      = out.data_handle();
#if defined(LAYOUT_TILED_BENCH_ENABLE_OMP)
#pragma omp parallel for schedule(static)
#endif
  for (index_type x = D; x < M - D; x += TM) {
    for (index_type y = D; y < N - D; y += TN) {
      // Compute offset of first element in tile (outside domain if in first tile row/column)
      const index_type center_toff = in.mapping().tile_offset_outer(x, y);

      for (index_type i = x; i < std::min(x + TM, M - D); ++i) {
        const index_type li         = i % TM;
        const index_type center_row = li * TN;
        const index_type north_row  = (li - D + TM) * TN;
        const index_type south_row  = (li + D - TM) * TN;

        for (index_type j = y; j < std::min(y + TN, N - D); ++j) {
          const index_type lj     = j % TN;
          const index_type center = center_toff + center_row + lj;

          // North neighbor: i-D, j
          index_type north = 0;
          if (li - D >= 0 || x == 0) {
            // Same tile or first row of tiles
            north = center - TN;
          } else {
            // Different tile
            //  ┌─────────┬─────────┬─────────┬─────────┐
            //  │         │△═╗2     │         │         │  If the target element (@) is in the tile above us (*), we do:
            //  │         │║ ║  3   │         │         │
            //  │         │║1▽══▶@  │         │         │  1. Subtract $N_elems_per_tile \times N_tiles_in_row$ from the
            //  ├─────────┼║─────↑──┼─────────┼─────────┤     base tile offset (■);
            //  │         │■   ← * →│         │         │  2. Add the pre-computed north_row offset;
            //  │         │      ↓  │         │         │  3. Add the local j offset.
            //  │         │         │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  └─────────┴─────────┴─────────┴─────────┘
            north = center_toff - nb_elems_in_tile_row + north_row + lj;
          }

          // South neighbor: i+D, j
          index_type south = 0;
          if (li + D < TM && i + D < M) {
            // Same tile
            south = center + TN;
          } else {
            // Different tile
            //  ┌─────────┬─────────┬─────────┬─────────┐
            //  │         │         │         │         │  If the target element (@) is in the tile below us (*), we do:
            //  │         │         │         │         │
            //  │         │         │         │         │  1. Add $N_elems_per_tile \times N_tiles_in_row$ from the
            //  ├─────────┼─────────┼─────────┼─────────┤     base tile offset (■);
            //  │         │■        │         │         │  2. Add the pre-computed south_row offset (may not be 0 if
            //  │         │║     ↑  │         │         │     D > 1);
            //  │         │║1  ← * →│         │         │  3. Add the local j offset.
            //  ├─────────┼║─────↓──┼─────────┼─────────┤
            //  │         │▽════▶@  │         │         │
            //  │         │  2&3    │         │         │
            //  │         │         │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  └─────────┴─────────┴─────────┴─────────┘
            south = center_toff + nb_elems_in_tile_row + south_row + lj;
          }

          // West neighbor: i, j-D
          index_type west = 0;
          if (lj - D >= 0 || y == 0) {
            // Same tile or first column of tiles
            west = center - D;
          } else {
            // Different tile
            //  ┌─────────┬─────────┬─────────┬─────────┐
            //  │         │         │         │         │  If the target element (@) is in the tile left of us (*), we
            //  │         │         │         │         │  do:
            //  │         │         │         │         │
            //  ├────1────┼─────────┼─────────┼─────────┤  1. Subtract $N_elems_per_tile$ from the base tile offset (■);
            //  │◁═════════■↑       │         │         │  2. Add the pre-computed center_row offset;
            //  2▽══════▶@← * →     │         │         │  3. Add the local j - D + TN offset.
            //  │         │ ↓       │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  └─────────┴─────────┴─────────┴─────────┘
            west = center_toff - tile_size + center_row + (lj - D + TN);
          }

          // East neighbor: i, j+D
          index_type east;
          if (lj - D < TN && j + D < N) {
            // Same tile
            east = center + D;
          } else {
            // Different tile
            //  ┌─────────┬─────────┬─────────┬─────────┐
            //  │         │         │         │         │  If the target element (@) is in the tile right of us (*), we
            //  │         │         │         │         │  do:
            //  │         │         │         │         │
            //  ├─────────┼────1────┼2────────┼─────────┤  1. Add $N_elems_per_tile$ from the base tile offset (■);
            //  │         │■══════↑══▼ 3      │         │  2. Add the pre-computed center_row offset;
            //  │         │     ← * →@        │         │  3. Add the local j + D - TN offset.
            //  │         │       ↓ │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  ├─────────┼─────────┼─────────┼─────────┤
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  │         │         │         │         │
            //  └─────────┴─────────┴─────────┴─────────┘
            east = center_toff + tile_size + center_row + (lj + D - TN);
          }

          // clang-format off
          hout[center] =             ( hin[north]
                         + hin[west] + hin[center] + hin[east]
                                     + hin[south]  ) / 5.0;
          // clang-format on
        }
      }
    }
  }
}

// TODO(gdossantos): update the runtime-only version
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

template <size_t TM = TILE_L1, size_t TN = TILE_L1, size_t D = DELTA, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto manual_blocked_tiled_subview(const MdSpanIn in, MdSpanOut out) -> void {
  using index_type   = typename MdSpanIn::index_type;
  using mapping_type = typename MdSpanIn::mapping_type;
  using value_type   = typename MdSpanIn::value_type;

  constexpr index_type tile_size = mapping_type::tile_size();

  const index_type M = in.extent(0);
  const index_type N = in.extent(1);

  const index_type ntiles_M = in.mapping().ntiles_in_extent(0);
  const index_type ntiles_N = in.mapping().ntiles_in_extent(1);

  // Halo tile dimensions
  constexpr index_type halo_TM = TM + 2 * D;
  constexpr index_type halo_TN = TN + 2 * D;

  using HaloExts = std::extents<size_t, halo_TM, halo_TN>;

  // Pre-allocate halo buffer
  alignas(64) std::array<value_type, halo_TM * halo_TN> halo_buf{};

  for (index_type x = 0; x < ntiles_M; x += TM) {
    for (index_type y = 0; y < ntiles_N; y += TN) {
      for (index_type i = x; i < std::min(x + TM, ntiles_M); ++i) {
        const index_type ti     = i * TM;
        const index_type ti_end = std::min(ti + TM, M);
        for (index_type j = y; j < std::min(y + TN, ntiles_N); ++j) {
          const index_type tj     = j * TN;
          const index_type tj_end = std::min(tj + TN, N);

          // Skip tiles without valid stencil region
          if (ti + D >= M - D || tj + D >= N - D || ti_end <= D || tj_end <= D) {
            continue;
          }

          // Fill halo buffer
          for (index_type hi = 0; hi < halo_TM; ++hi) {
            const index_type global_i    = ti + hi - D;
            const index_type src_tile_i  = global_i / TM;
            const index_type src_local_i = global_i % TM;
            const index_type halo_i      = hi * halo_TN;
            for (index_type hj = 0; hj < halo_TN; ++hj) {
              const index_type global_j      = tj + hj - D;
              const index_type src_tile_j    = global_j / TN;
              const index_type src_local_j   = global_j % TN;
              const index_type src_tile_base = (src_tile_i * ntiles_N + src_tile_j) * tile_size;
              const index_type src_offset    = src_tile_base + src_local_i * TN + src_local_j;
              const index_type halo_offset   = halo_i + hj;

              halo_buf[halo_offset] = in.data_handle()[src_offset];
            }
          }

          // Create local mdspan with layout_right over halo buffer
          std::mdspan<value_type, HaloExts, std::layout_right> hs(halo_buf.data());
          for (index_type hi = D; hi < std::min(TM + D, ti_end - ti + D); ++hi) {
            const index_type global_i    = ti + hi - D;
            const index_type dst_tile_i  = global_i / TM;
            const index_type dst_local_i = global_i % TM;
            for (index_type hj = D; hj < std::min(TN + D, tj_end - tj + D); ++hj) {
              // Global coordinates
              const index_type global_j = tj + hj - D;

              // Skip global boundaries
              if (global_i < D || global_i >= M - D || global_j < D || global_j >= N - D) {
                continue;
              }

              // Write to output - compute destination offset manually
              const index_type dst_tile_j    = global_j / TN;
              const index_type dst_local_j   = global_j % TN;
              const index_type dst_tile_base = (dst_tile_i * ntiles_N + dst_tile_j) * tile_size;
              const index_type dst_offset    = dst_tile_base + dst_local_i * TN + dst_local_j;

              // clang-format off
              out.data_handle()[dst_offset] =                ( hs[hi-D, hj]
                                              + hs[hi, hj-D] + hs[  hi, hj] + hs[hi, hj+D]
                                                             + hs[hi+D, hj] ) / 5.0;
              // clang-format on
            }
          }
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
  benchmark_utils::fill_2d_uniform01(mlr_in, rng_naive);
  benchmark_utils::fill_2d_copy(mlrtr_l1_in, mlr_in);
  benchmark_utils::fill_2d_copy(mlrtr_l2_in, mlr_in);

#if defined(LAYOUT_TILED_BENCH_ENABLE_CHECKS)
  // Checks
  const auto n       = static_cast<double>(cfg.n_);
  const double big_o = 5.0 * n * n;
  int res            = 0;

  auto ref_out  = MemorySlice<Type>(cfg.n_ * cfg.n_);
  auto mref_out = std::mdspan<Type, Exts>(ref_out.get(), cfg.n_, cfg.n_);
  naive(mlr_in, mref_out);
  double ref = benchmark_utils::l2_norm(mref_out);
  benchmark_utils::fill_zeroes(mlr_out);

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
  // benchmark_utils::compare_2d_vals(mref_out, mlrtr_l1_out);
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
