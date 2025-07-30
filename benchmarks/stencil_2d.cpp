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

namespace {

template <size_t D = DELTA, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto naive(const MdSpanIn in, MdSpanOut out) -> void {
  assert(in.extent(0) == out.extent(0));
  assert(in.extent(1) == out.extent(1));
  assert(D != 0);

  const size_t M = out.extent(0) - 2 * D;
  const size_t N = out.extent(1) - 2 * D;

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t i = D; i < M; ++i) {
    for (size_t j = D; j < N; ++j) {
      // clang-format off
      out[i, j] =              ( in[i-D, j]
                  + in[i, j-D] + in[  i, j] + in[i, j+D]
                               + in[i+D, j] ) / 5.0;
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

  const size_t M = out.extent(0) - 2 * D;
  const size_t N = out.extent(1) - 2 * D;

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (size_t x = D; x < M; x += TM) {
    for (size_t y = D; y < N; y += TN) {
      for (size_t i = x; i < std::min(x + TM, M); ++i) {
        for (size_t j = y; j < std::min(y + TN, N); ++j) {
          // clang-format off
          out[i, j] =              ( in[i-D, j]
                      + in[i, j-D] + in[  i, j] + in[i, j+D]
                                   + in[i+D, j] ) / 5.0;
          // clang-format on
        }
      }
    }
  }
}

template <size_t TM = TILE_L1, size_t TN = TILE_L1, size_t D = DELTA, class MdSpanIn, class MdSpanOut>
  requires IsMdSpan2D<MdSpanIn> && IsMdSpan2D<MdSpanOut>
auto manual_blocked_tiled(const MdSpanIn in, MdSpanOut out) -> void {
  using index_type = typename MdSpanIn::index_type;

  const auto& in_mapping = in.mapping();

  // Get extents and tile info
  const index_type M             = in.extent(0);
  const index_type N             = in.extent(1);
  const index_type tile_size     = in_mapping.tile_size();
  const index_type num_tile_cols = in_mapping.ntiles_in_extent(1);

  // Match the naive implementation's loop bounds exactly
  const index_type naive_M = M - 2 * D;
  const index_type naive_N = N - 2 * D;

  // Precompute common values
  const index_type tile_row_stride = num_tile_cols * tile_size;

  // Process in cache-friendly blocks
  for (index_type block_i = D; block_i < D + naive_M; block_i += TM) {
    for (index_type block_j = D; block_j < D + naive_N; block_j += TN) {

      const index_type block_end_i = std::min(block_i + TM, D + naive_M);
      const index_type block_end_j = std::min(block_j + TN, D + naive_N);

      for (index_type i = block_i; i < block_end_i; ++i) {
        // Precompute tile row for current i
        const index_type center_tile_i         = i / TM;
        const index_type center_local_i        = i % TM;
        const index_type center_tile_row_base  = center_tile_i * tile_row_stride;
        const index_type center_local_row_base = center_local_i * TN;

        // Precompute for up neighbor
        const index_type up_i              = i - D;
        const index_type up_tile_i         = up_i / TM;
        const index_type up_local_i        = up_i % TM;
        const index_type up_tile_row_base  = up_tile_i * tile_row_stride;
        const index_type up_local_row_base = up_local_i * TN;

        // Precompute for down neighbor
        const index_type down_i              = i + D;
        const index_type down_tile_i         = down_i / TM;
        const index_type down_local_i        = down_i % TM;
        const index_type down_tile_row_base  = down_tile_i * tile_row_stride;
        const index_type down_local_row_base = down_local_i * TN;

        for (index_type j = block_j; j < block_end_j; ++j) {

          // Center point offset
          const index_type center_tile_j  = j / TN;
          const index_type center_local_j = j % TN;
          const index_type center_offset
            = center_tile_row_base + center_tile_j * tile_size + center_local_row_base + center_local_j;

          // Up point offset
          const index_type up_tile_j  = j / TN;
          const index_type up_local_j = j % TN;
          const index_type up_offset  = up_tile_row_base + up_tile_j * tile_size + up_local_row_base + up_local_j;

          // Left point offset
          const index_type left_j       = j - D;
          const index_type left_tile_j  = left_j / TN;
          const index_type left_local_j = left_j % TN;
          const index_type left_offset
            = center_tile_row_base + left_tile_j * tile_size + center_local_row_base + left_local_j;

          // Right point offset
          const index_type right_j       = j + D;
          const index_type right_tile_j  = right_j / TN;
          const index_type right_local_j = right_j % TN;
          const index_type right_offset
            = center_tile_row_base + right_tile_j * tile_size + center_local_row_base + right_local_j;

          // Down point offset
          const index_type down_tile_j  = j / TN;
          const index_type down_local_j = j % TN;
          const index_type down_offset
            = down_tile_row_base + down_tile_j * tile_size + down_local_row_base + down_local_j;

          // Perform the stencil computation
          const auto result
            = (in.data_handle()[up_offset] + in.data_handle()[left_offset] + in.data_handle()[center_offset]
               + in.data_handle()[right_offset] + in.data_handle()[down_offset])
              / 5.0;

          out.data_handle()[center_offset] = result;
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

  const auto& mapping = in.mapping();
  const auto& extents = mapping.extents();

  constexpr auto tile_extents    = mapping_type::textents();
  constexpr index_type tile_m    = tile_extents[0];
  constexpr index_type tile_n    = tile_extents[1];
  constexpr index_type tile_size = mapping_type::tile_size();

  const index_type M = extents.extent(0);
  const index_type N = extents.extent(1);

  const index_type ntiles_m = mapping.ntiles_in_extent(0);
  const index_type ntiles_n = mapping.ntiles_in_extent(1);

  constexpr index_type cache_tiles_m = (TM + tile_m - 1) / tile_m;
  constexpr index_type cache_tiles_n = (TN + tile_n - 1) / tile_n;

  // Halo tile dimensions
  constexpr index_type halo_tile_m = tile_m + 2 * D;
  constexpr index_type halo_tile_n = tile_n + 2 * D;

  // Pre-allocate halo buffers
  alignas(64) value_type halo_in_buffer[halo_tile_m * halo_tile_n];
  alignas(64) value_type halo_out_buffer[halo_tile_m * halo_tile_n];

#if defined(LAYOUT_TILED_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (index_type cb_i = 0; cb_i < ntiles_m; cb_i += cache_tiles_m) {
    for (index_type cb_j = 0; cb_j < ntiles_n; cb_j += cache_tiles_n) {

      const index_type tile_end_i = std::min(cb_i + cache_tiles_m, ntiles_m);
      const index_type tile_end_j = std::min(cb_j + cache_tiles_n, ntiles_n);

      for (index_type tile_i = cb_i; tile_i < tile_end_i; ++tile_i) {
        for (index_type tile_j = cb_j; tile_j < tile_end_j; ++tile_j) {

          const index_type phys_start_i = tile_i * tile_m;
          const index_type phys_start_j = tile_j * tile_n;
          const index_type phys_end_i   = std::min(phys_start_i + tile_m, M);
          const index_type phys_end_j   = std::min(phys_start_j + tile_n, N);

          // Skip tiles without valid stencil region
          if (phys_start_i + D >= M - D || phys_start_j + D >= N - D || phys_end_i <= D || phys_end_j <= D) {
            continue;
          }

          // Fill input halo buffer
          for (index_type hi = 0; hi < halo_tile_m; ++hi) {
            for (index_type hj = 0; hj < halo_tile_n; ++hj) {
              const index_type global_i = phys_start_i + hi - D;
              const index_type global_j = phys_start_j + hj - D;

              // Compute source offset manually
              const index_type src_tile_i    = global_i / tile_m;
              const index_type src_tile_j    = global_j / tile_n;
              const index_type src_local_i   = global_i % tile_m;
              const index_type src_local_j   = global_j % tile_n;
              const index_type src_tile_base = (src_tile_i * ntiles_n + src_tile_j) * tile_size;
              const index_type src_offset    = src_tile_base + src_local_i * tile_n + src_local_j;

              halo_in_buffer[hi * halo_tile_n + hj] = in.data_handle()[src_offset];
            }
          }

          // Create local mdspans with layout_right over halo buffers
          std::extents<index_type, std::dynamic_extent, std::dynamic_extent> halo_ext{halo_tile_m, halo_tile_n};
          std::mdspan<const value_type, decltype(halo_ext), std::layout_right> halo_in_span{halo_in_buffer, halo_ext};
          std::mdspan<value_type, decltype(halo_ext), std::layout_right> halo_out_span{halo_out_buffer, halo_ext};

          // Compute stencil over valid region
          const index_type valid_start_i = std::max(D, D);
          const index_type valid_start_j = std::max(D, D);
          const index_type valid_end_i = std::min(tile_m + D, static_cast<index_type>(phys_end_i - phys_start_i + D));
          const index_type valid_end_j = std::min(tile_n + D, static_cast<index_type>(phys_end_j - phys_start_j + D));

          for (index_type hi = valid_start_i; hi < valid_end_i; ++hi) {
            // Pre-compute i-dependent values
            const index_type global_i  = phys_start_i + hi - D;
            const index_type north_row = hi - D;
            const index_type south_row = hi + D;

            // Skip if global_i violates boundaries
            if (global_i < D || global_i >= M - D) {
              continue;
            }

            for (index_type hj = valid_start_j; hj < valid_end_j; ++hj) {
              const index_type global_j = phys_start_j + hj - D;

              // Skip global boundaries
              if (global_j < D || global_j >= N - D) {
                continue;
              }

              // Pre-compute j-dependent values
              const index_type west_col = hj - D;
              const index_type east_col = hj + D;

              // Stencil computation using local halo mdspans
              const auto center = halo_in_span[hi, hj];
              const auto north  = halo_in_span[north_row, hj];
              const auto south  = halo_in_span[south_row, hj];
              const auto west   = halo_in_span[hi, west_col];
              const auto east   = halo_in_span[hi, east_col];

              const auto result = (north + west + center + east + south) / 5.0;

              // Write to output halo buffer
              halo_out_span[hi, hj] = result;
            }
          }

          // Write back output halo to tiled mdspan
          for (index_type hi = valid_start_i; hi < valid_end_i; ++hi) {
            const index_type global_i = phys_start_i + hi - D;

            if (global_i < D || global_i >= M - D) {
              continue;
            }

            // Pre-compute i-dependent offset components
            const index_type center_tile_i  = global_i / tile_m;
            const index_type center_local_i = global_i % tile_m;

            for (index_type hj = valid_start_j; hj < valid_end_j; ++hj) {
              const index_type global_j = phys_start_j + hj - D;

              if (global_j < D || global_j >= N - D) {
                continue;
              }

              // Complete offset calculation
              const index_type center_tile_j    = global_j / tile_n;
              const index_type center_local_j   = global_j % tile_n;
              const index_type center_tile_base = (center_tile_i * ntiles_n + center_tile_j) * tile_size;
              const index_type center_offset    = center_tile_base + center_local_i * tile_n + center_local_j;

              out.data_handle()[center_offset] = halo_out_span[hi, hj];
            }
          }
        }
      }
    }
  }
}

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
  auto rng_naive = ankerl::nanobench::Rng(42);
  benchmark_utils::fill_2d_uniform01(mlr_in, rng_naive);
  benchmark_utils::fill_2d_copy(mlrtr_l1_in, mlr_in);
  benchmark_utils::fill_2d_copy(mlrtr_l2_in, mlr_in);

  // Checks
#if defined(LAYOUT_TILED_ENABLE_CHECKS)
  const double n     = static_cast<double>(cfg.n_);
  const double big_o = 5.0 * n * n;
  naive(mlr_in, mlr_out);
  double ref = benchmark_utils::l2_norm(mlr_out);

  naive(mlrtr_l1_in, mlrtr_l1_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_naive", big_o);
  naive(mlrtr_l2_in, mlrtr_l2_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_naive", big_o);

  cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L1", big_o);
  cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlr_out), "R_cb-L2", big_o);

  manual_blocked_tiled<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_mbt", big_o);
  manual_blocked_tiled<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_mbt", big_o);

  manual_blocked_tiled_subview<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l1_out), "RTR-L1_mbtsv", big_o);
  manual_blocked_tiled_subview<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  benchmark_utils::check(ref, benchmark_utils::l2_norm(mlrtr_l2_out), "RTR-L2_mbtsv", big_o);
#endif

  // Benchmarks
  bs.run("R naive", [&]() { naive(mlr_in, mlr_out); });

  bs.run(fmt::format("R cb-{0}x{0}", TILE_L1), [&]() { cache_blocked<TILE_L1, TILE_L1>(mlr_in, mlr_out); });
  bs.run(fmt::format("R cb-{0}x{0}", TILE_L2), [&]() { cache_blocked<TILE_L2, TILE_L2>(mlr_in, mlr_out); });

  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L1), [&]() { naive(mlrtr_l1_in, mlrtr_l1_out); });
  bs.run(fmt::format("RTR-{0}x{0} naive", TILE_L2), [&]() { naive(mlrtr_l2_in, mlrtr_l2_out); });

  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L1), [&]() {
    manual_blocked_tiled<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbt", TILE_L2), [&]() {
    manual_blocked_tiled<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  });

  bs.run(fmt::format("RTR-{0}x{0} mbtsv", TILE_L1), [&]() {
    manual_blocked_tiled_subview<TILE_L1, TILE_L1>(mlrtr_l1_in, mlrtr_l1_out);
  });
  bs.run(fmt::format("RTR-{0}x{0} mbtsv", TILE_L2), [&]() {
    manual_blocked_tiled_subview<TILE_L2, TILE_L2>(mlrtr_l2_in, mlrtr_l2_out);
  });

  // Output
  bs.render(ankerl::nanobench::templates::csv(), *out_stream);
  file_stream.close();
}

} // namespace

auto main(int argc, char* argv[]) -> int {
  static std::string HELP_MSG = "Execute 2D stencil benchmark\n\n"
                                "Usage: layout_tiled.benchmarks.stencil_2d [OPTIONS]\n\n"
                                "Options:\n"
                                "  -n, --size <SIZE>             Matrix size [default: 512]\n"
                                // "  -t, --tile <SIZE>             Tile size [default: 64]\n"
                                // "  -d, --delta <VAL>             Stencil delta (order) [default: 1]\n"
                                "  -o, --output <FILENAME>       Output file name\n"
                                "  -h, --help                    Show this help\n";

  static struct option long_options[] = {
    {"size", required_argument, nullptr, 'n'},
    // {"tile", required_argument, nullptr, 't'},
    // {"delta", required_argument, nullptr, 'd'},
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
