// include/iai/detail/native_tiled_blocked_fragment_reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_NATIVE_TILED_BLOCKED_FRAGMENT_REDUCE
#define INCLUDED_IAI_DETAIL_NATIVE_TILED_BLOCKED_FRAGMENT_REDUCE

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "native_tiled_cache_block.hpp"
#include "native_tiled_fragment_reduce.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define IAI_NATIVE_BLOCKED_FRAGMENT_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define IAI_NATIVE_BLOCKED_FRAGMENT_ALWAYS_INLINE inline
#endif

namespace iai::detail {

template <class Output, class OutputTile, std::size_t ProblemRank>
IAI_NATIVE_BLOCKED_FRAGMENT_ALWAYS_INLINE auto initialize_owned_native_output_tile(
  const Output& output,
  const OutputTile& output_tile,
  const std::array<std::size_t, ProblemRank>& tile_extents
) -> void {
  if constexpr (!Output::is_inout) {
    auto local_coordinates = std::array<std::size_t, ProblemRank>{};
    auto initialize        = [&](const auto& output_local_coordinates) {
      output_tile.at(output_local_coordinates) = static_cast<typename Output::value_type>(output.identity());
    };
    for_each_native_output_element<0, Output>(tile_extents, local_coordinates, initialize);
  }
}

template <
  std::size_t Rows,
  std::size_t Columns,
  class ReductionAxes,
  class InputTuple,
  std::size_t ProblemRank,
  class OutputTile,
  class InputTiles,
  class Updater>
IAI_NATIVE_BLOCKED_FRAGMENT_ALWAYS_INLINE auto execute_full_native_reduction_tile_fragment(
  const std::size_t row_begin,
  const std::size_t column_begin,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const OutputTile& output_tile,
  const InputTiles& input_tiles,
  Updater& updater
) -> void {
  using output_type                   = typename OutputTile::binding_type;
  using value_type                    = typename output_type::value_type;
  constexpr std::size_t simd_axis     = native_fragment_vector_axis<output_type>();
  constexpr std::size_t unrolled_axis = native_fragment_row_axis<output_type>();
  std::array<std::array<value_type, Columns>, Rows> fragment{};

  static_for<Rows>([&]<std::size_t Row>() {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (std::size_t column = 0; column < Columns; ++column) {
      if constexpr (output_type::axes_type::size > 1) {
        local_coordinates[unrolled_axis] = row_begin + Row;
      }
      local_coordinates[simd_axis] = column_begin + column;
      fragment[Row][column]        = static_cast<value_type>(output_tile.at(local_coordinates));
    }
  });

  auto update = [&](const auto&, const auto& reduction_local_coordinates) {
    local_coordinates = reduction_local_coordinates;
    static_for<Rows>([&]<std::size_t Row>() {
      if constexpr (output_type::axes_type::size > 1) {
        local_coordinates[unrolled_axis] = row_begin + Row;
      }
#if defined(_OPENMP)
#pragma omp simd
#endif
      for (std::size_t column = 0; column < Columns; ++column) {
        local_coordinates[simd_axis] = column_begin + column;
        invoke_updater(
          updater,
          fragment[Row][column],
          input_tiles,
          local_coordinates,
          std::make_index_sequence<std::tuple_size_v<InputTuple>>()
        );
      }
    });
  };
  for_each_native_reduction_element<0, ReductionAxes, InputTuple>(
    tile_extents,
    local_coordinates,
    input_tiles,
    update
  );

  static_for<Rows>([&]<std::size_t Row>() {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (std::size_t column = 0; column < Columns; ++column) {
      if constexpr (output_type::axes_type::size > 1) {
        local_coordinates[unrolled_axis] = row_begin + Row;
      }
      local_coordinates[simd_axis]      = column_begin + column;
      output_tile.at(local_coordinates) = fragment[Row][column];
    }
  });
}

template <
  class ReductionAxes,
  class InputTuple,
  std::size_t ProblemRank,
  class OutputTile,
  class InputTiles,
  class Updater>
IAI_NATIVE_BLOCKED_FRAGMENT_ALWAYS_INLINE auto execute_bound_native_reduction_tile_fragments(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const OutputTile& output_tile,
  const InputTiles& input_tiles,
  Updater& updater
) -> void {
  using output_type                   = typename OutputTile::binding_type;
  using value_type                    = typename output_type::value_type;
  constexpr std::size_t simd_axis     = native_fragment_vector_axis<output_type>();
  constexpr std::size_t unrolled_axis = native_fragment_row_axis<output_type>();
  constexpr std::size_t mr            = output_type::axes_type::size > 1 ? 4 : 1;
  constexpr std::size_t nr            = 2 * native_simd_lanes<value_type>();

  auto local_coordinates = std::array<std::size_t, ProblemRank>{};
  auto plane             = [&](const auto& plane_coordinates) {
    local_coordinates            = plane_coordinates;
    const std::size_t row_extent = output_type::axes_type::size > 1 ? tile_extents[unrolled_axis] : std::size_t{1};
    for (std::size_t row_begin = 0; row_begin < row_extent; row_begin += mr) {
      const std::size_t row_count = std::min(mr, row_extent - row_begin);
      for (std::size_t column_begin = 0; column_begin < tile_extents[simd_axis]; column_begin += nr) {
        const std::size_t column_count = std::min(nr, tile_extents[simd_axis] - column_begin);
        if (row_count == mr && column_count == nr) {
          execute_full_native_reduction_tile_fragment<mr, nr, ReductionAxes, InputTuple>(
            row_begin,
            column_begin,
            tile_extents,
            local_coordinates,
            output_tile,
            input_tiles,
            updater
          );
          continue;
        }

        std::array<std::array<value_type, nr>, mr> fragment{};
        for (std::size_t row = 0; row < row_count; ++row) {
#if defined(_OPENMP)
#pragma omp simd
#endif
          for (std::size_t column = 0; column < column_count; ++column) {
            if constexpr (output_type::axes_type::size > 1) {
              local_coordinates[unrolled_axis] = row_begin + row;
            }
            local_coordinates[simd_axis] = column_begin + column;
            fragment[row][column]        = static_cast<value_type>(output_tile.at(local_coordinates));
          }
        }

        auto update = [&](const auto&, const auto& reduction_local_coordinates) {
          local_coordinates = reduction_local_coordinates;
          for (std::size_t row = 0; row < row_count; ++row) {
            if constexpr (output_type::axes_type::size > 1) {
              local_coordinates[unrolled_axis] = row_begin + row;
            }
#if defined(_OPENMP)
#pragma omp simd
#endif
            for (std::size_t column = 0; column < column_count; ++column) {
              local_coordinates[simd_axis] = column_begin + column;
              invoke_updater(
                updater,
                fragment[row][column],
                input_tiles,
                local_coordinates,
                std::make_index_sequence<std::tuple_size_v<InputTuple>>()
              );
            }
          }
        };
        for_each_native_reduction_element<0, ReductionAxes, InputTuple>(
          tile_extents,
          local_coordinates,
          input_tiles,
          update
        );

        for (std::size_t row = 0; row < row_count; ++row) {
#if defined(_OPENMP)
#pragma omp simd
#endif
          for (std::size_t column = 0; column < column_count; ++column) {
            if constexpr (output_type::axes_type::size > 1) {
              local_coordinates[unrolled_axis] = row_begin + row;
            }
            local_coordinates[simd_axis]      = column_begin + column;
            output_tile.at(local_coordinates) = fragment[row][column];
          }
        }
      }
    }
  };
  for_each_native_fragment_plane<0, output_type>(tile_extents, local_coordinates, plane);
}

template <class ReductionAxes, std::size_t ProblemRank, class Output, class InputTuple, class Updater>
auto execute_reduction_blocked_native_fragment_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  const Output& output,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  const auto output_tile = bind_projected_native_tile<ProblemRank>(output, output_tile_coordinates);
  initialize_owned_native_output_tile(output, output_tile, tile_extents);
  const auto input_ranges = bind_projected_native_input_tile_ranges(
    inputs,
    output_tile_coordinates,
    std::make_index_sequence<std::tuple_size_v<InputTuple>>()
  );

  auto reduction_tile_coordinates = output_tile_coordinates;
  auto reduction_tile             = [&](const auto& input_tiles) {
    execute_bound_native_reduction_tile_fragments<ReductionAxes, InputTuple>(
      tile_extents,
      output_tile,
      input_tiles,
      updater
    );
  };
  for_each_bound_native_reduction_tile_in_ranges<0, ReductionAxes, ProblemRank, InputTuple>(
    tile_counts,
    reduction_tile_coordinates,
    input_ranges,
    reduction_tile
  );
}

template <class ReductionAxes, class Output, class InputTuple, class Updater>
auto host_reduction_blocked_native_fragment_reduce(const Output& output, const InputTuple& inputs, Updater& updater)
  -> void {
  static_assert(native_complete_fragment_compatible_v<Output, InputTuple>);
  auto execute_tile = []<std::size_t ProblemRank>(
                        const std::array<std::size_t, ProblemRank>& tile_counts,
                        const std::array<std::size_t, ProblemRank>& tile_extents,
                        const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
                        const auto& selected_output,
                        const auto& selected_inputs,
                        auto& selected_updater
                      ) {
    execute_reduction_blocked_native_fragment_tile<ReductionAxes>(
      tile_counts,
      tile_extents,
      output_tile_coordinates,
      selected_output,
      selected_inputs,
      selected_updater
    );
  };
  host_native_tiled_reduce<ReductionAxes>(output, inputs, updater, execute_tile);
}

template <class ReductionAxes, std::size_t ProblemRank, class Output, class InputTuple, class Updater>
auto execute_rectangular_reduction_blocked_native_fragment(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_block_coordinates,
  const native_tile_block_shape<ProblemRank>& block_shape,
  const Output& output,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  const auto output_block      = make_native_tile_block(output_block_coordinates, tile_counts, block_shape);
  auto output_tile_coordinates = output_block.begin;
  auto initialize_output_tile  = [&](const auto& coordinates) {
    const auto output_tile = bind_projected_native_tile<ProblemRank>(output, coordinates);
    initialize_owned_native_output_tile(output, output_tile, tile_extents);
  };
  for_each_native_tile_in_block<0, typename Output::axes_type>(
    output_block,
    output_tile_coordinates,
    initialize_output_tile
  );

  const auto block_counts = native_tile_block_counts(tile_counts, block_shape);
  native_tile_block<ProblemRank> reduction_block_grid{};
  reduction_block_grid.extents     = block_counts;
  auto reduction_block_coordinates = output_block_coordinates;
  auto reduction_block             = [&](const auto& current_reduction_block_coordinates) {
    auto combined_block_coordinates = output_block_coordinates;
    for (const std::size_t axis : ReductionAxes::values) {
      combined_block_coordinates[axis] = current_reduction_block_coordinates[axis];
    }
    const auto current_block = make_native_tile_block(combined_block_coordinates, tile_counts, block_shape);
    const auto input_ranges  = bind_projected_native_input_tile_ranges(
      inputs,
      current_block.begin,
      std::make_index_sequence<std::tuple_size_v<InputTuple>>()
    );

    auto reduction_tile_coordinates = current_block.begin;
    auto reduction_tile             = [&](const auto& current_reduction_tile_coordinates) {
      auto current_output_tile_coordinates = current_block.begin;
      auto output_tile                     = [&](const auto& current_output_coordinates) {
        auto problem_tile_coordinates = current_reduction_tile_coordinates;
        for (const std::size_t axis : Output::axes_type::values) {
          problem_tile_coordinates[axis] = current_output_coordinates[axis];
        }
        const auto bound_output_tile = bind_projected_native_tile<ProblemRank>(output, problem_tile_coordinates);
        const auto input_tiles       = bind_projected_native_input_tiles_from_ranges(
          input_ranges,
          problem_tile_coordinates,
          std::make_index_sequence<std::tuple_size_v<InputTuple>>()
        );
        execute_bound_native_reduction_tile_fragments<ReductionAxes, InputTuple>(
          tile_extents,
          bound_output_tile,
          input_tiles,
          updater
        );
      };
      for_each_native_tile_in_block<0, typename Output::axes_type>(
        current_block,
        current_output_tile_coordinates,
        output_tile
      );
    };
    for_each_native_tile_in_block<0, ReductionAxes>(current_block, reduction_tile_coordinates, reduction_tile);
  };
  for_each_native_tile_in_block<0, ReductionAxes>(reduction_block_grid, reduction_block_coordinates, reduction_block);
}

template <class ReductionAxes, std::size_t ProblemRank, class Output, class InputTuple, class Updater>
auto host_rectangular_reduction_blocked_native_fragment_reduce(
  const Output& output,
  const InputTuple& inputs,
  Updater& updater,
  const native_tile_block_shape<ProblemRank>& block_shape
) -> void {
  static_assert(native_complete_fragment_compatible_v<Output, InputTuple>);
  using all_bindings_type           = decltype(std::tuple_cat(std::tuple{output}, inputs));
  constexpr std::size_t input_count = std::tuple_size_v<InputTuple>;
  constexpr std::size_t problem_rank_value
    = problem_rank<ReductionAxes, all_bindings_type>(std::make_index_sequence<input_count + 1>());
  static_assert(ProblemRank == problem_rank_value);
  constexpr auto tile_extents
    = common_native_tile_extents<ProblemRank, all_bindings_type>(std::make_index_sequence<input_count + 1>());

  const auto all_bindings = std::tuple_cat(std::tuple{output}, inputs);
  const auto extents = merge_dynamic_extents<ProblemRank>(all_bindings, std::make_index_sequence<input_count + 1>());
  assert_no_aliases(output, inputs, std::make_index_sequence<input_count>());
  assert(block_shape.valid());

  std::array<std::size_t, ProblemRank> tile_counts{};
  for (std::size_t axis = 0; axis < ProblemRank; ++axis) {
    assert(extents[axis] % tile_extents[axis] == 0);
    tile_counts[axis] = extents[axis] / tile_extents[axis];
  }
  const auto block_counts = native_tile_block_counts(tile_counts, block_shape);

  std::size_t output_block_count = 1;
  for (const std::size_t axis : Output::axes_type::values) {
    output_block_count *= block_counts[axis];
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t work_item = 0; work_item < output_block_count; ++work_item) {
    const auto output_block_coordinates = decode_native_output_tile<ProblemRank, Output>(work_item, block_counts);
    execute_rectangular_reduction_blocked_native_fragment<
      ReductionAxes>(tile_counts, tile_extents, output_block_coordinates, block_shape, output, inputs, updater);
  }
}

} // namespace iai::detail

#undef IAI_NATIVE_BLOCKED_FRAGMENT_ALWAYS_INLINE

#endif // INCLUDED_IAI_DETAIL_NATIVE_TILED_BLOCKED_FRAGMENT_REDUCE
