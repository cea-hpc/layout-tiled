// include/iai/detail/native_tiled_fragment_reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_NATIVE_TILED_FRAGMENT_REDUCE
#define INCLUDED_IAI_DETAIL_NATIVE_TILED_FRAGMENT_REDUCE

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "native_tiled_reduce.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define IAI_NATIVE_FRAGMENT_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define IAI_NATIVE_FRAGMENT_ALWAYS_INLINE inline
#endif

namespace iai::detail {

template <class Output>
consteval auto native_fragment_vector_axis() -> std::size_t {
  if constexpr (std::is_same_v<typename native_mapping_traits<Output>::inner_layout_type, std::layout_right>) {
    return Output::axes_type::values.back();
  } else {
    return Output::axes_type::values.front();
  }
}

template <class Output>
consteval auto native_fragment_row_axis() -> std::size_t {
  if constexpr (Output::axes_type::size == 1) {
    return native_fragment_vector_axis<Output>();
  } else if constexpr (std::is_same_v<typename native_mapping_traits<Output>::inner_layout_type, std::layout_right>) {
    return Output::axes_type::values[Output::axes_type::size - 2];
  } else {
    return Output::axes_type::values[1];
  }
}

template <class Input, std::size_t VectorAxis>
consteval auto native_input_is_contiguous_or_invariant() -> bool {
  constexpr std::size_t dimension = dimension_for_axis<Input>(VectorAxis);
  if constexpr (dimension == Input::axes_type::size) {
    return true;
  } else if constexpr (std::is_same_v<typename native_mapping_traits<Input>::inner_layout_type, std::layout_right>) {
    return dimension + 1 == Input::axes_type::size;
  } else if constexpr (std::is_same_v<typename native_mapping_traits<Input>::inner_layout_type, std::layout_left>) {
    return dimension == 0;
  } else {
    return false;
  }
}

template <class Output, class InputTuple, std::size_t... InputIndexes>
consteval auto native_complete_fragment_compatible(std::index_sequence<InputIndexes...>) -> bool {
  constexpr std::size_t simd_axis = native_fragment_vector_axis<Output>();
  return std::is_arithmetic_v<typename Output::value_type>
         && (std::is_arithmetic_v<typename std::tuple_element_t<InputIndexes, InputTuple>::value_type> && ...)
         && (native_input_is_contiguous_or_invariant<std::tuple_element_t<InputIndexes, InputTuple>, simd_axis>() && ...);
}

template <class Output, class InputTuple>
inline constexpr bool native_complete_fragment_compatible_v
  = native_complete_fragment_compatible<Output, InputTuple>(std::make_index_sequence<std::tuple_size_v<InputTuple>>());

template <class Value>
consteval auto native_simd_lanes() -> std::size_t {
#if defined(__AVX512F__)
  constexpr std::size_t register_bytes = 64;
#elif defined(__AVX__)
  constexpr std::size_t register_bytes = 32;
#elif defined(__SSE2__) || defined(__aarch64__)
  constexpr std::size_t register_bytes = 16;
#else
  constexpr std::size_t register_bytes = sizeof(Value);
#endif
  return std::max<std::size_t>(1, register_bytes / sizeof(Value));
}

template <std::size_t Depth, class Output, std::size_t ProblemRank, class Function>
IAI_NATIVE_FRAGMENT_ALWAYS_INLINE auto for_each_native_fragment_plane(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  Function& function
) -> void {
  constexpr auto order           = native_output_inner_order<Output>();
  constexpr std::size_t simd     = native_fragment_vector_axis<Output>();
  constexpr std::size_t unrolled = native_fragment_row_axis<Output>();
  if constexpr (Depth == Output::axes_type::size) {
    function(local_coordinates);
  } else {
    constexpr std::size_t axis = order[Depth];
    if constexpr (axis == simd || axis == unrolled) {
      for_each_native_fragment_plane<Depth + 1, Output>(tile_extents, local_coordinates, function);
    } else {
      for (std::size_t coordinate = 0; coordinate < tile_extents[axis]; ++coordinate) {
        local_coordinates[axis] = coordinate;
        for_each_native_fragment_plane<Depth + 1, Output>(tile_extents, local_coordinates, function);
      }
    }
  }
}

template <class Function, std::size_t... Indexes>
IAI_NATIVE_FRAGMENT_ALWAYS_INLINE auto static_for_impl(Function& function, std::index_sequence<Indexes...>) -> void {
  (function.template operator()<Indexes>(), ...);
}

template <std::size_t Count, class Function>
IAI_NATIVE_FRAGMENT_ALWAYS_INLINE auto static_for(Function&& function) -> void {
  static_for_impl(function, std::make_index_sequence<Count>());
}

template <
  std::size_t Rows,
  std::size_t Columns,
  class ReductionAxes,
  class InputTuple,
  std::size_t ProblemRank,
  class Output,
  class OutputTile,
  class InputRangeTuple,
  class Updater>
IAI_NATIVE_FRAGMENT_ALWAYS_INLINE auto execute_full_native_fragment(
  const std::size_t row_begin,
  const std::size_t column_begin,
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const Output& output,
  const OutputTile& output_tile,
  const InputRangeTuple& input_ranges,
  Updater& updater
) -> void {
  using value_type                    = typename Output::value_type;
  constexpr std::size_t simd_axis     = native_fragment_vector_axis<Output>();
  constexpr std::size_t unrolled_axis = native_fragment_row_axis<Output>();
  std::array<std::array<value_type, Columns>, Rows> fragment{};

  static_for<Rows>([&]<std::size_t Row>() {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (std::size_t column = 0; column < Columns; ++column) {
      if constexpr (Output::axes_type::size > 1) {
        local_coordinates[unrolled_axis] = row_begin + Row;
      }
      local_coordinates[simd_axis] = column_begin + column;
      fragment[Row][column]        = initialize_native_accumulator(output, output_tile, local_coordinates);
    }
  });

  auto tile_coordinates = output_tile_coordinates;
  auto update           = [&](const auto& input_tiles, const auto& reduction_local_coordinates) {
    local_coordinates = reduction_local_coordinates;
    static_for<Rows>([&]<std::size_t Row>() {
      if constexpr (Output::axes_type::size > 1) {
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
  for_each_native_reduction_tile_in_ranges<0, ReductionAxes, ProblemRank, InputTuple>(
    tile_counts,
    tile_extents,
    tile_coordinates,
    local_coordinates,
    input_ranges,
    update
  );

  static_for<Rows>([&]<std::size_t Row>() {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (std::size_t column = 0; column < Columns; ++column) {
      if constexpr (Output::axes_type::size > 1) {
        local_coordinates[unrolled_axis] = row_begin + Row;
      }
      local_coordinates[simd_axis]      = column_begin + column;
      output_tile.at(local_coordinates) = fragment[Row][column];
    }
  });
}

template <class ReductionAxes, std::size_t ProblemRank, class Output, class InputTuple, class Updater>
auto execute_complete_native_fragment_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  const Output& output,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  using value_type                    = typename Output::value_type;
  constexpr std::size_t simd_axis     = native_fragment_vector_axis<Output>();
  constexpr std::size_t unrolled_axis = native_fragment_row_axis<Output>();
  constexpr std::size_t mr            = Output::axes_type::size > 1 ? 4 : 1;
  constexpr std::size_t nr            = 2 * native_simd_lanes<value_type>();

  auto output_tile        = bind_projected_native_tile<ProblemRank>(output, output_tile_coordinates);
  const auto input_ranges = bind_projected_native_input_tile_ranges(
    inputs,
    output_tile_coordinates,
    std::make_index_sequence<std::tuple_size_v<InputTuple>>()
  );
  auto local_coordinates = std::array<std::size_t, ProblemRank>{};

  auto plane = [&](const auto& plane_coordinates) {
    local_coordinates            = plane_coordinates;
    const std::size_t row_extent = Output::axes_type::size > 1 ? tile_extents[unrolled_axis] : std::size_t{1};
    for (std::size_t row_begin = 0; row_begin < row_extent; row_begin += mr) {
      const std::size_t row_count = std::min(mr, row_extent - row_begin);
      for (std::size_t column_begin = 0; column_begin < tile_extents[simd_axis]; column_begin += nr) {
        const std::size_t column_count = std::min(nr, tile_extents[simd_axis] - column_begin);
        if (row_count == mr && column_count == nr) {
          execute_full_native_fragment<mr, nr, ReductionAxes, InputTuple>(
            row_begin,
            column_begin,
            tile_counts,
            tile_extents,
            output_tile_coordinates,
            local_coordinates,
            output,
            output_tile,
            input_ranges,
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
            if constexpr (Output::axes_type::size > 1) {
              local_coordinates[unrolled_axis] = row_begin + row;
            }
            local_coordinates[simd_axis] = column_begin + column;
            fragment[row][column]        = initialize_native_accumulator(output, output_tile, local_coordinates);
          }
        }

        auto tile_coordinates = output_tile_coordinates;
        auto update           = [&](const auto& input_tiles, const auto& reduction_local_coordinates) {
          local_coordinates = reduction_local_coordinates;
          for (std::size_t row = 0; row < row_count; ++row) {
            if constexpr (Output::axes_type::size > 1) {
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
        for_each_native_reduction_tile_in_ranges<0, ReductionAxes, ProblemRank, InputTuple>(
          tile_counts,
          tile_extents,
          tile_coordinates,
          local_coordinates,
          input_ranges,
          update
        );

        for (std::size_t row = 0; row < row_count; ++row) {
#if defined(_OPENMP)
#pragma omp simd
#endif
          for (std::size_t column = 0; column < column_count; ++column) {
            if constexpr (Output::axes_type::size > 1) {
              local_coordinates[unrolled_axis] = row_begin + row;
            }
            local_coordinates[simd_axis]      = column_begin + column;
            output_tile.at(local_coordinates) = fragment[row][column];
          }
        }
      }
    }
  };
  for_each_native_fragment_plane<0, Output>(tile_extents, local_coordinates, plane);
}

template <class ReductionAxes, class Output, class InputTuple, class Updater>
auto host_complete_native_fragment_reduce(const Output& output, const InputTuple& inputs, Updater& updater) -> void {
  auto execute_tile = []<std::size_t ProblemRank>(
                        const std::array<std::size_t, ProblemRank>& tile_counts,
                        const std::array<std::size_t, ProblemRank>& tile_extents,
                        const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
                        const auto& selected_output,
                        const auto& selected_inputs,
                        auto& selected_updater
                      ) {
    execute_complete_native_fragment_tile<ReductionAxes>(
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

} // namespace iai::detail

#undef IAI_NATIVE_FRAGMENT_ALWAYS_INLINE

#endif // INCLUDED_IAI_DETAIL_NATIVE_TILED_FRAGMENT_REDUCE
