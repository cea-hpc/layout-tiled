// include/iai/detail/native_tiled_reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_NATIVE_TILED_REDUCE
#define INCLUDED_IAI_DETAIL_NATIVE_TILED_REDUCE

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "layout_tiled_mapping.hpp"
#include "md_reduce.hpp"
#include "native_tiled_view.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE inline
#endif

namespace iai::detail {

template <class Binding>
using native_mapping_traits = layout_tiled_mapping_traits<typename Binding::view_type::mapping_type>;

template <class Binding>
consteval auto native_tile_extent_for_axis(const std::size_t axis) -> std::size_t {
  const std::size_t dimension = dimension_for_axis<Binding>(axis);
  if (dimension == Binding::axes_type::size) {
    return 0;
  }
  return native_mapping_traits<Binding>::tile_extents[dimension];
}

template <class Tuple, std::size_t... BindingIndexes>
consteval auto native_tile_extents_are_compatible(const std::size_t rank, std::index_sequence<BindingIndexes...>)
  -> bool {
  for (std::size_t axis = 0; axis < rank; ++axis) {
    std::size_t known = 0;
    bool compatible   = true;
    const auto merge  = [&](const std::size_t candidate) {
      if (candidate == 0) {
        return;
      }
      if (known != 0 && known != candidate) {
        compatible = false;
      }
      known = candidate;
    };
    (merge(native_tile_extent_for_axis<std::tuple_element_t<BindingIndexes, Tuple>>(axis)), ...);
    if (!compatible || known == 0) {
      return false;
    }
  }
  return true;
}

template <class ReductionAxes, class... Arguments>
consteval auto valid_native_tiled_reduce_call() -> bool {
  if constexpr (!valid_reduce_call<ReductionAxes, Arguments...>()) {
    return false;
  } else {
    using tuple_type                    = std::tuple<std::remove_cvref_t<Arguments>...>;
    constexpr std::size_t binding_count = sizeof...(Arguments) - 1;
    using binding_indexes               = std::make_index_sequence<binding_count>;
    constexpr bool all_native_tiled     = []<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
      return (recognized_layout_tiled_mdspan<typename std::tuple_element_t<Indexes, tuple_type>::view_type> && ...);
    }(binding_indexes{});
    constexpr std::size_t rank = problem_rank<ReductionAxes, tuple_type>(binding_indexes{});
    if constexpr (!all_native_tiled) {
      return false;
    } else {
      return native_tile_extents_are_compatible<tuple_type>(rank, binding_indexes{});
    }
  }
}

template <class Binding, std::size_t ProblemRank>
class projected_native_tile {
public:
  using binding_type = Binding;
  using tiled_type   = native_tiled_view<typename binding_type::view_type>;
  using tile_type    = typename tiled_type::tile_type;

  explicit constexpr projected_native_tile(tile_type tile) noexcept : tile_(tile) {}

  [[nodiscard]] IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE decltype(auto)
  at(const std::array<std::size_t, ProblemRank>& local_coordinates) const noexcept {
    const auto coordinates = project_coordinates<ProblemRank, binding_type>(local_coordinates);
    return tile_[coordinates];
  }

private:
  tile_type tile_;
};

template <class Binding, std::size_t ProblemRank>
class projected_native_tile_range {
public:
  using binding_type = Binding;
  using tiled_type   = native_tiled_view<typename binding_type::view_type>;
  using range_type   = native_tile_range<typename binding_type::view_type>;
  using tile_type    = projected_native_tile<binding_type, ProblemRank>;

  explicit constexpr projected_native_tile_range(
    const binding_type& binding,
    const std::array<std::size_t, ProblemRank>& problem_tile_origin
  ) noexcept :
      range_(
        tiled_type(binding.view()),
        project_coordinates<ProblemRank, binding_type, typename binding_type::view_type::index_type>(
          problem_tile_origin
        )
      ) {}

  [[nodiscard]] IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto
  bind_tile(const std::array<std::size_t, ProblemRank>& problem_tile_coordinates) const noexcept -> tile_type {
    const auto projected_coordinates
      = project_coordinates<ProblemRank, binding_type, typename binding_type::view_type::index_type>(
        problem_tile_coordinates
      );
    return tile_type(range_.bind_tile(projected_coordinates));
  }

private:
  range_type range_;
};

template <std::size_t ProblemRank, class Binding>
[[nodiscard]] IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto bind_projected_native_tile(
  const Binding& binding,
  const std::array<std::size_t, ProblemRank>& problem_tile_coordinates
) noexcept -> projected_native_tile<Binding, ProblemRank> {
  const auto projected_coordinates
    = project_coordinates<ProblemRank, Binding, typename Binding::view_type::index_type>(problem_tile_coordinates);
  const auto tiled = native_tiled_view(binding.view());
  return projected_native_tile<Binding, ProblemRank>(tiled.bind_tile(projected_coordinates));
}

template <std::size_t ProblemRank, class BindingTuple, std::size_t... BindingIndexes>
constexpr auto common_native_tile_extents(std::index_sequence<BindingIndexes...>)
  -> std::array<std::size_t, ProblemRank> {
  std::array<std::size_t, ProblemRank> result{};
  for (std::size_t axis = 0; axis < ProblemRank; ++axis) {
    ((result[axis]
      = std::max(result[axis], native_tile_extent_for_axis<std::tuple_element_t<BindingIndexes, BindingTuple>>(axis))),
     ...);
  }
  return result;
}

template <class Binding>
consteval auto native_inner_stride_for_axis(const std::size_t axis) -> std::size_t {
  const std::size_t dimension = dimension_for_axis<Binding>(axis);
  if (dimension == Binding::axes_type::size) {
    return std::numeric_limits<std::size_t>::max();
  }

  std::size_t stride = 1;
  if constexpr (std::is_same_v<typename native_mapping_traits<Binding>::inner_layout_type, std::layout_right>) {
    for (std::size_t current = dimension + 1; current < Binding::axes_type::size; ++current) {
      stride *= native_mapping_traits<Binding>::tile_extents[current];
    }
  } else {
    for (std::size_t current = 0; current < dimension; ++current) {
      stride *= native_mapping_traits<Binding>::tile_extents[current];
    }
  }
  return stride;
}

template <class InputTuple, std::size_t... InputIndexes>
consteval auto minimum_native_input_stride(const std::size_t axis, std::index_sequence<InputIndexes...>)
  -> std::size_t {
  std::size_t result = std::numeric_limits<std::size_t>::max();
  ((result = std::min(result, native_inner_stride_for_axis<std::tuple_element_t<InputIndexes, InputTuple>>(axis))),
   ...);
  return result;
}

template <class ReductionAxes, class InputTuple>
consteval auto ordered_native_reduction_axes() -> std::array<std::size_t, ReductionAxes::size> {
  auto result   = ReductionAxes::values;
  using indexes = std::make_index_sequence<std::tuple_size_v<InputTuple>>;
  for (std::size_t left = 0; left < result.size(); ++left) {
    for (std::size_t right = left + 1; right < result.size(); ++right) {
      if (minimum_native_input_stride<InputTuple>(result[right], indexes{})
          > minimum_native_input_stride<InputTuple>(result[left], indexes{})) {
        std::swap(result[left], result[right]);
      }
    }
  }
  return result;
}

template <class Output>
consteval auto native_output_inner_order() -> std::array<std::size_t, Output::axes_type::size> {
  auto result = Output::axes_type::values;
  if constexpr (std::is_same_v<typename native_mapping_traits<Output>::inner_layout_type, std::layout_left>) {
    std::reverse(result.begin(), result.end());
  }
  return result;
}

template <std::size_t ProblemRank, class InputTuple, std::size_t... InputIndexes>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto bind_projected_native_input_tiles(
  const InputTuple& inputs,
  const std::array<std::size_t, ProblemRank>& tile_coordinates,
  std::index_sequence<InputIndexes...>
) noexcept {
  return std::tuple{bind_projected_native_tile<ProblemRank>(std::get<InputIndexes>(inputs), tile_coordinates)...};
}

template <std::size_t ProblemRank, class InputTuple, std::size_t... InputIndexes>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto bind_projected_native_input_tile_ranges(
  const InputTuple& inputs,
  const std::array<std::size_t, ProblemRank>& tile_origin,
  std::index_sequence<InputIndexes...>
) noexcept {
  return std::tuple{projected_native_tile_range<std::tuple_element_t<InputIndexes, InputTuple>, ProblemRank>(
    std::get<InputIndexes>(inputs),
    tile_origin
  )...};
}

template <std::size_t ProblemRank, class InputRangeTuple, std::size_t... InputIndexes>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto bind_projected_native_input_tiles_from_ranges(
  const InputRangeTuple& ranges,
  const std::array<std::size_t, ProblemRank>& tile_coordinates,
  std::index_sequence<InputIndexes...>
) noexcept {
  return std::tuple{std::get<InputIndexes>(ranges).bind_tile(tile_coordinates)...};
}

template <
  std::size_t Depth,
  class ReductionAxes,
  class InputTuple,
  std::size_t ProblemRank,
  class InputTiles,
  class Function>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto for_each_native_reduction_element(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const InputTiles& input_tiles,
  Function& function
) -> void {
  constexpr auto order = ordered_native_reduction_axes<ReductionAxes, InputTuple>();
  if constexpr (Depth == ReductionAxes::size) {
    function(input_tiles, local_coordinates);
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_extents[axis]; ++coordinate) {
      local_coordinates[axis] = coordinate;
      for_each_native_reduction_element<Depth + 1, ReductionAxes, InputTuple>(
        tile_extents,
        local_coordinates,
        input_tiles,
        function
      );
    }
  }
}

template <std::size_t Depth, class ReductionAxes, std::size_t ProblemRank, class InputTuple, class Function>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto for_each_bound_native_reduction_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  std::array<std::size_t, ProblemRank>& tile_coordinates,
  const InputTuple& inputs,
  Function& function
) -> void {
  constexpr auto order = ordered_native_reduction_axes<ReductionAxes, InputTuple>();
  if constexpr (Depth == ReductionAxes::size) {
    const auto input_tiles = bind_projected_native_input_tiles(
      inputs,
      tile_coordinates,
      std::make_index_sequence<std::tuple_size_v<InputTuple>>()
    );
    function(input_tiles);
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_counts[axis]; ++coordinate) {
      tile_coordinates[axis] = coordinate;
      for_each_bound_native_reduction_tile<Depth + 1, ReductionAxes>(tile_counts, tile_coordinates, inputs, function);
    }
  }
}

template <std::size_t Depth, class ReductionAxes, std::size_t ProblemRank, class InputTuple, class Function>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto for_each_native_reduction_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& tile_coordinates,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const InputTuple& inputs,
  Function& function
) -> void {
  auto visit_bound_tile = [&](const auto& input_tiles) {
    for_each_native_reduction_element<0, ReductionAxes, InputTuple>(
      tile_extents,
      local_coordinates,
      input_tiles,
      function
    );
  };
  for_each_bound_native_reduction_tile<Depth, ReductionAxes>(tile_counts, tile_coordinates, inputs, visit_bound_tile);
}

template <
  std::size_t Depth,
  class ReductionAxes,
  std::size_t ProblemRank,
  class InputTuple,
  class InputRangeTuple,
  class Function>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto for_each_bound_native_reduction_tile_in_ranges(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  std::array<std::size_t, ProblemRank>& tile_coordinates,
  const InputRangeTuple& input_ranges,
  Function& function
) -> void {
  constexpr auto order = ordered_native_reduction_axes<ReductionAxes, InputTuple>();
  if constexpr (Depth == ReductionAxes::size) {
    const auto input_tiles = bind_projected_native_input_tiles_from_ranges(
      input_ranges,
      tile_coordinates,
      std::make_index_sequence<std::tuple_size_v<InputRangeTuple>>()
    );
    function(input_tiles);
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_counts[axis]; ++coordinate) {
      tile_coordinates[axis] = coordinate;
      for_each_bound_native_reduction_tile_in_ranges<Depth + 1, ReductionAxes, ProblemRank, InputTuple>(
        tile_counts,
        tile_coordinates,
        input_ranges,
        function
      );
    }
  }
}

template <
  std::size_t Depth,
  class ReductionAxes,
  std::size_t ProblemRank,
  class InputTuple,
  class InputRangeTuple,
  class Function>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto for_each_native_reduction_tile_in_ranges(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& tile_coordinates,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const InputRangeTuple& input_ranges,
  Function& function
) -> void {
  auto visit_bound_tile = [&](const auto& input_tiles) {
    for_each_native_reduction_element<0, ReductionAxes, InputTuple>(
      tile_extents,
      local_coordinates,
      input_tiles,
      function
    );
  };
  for_each_bound_native_reduction_tile_in_ranges<Depth, ReductionAxes, ProblemRank, InputTuple>(
    tile_counts,
    tile_coordinates,
    input_ranges,
    visit_bound_tile
  );
}

template <std::size_t Depth, class Output, std::size_t ProblemRank, class Function>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto for_each_native_output_element(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  Function& function
) -> void {
  constexpr auto order = native_output_inner_order<Output>();
  if constexpr (Depth == Output::axes_type::size) {
    function(local_coordinates);
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_extents[axis]; ++coordinate) {
      local_coordinates[axis] = coordinate;
      for_each_native_output_element<Depth + 1, Output>(tile_extents, local_coordinates, function);
    }
  }
}

template <class Output, class OutputTile, std::size_t ProblemRank>
IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE auto initialize_native_accumulator(
  const Output& output,
  const OutputTile& output_tile,
  const std::array<std::size_t, ProblemRank>& local_coordinates
) -> typename Output::value_type {
  if constexpr (Output::is_inout) {
    return static_cast<typename Output::value_type>(output_tile.at(local_coordinates));
  } else {
    return static_cast<typename Output::value_type>(output.identity());
  }
}

template <class ReductionAxes, std::size_t ProblemRank, class Output, class InputTuple, class Updater>
auto execute_scalar_native_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  const Output& output,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  auto tile_coordinates  = output_tile_coordinates;
  auto local_coordinates = std::array<std::size_t, ProblemRank>{};
  auto output_tile       = bind_projected_native_tile<ProblemRank>(output, output_tile_coordinates);

  auto output_element = [&](const auto& output_local_coordinates) {
    local_coordinates = output_local_coordinates;
    auto accumulator  = initialize_native_accumulator(output, output_tile, local_coordinates);
    auto update       = [&](const auto& input_tiles, const auto& reduction_local_coordinates) {
      invoke_updater(
        updater,
        accumulator,
        input_tiles,
        reduction_local_coordinates,
        std::make_index_sequence<std::tuple_size_v<InputTuple>>()
      );
    };
    for_each_native_reduction_tile<0, ReductionAxes>(
      tile_counts,
      tile_extents,
      tile_coordinates,
      local_coordinates,
      inputs,
      update
    );
    output_tile.at(local_coordinates) = accumulator;
  };
  for_each_native_output_element<0, Output>(tile_extents, local_coordinates, output_element);
}

template <std::size_t ProblemRank, class Output>
auto decode_native_output_tile(std::size_t work_item, const std::array<std::size_t, ProblemRank>& tile_counts)
  -> std::array<std::size_t, ProblemRank> {
  std::array<std::size_t, ProblemRank> coordinates{};
  if constexpr (std::is_same_v<typename native_mapping_traits<Output>::outer_layout_type, std::layout_right>) {
    for (std::size_t position = Output::axes_type::size; position > 0; --position) {
      const std::size_t axis = Output::axes_type::values[position - 1];
      coordinates[axis]      = work_item % tile_counts[axis];
      work_item /= tile_counts[axis];
    }
  } else {
    for (const std::size_t axis : Output::axes_type::values) {
      coordinates[axis] = work_item % tile_counts[axis];
      work_item /= tile_counts[axis];
    }
  }
  return coordinates;
}

template <class ReductionAxes, class Output, class InputTuple, class Updater, class TileExecutor>
auto host_native_tiled_reduce(
  const Output& output,
  const InputTuple& inputs,
  Updater& updater,
  TileExecutor& execute_tile
) -> void {
  using all_bindings_type           = decltype(std::tuple_cat(std::tuple{output}, inputs));
  constexpr std::size_t input_count = std::tuple_size_v<InputTuple>;
  constexpr std::size_t problem_rank_value
    = problem_rank<ReductionAxes, all_bindings_type>(std::make_index_sequence<input_count + 1>());
  constexpr auto tile_extents
    = common_native_tile_extents<problem_rank_value, all_bindings_type>(std::make_index_sequence<input_count + 1>());

  const auto all_bindings = std::tuple_cat(std::tuple{output}, inputs);
  const auto extents
    = merge_dynamic_extents<problem_rank_value>(all_bindings, std::make_index_sequence<input_count + 1>());
  assert_no_aliases(output, inputs, std::make_index_sequence<input_count>());

  std::array<std::size_t, problem_rank_value> tile_counts{};
  for (std::size_t axis = 0; axis < problem_rank_value; ++axis) {
    assert(extents[axis] % tile_extents[axis] == 0);
    tile_counts[axis] = extents[axis] / tile_extents[axis];
  }

  std::size_t output_tile_count = 1;
  for (const std::size_t axis : Output::axes_type::values) {
    output_tile_count *= tile_counts[axis];
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t work_item = 0; work_item < output_tile_count; ++work_item) {
    const auto output_tile_coordinates = decode_native_output_tile<problem_rank_value, Output>(work_item, tile_counts);
    execute_tile.template
    operator()<problem_rank_value>(tile_counts, tile_extents, output_tile_coordinates, output, inputs, updater);
  }
}

template <class ReductionAxes, class Output, class InputTuple, class Updater>
auto host_scalar_native_tiled_reduce(const Output& output, const InputTuple& inputs, Updater& updater) -> void {
  auto execute_tile = []<std::size_t ProblemRank>(
                        const std::array<std::size_t, ProblemRank>& tile_counts,
                        const std::array<std::size_t, ProblemRank>& tile_extents,
                        const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
                        const auto& selected_output,
                        const auto& selected_inputs,
                        auto& selected_updater
                      ) {
    execute_scalar_native_tile<ReductionAxes>(
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

template <class ReductionAxes, class ArgumentTuple, std::size_t... InputIndexes>
void dispatch_scalar_native_tiled_reduce(ArgumentTuple& arguments, std::index_sequence<InputIndexes...>) {
  constexpr std::size_t argument_count = std::tuple_size_v<std::remove_reference_t<ArgumentTuple>>;
  auto inputs                          = std::tuple{std::get<InputIndexes + 1>(arguments)...};
  auto&& updater                       = std::get<argument_count - 1>(arguments);
  host_scalar_native_tiled_reduce<ReductionAxes>(std::get<0>(arguments), inputs, updater);
}

} // namespace iai::detail

#undef IAI_NATIVE_TILED_REDUCE_ALWAYS_INLINE

#endif // INCLUDED_IAI_DETAIL_NATIVE_TILED_REDUCE
