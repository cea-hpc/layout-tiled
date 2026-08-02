// include/layout_tiled/detail/reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_DETAIL_REDUCE
#define INCLUDED_LAYOUT_TILED_DETAIL_REDUCE

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tiled_view.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define LAYOUT_TILED_REDUCE_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define LAYOUT_TILED_REDUCE_ALWAYS_INLINE inline
#endif

namespace layout_tiled::detail {

template <std::size_t... Axes>
struct axis_list {
  static constexpr std::size_t size = sizeof...(Axes);
  static constexpr std::array<std::size_t, size> values{Axes...};

  template <std::size_t Axis>
  static constexpr bool contains = ((Axis == Axes) || ...);

  static consteval auto is_unique() -> bool {
    for (std::size_t lhs = 0; lhs < size; ++lhs) {
      for (std::size_t rhs = lhs + 1; rhs < size; ++rhs) {
        if (values[lhs] == values[rhs]) {
          return false;
        }
      }
    }
    return true;
  }
};

template <class View, class Axes>
concept valid_binding = tiled_mdspan<std::remove_cvref_t<View>> && (std::remove_cvref_t<View>::rank() == Axes::size)
                        && (Axes::size > 0) && Axes::is_unique();

template <class View, class Identity, class Axes>
concept valid_output_binding
  = valid_binding<View, Axes> && (!std::is_const_v<typename std::remove_cvref_t<View>::element_type>)
    && std::is_constructible_v<typename std::remove_cvref_t<View>::value_type, Identity>
    && std::
      is_assignable_v<typename std::remove_cvref_t<View>::reference, typename std::remove_cvref_t<View>::value_type>;

template <class View, class Axes>
concept valid_inout_binding
  = valid_binding<View, Axes> && (!std::is_const_v<typename std::remove_cvref_t<View>::element_type>)
    && std::is_copy_constructible_v<typename std::remove_cvref_t<View>::value_type>
    && std::
      is_assignable_v<typename std::remove_cvref_t<View>::reference, typename std::remove_cvref_t<View>::value_type>;

struct no_identity {};

template <class View, class Axes>
class input_binding {
public:
  using view_type  = View;
  using axes_type  = Axes;
  using value_type = typename view_type::value_type;
  using tiled_type = basic_tiled_view<view_type>;

  static constexpr bool is_input  = true;
  static constexpr bool is_output = false;

  explicit constexpr input_binding(view_type view) : tiled_(view) {}

  [[nodiscard]] constexpr auto view() const noexcept -> const view_type& {
    return tiled_.source();
  }

  [[nodiscard]] constexpr auto tiled() const noexcept -> const tiled_type& {
    return tiled_;
  }

private:
  tiled_type tiled_;
};

template <class View, class Axes, class Identity, bool IsInout>
class output_binding {
public:
  using view_type     = View;
  using axes_type     = Axes;
  using value_type    = typename view_type::value_type;
  using identity_type = Identity;
  using tiled_type    = basic_tiled_view<view_type>;

  static constexpr bool is_input  = false;
  static constexpr bool is_output = true;
  static constexpr bool is_inout  = IsInout;

  explicit constexpr output_binding(view_type view)
    requires IsInout
      : tiled_(view) {}

  constexpr output_binding(view_type view, identity_type identity)
    requires (!IsInout)
      : tiled_(view), identity_(std::move(identity)) {}

  [[nodiscard]] constexpr auto view() const noexcept -> const view_type& {
    return tiled_.source();
  }

  [[nodiscard]] constexpr auto tiled() const noexcept -> const tiled_type& {
    return tiled_;
  }

  [[nodiscard]] constexpr auto identity() const noexcept -> const identity_type&
    requires (!IsInout)
  {
    return identity_;
  }

private:
  tiled_type tiled_;
  [[no_unique_address]] identity_type identity_{};
};

template <class>
inline constexpr bool is_binding_v = false;

template <class View, class Axes>
inline constexpr bool is_binding_v<input_binding<View, Axes>> = true;

template <class View, class Axes, class Identity, bool IsInout>
inline constexpr bool is_binding_v<output_binding<View, Axes, Identity, IsInout>> = true;

template <class Type>
inline constexpr bool is_input_binding_v
  = is_binding_v<std::remove_cvref_t<Type>> && std::remove_cvref_t<Type>::is_input;

template <class Type>
inline constexpr bool is_output_binding_v
  = is_binding_v<std::remove_cvref_t<Type>> && std::remove_cvref_t<Type>::is_output;

template <class Binding>
consteval auto maximum_axis() -> std::size_t {
  std::size_t result = 0;
  for (const std::size_t axis : Binding::axes_type::values) {
    result = std::max(result, axis);
  }
  return result;
}

template <class ReductionAxes, class Tuple, std::size_t... BindingIndexes>
consteval auto problem_rank(std::index_sequence<BindingIndexes...>) -> std::size_t {
  std::size_t maximum = 0;
  for (const std::size_t axis : ReductionAxes::values) {
    maximum = std::max(maximum, axis);
  }
  ((maximum = std::max(maximum, maximum_axis<std::tuple_element_t<BindingIndexes, Tuple>>())), ...);
  return maximum + 1;
}

template <class Binding>
consteval auto binding_contains_axis(std::size_t axis) -> bool {
  for (const std::size_t candidate : Binding::axes_type::values) {
    if (candidate == axis) {
      return true;
    }
  }
  return false;
}

template <class Axes>
consteval auto axes_contain(const std::size_t axis) -> bool {
  for (const std::size_t candidate : Axes::values) {
    if (candidate == axis) {
      return true;
    }
  }
  return false;
}

template <class Binding>
consteval auto dimension_for_axis(const std::size_t axis) -> std::size_t {
  for (std::size_t dimension = 0; dimension < Binding::axes_type::size; ++dimension) {
    if (Binding::axes_type::values[dimension] == axis) {
      return dimension;
    }
  }
  return Binding::axes_type::size;
}

template <class Binding>
consteval auto static_extent_for_axis(const std::size_t axis) -> std::size_t {
  const std::size_t dimension = dimension_for_axis<Binding>(axis);
  if (dimension == Binding::axes_type::size) {
    return std::dynamic_extent;
  }
  return Binding::view_type::static_extent(dimension);
}

template <class Binding>
consteval auto tile_extent_for_axis(const std::size_t axis) -> std::size_t {
  const std::size_t dimension = dimension_for_axis<Binding>(axis);
  if (dimension == Binding::axes_type::size) {
    return 0;
  }
  return static_cast<std::size_t>(Binding::view_type::mapping_type::textent(dimension));
}

template <class Tuple, std::size_t... BindingIndexes>
consteval auto all_axes_are_dense(const std::size_t rank, std::index_sequence<BindingIndexes...>) -> bool {
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const bool present = (binding_contains_axis<std::tuple_element_t<BindingIndexes, Tuple>>(axis) || ...);
    if (!present) {
      return false;
    }
  }
  return true;
}

template <class Tuple, std::size_t... BindingIndexes>
consteval auto static_extents_are_compatible(const std::size_t rank, std::index_sequence<BindingIndexes...>) -> bool {
  for (std::size_t axis = 0; axis < rank; ++axis) {
    std::size_t known = std::dynamic_extent;
    bool compatible   = true;
    const auto merge  = [&](const std::size_t candidate) {
      if (candidate == std::dynamic_extent) {
        return;
      }
      if (known != std::dynamic_extent && known != candidate) {
        compatible = false;
      }
      known = candidate;
    };
    (merge(static_extent_for_axis<std::tuple_element_t<BindingIndexes, Tuple>>(axis)), ...);
    if (!compatible) {
      return false;
    }
  }
  return true;
}

template <class Tuple, std::size_t... BindingIndexes>
consteval auto tile_extents_are_compatible(const std::size_t rank, std::index_sequence<BindingIndexes...>) -> bool {
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
    (merge(tile_extent_for_axis<std::tuple_element_t<BindingIndexes, Tuple>>(axis)), ...);
    if (!compatible || known == 0) {
      return false;
    }
  }
  return true;
}

template <class ReductionAxes, class Tuple, std::size_t... InputIndexes>
consteval auto updater_is_valid(std::index_sequence<InputIndexes...>) -> bool {
  using output_type   = std::tuple_element_t<0, Tuple>;
  using callable_type = std::tuple_element_t<std::tuple_size_v<Tuple> - 1, Tuple>;
  if constexpr (!std::is_invocable_v<
                  callable_type&,
                  typename output_type::value_type&,
                  const typename std::tuple_element_t<InputIndexes + 1, Tuple>::value_type&...>) {
    return false;
  } else {
    return std::is_void_v<std::invoke_result_t<
      callable_type&,
      typename output_type::value_type&,
      const typename std::tuple_element_t<InputIndexes + 1, Tuple>::value_type&...>>;
  }
}

template <class ReductionAxes, class... Arguments>
consteval auto valid_reduce_call() -> bool {
  if constexpr (sizeof...(Arguments) < 3 || ReductionAxes::size == 0 || !ReductionAxes::is_unique()) {
    return false;
  } else {
    using tuple_type                     = std::tuple<std::remove_cvref_t<Arguments>...>;
    constexpr std::size_t argument_count = sizeof...(Arguments);
    constexpr std::size_t binding_count  = argument_count - 1;

    if constexpr (!is_output_binding_v<std::tuple_element_t<0, tuple_type>>) {
      return false;
    } else if constexpr (![]<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
                           return (is_input_binding_v<std::tuple_element_t<Indexes + 1, tuple_type>> && ...);
                         }(std::make_index_sequence<binding_count - 1>())) {
      return false;
    } else if constexpr (is_binding_v<std::tuple_element_t<argument_count - 1, tuple_type>>) {
      return false;
    } else {
      using binding_indexes      = std::make_index_sequence<binding_count>;
      using input_indexes        = std::make_index_sequence<binding_count - 1>;
      using output_type          = std::tuple_element_t<0, tuple_type>;
      constexpr std::size_t rank = problem_rank<ReductionAxes, tuple_type>(binding_indexes{});

      if constexpr (output_type::axes_type::size == 0) {
        return false;
      }
      for (const std::size_t axis : output_type::axes_type::values) {
        if (axes_contain<ReductionAxes>(axis)) {
          return false;
        }
      }
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const bool reduction = axes_contain<ReductionAxes>(axis);
        const bool output    = binding_contains_axis<output_type>(axis);
        if (output == reduction) {
          return false;
        }
        if (reduction) {
          bool found_in_input = false;
          [&]<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
            ((found_in_input
              = found_in_input || binding_contains_axis<std::tuple_element_t<Indexes + 1, tuple_type>>(axis)),
             ...);
          }(input_indexes{});
          if (!found_in_input) {
            return false;
          }
        }
      }
      return all_axes_are_dense<tuple_type>(rank, binding_indexes{})
             && static_extents_are_compatible<tuple_type>(rank, binding_indexes{})
             && tile_extents_are_compatible<tuple_type>(rank, binding_indexes{})
             && updater_is_valid<ReductionAxes, tuple_type>(input_indexes{});
    }
  }
}

template <class Binding, std::size_t ProblemRank>
class projected_tile {
public:
  using binding_type = Binding;
  using tile_type    = typename binding_type::tiled_type::tile_type;
  using index_type   = typename binding_type::view_type::index_type;

  explicit constexpr projected_tile(tile_type tile) noexcept : tile_(tile) {}

  [[nodiscard]] LAYOUT_TILED_REDUCE_ALWAYS_INLINE decltype(auto)
  at(const std::array<std::size_t, ProblemRank>& local_coordinates) const noexcept {
    std::array<index_type, binding_type::axes_type::size> coordinates{};
    for (std::size_t dimension = 0; dimension < binding_type::axes_type::size; ++dimension) {
      coordinates[dimension] = static_cast<index_type>(local_coordinates[binding_type::axes_type::values[dimension]]);
    }
    return tile_[coordinates];
  }

private:
  tile_type tile_;
};

template <std::size_t ProblemRank, class Binding>
[[nodiscard]] LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto bind_projected_tile(
  const Binding& binding,
  const std::array<std::size_t, ProblemRank>& problem_tile_coordinates
) noexcept -> projected_tile<Binding, ProblemRank> {
  std::array<std::size_t, Binding::axes_type::size> coordinates{};
  for (std::size_t dimension = 0; dimension < Binding::axes_type::size; ++dimension) {
    coordinates[dimension] = problem_tile_coordinates[Binding::axes_type::values[dimension]];
  }
  return projected_tile<Binding, ProblemRank>(binding.tiled().tile(coordinates));
}

template <class BindingTuple, std::size_t... BindingIndexes>
consteval auto compile_time_problem_rank(std::index_sequence<BindingIndexes...>) -> std::size_t {
  std::size_t maximum = 0;
  ((maximum = std::max(maximum, maximum_axis<std::tuple_element_t<BindingIndexes, BindingTuple>>())), ...);
  return maximum + 1;
}

template <std::size_t ProblemRank, class BindingTuple, std::size_t... BindingIndexes>
constexpr auto common_tile_extents(std::index_sequence<BindingIndexes...>) -> std::array<std::size_t, ProblemRank> {
  std::array<std::size_t, ProblemRank> result{};
  for (std::size_t axis = 0; axis < ProblemRank; ++axis) {
    ((result[axis]
      = std::max(result[axis], tile_extent_for_axis<std::tuple_element_t<BindingIndexes, BindingTuple>>(axis))),
     ...);
  }
  return result;
}

template <std::size_t ProblemRank, class BindingTuple, std::size_t... BindingIndexes>
auto validate_dynamic_extents(const BindingTuple& bindings, std::index_sequence<BindingIndexes...>) noexcept
  -> std::array<std::size_t, ProblemRank> {
  std::array<std::size_t, ProblemRank> extents{};
  const auto merge_binding = [&]<class Binding>(const Binding& binding) {
    for (std::size_t dimension = 0; dimension < Binding::axes_type::size; ++dimension) {
      const std::size_t axis   = Binding::axes_type::values[dimension];
      const std::size_t extent = static_cast<std::size_t>(binding.view().extent(dimension));
      assert((extents[axis] == 0 || extents[axis] == extent) && "layout_tiled::reduce: incompatible logical extents");
      extents[axis] = extent;
    }
  };
  (merge_binding(std::get<BindingIndexes>(bindings)), ...);
  return extents;
}

template <class Output, class Input>
auto assert_no_overlap(const Output& output, const Input& input) noexcept -> void {
#ifndef NDEBUG
  using output_handle = typename Output::view_type::data_handle_type;
  using input_handle  = typename Input::view_type::data_handle_type;
  if constexpr (std::is_pointer_v<output_handle> && std::is_pointer_v<input_handle>) {
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output.view().data_handle());
    const auto input_begin  = reinterpret_cast<std::uintptr_t>(input.view().data_handle());
    const auto output_end
      = output_begin + output.view().mapping().required_span_size() * sizeof(typename Output::value_type);
    const auto input_end
      = input_begin + input.view().mapping().required_span_size() * sizeof(typename Input::value_type);
    assert((output_end <= input_begin || input_end <= output_begin) && "layout_tiled::reduce: output overlaps input");
  }
#else
  (void)output;
  (void)input;
#endif
}

template <class Output, class InputTuple, std::size_t... InputIndexes>
auto assert_no_aliases(const Output& output, const InputTuple& inputs, std::index_sequence<InputIndexes...>) noexcept
  -> void {
  (assert_no_overlap(output, std::get<InputIndexes>(inputs)), ...);
}

template <class Binding>
consteval auto inner_stride_for_axis(const std::size_t axis) -> std::size_t {
  const std::size_t dimension = dimension_for_axis<Binding>(axis);
  if (dimension == Binding::axes_type::size) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t stride = 1;
  if constexpr (std::is_same_v<typename Binding::view_type::mapping_type::inner_layout_type, std::layout_right>) {
    for (std::size_t current = dimension + 1; current < Binding::axes_type::size; ++current) {
      stride *= static_cast<std::size_t>(Binding::view_type::mapping_type::textent(current));
    }
  } else {
    for (std::size_t current = 0; current < dimension; ++current) {
      stride *= static_cast<std::size_t>(Binding::view_type::mapping_type::textent(current));
    }
  }
  return stride;
}

template <class InputTuple, std::size_t... InputIndexes>
consteval auto minimum_input_stride(const std::size_t axis, std::index_sequence<InputIndexes...>) -> std::size_t {
  std::size_t result = std::numeric_limits<std::size_t>::max();
  ((result = std::min(result, inner_stride_for_axis<std::tuple_element_t<InputIndexes, InputTuple>>(axis))), ...);
  return result;
}

template <class ReductionAxes, class InputTuple>
consteval auto ordered_reduction_axes() -> std::array<std::size_t, ReductionAxes::size> {
  auto result   = ReductionAxes::values;
  using indexes = std::make_index_sequence<std::tuple_size_v<InputTuple>>;
  for (std::size_t left = 0; left < result.size(); ++left) {
    for (std::size_t right = left + 1; right < result.size(); ++right) {
      if (minimum_input_stride<InputTuple>(result[right], indexes{})
          > minimum_input_stride<InputTuple>(result[left], indexes{})) {
        std::swap(result[left], result[right]);
      }
    }
  }
  return result;
}

template <class Output>
consteval auto output_inner_order() -> std::array<std::size_t, Output::axes_type::size> {
  auto result = Output::axes_type::values;
  if constexpr (std::is_same_v<typename Output::view_type::mapping_type::inner_layout_type, std::layout_left>) {
    std::reverse(result.begin(), result.end());
  }
  return result;
}

template <class Output>
consteval auto vector_axis() -> std::size_t {
  if constexpr (std::is_same_v<typename Output::view_type::mapping_type::inner_layout_type, std::layout_right>) {
    return Output::axes_type::values.back();
  } else {
    return Output::axes_type::values.front();
  }
}

template <class Output>
consteval auto row_axis() -> std::size_t {
  if constexpr (Output::axes_type::size == 1) {
    return vector_axis<Output>();
  } else if constexpr (std::
                         is_same_v<typename Output::view_type::mapping_type::inner_layout_type, std::layout_right>) {
    return Output::axes_type::values[Output::axes_type::size - 2];
  } else {
    return Output::axes_type::values[1];
  }
}

template <class Input, std::size_t VectorAxis>
consteval auto input_is_contiguous_or_invariant() -> bool {
  constexpr std::size_t dimension = dimension_for_axis<Input>(VectorAxis);
  if constexpr (dimension == Input::axes_type::size) {
    return true;
  } else if constexpr (std::is_same_v<typename Input::view_type::mapping_type::inner_layout_type, std::layout_right>) {
    return dimension + 1 == Input::axes_type::size;
  } else if constexpr (std::is_same_v<typename Input::view_type::mapping_type::inner_layout_type, std::layout_left>) {
    return dimension == 0;
  } else {
    return false;
  }
}

template <class Output, class InputTuple, std::size_t... InputIndexes>
consteval auto fragment_compatible_impl(std::index_sequence<InputIndexes...>) -> bool {
  constexpr std::size_t simd_axis = vector_axis<Output>();
  return std::is_arithmetic_v<typename Output::value_type>
         && (std::is_arithmetic_v<typename std::tuple_element_t<InputIndexes, InputTuple>::value_type> && ...)
         && (input_is_contiguous_or_invariant<std::tuple_element_t<InputIndexes, InputTuple>, simd_axis>() && ...);
}

template <class Output, class InputTuple>
inline constexpr bool fragment_compatible_v
  = fragment_compatible_impl<Output, InputTuple>(std::make_index_sequence<std::tuple_size_v<InputTuple>>());

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

template <std::size_t ProblemRank, class InputTuple, std::size_t... InputIndexes>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto bind_input_tiles(
  const InputTuple& inputs,
  const std::array<std::size_t, ProblemRank>& tile_coordinates,
  std::index_sequence<InputIndexes...>
) noexcept {
  return std::tuple{bind_projected_tile<ProblemRank>(std::get<InputIndexes>(inputs), tile_coordinates)...};
}

template <class Updater, class Accumulator, class InputTiles, std::size_t ProblemRank, std::size_t... InputIndexes>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto invoke_updater(
  Updater& updater,
  Accumulator& accumulator,
  const InputTiles& input_tiles,
  const std::array<std::size_t, ProblemRank>& local_coordinates,
  std::index_sequence<InputIndexes...>
) -> void {
  std::invoke(updater, accumulator, std::as_const(std::get<InputIndexes>(input_tiles).at(local_coordinates))...);
}

// `InputTuple` and a tuple of projected input tiles have corresponding binding
// types, but different wrappers. Keep the reduction order supplied explicitly.
template <
  std::size_t Depth,
  class ReductionAxes,
  class InputTuple,
  std::size_t ProblemRank,
  class InputTiles,
  class Function>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto for_each_reduction_element_ordered(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const InputTiles& input_tiles,
  Function& function
) -> void {
  constexpr auto order = ordered_reduction_axes<ReductionAxes, InputTuple>();
  if constexpr (Depth == ReductionAxes::size) {
    function(input_tiles, local_coordinates);
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_extents[axis]; ++coordinate) {
      local_coordinates[axis] = coordinate;
      for_each_reduction_element_ordered<Depth + 1, ReductionAxes, InputTuple>(
        tile_extents,
        local_coordinates,
        input_tiles,
        function
      );
    }
  }
}

template <std::size_t Depth, class ReductionAxes, std::size_t ProblemRank, class InputTuple, class Function>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto for_each_reduction_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& tile_coordinates,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const InputTuple& inputs,
  Function& function
) -> void {
  constexpr auto order = ordered_reduction_axes<ReductionAxes, InputTuple>();
  if constexpr (Depth == ReductionAxes::size) {
    const auto input_tiles
      = bind_input_tiles(inputs, tile_coordinates, std::make_index_sequence<std::tuple_size_v<InputTuple>>());
    for_each_reduction_element_ordered<0, ReductionAxes, InputTuple>(
      tile_extents,
      local_coordinates,
      input_tiles,
      function
    );
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_counts[axis]; ++coordinate) {
      tile_coordinates[axis] = coordinate;
      for_each_reduction_tile<Depth + 1, ReductionAxes>(
        tile_counts,
        tile_extents,
        tile_coordinates,
        local_coordinates,
        inputs,
        function
      );
    }
  }
}

template <std::size_t Depth, class Output, std::size_t ProblemRank, class Function>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto for_each_output_element(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  Function& function
) -> void {
  constexpr auto order = output_inner_order<Output>();
  if constexpr (Depth == Output::axes_type::size) {
    function(local_coordinates);
  } else {
    constexpr std::size_t axis = order[Depth];
    for (std::size_t coordinate = 0; coordinate < tile_extents[axis]; ++coordinate) {
      local_coordinates[axis] = coordinate;
      for_each_output_element<Depth + 1, Output>(tile_extents, local_coordinates, function);
    }
  }
}

template <class Output, class OutputTile, std::size_t ProblemRank>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto initialize_accumulator(
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
auto execute_scalar_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  const Output& output,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  auto tile_coordinates  = output_tile_coordinates;
  auto local_coordinates = std::array<std::size_t, ProblemRank>{};
  auto output_tile       = bind_projected_tile<ProblemRank>(output, output_tile_coordinates);

  auto output_element = [&](const auto& output_local_coordinates) {
    local_coordinates = output_local_coordinates;
    auto accumulator  = initialize_accumulator(output, output_tile, local_coordinates);
    auto update       = [&](const auto& input_tiles, const auto& reduction_local_coordinates) {
      invoke_updater(
        updater,
        accumulator,
        input_tiles,
        reduction_local_coordinates,
        std::make_index_sequence<std::tuple_size_v<InputTuple>>()
      );
    };
    for_each_reduction_tile<0, ReductionAxes>(
      tile_counts,
      tile_extents,
      tile_coordinates,
      local_coordinates,
      inputs,
      update
    );
    output_tile.at(local_coordinates) = accumulator;
  };
  for_each_output_element<0, Output>(tile_extents, local_coordinates, output_element);
}

template <std::size_t Depth, class Output, std::size_t ProblemRank, class Function>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto for_each_fragment_plane(
  const std::array<std::size_t, ProblemRank>& tile_extents,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  Function& function
) -> void {
  constexpr auto order           = output_inner_order<Output>();
  constexpr std::size_t simd     = vector_axis<Output>();
  constexpr std::size_t unrolled = row_axis<Output>();
  if constexpr (Depth == Output::axes_type::size) {
    function(local_coordinates);
  } else {
    constexpr std::size_t axis = order[Depth];
    if constexpr (axis == simd || axis == unrolled) {
      for_each_fragment_plane<Depth + 1, Output>(tile_extents, local_coordinates, function);
    } else {
      for (std::size_t coordinate = 0; coordinate < tile_extents[axis]; ++coordinate) {
        local_coordinates[axis] = coordinate;
        for_each_fragment_plane<Depth + 1, Output>(tile_extents, local_coordinates, function);
      }
    }
  }
}

template <class Function, std::size_t... Indexes>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto static_for_impl(Function& function, std::index_sequence<Indexes...>) -> void {
  (function.template operator()<Indexes>(), ...);
}

template <std::size_t Count, class Function>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto static_for(Function&& function) -> void {
  static_for_impl(function, std::make_index_sequence<Count>());
}

template <
  std::size_t Rows,
  std::size_t Columns,
  class ReductionAxes,
  std::size_t ProblemRank,
  class Output,
  class OutputTile,
  class InputTuple,
  class Updater>
LAYOUT_TILED_REDUCE_ALWAYS_INLINE auto execute_full_fragment_block(
  const std::size_t row_begin,
  const std::size_t column_begin,
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  std::array<std::size_t, ProblemRank>& local_coordinates,
  const Output& output,
  const OutputTile& output_tile,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  using value_type                    = typename Output::value_type;
  constexpr std::size_t simd_axis     = vector_axis<Output>();
  constexpr std::size_t unrolled_axis = row_axis<Output>();
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
      fragment[Row][column]        = initialize_accumulator(output, output_tile, local_coordinates);
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
  for_each_reduction_tile<0, ReductionAxes>(
    tile_counts,
    tile_extents,
    tile_coordinates,
    local_coordinates,
    inputs,
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
auto execute_fragment_tile(
  const std::array<std::size_t, ProblemRank>& tile_counts,
  const std::array<std::size_t, ProblemRank>& tile_extents,
  const std::array<std::size_t, ProblemRank>& output_tile_coordinates,
  const Output& output,
  const InputTuple& inputs,
  Updater& updater
) -> void {
  using value_type                    = typename Output::value_type;
  constexpr std::size_t simd_axis     = vector_axis<Output>();
  constexpr std::size_t unrolled_axis = row_axis<Output>();
  constexpr std::size_t mr            = Output::axes_type::size > 1 ? 4 : 1;
  constexpr std::size_t nr            = 2 * native_simd_lanes<value_type>();

  auto output_tile       = bind_projected_tile<ProblemRank>(output, output_tile_coordinates);
  auto local_coordinates = std::array<std::size_t, ProblemRank>{};

  auto plane = [&](const auto& plane_coordinates) {
    local_coordinates            = plane_coordinates;
    const std::size_t row_extent = Output::axes_type::size > 1 ? tile_extents[unrolled_axis] : std::size_t{1};
    for (std::size_t row_begin = 0; row_begin < row_extent; row_begin += mr) {
      const std::size_t row_count = std::min(mr, row_extent - row_begin);
      for (std::size_t column_begin = 0; column_begin < tile_extents[simd_axis]; column_begin += nr) {
        const std::size_t column_count = std::min(nr, tile_extents[simd_axis] - column_begin);
        if (row_count == mr && column_count == nr) {
          execute_full_fragment_block<mr, nr, ReductionAxes>(
            row_begin,
            column_begin,
            tile_counts,
            tile_extents,
            output_tile_coordinates,
            local_coordinates,
            output,
            output_tile,
            inputs,
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
            fragment[row][column]        = initialize_accumulator(output, output_tile, local_coordinates);
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
        for_each_reduction_tile<0, ReductionAxes>(
          tile_counts,
          tile_extents,
          tile_coordinates,
          local_coordinates,
          inputs,
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
  for_each_fragment_plane<0, Output>(tile_extents, local_coordinates, plane);
}

template <std::size_t ProblemRank, class Output>
auto decode_output_tile(std::size_t work_item, const std::array<std::size_t, ProblemRank>& tile_counts)
  -> std::array<std::size_t, ProblemRank> {
  std::array<std::size_t, ProblemRank> coordinates{};
  if constexpr (std::is_same_v<typename Output::view_type::mapping_type::outer_layout_type, std::layout_right>) {
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

template <class ReductionAxes, class Output, class InputTuple, class Updater>
auto host_reduce(const Output& output, const InputTuple& inputs, Updater& updater) -> void {
  using all_bindings_type           = decltype(std::tuple_cat(std::tuple{output}, inputs));
  constexpr std::size_t input_count = std::tuple_size_v<InputTuple>;
  constexpr std::size_t problem_rank_value
    = compile_time_problem_rank<all_bindings_type>(std::make_index_sequence<input_count + 1>());
  constexpr auto tile_extents
    = common_tile_extents<problem_rank_value, all_bindings_type>(std::make_index_sequence<input_count + 1>());

  const auto all_bindings = std::tuple_cat(std::tuple{output}, inputs);
  const auto extents
    = validate_dynamic_extents<problem_rank_value>(all_bindings, std::make_index_sequence<input_count + 1>());
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
    const auto output_tile_coordinates = decode_output_tile<problem_rank_value, Output>(work_item, tile_counts);
    if constexpr (fragment_compatible_v<Output, InputTuple>) {
      execute_fragment_tile<ReductionAxes>(
        tile_counts,
        tile_extents,
        output_tile_coordinates,
        output,
        inputs,
        updater
      );
    } else {
      execute_scalar_tile<ReductionAxes>(tile_counts, tile_extents, output_tile_coordinates, output, inputs, updater);
    }
  }
}

template <class ReductionAxes, class ArgumentTuple, std::size_t... InputIndexes>
void dispatch_reduce(ArgumentTuple& arguments, std::index_sequence<InputIndexes...>) {
  constexpr std::size_t argument_count = std::tuple_size_v<std::remove_reference_t<ArgumentTuple>>;
  auto inputs                          = std::tuple{std::get<InputIndexes + 1>(arguments)...};
  auto&& updater                       = std::get<argument_count - 1>(arguments);
  host_reduce<ReductionAxes>(std::get<0>(arguments), inputs, updater);
}

} // namespace layout_tiled::detail

#undef LAYOUT_TILED_REDUCE_ALWAYS_INLINE

#endif // INCLUDED_LAYOUT_TILED_DETAIL_REDUCE
