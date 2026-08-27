// include/iai/detail/md_reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_MD_REDUCE
#define INCLUDED_IAI_DETAIL_MD_REDUCE

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "access_transform.hpp"
#include "projected_view.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define IAI_DETAIL_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define IAI_DETAIL_ALWAYS_INLINE inline
#endif

namespace iai::detail {

template <class Binding>
consteval auto static_extent_for_axis(const std::size_t axis) -> std::size_t {
  const std::size_t dimension = dimension_for_axis<Binding>(axis);
  if (dimension == Binding::axes_type::size) {
    return std::dynamic_extent;
  }
  return Binding::view_type::static_extent(dimension);
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

template <class Tuple, std::size_t... InputIndexes>
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
             && updater_is_valid<tuple_type>(input_indexes{});
    }
  }
}

template <std::size_t ProblemRank, class Binding, class Coordinate = typename Binding::view_type::index_type>
[[nodiscard]] constexpr auto
project_coordinates(const std::array<std::size_t, ProblemRank>& problem_coordinates) noexcept
  -> std::array<Coordinate, Binding::axes_type::size> {
  return named_axis_projection<ProblemRank, typename Binding::axes_type, Coordinate>{}(problem_coordinates);
}

template <std::size_t ProblemRank, class BindingTuple, std::size_t... BindingIndexes>
auto merge_dynamic_extents(const BindingTuple& bindings, std::index_sequence<BindingIndexes...>) noexcept
  -> std::array<std::size_t, ProblemRank> {
  std::array<std::size_t, ProblemRank> extents{};
  const auto merge_binding = [&]<class Binding>(const Binding& binding) {
    for (std::size_t dimension = 0; dimension < Binding::axes_type::size; ++dimension) {
      const std::size_t axis   = Binding::axes_type::values[dimension];
      const std::size_t extent = static_cast<std::size_t>(binding.view().extent(dimension));
      assert((extents[axis] == 0 || extents[axis] == extent) && "iai::md_reduce: incompatible logical extents");
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
    assert((output_end <= input_begin || input_end <= output_begin) && "iai::md_reduce: output overlaps input");
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

template <class Updater, class Accumulator, class ProjectedInputs, class Coordinates, std::size_t... InputIndexes>
IAI_DETAIL_ALWAYS_INLINE auto invoke_updater(
  Updater& updater,
  Accumulator& accumulator,
  const ProjectedInputs& inputs,
  const Coordinates& coordinates,
  std::index_sequence<InputIndexes...>
) -> void {
  std::invoke(updater, accumulator, std::as_const(std::get<InputIndexes>(inputs).at(coordinates))...);
}

} // namespace iai::detail

#undef IAI_DETAIL_ALWAYS_INLINE

#endif // INCLUDED_IAI_DETAIL_MD_REDUCE
