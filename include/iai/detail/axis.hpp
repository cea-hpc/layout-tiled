// include/iai/detail/axis.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_AXIS
#define INCLUDED_IAI_DETAIL_AXIS

#include <algorithm>
#include <array>
#include <cstddef>

namespace iai::detail {

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

template <class Axes>
constexpr auto axes_contain(const std::size_t axis) -> bool {
  for (const std::size_t candidate : Axes::values) {
    if (candidate == axis) {
      return true;
    }
  }
  return false;
}

template <class Binding>
consteval auto binding_contains_axis(const std::size_t axis) -> bool {
  return axes_contain<typename Binding::axes_type>(axis);
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
consteval auto maximum_axis() -> std::size_t {
  std::size_t result = 0;
  for (const std::size_t axis : Binding::axes_type::values) {
    result = std::max(result, axis);
  }
  return result;
}

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_AXIS
