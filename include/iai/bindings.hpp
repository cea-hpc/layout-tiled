// include/iai/bindings.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_BINDINGS
#define INCLUDED_IAI_BINDINGS

#include <cstddef>
#include <type_traits>
#include <utility>

#include "detail/projected_view.hpp"

namespace iai {

/// Lazy view of a tiled input whose dimensions project onto `Axes...`.
template <class View, std::size_t... Axes>
using projected_input_view = detail::projected_input_view<View, detail::axis_list<Axes...>>;

/// Lazy view of a tiled output and the identity used to initialize its elements.
template <class View, class Identity, std::size_t... Axes>
using projected_output_view = detail::projected_output_view<View, detail::axis_list<Axes...>, Identity>;

/// Lazy view of a tiled output whose current elements initialize the reduction.
template <class View, std::size_t... Axes>
using projected_inout_view = detail::projected_inout_view<View, detail::axis_list<Axes...>>;

/// Lazily bind a read-only tiled mdspan whose dimensions project onto `Axes...`.
/// Constructing the binding does not access elements or enumerate execution work.
template <std::size_t... Axes, class View>
  requires detail::valid_binding<View, detail::axis_list<Axes...>>
[[nodiscard]] constexpr auto in(View view) {
  return projected_input_view<View, Axes...>(view);
}

/// Lazily bind a tiled output and an identity used to initialize each result.
/// Constructing the binding does not access elements or enumerate execution work.
template <std::size_t... Axes, class View, class Identity>
  requires detail::valid_output_binding<View, Identity, detail::axis_list<Axes...>>
[[nodiscard]] constexpr auto out(View view, Identity&& identity) {
  return projected_output_view<View, std::decay_t<Identity>, Axes...>(view, std::forward<Identity>(identity));
}

/// Lazily bind a tiled output whose current elements initialize each result.
/// Constructing the binding does not access elements or enumerate execution work.
template <std::size_t... Axes, class View>
  requires detail::valid_inout_binding<View, detail::axis_list<Axes...>>
[[nodiscard]] constexpr auto inout(View view) {
  return projected_inout_view<View, Axes...>(view);
}

} // namespace iai

#endif // INCLUDED_IAI_BINDINGS
