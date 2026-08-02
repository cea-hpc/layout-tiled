// include/layout_tiled/reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_REDUCE
#define INCLUDED_LAYOUT_TILED_REDUCE

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "detail/reduce.hpp"

namespace layout_tiled {

/// Bind a read-only tiled mdspan whose dimensions project onto `Axes...`.
template <std::size_t... Axes, class View>
  requires detail::valid_binding<View, detail::axis_list<Axes...>>
[[nodiscard]] constexpr auto in(View view) {
  return detail::input_binding<View, detail::axis_list<Axes...>>(view);
}

/// Bind a tiled output mdspan and an identity used to initialize each result.
template <std::size_t... Axes, class View, class Identity>
  requires detail::valid_output_binding<View, Identity, detail::axis_list<Axes...>>
[[nodiscard]] constexpr auto out(View view, Identity&& identity) {
  return detail::output_binding<View, detail::axis_list<Axes...>, std::decay_t<Identity>, false>(
    view,
    std::forward<Identity>(identity)
  );
}

/// Bind a tiled output mdspan whose current elements initialize the results.
template <std::size_t... Axes, class View>
  requires detail::valid_inout_binding<View, detail::axis_list<Axes...>>
[[nodiscard]] constexpr auto inout(View view) {
  return detail::output_binding<View, detail::axis_list<Axes...>, detail::no_identity, true>(view);
}

/// Apply a projected N-D reduction. The first argument is the sole output,
/// followed by one or more inputs and the updater callable as the final argument.
template <std::size_t... ReductionAxes, class... Arguments>
  requires (detail::valid_reduce_call<detail::axis_list<ReductionAxes...>, Arguments...>())
void reduce(Arguments&&... arguments) {
  auto argument_tuple = std::forward_as_tuple(std::forward<Arguments>(arguments)...);
  detail::dispatch_reduce<detail::axis_list<ReductionAxes...>>(
    argument_tuple,
    std::make_index_sequence<sizeof...(Arguments) - 2>()
  );
}

} // namespace layout_tiled

#endif // INCLUDED_LAYOUT_TILED_REDUCE
