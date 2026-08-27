// include/iai/md_reduce.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_MD_REDUCE
#define INCLUDED_IAI_MD_REDUCE

#include <cstddef>
#include <tuple>
#include <utility>

#include "bindings.hpp"
#include "detail/native_tiled_reduce_policy.hpp"

namespace iai {

/// Eagerly apply a projected N-D reduction.
///
/// The first argument is the sole output, followed by one or more inputs and
/// the updater callable as the final argument. The operation is complete when
/// this function returns.
template <std::size_t... ReductionAxes, class... Arguments>
  requires (detail::valid_native_tiled_reduce_call<detail::axis_list<ReductionAxes...>, Arguments...>())
void md_reduce(Arguments&&... arguments) {
  auto argument_tuple = std::forward_as_tuple(std::forward<Arguments>(arguments)...);
  detail::dispatch_native_tiled_reduce<detail::axis_list<ReductionAxes...>>(
    argument_tuple,
    std::make_index_sequence<sizeof...(Arguments) - 2>()
  );
}

} // namespace iai

#endif // INCLUDED_IAI_MD_REDUCE
