// include/iai/detail/native_tiled_reduce_policy.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_NATIVE_TILED_REDUCE_POLICY
#define INCLUDED_IAI_DETAIL_NATIVE_TILED_REDUCE_POLICY

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "native_tiled_blocked_fragment_reduce.hpp"

namespace iai::detail {

enum class native_tiled_reduction_path {
  scalar,
  complete_fragment,
  reduction_blocked_fragment
};

enum class native_tiled_reduction_override {
  automatic,
  force_scalar,
  force_complete_fragment,
  force_reduction_blocked_fragment
};

struct host_native_tiled_schedule_policy {
  // Portable conservative budgets. A backend may replace this policy with
  // target-specific values without changing reduction semantics.
  static constexpr std::size_t COMPLETE_REDUCTION_BUDGET_BYTES = 256 * 1024;
  static constexpr std::size_t CACHE_BLOCK_BUDGET_BYTES        = 32 * 1024;
};

template <std::size_t Rank>
struct native_tiled_reduction_schedule {
  native_tiled_reduction_path path{native_tiled_reduction_path::scalar};
  native_tile_block_shape<Rank> block_shape{};
  std::size_t complete_working_set_bytes{};
  std::size_t block_working_set_bytes{};
  std::size_t reduction_tile_count{1};
};

template <std::size_t Rank>
[[nodiscard]] constexpr auto clamped_native_block_extent(
  const std::array<std::size_t, Rank>& tile_counts,
  const std::size_t axis,
  const std::size_t requested
) -> std::size_t {
  return std::max<std::size_t>(1, std::min(tile_counts[axis], requested));
}

template <std::size_t Rank, class ReductionAxes>
[[nodiscard]] constexpr auto native_tiled_block_candidates(const std::array<std::size_t, Rank>& tile_counts)
  -> std::array<native_tile_block_shape<Rank>, 6> {
  constexpr std::array<std::array<std::size_t, 2>, 6> factors{{
    {1, 1},
    {2, 1},
    {1, 2},
    {2, 2},
    {4, 1},
    {2, 4},
  }};
  std::array<native_tile_block_shape<Rank>, factors.size()> result{};
  for (std::size_t candidate = 0; candidate < factors.size(); ++candidate) {
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      const std::size_t requested = axes_contain<ReductionAxes>(axis) ? factors[candidate][1] : factors[candidate][0];
      result[candidate].tile_extents[axis] = clamped_native_block_extent(tile_counts, axis, requested);
    }
  }
  return result;
}

template <std::size_t Rank, class ReductionAxes>
[[nodiscard]] constexpr auto complete_native_tiled_reduction_shape(const std::array<std::size_t, Rank>& tile_counts)
  -> native_tile_block_shape<Rank> {
  native_tile_block_shape<Rank> result{};
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    result.tile_extents[axis]
      = axes_contain<ReductionAxes>(axis) ? std::max<std::size_t>(1, tile_counts[axis]) : std::size_t{1};
  }
  return result;
}

template <
  class ReductionAxes,
  class Output,
  class InputTuple,
  native_tiled_reduction_override Override = native_tiled_reduction_override::automatic,
  class CachePolicy                        = host_native_tiled_schedule_policy,
  std::size_t Rank>
[[nodiscard]] constexpr auto
select_native_tiled_reduction_schedule_from_tile_counts(const std::array<std::size_t, Rank>& tile_counts)
  -> native_tiled_reduction_schedule<Rank> {
  native_tiled_reduction_schedule<Rank> result{};
  if constexpr (!native_complete_fragment_compatible_v<Output, InputTuple>) {
    return result;
  } else {
    using all_bindings_type = decltype(std::tuple_cat(std::tuple{std::declval<Output>()}, std::declval<InputTuple>()));
    const auto complete_shape         = complete_native_tiled_reduction_shape<Rank, ReductionAxes>(tile_counts);
    result.complete_working_set_bytes = native_tile_block_working_set_bytes<Rank, all_bindings_type>(complete_shape);
    for (const std::size_t axis : ReductionAxes::values) {
      result.reduction_tile_count *= tile_counts[axis];
    }

    const auto candidates = native_tiled_block_candidates<Rank, ReductionAxes>(tile_counts);
    const auto selected_block
      = select_native_tile_block_shape<Rank, all_bindings_type>(candidates, CachePolicy::CACHE_BLOCK_BUDGET_BYTES);
    if (selected_block.found) {
      result.block_shape             = selected_block.shape;
      result.block_working_set_bytes = selected_block.working_set_bytes;
    } else {
      result.block_shape.tile_extents.fill(1);
      result.block_working_set_bytes
        = native_tile_block_working_set_bytes<Rank, all_bindings_type>(result.block_shape);
    }

    if constexpr (Override == native_tiled_reduction_override::force_scalar) {
      result.path = native_tiled_reduction_path::scalar;
    } else if constexpr (Override == native_tiled_reduction_override::force_complete_fragment) {
      result.path = native_tiled_reduction_path::complete_fragment;
    } else if constexpr (Override == native_tiled_reduction_override::force_reduction_blocked_fragment) {
      result.path = native_tiled_reduction_path::reduction_blocked_fragment;
    } else if (result.reduction_tile_count <= 1
               || result.complete_working_set_bytes <= CachePolicy::COMPLETE_REDUCTION_BUDGET_BYTES) {
      result.path = native_tiled_reduction_path::complete_fragment;
    } else {
      result.path = native_tiled_reduction_path::reduction_blocked_fragment;
    }
    return result;
  }
}

template <
  class ReductionAxes,
  native_tiled_reduction_override Override = native_tiled_reduction_override::automatic,
  class CachePolicy                        = host_native_tiled_schedule_policy,
  class Output,
  class InputTuple>
[[nodiscard]] auto select_native_tiled_reduction_schedule(const Output& output, const InputTuple& inputs) {
  using all_bindings_type           = decltype(std::tuple_cat(std::tuple{output}, inputs));
  constexpr std::size_t input_count = std::tuple_size_v<InputTuple>;
  constexpr std::size_t rank
    = problem_rank<ReductionAxes, all_bindings_type>(std::make_index_sequence<input_count + 1>());
  constexpr auto tile_extents
    = common_native_tile_extents<rank, all_bindings_type>(std::make_index_sequence<input_count + 1>());
  const auto all_bindings = std::tuple_cat(std::tuple{output}, inputs);
  const auto extents      = merge_dynamic_extents<rank>(all_bindings, std::make_index_sequence<input_count + 1>());
  std::array<std::size_t, rank> tile_counts{};
  for (std::size_t axis = 0; axis < rank; ++axis) {
    assert(extents[axis] % tile_extents[axis] == 0);
    tile_counts[axis] = extents[axis] / tile_extents[axis];
  }
  return select_native_tiled_reduction_schedule_from_tile_counts<
    ReductionAxes,
    Output,
    InputTuple,
    Override,
    CachePolicy>(tile_counts);
}

template <
  class ReductionAxes,
  native_tiled_reduction_override Override = native_tiled_reduction_override::automatic,
  class CachePolicy                        = host_native_tiled_schedule_policy,
  class ArgumentTuple,
  std::size_t... InputIndexes>
void dispatch_native_tiled_reduce(ArgumentTuple& arguments, std::index_sequence<InputIndexes...>) {
  constexpr std::size_t argument_count = std::tuple_size_v<std::remove_reference_t<ArgumentTuple>>;
  auto inputs                          = std::tuple{std::get<InputIndexes + 1>(arguments)...};
  auto&& updater                       = std::get<argument_count - 1>(arguments);
  using output_type                    = std::remove_cvref_t<decltype(std::get<0>(arguments))>;
  using input_tuple_type               = decltype(inputs);
  if constexpr (native_complete_fragment_compatible_v<output_type, input_tuple_type>) {
    const auto schedule
      = select_native_tiled_reduction_schedule<ReductionAxes, Override, CachePolicy>(std::get<0>(arguments), inputs);

    switch (schedule.path) {
    case native_tiled_reduction_path::complete_fragment:
      host_complete_native_fragment_reduce<ReductionAxes>(std::get<0>(arguments), inputs, updater);
      return;
    case native_tiled_reduction_path::reduction_blocked_fragment:
      host_rectangular_reduction_blocked_native_fragment_reduce<ReductionAxes>(
        std::get<0>(arguments),
        inputs,
        updater,
        schedule.block_shape
      );
      return;
    case native_tiled_reduction_path::scalar:
      host_scalar_native_tiled_reduce<ReductionAxes>(std::get<0>(arguments), inputs, updater);
      return;
    }
  } else {
    host_scalar_native_tiled_reduce<ReductionAxes>(std::get<0>(arguments), inputs, updater);
  }
}

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_NATIVE_TILED_REDUCE_POLICY
