// include/iai/detail/native_tiled_cache_block.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_NATIVE_TILED_CACHE_BLOCK
#define INCLUDED_IAI_DETAIL_NATIVE_TILED_CACHE_BLOCK

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>

#include "native_tiled_reduce.hpp"

namespace iai::detail {

template <std::size_t Rank>
struct native_tile_block_shape {
  using coordinate_type = std::array<std::size_t, Rank>;

  coordinate_type tile_extents{};

  [[nodiscard]] constexpr auto valid() const noexcept -> bool {
    return std::all_of(tile_extents.begin(), tile_extents.end(), [](const std::size_t extent) { return extent != 0; });
  }

  [[nodiscard]] constexpr auto extent(const std::size_t axis) const noexcept -> std::size_t {
    assert(axis < Rank);
    return tile_extents[axis];
  }

  [[nodiscard]] constexpr auto tile_count() const noexcept -> std::size_t {
    std::size_t result = 1;
    for (const std::size_t extent : tile_extents) {
      result *= extent;
    }
    return result;
  }

  friend constexpr auto operator==(const native_tile_block_shape&, const native_tile_block_shape&) -> bool = default;
};

template <std::size_t Rank>
struct native_tile_block {
  using coordinate_type = std::array<std::size_t, Rank>;

  coordinate_type begin{};
  coordinate_type extents{};

  [[nodiscard]] constexpr auto end(const std::size_t axis) const noexcept -> std::size_t {
    assert(axis < Rank);
    return begin[axis] + extents[axis];
  }

  [[nodiscard]] constexpr auto tile_count() const noexcept -> std::size_t {
    std::size_t result = 1;
    for (const std::size_t extent : extents) {
      result *= extent;
    }
    return result;
  }

  friend constexpr auto operator==(const native_tile_block&, const native_tile_block&) -> bool = default;
};

template <std::size_t Rank>
[[nodiscard]] constexpr auto
native_tile_block_counts(const std::array<std::size_t, Rank>& tile_counts, const native_tile_block_shape<Rank>& shape)
  -> std::array<std::size_t, Rank> {
  assert(shape.valid());
  std::array<std::size_t, Rank> result{};
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    result[axis] = (tile_counts[axis] + shape.extent(axis) - 1) / shape.extent(axis);
  }
  return result;
}

template <std::size_t Rank>
[[nodiscard]] constexpr auto make_native_tile_block(
  const std::array<std::size_t, Rank>& block_coordinates,
  const std::array<std::size_t, Rank>& tile_counts,
  const native_tile_block_shape<Rank>& shape
) -> native_tile_block<Rank> {
  assert(shape.valid());
  native_tile_block<Rank> result{};
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    result.begin[axis] = block_coordinates[axis] * shape.extent(axis);
    assert(result.begin[axis] < tile_counts[axis]);
    result.extents[axis] = std::min(shape.extent(axis), tile_counts[axis] - result.begin[axis]);
  }
  return result;
}

template <std::size_t Depth, class Axes, std::size_t Rank, class Function>
constexpr auto for_each_native_tile_in_block(
  const native_tile_block<Rank>& block,
  std::array<std::size_t, Rank>& tile_coordinates,
  Function& function
) -> void {
  if constexpr (Depth == Axes::size) {
    function(tile_coordinates);
  } else {
    constexpr std::size_t axis = Axes::values[Depth];
    for (std::size_t coordinate = block.begin[axis]; coordinate < block.end(axis); ++coordinate) {
      tile_coordinates[axis] = coordinate;
      for_each_native_tile_in_block<Depth + 1, Axes>(block, tile_coordinates, function);
    }
  }
}

struct native_tile_block_operand_metrics {
  std::size_t tile_bytes{};
  std::size_t distinct_tile_count{};
  std::size_t working_set_bytes{};
  std::size_t reuse_factor{};

  friend constexpr auto operator==(const native_tile_block_operand_metrics&, const native_tile_block_operand_metrics&)
    -> bool
    = default;
};

template <class Binding>
consteval auto native_binding_tile_element_count() -> std::size_t {
  std::size_t result = 1;
  for (const std::size_t extent : native_mapping_traits<Binding>::tile_extents) {
    result *= extent;
  }
  return result;
}

template <class Binding>
constexpr auto native_binding_carries_axis(const std::size_t axis) -> bool {
  return std::find(Binding::axes_type::values.begin(), Binding::axes_type::values.end(), axis)
         != Binding::axes_type::values.end();
}

template <std::size_t Rank, class Binding>
[[nodiscard]] constexpr auto native_tile_block_operand_metrics_for(const native_tile_block_shape<Rank>& shape)
  -> native_tile_block_operand_metrics {
  assert(shape.valid());
  native_tile_block_operand_metrics result{};
  result.tile_bytes          = native_binding_tile_element_count<Binding>() * sizeof(typename Binding::value_type);
  result.distinct_tile_count = 1;
  result.reuse_factor        = 1;
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    if (native_binding_carries_axis<Binding>(axis)) {
      result.distinct_tile_count *= shape.extent(axis);
    } else {
      result.reuse_factor *= shape.extent(axis);
    }
  }
  result.working_set_bytes = result.tile_bytes * result.distinct_tile_count;
  return result;
}

template <std::size_t Rank, class BindingTuple, std::size_t... BindingIndexes>
[[nodiscard]] constexpr auto
native_tile_block_metrics(const native_tile_block_shape<Rank>& shape, std::index_sequence<BindingIndexes...>)
  -> std::array<native_tile_block_operand_metrics, sizeof...(BindingIndexes)> {
  return {native_tile_block_operand_metrics_for<Rank, std::tuple_element_t<BindingIndexes, BindingTuple>>(shape)...};
}

template <std::size_t Rank, class BindingTuple>
[[nodiscard]] constexpr auto native_tile_block_metrics(const native_tile_block_shape<Rank>& shape) {
  return native_tile_block_metrics<Rank, BindingTuple>(
    shape,
    std::make_index_sequence<std::tuple_size_v<BindingTuple>>()
  );
}

template <std::size_t Rank, class BindingTuple>
[[nodiscard]] constexpr auto native_tile_block_working_set_bytes(const native_tile_block_shape<Rank>& shape)
  -> std::size_t {
  std::size_t result = 0;
  for (const auto metrics : native_tile_block_metrics<Rank, BindingTuple>(shape)) {
    result += metrics.working_set_bytes;
  }
  return result;
}

template <std::size_t Rank, class BindingTuple>
[[nodiscard]] constexpr auto native_tile_block_reused_bytes(const native_tile_block_shape<Rank>& shape)
  -> std::size_t {
  std::size_t result = 0;
  for (const auto metrics : native_tile_block_metrics<Rank, BindingTuple>(shape)) {
    result += metrics.working_set_bytes * (metrics.reuse_factor - 1);
  }
  return result;
}

template <std::size_t Rank>
struct native_tile_block_selection {
  native_tile_block_shape<Rank> shape{};
  std::size_t working_set_bytes{};
  std::size_t reused_bytes{};
  bool found{};
};

template <std::size_t Rank, class BindingTuple, std::size_t CandidateCount>
[[nodiscard]] constexpr auto select_native_tile_block_shape(
  const std::array<native_tile_block_shape<Rank>, CandidateCount>& candidates,
  const std::size_t cache_budget_bytes
) -> native_tile_block_selection<Rank> {
  native_tile_block_selection<Rank> result{};
  std::size_t selected_volume = 0;
  for (const auto& candidate : candidates) {
    if (!candidate.valid()) {
      continue;
    }
    const std::size_t working_set_bytes = native_tile_block_working_set_bytes<Rank, BindingTuple>(candidate);
    if (working_set_bytes > cache_budget_bytes) {
      continue;
    }
    const std::size_t reused_bytes = native_tile_block_reused_bytes<Rank, BindingTuple>(candidate);
    const std::size_t volume       = candidate.tile_count();
    if (!result.found || reused_bytes > result.reused_bytes
        || (reused_bytes == result.reused_bytes && volume > selected_volume)
        || (reused_bytes == result.reused_bytes && volume == selected_volume && working_set_bytes < result.working_set_bytes)) {
      result.shape             = candidate;
      result.working_set_bytes = working_set_bytes;
      result.reused_bytes      = reused_bytes;
      result.found             = true;
      selected_volume          = volume;
    }
  }
  return result;
}

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_NATIVE_TILED_CACHE_BLOCK
