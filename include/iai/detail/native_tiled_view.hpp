// include/iai/detail/native_tiled_view.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_NATIVE_TILED_VIEW
#define INCLUDED_IAI_DETAIL_NATIVE_TILED_VIEW

#include <array>
#include <cassert>
#include <cstddef>
#include <experimental/mdspan>
#include <type_traits>
#include <utility>

#include "layout_tiled_mapping.hpp"

namespace iai::detail {

template <class Index, std::size_t Rank>
struct hierarchical_tile_coordinate {
  std::array<Index, Rank> local{};
  std::array<Index, Rank> tile{};
};

/// Recognized native tiled storage represented as separate local-tile and
/// outer tile-grid modes. This is an internal coordinate-to-offset transform;
/// it neither owns nor accesses elements.
template <recognized_layout_tiled_mapping Mapping>
class native_tile_decomposition {
public:
  using mapping_type      = std::remove_cvref_t<Mapping>;
  using traits_type       = layout_tiled_mapping_traits<mapping_type>;
  using index_type        = typename mapping_type::index_type;
  using inner_layout_type = typename traits_type::inner_layout_type;

  static constexpr std::size_t rank  = mapping_type::extents_type::rank();
  using coordinate_type              = std::array<index_type, rank>;
  using hierarchical_coordinate_type = hierarchical_tile_coordinate<index_type, rank>;

private:
  template <std::size_t... Dimensions>
  static auto make_local_extents(std::index_sequence<Dimensions...>)
    -> std::extents<index_type, traits_type::tile_extents[Dimensions]...>;

public:
  using local_extents_type = decltype(make_local_extents(std::make_index_sequence<rank>()));
  using local_mapping_type = typename inner_layout_type::template mapping<local_extents_type>;

  explicit constexpr native_tile_decomposition(mapping_type mapping) noexcept : mapping_(mapping) {}

  [[nodiscard]] static constexpr auto local_extents() noexcept -> coordinate_type {
    coordinate_type result{};
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      result[dimension] = static_cast<index_type>(traits_type::tile_extents[dimension]);
    }
    return result;
  }

  [[nodiscard]] constexpr auto tile_grid_extents() const noexcept -> coordinate_type {
    coordinate_type result{};
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      result[dimension] = mapping_.ntiles_in_extent(static_cast<typename mapping_type::rank_type>(dimension));
    }
    return result;
  }

  [[nodiscard]] constexpr auto split(const coordinate_type& logical) const noexcept -> hierarchical_coordinate_type {
    hierarchical_coordinate_type result{};
    const coordinate_type extents = local_extents();
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      assert(logical[dimension] < mapping_.extents().extent(dimension));
      result.local[dimension] = logical[dimension] % extents[dimension];
      result.tile[dimension]  = logical[dimension] / extents[dimension];
    }
    return result;
  }

  [[nodiscard]] constexpr auto tile_origin(const coordinate_type& tile) const noexcept -> coordinate_type {
    coordinate_type result{};
    const coordinate_type extents = local_extents();
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      assert(tile[dimension] < mapping_.ntiles_in_extent(static_cast<typename mapping_type::rank_type>(dimension)));
      result[dimension] = tile[dimension] * extents[dimension];
    }
    return result;
  }

  /// Partial evaluation of the fixed tile-grid coordinate.
  [[nodiscard]] constexpr auto fixed_tile_offset(const coordinate_type& tile) const noexcept -> index_type {
    return outer_offset(tile_origin(tile), std::make_index_sequence<rank>());
  }

  [[nodiscard]] constexpr auto operator()(const hierarchical_coordinate_type& coordinate) const noexcept
    -> index_type {
    return fixed_tile_offset(coordinate.tile)
           + local_offset(
             coordinate.local,
             local_mapping_type(local_extents_type{}),
             std::make_index_sequence<rank>()
           );
  }

private:
  template <std::size_t... Dimensions>
  [[nodiscard]] constexpr auto
  outer_offset(const coordinate_type& origin, std::index_sequence<Dimensions...>) const noexcept -> index_type {
    return mapping_.tile_offset_outer(origin[Dimensions]...);
  }

  template <std::size_t... Dimensions>
  [[nodiscard]] static constexpr auto local_offset(
    const coordinate_type& local,
    const local_mapping_type& mapping,
    std::index_sequence<Dimensions...>
  ) noexcept -> index_type {
    return mapping(local[Dimensions]...);
  }

  mapping_type mapping_;
};

template <recognized_layout_tiled_mapping Mapping>
native_tile_decomposition(Mapping) -> native_tile_decomposition<Mapping>;

/// A lazy view of a recognized tiled mdspan as an N-D grid of affine native
/// tiles. Binding a tile offsets the source handle exactly once and leaves a
/// static affine inner mapping for the hot local loops.
template <class MdSpan>
  requires recognized_layout_tiled_mdspan<MdSpan>
class native_tiled_view {
public:
  using source_type        = MdSpan;
  using mapping_type       = typename source_type::mapping_type;
  using decomposition_type = native_tile_decomposition<mapping_type>;
  using element_type       = typename source_type::element_type;
  using index_type         = typename source_type::index_type;
  using accessor_type      = typename source_type::accessor_type;
  using data_handle_type   = typename source_type::data_handle_type;
  using inner_layout_type  = typename decomposition_type::inner_layout_type;

  static constexpr std::size_t rank = source_type::rank();
  using coordinate_type             = std::array<index_type, rank>;
  using tile_extents_type           = typename decomposition_type::local_extents_type;
  using tile_mapping_type           = typename decomposition_type::local_mapping_type;
  using tile_type                   = std::mdspan<element_type, tile_extents_type, inner_layout_type, accessor_type>;

  explicit constexpr native_tiled_view(source_type source) noexcept : source_(source) {}

  [[nodiscard]] constexpr auto source() const noexcept -> const source_type& {
    return source_;
  }

  [[nodiscard]] constexpr auto decomposition() const noexcept -> decomposition_type {
    return decomposition_type(source_.mapping());
  }

  [[nodiscard]] constexpr auto tile_grid_extents() const noexcept -> coordinate_type {
    return decomposition().tile_grid_extents();
  }

  [[nodiscard]] constexpr auto bind_tile(const coordinate_type& tile_coordinates) const noexcept -> tile_type {
    const index_type offset = decomposition().fixed_tile_offset(tile_coordinates);
    data_handle_type handle = source_.accessor().offset(source_.data_handle(), static_cast<std::size_t>(offset));
    return tile_type(handle, tile_mapping_type(tile_extents_type{}), source_.accessor());
  }

private:
  source_type source_;
};

template <class MdSpan>
native_tiled_view(MdSpan) -> native_tiled_view<MdSpan>;

/// A partially evaluated rectangular range of native tiles. The origin is
/// bound once through the source accessor; subsequent tiles use the affine
/// outer tile-grid strides relative to that origin.
template <class MdSpan>
  requires recognized_layout_tiled_mdspan<MdSpan>
class native_tile_range {
public:
  using tiled_type        = native_tiled_view<MdSpan>;
  using mapping_type      = typename tiled_type::mapping_type;
  using traits_type       = layout_tiled_mapping_traits<mapping_type>;
  using index_type        = typename tiled_type::index_type;
  using coordinate_type   = typename tiled_type::coordinate_type;
  using tile_type         = typename tiled_type::tile_type;
  using data_handle_type  = typename tiled_type::data_handle_type;
  using tile_extents_type = typename tiled_type::tile_extents_type;
  using tile_mapping_type = typename tiled_type::tile_mapping_type;

  static constexpr std::size_t rank = tiled_type::rank;

  explicit constexpr native_tile_range(tiled_type tiled, const coordinate_type& origin) noexcept :
      origin_(origin),
      origin_tile_(tiled.bind_tile(origin)),
      tile_strides_(make_tile_strides(tiled)) {}

  [[nodiscard]] constexpr auto origin() const noexcept -> const coordinate_type& {
    return origin_;
  }

  [[nodiscard]] constexpr auto bind_tile(const coordinate_type& coordinates) const noexcept -> tile_type {
    std::size_t relative_offset = 0;
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      assert(coordinates[dimension] >= origin_[dimension]);
      relative_offset
        += static_cast<std::size_t>(coordinates[dimension] - origin_[dimension]) * tile_strides_[dimension];
    }
    if (relative_offset == 0) {
      return origin_tile_;
    }
    data_handle_type handle = origin_tile_.accessor().offset(origin_tile_.data_handle(), relative_offset);
    return tile_type(handle, origin_tile_.mapping(), origin_tile_.accessor());
  }

private:
  [[nodiscard]] static constexpr auto make_tile_strides(const tiled_type& tiled) noexcept
    -> std::array<std::size_t, rank> {
    std::array<std::size_t, rank> result{};
    const auto grid_extents = tiled.tile_grid_extents();
    std::size_t stride      = tile_mapping_type(tile_extents_type{}).required_span_size();
    if constexpr (std::is_same_v<typename traits_type::outer_layout_type, std::layout_right>) {
      for (std::size_t position = rank; position > 0; --position) {
        const std::size_t dimension = position - 1;
        result[dimension]           = stride;
        stride *= static_cast<std::size_t>(grid_extents[dimension]);
      }
    } else {
      for (std::size_t dimension = 0; dimension < rank; ++dimension) {
        result[dimension] = stride;
        stride *= static_cast<std::size_t>(grid_extents[dimension]);
      }
    }
    return result;
  }

  coordinate_type origin_{};
  tile_type origin_tile_;
  std::array<std::size_t, rank> tile_strides_{};
};

template <class MdSpan>
native_tile_range(native_tiled_view<MdSpan>, const typename native_tiled_view<MdSpan>::coordinate_type&)
  -> native_tile_range<MdSpan>;

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_NATIVE_TILED_VIEW
