// include/layout_tiled/detail/tiled_view.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_DETAIL_TILED_VIEW
#define INCLUDED_LAYOUT_TILED_DETAIL_TILED_VIEW

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <experimental/mdspan>
#include <type_traits>
#include <utility>

namespace layout_tiled::detail {

template <class MdSpan>
concept tiled_mdspan = requires(const MdSpan& view, std::size_t dimension) {
  typename MdSpan::mapping_type::outer_layout_type;
  typename MdSpan::mapping_type::inner_layout_type;
  { MdSpan::rank() } -> std::convertible_to<std::size_t>;
  { MdSpan::mapping_type::textent(dimension) } -> std::convertible_to<typename MdSpan::index_type>;
  { view.mapping().ntiles_in_extent(dimension) } -> std::convertible_to<typename MdSpan::index_type>;
};

/// An N-D tiled mdspan viewed as an N-D grid of affine inner-layout tiles.
///
/// The tile mdspan preserves the source element type, accessor, data handle,
/// static tile extents, and the tiled mapping's inner layout.
template <tiled_mdspan MdSpan>
class basic_tiled_view {
public:
  using source_type       = MdSpan;
  using mapping_type      = typename source_type::mapping_type;
  using element_type      = typename source_type::element_type;
  using index_type        = typename source_type::index_type;
  using accessor_type     = typename source_type::accessor_type;
  using data_handle_type  = typename source_type::data_handle_type;
  using inner_layout_type = typename mapping_type::inner_layout_type;

  static constexpr std::size_t rank = source_type::rank();

private:
  template <std::size_t... Dimensions>
  static auto make_tile_extents(std::index_sequence<Dimensions...>)
    -> std::extents<index_type, mapping_type::textent(Dimensions)...>;

public:
  using tile_extents_type = decltype(make_tile_extents(std::make_index_sequence<rank>()));
  using tile_mapping_type = typename inner_layout_type::template mapping<tile_extents_type>;
  using tile_type         = std::mdspan<element_type, tile_extents_type, inner_layout_type, accessor_type>;

  explicit constexpr basic_tiled_view(source_type source) noexcept : source_(source) {}

  [[nodiscard]] constexpr auto source() const noexcept -> const source_type& {
    return source_;
  }

  [[nodiscard]] constexpr auto extent(std::size_t dimension) const noexcept -> index_type {
    assert(dimension < rank);
    return source_.mapping().ntiles_in_extent(static_cast<typename mapping_type::rank_type>(dimension));
  }

  [[nodiscard]] constexpr auto tile(const std::array<std::size_t, rank>& tile_coordinates) const noexcept
    -> tile_type {
    std::array<index_type, rank> origins{};
    for (std::size_t dimension = 0; dimension < rank; ++dimension) {
      assert(tile_coordinates[dimension] < static_cast<std::size_t>(extent(dimension)));
      origins[dimension] = static_cast<index_type>(
        tile_coordinates[dimension] * static_cast<std::size_t>(mapping_type::textent(dimension))
      );
    }

    const index_type offset = tile_offset(origins, std::make_index_sequence<rank>());
    data_handle_type handle = source_.accessor().offset(source_.data_handle(), static_cast<std::size_t>(offset));
    return tile_type(handle, tile_mapping_type(tile_extents_type{}), source_.accessor());
  }

private:
  template <std::size_t... Dimensions>
  [[nodiscard]] constexpr auto
  tile_offset(const std::array<index_type, rank>& origins, std::index_sequence<Dimensions...>) const noexcept
    -> index_type {
    return source_.mapping().tile_offset_outer(origins[Dimensions]...);
  }

  source_type source_;
};

template <tiled_mdspan MdSpan>
basic_tiled_view(MdSpan) -> basic_tiled_view<MdSpan>;

} // namespace layout_tiled::detail

#endif // INCLUDED_LAYOUT_TILED_DETAIL_TILED_VIEW
