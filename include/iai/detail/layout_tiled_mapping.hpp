// include/iai/detail/layout_tiled_mapping.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_IAI_DETAIL_LAYOUT_TILED_MAPPING
#define INCLUDED_IAI_DETAIL_LAYOUT_TILED_MAPPING

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include <layout_tiled/layout_ltl.hpp>
#include <layout_tiled/layout_ltr.hpp>
#include <layout_tiled/layout_rtl.hpp>
#include <layout_tiled/layout_rtr.hpp>

namespace iai::detail {

template <class>
struct known_layout_tiled_layout {
  static constexpr bool is_recognized = false;
};

template <std::size_t... TileExtents>
struct known_layout_tiled_layout<layout_tiled::layout_rtr<TileExtents...>> {
  static constexpr bool is_recognized = true;
  static constexpr std::array<std::size_t, sizeof...(TileExtents)> tile_extents{TileExtents...};
  using outer_layout_type = std::layout_right;
  using inner_layout_type = std::layout_right;
};

template <std::size_t... TileExtents>
struct known_layout_tiled_layout<layout_tiled::layout_rtl<TileExtents...>> {
  static constexpr bool is_recognized = true;
  static constexpr std::array<std::size_t, sizeof...(TileExtents)> tile_extents{TileExtents...};
  using outer_layout_type = std::layout_right;
  using inner_layout_type = std::layout_left;
};

template <std::size_t... TileExtents>
struct known_layout_tiled_layout<layout_tiled::layout_ltr<TileExtents...>> {
  static constexpr bool is_recognized = true;
  static constexpr std::array<std::size_t, sizeof...(TileExtents)> tile_extents{TileExtents...};
  using outer_layout_type = std::layout_left;
  using inner_layout_type = std::layout_right;
};

template <std::size_t... TileExtents>
struct known_layout_tiled_layout<layout_tiled::layout_ltl<TileExtents...>> {
  static constexpr bool is_recognized = true;
  static constexpr std::array<std::size_t, sizeof...(TileExtents)> tile_extents{TileExtents...};
  using outer_layout_type = std::layout_left;
  using inner_layout_type = std::layout_left;
};

template <class Mapping>
consteval auto is_exact_layout_tiled_mapping() -> bool {
  using mapping_type = std::remove_cvref_t<Mapping>;
  if constexpr (!requires {
                  typename mapping_type::layout_type;
                  typename mapping_type::extents_type;
                }) {
    return false;
  } else {
    using layout_type = typename mapping_type::layout_type;
    if constexpr (!known_layout_tiled_layout<layout_type>::is_recognized) {
      return false;
    } else {
      using expected_mapping = typename layout_type::template mapping<typename mapping_type::extents_type>;
      return std::same_as<mapping_type, expected_mapping>;
    }
  }
}

template <class Mapping>
struct layout_tiled_mapping_traits {
  static constexpr bool is_recognized = false;
};

template <class Mapping>
  requires (is_exact_layout_tiled_mapping<Mapping>())
struct layout_tiled_mapping_traits<Mapping> : known_layout_tiled_layout<typename Mapping::layout_type> {
  using mapping_type = Mapping;
  using extents_type = typename mapping_type::extents_type;
  using layout_type  = typename mapping_type::layout_type;
};

template <class Mapping>
concept recognized_layout_tiled_mapping = layout_tiled_mapping_traits<std::remove_cvref_t<Mapping>>::is_recognized;

template <class View>
concept recognized_layout_tiled_mdspan = requires {
  typename std::remove_cvref_t<View>::mapping_type;
} && recognized_layout_tiled_mapping<typename std::remove_cvref_t<View>::mapping_type>;

} // namespace iai::detail

#endif // INCLUDED_IAI_DETAIL_LAYOUT_TILED_MAPPING
