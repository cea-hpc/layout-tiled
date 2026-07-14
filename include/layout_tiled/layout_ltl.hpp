// include/layout_tiled/layout_ltl.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_LAYOUT_LTL
#define INCLUDED_LAYOUT_TILED_LAYOUT_LTL

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <utility>

#if defined(LAYOUT_TILED_KOKKOS_MDSPAN)
#include <experimental/mdspan>
#else
#include <mdspan>
#endif

#include "detail/common.hpp"
#include "detail/macros.hpp"

namespace layout_tiled {

template <size_t... TileExtents>
class layout_ltl {
public:
  template <class Extents>
  struct mapping {
    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_ltl<TileExtents...>;

    using outer_layout_type = std::layout_left;
    using inner_layout_type = std::layout_left;

    static_assert(extents_type::rank() > 0, "layout_tiled::layout_ltl::mapping: number of extents must be positive");
    static_assert(
      extents_type::rank() == sizeof...(TileExtents),
      "layout_tiled::layout_ltl::mapping: number of tile extents must match extents rank"
    );
    static_assert(((TileExtents > 0) && ...), "layout_tiled::layout_ltl::mapping: all tile extents must be positive");
    static_assert(
      [] {
        size_t product           = 1;
        constexpr auto max_index = static_cast<size_t>((std::numeric_limits<index_type>::max)());
        for (const size_t extent : std::array<size_t, sizeof...(TileExtents)>{TileExtents...}) {
          if (extent > max_index / product) {
            return false;
          }
          product *= extent;
        }
        return true;
      }(),
      "layout_tiled::layout_ltl::mapping: tile size must be representable as index_type"
    );

  private:
    static constexpr auto required_span_size_is_representable(const extents_type& ext) -> bool {
      for (rank_type r = 0; r < extents_type::rank(); ++r) {
        if (ext.extent(r) == 0) {
          return true;
        }
      }
      index_type prod = 1;
      for (rank_type r = 0; r < extents_type::rank(); ++r) {
        bool overflowed = __builtin_mul_overflow(prod, ext.extent(r), &prod);
        if (overflowed) {
          return false;
        }
      }
      return true;
    }
    static_assert(
      extents_type::rank_dynamic() > 0 || required_span_size_is_representable(extents_type()),
      "layout_tiled::layout_ltl::mapping: product of static extents must be representable as index_type"
    );

    static constexpr auto tile_extents_are_multiples_of_extents(const extents_type& ext) -> bool {
      if constexpr (extents_type::rank() == 0) {
        return true;
      }
      constexpr std::array<size_t, sizeof...(TileExtents)> tile_extents{TileExtents...};
      for (rank_type r = 0; r < extents_type::rank(); ++r) {
        if (0 != (static_cast<size_t>(ext.extent(r)) % tile_extents[r])) {
          return false;
        }
      }
      return true;
    }
    static_assert(
      extents_type::rank_dynamic() > 0 || tile_extents_are_multiples_of_extents(extents_type()),
      "layout_tiled::layout_ltl::mapping: all static extents must be multiples of their tile extents"
    );

  public:
    // [mdspan.layout_tiled::layout_ltl.cons]
    LAYOUT_TILED_INLINE_FUNCTION
    constexpr mapping() noexcept = default;

    LAYOUT_TILED_INLINE_FUNCTION
    constexpr mapping(const mapping&) noexcept = default;
    LAYOUT_TILED_INLINE_FUNCTION
    constexpr auto operator=(const mapping&) noexcept -> mapping& = default;

    LAYOUT_TILED_INLINE_FUNCTION
    constexpr mapping(mapping&&) noexcept = default;
    LAYOUT_TILED_INLINE_FUNCTION
    constexpr auto operator=(mapping&&) noexcept -> mapping& = default;

    LAYOUT_TILED_INLINE_FUNCTION
    constexpr ~mapping() noexcept = default;

    LAYOUT_TILED_INLINE_FUNCTION
    constexpr explicit mapping(const extents_type& exts) noexcept : extents_(exts) {
      assert(required_span_size_is_representable(exts));
      assert(tile_extents_are_multiples_of_extents(exts));
    }

    template <class OtherExtents>
      requires std::is_constructible_v<extents_type, OtherExtents>
    explicit(!std::is_convertible_v<OtherExtents, extents_type>) LAYOUT_TILED_INLINE_FUNCTION
      constexpr mapping(const layout_ltl<TileExtents...>::template mapping<OtherExtents>& other) noexcept :
        extents_(other.extents()) {
      assert(required_span_size_is_representable(extents_));
      assert(tile_extents_are_multiples_of_extents(extents_));
    }

    /* --- REQUIRED MEMBERS --- */
    // [mdspan.layout_tiled::layout_ltl.obs]
    LAYOUT_TILED_INLINE_FUNCTION
    constexpr auto extents() const noexcept -> const extents_type& {
      return extents_;
    }

    LAYOUT_TILED_INLINE_FUNCTION
    constexpr auto required_span_size() const noexcept -> index_type {
      for (rank_type i = 0; i < extents_type::rank(); ++i) {
        if (extents_.extent(i) == 0) {
          return 0;
        }
      }
      index_type res = 1;
      for (rank_type i = 0; i < extents_type::rank(); ++i) {
        res *= extents_.extent(i);
      }
      return res;
    }

    template <class... Idxs>
      requires (extents_type::rank() == sizeof...(Idxs)) && (std::is_convertible_v<Idxs, index_type> && ...)
    LAYOUT_TILED_IMPL_HOST_DEVICE constexpr auto operator()(Idxs... idxs) const noexcept -> index_type {
      assert(
        "layout_tiled::layout_ltl::mapping: out of bounds indexing"
        && detail::is_md_idx_in_span(std::make_index_sequence<extents_type::rank()>(), extents_, idxs...)
      );

      return tile_offset_outer(static_cast<index_type>(idxs)...) + tile_offset_inner(static_cast<index_type>(idxs)...);
    }

    template <class OtherExtents>
      requires (extents_type::rank() == OtherExtents::rank())
    LAYOUT_TILED_INLINE_FUNCTION friend constexpr auto
    operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept -> bool {
      return lhs.extents() == rhs.extents();
    }

    /// Mapping is always unique.
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto is_always_unique() noexcept -> bool {
      return true;
    }
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto is_always_exhaustive() noexcept -> bool {
      return true;
    }
    /// Not always strided: stride is regular within tiles but changes at tile boundaries.
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto is_always_strided() noexcept -> bool {
      return false;
    }

    /// Always unique.
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto is_unique() noexcept -> bool {
      return true;
    }
    /// Always exhaustive.
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto is_exhaustive() noexcept -> bool {
      return true;
    }
    /// Not strided: stride changes at tile boundaries.
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto is_strided() noexcept -> bool {
      return false;
    }

    /// Stride is the forward product of tile extents.
    LAYOUT_TILED_INLINE_FUNCTION
    constexpr auto stride(rank_type r) const noexcept -> index_type {
      return detail::fwd_prod_of_extents<index_type, TileExtents...>(r);
    }
    /* --- END REQUIRED MEMBERS --- */

    /* --- PUBLIC HELPER MEMBERS --- */
    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto textents() noexcept -> std::array<index_type, sizeof...(TileExtents)> {
      return {static_cast<index_type>(TileExtents)...};
    }

    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto textent(rank_type i) noexcept -> index_type {
      return textents()[i];
    }

    LAYOUT_TILED_INLINE_FUNCTION
    static constexpr auto tile_size() noexcept -> index_type {
      return (index_type{1} * ... * static_cast<index_type>(TileExtents));
    }

    LAYOUT_TILED_INLINE_FUNCTION
    constexpr auto ntiles_in_extent(rank_type ext) const noexcept -> index_type {
      return extents_.extent(ext) / textent(ext) + index_type((extents_.extent(ext) % textent(ext)) != 0);
    }
    /* --- END PUBLIC HELPER MEMBERS --- */

    /* --- HELPER MEMBERS --- */
    /// Finds the starting offset of the tile in which the indexed element is located.
    template <class... Idxs>
    LAYOUT_TILED_IMPL_HOST_DEVICE constexpr auto tile_offset_outer(Idxs... idxs) const noexcept -> index_type {
      auto idx_arr      = std::to_array({idxs...});
      index_type offset = 0;
      index_type stride = 1;
      for (rank_type i = 0; i < extents_type::rank(); ++i) {
        offset += idx_arr[i] / textent(i) * stride;
        stride *= ntiles_in_extent(i);
      }
      return offset * tile_size();
    }

    /// Finds the offset of the indexed element within a tile.
    template <class... Idxs>
    LAYOUT_TILED_IMPL_HOST_DEVICE constexpr auto tile_offset_inner(Idxs... idxs) const noexcept -> index_type {
      auto idx_arr      = std::to_array({idxs...});
      index_type offset = 0;
      index_type stride = 1;
      for (rank_type i = 0; i < extents_type::rank(); ++i) {
        offset += idx_arr[i] % textent(i) * stride;
        stride *= textent(i);
      }
      return offset;
    }
    /* --- END HELPER MEMBERS --- */

  private:
    extents_type extents_;
  };
};

} // namespace layout_tiled

#endif // INCLUDED_LAYOUT_TILED_LAYOUT_LTL
