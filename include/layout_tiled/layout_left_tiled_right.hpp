// Copyright (C) 2024-2025 CEA
// Copyright (C) 2024-2025 Gabriel Dos Santos
//
// Distributed under the MIT License.
// You may not use this file except in compliance with the License.
// See https://opensource.org/license/MIT for license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <utility>

#include "details/common.hpp"

template <std::size_t... TileExtents>
class layout_left_tiled_right {
public:
  template <class Extents>
  struct mapping {
    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_left_tiled_right;

    static_assert(extents_type::rank() > 0, "layout_left_tiled_right::mapping: number of extents must be positive");
    static_assert(
      extents_type::rank() == sizeof...(TileExtents),
      "layout_left_tiled_right::mapping: number of tile extents must match extents rank"
    );
    static_assert(((TileExtents > 0) && ...), "layout_left_tiled_right::mapping: all tile extents must be positive");

  private:
    static constexpr auto required_span_size_is_representable(extents_type const& ext) -> bool {
      if constexpr (extents_type::rank() == 0) {
        return true;
      }

      index_type prod = ext.extent(0);
      for (rank_type r = 1; r < extents_type::rank(); r++) {
        bool overflowed = __builtin_mul_overflow(prod, ext.extent(r), &prod);
        if (overflowed) {
          return false;
        }
      }
      return true;
    }

    static_assert(
      extents_type::rank_dynamic() > 0 || required_span_size_is_representable(extents_type()),
      "layout_left_tiled_right::mapping: product of static extents must be representable as index_type"
    );

  public:
    // [mdspan.layout_left_tiled_right.cons]
    constexpr mapping() noexcept               = default;
    constexpr mapping(mapping const&) noexcept = default;
    mapping(extents_type const& exts) noexcept
        : extents_(exts) {
      assert(required_span_size_is_representable(exts));
      for (rank_type i = 0; i < exts.rank(); ++i) {
        assert(exts.extent(i) > 0);
      }
    }

    constexpr auto operator=(mapping const&) noexcept -> mapping& = default;

    /* --- REQUIRED MEMBERS --- */
    // [mdspan.layout_left_tiled_right.obs]
    constexpr auto extents() const -> extents_type const& {
      return extents_;
    }

    constexpr auto required_span_size() const noexcept -> index_type {
      index_type res = 1;
      for (std::size_t i = 0; i < extents_type::rank(); ++i) {
        res *= extents_.extent(i);
      }
      return res;
    }

    template <class... Idxs>
    constexpr auto operator()(Idxs... idxs) const noexcept -> index_type
      requires(extents_type::rank() == sizeof...(Idxs))
    {
      assert(
        "layout_left_tiled_right::mapping: out of bounds indexing" &&
        details::is_md_idx_in_span(std::make_index_sequence<extents_type::rank()>(), extents_, idxs...)
      );

      return tile_offset_outer(idxs...) + tile_offset_inner(idxs...);
    }

    // Mapping is always unique.
    static constexpr auto is_always_unique() noexcept -> bool {
      return true;
    }
    // Only exhaustive if `extents_.extent(i) % ntiles_in_extent(i) == 0`, so not always.
    static constexpr auto is_always_exhaustive() noexcept -> bool {
      return false;
    }
    // There is not always a regular stride between elements in a given dimension.
    static constexpr auto is_always_strided() noexcept -> bool {
      return false;
    }

    static constexpr auto is_unique() noexcept -> bool {
      return true;
    }
    // Only exhaustive if extents fit exactly into tile sizes.
    constexpr auto is_exhaustive() const noexcept -> bool {
      return []<std::size_t... Is>(auto const& self, std::index_sequence<Is...>) {
        return ((self.extents_.extent(Is) % self.textent(Is) == 0) && ...);
      }(*this, std::make_index_sequence<extents_type::rank()>{});
    }
    // There are some circumstances where this is strided, but we're currently not concerned about that optimization, so
    // we're allowed to just return false here.
    constexpr auto is_strided() const noexcept -> bool {
      return false;
    }
    /* --- END REQUIRED MEMBERS --- */

    /* --- PUBLIC HELPER MEMBERS --- */
    constexpr auto tile_size() const noexcept -> index_type {
      return (1 * ... * TileExtents);
    }

    constexpr auto textent(index_type i) const noexcept -> index_type {
      constexpr index_type pack_values[] = {TileExtents...};
      return pack_values[i];
    }

    constexpr auto ntiles_in_extent(index_type ext) const noexcept -> index_type {
      return extents_.extent(ext) / textent(ext) + index_type((extents_.extent(ext) % textent(ext)) != 0);
    }
    /* --- END PUBLIC HELPER MEMBERS --- */

  private:
    /* --- HELPER MEMBERS --- */
    /// Finds the starting offset of the tile in which the indexed element is located.
    template <class... Idxs>
    constexpr auto tile_offset_outer(Idxs... idxs) const noexcept -> index_type {
      constexpr rank_type rank = extents_type::rank();
      std::array<std::size_t, rank> idx_arr{idxs...};

      index_type offset = 0;
      index_type stride = 1;
      for (std::size_t i = 0; i < rank; ++i) {
        offset += idx_arr[i] / textent(i) * stride;
        stride *= ntiles_in_extent(i);
      }

      return offset * tile_size();
    }

    /// Finds the offset of the indexed element within a tile.
    template <class... Idxs>
    constexpr auto tile_offset_inner(Idxs... idxs) const noexcept -> index_type {
      constexpr rank_type rank        = extents_type::rank();
      std::array<size_t, rank> coords = [this, idxs...]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array<index_type, rank>{(idxs % textent(Is))...};
      }(std::make_index_sequence<rank>());

      index_type offset = 0;
      index_type stride = 1;
      for (size_type i = rank; i > 0; --i) {
        offset += coords[i - 1] * stride;
        stride *= textent(i - 1);
      }

      return offset;
    }
    /* --- END HELPER MEMBERS --- */

    extents_type extents_;
  };
};
