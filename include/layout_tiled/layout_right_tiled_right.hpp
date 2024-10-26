/// Copyright (C) 2024, CEA
///
/// Licensed under the MIT License ("the License" hereafter);
/// You may not use this file except in compliance with the License.
/// You should have received a full copy of the License along with this program;
/// if not, you can obtain a copy at:
///
///    https://opensource.org/license/MIT
///
/// The software is provided “as is”, without warranty of any kind, express
/// or implied, including but not limited to the warranties of merchantability,
/// fitness for a particular purpose and noninfringement. In no event shall
/// the authors or copyright holders be liable for any claim, damages or other
/// liability, whether in an action of contract, tort or otherwise, arising from,
/// out of or in connection with the software or the use or other dealings
/// in the software.
///
/// Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>

#pragma once

#include <cassert>

template <typename tile_size_type, tile_size_type RowTileSize, tile_size_type ColTileSize>
class layout_right_tiled_right {
public:
  template <class Extents>
  struct mapping {
    static_assert(Extents::rank() == 2, "This implementation is for 2D only");
    static_assert(RowTileSize > 0, "Row tile size must be positive");
    static_assert(ColTileSize > 0, "Column tile size must be positive");

    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_right_tiled_right;

    constexpr mapping() noexcept                          = default;
    constexpr mapping(mapping const&) noexcept            = default;
    constexpr mapping& operator=(mapping const&) noexcept = default;

    mapping(extents_type const& exts) noexcept : extents_(exts) {
      assert(exts.extent(0) > 0);
      assert(exts.extent(1) > 0);
    }

    /* --- REQUIRED MEMBERS --- */
    constexpr auto extents() const -> extents_type const& {
      return extents_;
    }

    constexpr auto required_span_size() const noexcept -> size_type {
      return nb_row_tiles() * nb_col_tiles() * tile_size();
    }

    template <class RowIndex, class ColIndex>
    constexpr auto operator()(RowIndex row, ColIndex col) const noexcept -> size_type {
      return tile_offset(row, col) + offset_in_tile(row, col);
    }

    // Mapping is always unique.
    static constexpr auto is_always_unique() noexcept -> bool {
      return true;
    }
    // Only exhaustive if `extents_.extent(0) % col_tile_size == 0`, so not always.
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
      return (extents_.extent(0) % RowTileSize == 0) && (extents_.extent(1) % ColTileSize == 0);
    }
    // There are some circumstances where this is strided, but we're not concerned about that optimization, so
    // we're allowed to just return false here.
    constexpr auto is_strided() const noexcept -> bool {
      return false;
    }

  private:
    /* --- HELPER MEMBERS --- */
    constexpr auto nb_row_tiles() const noexcept -> size_type {
      return extents_.extent(0) / RowTileSize + size_type((extents_.extent(0) % RowTileSize) != 0);
    }

    constexpr auto nb_col_tiles() const noexcept -> size_type {
      return extents_.extent(1) / ColTileSize + size_type((extents_.extent(1) % ColTileSize) != 0);
    }

    constexpr auto tile_size() const noexcept -> size_type {
      return RowTileSize * ColTileSize;
    }

    constexpr auto tile_offset(size_type row, size_type col) const noexcept -> size_type {
      auto row_tile = row / RowTileSize;
      auto col_tile = col / ColTileSize;
      return (row_tile * nb_col_tiles() + col_tile) * tile_size();
    }

    constexpr auto offset_in_tile(size_type row, size_type col) const noexcept -> size_type {
      auto t_row = row % RowTileSize;
      auto t_col = col % ColTileSize;
      return t_row * ColTileSize + t_col;
    }

    extents_type extents_;
  };
};
