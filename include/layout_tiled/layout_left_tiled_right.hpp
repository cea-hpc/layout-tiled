#pragma once

#include <cassert>

class layout_left_tiled_right {
public:
  template <class Extents>
  struct mapping {
    static_assert(Extents::rank() == 2, "This implementation is for 2D only");

    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_left_tiled_right;

    constexpr mapping() noexcept                          = default;
    constexpr mapping(mapping const&) noexcept            = default;
    constexpr mapping& operator=(mapping const&) noexcept = default;

    mapping(extents_type const& exts, size_type trow, size_type tcol) noexcept
        : extents_(exts), row_tile_size_(trow), col_tile_size_(tcol) {
      assert(trow > 0);
      assert(tcol > 0);
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
      return (extents_.extent(0) % row_tile_size_ == 0) && (extents_.extent(1) % col_tile_size_ == 0);
    }
    // There are some circumstances where this is strided, but we're not concerned about that optimization, so
    // we're allowed to just return false here.
    constexpr auto is_strided() const noexcept -> bool {
      return false;
    }

  private:
    /* --- HELPER MEMBERS --- */
    constexpr auto nb_row_tiles() const noexcept -> size_type {
      return extents_.extent(0) / row_tile_size_ + size_type((extents_.extent(0) % row_tile_size_) != 0);
    }

    constexpr auto nb_col_tiles() const noexcept -> size_type {
      return extents_.extent(1) / col_tile_size_ + size_type((extents_.extent(1) % col_tile_size_) != 0);
    }

    constexpr auto tile_size() const noexcept -> size_type {
      return row_tile_size_ * col_tile_size_;
    }

    auto tile_offset(size_type row, size_type col) const noexcept -> size_type {
      auto row_tile = row / row_tile_size_;
      auto col_tile = col / col_tile_size_;
      return (col_tile * nb_row_tiles() + row_tile) * tile_size();
    }

    auto offset_in_tile(size_type row, size_type col) const noexcept -> size_type {
      auto t_row = row % row_tile_size_;
      auto t_col = col % col_tile_size_;
      return t_row * col_tile_size_ + t_col;
    }

    extents_type extents_;
    size_type row_tile_size_;
    size_type col_tile_size_;
  };
};
