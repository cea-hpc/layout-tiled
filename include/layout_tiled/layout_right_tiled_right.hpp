#pragma once

#include <cassert>

class layout_right_tiled_right {
public:
  template <class Extents>
  struct mapping {
    static_assert(Extents::rank() == 2, "This implementation is for 2D only");

    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_right_tiled_right;

    constexpr mapping() noexcept                          = default;
    constexpr mapping(mapping const&) noexcept            = default;
    constexpr mapping& operator=(mapping const&) noexcept = default;

    mapping(extents_type const& exts, size_type tm, size_type tn) noexcept
        : extents_(exts), tile_m_size_(tm), tile_n_size_(tn) {
      assert(tm > 0);
      assert(tn > 0);
      assert(exts.extent(0) > 0);
      assert(exts.extent(1) > 0);
    }
    //------------------------------------------------------------
    // Helper members (not part of the layout concept)

    constexpr size_type nb_tiles_m() const noexcept {
      return extents_.extent(0) / tile_m_size_ + size_type((extents_.extent(0) % tile_m_size_) != 0);
    }

    constexpr size_type nb_tiles_n() const noexcept {
      return extents_.extent(1) / tile_n_size_ + size_type((extents_.extent(1) % tile_n_size_) != 0);
    }

    constexpr size_type tile_size() const noexcept { return tile_m_size_ * tile_n_size_; }

    size_type tile_offset(size_type m, size_type n) const noexcept {
      // This could probably be more efficient, but for example purposes...
      auto m_tile = m / tile_m_size_;
      auto n_tile = n / tile_n_size_;
      return (m_tile * nb_tiles_m() + n_tile) * tile_size();
    }

    size_type offset_in_tile(size_type m, size_type n) const noexcept {
      auto t_m = m % tile_m_size_;
      auto t_n = n % tile_n_size_;
      return t_n * tile_m_size_ + t_m;
    }

    //------------------------------------------------------------
    // Required members

    constexpr extents_type const& extents() const { return extents_; }

    constexpr size_type required_span_size() const noexcept { return nb_tiles_m() * nb_tiles_n() * tile_size(); }

    template <class RowIndex, class ColIndex>
    // requires(is_convertible_v<RowIndex, size_type> &&
    //   is_convertible_v<ColIndex, size_type> &&
    //   is_nothrow_constructible_v<size_type, RowIndex> &&
    //   is_nothrow_constructible_v<size_type, ColIndex>)
    constexpr size_type operator()(RowIndex row, ColIndex col) const noexcept {
      // TODO (mfh 2022/08/04 check precondition that
      // extents_type::index-cast(row, col)
      // is a multidimensional index in extents_.
      return tile_offset(row, col) + offset_in_tile(row, col);
    }

    // Mapping is always unique
    static constexpr bool is_always_unique() noexcept { return true; }
    // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not always
    static constexpr bool is_always_exhaustive() noexcept { return false; }
    // There is not always a regular stride between elements in a given dimension
    static constexpr bool is_always_strided() noexcept { return false; }

    static constexpr bool is_unique() noexcept { return true; }
    constexpr bool is_exhaustive() const noexcept {
      // Only exhaustive if extents fit exactly into tile sizes...
      return (extents_.extent(0) % tile_m_size_ == 0) && (extents_.extent(1) % tile_n_size_ == 0);
    }
    // There are some circumstances where this is strided, but we're not
    // concerned about that optimization, so we're allowed to just return false here
    constexpr bool is_strided() const noexcept { return false; }

  private:
    extents_type extents_;
    size_type tile_m_size_;
    size_type tile_n_size_;
  };
};
