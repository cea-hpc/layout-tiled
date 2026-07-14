// include/layout_tiled/detail/sub_tiled_mapping.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_DETAIL_SUB_TILED_MAPPING
#define INCLUDED_LAYOUT_TILED_DETAIL_SUB_TILED_MAPPING

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(LAYOUT_TILED_KOKKOS_MDSPAN)
#include <experimental/mdspan>
#else
#include <mdspan>
#endif

#include "common.hpp"
#include "macros.hpp"

#if defined(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN)
namespace layout_tiled::detail {

template <class DstExtents, class OrigMapping, class... Slices>
class sub_tiled_mapping;

template <class Mapping, class... Slices>
using sub_tiled_mapping_result_t = std::submdspan_mapping_result<sub_tiled_mapping<
  decltype(std::submdspan_extents(std::declval<typename Mapping::extents_type>(), std::declval<Slices>()...)),
  Mapping,
  Slices...>>;

template <class Mapping, class... Slices>
  requires (sizeof...(Slices) == Mapping::extents_type::rank())
LAYOUT_TILED_INLINE_FUNCTION constexpr auto make_sub_tiled_mapping(const Mapping& src, Slices... slices)
  -> sub_tiled_mapping_result_t<Mapping, Slices...>;

/// Layout policy used by mdspans returned from submdspan.
///
/// Keeping the source mapping and slice types in the policy makes its mapping
/// alias reconstruct the exact stateful mapping type required by mdspan.
template <class OrigMapping, class... Slices>
struct sub_tiled_layout {
  template <class Extents>
  using mapping = sub_tiled_mapping<Extents, OrigMapping, Slices...>;
};

template <class OrigMapping, class... Slices>
inline constexpr size_t sub_rank_from_slices_v
  = (0 + ... + (is_index_slice_v<Slices, typename OrigMapping::index_type> ? 0 : 1));

/// Maps each source dimension to its retained destination dimension, or -1
/// when an index slice removes that dimension.
template <size_t OrigRank, class IndexType, class... Slices>
struct sub_idx_pos_helper {
  static constexpr auto get() noexcept -> std::array<int, OrigRank> {
    std::array<int, OrigRank> result{};
    int sub_pos = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      ((result[Is] = (is_index_slice_v<Slices, IndexType> ? -1 : sub_pos++)), ...);
    }(std::make_index_sequence<OrigRank>());
    return result;
  }

  static constexpr std::array<int, OrigRank> value = get();
};

/// A stateful layout mapping for a subview of a tiled mapping.
///
/// Destination indices are expanded back into source indices using a stored
/// origin and stride for every source dimension. The source mapping result is
/// then normalized by the subview's base offset.
template <class DstExtents, class OrigMapping, class... Slices>
class sub_tiled_mapping {
  static constexpr size_t orig_rank = OrigMapping::extents_type::rank();
  static constexpr size_t sub_rank  = DstExtents::rank();

  static_assert(orig_rank == sizeof...(Slices));
  static_assert(sub_rank == sub_rank_from_slices_v<OrigMapping, Slices...>);

  static constexpr std::array<int, orig_rank> s_sub_idx_pos
    = sub_idx_pos_helper<orig_rank, typename OrigMapping::index_type, Slices...>::value;

public:
  using extents_type = DstExtents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = sub_tiled_layout<OrigMapping, Slices...>;

  static_assert(std::is_same_v<typename layout_type::template mapping<extents_type>, sub_tiled_mapping>);

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr sub_tiled_mapping(const sub_tiled_mapping&) noexcept = default;
  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto operator=(const sub_tiled_mapping&) noexcept -> sub_tiled_mapping& = default;
  LAYOUT_TILED_INLINE_FUNCTION
  constexpr sub_tiled_mapping(sub_tiled_mapping&&) noexcept = default;
  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto operator=(sub_tiled_mapping&&) noexcept -> sub_tiled_mapping& = default;
  LAYOUT_TILED_INLINE_FUNCTION
  constexpr ~sub_tiled_mapping() noexcept = default;

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto extents() const noexcept -> const extents_type& {
    return m_extents;
  }

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto required_span_size() const noexcept -> index_type {
    if (empty()) {
      return 0;
    }
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
      return m_orig(last_orig_idx<Is>()...) - m_base_offset + 1;
    }(std::make_index_sequence<orig_rank>());
  }

  template <class... SubIdxs>
    requires (sub_rank == sizeof...(SubIdxs)) && (std::is_convertible_v<SubIdxs, index_type> && ...)
  LAYOUT_TILED_INLINE_FUNCTION constexpr auto operator()(SubIdxs... sub_idxs) const noexcept -> index_type {
    assert(
      "layout_tiled::detail::sub_tiled_mapping: out of bounds indexing"
      && is_md_idx_in_span(std::make_index_sequence<sub_rank>(), m_extents, sub_idxs...)
    );
    const auto sub_arr = std::array<index_type, sub_rank>{static_cast<index_type>(sub_idxs)...};
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
      return m_orig(expanded_idx<Is>(sub_arr)...) - m_base_offset;
    }(std::make_index_sequence<orig_rank>());
  }

  LAYOUT_TILED_INLINE_FUNCTION
  static constexpr auto is_always_unique() noexcept -> bool {
    return OrigMapping::is_always_unique();
  }

  LAYOUT_TILED_INLINE_FUNCTION
  static constexpr auto is_always_exhaustive() noexcept -> bool {
    return sub_rank == 0 || static_domain_is_empty()
           || (OrigMapping::is_always_exhaustive() && (is_full_extent_slice_v<Slices> && ...));
  }

  LAYOUT_TILED_INLINE_FUNCTION
  static constexpr auto is_always_strided() noexcept -> bool {
    return false;
  }

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto is_unique() const noexcept -> bool {
    return m_orig.is_unique();
  }

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto is_exhaustive() const noexcept -> bool {
    return required_span_size() == domain_size();
  }

  LAYOUT_TILED_INLINE_FUNCTION
  static constexpr auto is_strided() noexcept -> bool {
    return false;
  }

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto stride(rank_type) const noexcept -> index_type {
    return 0;
  }

  LAYOUT_TILED_INLINE_FUNCTION
  friend constexpr auto operator==(const sub_tiled_mapping& lhs, const sub_tiled_mapping& rhs) noexcept -> bool {
    if (lhs.m_orig != rhs.m_orig || lhs.m_extents != rhs.m_extents || lhs.m_base_offset != rhs.m_base_offset) {
      return false;
    }
    for (size_t r = 0; r < orig_rank; ++r) {
      if (lhs.m_origins[r] != rhs.m_origins[r] || lhs.m_strides[r] != rhs.m_strides[r]) {
        return false;
      }
    }
    return true;
  }

private:
  template <class Mapping, class... SliceSpecifiers>
    requires (sizeof...(SliceSpecifiers) == Mapping::extents_type::rank())
  friend LAYOUT_TILED_INLINE_FUNCTION constexpr auto make_sub_tiled_mapping(const Mapping&, SliceSpecifiers...)
    -> sub_tiled_mapping_result_t<Mapping, SliceSpecifiers...>;

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr sub_tiled_mapping(
    const OrigMapping& orig,
    const DstExtents& dst_ext,
    std::array<index_type, orig_rank> origins,
    std::array<index_type, orig_rank> strides,
    index_type base_offset
  ) noexcept :
      m_orig(orig),
      m_extents(dst_ext),
      m_origins(origins),
      m_strides(strides),
      m_base_offset(base_offset) {}

  LAYOUT_TILED_INLINE_FUNCTION
  static constexpr auto static_domain_is_empty() noexcept -> bool {
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return ((extents_type::static_extent(Is) == 0) || ...);
    }(std::make_index_sequence<sub_rank>());
  }

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto empty() const noexcept -> bool {
    for (rank_type r = 0; r < sub_rank; ++r) {
      if (m_extents.extent(r) == 0) {
        return true;
      }
    }
    return false;
  }

  LAYOUT_TILED_INLINE_FUNCTION
  constexpr auto domain_size() const noexcept -> index_type {
    if (empty()) {
      return 0;
    }
    index_type result = 1;
    for (rank_type r = 0; r < sub_rank; ++r) {
      result *= m_extents.extent(r);
    }
    return result;
  }

  template <size_t Dim>
  LAYOUT_TILED_INLINE_FUNCTION constexpr auto
  expanded_idx(const std::array<index_type, sub_rank>& sub_arr) const noexcept -> index_type {
    constexpr int pos = s_sub_idx_pos[Dim];
    if constexpr (pos >= 0) {
      return m_origins[Dim] + sub_arr[static_cast<size_t>(pos)] * m_strides[Dim];
    } else {
      return m_origins[Dim];
    }
  }

  template <size_t Dim>
  LAYOUT_TILED_INLINE_FUNCTION constexpr auto last_orig_idx() const noexcept -> index_type {
    constexpr int pos = s_sub_idx_pos[Dim];
    if constexpr (pos >= 0) {
      return m_origins[Dim] + (m_extents.extent(static_cast<rank_type>(pos)) - 1) * m_strides[Dim];
    } else {
      return m_origins[Dim];
    }
  }

  OrigMapping m_orig;
  DstExtents m_extents;
  std::array<index_type, orig_rank> m_origins;
  std::array<index_type, orig_rank> m_strides;
  index_type m_base_offset;
};

template <class Mapping, class... Slices>
  requires (sizeof...(Slices) == Mapping::extents_type::rank())
LAYOUT_TILED_INLINE_FUNCTION constexpr auto make_sub_tiled_mapping(const Mapping& src, Slices... slices)
  -> sub_tiled_mapping_result_t<Mapping, Slices...> {
  using orig_extents_t = typename Mapping::extents_type;
  using index_type     = typename Mapping::index_type;
  constexpr size_t N   = orig_extents_t::rank();

  auto dst_ext = std::submdspan_extents(src.extents(), slices...);

  const bool base_is_end = any_slice_out_of_bounds(src.extents(), slices...);
  const auto base_offset
    = static_cast<index_type>(base_is_end ? src.required_span_size() : src(first_of<index_type>(slices)...));
  const auto origins = std::array<index_type, N>{first_of<index_type>(slices)...};
  const auto strides = std::array<index_type, N>{stride_of<index_type>(slices)...};

  using dst_extents_t    = decltype(dst_ext);
  using result_mapping_t = sub_tiled_mapping<dst_extents_t, Mapping, Slices...>;
  return std::submdspan_mapping_result<result_mapping_t>{
    result_mapping_t(src, dst_ext, origins, strides, base_offset),
    static_cast<size_t>(base_offset)
  };
}

/// Make subview mappings recursively sliceable through ADL.
template <class DstExtents, class OrigMapping, class... OrigSlices, class... Slices>
  requires (sizeof...(Slices) == DstExtents::rank())
LAYOUT_TILED_INLINE_FUNCTION constexpr auto
submdspan_mapping(const sub_tiled_mapping<DstExtents, OrigMapping, OrigSlices...>& src, Slices... slices) {
  return make_sub_tiled_mapping(src, slices...);
}

} // namespace layout_tiled::detail
#endif // defined(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN)

#endif // INCLUDED_LAYOUT_TILED_DETAIL_SUB_TILED_MAPPING
