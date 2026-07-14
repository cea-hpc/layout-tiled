// include/layout_tiled/detail/common.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_DETAIL_COMMON
#define INCLUDED_LAYOUT_TILED_DETAIL_COMMON

#include <array>
#include <complex>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "macros.hpp"

namespace layout_tiled::detail {

template <std::integral IdxType, class From>
  requires std::integral<From>
LAYOUT_TILED_IMPL_HOST_DEVICE constexpr auto is_idx_in_extent(IdxType extent, From value) -> bool {
  if constexpr (std::is_signed_v<From>) {
    if (value < 0) {
      return false;
    }
  }
  using Tp = std::common_type_t<IdxType, From>;
  return static_cast<Tp>(value) < static_cast<Tp>(extent);
}

template <size_t... Idxs, class Extents, class... From>
LAYOUT_TILED_INLINE_FUNCTION constexpr auto
is_md_idx_in_span([[maybe_unused]] std::index_sequence<Idxs...> _, const Extents& ext, From... values) -> bool {
  return (is_idx_in_extent(ext.extent(Idxs), static_cast<typename Extents::index_type>(values)) && ...);
}

/// If `i > 0`, computes the product of extents up to `i` for all `k` in the range [0, i), otherwise 1.
///
/// Pre-condition: `i < sizeof...(Exts)`
template <class E, E... Exts>
LAYOUT_TILED_IMPL_HOST_DEVICE constexpr auto fwd_prod_of_extents(E i) noexcept -> E {
  if (0 == i) {
    return 1;
  }
  constexpr std::array<E, sizeof...(Exts)> extents{Exts...};
  E product = 1;
  for (E k = 0; k < i; ++k) {
    product *= extents[k];
  }
  return product;
}

/// If `i < sizeof...(Exts)`, computes the product of extents up to `i` for all `k` in the range
/// [i+1, sizeof...(Exts)), otherwise 1.
///
/// Pre-condition: `i < sizeof...(Exts)`
template <class E, E... Exts>
LAYOUT_TILED_IMPL_HOST_DEVICE constexpr auto rev_prod_of_extents(E i) noexcept -> E {
  if (sizeof...(Exts) < i + 1) {
    return 1;
  }
  constexpr std::array<E, sizeof...(Exts)> extents{Exts...};
  E product = 1;
  for (E k = i + 1; k < sizeof...(Exts); ++k) {
    product *= extents[k];
  }
  return product;
}

#if defined(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN)
template <class T>
struct is_strided_slice : std::false_type {};

template <class OffsetType, class ExtentType, class StrideType>
struct is_strided_slice<std::strided_slice<OffsetType, ExtentType, StrideType>> : std::true_type {};

template <class T>
inline constexpr bool is_strided_slice_v = is_strided_slice<std::remove_cvref_t<T>>::value;

template <class T>
inline constexpr bool is_full_extent_slice_v = std::is_convertible_v<T, std::full_extent_t>;

template <class T, class IndexType>
inline constexpr bool is_index_slice_v = !is_full_extent_slice_v<T> && std::is_convertible_v<T, IndexType>;

template <size_t I, class T>
LAYOUT_TILED_INLINE_FUNCTION constexpr auto slice_get(const T& value) noexcept {
  using std::get;
  return get<I>(value);
}

template <size_t I, class T>
  requires (I < 2)
LAYOUT_TILED_INLINE_FUNCTION constexpr auto slice_get(const std::complex<T>& value) noexcept -> T {
  if constexpr (I == 0) {
    return value.real();
  } else {
    return value.imag();
  }
}

/// Returns the lower bound of a slice specifier.
///
/// For `full_extent_t` this is zero, for an index slice it is the index,
/// and for a range or strided slice it is the range offset.
template <class IndexType, class T>
LAYOUT_TILED_INLINE_FUNCTION constexpr auto first_of(const T& s) noexcept -> IndexType {
  if constexpr (is_full_extent_slice_v<T>) {
    return IndexType{0};
  } else if constexpr (is_index_slice_v<T, IndexType>) {
    return static_cast<IndexType>(s);
  } else if constexpr (is_strided_slice_v<T>) {
    return static_cast<IndexType>(s.offset);
  } else {
    return static_cast<IndexType>(slice_get<0>(s));
  }
}

/// Returns the multiplier used to expand a destination index.
template <class IndexType, class T>
LAYOUT_TILED_INLINE_FUNCTION constexpr auto stride_of(const T& s) noexcept -> IndexType {
  if constexpr (is_strided_slice_v<T>) {
    return static_cast<IndexType>(s.stride);
  } else {
    return IndexType{1};
  }
}

/// True if any slice's lower bound meets or exceeds the corresponding extent.
///
/// Per LWG 4060: when out of bounds, `submdspan_mapping` must return
/// `required_span_size()` as the offset instead of evaluating the mapping.
template <class Extents, class... Slices>
LAYOUT_TILED_INLINE_FUNCTION constexpr auto
any_slice_out_of_bounds(const Extents& exts, const Slices&... slices) noexcept -> bool {
  return [&]<size_t... Is>(std::index_sequence<Is...>) {
    return (... || (first_of<typename Extents::index_type>(slices) >= exts.extent(Is)));
  }(std::make_index_sequence<sizeof...(Slices)>());
}
#endif

} // namespace layout_tiled::detail

#endif // INCLUDED_LAYOUT_TILED_DETAIL_COMMON
