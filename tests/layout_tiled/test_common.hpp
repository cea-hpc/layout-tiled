// tests/layout_tiled/test_common.hpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDED_LAYOUT_TILED_TESTS_COMMON
#define INCLUDED_LAYOUT_TILED_TESTS_COMMON

#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

// --- Types for testing operator() convertibility constraints ---

struct IntType {
  int val;
  constexpr IntType() = default;
  constexpr IntType(int v) noexcept : val(v) {}
  constexpr operator int() const noexcept {
    return val;
  }
};

struct IntTypeNC {
  int val;
  constexpr IntTypeNC() = default;
  constexpr IntTypeNC(int v) noexcept : val(v) {}
  constexpr operator int() noexcept {
    return val;
  }
};

// --- SFINAE helpers to check if mapping(idxs...) is valid ---

template <class Mapping, class... Idxs>
concept operator_constraints = requires(Mapping m, Idxs... idxs) {
  { m(idxs...) } -> std::same_as<typename Mapping::index_type>;
};

template <class Mapping, class... Idxs>
  requires operator_constraints<Mapping, Idxs...>
constexpr auto check_operator_constraints(Mapping, Idxs...) -> bool {
  return true;
}

template <class Mapping, class... Idxs>
constexpr auto check_operator_constraints(Mapping, Idxs...) -> bool {
  return false;
}

// --- Manual reference computations for each layout variant ---

template <size_t... TileExtents>
constexpr auto
ltl_offset(std::array<size_t, sizeof...(TileExtents)> extents, std::array<size_t, sizeof...(TileExtents)> idx)
  -> size_t {
  constexpr size_t R     = sizeof...(TileExtents);
  constexpr auto textent = std::array<size_t, R>{TileExtents...};

  size_t outer_offset = 0;
  size_t outer_stride = 1;
  for (size_t i = 0; i < R; ++i) {
    outer_offset += (idx[i] / textent[i]) * outer_stride;
    outer_stride *= extents[i] / textent[i];
  }

  size_t inner_offset = 0;
  size_t inner_stride = 1;
  for (size_t i = 0; i < R; ++i) {
    inner_offset += (idx[i] % textent[i]) * inner_stride;
    inner_stride *= textent[i];
  }

  size_t tile_size = 1;
  for (auto t : textent)
    tile_size *= t;

  return outer_offset * tile_size + inner_offset;
}

template <size_t... TileExtents>
constexpr auto
ltr_offset(std::array<size_t, sizeof...(TileExtents)> extents, std::array<size_t, sizeof...(TileExtents)> idx)
  -> size_t {
  constexpr size_t R     = sizeof...(TileExtents);
  constexpr auto textent = std::array<size_t, R>{TileExtents...};

  size_t outer_offset = 0;
  size_t outer_stride = 1;
  for (size_t i = 0; i < R; ++i) {
    outer_offset += (idx[i] / textent[i]) * outer_stride;
    outer_stride *= extents[i] / textent[i];
  }

  size_t inner_offset = 0;
  size_t inner_stride = 1;
  for (size_t i = R; i > 0; --i) {
    inner_offset += (idx[i - 1] % textent[i - 1]) * inner_stride;
    inner_stride *= textent[i - 1];
  }

  size_t tile_size = 1;
  for (auto t : textent)
    tile_size *= t;

  return outer_offset * tile_size + inner_offset;
}

template <size_t... TileExtents>
constexpr auto
rtl_offset(std::array<size_t, sizeof...(TileExtents)> extents, std::array<size_t, sizeof...(TileExtents)> idx)
  -> size_t {
  constexpr size_t R     = sizeof...(TileExtents);
  constexpr auto textent = std::array<size_t, R>{TileExtents...};

  size_t outer_offset = 0;
  size_t outer_stride = 1;
  for (size_t i = R; i > 0; --i) {
    outer_offset += (idx[i - 1] / textent[i - 1]) * outer_stride;
    outer_stride *= extents[i - 1] / textent[i - 1];
  }

  size_t inner_offset = 0;
  size_t inner_stride = 1;
  for (size_t i = 0; i < R; ++i) {
    inner_offset += (idx[i] % textent[i]) * inner_stride;
    inner_stride *= textent[i];
  }

  size_t tile_size = 1;
  for (auto t : textent)
    tile_size *= t;

  return outer_offset * tile_size + inner_offset;
}

template <size_t... TileExtents>
constexpr auto
rtr_offset(std::array<size_t, sizeof...(TileExtents)> extents, std::array<size_t, sizeof...(TileExtents)> idx)
  -> size_t {
  constexpr size_t R     = sizeof...(TileExtents);
  constexpr auto textent = std::array<size_t, R>{TileExtents...};

  size_t outer_offset = 0;
  size_t outer_stride = 1;
  for (size_t i = R; i > 0; --i) {
    outer_offset += (idx[i - 1] / textent[i - 1]) * outer_stride;
    outer_stride *= extents[i - 1] / textent[i - 1];
  }

  size_t inner_offset = 0;
  size_t inner_stride = 1;
  for (size_t i = R; i > 0; --i) {
    inner_offset += (idx[i - 1] % textent[i - 1]) * inner_stride;
    inner_stride *= textent[i - 1];
  }

  size_t tile_size = 1;
  for (auto t : textent)
    tile_size *= t;

  return outer_offset * tile_size + inner_offset;
}

// --- Generic helpers ---

/// Computes the product of all elements in the array.
template <size_t N>
constexpr auto prod(std::array<size_t, N> arr) -> size_t {
  size_t p = 1;
  for (auto x : arr)
    p *= x;
  return p;
}

template <class Layout>
constexpr auto base_mapping_is_constexpr() -> bool {
  using extents_t = std::dextents<int, 2>;
  using mapping_t = typename Layout::template mapping<extents_t>;

  const mapping_t mapping(extents_t(6, 12));
  return mapping.extents() == extents_t(6, 12) && mapping.required_span_size() == 72
         && mapping(IntType(5), IntType(11)) == mapping(5, 11) && mapping == mapping
         && mapping_t::textents() == std::array<int, 2>{2, 3} && mapping_t::tile_size() == 6;
}

template <class Layout>
void check_mdspan_end_to_end() {
  using extents_t = std::dextents<int, 2>;
  using mapping_t = typename Layout::template mapping<extents_t>;
  using mdspan_t  = std::mdspan<int, extents_t, Layout>;

  std::array<int, 72> storage{};
  std::array<bool, 72> seen{};
  const mapping_t mapping(extents_t(6, 12));
  const mdspan_t view(storage.data(), mapping);

  for (int i = 0; i < view.extent(0); ++i) {
    for (int j = 0; j < view.extent(1); ++j) {
      const int offset = mapping(IntType(i), IntType(j));
      ASSERT_GE(offset, 0);
      ASSERT_LT(static_cast<size_t>(offset), storage.size());
      EXPECT_FALSE(seen[static_cast<size_t>(offset)]);
      seen[static_cast<size_t>(offset)] = true;

      storage[static_cast<size_t>(offset)] = 1000 + i * view.extent(1) + j;
      EXPECT_EQ(&(view[i, j]), storage.data() + offset);
      EXPECT_EQ((view[i, j]), 1000 + i * view.extent(1) + j);
    }
  }

  for (const bool visited : seen) {
    EXPECT_TRUE(visited);
  }
}

template <class Layout>
void check_mapping_conversions() {
  using dynamic_extents_t      = std::dextents<int, 2>;
  using long_extents_t         = std::dextents<long long, 2>;
  using static_extents_t       = std::extents<int, 4, 6>;
  using mixed_extents_t        = std::extents<int, 4, std::dynamic_extent>;
  using incompatible_extents_t = std::extents<int, 8, 6>;

  using dynamic_mapping_t      = typename Layout::template mapping<dynamic_extents_t>;
  using long_mapping_t         = typename Layout::template mapping<long_extents_t>;
  using static_mapping_t       = typename Layout::template mapping<static_extents_t>;
  using mixed_mapping_t        = typename Layout::template mapping<mixed_extents_t>;
  using incompatible_mapping_t = typename Layout::template mapping<incompatible_extents_t>;

  static_assert(std::is_constructible_v<dynamic_mapping_t, static_mapping_t>);
  static_assert(std::is_convertible_v<static_mapping_t, dynamic_mapping_t>);
  static_assert(std::is_constructible_v<static_mapping_t, dynamic_mapping_t>);
  static_assert(!std::is_convertible_v<dynamic_mapping_t, static_mapping_t>);
  static_assert(std::is_constructible_v<mixed_mapping_t, static_mapping_t>);
  static_assert(!std::is_constructible_v<static_mapping_t, incompatible_mapping_t>);
  static_assert(std::is_constructible_v<dynamic_mapping_t, long_mapping_t>);

  const static_mapping_t static_mapping;
  const dynamic_mapping_t dynamic_mapping(static_mapping);
  const mixed_mapping_t mixed_mapping(static_mapping);
  const static_mapping_t static_from_dynamic(dynamic_mapping);
  const long_mapping_t long_mapping(long_extents_t(4, 6));
  const dynamic_mapping_t narrowed_mapping(long_mapping);

  EXPECT_EQ(dynamic_mapping.extents(), dynamic_extents_t(4, 6));
  EXPECT_EQ(mixed_mapping.extents(), mixed_extents_t(6));
  EXPECT_EQ(static_from_dynamic.extents(), static_extents_t());
  EXPECT_EQ(narrowed_mapping.extents(), dynamic_extents_t(4, 6));
  EXPECT_EQ(dynamic_mapping, static_mapping);
  EXPECT_EQ(static_mapping, dynamic_mapping);
  EXPECT_EQ(narrowed_mapping(3, 5), long_mapping(3, 5));
}

#if defined(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN)
struct FullExtentConvertible {
  constexpr operator std::full_extent_t() const noexcept {
    return std::full_extent;
  }
};

struct IntOnlyIndex {
  int value;

  template <class T>
    requires std::same_as<T, int>
  constexpr operator T() const noexcept {
    return value;
  }
};

template <class Layout>
constexpr auto dynamic_submdspan_is_constexpr() -> bool {
  using extents_t = std::dextents<int, 2>;
  using mapping_t = typename Layout::template mapping<extents_t>;

  const mapping_t mapping(extents_t(4, 6));
  const auto result = submdspan_mapping(mapping, std::pair<int, int>{1, 3}, std::full_extent);
  return result.offset == static_cast<size_t>(mapping(1, 0)) && result.mapping.extents().extent(0) == 2
         && result.mapping.extents().extent(1) == 6
         && result.mapping(1, 5) == mapping(2, 5) - static_cast<int>(result.offset);
}

template <class Layout>
void check_submdspan_end_to_end() {
  using extents_t = std::dextents<size_t, 2>;
  using mapping_t = typename Layout::template mapping<extents_t>;
  using mdspan_t  = std::mdspan<int, extents_t, Layout>;

  std::array<int, 24> storage{};
  for (size_t i = 0; i < storage.size(); ++i) {
    storage[i] = static_cast<int>(i);
  }

  const mapping_t mapping(extents_t(4, 6));
  const mdspan_t src(storage.data(), mapping);

  const auto full      = std::submdspan(src, std::full_extent, std::full_extent);
  using full_mapping_t = std::remove_cvref_t<decltype(full.mapping())>;
  using state_array_t  = std::array<typename full_mapping_t::index_type, mapping_t::extents_type::rank()>;
  static_assert(std::is_same_v<
                full_mapping_t,
                typename full_mapping_t::layout_type::template mapping<typename full_mapping_t::extents_type>>);
  static_assert(!std::is_constructible_v<
                full_mapping_t,
                const mapping_t&,
                const typename full_mapping_t::extents_type&,
                state_array_t,
                state_array_t,
                typename full_mapping_t::index_type>);
  static_assert(full_mapping_t::is_always_unique());
  static_assert(full_mapping_t::is_always_exhaustive());
  static_assert(!full_mapping_t::is_always_strided());
  EXPECT_EQ(full.extent(0), 4);
  EXPECT_EQ(full.extent(1), 6);
  EXPECT_EQ((&full[3, 5]), (&src[3, 5]));
  EXPECT_TRUE(full.mapping().is_unique());
  EXPECT_TRUE(full.mapping().is_exhaustive());
  EXPECT_FALSE(full.mapping().is_strided());

  const auto converted_full      = std::submdspan(src, FullExtentConvertible{}, std::full_extent);
  using converted_full_mapping_t = std::remove_cvref_t<decltype(converted_full.mapping())>;
  static_assert(converted_full_mapping_t::is_always_exhaustive());
  EXPECT_EQ(converted_full.extent(0), 4);
  EXPECT_EQ(converted_full.extent(1), 6);
  EXPECT_EQ((&converted_full[3, 5]), (&src[3, 5]));

  const auto complex_range = std::submdspan(src, std::complex<size_t>{1, 3}, std::full_extent);
  EXPECT_EQ(complex_range.extent(0), 2);
  EXPECT_EQ(complex_range.extent(1), 6);
  for (size_t i = 0; i < complex_range.extent(0); ++i) {
    for (size_t j = 0; j < complex_range.extent(1); ++j) {
      EXPECT_EQ((&complex_range[i, j]), (&src[1 + i, j]));
    }
  }

  const auto indexed = std::submdspan(src, std::integral_constant<size_t, 1>{}, std::full_extent);
  EXPECT_EQ(indexed.rank(), 1);
  EXPECT_EQ(indexed.extent(0), 6);
  for (size_t j = 0; j < indexed.extent(0); ++j) {
    EXPECT_EQ((&indexed[j]), (&src[1, j]));
  }

  const auto strided = std::submdspan(
    src,
    std::strided_slice<size_t, size_t, size_t>{0, 4, 2},
    std::strided_slice<size_t, size_t, size_t>{1, 5, 2}
  );
  EXPECT_EQ(strided.extent(0), 2);
  EXPECT_EQ(strided.extent(1), 3);
  for (size_t i = 0; i < strided.extent(0); ++i) {
    for (size_t j = 0; j < strided.extent(1); ++j) {
      EXPECT_EQ((&strided[i, j]), (&src[2 * i, 1 + 2 * j]));
    }
  }

  const auto scalar = std::submdspan(src, size_t(2), size_t(4));
  static_assert(decltype(scalar)::rank() == 0);
  EXPECT_EQ((&scalar[]), (&src[2, 4]));
  EXPECT_EQ(scalar.mapping().required_span_size(), 1);
  EXPECT_TRUE(scalar.mapping().is_exhaustive());
  static_assert(std::remove_cvref_t<decltype(scalar.mapping())>::is_always_exhaustive());

  const auto nested_scalar = std::submdspan(scalar);
  static_assert(decltype(nested_scalar)::rank() == 0);
  EXPECT_EQ((&nested_scalar[]), (&src[2, 4]));
  EXPECT_EQ(nested_scalar.mapping().required_span_size(), 1);

  const auto empty = std::submdspan(src, std::pair<size_t, size_t>{1, 1}, std::full_extent);
  EXPECT_EQ(empty.extent(0), 0);
  EXPECT_EQ(empty.mapping().required_span_size(), 0);
  EXPECT_TRUE(empty.mapping().is_exhaustive());

  const auto static_empty = std::submdspan(
    src,
    std::strided_slice<size_t, std::integral_constant<size_t, 0>, std::integral_constant<size_t, 0>>{0, {}, {}},
    std::full_extent
  );
  static_assert(std::remove_cvref_t<decltype(static_empty.mapping())>::is_always_exhaustive());
  EXPECT_EQ(static_empty.mapping().required_span_size(), 0);

  const auto end_empty = std::submdspan(src, std::pair<size_t, size_t>{4, 4}, std::full_extent);
  EXPECT_EQ(end_empty.mapping().required_span_size(), 0);
  EXPECT_EQ(end_empty.data_handle(), storage.data() + mapping.required_span_size());

  const auto outer = std::submdspan(src, std::pair<size_t, size_t>{0, 4}, std::pair<size_t, size_t>{1, 6});
  const auto nested
    = std::submdspan(outer, std::strided_slice<size_t, size_t, size_t>{0, 4, 2}, std::integral_constant<size_t, 2>{});
  EXPECT_EQ(nested.extent(0), 2);
  EXPECT_EQ((&nested[0]), (&src[0, 3]));
  EXPECT_EQ((&nested[1]), (&src[2, 3]));

  const auto left  = submdspan_mapping(mapping, std::pair<size_t, size_t>{0, 3}, std::full_extent);
  const auto right = submdspan_mapping(mapping, std::pair<size_t, size_t>{1, 4}, std::full_extent);
  EXPECT_NE(left.mapping, right.mapping);

  const auto complete = submdspan_mapping(mapping, std::pair<size_t, size_t>{0, 4}, std::pair<size_t, size_t>{0, 6});
  EXPECT_TRUE(complete.mapping.is_exhaustive());

  using int_extents_t = std::dextents<int, 2>;
  using int_mapping_t = typename Layout::template mapping<int_extents_t>;
  using int_mdspan_t  = std::mdspan<int, int_extents_t, Layout>;
  const int_mapping_t int_mapping(int_extents_t(4, 6));
  const int_mdspan_t int_src(storage.data(), int_mapping);
  static_assert(std::is_same_v<decltype(int_mapping_t::textents()), std::array<int, 2>>);
  static_assert(layout_tiled::detail::is_index_slice_v<IntOnlyIndex, int>);
  static_assert(!layout_tiled::detail::is_index_slice_v<IntOnlyIndex, size_t>);
  static_assert(layout_tiled::detail::sub_rank_from_slices_v<int_mapping_t, IntOnlyIndex, std::full_extent_t> == 1);
  static_assert(layout_tiled::detail::first_of<int>(IntOnlyIndex{1}) == 1);
  EXPECT_EQ(int_mapping_t::textents(), (std::array<int, 2>{2, 3}));
  EXPECT_EQ(int_mapping_t::tile_size(), 6);

  const auto int_indexed = std::submdspan(int_src, int{1}, std::full_extent);
  static_assert(decltype(int_indexed)::rank() == 1);
  EXPECT_EQ(int_indexed.extent(0), 6);
  for (int j = 0; j < int_indexed.extent(0); ++j) {
    EXPECT_EQ((&int_indexed[j]), (&int_src[1, j]));
  }
  EXPECT_EQ(int_indexed.mapping()(IntType(5)), int_indexed.mapping()(5));

  static_assert(dynamic_submdspan_is_constexpr<Layout>());
}
#endif

#endif // INCLUDED_LAYOUT_TILED_TESTS_COMMON
