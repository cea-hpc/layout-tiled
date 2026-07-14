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

#endif // INCLUDED_LAYOUT_TILED_TESTS_COMMON
