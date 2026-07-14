// tests/layout_tiled/layout_rtr.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <concepts>
#if defined(LAYOUT_TILED_KOKKOS_MDSPAN)
#include <experimental/mdspan>
#else
#include <mdspan>
#endif
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <layout_tiled/layout_rtr.hpp>

#include "test_common.hpp"

namespace {

using layout_t  = layout_tiled::layout_rtr<2, 3>;
using extents_t = std::dextents<size_t, 2>;
using mapping_t = layout_t::mapping<extents_t>;

} // namespace

static_assert(base_mapping_is_constexpr<layout_t>());

TEST(layout_rtr, static_requirements) {
  ASSERT_TRUE((std::is_same_v<mapping_t::extents_type, extents_t>));
  ASSERT_TRUE((std::is_same_v<mapping_t::index_type, extents_t::index_type>));
  ASSERT_TRUE((std::is_same_v<mapping_t::size_type, extents_t::size_type>));
  ASSERT_TRUE((std::is_same_v<mapping_t::rank_type, extents_t::rank_type>));
  ASSERT_TRUE((std::is_same_v<mapping_t::layout_type, layout_t>));
  ASSERT_TRUE((std::is_same_v<mapping_t::layout_type::template mapping<extents_t>, mapping_t>));

  ASSERT_TRUE(std::is_copy_constructible_v<mapping_t>);
  ASSERT_TRUE(std::is_nothrow_move_constructible_v<mapping_t>);
  ASSERT_TRUE(std::is_nothrow_move_assignable_v<mapping_t>);
  ASSERT_TRUE(std::is_nothrow_swappable_v<mapping_t>);

  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>().extents()), const extents_t&>));
  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>().required_span_size()), size_t>));
  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>().is_unique()), bool>));
  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>().is_exhaustive()), bool>));
  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>().is_strided()), bool>));
  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>().stride(0)), size_t>));
  ASSERT_TRUE((std::is_same_v<decltype(mapping_t::is_always_unique()), bool>));
  ASSERT_TRUE((std::is_same_v<decltype(mapping_t::is_always_exhaustive()), bool>));
  ASSERT_TRUE((std::is_same_v<decltype(mapping_t::is_always_strided()), bool>));
  ASSERT_TRUE((std::is_same_v<decltype(std::declval<const mapping_t&>()(size_t(0), size_t(0))), size_t>));

  ASSERT_TRUE(noexcept(std::declval<const mapping_t&>().required_span_size()));
  ASSERT_TRUE(noexcept(std::declval<const mapping_t&>().is_unique()));
  ASSERT_TRUE(noexcept(std::declval<const mapping_t&>().is_exhaustive()));
  ASSERT_TRUE(noexcept(std::declval<const mapping_t&>().is_strided()));
  ASSERT_TRUE(noexcept(std::declval<const mapping_t&>().stride(0)));
  ASSERT_TRUE(noexcept(mapping_t::is_always_unique()));
  ASSERT_TRUE(noexcept(mapping_t::is_always_exhaustive()));
  ASSERT_TRUE(noexcept(mapping_t::is_always_strided()));
}

TEST(layout_rtr, default_construction) {
  mapping_t m;
  EXPECT_EQ(m.extents(), extents_t());
  EXPECT_EQ(m.required_span_size(), 0);
}

TEST(layout_rtr, construction_from_extents) {
  auto exts = extents_t(4, 6);
  mapping_t m(exts);
  EXPECT_EQ(m.extents(), exts);
  EXPECT_EQ(m.required_span_size(), 24);
}

TEST(layout_rtr, mdspan_end_to_end) {
  check_mdspan_end_to_end<layout_t>();
}

TEST(layout_rtr, properties) {
  EXPECT_TRUE(mapping_t::is_always_unique());
  EXPECT_TRUE(mapping_t::is_always_exhaustive());
  EXPECT_FALSE(mapping_t::is_always_strided());

  mapping_t m(extents_t(4, 6));
  EXPECT_TRUE(m.is_unique());
  EXPECT_TRUE(m.is_exhaustive());
  EXPECT_FALSE(m.is_strided());
}

TEST(layout_rtr, equality) {
  mapping_t a(extents_t(4, 6));
  mapping_t b(extents_t(4, 6));
  mapping_t c(extents_t(8, 6));

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(layout_rtr, index_operator_2d) {
  constexpr size_t ROWS = 4, COLS = 6;
  auto exts = extents_t(ROWS, COLS);
  mapping_t m(exts);

  for (size_t i = 0; i < ROWS; ++i) {
    for (size_t j = 0; j < COLS; ++j) {
      auto expected = rtr_offset<2, 3>({ROWS, COLS}, {i, j});
      EXPECT_EQ(m(i, j), expected);
    }
  }
}

TEST(layout_rtr, index_operator_3d) {
  using extents3d_t = std::dextents<size_t, 3>;
  using mapping3d_t = layout_tiled::layout_rtr<2, 3, 4>::mapping<extents3d_t>;

  auto exts = extents3d_t(4, 6, 8);
  mapping3d_t m(exts);

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      for (size_t k = 0; k < 8; ++k) {
        auto expected = rtr_offset<2, 3, 4>({4, 6, 8}, {i, j, k});
        EXPECT_EQ(m(i, j, k), expected);
      }
    }
  }
}

TEST(layout_rtr, index_operator_1d) {
  using extents1d_t = std::dextents<size_t, 1>;
  using mapping1d_t = layout_tiled::layout_rtr<4>::mapping<extents1d_t>;

  auto exts = extents1d_t(12);
  mapping1d_t m(exts);

  for (size_t i = 0; i < 12; ++i) {
    auto expected = rtr_offset<4>({12}, {i});
    EXPECT_EQ(m(i), expected);
  }
}

TEST(layout_rtr, index_operator_constraints) {
  mapping_t m(extents_t(4, 6));

  EXPECT_TRUE(check_operator_constraints(m, size_t(0), size_t(0)));
  EXPECT_FALSE(check_operator_constraints(m, size_t(0)));
  EXPECT_FALSE(check_operator_constraints(m, size_t(0), size_t(0), size_t(0)));
  EXPECT_TRUE(check_operator_constraints(m, IntType(0), IntType(0)));
  EXPECT_EQ(m(IntType(1), IntType(2)), m(size_t(1), size_t(2)));
  EXPECT_EQ(m(IntTypeNC(1), IntTypeNC(2)), m(size_t(1), size_t(2)));
}

TEST(layout_rtr, stride) {
  mapping_t m(extents_t(4, 6));
  // layout_rtr: stride(r) = rev_prod_of_extents<TileExtents...>(r)
  EXPECT_EQ(m.stride(0), 3);
  EXPECT_EQ(m.stride(1), 1);
}

TEST(layout_rtr, helpers) {
  EXPECT_EQ((mapping_t::textents()), (std::array<size_t, 2>{2, 3}));
  EXPECT_EQ(mapping_t::textent(0), 2);
  EXPECT_EQ(mapping_t::textent(1), 3);
  EXPECT_EQ(mapping_t::tile_size(), 6);

  mapping_t m(extents_t(4, 6));
  EXPECT_EQ(m.ntiles_in_extent(0), 2);
  EXPECT_EQ(m.ntiles_in_extent(1), 2);
}

// ---------------------------------------------------------------------------
// Converting constructor from same layout with different extents
// ---------------------------------------------------------------------------

using rtr_static_mapping_t = layout_tiled::layout_rtr<2, 3>::mapping<std::extents<size_t, 4, 6>>;

static_assert(std::is_constructible_v<mapping_t, rtr_static_mapping_t>);

TEST(layout_rtr, converting_constructor) {
  rtr_static_mapping_t src;
  mapping_t dst(src);
  EXPECT_EQ(dst.extents(), extents_t(4, 6));
  EXPECT_EQ(dst.required_span_size(), 24);

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      EXPECT_EQ(dst(i, j), src(i, j));
    }
  }
}

TEST(layout_rtr, converting_constructor_constraints) {
  check_mapping_conversions<layout_t>();
}

#if defined(LAYOUT_TILED_ENABLE_P2630_SUBMDSPAN)
// ---------------------------------------------------------------------------
// submdspan_mapping
// ---------------------------------------------------------------------------

TEST(layout_rtr, submdspan_mapping_full_extent) {
  mapping_t m(extents_t(4, 6));
  auto r = submdspan_mapping(m, std::full_extent, std::full_extent);
  EXPECT_EQ(r.mapping.extents(), extents_t(4, 6));
  EXPECT_EQ(r.offset, size_t(0));
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 6; ++j)
      EXPECT_EQ(r.mapping(i, j), m(i, j));
}

TEST(layout_rtr, submdspan_mapping_integral) {
  mapping_t m(extents_t(4, 6));
  auto r = submdspan_mapping(m, size_t(1), std::full_extent);
  EXPECT_EQ(r.mapping.extents().rank(), 1);
  EXPECT_EQ(r.mapping.extents().extent(0), size_t(6));
  EXPECT_EQ(r.offset, m(1, 0));
  for (size_t j = 0; j < 6; ++j)
    EXPECT_EQ(r.mapping(j), m(1, j) - r.offset);
}

TEST(layout_rtr, submdspan_mapping_range) {
  mapping_t m(extents_t(4, 6));
  auto slices = std::pair<size_t, size_t>{2, 4};
  auto slicer = std::pair<size_t, size_t>{3, 6};
  auto r      = submdspan_mapping(m, slices, slicer);
  EXPECT_EQ(r.mapping.extents().rank(), 2);
  EXPECT_EQ(r.mapping.extents().extent(0), size_t(2));
  EXPECT_EQ(r.mapping.extents().extent(1), size_t(3));
  EXPECT_EQ(r.offset, m(2, 3));
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_EQ(r.mapping(i, j), m(2 + i, 3 + j) - r.offset);
}

TEST(layout_rtr, submdspan_mapping_mixed) {
  mapping_t m(extents_t(4, 6));
  auto r = submdspan_mapping(m, std::full_extent, size_t(2));
  EXPECT_EQ(r.mapping.extents().rank(), 1);
  EXPECT_EQ(r.mapping.extents().extent(0), size_t(4));
  EXPECT_EQ(r.offset, m(0, 2));
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(r.mapping(i), m(i, 2) - r.offset);
}

TEST(layout_rtr, submdspan_mapping_properties) {
  mapping_t m(extents_t(4, 6));
  auto r     = submdspan_mapping(m, std::full_extent, std::full_extent);
  using sm_t = std::decay_t<decltype(r.mapping)>;
  EXPECT_TRUE(sm_t::is_always_unique());
  EXPECT_TRUE(sm_t::is_always_exhaustive());
  EXPECT_FALSE(sm_t::is_always_strided());
  EXPECT_TRUE(r.mapping.is_unique());
  EXPECT_TRUE(r.mapping.is_exhaustive());
  EXPECT_FALSE(r.mapping.is_strided());
}

TEST(layout_rtr, submdspan_end_to_end) {
  check_submdspan_end_to_end<layout_t>();
}
#endif

// ---------------------------------------------------------------------------
// TDD: converting constructors from other layouts must not be available
// ---------------------------------------------------------------------------

static_assert(!std::is_constructible_v<mapping_t, std::layout_left::mapping<extents_t>>);
static_assert(!std::is_constructible_v<mapping_t, std::layout_right::mapping<extents_t>>);
static_assert(!std::is_constructible_v<mapping_t, std::layout_stride::mapping<extents_t>>);
