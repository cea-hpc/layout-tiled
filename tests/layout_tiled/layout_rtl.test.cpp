// tests/layout_tiled/layout_rtl.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <experimental/mdspan>

#include <gtest/gtest.h>
#include <layout_tiled/layout_rtl.hpp>

TEST(layout_rtl, maps_to_2d_linear_storage) {
  constexpr size_t ROWS  = 4;
  constexpr size_t COLS  = 6;
  constexpr size_t TROWS = 2;
  constexpr size_t TCOLS = 3;

  using element_type = int;
  using extents_type = std::dextents<size_t, 2>;
  using layout_rtl   = layout_tiled::layout_rtl<TROWS, TCOLS>;

  // clang-format off
  // Linearly ordered data (row-major)
  std::array<element_type, ROWS * COLS> row{
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
  };

  // Manually tiled data (row-major over tiles, row-major within tiles)
  std::array<element_type, ROWS * COLS> rtl{
     0,  6,  1,  7,  2,  8,
     3,  9,  4, 10,  5, 11,
    12, 18, 13, 19, 14, 20,
    15, 21, 16, 22, 17, 23,
  };
  // clang-format on

  auto m_row = std::mdspan<element_type, extents_type>(row.data(), ROWS, COLS);
  auto m_rtl = std::mdspan<element_type, extents_type, layout_rtl>(rtl.data(), ROWS, COLS);

  for (size_t i = 0; i < ROWS; ++i) {
    for (size_t j = 0; j < COLS; ++j) {
      auto row_val = m_row[i, j];
      auto rtl_val = m_rtl[i, j];
      EXPECT_EQ(row_val, rtl_val);
    }
  }
}
